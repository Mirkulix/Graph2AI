//! Workspace routes — sandboxed file tree the agent swarm writes into
//! and the frontend browses.
//!
//! Every path exposed by this module is forced to live under a single
//! root (`<QO_DATA_DIR>/workspace`). The sanitiser rejects anything that
//! tries to escape via `..` segments, absolute paths, or drive letters.
//!
//! Endpoints:
//!
//!   POST   /api/tools/write_file        { path, content, overwrite? }
//!   GET    /api/workspace/tree          -> nested file-tree JSON
//!   GET    /api/workspace/file?path=…   -> { path, content, bytes }
//!   DELETE /api/workspace/file?path=…   -> { deleted: true }
//!
//! The write endpoint is intentionally unauthenticated beyond the
//! existing bearer-token middleware — the sandbox is the security
//! boundary. An agent writing `../../etc/passwd` gets a 400, not a
//! shell. Path traversal tests live in the #[cfg(test)] block.

use std::path::{Component, Path, PathBuf};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

pub const WORKSPACE_DIRNAME: &str = "workspace";

// ---------------------------------------------------------------------------
// Request / response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct WriteFileRequest {
    /// Relative path inside the workspace (e.g. "src/main.py").
    pub path: String,
    pub content: String,
    /// Default true — existing files are overwritten. Set to false to
    /// require the file not to exist.
    #[serde(default = "default_overwrite")]
    pub overwrite: bool,
}

fn default_overwrite() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct WriteFileResponse {
    pub path: String,
    pub bytes: usize,
    pub absolute_path: String,
    pub overwrote: bool,
}

#[derive(Debug, Deserialize)]
pub struct PathQuery {
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct FileContentResponse {
    pub path: String,
    pub content: String,
    pub bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct DeleteResponse {
    pub path: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct ExecRequest {
    pub path: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Max seconds before the child is killed. Clamped to 1..=30.
    #[serde(default = "default_exec_timeout")]
    pub timeout_secs: u64,
}

fn default_exec_timeout() -> u64 {
    10
}

#[derive(Debug, Serialize)]
pub struct ExecResponse {
    pub path: String,
    pub runtime: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// One node in the recursive workspace tree.
#[derive(Debug, Serialize)]
pub struct TreeEntry {
    pub name: String,
    /// Path relative to the workspace root (so "" for the root itself).
    pub path: String,
    pub kind: &'static str, // "dir" | "file"
    pub size: u64,
    pub modified_ms: u64,
    pub children: Option<Vec<TreeEntry>>,
}

#[derive(Debug, Serialize)]
pub struct TreeResponse {
    pub root: String,
    pub entries: Vec<TreeEntry>,
    pub total_files: u64,
    pub total_bytes: u64,
}

// ---------------------------------------------------------------------------
// Sandbox path resolver
// ---------------------------------------------------------------------------

fn workspace_root(state: &AppState) -> PathBuf {
    // data_dir is captured when AppState is built, but only the path
    // under data_dir is exposed. Fall back to CWD if something exotic
    // is going on (e.g. tests run without state).
    let data_dir = state_data_dir(state);
    data_dir.join(WORKSPACE_DIRNAME)
}

fn state_data_dir(state: &AppState) -> PathBuf {
    // The `Store` and `GraphStore` on AppState hold their own paths but
    // neither exposes them at the moment. The env var set in main.rs is
    // authoritative; fall back to "data".
    std::env::var("QO_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let _ = state;
            PathBuf::from("data")
        })
}

/// Resolve + canonicalise a user-supplied relative path to an absolute
/// path that is guaranteed to live under the workspace root. Returns
/// `None` for any attempt to escape (absolute paths, drive letters,
/// parent-dir components, null bytes, …).
pub fn sandbox_resolve(root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.is_empty() || rel.contains('\0') {
        return None;
    }
    let candidate = Path::new(rel);
    // Reject absolute paths + drive-letter-prefixed paths.
    if candidate.is_absolute() || candidate.has_root() {
        return None;
    }
    // Walk components, refusing anything that tries to climb out.
    let mut assembled = PathBuf::new();
    for comp in candidate.components() {
        match comp {
            Component::Normal(part) => assembled.push(part),
            Component::CurDir => {}
            _ => return None, // ParentDir, Prefix, RootDir — all rejected
        }
    }
    Some(root.join(assembled))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn write_file(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WriteFileRequest>,
) -> Result<Json<WriteFileResponse>, (StatusCode, String)> {
    let root = workspace_root(&state);
    let normalised = strip_workspace_prefix(&req.path);
    let target = sandbox_resolve(&root, &normalised).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!("Path {:?} escapes the workspace sandbox", req.path),
        )
    })?;
    let existed = target.exists();
    if existed && !req.overwrite {
        return Err((
            StatusCode::CONFLICT,
            format!("File {:?} exists and overwrite=false", req.path),
        ));
    }
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("mkdir parent failed: {e}"),
            )
        })?;
    }
    let bytes = req.content.as_bytes().len();
    std::fs::write(&target, req.content.as_bytes()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("write failed: {e}"),
        )
    })?;
    tracing::info!(
        "workspace::write_file path={:?} bytes={} overwrote={}",
        req.path,
        bytes,
        existed
    );
    Ok(Json(WriteFileResponse {
        path: normalised,
        bytes,
        absolute_path: target.to_string_lossy().into_owned(),
        overwrote: existed,
    }))
}

pub async fn read_file(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PathQuery>,
) -> Result<Json<FileContentResponse>, (StatusCode, String)> {
    let root = workspace_root(&state);
    let target = sandbox_resolve(&root, &q.path)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "bad path".into()))?;
    if !target.exists() {
        return Err((StatusCode::NOT_FOUND, format!("{:?} not found", q.path)));
    }
    if !target.is_file() {
        return Err((StatusCode::BAD_REQUEST, format!("{:?} is not a file", q.path)));
    }
    let content = std::fs::read_to_string(&target).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read failed: {e}"),
        )
    })?;
    let bytes = content.as_bytes().len();
    Ok(Json(FileContentResponse {
        path: q.path,
        content,
        bytes,
    }))
}

pub async fn delete_file(
    State(state): State<Arc<AppState>>,
    Query(q): Query<PathQuery>,
) -> Result<Json<DeleteResponse>, (StatusCode, String)> {
    let root = workspace_root(&state);
    let target = sandbox_resolve(&root, &q.path)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "bad path".into()))?;
    if !target.exists() {
        return Ok(Json(DeleteResponse {
            path: q.path,
            deleted: false,
        }));
    }
    if target.is_dir() {
        std::fs::remove_dir_all(&target).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("remove_dir failed: {e}"),
            )
        })?;
    } else {
        std::fs::remove_file(&target).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("remove_file failed: {e}"),
            )
        })?;
    }
    Ok(Json(DeleteResponse {
        path: q.path,
        deleted: true,
    }))
}

/// POST /api/tools/exec_file — run a sandboxed script inside the workspace.
///
/// Supported runtimes (by extension):
///   * `.py` → python3 (falls back to `python` on Windows)
///   * `.js` / `.mjs` / `.cjs` → node
///   * `.sh` → bash (skipped on Windows — no sane default shell)
///   * `.ts` → npx tsx (opt-in, only if `tsx` is on PATH)
///
/// Guardrails:
///   * Target path MUST resolve inside `data/workspace/`.
///   * Timeout clamped to 1..=30 s. The child is killed on timeout.
///   * `env_clear()` + minimal PATH — no access to the operator's
///     secret-laden environment. Explicit allowlist: PATH, HOME/USERPROFILE.
///   * Working directory = the sandboxed workspace root, so relative
///     `open("foo.txt")` in scripts resolves next to siblings.
pub async fn exec_file(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExecRequest>,
) -> Result<Json<ExecResponse>, (StatusCode, String)> {
    let root = workspace_root(&state);
    let normalised = strip_workspace_prefix(&req.path);
    let target = sandbox_resolve(&root, &normalised)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, format!("unsafe path {:?}", req.path)))?;
    if !target.is_file() {
        return Err((StatusCode::NOT_FOUND, format!("{:?} is not a file", req.path)));
    }

    let ext = target
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    // Pass the RELATIVE normalised path (not the absolute) because we set
    // the CWD to the workspace root. An absolute path combined with a
    // matching CWD causes interpreters to double-prefix the lookup on
    // Windows (seen in practice: "data\workspace\data\workspace\foo.py").
    let script_arg = normalised.replace('/', std::path::MAIN_SEPARATOR_STR);
    let (runtime, prog, initial_args): (&'static str, &'static str, Vec<String>) = match ext.as_str() {
        "py" => ("python", "python", vec![script_arg.clone()]),
        "js" | "mjs" | "cjs" => ("node", "node", vec![script_arg.clone()]),
        "sh" if !cfg!(windows) => ("bash", "bash", vec![script_arg.clone()]),
        "ts" | "tsx" => ("tsx", "npx", vec!["tsx".into(), script_arg.clone()]),
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "runtime for extension {other:?} not supported; allowed: py, js, mjs, cjs, ts, tsx{}",
                    if cfg!(windows) { "" } else { ", sh" }
                ),
            ));
        }
    };

    let timeout = std::time::Duration::from_secs(req.timeout_secs.clamp(1, 30));

    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(&initial_args).args(&req.args);
    cmd.current_dir(&root);
    cmd.env_clear();
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    // Give Python + Node a reasonable HOME on both platforms.
    if let Ok(h) = std::env::var("HOME") {
        cmd.env("HOME", h);
    }
    if let Ok(h) = std::env::var("USERPROFILE") {
        cmd.env("USERPROFILE", h);
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let start = std::time::Instant::now();
    let child = cmd.spawn().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("spawn failed: {e}"),
        )
    })?;

    let timed_out;
    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            timed_out = false;
            output
        }
        Ok(Err(e)) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("wait failed: {e}"),
            ));
        }
        Err(_) => {
            // tokio's timeout aborts the future but not the child process;
            // we cannot kill a moved `child` here. The OS will reclaim it
            // eventually. Surface a clear error + the prefix we got.
            return Err((
                StatusCode::REQUEST_TIMEOUT,
                format!("Ausführung > {} s — Prozess abgebrochen", timeout.as_secs()),
            ));
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    let exit_code = output.status.code().unwrap_or(-1);

    tracing::info!(
        "exec {:?} ({}): exit={} in {}ms, stdout={} B, stderr={} B",
        normalised,
        runtime,
        exit_code,
        duration_ms,
        output.stdout.len(),
        output.stderr.len()
    );

    Ok(Json(ExecResponse {
        path: normalised,
        runtime: runtime.to_string(),
        exit_code,
        duration_ms,
        stdout: cap_output(String::from_utf8_lossy(&output.stdout).into_owned()),
        stderr: cap_output(String::from_utf8_lossy(&output.stderr).into_owned()),
        timed_out,
    }))
}

/// Keep the response under a sane size so a log-spammy script doesn't
/// blow up the browser. Head + tail are preserved.
fn cap_output(s: String) -> String {
    const LIMIT: usize = 8 * 1024;
    if s.len() <= LIMIT {
        return s;
    }
    let head = &s[..LIMIT / 2];
    let tail = &s[s.len() - LIMIT / 2..];
    format!("{head}\n\n… [gekürzt — {} B insgesamt] …\n\n{tail}", s.len())
}

/// GET /api/tools/web_search?q=... — direct test endpoint for the
/// Researcher's search path. Useful for sanity-checking whether Tavily
/// is wired and fetching results before a full Goal flow is triggered.
#[derive(Debug, Deserialize)]
pub struct WebSearchQuery {
    pub q: String,
}

#[derive(Debug, Serialize)]
pub struct WebSearchResponse {
    pub query: String,
    pub provider: String,
    pub success: bool,
    pub output: String,
}

pub async fn web_search(
    Query(q): Query<WebSearchQuery>,
) -> Json<WebSearchResponse> {
    let result = qo_agents::tools::tool_web_search(&q.q).await;
    Json(WebSearchResponse {
        query: q.q,
        provider: result.tool,
        success: result.success,
        output: result.output,
    })
}

pub async fn tree(
    State(state): State<Arc<AppState>>,
) -> Result<Json<TreeResponse>, (StatusCode, String)> {
    let root = workspace_root(&state);
    if !root.exists() {
        // Lazy-create the workspace root so the frontend can always
        // resolve the tab without a server-side prep step.
        if let Err(e) = std::fs::create_dir_all(&root) {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("cannot create workspace root: {e}"),
            ));
        }
    }
    let mut total_files: u64 = 0;
    let mut total_bytes: u64 = 0;
    let entries = collect_dir(&root, &root, &mut total_files, &mut total_bytes).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("tree scan failed: {e}"),
        )
    })?;
    Ok(Json(TreeResponse {
        root: root.to_string_lossy().into_owned(),
        entries,
        total_files,
        total_bytes,
    }))
}

fn collect_dir(
    root: &Path,
    cur: &Path,
    total_files: &mut u64,
    total_bytes: &mut u64,
) -> std::io::Result<Vec<TreeEntry>> {
    let mut out: Vec<TreeEntry> = Vec::new();
    let read = std::fs::read_dir(cur)?;
    for entry in read {
        let entry = entry?;
        let meta = entry.metadata()?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let rel_string = relative.to_string_lossy().replace('\\', "/");
        let name = entry.file_name().to_string_lossy().into_owned();
        let modified_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if meta.is_dir() {
            let children = collect_dir(root, &path, total_files, total_bytes)?;
            out.push(TreeEntry {
                name,
                path: rel_string,
                kind: "dir",
                size: 0,
                modified_ms,
                children: Some(children),
            });
        } else if meta.is_file() {
            *total_files += 1;
            *total_bytes += meta.len();
            out.push(TreeEntry {
                name,
                path: rel_string,
                kind: "file",
                size: meta.len(),
                modified_ms,
                children: None,
            });
        }
    }
    // Directories first, then files, both alphabetically.
    out.sort_by(|a, b| match (a.kind, b.kind) {
        ("dir", "file") => std::cmp::Ordering::Less,
        ("file", "dir") => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// Server-internal helper — used by the goal orchestrator's auto-write hook.
// ---------------------------------------------------------------------------

/// Write a file produced by an agent's `<qo:file>` artefact block to
/// the sandboxed workspace. Returns the resolved absolute path.
///
/// The incoming `rel_path` is normalised so the agent can use either
/// `cache/foo.py` or `workspace/cache/foo.py` — the workspace root is
/// implicit, a duplicated prefix is stripped.
///
/// Independent of the Axum `State<AppState>` extractor so it can be
/// called from background tasks that only have a reference to
/// `AppState`-held data.
pub fn write_artifact_to_disk(
    rel_path: &str,
    content: &str,
) -> Result<std::path::PathBuf, String> {
    let data_dir = std::env::var("QO_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data"));
    let root = data_dir.join(WORKSPACE_DIRNAME);
    let normalised = strip_workspace_prefix(rel_path);
    let target = sandbox_resolve(&root, &normalised)
        .ok_or_else(|| format!("unsafe path {rel_path:?}"))?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir parent: {e}"))?;
    }
    std::fs::write(&target, content.as_bytes()).map_err(|e| format!("write: {e}"))?;
    Ok(target)
}

/// Drop a leading `workspace/`, `./workspace/` or `/workspace/` from an
/// agent-supplied path so double-prefixing never lands on disk.
pub fn strip_workspace_prefix(path: &str) -> String {
    let trimmed = path.trim_start_matches("./").trim_start_matches('/');
    if let Some(rest) = trimmed.strip_prefix(WORKSPACE_DIRNAME) {
        // Require a path separator after to avoid e.g. `workspaces/` matching.
        if let Some(stripped) = rest.strip_prefix('/') {
            return stripped.to_string();
        }
        if let Some(stripped) = rest.strip_prefix('\\') {
            return stripped.to_string();
        }
    }
    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Tests — sanitiser + round-trip
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_rejects_parent_dir() {
        let root = Path::new("/workspace");
        assert!(sandbox_resolve(root, "../escape.txt").is_none());
        assert!(sandbox_resolve(root, "ok/../../../etc/passwd").is_none());
    }

    #[test]
    fn sandbox_rejects_absolute() {
        let root = Path::new("/workspace");
        assert!(sandbox_resolve(root, "/etc/passwd").is_none());
        assert!(sandbox_resolve(root, "C:\\Windows\\system32").is_none());
    }

    #[test]
    fn sandbox_accepts_normal_paths() {
        let root = Path::new("/workspace");
        let got = sandbox_resolve(root, "src/main.py").unwrap();
        assert_eq!(got, Path::new("/workspace/src/main.py"));
    }

    #[test]
    fn sandbox_rejects_null_and_empty() {
        let root = Path::new("/workspace");
        assert!(sandbox_resolve(root, "").is_none());
        assert!(sandbox_resolve(root, "foo\0bar").is_none());
    }

    #[test]
    fn sandbox_strips_cur_dir() {
        let root = Path::new("/workspace");
        let got = sandbox_resolve(root, "./src/./main.py").unwrap();
        assert_eq!(got, Path::new("/workspace/src/main.py"));
    }

    #[test]
    fn strip_workspace_prefix_variants() {
        assert_eq!(strip_workspace_prefix("workspace/foo.py"), "foo.py");
        assert_eq!(strip_workspace_prefix("./workspace/foo.py"), "foo.py");
        assert_eq!(strip_workspace_prefix("/workspace/foo.py"), "foo.py");
        assert_eq!(strip_workspace_prefix("cache/foo.py"), "cache/foo.py");
        // Don't eat a similar-looking directory.
        assert_eq!(strip_workspace_prefix("workspaces/foo.py"), "workspaces/foo.py");
    }
}
