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
