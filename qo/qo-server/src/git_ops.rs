//! Git workspace helpers for the autonomous codebase improver.
//!
//! All operations execute in the OrbitQLang repo directory (configurable
//! via the `ORBITQ_REPO` env var, default `c:/Users/a.b/Graph/OrbitQLang`).
//! The git binary is `git` from PATH unless `PORTABLE_GIT` is set, in
//! which case `/c/Users/a.b/PortableGit/cmd/git.exe` is used.
//!
//! No new dependencies — everything goes through `tokio::process::Command`.
//! Errors are returned as plain `String` for ergonomic propagation through
//! the swarm orchestrator's `tracing::warn!` paths.

use std::time::{Duration, Instant};
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_REPO: &str = "c:/Users/a.b/Graph/OrbitQLang";
const PORTABLE_GIT_DEFAULT: &str = r"C:\Users\a.b\PortableGit\cmd\git.exe";

/// Resolve the repo root once per call. Cheap — no caching needed.
fn repo_dir() -> String {
    std::env::var("ORBITQ_REPO").unwrap_or_else(|_| DEFAULT_REPO.to_string())
}

/// Resolve the git binary path. PORTABLE_GIT env var (if non-empty) wins;
/// otherwise the hardcoded Windows portable git default; otherwise just
/// "git" from PATH.
fn git_bin() -> String {
    match std::env::var("PORTABLE_GIT") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => {
            if std::path::Path::new(PORTABLE_GIT_DEFAULT).exists() {
                PORTABLE_GIT_DEFAULT.to_string()
            } else {
                "git".to_string()
            }
        }
    }
}

/// Run `git <args>` in the repo dir. Returns (stdout, stderr) on success
/// or a formatted error string on non-zero exit / spawn failure.
async fn git(args: &[&str]) -> Result<(String, String), String> {
    let bin = git_bin();
    let dir = repo_dir();
    let output = Command::new(&bin)
        .args(args)
        .current_dir(&dir)
        .output()
        .await
        .map_err(|e| format!("git spawn ({} {:?}): {}", bin, args, e))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(format!(
            "git {:?} exit={:?}: {}",
            args,
            output.status.code(),
            stderr.chars().take(400).collect::<String>(),
        ));
    }
    Ok((stdout, stderr))
}

// ---------------------------------------------------------------------------
// Branch / status / commit primitives
// ---------------------------------------------------------------------------

/// Create and switch to a new branch. Fails if the branch exists or the
/// worktree is dirty.
pub async fn create_branch(name: &str) -> Result<(), String> {
    let _ = git(&["checkout", "-b", name]).await?;
    Ok(())
}

/// Return the current branch name (or `HEAD` for detached states).
pub async fn current_branch() -> Result<String, String> {
    let (stdout, _) = git(&["rev-parse", "--abbrev-ref", "HEAD"]).await?;
    Ok(stdout.trim().to_string())
}

/// True iff `git status --porcelain` produces any non-empty output.
pub async fn has_changes() -> Result<bool, String> {
    let (stdout, _) = git(&["status", "--porcelain"]).await?;
    Ok(!stdout.trim().is_empty())
}

/// Capture the current `git status --porcelain HEAD` output as a list
/// of file paths (rename targets only — the new name).
///
/// Used by the autonomous swarm git pipeline to capture the dirty-file
/// snapshot BEFORE the agent runs, so `commit_all` can later commit only
/// the files the agent actually touched (instead of dragging in
/// pre-existing dirty state via `git add -A`).
pub async fn capture_status() -> Result<Vec<String>, String> {
    let (stdout, _) = git(&["status", "--porcelain", "HEAD"]).await?;
    Ok(stdout
        .lines()
        .filter_map(|line| {
            // Porcelain format: "XY path" where XY are status codes.
            // Path starts at column 3 (0-indexed). For renames "R  old -> new"
            // we keep only the new path.
            if line.len() < 4 {
                return None;
            }
            let path_part = &line[3..];
            let path = if let Some(arrow) = path_part.find(" -> ") {
                &path_part[arrow + 4..]
            } else {
                path_part
            };
            Some(path.trim().to_string())
        })
        .collect())
}

/// Files that are dirty NOW but were not dirty BEFORE — i.e. the set of
/// files the agent touched during the swarm run.
pub fn diff_status(before: &[String], after: &[String]) -> Vec<String> {
    let before_set: std::collections::HashSet<&String> = before.iter().collect();
    after
        .iter()
        .filter(|f| !before_set.contains(f))
        .cloned()
        .collect()
}

/// Stage specific files (or everything if `files` is empty) and create
/// one commit. Returns the new commit SHA.
///
/// Pass an explicit file list from `diff_status` for autonomous swarm
/// runs so the commit only contains the agent's changes, not whatever
/// dirty state existed before. Pass `&[]` for legacy `git add -A`
/// behaviour.
pub async fn commit_all(message: &str, files: &[String]) -> Result<String, String> {
    if files.is_empty() {
        let _ = git(&["add", "-A"]).await?;
    } else {
        let mut args = vec!["add".to_string(), "--".to_string()];
        for f in files {
            args.push(f.clone());
        }
        let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let _ = git(&arg_refs).await?;
    }
    let _ = git(&["commit", "-m", message]).await?;
    let (sha, _) = git(&["rev-parse", "HEAD"]).await?;
    Ok(sha.trim().to_string())
}

/// Switch to an existing branch / commit / ref.
pub async fn checkout(target: &str) -> Result<(), String> {
    let _ = git(&["checkout", target]).await?;
    Ok(())
}

/// Force-delete a local branch.
pub async fn delete_branch(name: &str) -> Result<(), String> {
    let _ = git(&["branch", "-D", name]).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Diff summary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffSummary {
    pub files_changed: u32,
    pub insertions: u32,
    pub deletions: u32,
    pub raw: String,
}

/// `git diff --stat <base>..<branch>` parsed into counts.
///
/// Parses the trailing summary line, which `git diff --stat` produces in
/// the shape:
///   ` 3 files changed, 42 insertions(+), 7 deletions(-)`
/// Any of the three numbers may be missing when the diff is empty/one-sided.
pub async fn diff_summary(branch: &str, base: &str) -> Result<DiffSummary, String> {
    let range = format!("{}..{}", base, branch);
    let (stdout, _) = git(&["diff", "--stat", &range]).await?;
    let raw = stdout.clone();
    let summary_line = stdout
        .lines()
        .rev()
        .find(|l| l.contains("changed"))
        .unwrap_or("")
        .to_string();
    let (files_changed, insertions, deletions) = parse_diff_stat_summary(&summary_line);
    Ok(DiffSummary {
        files_changed,
        insertions,
        deletions,
        raw,
    })
}

/// Pull the three numbers out of a `git diff --stat` summary line.
/// Robust to missing fields (e.g. a pure addition has no "deletions(-)").
fn parse_diff_stat_summary(line: &str) -> (u32, u32, u32) {
    let mut files = 0u32;
    let mut ins = 0u32;
    let mut del = 0u32;
    for chunk in line.split(',') {
        let t = chunk.trim();
        // First number followed by a keyword.
        let mut parts = t.split_whitespace();
        if let (Some(num), Some(word)) = (parts.next(), parts.next()) {
            if let Ok(n) = num.parse::<u32>() {
                if word.starts_with("file") {
                    files = n;
                } else if word.starts_with("insertion") {
                    ins = n;
                } else if word.starts_with("deletion") {
                    del = n;
                }
            }
        }
    }
    (files, ins, del)
}

/// `git branch --list "auto/*"` — returns branch names with leading
/// markers (`* `, whitespace) stripped.
pub async fn list_auto_branches() -> Result<Vec<String>, String> {
    let (stdout, _) = git(&["branch", "--list", "auto/*"]).await?;
    let mut out = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim_start_matches('*').trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    Ok(out)
}

/// Last commit metadata for a ref.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CommitInfo {
    pub sha: String,
    pub message: String,
    pub date: String,
}

/// `git log -1 --format=%H%n%s%n%cI <branch>`. Returns the most recent
/// commit's SHA, subject and ISO-8601 committer date.
pub async fn last_commit(branch: &str) -> Result<CommitInfo, String> {
    let (stdout, _) = git(&["log", "-1", "--format=%H%n%s%n%cI", branch]).await?;
    let mut lines = stdout.lines();
    let sha = lines.next().unwrap_or("").trim().to_string();
    let message = lines.next().unwrap_or("").trim().to_string();
    let date = lines.next().unwrap_or("").trim().to_string();
    Ok(CommitInfo { sha, message, date })
}

/// Full unified-diff text between `<base>..<branch>`. Capped at 200 KB
/// because the cockpit is the only consumer and rendering a multi-MB
/// diff would lock the browser.
pub async fn diff_text(branch: &str, base: &str) -> Result<String, String> {
    let range = format!("{}..{}", base, branch);
    let (mut stdout, _) = git(&["diff", &range]).await?;
    const CAP: usize = 200 * 1024;
    if stdout.len() > CAP {
        stdout.truncate(CAP);
        stdout.push_str("\n…(truncated at 200 KB)");
    }
    Ok(stdout)
}

/// `git merge --no-ff <branch>` against the currently-checked-out branch.
pub async fn merge_no_ff(branch: &str) -> Result<String, String> {
    let (stdout, stderr) = git(&["merge", "--no-ff", branch]).await?;
    Ok(format!("{}{}", stdout, stderr))
}

// ---------------------------------------------------------------------------
// Build & test runner
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct TestResult {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

const STDOUT_CAP: usize = 8 * 1024;
const STDERR_CAP: usize = 4 * 1024;

fn cap_to(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}\n…(truncated)", &s[..max])
    }
}

/// Run the standard build + frontend type-check sequence. The two phases
/// are sequential; the first failure short-circuits.
///
/// Phase 1: `cargo build --bin qo --no-default-features` (5 min cap)
/// Phase 2: `npx tsc --noEmit` in `frontend/` (2 min cap)
pub async fn run_build_and_test() -> TestResult {
    let started = Instant::now();
    let dir = repo_dir();

    // Inject our cargo path the same way the manual builds do. We don't
    // mutate the global env — only this child process inherits it.
    let path_with_cargo = match std::env::var("PATH") {
        Ok(existing) => format!("/c/Users/a.b/.cargo/bin:{}", existing),
        Err(_) => "/c/Users/a.b/.cargo/bin".to_string(),
    };

    // ── Phase 1: cargo build ─────────────────────────────────────
    let cargo_fut = Command::new("cargo")
        .args(["build", "--bin", "qo", "--no-default-features"])
        .current_dir(&dir)
        .env("PATH", &path_with_cargo)
        .output();

    let cargo_res = match timeout(Duration::from_secs(300), cargo_fut).await {
        Err(_) => {
            return TestResult {
                success: false,
                stdout: String::new(),
                stderr: "cargo build: timed out after 300s".to_string(),
                duration_ms: started.elapsed().as_millis() as u64,
            };
        }
        Ok(Err(e)) => {
            return TestResult {
                success: false,
                stdout: String::new(),
                stderr: format!("cargo build spawn: {}", e),
                duration_ms: started.elapsed().as_millis() as u64,
            };
        }
        Ok(Ok(o)) => o,
    };

    if !cargo_res.status.success() {
        return TestResult {
            success: false,
            stdout: cap_to(&String::from_utf8_lossy(&cargo_res.stdout), STDOUT_CAP),
            stderr: cap_to(&String::from_utf8_lossy(&cargo_res.stderr), STDERR_CAP),
            duration_ms: started.elapsed().as_millis() as u64,
        };
    }

    // ── Phase 2: tsc --noEmit in frontend/ ───────────────────────
    let frontend_dir = format!("{}/frontend", dir.trim_end_matches('/'));
    // On Windows we need to invoke npx via cmd.exe so the shim resolves.
    #[cfg(windows)]
    let tsc_fut = Command::new("cmd")
        .args(["/C", "npx", "tsc", "--noEmit"])
        .current_dir(&frontend_dir)
        .env("PATH", &path_with_cargo)
        .output();
    #[cfg(not(windows))]
    let tsc_fut = Command::new("npx")
        .args(["tsc", "--noEmit"])
        .current_dir(&frontend_dir)
        .env("PATH", &path_with_cargo)
        .output();

    let tsc_res = match timeout(Duration::from_secs(120), tsc_fut).await {
        Err(_) => {
            return TestResult {
                success: false,
                stdout: cap_to(&String::from_utf8_lossy(&cargo_res.stdout), STDOUT_CAP),
                stderr: "tsc --noEmit: timed out after 120s".to_string(),
                duration_ms: started.elapsed().as_millis() as u64,
            };
        }
        Ok(Err(e)) => {
            return TestResult {
                success: false,
                stdout: cap_to(&String::from_utf8_lossy(&cargo_res.stdout), STDOUT_CAP),
                stderr: format!("tsc spawn: {}", e),
                duration_ms: started.elapsed().as_millis() as u64,
            };
        }
        Ok(Ok(o)) => o,
    };

    let combined_stdout = format!(
        "--- cargo build ---\n{}\n--- tsc --noEmit ---\n{}",
        String::from_utf8_lossy(&cargo_res.stdout),
        String::from_utf8_lossy(&tsc_res.stdout),
    );
    let combined_stderr = format!(
        "--- cargo build ---\n{}\n--- tsc --noEmit ---\n{}",
        String::from_utf8_lossy(&cargo_res.stderr),
        String::from_utf8_lossy(&tsc_res.stderr),
    );

    TestResult {
        success: tsc_res.status.success(),
        stdout: cap_to(&combined_stdout, STDOUT_CAP),
        stderr: cap_to(&combined_stderr, STDERR_CAP),
        duration_ms: started.elapsed().as_millis() as u64,
    }
}

// ---------------------------------------------------------------------------
// Tests — pure parsing only (no shelling out in unit tests).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_summary() {
        let line = " 3 files changed, 42 insertions(+), 7 deletions(-)";
        assert_eq!(parse_diff_stat_summary(line), (3, 42, 7));
    }

    #[test]
    fn parse_only_insertions() {
        let line = " 1 file changed, 9 insertions(+)";
        assert_eq!(parse_diff_stat_summary(line), (1, 9, 0));
    }

    #[test]
    fn parse_only_deletions() {
        let line = " 2 files changed, 5 deletions(-)";
        assert_eq!(parse_diff_stat_summary(line), (2, 0, 5));
    }

    #[test]
    fn parse_empty_line_returns_zeros() {
        assert_eq!(parse_diff_stat_summary(""), (0, 0, 0));
    }

    #[test]
    fn cap_to_truncates() {
        let s = "a".repeat(20);
        let out = cap_to(&s, 5);
        assert!(out.starts_with("aaaaa"));
        assert!(out.contains("(truncated)"));
    }

    #[test]
    fn cap_to_passes_short() {
        assert_eq!(cap_to("hi", 100), "hi");
    }
}
