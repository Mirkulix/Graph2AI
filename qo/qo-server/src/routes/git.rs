//! `GET/POST /api/git/*` — cockpit endpoints for browsing and managing
//! the auto-improver branches produced by autonomous swarms.
//!
//! These endpoints are intentionally read-mostly. The destructive
//! operations (`merge`, `discard`) require an explicit POST so a
//! mis-clicked GET can never lose work.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::git_ops;
use crate::AppState;

const DEFAULT_BASE: &str = "main";

fn base_branch() -> String {
    std::env::var("ORBITQ_BASE_BRANCH").unwrap_or_else(|_| DEFAULT_BASE.to_string())
}

#[derive(Debug, Serialize)]
pub struct AutoBranchInfo {
    pub name: String,
    pub last_commit_sha: String,
    pub last_commit_message: String,
    pub last_commit_date: String,
    pub diff: Option<git_ops::DiffSummary>,
}

#[derive(Debug, Serialize)]
pub struct DiffDetail {
    pub branch: String,
    pub base: String,
    pub summary: git_ops::DiffSummary,
    pub diff: String,
}

#[derive(Debug, Deserialize)]
pub struct BranchRequest {
    pub branch: String,
}

#[derive(Debug, Serialize)]
pub struct MergeResponse {
    pub success: bool,
    pub merged_branch: String,
    pub base: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct DiscardResponse {
    pub deleted: String,
}

/// `GET /api/git/branches` — list every `auto/*` branch with last-commit
/// metadata and an optional diff summary against the configured base
/// branch (default `main`). Best-effort: a failure on any one branch
/// degrades to `diff: None` rather than poisoning the whole response.
pub async fn list_auto_branches(
    State(_state): State<Arc<AppState>>,
) -> Result<Json<Vec<AutoBranchInfo>>, (StatusCode, String)> {
    let names = git_ops::list_auto_branches()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let base = base_branch();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let commit = match git_ops::last_commit(&name).await {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(branch = %name, "git: last_commit failed: {}", e);
                continue;
            }
        };
        let diff = git_ops::diff_summary(&name, &base).await.ok();
        out.push(AutoBranchInfo {
            name,
            last_commit_sha: commit.sha,
            last_commit_message: commit.message,
            last_commit_date: commit.date,
            diff,
        });
    }
    Ok(Json(out))
}

/// `GET /api/git/diff/{branch}` — full unified diff vs the base branch,
/// capped at 200 KB by `git_ops::diff_text`.
pub async fn diff_branch(
    State(_state): State<Arc<AppState>>,
    Path(branch): Path<String>,
) -> Result<Json<DiffDetail>, (StatusCode, String)> {
    let base = base_branch();
    let summary = git_ops::diff_summary(&branch, &base)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let diff = git_ops::diff_text(&branch, &base)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(DiffDetail {
        branch,
        base,
        summary,
        diff,
    }))
}

/// `POST /api/git/merge { branch }` — checkout the base branch, then
/// `git merge --no-ff <branch>`. Conflicts are surfaced via a non-success
/// response with the merge driver's stderr so the cockpit can show them.
pub async fn merge_branch(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<BranchRequest>,
) -> Result<Json<MergeResponse>, (StatusCode, String)> {
    let base = base_branch();
    if req.branch.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "branch must be non-empty".into()));
    }
    if req.branch == base {
        return Err((StatusCode::BAD_REQUEST, "cannot merge base into itself".into()));
    }
    git_ops::checkout(&base)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("checkout {}: {}", base, e)))?;
    match git_ops::merge_no_ff(&req.branch).await {
        Ok(message) => {
            tracing::info!(branch = %req.branch, "git: merged into {}", base);
            Ok(Json(MergeResponse {
                success: true,
                merged_branch: req.branch,
                base,
                message,
            }))
        }
        Err(e) => {
            tracing::warn!(branch = %req.branch, "git: merge failed: {}", e);
            Ok(Json(MergeResponse {
                success: false,
                merged_branch: req.branch,
                base,
                message: e,
            }))
        }
    }
}

/// `POST /api/git/discard { branch }` — `git branch -D <branch>`.
pub async fn discard_branch(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<BranchRequest>,
) -> Result<Json<DiscardResponse>, (StatusCode, String)> {
    if req.branch.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "branch must be non-empty".into()));
    }
    if !req.branch.starts_with("auto/") {
        // Refuse to delete anything outside the auto-improver namespace.
        return Err((
            StatusCode::FORBIDDEN,
            "discard restricted to branches under auto/*".into(),
        ));
    }
    git_ops::delete_branch(&req.branch)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    tracing::info!(branch = %req.branch, "git: branch discarded");
    Ok(Json(DiscardResponse { deleted: req.branch }))
}
