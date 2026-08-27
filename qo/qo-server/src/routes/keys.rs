//! Admin seat management over HTTP — the cockpit half of `qlang keys`.
//!
//! The store is shared with the auth middleware (behind an `RwLock`) and is
//! persisted back to the same file the CLI edits, so a seat issued here
//! survives a restart. Secrets are returned exactly once, at issue time.

use axum::extract::{Extension, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::api_keys::{ApiKeyStore, Principal, Role};
use crate::AppState;

/// `GET /api/me` — the authenticated seat's own identity (label + role).
/// Any authenticated principal may call it; the open (no-seats) case reports
/// the implicit `root` admin.
pub async fn me(Extension(principal): Extension<Principal>) -> Json<Value> {
    Json(json!({ "label": principal.label, "role": principal.role.as_str() }))
}

#[derive(Deserialize)]
pub struct IssueRequest {
    pub label: String,
    pub role: String,
}

#[derive(Deserialize)]
pub struct RevokeRequest {
    pub label: String,
}

/// `GET /api/keys` — list the seats. Never returns the secrets.
pub async fn list_keys(State(state): State<Arc<AppState>>) -> Json<Value> {
    let store = state.api_keys.read().await;
    let seats: Vec<Value> = store
        .keys
        .iter()
        .map(|key| {
            json!({
                "label": key.label,
                "role": key.role.as_str(),
                "revoked": key.revoked,
            })
        })
        .collect();
    Json(json!({ "seats": seats, "active_seats": store.active_seats() }))
}

/// `POST /api/keys/issue` — create a seat and return the secret exactly once.
pub async fn issue_key(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IssueRequest>,
) -> (StatusCode, Json<Value>) {
    let role = match req.role.as_str() {
        "viewer" => Role::Viewer,
        "member" => Role::Member,
        "admin" => Role::Admin,
        other => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("unknown role {other:?}; use member|admin|viewer") })),
            );
        }
    };
    if req.label.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "label must not be empty" })));
    }

    // A fresh 32-byte secret from the same RNG the protocol trusts.
    let secret = hex(&qlang_core::crypto::Keypair::generate().secret_seed());

    let mut store = state.api_keys.write().await;
    store.issue(&req.label, &secret, role);
    if let Err(e) = persist(&state, &store) {
        store.revoke(&req.label); // roll back: nothing that would not survive a restart
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("cannot persist key store: {e}") })),
        );
    }

    (
        StatusCode::OK,
        Json(json!({ "label": req.label, "role": role.as_str(), "secret": secret })),
    )
}

/// `POST /api/keys/revoke` — revoke a seat by label.
pub async fn revoke_key(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RevokeRequest>,
) -> (StatusCode, Json<Value>) {
    let mut store = state.api_keys.write().await;
    if !store.revoke(&req.label) {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("no active seat labelled {:?}", req.label) })),
        );
    }
    if let Err(e) = persist(&state, &store) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("cannot persist key store: {e}") })),
        );
    }
    (
        StatusCode::OK,
        Json(json!({ "revoked": req.label, "active_seats": store.active_seats() })),
    )
}

/// Write the store back to `api_keys_path` with a tight file mode.
fn persist(state: &AppState, store: &ApiKeyStore) -> Result<(), String> {
    let json = store.to_json().map_err(|e| e.to_string())?;
    if let Some(parent) = state.api_keys_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&state.api_keys_path)
            .map_err(|e| e.to_string())?;
        std::fs::set_permissions(&state.api_keys_path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .map_err(|e| e.to_string())?;
        file.write_all(json.as_bytes()).map_err(|e| e.to_string())
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&state.api_keys_path, json).map_err(|e| e.to_string())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
