//! Presence registry — connected IDEs/agents register themselves so the
//! mesh can discover who is online and route handovers accordingly.
//!
//! Endpoints (all under `/api/presence`):
//!
//!   POST   /api/presence/register
//!       → register an IDE/agent. TTL is 60s; client must heartbeat.
//!   POST   /api/presence/heartbeat/{identity}
//!       → bump expires_at without changing other fields.
//!   DELETE /api/presence/{identity}
//!       → explicit deregister (e.g. on VS Code extension deactivate).
//!   GET    /api/presence
//!       → list all currently-registered identities.
//!
//! Storage is purely in-memory: presence is ephemeral by design — a
//! restart of `qo` should wipe it, so dead IDEs aren't resurrected from
//! disk. A background sweeper task (spawned from `lib.rs::build_app`)
//! evicts expired entries every 30s.
//!
//! No auth here — the existing `auth_middleware` already wraps the API
//! router if `QO_AUTH_TOKEN` is set.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::AppState;

/// Time-to-live for a presence entry. Each register/heartbeat pushes
/// `expires_at` forward by this much. Picked so that a 30s sweeper plus
/// a 5s frontend poll see a consistent picture even with one missed beat.
pub const PRESENCE_TTL: Duration = Duration::from_secs(60);

/// How often the sweeper task runs to evict expired entries.
pub const SWEEPER_INTERVAL: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Public, frontend-visible record of one registered IDE/agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceEntry {
    pub identity: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ide_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_model: Option<String>,
    pub registered_at: String,
    pub last_seen_at: String,
    pub expires_at: String,
}

#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub identity: String,
    #[serde(default)]
    pub ide_name: Option<String>,
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub llm_provider: Option<String>,
    #[serde(default)]
    pub llm_model: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RegisterResponse {
    pub identity: String,
    pub registered_at: String,
    pub expires_at: String,
}

#[derive(Debug, Serialize)]
pub struct HeartbeatResponse {
    pub ok: bool,
    pub expires_at: String,
}

#[derive(Debug, Serialize)]
pub struct OkResponse {
    pub ok: bool,
}

// ---------------------------------------------------------------------------
// Time helpers
// ---------------------------------------------------------------------------

/// Format a `SystemTime` as an ISO 8601 / RFC 3339 UTC string,
/// e.g. `"2026-04-23T07:25:30Z"`.
///
/// Hand-rolled (no `chrono` dependency in this crate). Uses the standard
/// civil-from-days algorithm from Howard Hinnant's date library.
fn format_iso8601(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400) as u32;

    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    // civil_from_days (Hinnant). days are days since 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, m, d, hour, minute, second
    )
}

fn now_iso() -> String {
    format_iso8601(SystemTime::now())
}

fn expires_iso() -> String {
    format_iso8601(SystemTime::now() + PRESENCE_TTL)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `POST /api/presence/register` — create or refresh a presence entry.
///
/// Re-registering the same `identity` overwrites the previous record
/// (lets a VS Code extension `activate` cleanly after a reload without
/// orphaning the old entry).
pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<RegisterResponse>, (StatusCode, String)> {
    if req.identity.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "identity must be non-empty".to_string(),
        ));
    }

    let now = now_iso();
    let exp = expires_iso();

    let entry = PresenceEntry {
        identity: req.identity.clone(),
        ide_name: req.ide_name,
        host: req.host,
        capabilities: req.capabilities.unwrap_or_default(),
        llm_provider: req.llm_provider,
        llm_model: req.llm_model,
        registered_at: now.clone(),
        last_seen_at: now.clone(),
        expires_at: exp.clone(),
    };

    {
        let mut map = state.presence.lock().await;
        map.insert(req.identity.clone(), entry);
    }

    tracing::info!(
        identity = %req.identity,
        "presence: registered"
    );

    Ok(Json(RegisterResponse {
        identity: req.identity,
        registered_at: now,
        expires_at: exp,
    }))
}

/// `POST /api/presence/heartbeat/{identity}` — bump TTL.
///
/// Returns 404 if the identity is unknown (e.g. swept after a long
/// network glitch). Clients should react by re-registering.
pub async fn heartbeat(
    State(state): State<Arc<AppState>>,
    Path(identity): Path<String>,
) -> Result<Json<HeartbeatResponse>, (StatusCode, String)> {
    let now = now_iso();
    let exp = expires_iso();

    let mut map = state.presence.lock().await;
    match map.get_mut(&identity) {
        Some(entry) => {
            entry.last_seen_at = now;
            entry.expires_at = exp.clone();
            tracing::debug!(identity = %identity, "presence: heartbeat");
            Ok(Json(HeartbeatResponse { ok: true, expires_at: exp }))
        }
        None => Err((
            StatusCode::NOT_FOUND,
            format!("unknown identity: {}", identity),
        )),
    }
}

/// `DELETE /api/presence/{identity}` — explicit deregister.
///
/// Idempotent: returns `{ok: true}` even if the identity wasn't present,
/// so the VS Code extension can call this on `deactivate` without
/// caring whether the sweeper already ran.
pub async fn deregister(
    State(state): State<Arc<AppState>>,
    Path(identity): Path<String>,
) -> Json<OkResponse> {
    let removed = {
        let mut map = state.presence.lock().await;
        map.remove(&identity).is_some()
    };
    if removed {
        tracing::info!(identity = %identity, "presence: deregistered");
    } else {
        tracing::debug!(identity = %identity, "presence: deregister noop (already gone)");
    }
    Json(OkResponse { ok: true })
}

/// `GET /api/presence` — snapshot of every live presence entry.
pub async fn list(State(state): State<Arc<AppState>>) -> Json<Vec<PresenceEntry>> {
    let map = state.presence.lock().await;
    let mut entries: Vec<PresenceEntry> = map.values().cloned().collect();
    // Stable order so the cockpit doesn't reshuffle on every poll.
    entries.sort_by(|a, b| a.identity.cmp(&b.identity));
    Json(entries)
}

// ---------------------------------------------------------------------------
// Sweeper
// ---------------------------------------------------------------------------

/// Background task: every `SWEEPER_INTERVAL`, drop entries whose
/// `expires_at` has passed. Holds the mutex briefly — scan, collect
/// dead keys, remove. Spawned once from `build_app`.
pub async fn sweeper_loop(state: Arc<AppState>) {
    let mut ticker = tokio::time::interval(SWEEPER_INTERVAL);
    // Skip the immediate tick — nothing to sweep at startup.
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let now = now_iso();
        let removed: Vec<String> = {
            let mut map = state.presence.lock().await;
            // ISO 8601 UTC strings are lexicographically ordered, so a
            // simple `<` comparison works for "is this in the past?".
            let dead: Vec<String> = map
                .iter()
                .filter_map(|(k, v)| if v.expires_at < now { Some(k.clone()) } else { None })
                .collect();
            for k in &dead {
                map.remove(k);
            }
            dead
        };
        if !removed.is_empty() {
            tracing::info!(
                count = removed.len(),
                identities = ?removed,
                "presence: swept expired entries"
            );
        }
    }
}
