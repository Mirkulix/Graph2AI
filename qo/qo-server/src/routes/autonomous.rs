//! `POST/GET /api/autonomous/*` — autonomous swarm scheduler.
//!
//! Runs OrbitQO in unattended mode: every `interval_seconds` the scheduler
//! pops a goal from the user-managed FIFO queue (or asks the meta-agent to
//! invent one) and starts a fresh swarm via `swarm::start_swarm_internal`.
//! Hard caps — daily USD budget and per-day swarm count — are enforced at
//! every tick and are NEVER bypassed; once the budget is exceeded the
//! scheduler disables itself with `paused_reason="daily_budget_exceeded"`.
//!
//! Lifecycle:
//!
//!   /start ─▶ enabled=true, scheduler task spawned ONCE (idempotent —
//!             guarded by `autonomous_loop_started: AtomicBool`).
//!   tick   ─▶ every 10s: budget check, time check, pick goal, run swarm,
//!             wait for completion (max 5 min), persist run history.
//!   /stop  ─▶ enabled=false, paused_reason="user_stopped". The scheduler
//!             task itself stays alive but every tick is a cheap no-op.
//!
//! No new dependencies — uses tokio, serde, axum, qo_llm that the crate
//! already pulls in.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::routes::swarm::{self, StartSwarmRequest};
use crate::AppState;

// ---------------------------------------------------------------------------
// Defaults — conservative on purpose. Users opt into more aggressive limits
// by passing them explicitly to `/api/autonomous/start`.
// ---------------------------------------------------------------------------

const DEFAULT_INTERVAL_SECS: u64 = 1800; // 30 minutes
const DEFAULT_DAILY_BUDGET_USD: f64 = 0.50;
const DEFAULT_MAX_SWARMS_PER_HOUR: u32 = 4;
const DEFAULT_SWARM_MAX_ROUNDS: u32 = 2;
const DEFAULT_SWARM_MAX_TOKENS: u32 = 10_000;
const HISTORY_CAP: usize = 20;
const TICK_INTERVAL_SECS: u64 = 10;
const SWARM_WAIT_TIMEOUT_SECS: u64 = 300; // 5 min safety ceiling per swarm
const SWARM_POLL_INTERVAL_SECS: u64 = 5;

// ---------------------------------------------------------------------------
// State types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousConfig {
    #[serde(default = "default_interval_seconds")]
    pub interval_seconds: u64,
    #[serde(default = "default_daily_budget_usd")]
    pub daily_budget_usd: f64,
    #[serde(default = "default_max_swarms_per_hour")]
    pub max_swarms_per_hour: u32,
    #[serde(default)]
    pub goals_queue: Vec<String>,
    #[serde(default = "default_meta_agent_enabled")]
    pub meta_agent_enabled: bool,
    #[serde(default = "default_swarm_max_rounds")]
    pub swarm_max_rounds: u32,
    #[serde(default = "default_swarm_max_tokens")]
    pub swarm_max_tokens: u32,
}

fn default_interval_seconds() -> u64 { DEFAULT_INTERVAL_SECS }
fn default_daily_budget_usd() -> f64 { DEFAULT_DAILY_BUDGET_USD }
fn default_max_swarms_per_hour() -> u32 { DEFAULT_MAX_SWARMS_PER_HOUR }
fn default_meta_agent_enabled() -> bool { true }
fn default_swarm_max_rounds() -> u32 { DEFAULT_SWARM_MAX_ROUNDS }
fn default_swarm_max_tokens() -> u32 { DEFAULT_SWARM_MAX_TOKENS }

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            interval_seconds: DEFAULT_INTERVAL_SECS,
            daily_budget_usd: DEFAULT_DAILY_BUDGET_USD,
            max_swarms_per_hour: DEFAULT_MAX_SWARMS_PER_HOUR,
            goals_queue: Vec::new(),
            meta_agent_enabled: true,
            swarm_max_rounds: DEFAULT_SWARM_MAX_ROUNDS,
            swarm_max_tokens: DEFAULT_SWARM_MAX_TOKENS,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousRun {
    pub swarm_id: u64,
    pub goal: String,
    /// "queue" | "meta-agent"
    pub goal_source: String,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    pub cost_usd: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousState {
    pub enabled: bool,
    pub config: AutonomousConfig,
    pub started_at: Option<u64>,
    /// "daily_budget_exceeded" | "user_stopped" | "rate_limited" | …
    pub paused_reason: Option<String>,
    pub swarms_today: u32,
    pub spent_today_usd: f64,
    pub last_swarm_at: Option<u64>,
    pub next_swarm_at: Option<u64>,
    pub current_swarm_id: Option<u64>,
    /// UTC day-of-year * 10000 + year, used to detect a new day for the
    /// counter reset. 0 means "no swarm has run yet".
    #[serde(default)]
    pub last_day_key: u64,
    /// Last 20 runs (newest at the back).
    pub history: Vec<AutonomousRun>,
}

impl Default for AutonomousState {
    fn default() -> Self {
        Self {
            enabled: false,
            config: AutonomousConfig::default(),
            started_at: None,
            paused_reason: None,
            swarms_today: 0,
            spent_today_usd: 0.0,
            last_swarm_at: None,
            next_swarm_at: None,
            current_swarm_id: None,
            last_day_key: 0,
            history: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Request payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SetQueueRequest {
    pub goals: Vec<String>,
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// `POST /api/autonomous/start` — enable autonomous mode and spawn the
/// scheduler loop (once). Body is the full or partial `AutonomousConfig`;
/// missing fields fall back to defaults via serde.
pub async fn start_autonomous(
    State(state): State<Arc<AppState>>,
    Json(config): Json<AutonomousConfig>,
) -> Result<Json<AutonomousState>, (StatusCode, String)> {
    {
        let mut auto = state.autonomous.write().await;
        auto.config = config;
        auto.enabled = true;
        auto.paused_reason = None;
        if auto.started_at.is_none() {
            auto.started_at = Some(now_secs());
        }
        tracing::info!(
            interval_seconds = auto.config.interval_seconds,
            daily_budget_usd = auto.config.daily_budget_usd,
            max_swarms_per_hour = auto.config.max_swarms_per_hour,
            queue_len = auto.config.goals_queue.len(),
            meta_agent_enabled = auto.config.meta_agent_enabled,
            "autonomous: enabled"
        );
    }

    // Idempotent spawn — only the first /start call gets the loop.
    if !state.autonomous_loop_started.swap(true, Ordering::SeqCst) {
        let state_clone = state.clone();
        tokio::spawn(async move {
            scheduler_loop(state_clone).await;
        });
        tracing::info!("autonomous: scheduler loop spawned");
    } else {
        tracing::info!("autonomous: scheduler loop already running — config updated in place");
    }

    let snapshot = state.autonomous.read().await.clone();
    Ok(Json(snapshot))
}

/// `POST /api/autonomous/stop` — disable autonomous mode. The scheduler
/// task stays alive but goes idle.
pub async fn stop_autonomous(
    State(state): State<Arc<AppState>>,
) -> Json<AutonomousState> {
    {
        let mut auto = state.autonomous.write().await;
        auto.enabled = false;
        auto.paused_reason = Some("user_stopped".to_string());
        tracing::info!("autonomous: stopped by user");
    }
    let snapshot = state.autonomous.read().await.clone();
    Json(snapshot)
}

/// `GET /api/autonomous/status` — full state snapshot.
pub async fn get_status(
    State(state): State<Arc<AppState>>,
) -> Json<AutonomousState> {
    let snapshot = state.autonomous.read().await.clone();
    Json(snapshot)
}

/// `PUT /api/autonomous/queue` — replace the goal queue (FIFO). Empty
/// strings are dropped so `["", "foo"]` → `["foo"]`.
pub async fn set_queue(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetQueueRequest>,
) -> Json<AutonomousState> {
    let cleaned: Vec<String> = req
        .goals
        .into_iter()
        .map(|g| g.trim().to_string())
        .filter(|g| !g.is_empty())
        .collect();
    {
        let mut auto = state.autonomous.write().await;
        auto.config.goals_queue = cleaned;
        tracing::info!(
            queue_len = auto.config.goals_queue.len(),
            "autonomous: goal queue replaced"
        );
    }
    let snapshot = state.autonomous.read().await.clone();
    Json(snapshot)
}

// ---------------------------------------------------------------------------
// Scheduler loop — single global tokio task, spawned at first /start.
// Lock-discipline: all `state.autonomous` mutations happen inside a
// short critical section; we never hold the write guard across an
// `.await` on the LLM call.
// ---------------------------------------------------------------------------

async fn scheduler_loop(state: Arc<AppState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(TICK_INTERVAL_SECS)).await;

        // Snapshot the parts we need for this tick.
        let (
            enabled,
            current_swarm,
            last_swarm_at,
            interval_secs,
            spent_today,
            daily_budget,
            swarms_today,
            max_per_hour,
            queue_first,
            meta_enabled,
            swarm_max_rounds,
            swarm_max_tokens,
            last_day_key,
        ) = {
            let auto = state.autonomous.read().await;
            (
                auto.enabled,
                auto.current_swarm_id,
                auto.last_swarm_at,
                auto.config.interval_seconds,
                auto.spent_today_usd,
                auto.config.daily_budget_usd,
                auto.swarms_today,
                auto.config.max_swarms_per_hour,
                auto.config.goals_queue.first().cloned(),
                auto.config.meta_agent_enabled,
                auto.config.swarm_max_rounds,
                auto.config.swarm_max_tokens,
                auto.last_day_key,
            )
        };

        if !enabled {
            tracing::debug!("autonomous tick: disabled — skip");
            continue;
        }

        // A swarm is still running — give it space.
        if current_swarm.is_some() {
            tracing::debug!(
                current = ?current_swarm,
                "autonomous tick: swarm still in flight — skip"
            );
            continue;
        }

        let now = now_secs();
        let today_key = utc_day_key(now);

        // Daily reset: counters roll over the moment we observe a new UTC day.
        if last_day_key != 0 && today_key != last_day_key {
            let mut auto = state.autonomous.write().await;
            tracing::info!(
                old_day = last_day_key,
                new_day = today_key,
                spent_yesterday = auto.spent_today_usd,
                swarms_yesterday = auto.swarms_today,
                "autonomous tick: new UTC day — resetting daily counters"
            );
            auto.swarms_today = 0;
            auto.spent_today_usd = 0.0;
            auto.last_day_key = today_key;
            // Re-read after mutation so the rest of this tick sees fresh values.
            // (We continue rather than complicate the snapshot dance.)
            continue;
        }

        // Budget guard — the daily USD cap is sacrosanct.
        if spent_today >= daily_budget {
            let mut auto = state.autonomous.write().await;
            auto.enabled = false;
            auto.paused_reason = Some("daily_budget_exceeded".to_string());
            tracing::warn!(
                spent_today_usd = spent_today,
                daily_budget_usd = daily_budget,
                "autonomous tick: daily budget exceeded — pausing"
            );
            continue;
        }

        // Hard rate limit: max_swarms_per_hour * 24 across the day.
        let daily_swarm_cap = max_per_hour.saturating_mul(24);
        if swarms_today >= daily_swarm_cap {
            tracing::debug!(
                swarms_today,
                daily_swarm_cap,
                "autonomous tick: per-day swarm cap reached — skip"
            );
            continue;
        }

        // Time guard — respect the user-configured cooldown.
        if let Some(last) = last_swarm_at {
            if now < last + interval_secs {
                let next = last + interval_secs;
                {
                    let mut auto = state.autonomous.write().await;
                    auto.next_swarm_at = Some(next);
                }
                tracing::debug!(
                    last_swarm_at = last,
                    next_swarm_at = next,
                    "autonomous tick: still in cooldown — skip"
                );
                continue;
            }
        }

        // ── Pick a goal ────────────────────────────────────────────
        let (goal, source) = if let Some(g) = queue_first {
            (g, "queue".to_string())
        } else if meta_enabled {
            match call_meta_agent(&state).await {
                Ok(g) if !g.trim().is_empty() => (g, "meta-agent".to_string()),
                Ok(_) => {
                    tracing::warn!("autonomous tick: meta-agent returned empty goal — skip");
                    continue;
                }
                Err(e) => {
                    tracing::warn!("autonomous tick: meta-agent failed: {e} — skip");
                    continue;
                }
            }
        } else {
            tracing::debug!(
                "autonomous tick: queue empty and meta-agent disabled — skip"
            );
            continue;
        };

        tracing::info!(
            goal = %goal,
            source = %source,
            "autonomous tick: goal selected"
        );

        // Pop the queue if that's where the goal came from.
        if source == "queue" {
            let mut auto = state.autonomous.write().await;
            if !auto.config.goals_queue.is_empty() {
                auto.config.goals_queue.remove(0);
            }
        }

        // ── Start the swarm ────────────────────────────────────────
        let req = StartSwarmRequest {
            goal: goal.clone(),
            max_rounds: Some(swarm_max_rounds),
            max_tokens: Some(swarm_max_tokens),
            // Mark this as scheduler-originated so the post-swarm git +
            // build/test pipeline runs on completion. Manual swarms
            // started from the cockpit leave this field at its default
            // (`false`) and skip the pipeline.
            auto_originated: true,
        };
        let swarm_id = match swarm::start_swarm_internal(state.clone(), req).await {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!("autonomous tick: failed to start swarm: {e}");
                continue;
            }
        };
        tracing::info!(swarm_id, "autonomous tick: swarm started");

        // Record the run + bookkeeping in one short critical section.
        {
            let mut auto = state.autonomous.write().await;
            auto.current_swarm_id = Some(swarm_id);
            auto.last_swarm_at = Some(now);
            auto.last_day_key = today_key;
            auto.swarms_today += 1;
            auto.next_swarm_at = Some(now + auto.config.interval_seconds);
            auto.history.push(AutonomousRun {
                swarm_id,
                goal: goal.clone(),
                goal_source: source.clone(),
                started_at: now,
                finished_at: None,
                cost_usd: 0.0,
                status: "running".to_string(),
            });
            while auto.history.len() > HISTORY_CAP {
                auto.history.remove(0);
            }
        }

        // ── Wait for the swarm to finish (with a 5-min ceiling) ────
        let waited = wait_for_swarm(&state, swarm_id).await;

        // Update history + clear current_swarm_id.
        {
            let mut auto = state.autonomous.write().await;
            if let Some(run) = auto
                .history
                .iter_mut()
                .rev()
                .find(|r| r.swarm_id == swarm_id)
            {
                run.cost_usd = waited.cost_usd;
                run.finished_at = Some(now_secs());
                run.status = waited.status.clone();
            }
            auto.spent_today_usd += waited.cost_usd;
            auto.current_swarm_id = None;
            tracing::info!(
                swarm_id,
                status = %waited.status,
                cost_usd = waited.cost_usd,
                spent_today_usd = auto.spent_today_usd,
                "autonomous tick: swarm finished — accounting updated"
            );
        }
    }
}

/// Result of waiting for a swarm — captures the final cost and status so
/// the scheduler can persist them in `AutonomousRun` history.
struct SwarmOutcome {
    cost_usd: f64,
    status: String,
}

async fn wait_for_swarm(state: &Arc<AppState>, swarm_id: u64) -> SwarmOutcome {
    let start = now_secs();
    loop {
        tokio::time::sleep(Duration::from_secs(SWARM_POLL_INTERVAL_SECS)).await;
        let snapshot = {
            let map = state.swarms.read().await;
            map.get(&swarm_id).cloned()
        };
        match snapshot {
            Some(s) => {
                if matches!(s.status.as_str(), "done" | "stopped" | "error") {
                    return SwarmOutcome {
                        cost_usd: s.cost_usd,
                        status: s.status,
                    };
                }
                if now_secs().saturating_sub(start) > SWARM_WAIT_TIMEOUT_SECS {
                    tracing::warn!(
                        swarm_id,
                        elapsed = now_secs().saturating_sub(start),
                        "autonomous tick: swarm wait timed out — recording last-known cost"
                    );
                    return SwarmOutcome {
                        cost_usd: s.cost_usd,
                        status: format!("timeout({})", s.status),
                    };
                }
            }
            None => {
                // Swarm vanished — nothing to charge or report beyond a marker.
                return SwarmOutcome {
                    cost_usd: 0.0,
                    status: "missing".to_string(),
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Meta-agent — asks DeepSeek to invent a single bounded improvement task
// when the user-managed queue is empty. Free-form text, no JSON parsing.
// ---------------------------------------------------------------------------

async fn call_meta_agent(state: &Arc<AppState>) -> Result<String, String> {
    let prompt = "You are the autonomous improvement director for the OrbitQO project — \
        a Rust-based AI-to-AI graph protocol with React cockpit.\n\
        \n\
        Suggest ONE concrete, bounded improvement task that could be done in a single \
        2-round agent swarm. Avoid: vague refactors, anything requiring user input, \
        anything requiring new dependencies, anything > 200 LOC of code.\n\
        \n\
        Good examples:\n\
        - 'Identify 3 functions in qo/qo-server/src/routes/consensus.rs that could benefit \
          from clearer error messages, propose the new messages.'\n\
        - 'Review frontend/src/lib/sse.ts for testability gaps and suggest mockable seams.'\n\
        - 'Audit qo/qo-server/src/tools.rs for path-traversal sandbox edge cases.'\n\
        \n\
        Bad examples:\n\
        - 'Refactor the codebase' (too vague)\n\
        - 'Add a new feature' (needs user input)\n\
        - 'Improve performance' (no specific target)\n\
        \n\
        Respond with ONLY the goal text, no preamble, no quotes, no JSON.";

    let msgs: Vec<(String, String)> = vec![
        ("system".to_string(), prompt.to_string()),
        ("user".to_string(), "Pick the next goal.".to_string()),
    ];
    let (response, _tier_used) = state
        .llm
        .chat_with_model(
            Some(qo_llm::Tier::DeepSeek),
            Some("deepseek-reasoner".to_string()),
            msgs,
        )
        .await
        .map_err(|e| e.to_string())?;
    Ok(response.trim().to_string())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Cheap UTC day key: the integer number of days since the Unix epoch.
/// Two epoch-seconds values produce the same key iff they fall on the
/// same UTC calendar day, which is exactly what the daily-reset logic
/// needs without dragging chrono in.
fn utc_day_key(epoch_secs: u64) -> u64 {
    epoch_secs / 86_400
}

// ---------------------------------------------------------------------------
// Tests — pure helpers, no async runtime needed.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_conservative() {
        let cfg = AutonomousConfig::default();
        assert_eq!(cfg.interval_seconds, 1800);
        assert_eq!(cfg.daily_budget_usd, 0.50);
        assert_eq!(cfg.max_swarms_per_hour, 4);
        assert_eq!(cfg.swarm_max_rounds, 2);
        assert_eq!(cfg.swarm_max_tokens, 10_000);
        assert!(cfg.meta_agent_enabled);
        assert!(cfg.goals_queue.is_empty());
    }

    #[test]
    fn state_default_is_disabled() {
        let s = AutonomousState::default();
        assert!(!s.enabled);
        assert!(s.started_at.is_none());
        assert_eq!(s.swarms_today, 0);
        assert_eq!(s.spent_today_usd, 0.0);
        assert!(s.history.is_empty());
    }

    #[test]
    fn day_key_changes_at_utc_midnight() {
        // Pick an arbitrary UTC midnight.
        let midnight = 1_700_000_000u64 - (1_700_000_000u64 % 86_400);
        let just_before = utc_day_key(midnight - 1);
        let at_midnight = utc_day_key(midnight);
        let later_same_day = utc_day_key(midnight + 86_399);
        let next_day = utc_day_key(midnight + 86_400);
        assert_ne!(just_before, at_midnight);
        assert_eq!(at_midnight, later_same_day);
        assert_ne!(at_midnight, next_day);
    }

    #[test]
    fn partial_config_uses_defaults_via_serde() {
        // Empty JSON object → all defaults populated.
        let cfg: AutonomousConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.interval_seconds, 1800);
        assert_eq!(cfg.daily_budget_usd, 0.50);
        assert!(cfg.meta_agent_enabled);
    }

    #[test]
    fn partial_config_overrides_only_provided_fields() {
        let cfg: AutonomousConfig = serde_json::from_str(
            r#"{ "interval_seconds": 60, "daily_budget_usd": 1.5 }"#,
        )
        .unwrap();
        assert_eq!(cfg.interval_seconds, 60);
        assert_eq!(cfg.daily_budget_usd, 1.5);
        // Untouched fields keep their defaults.
        assert_eq!(cfg.max_swarms_per_hour, 4);
        assert_eq!(cfg.swarm_max_rounds, 2);
    }
}
