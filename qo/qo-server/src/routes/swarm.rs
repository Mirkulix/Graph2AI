//! POST /api/swarm/* — autonomous multi-agent swarm orchestrator.
//!
//! A swarm is a finite, cost-capped, multi-round conversation between the
//! 6 server-side agents (and any IDE identities currently in the
//! presence registry). The strategist drives planning and evaluation; the
//! other agents execute subtasks in parallel. Each LLM call goes through
//! the per-agent (tier, model) mapping in `agent_models` so the swarm
//! benefits from the DeepSeek-coder / DeepSeek-reasoner / Ollama mix.
//!
//! Lifecycle (per swarm):
//!
//!   start_swarm  ── inserts SwarmState (status=planning), spawns task
//!         │
//!         ▼
//!   round 1..=max_rounds:
//!     ├─ PLANNING:  strategist breaks goal into 3-5 JSON subtasks
//!     ├─ DISPATCH:  parallel tokio tasks, one per subtask
//!     │              - server agents → direct LlmRouter call
//!     │              - IDE identities → bus.send(Execute) + 10s wait
//!     └─ EVAL:      strategist returns { converged, comments }
//!         │
//!         ▼
//!   status=done | stopped | error,  finished_at=now
//!
//! Hard caps:
//!   - max_tokens (default 20_000) — checked at every round boundary
//!   - stop_requested flag — same check
//!   - 60s per-round dispatch wait
//!   - 10s per-IDE-subtask wait
//!
//! No new dependencies — uses tokio, serde, axum, qo_llm, qlang_agent
//! that are already pulled in by the crate.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use qlang_agent::protocol::{AgentId, Capability, GraphMessage, MessageIntent};
use qlang_core::graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap as StdHashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use crate::agent_models;
use crate::git_ops;
use crate::routes::llm_proxy::call_claude_cli_agent;
use crate::AppState;

// ---------------------------------------------------------------------------
// State types — these are public because AppState holds a HashMap<u64, ..>.
// ---------------------------------------------------------------------------

/// One LLM/IDE call within a swarm round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmSubtask {
    pub id: String,
    pub assigned_to: String,
    pub prompt: String,
    pub response: Option<String>,
    pub status: String, // pending | running | done | error
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
}

/// One full plan/dispatch/eval cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmRound {
    pub round_number: u32,
    pub plan: String,
    pub subtasks: Vec<SwarmSubtask>,
    pub eval: Option<String>,
}

/// Full swarm state — read by `/api/swarm/{id}`, mutated by the
/// orchestrator task. Held inside an `Arc<RwLock<HashMap<u64, _>>>`
/// in `AppState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    pub id: u64,
    pub goal: String,
    pub status: String, // planning | dispatching | evaluating | done | stopped | error
    pub rounds: Vec<SwarmRound>,
    pub tokens_used: u64,
    pub cost_usd: f64,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    pub stop_requested: bool,
    /// True when the autonomous scheduler started this swarm. Manual
    /// `/api/swarm/start` calls leave this false. The post-swarm git/
    /// test pipeline only runs for auto-originated swarms.
    #[serde(default)]
    pub auto_originated: bool,
    /// Branch produced by the post-swarm git commit, if any.
    #[serde(default)]
    pub branch_name: Option<String>,
    /// Result of `cargo build` + `tsc --noEmit` on that branch.
    #[serde(default)]
    pub tests_passed: Option<bool>,
    #[serde(default)]
    pub diff_files_changed: Option<u32>,
    #[serde(default)]
    pub diff_insertions: Option<u32>,
    #[serde(default)]
    pub diff_deletions: Option<u32>,
}

// ---------------------------------------------------------------------------
// Request / response payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct StartSwarmRequest {
    pub goal: String,
    #[serde(default)]
    pub max_rounds: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Internal flag. Set by the autonomous scheduler so the
    /// orchestrator knows to run the post-swarm git+test pipeline at
    /// completion. Manual swarms leave this `false`/missing.
    #[serde(default)]
    pub auto_originated: bool,
}

#[derive(Debug, Serialize)]
pub struct StartSwarmResponse {
    pub swarm_id: u64,
}

#[derive(Debug, Serialize)]
pub struct StopSwarmResponse {
    pub ok: bool,
    pub status: String,
}

// ---------------------------------------------------------------------------
// ID generation — monotonic counter seeded with current epoch ms so
// restarts don't reuse old IDs that the cockpit might still poll.
// ---------------------------------------------------------------------------

static SWARM_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_swarm_id() -> u64 {
    let base = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let seq = SWARM_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    (base << 16) | (seq & 0xFFFF)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Cost model — derived from `qo_llm::config::provider_templates()`.
// Returns USD cost per 1k tokens for the (tier, model) combo. We don't
// hold a reference to ProviderTemplate at runtime because the templates
// table lives in qo-llm and is keyed by id strings; instead we hard-code
// the cost values that match the templates to avoid a cross-crate lookup
// in the hot path. If the templates change, update here too.
// ---------------------------------------------------------------------------

fn cost_per_1k_for(tier: qo_llm::Tier, model: &Option<String>) -> f64 {
    use qo_llm::Tier;
    match tier {
        Tier::Local => 0.0, // Ollama is free
        Tier::Groq => 0.0,  // Groq free tier
        Tier::DeepSeek => match model.as_deref() {
            Some("deepseek-coder") => 0.00014,
            Some("deepseek-reasoner") => 0.00055,
            Some("deepseek-chat") | None => 0.00027,
            _ => 0.00027,
        },
        Tier::Cloud => 0.01, // generic OpenAI-compatible upper bound
    }
}

/// Estimate token count from a UTF-8 string. ~4 chars per token is the
/// industry-standard rough rule used by every other place in the codebase
/// (see `cost_tracker` updates in `router.rs`).
fn estimate_tokens(text: &str) -> u64 {
    (text.len() / 4) as u64
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

/// `POST /api/swarm/start` — kick off a new swarm.
pub async fn start_swarm(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartSwarmRequest>,
) -> Result<Json<StartSwarmResponse>, (StatusCode, String)> {
    let id = start_swarm_internal(state, req)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(StartSwarmResponse { swarm_id: id }))
}

/// Direct entry point for the swarm orchestrator — same work as the HTTP
/// handler, but callable from non-HTTP contexts (e.g. the autonomous
/// scheduler loop). Returns the new swarm id on success.
pub async fn start_swarm_internal(
    state: Arc<AppState>,
    req: StartSwarmRequest,
) -> Result<u64, String> {
    if req.goal.trim().is_empty() {
        return Err("goal must be non-empty".to_string());
    }
    let max_rounds = req.max_rounds.unwrap_or(3).max(1);
    let max_tokens: u64 = req.max_tokens.unwrap_or(20_000).into();

    let id = next_swarm_id();
    let initial = SwarmState {
        id,
        goal: req.goal.clone(),
        status: "planning".to_string(),
        rounds: Vec::new(),
        tokens_used: 0,
        cost_usd: 0.0,
        started_at: now_secs(),
        finished_at: None,
        stop_requested: false,
        auto_originated: req.auto_originated,
        branch_name: None,
        tests_passed: None,
        diff_files_changed: None,
        diff_insertions: None,
        diff_deletions: None,
    };

    {
        let mut map = state.swarms.write().await;
        map.insert(id, initial);
    }

    tracing::info!(
        swarm_id = id,
        goal = %req.goal,
        max_rounds,
        max_tokens,
        "swarm: started"
    );

    let state_clone = state.clone();
    let goal_clone = req.goal.clone();
    tokio::spawn(async move {
        run_swarm(state_clone, id, goal_clone, max_rounds, max_tokens).await;
    });

    Ok(id)
}

/// `GET /api/swarm/{id}` — return the full SwarmState snapshot.
pub async fn get_swarm(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<SwarmState>, (StatusCode, String)> {
    let map = state.swarms.read().await;
    match map.get(&id) {
        Some(s) => Ok(Json(s.clone())),
        None => Err((StatusCode::NOT_FOUND, format!("swarm {} not found", id))),
    }
}

/// `POST /api/swarm/{id}/stop` — set the stop flag. The orchestrator
/// polls it at every round boundary and exits with status="stopped".
pub async fn stop_swarm(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<StopSwarmResponse>, (StatusCode, String)> {
    let mut map = state.swarms.write().await;
    match map.get_mut(&id) {
        Some(s) => {
            s.stop_requested = true;
            tracing::info!(swarm_id = id, "swarm: stop requested");
            Ok(Json(StopSwarmResponse {
                ok: true,
                status: s.status.clone(),
            }))
        }
        None => Err((StatusCode::NOT_FOUND, format!("swarm {} not found", id))),
    }
}

/// `GET /api/swarm/active` — list swarms that are not done/stopped/error.
pub async fn list_active(State(state): State<Arc<AppState>>) -> Json<Vec<SwarmState>> {
    let map = state.swarms.read().await;
    let mut active: Vec<SwarmState> = map
        .values()
        .filter(|s| {
            !matches!(
                s.status.as_str(),
                "done" | "stopped" | "error"
            )
        })
        .cloned()
        .collect();
    // Newest first so the cockpit sees the most relevant swarm at the top.
    active.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    Json(active)
}

// ---------------------------------------------------------------------------
// Orchestrator — runs in a tokio task. All mutations to `SwarmState`
// take the RwLock briefly, copy out, mutate, write back. We never hold
// the lock across an `.await` on the LLM call.
// ---------------------------------------------------------------------------

async fn run_swarm(
    state: Arc<AppState>,
    swarm_id: u64,
    _goal: String,
    max_rounds: u32,
    max_tokens: u64,
) {
    let known_agents = ["developer", "researcher", "guardian", "strategist", "artisan", "ceo"];

    for round_no in 1..=max_rounds {
        // ── Cap checks ────────────────────────────────────────────
        if !precheck_continue(&state.swarms, swarm_id, max_tokens).await {
            break;
        }

        set_status(&state.swarms, swarm_id, "planning").await;

        // ── PLANNING: strategist returns JSON subtasks ────────────
        let prior_summary = summarize_prior_rounds(&state.swarms, swarm_id).await;
        let ide_list = collect_ide_list(&state).await;
        let plan_prompt = build_plan_prompt(
            &goal_of(&state.swarms, swarm_id).await,
            &prior_summary,
            &known_agents,
            &ide_list,
        );

        tracing::info!(
            swarm_id,
            round = round_no,
            "swarm: planning round"
        );

        let plan_text = match llm_call(
            &state,
            swarm_id,
            "strategist",
            "You are the strategist for an autonomous multi-agent swarm. \
             Break work into concrete subtasks. ALWAYS reply with strict JSON.",
            &plan_prompt,
        )
        .await
        {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(swarm_id, round = round_no, "planning LLM failed: {e}");
                push_error_round(&state.swarms, swarm_id, round_no, format!("plan failed: {e}"))
                    .await;
                continue;
            }
        };

        let parsed = parse_subtasks(&plan_text);
        let subtasks: Vec<SwarmSubtask> = match parsed {
            Ok(items) if !items.is_empty() => items
                .into_iter()
                .enumerate()
                .map(|(i, (assigned_to, prompt))| SwarmSubtask {
                    id: format!("r{}.{}", round_no, i + 1),
                    assigned_to,
                    prompt,
                    response: None,
                    status: "pending".to_string(),
                    started_at: None,
                    finished_at: None,
                })
                .collect(),
            Ok(_) => {
                tracing::warn!(swarm_id, round = round_no, "planner produced 0 subtasks");
                push_error_round(
                    &state.swarms,
                    swarm_id,
                    round_no,
                    format!("planner produced no subtasks. raw: {}", trim_for_log(&plan_text)),
                )
                .await;
                continue;
            }
            Err(e) => {
                tracing::warn!(swarm_id, round = round_no, "subtask JSON parse failed: {e}");
                push_error_round(
                    &state.swarms,
                    swarm_id,
                    round_no,
                    format!("plan JSON parse failed: {e}. raw: {}", trim_for_log(&plan_text)),
                )
                .await;
                continue;
            }
        };

        // Insert the round shell so the frontend can see "round 2 dispatching".
        {
            let mut map = state.swarms.write().await;
            if let Some(s) = map.get_mut(&swarm_id) {
                s.rounds.push(SwarmRound {
                    round_number: round_no,
                    plan: plan_text.clone(),
                    subtasks: subtasks.clone(),
                    eval: None,
                });
            }
        }

        set_status(&state.swarms, swarm_id, "dispatching").await;

        // ── DISPATCH: spawn one task per subtask ──────────────────
        let mut handles = Vec::with_capacity(subtasks.len());
        for st in &subtasks {
            let state_inner = state.clone();
            let subtask = st.clone();
            handles.push(tokio::spawn(async move {
                run_subtask(state_inner, swarm_id, round_no, subtask).await;
            }));
        }

        // 60s ceiling for the whole round's dispatch.
        let dispatch_timeout = Duration::from_secs(60);
        let _ = tokio::time::timeout(dispatch_timeout, async {
            for h in handles {
                let _ = h.await;
            }
        })
        .await;

        // ── EVAL: ask strategist if we converged ──────────────────
        if !precheck_continue(&state.swarms, swarm_id, max_tokens).await {
            break;
        }
        set_status(&state.swarms, swarm_id, "evaluating").await;

        let eval_prompt = build_eval_prompt(&state.swarms, swarm_id, round_no).await;
        let eval_text = match llm_call(
            &state,
            swarm_id,
            "strategist",
            "You are the strategist. Evaluate the round results vs the goal. ALWAYS reply with strict JSON.",
            &eval_prompt,
        )
        .await
        {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(swarm_id, round = round_no, "eval LLM failed: {e}");
                format!("{{\"converged\": false, \"comments\": \"eval failed: {}\"}}", e)
            }
        };

        // Persist the eval text on the round.
        {
            let mut map = state.swarms.write().await;
            if let Some(s) = map.get_mut(&swarm_id) {
                if let Some(round) = s.rounds.iter_mut().find(|r| r.round_number == round_no) {
                    round.eval = Some(eval_text.clone());
                }
            }
        }

        let converged = parse_converged(&eval_text);
        tracing::info!(
            swarm_id,
            round = round_no,
            converged,
            "swarm: round complete"
        );
        if converged {
            break;
        }
    }

    // ── Post-swarm git + test pipeline (auto-originated only) ────
    // We do this BEFORE flipping status to "done" so a slow build doesn't
    // race the autonomous scheduler's wait_for_swarm poll loop.
    let (auto_originated, goal_for_git) = {
        let map = state.swarms.read().await;
        match map.get(&swarm_id) {
            Some(s) => (s.auto_originated, s.goal.clone()),
            None => (false, String::new()),
        }
    };
    if auto_originated {
        run_post_swarm_git_pipeline(&state, swarm_id, &goal_for_git).await;
    }

    // ── Finalize ────────────────────────────────────────────────
    let mut map = state.swarms.write().await;
    if let Some(s) = map.get_mut(&swarm_id) {
        if s.stop_requested && s.status != "done" {
            s.status = "stopped".to_string();
        } else if s.status != "error" {
            s.status = "done".to_string();
        }
        s.finished_at = Some(now_secs());
        tracing::info!(
            swarm_id,
            status = %s.status,
            tokens_used = s.tokens_used,
            cost_usd = s.cost_usd,
            branch_name = ?s.branch_name,
            tests_passed = ?s.tests_passed,
            "swarm: finished"
        );
    }
}

// ---------------------------------------------------------------------------
// Post-swarm git + test pipeline
//
// Runs only for auto-originated swarms. Captures whatever the agents
// edited into a fresh `auto/<slug>-<unix>` branch, runs the standard
// build + tsc check, and parks the result on the SwarmState so the
// cockpit can review and merge or discard it.
//
// Failure-tolerant: every git/test error is logged via `tracing::warn`
// and folded back to a no-op for that step. We never panic the swarm.
// ---------------------------------------------------------------------------

async fn run_post_swarm_git_pipeline(state: &Arc<AppState>, swarm_id: u64, goal: &str) {
    // Step 1 — anything to commit?
    let dirty = match git_ops::has_changes().await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(swarm_id, "post-swarm: has_changes failed: {}", e);
            return;
        }
    };
    if !dirty {
        tracing::info!(swarm_id, "post-swarm: no changes — skipping branch/commit");
        return;
    }

    // Remember where we were so we can return after the build.
    let original_branch = match git_ops::current_branch().await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(swarm_id, "post-swarm: current_branch failed: {}", e);
            return;
        }
    };

    // Step 2 — branch + commit.
    let slug = goal_slug(goal);
    let unix = now_secs();
    let branch_name = format!("auto/{}-{}", slug, unix);
    if let Err(e) = git_ops::create_branch(&branch_name).await {
        tracing::warn!(swarm_id, "post-swarm: create_branch '{}' failed: {}", branch_name, e);
        return;
    }
    let goal_short: String = goal.chars().take(60).collect();
    let commit_msg = format!("auto: {} (swarm #{})", goal_short, swarm_id);
    let commit_sha = match git_ops::commit_all(&commit_msg).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(swarm_id, "post-swarm: commit failed: {}", e);
            // Try to flip back so we don't strand the user on the auto branch.
            let _ = git_ops::checkout(&original_branch).await;
            return;
        }
    };
    tracing::info!(
        swarm_id,
        branch = %branch_name,
        sha = %commit_sha,
        "post-swarm: committed agent edits"
    );

    // Step 3 — diff summary vs the branch we came from (best-effort).
    let summary = git_ops::diff_summary(&branch_name, &original_branch).await.ok();

    // Step 4 — build + tsc. Long-running; do NOT hold the swarm lock.
    let test_result = git_ops::run_build_and_test().await;
    tracing::info!(
        swarm_id,
        branch = %branch_name,
        passed = test_result.success,
        duration_ms = test_result.duration_ms,
        "post-swarm: build+test complete"
    );

    // Step 5 — record onto SwarmState.
    {
        let mut map = state.swarms.write().await;
        if let Some(s) = map.get_mut(&swarm_id) {
            s.branch_name = Some(branch_name.clone());
            s.tests_passed = Some(test_result.success);
            if let Some(ref ds) = summary {
                s.diff_files_changed = Some(ds.files_changed);
                s.diff_insertions = Some(ds.insertions);
                s.diff_deletions = Some(ds.deletions);
            }
        }
    }

    // Step 6 — return to the original branch so the user's working
    // copy is exactly where it was before the swarm started.
    if let Err(e) = git_ops::checkout(&original_branch).await {
        tracing::warn!(
            swarm_id,
            "post-swarm: failed to checkout back to '{}': {}",
            original_branch,
            e
        );
    }
}

/// Build a filesystem/branch-safe slug from the goal: lowercase, max 40
/// chars, non-alphanumerics collapsed to `-`, no leading/trailing dashes.
fn goal_slug(goal: &str) -> String {
    let mut out = String::with_capacity(40);
    let mut last_dash = true; // suppress leading dashes
    for ch in goal.chars().take(40) {
        let lc = ch.to_ascii_lowercase();
        if lc.is_ascii_alphanumeric() {
            out.push(lc);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("swarm");
    }
    out
}

// ---------------------------------------------------------------------------
// Per-subtask runner
// ---------------------------------------------------------------------------

async fn run_subtask(
    state: Arc<AppState>,
    swarm_id: u64,
    round_no: u32,
    subtask: SwarmSubtask,
) {
    // Mark running.
    update_subtask(&state.swarms, swarm_id, round_no, &subtask.id, |s| {
        s.status = "running".to_string();
        s.started_at = Some(now_secs());
    })
    .await;

    let assigned = subtask.assigned_to.clone();
    let prompt = subtask.prompt.clone();

    // Decide route: known server agent → direct LLM; otherwise treat as
    // an IDE identity registered in presence and round-trip via the bus.
    let is_server_agent = matches!(
        assigned.as_str(),
        "developer" | "researcher" | "guardian" | "strategist" | "artisan" | "ceo"
    );

    let result = if is_server_agent {
        // Server-side agents now run as full Claude Code sessions: each
        // subtask becomes a real agentic investigation with Read / Edit /
        // Write / Bash / Grep / Glob scoped to the OrbitQLang repo. The
        // per-agent (tier, model) mapping in `agent_models` still drives
        // non-swarm paths (mailbox loop in lib.rs); inside the swarm the
        // model is fixed to "sonnet" for now.
        run_server_agent_via_claude_cli(&state, swarm_id, &assigned, &prompt).await
    } else {
        ide_dispatch(&state, &assigned, &prompt).await
    };

    let (status, response_text) = match result {
        Ok(text) => ("done".to_string(), text),
        Err(e) => ("error".to_string(), format!("[error] {}", e)),
    };

    update_subtask(&state.swarms, swarm_id, round_no, &subtask.id, |s| {
        s.status = status.clone();
        s.response = Some(response_text.clone());
        s.finished_at = Some(now_secs());
    })
    .await;
}

// ---------------------------------------------------------------------------
// LLM call wrapper — applies per-agent (tier, model) mapping AND
// updates SwarmState.tokens_used / cost_usd in one shot. Never holds
// the lock across the `await`.
// ---------------------------------------------------------------------------

async fn llm_call(
    state: &Arc<AppState>,
    swarm_id: u64,
    agent: &str,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<String, String> {
    let (tier, model) = agent_models::model_for_agent(agent);
    let messages = vec![
        ("system".to_string(), system_prompt.to_string()),
        ("user".to_string(), user_prompt.to_string()),
    ];

    let res = state
        .llm
        .chat_with_model(Some(tier), model.clone(), messages)
        .await
        .map_err(|e| e.to_string())?;
    let (text, _tier_used) = res;

    // Charge tokens against the swarm budget. We approximate input + output
    // cost together using the system+user+response length so a long prompt
    // burns budget too.
    let total_chars = system_prompt.len() + user_prompt.len() + text.len();
    let tokens = (total_chars / 4) as u64;
    let cost = (tokens as f64 / 1000.0) * cost_per_1k_for(tier, &model);

    {
        let mut map = state.swarms.write().await;
        if let Some(s) = map.get_mut(&swarm_id) {
            s.tokens_used = s.tokens_used.saturating_add(tokens);
            s.cost_usd += cost;
        }
    }

    tracing::info!(
        swarm_id,
        agent,
        ?tier,
        ?model,
        tokens,
        chars = text.len(),
        "swarm: agent reply"
    );

    Ok(text)
}

// ---------------------------------------------------------------------------
// Server-agent dispatch via claude-cli-agent — spawns a full Claude Code
// session with Read/Edit/Write/Bash/Grep/Glob scoped to the OrbitQLang
// repo. The per-call cost is bounded by the CLI's `--max-budget-usd`
// flag; we still charge an estimated token count against the swarm
// budget so the existing token cap still terminates a runaway swarm.
// ---------------------------------------------------------------------------

async fn run_server_agent_via_claude_cli(
    state: &Arc<AppState>,
    swarm_id: u64,
    agent: &str,
    user_prompt: &str,
) -> Result<String, String> {
    let res = call_claude_cli_agent("sonnet", None, user_prompt).await?;

    // Charge an estimated token count so the swarm token-cap still bites
    // even though we no longer route through the LlmRouter. Cost is left
    // at 0 because the CLI handles its own per-call USD budget.
    let tokens = estimate_tokens(user_prompt) + estimate_tokens(&res);
    {
        let mut map = state.swarms.write().await;
        if let Some(s) = map.get_mut(&swarm_id) {
            s.tokens_used = s.tokens_used.saturating_add(tokens);
        }
    }
    tracing::info!(
        swarm_id,
        agent,
        chars = res.len(),
        tokens,
        "swarm: claude-cli-agent reply"
    );
    Ok(res)
}

// ---------------------------------------------------------------------------
// IDE dispatch — bus.send(Execute) and wait for a Result reply.
// We register an ephemeral mailbox under a unique requester name so the
// reply is correlated by `in_reply_to`.
// ---------------------------------------------------------------------------

async fn ide_dispatch(
    state: &Arc<AppState>,
    ide_identity: &str,
    prompt: &str,
) -> Result<String, String> {
    let bus = state.message_bus.clone();
    let requester_name = format!(
        "swarm-orchestrator-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    );
    let requester = AgentId {
        name: requester_name.clone(),
        capabilities: vec![Capability::Execute],
    };
    let mut mailbox = bus.register(requester.clone()).await;

    let mut metadata = StdHashMap::new();
    metadata.insert("source".to_string(), "swarm".to_string());
    metadata.insert("content".to_string(), prompt.to_string());

    let msg_id = qlang_agent::protocol::next_msg_id();
    let graph = Graph {
        id: format!("swarm-task-{}", msg_id),
        version: "1.0".to_string(),
        nodes: vec![],
        edges: vec![],
        constraints: vec![],
        metadata,
    };
    let exec = GraphMessage {
        id: msg_id,
        from: requester.clone(),
        to: AgentId {
            name: ide_identity.to_string(),
            capabilities: vec![Capability::Execute],
        },
        graph,
        inputs: StdHashMap::new(),
        intent: MessageIntent::Execute,
        in_reply_to: None,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    };

    // 10s timeout per spec.
    let result = bus
        .send_and_wait(exec, &mut mailbox, Duration::from_secs(10))
        .await;
    bus.unregister(&requester_name).await;

    match result {
        Ok(reply) => {
            let body = reply
                .graph
                .metadata
                .get("content")
                .cloned()
                .unwrap_or_else(|| "(no content)".to_string());
            Ok(body)
        }
        Err(e) => Err(format!("ide '{}' no response: {}", ide_identity, e)),
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

async fn precheck_continue(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
    max_tokens: u64,
) -> bool {
    let map = swarms.read().await;
    match map.get(&swarm_id) {
        Some(s) => {
            if s.stop_requested {
                tracing::info!(swarm_id, "swarm: stop_requested — exiting loop");
                return false;
            }
            if s.tokens_used > max_tokens {
                tracing::info!(
                    swarm_id,
                    tokens_used = s.tokens_used,
                    max_tokens,
                    "swarm: token cap reached — exiting loop"
                );
                return false;
            }
            true
        }
        None => false,
    }
}

async fn set_status(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
    status: &str,
) {
    let mut map = swarms.write().await;
    if let Some(s) = map.get_mut(&swarm_id) {
        s.status = status.to_string();
    }
}

async fn goal_of(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
) -> String {
    let map = swarms.read().await;
    map.get(&swarm_id).map(|s| s.goal.clone()).unwrap_or_default()
}

async fn summarize_prior_rounds(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
) -> String {
    let map = swarms.read().await;
    let s = match map.get(&swarm_id) {
        Some(s) => s,
        None => return String::new(),
    };
    if s.rounds.is_empty() {
        return "(none — this is the first round)".to_string();
    }
    let mut out = String::new();
    for round in &s.rounds {
        out.push_str(&format!("Round {} eval: {}\n", round.round_number,
            round.eval.as_deref().unwrap_or("(no eval)")));
        for st in &round.subtasks {
            let resp = st.response.as_deref().unwrap_or("(no response)");
            let snippet = if resp.len() > 400 { &resp[..400] } else { resp };
            out.push_str(&format!("  [{}] {} -> {}\n", st.id, st.assigned_to, snippet));
        }
    }
    out
}

async fn collect_ide_list(state: &Arc<AppState>) -> Vec<String> {
    let map = state.presence.lock().await;
    map.keys().cloned().collect()
}

async fn push_error_round(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
    round_no: u32,
    error_text: String,
) {
    let mut map = swarms.write().await;
    if let Some(s) = map.get_mut(&swarm_id) {
        s.rounds.push(SwarmRound {
            round_number: round_no,
            plan: error_text.clone(),
            subtasks: vec![SwarmSubtask {
                id: format!("r{}.err", round_no),
                assigned_to: "strategist".to_string(),
                prompt: "(planning failed)".to_string(),
                response: Some(error_text),
                status: "error".to_string(),
                started_at: Some(now_secs()),
                finished_at: Some(now_secs()),
            }],
            eval: None,
        });
    }
}

async fn update_subtask<F>(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
    round_no: u32,
    subtask_id: &str,
    mut mutator: F,
) where
    F: FnMut(&mut SwarmSubtask),
{
    let mut map = swarms.write().await;
    if let Some(s) = map.get_mut(&swarm_id) {
        if let Some(round) = s.rounds.iter_mut().find(|r| r.round_number == round_no) {
            if let Some(st) = round.subtasks.iter_mut().find(|x| x.id == subtask_id) {
                mutator(st);
            }
        }
    }
}

fn build_plan_prompt(
    goal: &str,
    prior_summary: &str,
    known_agents: &[&str],
    ide_list: &[String],
) -> String {
    let ide_render = if ide_list.is_empty() {
        "(none currently online)".to_string()
    } else {
        ide_list.join(", ")
    };
    format!(
        "You are the strategist. The goal is: {goal}\n\n\
         Prior round results:\n{prior}\n\n\
         Break this into 3-5 concrete subtasks. Output STRICT JSON in this exact shape, \
         and NOTHING else (no markdown fences, no commentary):\n\
         {{ \"subtasks\": [ {{ \"assigned_to\": \"developer\", \"prompt\": \"...\" }} ] }}\n\n\
         Available server agents: {agents}.\n\
         Available IDE identities (live): {ides}.\n\
         Pick wisely — match the subtask to the agent's strength.",
        goal = goal,
        prior = prior_summary,
        agents = known_agents.join(", "),
        ides = ide_render,
    )
}

async fn build_eval_prompt(
    swarms: &Arc<RwLock<std::collections::HashMap<u64, SwarmState>>>,
    swarm_id: u64,
    round_no: u32,
) -> String {
    let map = swarms.read().await;
    let (goal, round_summary) = match map.get(&swarm_id) {
        Some(s) => {
            let round = s.rounds.iter().find(|r| r.round_number == round_no);
            let mut summary = String::new();
            if let Some(round) = round {
                for st in &round.subtasks {
                    let resp = st.response.as_deref().unwrap_or("(none)");
                    let snippet = if resp.len() > 600 { &resp[..600] } else { resp };
                    summary.push_str(&format!(
                        "[{}] {} ({}): {}\n",
                        st.id, st.assigned_to, st.status, snippet
                    ));
                }
            }
            (s.goal.clone(), summary)
        }
        None => (String::new(), String::new()),
    };
    format!(
        "Goal: {goal}\n\nRound {round} results:\n{summary}\n\n\
         Evaluate these results vs the goal. Output STRICT JSON in this exact shape, \
         and NOTHING else (no markdown fences, no commentary):\n\
         {{ \"converged\": true, \"comments\": \"...\" }}\n\
         Set converged=true ONLY if the goal is materially achieved.",
        goal = goal,
        round = round_no,
        summary = round_summary,
    )
}

/// Per-agent system prompt, retained for non-claude-cli paths and tests.
/// Currently unused inside this file because subtasks now run via
/// `call_claude_cli_agent`, but kept around so swapping back to direct
/// LLM calls is a one-liner in `run_subtask`.
#[allow(dead_code)]
fn system_prompt_for(role: &str) -> &'static str {
    match role {
        "ceo"        => "You are CEO, a coordinator agent. Decompose, summarize, and assign work.",
        "developer"  => "You are Developer, a senior software engineer. Be precise. Use code blocks.",
        "researcher" => "You are Researcher, a knowledge synthesizer. Cite sources. Flag uncertainty.",
        "guardian"   => "You are Guardian, a security and safety reviewer. Find vulnerabilities and propose mitigations.",
        "strategist" => "You are Strategist, a planning advisor. Lay out trade-offs and second-order effects.",
        "artisan"    => "You are Artisan, a creative implementer. Generate concrete artifacts.",
        _ => "You are an AI assistant.",
    }
}

/// Parse `{ "subtasks": [...] }` into (assigned_to, prompt) pairs.
/// Tolerates leading/trailing text and code-fence wrappers because
/// LLMs love to ignore "JSON only" instructions.
fn parse_subtasks(raw: &str) -> Result<Vec<(String, String)>, String> {
    let cleaned = strip_code_fences(raw);
    let json_slice = extract_first_json_object(&cleaned).ok_or("no JSON object found")?;
    let v: serde_json::Value = serde_json::from_str(json_slice)
        .map_err(|e| format!("json parse: {e}"))?;
    let arr = v
        .get("subtasks")
        .and_then(|x| x.as_array())
        .ok_or("missing 'subtasks' array")?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let assigned_to = item
            .get("assigned_to")
            .and_then(|x| x.as_str())
            .ok_or("subtask missing 'assigned_to'")?
            .trim()
            .to_string();
        let prompt = item
            .get("prompt")
            .and_then(|x| x.as_str())
            .ok_or("subtask missing 'prompt'")?
            .trim()
            .to_string();
        if assigned_to.is_empty() || prompt.is_empty() {
            continue;
        }
        out.push((assigned_to, prompt));
    }
    Ok(out)
}

fn parse_converged(raw: &str) -> bool {
    let cleaned = strip_code_fences(raw);
    let Some(json_slice) = extract_first_json_object(&cleaned) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(json_slice)
        .ok()
        .and_then(|v| v.get("converged").and_then(|x| x.as_bool()))
        .unwrap_or(false)
}

fn strip_code_fences(s: &str) -> String {
    // Cheap: drop lines that are just ``` fences. Keep everything else.
    s.lines()
        .filter(|l| !l.trim().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find the first `{...}` slice with balanced braces. None if not found.
fn extract_first_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn trim_for_log(s: &str) -> String {
    if s.len() > 300 {
        format!("{}…", &s[..300])
    } else {
        s.to_string()
    }
}

#[allow(dead_code)]
fn _estimate_tokens_unused(s: &str) -> u64 {
    estimate_tokens(s)
}

// ---------------------------------------------------------------------------
// Tests — pure parser tests, no async runtime needed.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_clean_json() {
        let raw = r#"{ "subtasks": [
            { "assigned_to": "developer", "prompt": "write fn" },
            { "assigned_to": "guardian", "prompt": "review fn" }
        ] }"#;
        let parsed = parse_subtasks(raw).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], ("developer".to_string(), "write fn".to_string()));
    }

    #[test]
    fn parse_with_code_fence_and_prefix() {
        let raw = "Sure! Here's the plan:\n```json\n{\"subtasks\":[{\"assigned_to\":\"ceo\",\"prompt\":\"plan\"}]}\n```\nDone.";
        let parsed = parse_subtasks(raw).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].0, "ceo");
    }

    #[test]
    fn parse_missing_array_errors() {
        let raw = r#"{ "foo": 1 }"#;
        assert!(parse_subtasks(raw).is_err());
    }

    #[test]
    fn converged_true_extracted() {
        let raw = r#"{"converged": true, "comments": "ok"}"#;
        assert!(parse_converged(raw));
    }

    #[test]
    fn converged_false_extracted() {
        let raw = r#"{"converged": false, "comments": "more work"}"#;
        assert!(!parse_converged(raw));
    }

    #[test]
    fn converged_garbage_defaults_false() {
        assert!(!parse_converged("nonsense"));
    }

    #[test]
    fn extract_balanced_object() {
        let s = "before {\"a\": {\"b\": 1}} after";
        assert_eq!(
            extract_first_json_object(s),
            Some("{\"a\": {\"b\": 1}}")
        );
    }

    #[test]
    fn extract_handles_escaped_brace_in_string() {
        let s = r#"{"a": "}}}", "b": 2}"#;
        let extracted = extract_first_json_object(s).unwrap();
        let v: serde_json::Value = serde_json::from_str(extracted).unwrap();
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn cost_for_local_is_zero() {
        assert_eq!(cost_per_1k_for(qo_llm::Tier::Local, &None), 0.0);
    }

    #[test]
    fn cost_for_deepseek_coder_matches_template() {
        assert_eq!(
            cost_per_1k_for(qo_llm::Tier::DeepSeek, &Some("deepseek-coder".to_string())),
            0.00014
        );
    }
}
