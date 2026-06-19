use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures::stream::Stream;
use qo_agents::extract_artifacts::{extract_artifacts, Artifact};
use qo_llm::Tier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use crate::agent_models::model_for_agent;
use crate::routes::workspace::write_artifact_to_disk;
use crate::AppState;

const PLANNER_SYSTEM_PROMPT: &str = "\
Du bist Planner in einem DeepSeek-basierten Multi-Agent-System. \
Du strukturierst Ziele fuer einen Worker und einen Reviewer. \
Antworte immer auf Deutsch und liefere nur valides JSON.";

const WORKER_SYSTEM_PROMPT: &str = "\
Du bist Worker in einem DeepSeek-basierten Multi-Agent-System. \
Setze den Plan konkret um. Antworte auf Deutsch. \
Wenn du Dateien erzeugst, gib sie als <qo:file path=\"workspace/...\">...</qo:file> aus.";

const REVIEWER_SYSTEM_PROMPT: &str = "\
Du bist Reviewer in einem DeepSeek-basierten Multi-Agent-System. \
Pruefe hart gegen Ziel und Abnahmekriterien. Antworte auf Deutsch und liefere nur valides JSON.";

const MULTI_AGENT_MODE: &str = "deepseek_first_planner_worker_reviewer";

const MULTI_AGENT_RUN_HISTORY_CAP: usize = 100;

static MULTI_AGENT_RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

fn default_max_revisions() -> u32 {
    1
}

fn default_write_artifacts() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiAgentRunRequest {
    pub goal: String,
    #[serde(default = "default_max_revisions")]
    pub max_revisions: u32,
    #[serde(default = "default_write_artifacts")]
    pub write_artifacts: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiAgentPlan {
    pub goal_summary: String,
    pub deliverable: String,
    pub acceptance_criteria: Vec<String>,
    pub worker_instructions: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentOutput {
    pub agent: String,
    pub tier: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactWriteResult {
    pub path: String,
    pub written: bool,
    pub resolved_path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerRound {
    pub iteration: u32,
    pub tier: String,
    pub output: String,
    pub artifacts: Vec<Artifact>,
    pub artifact_writes: Vec<ArtifactWriteResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewerRound {
    pub iteration: u32,
    pub tier: String,
    pub approved: bool,
    pub feedback: String,
    pub final_answer: String,
    pub output: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiAgentRunResponse {
    pub run_id: u64,
    pub started_at: u64,
    pub finished_at: u64,
    pub mode: String,
    pub goal: String,
    pub status: String,
    pub plan: MultiAgentPlan,
    pub planner: AgentOutput,
    pub worker_rounds: Vec<WorkerRound>,
    pub reviewer_rounds: Vec<ReviewerRound>,
    pub deliverable: String,
    pub final_answer: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredMultiAgentRun {
    pub run_id: u64,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    pub request: MultiAgentRunRequest,
    pub goal: String,
    pub mode: String,
    pub status: String,
    pub phase: String,
    pub plan: Option<MultiAgentPlan>,
    pub planner: Option<AgentOutput>,
    pub worker_rounds: Vec<WorkerRound>,
    pub reviewer_rounds: Vec<ReviewerRound>,
    pub deliverable: Option<String>,
    pub final_answer: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiAgentRunEvent {
    pub kind: String,
    pub run: StoredMultiAgentRun,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiAgentRunStartedResponse {
    pub run_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiAgentRunSummary {
    pub run_id: u64,
    pub started_at: u64,
    pub finished_at: Option<u64>,
    pub mode: String,
    pub goal: String,
    pub status: String,
    pub worker_rounds: usize,
    pub reviewer_rounds: usize,
    pub artifacts_detected: usize,
    pub artifacts_written: usize,
}

#[derive(Debug)]
struct ParsedReview {
    approved: bool,
    feedback: String,
    final_answer: String,
}

pub async fn list_runs(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<MultiAgentRunSummary>> {
    let runs = state.multi_agent_runs.read().await;
    let mut summaries: Vec<MultiAgentRunSummary> = runs.values().map(summarize_run).collect();
    summaries.sort_by(|a, b| b.run_id.cmp(&a.run_id));
    Json(summaries)
}

pub async fn get_run(
    Path(run_id): Path<u64>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<StoredMultiAgentRun>, (StatusCode, String)> {
    let runs = state.multi_agent_runs.read().await;
    let run = runs
        .get(&run_id)
        .cloned()
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("multi-agent run {run_id} not found")))?;
    Ok(Json(run))
}

pub async fn stream_runs(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut rx = state.multi_agent_events_tx.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let json = serde_json::to_string(&event).unwrap_or_default();
                    yield Ok(Event::default().data(json));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

pub async fn start_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MultiAgentRunRequest>,
) -> Result<Json<MultiAgentRunStartedResponse>, (StatusCode, String)> {
    validate_request(&state, &req).await?;

    let run_id = next_multi_agent_run_id();
    let started_at = now_secs();
    let initial = StoredMultiAgentRun {
        run_id,
        started_at,
        finished_at: None,
        goal: req.goal.trim().to_string(),
        mode: MULTI_AGENT_MODE.to_string(),
        status: "queued".to_string(),
        phase: "queued".to_string(),
        request: req.clone(),
        plan: None,
        planner: None,
        worker_rounds: Vec::new(),
        reviewer_rounds: Vec::new(),
        deliverable: None,
        final_answer: None,
        error: None,
    };
    store_run(&state.multi_agent_runs, initial.clone()).await;
    emit_run_event(&state, "started", initial);

    let state_clone = state.clone();
    tokio::spawn(async move {
        if let Err(error) =
            execute_multi_agent_core(&state_clone, &req, run_id, started_at, true).await
        {
            finalize_run_error(&state_clone, run_id, error).await;
        }
    });

    Ok(Json(MultiAgentRunStartedResponse { run_id }))
}

pub async fn run_multi_agent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MultiAgentRunRequest>,
) -> Result<Json<MultiAgentRunResponse>, (StatusCode, String)> {
    validate_request(&state, &req).await?;
    let run_id = next_multi_agent_run_id();
    let started_at = now_secs();
    let response = execute_multi_agent_core(&state, &req, run_id, started_at, false)
        .await
        .map_err(internal_error)?;
    let record = record_from_response(req, &response);
    store_run(&state.multi_agent_runs, record.clone()).await;
    emit_run_event(&state, "completed", record);
    Ok(Json(response))
}

async fn validate_request(
    state: &Arc<AppState>,
    req: &MultiAgentRunRequest,
) -> Result<(), (StatusCode, String)> {
    if req.goal.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "goal must be non-empty".to_string()));
    }
    if !state.llm.tier_available(Tier::DeepSeek).await {
        return Err((
            StatusCode::PRECONDITION_FAILED,
            "DeepSeek provider is not configured".to_string(),
        ));
    }
    Ok(())
}

async fn execute_multi_agent_core(
    state: &Arc<AppState>,
    req: &MultiAgentRunRequest,
    run_id: u64,
    started_at: u64,
    live: bool,
) -> Result<MultiAgentRunResponse, String> {
    let goal = req.goal.trim().to_string();
    let max_revisions = req.max_revisions.min(3);
    let mut reviewer_feedback: Option<String> = None;
    let mut final_status = "needs_revision_limit_review".to_string();
    let mut final_deliverable = String::new();
    let mut final_answer = String::new();

    if live {
        update_run(state, run_id, "planning", |run| {
            run.status = "planning".to_string();
            run.phase = "planner".to_string();
        })
        .await;
    }

    let planner_prompt = build_planner_prompt(&goal);
    let planner = call_deepseek_agent(state, "planner", PLANNER_SYSTEM_PROMPT, &planner_prompt).await?;
    let plan = parse_plan(&goal, &planner.output);

    if live {
        let plan_for_run = plan.clone();
        let planner_for_run = planner.clone();
        update_run(state, run_id, "planner_result", move |run| {
            run.plan = Some(plan_for_run);
            run.planner = Some(planner_for_run);
        })
        .await;
    }

    let mut worker_rounds = Vec::new();
    let mut reviewer_rounds = Vec::new();

    for attempt in 0..=max_revisions {
        if live {
            update_run(state, run_id, "working", |run| {
                run.status = "working".to_string();
                run.phase = format!("worker_round_{}", attempt + 1);
            })
            .await;
        }

        let worker_prompt = build_worker_prompt(&goal, &plan, reviewer_feedback.as_deref(), attempt);
        let worker = call_deepseek_agent(state, "worker", WORKER_SYSTEM_PROMPT, &worker_prompt).await?;
        let artifacts = extract_artifacts(&worker.output);
        let artifact_writes = maybe_write_artifacts(&artifacts, req.write_artifacts);
        final_deliverable = worker.output.clone();
        let worker_round = WorkerRound {
            iteration: attempt + 1,
            tier: worker.tier.clone(),
            output: worker.output.clone(),
            artifacts,
            artifact_writes,
        };
        worker_rounds.push(worker_round.clone());

        if live {
            let round_for_run = worker_round.clone();
            let deliverable_for_run = final_deliverable.clone();
            update_run(state, run_id, "worker_result", move |run| {
                run.worker_rounds.push(round_for_run);
                run.deliverable = Some(deliverable_for_run);
            })
            .await;
        }

        if live {
            update_run(state, run_id, "reviewing", |run| {
                run.status = "reviewing".to_string();
                run.phase = format!("reviewer_round_{}", attempt + 1);
            })
            .await;
        }

        let reviewer_prompt = build_reviewer_prompt(&goal, &plan, &worker.output);
        let reviewer =
            call_deepseek_agent(state, "reviewer", REVIEWER_SYSTEM_PROMPT, &reviewer_prompt).await?;
        let review = parse_review(&worker.output, &reviewer.output);
        final_answer = review.final_answer.clone();
        reviewer_feedback = Some(review.feedback.clone());
        let reviewer_round = ReviewerRound {
            iteration: attempt + 1,
            tier: reviewer.tier.clone(),
            approved: review.approved,
            feedback: review.feedback.clone(),
            final_answer: review.final_answer.clone(),
            output: reviewer.output.clone(),
        };
        reviewer_rounds.push(reviewer_round.clone());

        if live {
            let round_for_run = reviewer_round.clone();
            let final_answer_for_run = final_answer.clone();
            let status_for_run = if review.approved {
                "approved".to_string()
            } else {
                "needs_revision".to_string()
            };
            update_run(state, run_id, "reviewer_result", move |run| {
                run.reviewer_rounds.push(round_for_run);
                run.final_answer = Some(final_answer_for_run);
                run.status = status_for_run;
                run.phase = "review_complete".to_string();
            })
            .await;
        }

        if review.approved {
            final_status = "approved".to_string();
            break;
        }
    }

    if final_answer.trim().is_empty() {
        final_answer = summarize_final_answer(&goal, &final_deliverable, &reviewer_feedback);
    }

    let _ = state
        .store
        .log_action("multi_agent_run", &goal, &final_status);

    let response = MultiAgentRunResponse {
        run_id,
        started_at,
        finished_at: now_secs(),
        mode: MULTI_AGENT_MODE.to_string(),
        goal,
        status: final_status.clone(),
        plan,
        planner,
        worker_rounds,
        reviewer_rounds,
        deliverable: final_deliverable,
        final_answer,
    };

    if live {
        let record = record_from_response(req.clone(), &response);
        store_run(&state.multi_agent_runs, record.clone()).await;
        emit_run_event(state, "completed", record);
    }

    Ok(response)
}

fn build_planner_prompt(goal: &str) -> String {
    format!(
        "Ziel:\n{goal}\n\n\
         Erstelle einen kompakten Arbeitsplan fuer genau einen Worker-Lauf. \
         Gib STRICT JSON in dieser Form zurueck und nichts sonst:\n\
         {{\n\
           \"goal_summary\": \"...\",\n\
           \"deliverable\": \"...\",\n\
           \"acceptance_criteria\": [\"...\", \"...\"],\n\
           \"worker_instructions\": \"...\"\n\
         }}\n\
         Die Abnahmekriterien muessen pruefbar und konkret sein.",
    )
}

fn build_worker_prompt(
    goal: &str,
    plan: &MultiAgentPlan,
    reviewer_feedback: Option<&str>,
    attempt: u32,
) -> String {
    let feedback = reviewer_feedback.unwrap_or("Kein Review-Feedback vorhanden.");
    format!(
        "Originalziel:\n{goal}\n\n\
         Zielzusammenfassung:\n{goal_summary}\n\n\
         Gewuenschtes Ergebnis:\n{deliverable}\n\n\
         Arbeitsanweisung:\n{instructions}\n\n\
         Abnahmekriterien:\n- {criteria}\n\n\
         Aktuelle Runde: {round}\n\
         Review-Feedback aus der letzten Runde:\n{feedback}\n\n\
         Liefere jetzt das Arbeitsergebnis. \
         Wenn Dateien sinnvoll sind, gib sie als <qo:file path=\"workspace/...\">...</qo:file> aus.",
        goal_summary = plan.goal_summary,
        deliverable = plan.deliverable,
        instructions = plan.worker_instructions,
        criteria = plan.acceptance_criteria.join("\n- "),
        round = attempt + 1,
        feedback = feedback,
    )
}

fn build_reviewer_prompt(goal: &str, plan: &MultiAgentPlan, worker_output: &str) -> String {
    format!(
        "Originalziel:\n{goal}\n\n\
         Zielzusammenfassung:\n{goal_summary}\n\n\
         Gewuenschtes Ergebnis:\n{deliverable}\n\n\
         Abnahmekriterien:\n- {criteria}\n\n\
         Worker-Ergebnis:\n{worker_output}\n\n\
         Pruefe, ob das Ergebnis materiell brauchbar ist. \
         Gib STRICT JSON in dieser Form zurueck und nichts sonst:\n\
         {{\n\
           \"approved\": true,\n\
           \"feedback\": \"...\",\n\
           \"final_answer\": \"...\"\n\
         }}\n\
         Setze approved nur dann auf true, wenn das Ziel und die Kriterien erfuellt sind.",
        goal_summary = plan.goal_summary,
        deliverable = plan.deliverable,
        criteria = plan.acceptance_criteria.join("\n- "),
        worker_output = trim_for_context(worker_output, 12_000),
    )
}

async fn call_deepseek_agent(
    state: &Arc<AppState>,
    agent: &str,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<AgentOutput, String> {
    let routed_agent = match agent {
        "planner" => "strategist",
        "worker" => "developer",
        "reviewer" => "researcher",
        other => other,
    };
    let (tier, model_override) = model_for_agent(routed_agent);
    if tier != Tier::DeepSeek {
        return Err(format!(
            "{agent} is mapped to {:?} instead of DeepSeek; fix agent_models routing",
            tier
        ));
    }

    let (output, tier) = state
        .llm
        .chat_with_model(
            Some(tier),
            model_override,
            vec![
                ("system".to_string(), system_prompt.to_string()),
                ("user".to_string(), user_prompt.to_string()),
            ],
        )
        .await
        .map_err(|e| format!("{agent} call failed: {e}"))?;

    if tier != Tier::DeepSeek {
        return Err(format!(
            "{agent} unexpectedly ran on {:?} instead of DeepSeek",
            tier
        ));
    }

    Ok(AgentOutput {
        agent: agent.to_string(),
        tier: tier_label(tier).to_string(),
        output,
    })
}

fn maybe_write_artifacts(
    artifacts: &[Artifact],
    write_artifacts: bool,
) -> Vec<ArtifactWriteResult> {
    if !write_artifacts {
        return artifacts
            .iter()
            .map(|artifact| ArtifactWriteResult {
                path: artifact.path.clone(),
                written: false,
                resolved_path: None,
                error: None,
            })
            .collect();
    }

    artifacts
        .iter()
        .map(|artifact| match write_artifact_to_disk(&artifact.path, &artifact.content) {
            Ok(path) => ArtifactWriteResult {
                path: artifact.path.clone(),
                written: true,
                resolved_path: Some(path.display().to_string()),
                error: None,
            },
            Err(error) => ArtifactWriteResult {
                path: artifact.path.clone(),
                written: false,
                resolved_path: None,
                error: Some(error),
            },
        })
        .collect()
}

fn parse_plan(goal: &str, raw: &str) -> MultiAgentPlan {
    let cleaned = strip_code_fences(raw);
    let parsed = extract_first_json_object(&cleaned)
        .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok());

    let fallback_summary: String = goal.chars().take(160).collect();

    let goal_summary = parsed
        .as_ref()
        .and_then(|value| value.get("goal_summary"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&fallback_summary)
        .to_string();

    let deliverable = parsed
        .as_ref()
        .and_then(|value| value.get("deliverable"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Ein verwertbares, direkt nutzbares Arbeitsergebnis")
        .to_string();

    let acceptance_criteria = parsed
        .as_ref()
        .and_then(|value| value.get("acceptance_criteria"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| {
            vec![
                "Das Ergebnis ist konkret und nicht nur abstrakt beschrieben.".to_string(),
                "Das Ergebnis greift das Originalziel direkt auf.".to_string(),
                "Das Ergebnis ist fuer den naechsten Arbeitsschritt nutzbar.".to_string(),
            ]
        });

    let worker_instructions = parsed
        .as_ref()
        .and_then(|value| value.get("worker_instructions"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Arbeite direkt auf das Ziel hin und liefere ein klares Ergebnis.")
        .to_string();

    MultiAgentPlan {
        goal_summary,
        deliverable,
        acceptance_criteria,
        worker_instructions,
    }
}

fn parse_review(worker_output: &str, raw: &str) -> ParsedReview {
    let cleaned = strip_code_fences(raw);
    let parsed = extract_first_json_object(&cleaned)
        .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok());

    let approved = parsed
        .as_ref()
        .and_then(|value| value.get("approved"))
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let feedback = parsed
        .as_ref()
        .and_then(|value| value.get("feedback"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Review konnte nicht sauber geparst werden; bitte Ergebnis nachschaerfen.")
        .to_string();

    let final_answer = parsed
        .as_ref()
        .and_then(|value| value.get("final_answer"))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| summarize_final_answer("", worker_output, &Some(feedback.clone())));

    ParsedReview {
        approved,
        feedback,
        final_answer,
    }
}

fn summarize_final_answer(goal: &str, deliverable: &str, reviewer_feedback: &Option<String>) -> String {
    let feedback = reviewer_feedback
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("Kein zusaetzliches Reviewer-Feedback.");
    format!(
        "Ziel: {goal}\n\nAktueller Stand:\n{deliverable}\n\nReviewer-Hinweis:\n{feedback}",
        deliverable = trim_for_context(deliverable, 2_000),
    )
}

fn strip_code_fences(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().starts_with("```"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_first_json_object(text: &str) -> Option<&str> {
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if start.is_none() {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    continue;
                }
                depth -= 1;
                if depth == 0 {
                    return start.map(|begin| &text[begin..=idx]);
                }
            }
            _ => {}
        }
    }

    None
}

fn trim_for_context(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "\n...[truncated]..."
}

fn tier_label(tier: Tier) -> &'static str {
    match tier {
        Tier::Local => "local",
        Tier::Groq => "groq",
        Tier::Cloud => "cloud",
        Tier::DeepSeek => "deepseek",
    }
}

fn internal_error(error: String) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, error)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn next_multi_agent_run_id() -> u64 {
    let base = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    let seq = MULTI_AGENT_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    (base << 16) | (seq & 0xFFFF)
}

async fn store_run(
    runs: &Arc<RwLock<HashMap<u64, StoredMultiAgentRun>>>,
    run: StoredMultiAgentRun,
) {
    let mut runs = runs.write().await;
    runs.insert(run.run_id, run);
    if runs.len() <= MULTI_AGENT_RUN_HISTORY_CAP {
        return;
    }

    if let Some(oldest_id) = runs
        .values()
        .min_by_key(|candidate| (candidate.started_at, candidate.run_id))
        .map(|candidate| candidate.run_id)
    {
        runs.remove(&oldest_id);
    }
}

async fn update_run<F>(state: &Arc<AppState>, run_id: u64, kind: &str, updater: F)
where
    F: FnOnce(&mut StoredMultiAgentRun),
{
    let snapshot = {
        let mut runs = state.multi_agent_runs.write().await;
        let Some(run) = runs.get_mut(&run_id) else {
            return;
        };
        updater(run);
        run.clone()
    };
    emit_run_event(state, kind, snapshot);
}

async fn finalize_run_error(state: &Arc<AppState>, run_id: u64, error: String) {
    let snapshot = {
        let mut runs = state.multi_agent_runs.write().await;
        let Some(run) = runs.get_mut(&run_id) else {
            return;
        };
        run.status = "error".to_string();
        run.phase = "error".to_string();
        run.error = Some(error);
        run.finished_at = Some(now_secs());
        run.clone()
    };
    emit_run_event(state, "error", snapshot);
}

fn emit_run_event(state: &Arc<AppState>, kind: &str, run: StoredMultiAgentRun) {
    let _ = state.multi_agent_events_tx.send(MultiAgentRunEvent {
        kind: kind.to_string(),
        run,
    });
}

fn record_from_response(
    request: MultiAgentRunRequest,
    response: &MultiAgentRunResponse,
) -> StoredMultiAgentRun {
    StoredMultiAgentRun {
        run_id: response.run_id,
        started_at: response.started_at,
        finished_at: Some(response.finished_at),
        request,
        goal: response.goal.clone(),
        mode: response.mode.clone(),
        status: response.status.clone(),
        phase: "complete".to_string(),
        plan: Some(response.plan.clone()),
        planner: Some(response.planner.clone()),
        worker_rounds: response.worker_rounds.clone(),
        reviewer_rounds: response.reviewer_rounds.clone(),
        deliverable: Some(response.deliverable.clone()),
        final_answer: Some(response.final_answer.clone()),
        error: None,
    }
}

fn summarize_run(run: &StoredMultiAgentRun) -> MultiAgentRunSummary {
    let artifacts_detected = run
        .worker_rounds
        .iter()
        .map(|round| round.artifacts.len())
        .sum();
    let artifacts_written = run
        .worker_rounds
        .iter()
        .flat_map(|round| round.artifact_writes.iter())
        .filter(|artifact| artifact.written)
        .count();

    MultiAgentRunSummary {
        run_id: run.run_id,
        started_at: run.started_at,
        finished_at: run.finished_at,
        mode: run.mode.clone(),
        goal: run.goal.clone(),
        status: run.status.clone(),
        worker_rounds: run.worker_rounds.len(),
        reviewer_rounds: run.reviewer_rounds.len(),
        artifacts_detected,
        artifacts_written,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn extracts_first_json_object_with_wrapping_text() {
        let raw = "note\n```json\n{\"approved\":true,\"feedback\":\"ok\"}\n```\ntrail";
        let stripped = strip_code_fences(raw);
        let json = extract_first_json_object(&stripped).unwrap();
        assert_eq!(json, "{\"approved\":true,\"feedback\":\"ok\"}");
    }

    #[test]
    fn parse_plan_reads_json_fields() {
        let raw = r#"
before
{
  "goal_summary": "Kurzer Plan",
  "deliverable": "API Entwurf",
  "acceptance_criteria": ["klar", "nutzbar"],
  "worker_instructions": "Baue die erste Version"
}
after
"#;
        let plan = parse_plan("irrelevant", raw);
        assert_eq!(plan.goal_summary, "Kurzer Plan");
        assert_eq!(plan.deliverable, "API Entwurf");
        assert_eq!(plan.acceptance_criteria, vec!["klar", "nutzbar"]);
        assert_eq!(plan.worker_instructions, "Baue die erste Version");
    }

    #[test]
    fn parse_plan_falls_back_when_json_is_missing() {
        let plan = parse_plan("Ein echtes Ziel", "kein json");
        assert!(plan.goal_summary.contains("Ein echtes Ziel"));
        assert!(!plan.acceptance_criteria.is_empty());
    }

    #[test]
    fn parse_review_reads_decision() {
        let review = parse_review(
            "worker output",
            r#"{"approved":true,"feedback":"passt","final_answer":"fertig"}"#,
        );
        assert!(review.approved);
        assert_eq!(review.feedback, "passt");
        assert_eq!(review.final_answer, "fertig");
    }

    #[test]
    fn parse_review_falls_back_to_safe_defaults() {
        let review = parse_review("worker output", "keine json antwort");
        assert!(!review.approved);
        assert!(review.feedback.contains("Review"));
        assert!(review.final_answer.contains("worker output"));
    }

    #[test]
    fn summarize_run_counts_artifacts() {
        let run = StoredMultiAgentRun {
            run_id: 7,
            started_at: 10,
            finished_at: Some(12),
            request: MultiAgentRunRequest {
                goal: "goal".to_string(),
                max_revisions: 1,
                write_artifacts: true,
            },
            goal: "goal".to_string(),
            mode: "local".to_string(),
            status: "approved".to_string(),
            phase: "complete".to_string(),
            plan: Some(MultiAgentPlan {
                goal_summary: "summary".to_string(),
                deliverable: "deliverable".to_string(),
                acceptance_criteria: vec!["a".to_string()],
                worker_instructions: "do it".to_string(),
            }),
            planner: Some(AgentOutput {
                agent: "planner".to_string(),
                tier: "local".to_string(),
                output: "{}".to_string(),
            }),
            worker_rounds: vec![WorkerRound {
                iteration: 1,
                tier: "local".to_string(),
                output: "out".to_string(),
                artifacts: vec![
                    Artifact {
                        path: "workspace/a.txt".to_string(),
                        content: "a".to_string(),
                    },
                    Artifact {
                        path: "workspace/b.txt".to_string(),
                        content: "b".to_string(),
                    },
                ],
                artifact_writes: vec![
                    ArtifactWriteResult {
                        path: "workspace/a.txt".to_string(),
                        written: true,
                        resolved_path: Some("data/workspace/a.txt".to_string()),
                        error: None,
                    },
                    ArtifactWriteResult {
                        path: "workspace/b.txt".to_string(),
                        written: false,
                        resolved_path: None,
                        error: Some("nope".to_string()),
                    },
                ],
            }],
            reviewer_rounds: vec![ReviewerRound {
                iteration: 1,
                tier: "local".to_string(),
                approved: true,
                feedback: "passt".to_string(),
                final_answer: "fertig".to_string(),
                output: "{}".to_string(),
            }],
            deliverable: Some("done".to_string()),
            final_answer: Some("fertig".to_string()),
            error: None,
        };

        let summary = summarize_run(&run);
        assert_eq!(summary.artifacts_detected, 2);
        assert_eq!(summary.artifacts_written, 1);
        assert_eq!(summary.worker_rounds, 1);
        assert_eq!(summary.reviewer_rounds, 1);
    }

    #[tokio::test]
    async fn store_run_caps_history() {
        let runs = Arc::new(RwLock::new(HashMap::new()));
        for idx in 0..=MULTI_AGENT_RUN_HISTORY_CAP {
            store_run(
                &runs,
                StoredMultiAgentRun {
                    run_id: idx as u64,
                    started_at: idx as u64,
                    finished_at: Some(idx as u64),
                    request: MultiAgentRunRequest {
                        goal: format!("goal-{idx}"),
                        max_revisions: 1,
                        write_artifacts: false,
                    },
                    goal: format!("goal-{idx}"),
                    mode: "local".to_string(),
                    status: "approved".to_string(),
                    phase: "complete".to_string(),
                    plan: Some(MultiAgentPlan {
                        goal_summary: "summary".to_string(),
                        deliverable: "deliverable".to_string(),
                        acceptance_criteria: vec!["a".to_string()],
                        worker_instructions: "do it".to_string(),
                    }),
                    planner: Some(AgentOutput {
                        agent: "planner".to_string(),
                        tier: "local".to_string(),
                        output: "{}".to_string(),
                    }),
                    worker_rounds: vec![],
                    reviewer_rounds: vec![],
                    deliverable: Some("done".to_string()),
                    final_answer: Some("fertig".to_string()),
                    error: None,
                },
            )
            .await;
        }

        let runs = runs.read().await;
        assert_eq!(runs.len(), MULTI_AGENT_RUN_HISTORY_CAP);
        assert!(!runs.contains_key(&0));
        assert!(runs.contains_key(&(MULTI_AGENT_RUN_HISTORY_CAP as u64)));
    }
}
