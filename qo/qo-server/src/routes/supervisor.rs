use axum::{
    extract::{Query, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        Html,
    },
    Json,
};
use futures::stream::Stream;
use qlang_agent::protocol::{next_msg_id, AgentId, Capability, GraphMessage, MessageIntent};
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Shape, TensorData, TensorType};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::AppState;

const QLMS_MAGIC: &[u8; 4] = b"QLMS";
const PAYLOAD_KEY: &str = "coding_handover.payload_json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SupervisorDaemonState {
    pub running: bool,
    pub process_id: Option<u32>,
    pub state_path: Option<String>,
    pub spawn: bool,
    pub interval_ms: u64,
    pub last_error: Option<String>,
    pub started_at: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorQuery {
    pub state: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorLogsQuery {
    pub state: Option<String>,
    pub session: Option<u64>,
    pub task: Option<u64>,
    pub tail: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SupervisorLogsResponse {
    pub state: String,
    pub session_id: u64,
    pub agent: String,
    pub task_id: Option<u64>,
    pub stdout_log_path: Option<String>,
    pub stderr_log_path: Option<String>,
    pub stdout_tail: Vec<String>,
    pub stderr_tail: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SupervisorStateFile {
    next_task_id: Option<u64>,
    next_session_id: Option<u64>,
    permission_profile: Option<PermissionProfile>,
    agents: Vec<SupervisorAgent>,
    sessions: Vec<SupervisorSession>,
    tasks: Vec<SupervisorTask>,
    event_log: Vec<SupervisorEvent>,
}

#[derive(Debug, Deserialize)]
struct SupervisorSession {
    id: u64,
    agent_name: String,
    current_task_id: Option<u64>,
    stdout_log_path: Option<String>,
    stderr_log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PermissionProfile {
    mode: String,
    allow_edit: bool,
    allow_execute: bool,
    allow_network: bool,
    allow_sensitive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisorAgent {
    name: String,
    kind: String,
    command: String,
    args: Vec<String>,
    working_dir: Option<String>,
    plugins: Vec<String>,
    skills: Vec<String>,
    labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisorTask {
    id: u64,
    title: String,
    goal: String,
    assigned_agent: Option<String>,
    status: String,
    depends_on: Vec<u64>,
    created_at: String,
    updated_at: String,
    next_action: String,
    latest_handover_path: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisorEvent {
    at: String,
    kind: String,
    message: String,
    task_id: Option<u64>,
    session_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisorStateWritable {
    version: String,
    permission_profile: PermissionProfile,
    agents: Vec<SupervisorAgent>,
    sessions: Vec<SupervisorSessionWritable>,
    tasks: Vec<SupervisorTask>,
    event_log: Vec<SupervisorEvent>,
    next_task_id: u64,
    next_session_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SupervisorSessionWritable {
    id: u64,
    agent_name: String,
    tab_label: String,
    status: String,
    current_task_id: Option<u64>,
    started_at: String,
    last_update: String,
    last_handover_path: Option<String>,
    launch_command: String,
    process_id: Option<u32>,
    exit_code: Option<i32>,
    stdout_log_path: Option<String>,
    stderr_log_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddAgentRequest {
    pub state: Option<String>,
    pub name: String,
    pub kind: Option<String>,
    pub command: String,
    pub args: Option<Vec<String>>,
    pub working_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddTaskRequest {
    pub state: Option<String>,
    pub title: String,
    pub goal: String,
    pub assigned_agent: Option<String>,
    pub next_action: Option<String>,
    pub depends_on: Option<Vec<u64>>,
    pub notes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorActionRequest {
    pub state: Option<String>,
    pub action: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentPreset {
    pub key: String,
    pub name: String,
    pub kind: String,
    pub command: String,
    pub args: Vec<String>,
    pub description: String,
    pub role: String,
    pub capabilities: Vec<String>,
    pub recommended_for: Vec<String>,
    pub default_next_action: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentPresetStatus {
    pub key: String,
    pub name: String,
    pub kind: String,
    pub command: String,
    pub args: Vec<String>,
    pub description: String,
    pub role: String,
    pub capabilities: Vec<String>,
    pub recommended_for: Vec<String>,
    pub default_next_action: String,
    pub available: bool,
    pub resolved_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InstallPresetRequest {
    pub state: Option<String>,
    pub preset: String,
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SuggestAgentRequest {
    pub title: Option<String>,
    pub goal: String,
    pub route: Option<String>,
    pub risk: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct DispatchPresetRequest {
    pub state: Option<String>,
    pub preset: String,
    pub title: String,
    pub goal: String,
    pub next_action: Option<String>,
    pub start_now: Option<bool>,
    pub spawn: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorTaskActionRequest {
    pub state: Option<String>,
    pub task_id: u64,
    pub action: String,
    pub note: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodingHandoverPayload {
    pub schema_version: String,
    pub phase: String,
    pub status: String,
    pub summary: String,
    pub request: String,
    pub next_action: String,
    pub files: Vec<String>,
    pub evidence: Vec<String>,
    pub risks: Vec<String>,
    pub proposed_changes: Vec<String>,
    pub tests: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateHandoverRequest {
    pub state: Option<String>,
    pub task_id: Option<u64>,
    pub output: String,
    pub from: String,
    pub to: String,
    pub phase: String,
    pub status: Option<String>,
    pub summary: String,
    pub request: String,
    pub next_action: String,
    pub files: Option<Vec<String>>,
    pub evidence: Option<Vec<String>>,
    pub risks: Option<Vec<String>>,
    pub proposed_changes: Option<Vec<String>>,
    pub tests: Option<Vec<String>>,
    pub notes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct ReplyHandoverRequest {
    pub state: Option<String>,
    pub task_id: Option<u64>,
    pub input: String,
    pub output: String,
    pub from: String,
    pub to: String,
    pub phase: String,
    pub status: Option<String>,
    pub summary: String,
    pub request: String,
    pub next_action: String,
    pub files: Option<Vec<String>>,
    pub evidence: Option<Vec<String>>,
    pub risks: Option<Vec<String>>,
    pub proposed_changes: Option<Vec<String>>,
    pub tests: Option<Vec<String>>,
    pub notes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct ShowHandoverQuery {
    pub input: String,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorStreamQuery {
    pub state: Option<String>,
    pub session: Option<u64>,
    pub task: Option<u64>,
    pub tail: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SupervisorDaemonStartRequest {
    pub state: Option<String>,
    pub spawn: Option<bool>,
    pub interval_ms: Option<u64>,
}

/// GET /api/supervisor/state
pub async fn state(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<SupervisorQuery>,
) -> Json<Value> {
    let state_path = resolve_state_path(query.state);
    match load_json(&state_path) {
        Ok(value) => Json(value),
        Err(err) => Json(json!({
            "state_path": state_path.display().to_string(),
            "error": err
        })),
    }
}

/// GET /api/supervisor/logs
pub async fn logs(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<SupervisorLogsQuery>,
) -> Json<Value> {
    let state_path = resolve_state_path(query.state.clone());
    let raw = match std::fs::read_to_string(&state_path) {
        Ok(raw) => raw,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": format!("failed to read supervisor state: {err}")
            }))
        }
    };
    let parsed: SupervisorStateFile = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": format!("failed to parse supervisor state: {err}")
            }))
        }
    };

    let session = if let Some(session_id) = query.session {
        parsed.sessions.iter().find(|session| session.id == session_id)
    } else if let Some(task_id) = query.task {
        parsed
            .sessions
            .iter()
            .rev()
            .find(|session| session.current_task_id == Some(task_id))
    } else {
        parsed.sessions.last()
    };

    let Some(session) = session else {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": "no matching supervisor session found"
        }));
    };

    let tail = query.tail.unwrap_or(60);
    let stdout_log_path = session
        .stdout_log_path
        .as_ref()
        .map(|path| resolve_log_path(&state_path, path));
    let stderr_log_path = session
        .stderr_log_path
        .as_ref()
        .map(|path| resolve_log_path(&state_path, path));

    let response = SupervisorLogsResponse {
        state: state_path.display().to_string(),
        session_id: session.id,
        agent: session.agent_name.clone(),
        task_id: session.current_task_id,
        stdout_log_path: stdout_log_path
            .as_ref()
            .map(|path| path.display().to_string()),
        stderr_log_path: stderr_log_path
            .as_ref()
            .map(|path| path.display().to_string()),
        stdout_tail: stdout_log_path
            .as_ref()
            .map(|path| tail_text_file(path, tail))
            .transpose()
            .unwrap_or_else(|err| Some(vec![format!("ERROR: {err}")]))
            .unwrap_or_default(),
        stderr_tail: stderr_log_path
            .as_ref()
            .map(|path| tail_text_file(path, tail))
            .transpose()
            .unwrap_or_else(|err| Some(vec![format!("ERROR: {err}")]))
            .unwrap_or_default(),
    };

    Json(serde_json::to_value(response).unwrap_or_else(|err| {
        json!({
            "state": state_path.display().to_string(),
            "error": format!("failed to serialize supervisor logs response: {err}")
        })
    }))
}

/// POST /api/supervisor/agent
pub async fn add_agent(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<AddAgentRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    let mut state = match load_supervisor_state(&state_path) {
        Ok(state) => state,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": err
            }))
        }
    };

    if state.agents.iter().any(|agent| agent.name == request.name) {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("agent '{}' already exists", request.name)
        }));
    }

    state.agents.push(SupervisorAgent {
        name: request.name.clone(),
        kind: request.kind.unwrap_or_else(|| "Custom".into()),
        command: request.command,
        args: request.args.unwrap_or_default(),
        working_dir: request.working_dir,
        plugins: Vec::new(),
        skills: Vec::new(),
        labels: Vec::new(),
    });
    state.event_log.push(SupervisorEvent {
        at: now_string(),
        kind: "add-agent".into(),
        message: format!("registered agent '{}'", request.name),
        task_id: None,
        session_id: None,
    });

    match save_supervisor_state(&state_path, &state) {
        Ok(()) => Json(json!({
            "state": state_path.display().to_string(),
            "status": "registered",
            "agent": request.name
        })),
        Err(err) => Json(json!({
            "state": state_path.display().to_string(),
            "error": err
        })),
    }
}

/// POST /api/supervisor/task
pub async fn add_task(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<AddTaskRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    let mut state = match load_supervisor_state(&state_path) {
        Ok(state) => state,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": err
            }))
        }
    };

    let now = now_string();
    let task_id = state.next_task_id;
    state.next_task_id += 1;
    state.tasks.push(SupervisorTask {
        id: task_id,
        title: request.title.clone(),
        goal: request.goal,
        assigned_agent: request.assigned_agent,
        status: "Queued".into(),
        depends_on: request.depends_on.unwrap_or_default(),
        created_at: now.clone(),
        updated_at: now.clone(),
        next_action: request.next_action.unwrap_or_else(|| "start work".into()),
        latest_handover_path: None,
        notes: request.notes.unwrap_or_default(),
    });
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "enqueue".into(),
        message: format!("enqueued task #{task_id}"),
        task_id: Some(task_id),
        session_id: None,
    });

    match save_supervisor_state(&state_path, &state) {
        Ok(()) => Json(json!({
            "state": state_path.display().to_string(),
            "status": "queued",
            "task_id": task_id,
            "title": request.title
        })),
        Err(err) => Json(json!({
            "state": state_path.display().to_string(),
            "error": err
        })),
    }
}

/// POST /api/supervisor/action
pub async fn action(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<SupervisorActionRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    match request.action.as_str() {
        "poll" => Json(run_poll(&state_path)),
        "tick" => Json(run_tick(&state_path, false)),
        "tick_spawn" => Json(run_tick(&state_path, true)),
        "cycle" => {
            let _ = run_poll(&state_path);
            Json(run_tick(&state_path, false))
        }
        "cycle_spawn" => {
            let _ = run_poll(&state_path);
            Json(run_tick(&state_path, true))
        }
        other => Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("unsupported supervisor action '{}'", other)
        })),
    }
}

/// GET /api/supervisor/presets
pub async fn presets() -> Json<Value> {
    Json(json!(agent_preset_statuses()))
}

/// POST /api/supervisor/install-preset
pub async fn install_preset(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<InstallPresetRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    let mut state = match load_supervisor_state(&state_path) {
        Ok(state) => state,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": err
            }))
        }
    };

    let Some(preset) = agent_presets()
        .into_iter()
        .find(|preset| preset.key == request.preset) else {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("unknown preset '{}'", request.preset)
        }));
    };

    if resolve_command_path(&preset.command).is_none() {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("preset '{}' is not available on this machine", preset.key)
        }));
    }

    let agent_name = request.name.unwrap_or_else(|| preset.key.clone());
    if state.agents.iter().any(|agent| agent.name == agent_name) {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("agent '{}' already exists", agent_name)
        }));
    }

    state.agents.push(SupervisorAgent {
        name: agent_name.clone(),
        kind: preset.kind.clone(),
        command: preset.command.clone(),
        args: preset.args.clone(),
        working_dir: None,
        plugins: Vec::new(),
        skills: preset.capabilities.clone(),
        labels: {
            let mut labels = vec![format!("preset:{}", preset.key), format!("role:{}", preset.role)];
            labels.extend(preset.capabilities.iter().map(|capability| format!("cap:{capability}")));
            labels
        },
    });
    state.event_log.push(SupervisorEvent {
        at: now_string(),
        kind: "install-preset".into(),
        message: format!("installed preset '{}' as agent '{}'", preset.key, agent_name),
        task_id: None,
        session_id: None,
    });

    match save_supervisor_state(&state_path, &state) {
        Ok(()) => Json(json!({
            "state": state_path.display().to_string(),
            "status": "installed",
            "preset": preset.key,
            "agent": agent_name
        })),
        Err(err) => Json(json!({
            "state": state_path.display().to_string(),
            "error": err
        })),
    }
}

/// POST /api/supervisor/suggest-agent
pub async fn suggest_agent(Json(request): Json<SuggestAgentRequest>) -> Json<Value> {
    let title = request.title.unwrap_or_default();
    let route = request.route.unwrap_or_default();
    let risk = request.risk.unwrap_or_default();
    let combined = format!("{} {} {} {}", title, request.goal, route, risk).to_lowercase();

    let preset = if combined.contains("smoke")
        || combined.contains("demo")
        || combined.contains("gui test")
        || combined.contains("cockpit")
    {
        "demo"
    } else if combined.contains("review")
        || combined.contains("patch")
        || combined.contains("fix")
        || combined.contains("test")
    {
        "codex"
    } else if combined.contains("research")
        || combined.contains("second opinion")
        || combined.contains("escalate")
        || combined.contains("broad")
    {
        "gemini"
    } else if combined.contains("context")
        || combined.contains("transcript")
        || combined.contains("long")
        || combined.contains("document")
    {
        "kimi"
    } else {
        "claude"
    };

    let preset_data = agent_presets()
        .into_iter()
        .find(|candidate| candidate.key == preset)
        .expect("preset exists");

    Json(json!({
        "preset": preset_data.key,
        "name": preset_data.name,
        "role": preset_data.role,
        "default_next_action": preset_data.default_next_action,
        "recommended_for": preset_data.recommended_for,
        "capabilities": preset_data.capabilities
    }))
}

/// POST /api/supervisor/dispatch
pub async fn dispatch_preset(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<DispatchPresetRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    let preset = match agent_presets()
        .into_iter()
        .find(|candidate| candidate.key == request.preset)
    {
        Some(preset) => preset,
        None => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": format!("unknown preset '{}'", request.preset)
            }))
        }
    };

    if resolve_command_path(&preset.command).is_none() {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": format!("preset '{}' is not available on this machine", preset.key)
        }));
    }

    let mut state = match load_supervisor_state(&state_path) {
        Ok(state) => state,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": err
            }))
        }
    };

    let agent_name = preset.key.clone();
    if !state.agents.iter().any(|agent| agent.name == agent_name) {
        state.agents.push(SupervisorAgent {
            name: agent_name.clone(),
            kind: preset.kind.clone(),
            command: preset.command.clone(),
            args: preset.args.clone(),
            working_dir: None,
            plugins: Vec::new(),
            skills: preset.capabilities.clone(),
            labels: {
                let mut labels = vec![format!("preset:{}", preset.key), format!("role:{}", preset.role)];
                labels.extend(preset.capabilities.iter().map(|capability| format!("cap:{capability}")));
                labels
            },
        });
        state.event_log.push(SupervisorEvent {
            at: now_string(),
            kind: "install-preset".into(),
            message: format!("installed preset '{}' as agent '{}'", preset.key, agent_name),
            task_id: None,
            session_id: None,
        });
    }

    let now = now_string();
    let task_id = state.next_task_id;
    state.next_task_id += 1;
    state.tasks.push(SupervisorTask {
        id: task_id,
        title: request.title,
        goal: request.goal,
        assigned_agent: Some(agent_name.clone()),
        status: "Queued".into(),
        depends_on: Vec::new(),
        created_at: now.clone(),
        updated_at: now.clone(),
        next_action: request
            .next_action
            .unwrap_or_else(|| preset.default_next_action.clone()),
        latest_handover_path: None,
        notes: vec![format!("preset:{}", preset.key), format!("role:{}", preset.role)],
    });
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "enqueue".into(),
        message: format!("enqueued task #{task_id} via preset '{}'", preset.key),
        task_id: Some(task_id),
        session_id: None,
    });

    if let Err(err) = save_supervisor_state(&state_path, &state) {
        return Json(json!({
            "state": state_path.display().to_string(),
            "error": err
        }));
    }

    let start_now = request.start_now.unwrap_or(false);
    let spawn = request.spawn.unwrap_or(true);
    let dispatch_result = if start_now {
        run_tick(&state_path, spawn)
    } else {
        json!({
            "state": state_path.display().to_string(),
            "status": "queued",
            "task_id": task_id,
            "agent": agent_name
        })
    };

    Json(json!({
        "preset": preset.key,
        "agent": agent_name,
        "task_id": task_id,
        "dispatched": true,
        "start_now": start_now,
        "dispatch_result": dispatch_result
    }))
}

/// POST /api/supervisor/task-action
pub async fn task_action(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<SupervisorTaskActionRequest>,
) -> Json<Value> {
    let state_path = resolve_state_path(request.state);
    let mut state = match load_supervisor_state(&state_path) {
        Ok(state) => state,
        Err(err) => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": err
            }))
        }
    };

    let now = now_string();
    let task = match state.tasks.iter_mut().find(|task| task.id == request.task_id) {
        Some(task) => task,
        None => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": format!("task #{} not found", request.task_id)
            }))
        }
    };

    match request.action.as_str() {
        "complete" => {
            task.status = "Done".into();
            task.updated_at = now.clone();
            if let Some(note) = &request.note {
                task.notes.push(note.clone());
            }
            if let Some(session) = state
                .sessions
                .iter_mut()
                .find(|session| session.current_task_id == Some(request.task_id) && session.status == "Running")
            {
                session.status = "Done".into();
                session.last_update = now.clone();
                session.exit_code = Some(0);
            }
            state.event_log.push(SupervisorEvent {
                at: now.clone(),
                kind: "complete".into(),
                message: format!("task #{} -> Done", request.task_id),
                task_id: Some(request.task_id),
                session_id: None,
            });
        }
        "fail" => {
            task.status = "Failed".into();
            task.updated_at = now.clone();
            if let Some(note) = &request.note {
                task.notes.push(note.clone());
            }
            if let Some(session) = state
                .sessions
                .iter_mut()
                .find(|session| session.current_task_id == Some(request.task_id) && session.status == "Running")
            {
                session.status = "Failed".into();
                session.last_update = now.clone();
                session.exit_code = Some(1);
            }
            state.event_log.push(SupervisorEvent {
                at: now.clone(),
                kind: "fail".into(),
                message: format!("task #{} -> Failed", request.task_id),
                task_id: Some(request.task_id),
                session_id: None,
            });
        }
        "handover" => {
            let Some(path) = request.path.clone() else {
                return Json(json!({
                    "state": state_path.display().to_string(),
                    "error": "path is required for handover"
                }));
            };
            task.latest_handover_path = Some(path.clone());
            task.updated_at = now.clone();
            if let Some(session) = state
                .sessions
                .iter_mut()
                .find(|session| session.current_task_id == Some(request.task_id))
            {
                session.last_handover_path = Some(path.clone());
                session.last_update = now.clone();
            }
            state.event_log.push(SupervisorEvent {
                at: now.clone(),
                kind: "handover".into(),
                message: format!("recorded handover for task #{}: {}", request.task_id, path),
                task_id: Some(request.task_id),
                session_id: None,
            });
        }
        other => {
            return Json(json!({
                "state": state_path.display().to_string(),
                "error": format!("unsupported task action '{}'", other)
            }))
        }
    }

    match save_supervisor_state(&state_path, &state) {
        Ok(()) => Json(json!({
            "state": state_path.display().to_string(),
            "status": "ok",
            "action": request.action,
            "task_id": request.task_id
        })),
        Err(err) => Json(json!({
            "state": state_path.display().to_string(),
            "error": err
        })),
    }
}

/// POST /api/supervisor/handover/create
pub async fn create_handover(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<CreateHandoverRequest>,
) -> Json<Value> {
    let payload = CodingHandoverPayload {
        schema_version: "coding-handover/v1".into(),
        phase: request.phase,
        status: request.status.unwrap_or_else(|| "open".into()),
        summary: request.summary,
        request: request.request,
        next_action: request.next_action,
        files: request.files.unwrap_or_default(),
        evidence: request.evidence.unwrap_or_default(),
        risks: request.risks.unwrap_or_default(),
        proposed_changes: request.proposed_changes.unwrap_or_default(),
        tests: request.tests.unwrap_or_default(),
        notes: request.notes.unwrap_or_default(),
    };
    let message = build_message(
        agent_id(&request.from),
        agent_id(&request.to),
        payload,
        MessageIntent::Compose,
        None,
    );
    let output = PathBuf::from(&request.output);
    if let Err(err) = write_conversation(&output, &[message]) {
        return Json(json!({ "error": err }));
    }
    if let Some(task_id) = request.task_id {
        let _ = attach_handover_to_task(&resolve_state_path(request.state), task_id, output.display().to_string());
    }
    Json(json!({
        "written": output.display().to_string(),
        "format": "QLMS",
        "purpose": "coding_handover"
    }))
}

/// POST /api/supervisor/handover/reply
pub async fn reply_handover(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<ReplyHandoverRequest>,
) -> Json<Value> {
    let input = PathBuf::from(&request.input);
    let mut conversation = match read_conversation(&input) {
        Ok(conversation) => conversation,
        Err(err) => return Json(json!({ "error": err })),
    };
    let previous = match conversation.last().cloned() {
        Some(message) => message,
        None => return Json(json!({ "error": "input conversation is empty" })),
    };
    let payload = CodingHandoverPayload {
        schema_version: "coding-handover/v1".into(),
        phase: request.phase,
        status: request.status.unwrap_or_else(|| "open".into()),
        summary: request.summary,
        request: request.request,
        next_action: request.next_action,
        files: request.files.unwrap_or_default(),
        evidence: request.evidence.unwrap_or_default(),
        risks: request.risks.unwrap_or_default(),
        proposed_changes: request.proposed_changes.unwrap_or_default(),
        tests: request.tests.unwrap_or_default(),
        notes: request.notes.unwrap_or_default(),
    };
    let reply = build_message(
        agent_id(&request.from),
        agent_id(&request.to),
        payload,
        MessageIntent::Result {
            original_message_id: previous.id,
        },
        Some(previous.id),
    );
    conversation.push(reply);
    let output = PathBuf::from(&request.output);
    if let Err(err) = write_conversation(&output, &conversation) {
        return Json(json!({ "error": err }));
    }
    if let Some(task_id) = request.task_id {
        let _ = attach_handover_to_task(&resolve_state_path(request.state), task_id, output.display().to_string());
    }
    Json(json!({
        "written": output.display().to_string(),
        "format": "QLMS",
        "purpose": "coding_handover_reply",
        "messages": conversation.len()
    }))
}

/// GET /api/supervisor/handover/show
pub async fn show_handover(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<ShowHandoverQuery>,
) -> Json<Value> {
    let input = PathBuf::from(&query.input);
    let conversation = match read_conversation(&input) {
        Ok(conversation) => conversation,
        Err(err) => return Json(json!({ "error": err })),
    };
    Json(json!(render_conversation(&conversation)))
}

/// GET /api/supervisor/stream
pub async fn stream(
    State(_state): State<Arc<AppState>>,
    Query(query): Query<SupervisorStreamQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let state_path = resolve_state_path(query.state.clone());
    let session_id = query.session;
    let task_id = query.task;
    let tail = query.tail.unwrap_or(80);

    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            let payload = build_stream_snapshot(&state_path, session_id, task_id, tail);
            let data = serde_json::to_string(&payload).unwrap_or_else(|err| {
                json!({
                    "state": state_path.display().to_string(),
                    "error": format!("failed to serialize supervisor stream payload: {err}")
                }).to_string()
            });
            yield Ok::<Event, Infallible>(Event::default().data(data));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// GET /api/supervisor/daemon/status
pub async fn daemon_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let daemon = state.supervisor_daemon.lock().await.clone();
    Json(json!(daemon))
}

/// POST /api/supervisor/daemon/start
pub async fn daemon_start(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SupervisorDaemonStartRequest>,
) -> Json<Value> {
    let mut daemon = state.supervisor_daemon.lock().await;
    if daemon.running {
        return Json(json!({
            "running": true,
            "process_id": daemon.process_id,
            "state_path": daemon.state_path,
            "interval_ms": daemon.interval_ms
        }));
    }

    let state_path = resolve_state_path(request.state);
    let spawn = request.spawn.unwrap_or(true);
    let interval_ms = request.interval_ms.unwrap_or(2_000);
    let current_exe = match std::env::current_exe() {
        Ok(path) => path,
        Err(err) => {
            daemon.last_error = Some(format!("failed to resolve current executable: {err}"));
            return Json(json!({ "error": daemon.last_error.clone() }))
        }
    };
    let exe_dir = match current_exe.parent() {
        Some(path) => path.to_path_buf(),
        None => {
            daemon.last_error = Some("current executable has no parent directory".into());
            return Json(json!({ "error": daemon.last_error.clone() }))
        }
    };
    let qlang_exe = exe_dir.join(if cfg!(windows) { "qlang.exe" } else { "qlang" });
    if !qlang_exe.exists() {
        daemon.last_error = Some(format!("qlang executable not found next to qo: {}", qlang_exe.display()));
        return Json(json!({ "error": daemon.last_error.clone() }))
    }

    let mut command = std::process::Command::new(&qlang_exe);
    command.arg("supervisor");
    command.arg("daemon");
    command.arg("--state");
    command.arg(state_path.display().to_string());
    command.arg("--interval-ms");
    command.arg(interval_ms.to_string());
    if spawn {
        command.arg("--spawn");
    }
    command.stdin(std::process::Stdio::null());
    command.stdout(std::process::Stdio::null());
    command.stderr(std::process::Stdio::null());

    match command.spawn() {
        Ok(child) => {
            *daemon = SupervisorDaemonState {
                running: true,
                process_id: Some(child.id()),
                state_path: Some(state_path.display().to_string()),
                spawn,
                interval_ms,
                last_error: None,
                started_at: Some(now_string()),
            };
            Json(json!({
                "running": true,
                "process_id": child.id(),
                "state_path": state_path.display().to_string(),
                "spawn": spawn,
                "interval_ms": interval_ms
            }))
        }
        Err(err) => {
            daemon.last_error = Some(format!("failed to start supervisor daemon: {err}"));
            Json(json!({ "error": daemon.last_error.clone() }))
        }
    }
}

/// POST /api/supervisor/daemon/stop
pub async fn daemon_stop(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut daemon = state.supervisor_daemon.lock().await;
    let Some(pid) = daemon.process_id else {
        *daemon = SupervisorDaemonState::default();
        return Json(json!({ "running": false, "status": "idle" }));
    };

    let stop_result = if cfg!(windows) {
        std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .map_err(|err| format!("taskkill failed: {err}"))
            .and_then(|status| {
                if status.success() {
                    Ok(())
                } else {
                    Err(format!("taskkill exited with status {status}"))
                }
            })
    } else {
        std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .map_err(|err| format!("kill failed: {err}"))
            .and_then(|status| {
                if status.success() {
                    Ok(())
                } else {
                    Err(format!("kill exited with status {status}"))
                }
            })
    };

    match stop_result {
        Ok(()) => {
            *daemon = SupervisorDaemonState::default();
            Json(json!({ "running": false, "status": "stopped", "process_id": pid }))
        }
        Err(err) => {
            daemon.last_error = Some(err.clone());
            Json(json!({ "running": daemon.running, "process_id": daemon.process_id, "error": err }))
        }
    }
}

/// GET /supervisor
pub async fn cockpit() -> Html<&'static str> {
    Html(SUPERVISOR_HTML)
}

fn resolve_state_path(value: Option<String>) -> PathBuf {
    value
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".qlang/supervisor.json"))
}

fn resolve_log_path(state_path: &Path, log_path: &str) -> PathBuf {
    let candidate = PathBuf::from(log_path);
    if candidate.is_absolute() {
        return candidate;
    }
    if let Some(parent) = state_path.parent() {
        return parent.join(candidate);
    }
    candidate
}

fn load_json(path: &Path) -> Result<Value, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse '{}': {err}", path.display()))
}

fn agent_presets() -> Vec<AgentPreset> {
    vec![
        AgentPreset {
            key: "demo".into(),
            name: "QLANG Demo Agent".into(),
            kind: "QlangDemo".into(),
            command: "powershell".into(),
            args: vec![
                "-NoProfile".into(),
                "-Command".into(),
                "Write-Output 'qlang-demo: session-start'; Start-Sleep -Milliseconds 350; Write-Output 'qlang-demo: analyzing task'; Start-Sleep -Milliseconds 350; Write-Output 'qlang-demo: session-complete'".into(),
            ],
            description: "Runs a short non-interactive demo session so the GUI, logs, sessions, and daemon flow can be tested safely.".into(),
            role: "gui-smoke-test".into(),
            capabilities: vec!["smoke_test".into(), "logging".into(), "supervisor_validation".into()],
            recommended_for: vec!["GUI smoke test".into(), "daemon validation".into(), "session log verification".into()],
            default_next_action: "Run a short supervisor smoke test and verify that session logs appear in the cockpit.".into(),
        },
        AgentPreset {
            key: "claude".into(),
            name: "Claude Code".into(),
            kind: "ClaudeCode".into(),
            command: "claude".into(),
            args: vec!["code".into()],
            description: "Starts Claude Code in its native CLI mode.".into(),
            role: "primary-builder".into(),
            capabilities: vec!["analyze".into(), "patch".into(), "handover".into()],
            recommended_for: vec!["repo analysis".into(), "implementation".into(), "multi-step fixes".into()],
            default_next_action: "Investigate the repository and prepare the first implementation step.".into(),
        },
        AgentPreset {
            key: "codex".into(),
            name: "Codex".into(),
            kind: "Codex".into(),
            command: "codex".into(),
            args: vec![],
            description: "Starts the Codex CLI worker.".into(),
            role: "reviewer-patcher".into(),
            capabilities: vec!["patch".into(), "review".into(), "tests".into()],
            recommended_for: vec!["targeted patches".into(), "review follow-up".into(), "test repair".into()],
            default_next_action: "Review the current context and produce the next patch or verification step.".into(),
        },
        AgentPreset {
            key: "gemini".into(),
            name: "Gemini".into(),
            kind: "Gemini".into(),
            command: "gemini".into(),
            args: vec![],
            description: "Starts the Gemini CLI worker.".into(),
            role: "research-escalation".into(),
            capabilities: vec!["research".into(), "summarize".into(), "escalate".into()],
            recommended_for: vec!["broad reasoning".into(), "second opinion".into(), "summary handoff".into()],
            default_next_action: "Provide a second-opinion analysis and summarize the best next move.".into(),
        },
        AgentPreset {
            key: "kimi".into(),
            name: "Kimi".into(),
            kind: "Kimi".into(),
            command: "kimi".into(),
            args: vec![],
            description: "Starts the Kimi CLI worker if installed.".into(),
            role: "long-context-analyst".into(),
            capabilities: vec!["long_context".into(), "summarize".into(), "search".into()],
            recommended_for: vec!["long transcripts".into(), "document-heavy tasks".into(), "context compression".into()],
            default_next_action: "Digest the large context and return a compact structured summary.".into(),
        },
    ]
}

fn agent_preset_statuses() -> Vec<AgentPresetStatus> {
    agent_presets()
        .into_iter()
        .map(|preset| {
            let resolved_path = resolve_command_path(&preset.command);
            AgentPresetStatus {
                key: preset.key,
                name: preset.name,
                kind: preset.kind,
                command: preset.command,
                args: preset.args,
                description: preset.description,
                role: preset.role,
                capabilities: preset.capabilities,
                recommended_for: preset.recommended_for,
                default_next_action: preset.default_next_action,
                available: resolved_path.is_some(),
                resolved_path,
            }
        })
        .collect()
}

fn build_stream_snapshot(state_path: &Path, session_id: Option<u64>, task_id: Option<u64>, tail: usize) -> Value {
    let state = load_json(state_path).unwrap_or_else(|err| json!({
        "state_path": state_path.display().to_string(),
        "error": err
    }));
    let logs = load_logs_value(state_path, session_id, task_id, tail);
    json!({
        "state": state,
        "logs": logs
    })
}

fn load_logs_value(state_path: &Path, session_id: Option<u64>, task_id: Option<u64>, tail: usize) -> Value {
    let raw = match std::fs::read_to_string(state_path) {
        Ok(raw) => raw,
        Err(err) => {
            return json!({
                "state": state_path.display().to_string(),
                "error": format!("failed to read supervisor state: {err}")
            })
        }
    };
    let parsed: SupervisorStateFile = match serde_json::from_str(&raw) {
        Ok(parsed) => parsed,
        Err(err) => {
            return json!({
                "state": state_path.display().to_string(),
                "error": format!("failed to parse supervisor state: {err}")
            })
        }
    };

    let session = if let Some(session_id) = session_id {
        parsed.sessions.iter().find(|session| session.id == session_id)
    } else if let Some(task_id) = task_id {
        parsed.sessions.iter().rev().find(|session| session.current_task_id == Some(task_id))
    } else {
        parsed.sessions.last()
    };

    let Some(session) = session else {
        return json!({
            "state": state_path.display().to_string(),
            "error": "no matching supervisor session found"
        });
    };

    let stdout_log_path = session
        .stdout_log_path
        .as_ref()
        .map(|path| resolve_log_path(state_path, path));
    let stderr_log_path = session
        .stderr_log_path
        .as_ref()
        .map(|path| resolve_log_path(state_path, path));

    serde_json::to_value(SupervisorLogsResponse {
        state: state_path.display().to_string(),
        session_id: session.id,
        agent: session.agent_name.clone(),
        task_id: session.current_task_id,
        stdout_log_path: stdout_log_path.as_ref().map(|path| path.display().to_string()),
        stderr_log_path: stderr_log_path.as_ref().map(|path| path.display().to_string()),
        stdout_tail: stdout_log_path
            .as_ref()
            .map(|path| tail_text_file(path, tail))
            .transpose()
            .unwrap_or_else(|err| Some(vec![format!("ERROR: {err}")]))
            .unwrap_or_default(),
        stderr_tail: stderr_log_path
            .as_ref()
            .map(|path| tail_text_file(path, tail))
            .transpose()
            .unwrap_or_else(|err| Some(vec![format!("ERROR: {err}")]))
            .unwrap_or_default(),
    }).unwrap_or_else(|err| json!({
        "state": state_path.display().to_string(),
        "error": format!("failed to serialize supervisor logs response: {err}")
    }))
}

fn attach_handover_to_task(state_path: &Path, task_id: u64, path: String) -> Result<(), String> {
    let mut state = load_supervisor_state(state_path)?;
    let now = now_string();
    if let Some(task) = state.tasks.iter_mut().find(|task| task.id == task_id) {
        task.latest_handover_path = Some(path.clone());
        task.updated_at = now.clone();
    }
    if let Some(session) = state
        .sessions
        .iter_mut()
        .find(|session| session.current_task_id == Some(task_id))
    {
        session.last_handover_path = Some(path.clone());
        session.last_update = now.clone();
    }
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "handover".into(),
        message: format!("recorded handover for task #{task_id}: {path}"),
        task_id: Some(task_id),
        session_id: None,
    });
    save_supervisor_state(state_path, &state)
}

fn load_supervisor_state(path: &Path) -> Result<SupervisorStateWritable, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse '{}': {err}", path.display()))
}

fn save_supervisor_state(path: &Path, state: &SupervisorStateWritable) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(state)
        .map_err(|err| format!("failed to serialize supervisor state: {err}"))?;
    std::fs::write(path, json)
        .map_err(|err| format!("failed to write '{}': {err}", path.display()))
}

fn build_message(
    from: AgentId,
    to: AgentId,
    payload: CodingHandoverPayload,
    intent: MessageIntent,
    in_reply_to: Option<u64>,
) -> GraphMessage {
    let payload_json = serde_json::to_string_pretty(&payload).expect("serialize coding payload");
    let mut inputs = HashMap::new();
    inputs.insert(PAYLOAD_KEY.to_string(), TensorData::from_string(&payload_json));
    inputs.insert("coding_handover.summary".to_string(), TensorData::from_string(&payload.summary));
    inputs.insert("coding_handover.request".to_string(), TensorData::from_string(&payload.request));
    inputs.insert("coding_handover.next_action".to_string(), TensorData::from_string(&payload.next_action));
    inputs.insert(
        "coding_handover.files_json".to_string(),
        TensorData::from_string(&serde_json::to_string(&payload.files).expect("serialize files")),
    );

    GraphMessage {
        id: next_msg_id(),
        from,
        to,
        graph: build_handover_graph(&payload),
        inputs,
        intent,
        in_reply_to,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    }
}

fn build_handover_graph(payload: &CodingHandoverPayload) -> Graph {
    let utf8 = TensorType::new(qlang_core::tensor::Dtype::Utf8, Shape(vec![Dim::Dynamic]));
    let mut graph = Graph::new(format!("coding-handover-{}", payload.phase));
    graph.metadata.insert("handover_kind".into(), "coding_context".into());
    graph.metadata.insert("schema_version".into(), payload.schema_version.clone());
    graph.metadata.insert("phase".into(), payload.phase.clone());
    graph.metadata.insert("status".into(), payload.status.clone());
    graph.metadata.insert("summary".into(), payload.summary.clone());
    let input = graph.add_node(
        Op::Input {
            name: PAYLOAD_KEY.to_string(),
        },
        vec![],
        vec![utf8.clone()],
    );
    let output = graph.add_node(
        Op::Output {
            name: "coding_handover.payload_out".to_string(),
        },
        vec![utf8.clone()],
        vec![],
    );
    graph.add_edge(input, 0, output, 0, utf8);
    graph
}

fn agent_id(name: &str) -> AgentId {
    AgentId {
        name: name.to_string(),
        capabilities: vec![Capability::Execute, Capability::Optimize],
    }
}

fn write_conversation(path: &Path, messages: &[GraphMessage]) -> Result<(), String> {
    let json = serde_json::to_vec(messages)
        .map_err(|e| format!("failed to serialize conversation: {e}"))?;
    let mut bytes = Vec::with_capacity(8 + json.len());
    bytes.extend_from_slice(QLMS_MAGIC);
    bytes.extend_from_slice(&(messages.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&json);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create output directory '{}': {e}", parent.display()))?;
        }
    }
    std::fs::write(path, bytes)
        .map_err(|e| format!("failed to write '{}': {e}", path.display()))
}

fn read_conversation(path: &Path) -> Result<Vec<GraphMessage>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
    parse_qlms_messages(&bytes)
        .map_err(|e| format!("failed to parse '{}': {e}", path.display()))
}

fn parse_qlms_messages(bytes: &[u8]) -> Result<Vec<GraphMessage>, String> {
    if bytes.len() < 8 {
        return Err("file too short for QLMS envelope".into());
    }
    if &bytes[0..4] != QLMS_MAGIC {
        return Err("invalid QLMS magic".into());
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let payload = &bytes[8..];
    let messages: Vec<GraphMessage> =
        serde_json::from_slice(payload).map_err(|e| format!("invalid QLMS JSON payload: {e}"))?;
    if messages.len() != count {
        return Err(format!(
            "QLMS message count mismatch: header says {count}, payload has {}",
            messages.len()
        ));
    }
    Ok(messages)
}

fn render_conversation(messages: &[GraphMessage]) -> Vec<Value> {
    messages
        .iter()
        .enumerate()
        .map(|(idx, msg)| {
            let payload = extract_payload(msg).unwrap_or_else(|err| CodingHandoverPayload {
                schema_version: "coding-handover/v1".into(),
                phase: "unknown".into(),
                status: "invalid".into(),
                summary: format!("payload parse failed: {err}"),
                request: String::new(),
                next_action: String::new(),
                files: Vec::new(),
                evidence: Vec::new(),
                risks: Vec::new(),
                proposed_changes: Vec::new(),
                tests: Vec::new(),
                notes: Vec::new(),
            });
            json!({
                "index": idx,
                "id": msg.id,
                "from": msg.from.name,
                "to": msg.to.name,
                "intent": intent_name(&msg.intent),
                "in_reply_to": msg.in_reply_to,
                "payload": payload
            })
        })
        .collect()
}

fn extract_payload(msg: &GraphMessage) -> Result<CodingHandoverPayload, String> {
    let tensor = msg
        .inputs
        .get(PAYLOAD_KEY)
        .ok_or_else(|| format!("missing payload tensor '{PAYLOAD_KEY}'"))?;
    let json = tensor
        .as_string()
        .ok_or_else(|| "payload tensor is not valid UTF-8".to_string())?;
    serde_json::from_str(&json).map_err(|e| format!("invalid payload JSON: {e}"))
}

fn intent_name(intent: &MessageIntent) -> String {
    match intent {
        MessageIntent::Execute => "execute".into(),
        MessageIntent::Optimize => "optimize".into(),
        MessageIntent::Compress { method } => format!("compress:{method}"),
        MessageIntent::Verify => "verify".into(),
        MessageIntent::Result { original_message_id } => format!("result:{original_message_id}"),
        MessageIntent::Compose => "compose".into(),
        MessageIntent::Train { epochs } => format!("train:{epochs}"),
    }
}

fn tail_text_file(path: &Path, tail: usize) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read log '{}': {err}", path.display()))?;
    let lines = raw.lines().map(|line| line.to_string()).collect::<Vec<_>>();
    let start = lines.len().saturating_sub(tail);
    Ok(lines[start..].to_vec())
}

fn now_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn run_poll(state_path: &Path) -> Value {
    let mut state = match load_supervisor_state(state_path) {
        Ok(state) => state,
        Err(err) => {
            return json!({
                "state": state_path.display().to_string(),
                "error": err
            })
        }
    };
    let now = now_string();
    let mut updates = 0usize;
    let mut completed_task_ids = Vec::new();

    for session in &mut state.sessions {
        if session.status != "Running" {
            continue;
        }
        let Some(pid) = session.process_id else {
            continue;
        };
        match is_process_running(pid) {
            Ok(true) => {}
            Ok(false) => {
                session.status = "Done".into();
                session.last_update = now.clone();
                session.exit_code = Some(0);
                if let Some(task_id) = session.current_task_id {
                    completed_task_ids.push(task_id);
                }
                updates += 1;
            }
            Err(err) => {
                state.event_log.push(SupervisorEvent {
                    at: now.clone(),
                    kind: "poll-error".into(),
                    message: format!("failed to inspect pid {}: {}", pid, err),
                    task_id: session.current_task_id,
                    session_id: Some(session.id),
                });
            }
        }
    }

    for task in &mut state.tasks {
        if completed_task_ids.contains(&task.id) && task.status == "Running" {
            task.status = "Done".into();
            task.updated_at = now.clone();
            updates += 1;
        }
    }

    if updates > 0 {
        state.event_log.push(SupervisorEvent {
            at: now.clone(),
            kind: "poll".into(),
            message: format!("applied {} session/task updates", updates),
            task_id: None,
            session_id: None,
        });
    }

    if let Err(err) = save_supervisor_state(state_path, &state) {
        return json!({
            "state": state_path.display().to_string(),
            "error": err
        });
    }

    json!({
        "state": state_path.display().to_string(),
        "status": "polled",
        "updates": updates
    })
}

fn run_tick(state_path: &Path, spawn: bool) -> Value {
    let mut state = match load_supervisor_state(state_path) {
        Ok(state) => state,
        Err(err) => {
            return json!({
                "state": state_path.display().to_string(),
                "error": err
            })
        }
    };
    let now = now_string();
    let done_ids = state
        .tasks
        .iter()
        .filter(|task| task.status == "Done")
        .map(|task| task.id)
        .collect::<Vec<_>>();

    let next_task_index = state.tasks.iter().position(|task| {
        task.status == "Queued"
            && task.depends_on.iter().all(|dependency| done_ids.contains(dependency))
    });

    let Some(task_index) = next_task_index else {
        return json!({
            "state": state_path.display().to_string(),
            "status": "idle",
            "message": "no runnable tasks"
        });
    };

    let task_id = state.tasks[task_index].id;
    let assigned_agent = state.tasks[task_index]
        .assigned_agent
        .clone()
        .or_else(|| state.agents.first().map(|agent| agent.name.clone()));
    let Some(assigned_agent) = assigned_agent else {
        return json!({
            "state": state_path.display().to_string(),
            "error": "no agents registered"
        });
    };
    state.tasks[task_index].status = "Running".into();
    state.tasks[task_index].updated_at = now.clone();
    state.tasks[task_index].assigned_agent = Some(assigned_agent.clone());

    let session_id = state.next_session_id;
    state.next_session_id += 1;
    let agent = match state.agents.iter().find(|agent| agent.name == assigned_agent) {
        Some(agent) => agent.clone(),
        None => {
            return json!({
                "state": state_path.display().to_string(),
                "error": format!("assigned agent '{}' is missing", assigned_agent)
            })
        }
    };
    let launch_command = render_launch_command(&agent);
    let (process_id, exit_code, session_status, stdout_log_path, stderr_log_path) = if spawn {
        match spawn_agent_process(state_path, session_id, &agent) {
            Ok(spawned) => (
                Some(spawned.process_id),
                None,
                "Running".to_string(),
                Some(spawned.stdout_log_path),
                Some(spawned.stderr_log_path),
            ),
            Err(err) => (
                None,
                Some(-1),
                "Failed".to_string(),
                Some(format!("ERROR: {err}")),
                None,
            ),
        }
    } else {
        (None, None, "Running".to_string(), None, None)
    };

    state.sessions.push(SupervisorSessionWritable {
        id: session_id,
        agent_name: assigned_agent.clone(),
        tab_label: format!("{assigned_agent}-task-{task_id}"),
        status: session_status.clone(),
        current_task_id: Some(task_id),
        started_at: now.clone(),
        last_update: now.clone(),
        last_handover_path: None,
        launch_command: launch_command.clone(),
        process_id,
        exit_code,
        stdout_log_path,
        stderr_log_path,
    });
    if session_status == "Failed" {
        state.tasks[task_index].status = "Failed".into();
    }
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "tick".into(),
        message: format!("scheduled task #{task_id} on agent '{assigned_agent}'"),
        task_id: Some(task_id),
        session_id: Some(session_id),
    });

    if let Err(err) = save_supervisor_state(state_path, &state) {
        return json!({
            "state": state_path.display().to_string(),
            "error": err
        });
    }

    json!({
        "state": state_path.display().to_string(),
        "status": "scheduled",
        "task_id": task_id,
        "agent": assigned_agent,
        "session_id": session_id,
        "launch_command": launch_command,
        "spawned": spawn,
        "process_id": process_id
    })
}

fn render_launch_command(agent: &SupervisorAgent) -> String {
    let mut parts = vec![agent.command.clone()];
    parts.extend(agent.args.clone());
    parts.join(" ")
}

struct SpawnedProcess {
    process_id: u32,
    stdout_log_path: String,
    stderr_log_path: String,
}

fn spawn_agent_process(state_path: &Path, session_id: u64, agent: &SupervisorAgent) -> Result<SpawnedProcess, String> {
    use std::fs::File;
    use std::process::{Command, Stdio};

    let (program, args) = resolve_launch(agent)?;
    let (stdout_log_path, stderr_log_path) = session_log_paths(state_path, session_id);
    if let Some(parent) = stdout_log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create session log directory '{}': {e}", parent.display()))?;
    }
    let stdout_file = File::create(&stdout_log_path)
        .map_err(|e| format!("failed to create stdout log '{}': {e}", stdout_log_path.display()))?;
    let stderr_file = File::create(&stderr_log_path)
        .map_err(|e| format!("failed to create stderr log '{}': {e}", stderr_log_path.display()))?;
    let mut command = Command::new(program);
    command.args(args);
    if let Some(cwd) = &agent.working_dir {
        command.current_dir(cwd);
    }
    command.stdin(Stdio::null());
    command.stdout(Stdio::from(stdout_file));
    command.stderr(Stdio::from(stderr_file));
    let child = command
        .spawn()
        .map_err(|e| format!("failed to spawn '{}': {e}", render_launch_command(agent)))?;
    Ok(SpawnedProcess {
        process_id: child.id(),
        stdout_log_path: stdout_log_path.display().to_string(),
        stderr_log_path: stderr_log_path.display().to_string(),
    })
}

fn resolve_launch(agent: &SupervisorAgent) -> Result<(String, Vec<String>), String> {
    let resolved = resolve_command_path(&agent.command).unwrap_or_else(|| agent.command.clone());
    let lower = resolved.to_ascii_lowercase();
    if lower.ends_with(".ps1") {
        let mut args = vec![
            "-NoProfile".to_string(),
            "-ExecutionPolicy".to_string(),
            "Bypass".to_string(),
            "-File".to_string(),
            resolved,
        ];
        args.extend(agent.args.clone());
        return Ok(("powershell".to_string(), args));
    }
    if lower.ends_with(".cmd") || lower.ends_with(".bat") {
        let mut args = vec!["/c".to_string(), resolved];
        args.extend(agent.args.clone());
        return Ok(("cmd".to_string(), args));
    }
    Ok((resolved, agent.args.clone()))
}

fn resolve_command_path(command: &str) -> Option<String> {
    use std::process::Command;
    if command.contains('\\') || command.contains('/') || command.contains(':') {
        return Some(command.to_string());
    }
    let output = Command::new("where").arg(command).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    stdout.lines().next().map(|line| line.trim().to_string())
}

fn session_log_paths(state_path: &Path, session_id: u64) -> (PathBuf, PathBuf) {
    let base_dir = state_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".qlang"));
    let session_dir = base_dir.join("logs").join(format!("session-{session_id}"));
    (session_dir.join("stdout.log"), session_dir.join("stderr.log"))
}

fn is_process_running(pid: u32) -> Result<bool, String> {
    use std::process::Command;
    let script = format!(
        "$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($null -eq $p) {{ exit 1 }} else {{ exit 0 }}"
    );
    let output = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .map_err(|e| format!("Get-Process probe failed: {e}"))?;
    if output.status.code() == Some(0) {
        return Ok(true);
    }
    if output.status.code() == Some(1) {
        return Ok(false);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!("Get-Process exited with status {}: {}", output.status, stderr.trim()))
}

const SUPERVISOR_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QLANG Supervisor Cockpit</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1f2630;
      --line: #30363d;
      --text: #e6edf3;
      --muted: #9da7b3;
      --ok: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --accent: #58a6ff;
      --mono: "Cascadia Code", "Consolas", monospace;
      --sans: "Segoe UI", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      background:
        radial-gradient(circle at top right, rgba(88,166,255,0.16), transparent 30%),
        radial-gradient(circle at left center, rgba(63,185,80,0.12), transparent 28%),
        var(--bg);
      color: var(--text);
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0.02em; }
    .muted { color: var(--muted); }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
    }
    input, button, select {
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(180deg, #2381ff, #1766ce);
      border-color: #1c5fb8;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.25fr 1fr;
      gap: 16px;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .panel {
      background: linear-gradient(180deg, rgba(31,38,48,0.92), rgba(22,27,34,0.92));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      min-height: 180px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat {
      background: rgba(13,17,23,0.72);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }
    .stat .label { font-size: 12px; color: var(--muted); text-transform: uppercase; }
    .stat .value { font-size: 24px; margin-top: 6px; font-weight: 700; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.03);
    }
    .badge.done { color: var(--ok); }
    .badge.running { color: var(--accent); }
    .badge.failed { color: var(--bad); }
    .badge.waiting, .badge.queued { color: var(--warn); }
    .list { display: grid; gap: 10px; }
    .card {
      border: 1px solid var(--line);
      background: rgba(13,17,23,0.58);
      border-radius: 14px;
      padding: 12px;
    }
    .card header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 8px;
    }
    .mono { font-family: var(--mono); }
    .small { font-size: 12px; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      background: rgba(13,17,23,0.85);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      min-height: 180px;
      overflow: auto;
    }
    .span-2 { grid-column: span 2; }
    @media (max-width: 1100px) {
      .stats { grid-template-columns: repeat(2, 1fr); }
      .grid { grid-template-columns: 1fr; }
      .span-2 { grid-column: span 1; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <h1>QLANG Supervisor Cockpit</h1>
        <div class="muted">One control plane for QLANG, Claude Code, Codex, Gemini, Kimi and future adapters.</div>
      </div>
      <div class="controls">
        <input id="statePath" value=".qlang/supervisor.json" size="32" />
        <select id="logTail">
          <option value="40">40 lines</option>
          <option value="80" selected>80 lines</option>
          <option value="160">160 lines</option>
        </select>
        <button id="refreshBtn" type="button">Refresh</button>
      </div>
    </div>

    <div class="stats" id="stats"></div>
    <div class="toolbar">
      <button type="button" onclick="runAction('poll')">Poll</button>
      <button type="button" onclick="runAction('tick')">Tick</button>
      <button type="button" onclick="runAction('tick_spawn')">Tick + Spawn</button>
      <button type="button" onclick="runAction('cycle_spawn')">Cycle + Spawn</button>
      <button type="button" onclick="startDaemon()" style="background:linear-gradient(180deg,#2f8f6f,#226b53); border-color:#205744;">Start Daemon</button>
      <button type="button" onclick="stopDaemon()" style="background:linear-gradient(180deg,#b64040,#8e2323); border-color:#7a1c1c;">Stop Daemon</button>
      <span class="small muted" id="actionResult"></span>
    </div>
    <div class="small muted" id="daemonStatus" style="margin-bottom:16px;">Daemon: unknown</div>

    <div class="grid">
      <section class="panel">
        <h2>Agent Presets</h2>
        <div class="list" id="presets"></div>
      </section>
        <section class="panel">
        <h2>Quick Routes</h2>
        <div class="list">
          <button type="button" onclick="dispatchQuick('demo', 'gui-smoke-test', 'Run a short supervisor smoke test and verify that session logs appear in the cockpit.', 'Run GUI Smoke Test')" style="background:linear-gradient(180deg,#5a6dd8,#3f53bb); border-color:#32449e;">Run GUI Smoke Test</button>
          <button type="button" onclick="dispatchQuick('claude', 'primary-builder', 'Analyze repository state and plan the next implementation step.', 'Analyze with Claude')">Analyze with Claude</button>
          <button type="button" onclick="dispatchQuick('codex', 'reviewer-patcher', 'Produce the next targeted patch and verification step.', 'Patch with Codex')">Patch with Codex</button>
          <button type="button" onclick="dispatchQuick('gemini', 'research-escalation', 'Provide a second-opinion analysis and summarize the best next move.', 'Escalate to Gemini')">Escalate to Gemini</button>
          <button type="button" onclick="dispatchQuick('kimi', 'long-context-analyst', 'Digest the large context and return a compact structured summary.', 'Summarize with Kimi')">Summarize with Kimi</button>
          <button type="button" onclick="suggestAgentForTask()" style="background:linear-gradient(180deg,#444f66,#303a4d); border-color:#293244;">Suggest Agent</button>
          <button type="button" onclick="prepareQuick('claude', 'primary-builder', 'Analyze repository state and plan the next implementation step.', 'Analyze with Claude')" style="background:linear-gradient(180deg,#444f66,#303a4d); border-color:#293244;">Prefill Analyze</button>
        </div>
      </section>
      <section class="panel">
        <h2>Add Agent</h2>
        <div class="list">
          <input id="agentName" placeholder="Agent name" />
          <input id="agentKind" placeholder="Kind (ClaudeCode/Codex/Gemini/Kimi/Custom)" value="Custom" />
          <input id="agentCommand" placeholder="Command" />
          <input id="agentArgs" placeholder="Args, separated by ||" />
          <button type="button" onclick="addAgent()">Register Agent</button>
        </div>
      </section>
      <section class="panel span-2">
        <h2>Enqueue Task</h2>
        <div class="list">
          <input id="taskTitle" placeholder="Task title" />
          <input id="taskGoal" placeholder="Goal" />
          <input id="taskAgent" placeholder="Assigned agent (optional)" />
          <input id="taskNextAction" placeholder="Next action" value="start work" />
          <button type="button" onclick="addTask()">Enqueue Task</button>
        </div>
      </section>
      <section class="panel span-2">
        <h2>QLMS Handover</h2>
        <div class="list">
          <input id="handoverTaskId" placeholder="Task id (optional)" />
          <input id="handoverInput" placeholder="Reply input path (only for reply)" />
          <input id="handoverOutput" placeholder="Output path" value="handoff.qlms" />
          <input id="handoverFrom" placeholder="From agent" />
          <input id="handoverTo" placeholder="To agent" />
          <input id="handoverPhase" placeholder="Phase" value="analyze" />
          <input id="handoverSummary" placeholder="Summary" />
          <input id="handoverRequest" placeholder="Request" />
          <input id="handoverNextAction" placeholder="Next action" />
          <input id="handoverFiles" placeholder="Files, separated by ||" />
          <input id="handoverEvidence" placeholder="Evidence, separated by ||" />
          <input id="handoverRisks" placeholder="Risks, separated by ||" />
          <input id="handoverChanges" placeholder="Changes, separated by ||" />
          <input id="handoverTests" placeholder="Tests, separated by ||" />
          <input id="handoverNotes" placeholder="Notes, separated by ||" />
          <div style="display:flex; gap:10px; flex-wrap:wrap;">
            <button type="button" onclick="createHandover()">Create Handover</button>
            <button type="button" onclick="replyHandover()" style="background:linear-gradient(180deg,#2f8f6f,#226b53); border-color:#205744;">Reply Handover</button>
            <button type="button" onclick="showHandover()" style="background:linear-gradient(180deg,#444f66,#303a4d); border-color:#293244;">Show Handover</button>
          </div>
          <pre id="handoverView"></pre>
        </div>
      </section>
      <section class="panel">
        <h2>Agents</h2>
        <div class="list" id="agents"></div>
      </section>
      <section class="panel">
        <h2>Tasks</h2>
        <div class="small muted" style="margin-bottom:10px;">Select a task and use the inline buttons to complete, fail, or attach a handover path.</div>
        <div class="list" id="tasks"></div>
      </section>
      <section class="panel">
        <h2>Sessions</h2>
        <div class="list" id="sessions"></div>
      </section>
      <section class="panel span-2">
        <h2>Recent Events</h2>
        <div class="list" id="events"></div>
      </section>
      <section class="panel">
        <h2>Selected Session Logs</h2>
        <div class="small muted" id="logMeta">No session selected.</div>
        <div style="display:grid; gap:12px; margin-top: 12px;">
          <div>
            <div class="small muted" style="margin-bottom:6px;">stdout</div>
            <pre id="stdoutLog"></pre>
          </div>
          <div>
            <div class="small muted" style="margin-bottom:6px;">stderr</div>
            <pre id="stderrLog"></pre>
          </div>
        </div>
      </section>
    </div>
  </div>
  <script>
    const stateInput = document.getElementById("statePath");
    const tailInput = document.getElementById("logTail");
    const refreshBtn = document.getElementById("refreshBtn");
    const actionResult = document.getElementById("actionResult");
    let selectedSessionId = null;
    let eventSource = null;

    function badge(status) {
      const normalized = String(status || "unknown").toLowerCase();
      return `<span class="badge ${normalized}">${status || "unknown"}</span>`;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    async function loadState() {
      const state = encodeURIComponent(stateInput.value.trim() || ".qlang/supervisor.json");
      const response = await fetch(`/api/supervisor/state?state=${state}`);
      const data = await response.json();
      renderState(data);
      await loadPresets();
      await refreshDaemonStatus();
      if (selectedSessionId || (data.sessions && data.sessions.length)) {
        if (!selectedSessionId && data.sessions.length) {
          selectedSessionId = data.sessions[data.sessions.length - 1].id;
        }
        await loadLogs(selectedSessionId);
      }
    }

    function renderState(data) {
      const stats = [
        ["Agents", data.agent_count ?? 0],
        ["Sessions", data.session_count ?? 0],
        ["Queued", data.queued_tasks ?? 0],
        ["Running", data.running_tasks ?? 0],
        ["Done", data.done_tasks ?? 0],
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) => `
        <div class="stat">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join("");

      document.getElementById("agents").innerHTML = (data.agents || []).map(agent => `
        <article class="card">
          <header>
            <strong>${escapeHtml(agent.name)}</strong>
            <span class="badge">${escapeHtml(agent.kind)}</span>
          </header>
          <div class="small muted mono">${escapeHtml([agent.command].concat(agent.args || []).join(" "))}</div>
          <div class="small muted" style="margin-top:8px;">skills=${escapeHtml((agent.skills || []).join(", ") || "-")}</div>
          <div class="small muted" style="margin-top:4px;">labels=${escapeHtml((agent.labels || []).join(", ") || "-")}</div>
        </article>
      `).join("") || `<div class="muted">No agents registered.</div>`;

      document.getElementById("tasks").innerHTML = (data.tasks || []).map(task => `
        <article class="card">
          <header>
            <strong>#${task.id} ${escapeHtml(task.title)}</strong>
            ${badge(task.status)}
          </header>
          <div class="small">${escapeHtml(task.goal)}</div>
          <div class="small muted" style="margin-top:8px;">
            agent=${escapeHtml(task.assigned_agent || "unassigned")} · next=${escapeHtml(task.next_action || "-")}
          </div>
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
            <button type="button" onclick="completeTask(${task.id})">Complete</button>
            <button type="button" onclick="failTask(${task.id})" style="background:linear-gradient(180deg,#b64040,#8e2323); border-color:#7a1c1c;">Fail</button>
            <button type="button" onclick="handoverTask(${task.id})" style="background:linear-gradient(180deg,#2f8f6f,#226b53); border-color:#205744;">Handover</button>
          </div>
        </article>
      `).join("") || `<div class="muted">No tasks queued.</div>`;

      document.getElementById("sessions").innerHTML = (data.sessions || []).map(session => `
        <article class="card" data-session-id="${session.id}" style="cursor:pointer;">
          <header>
            <strong>#${session.id} ${escapeHtml(session.agent_name)}</strong>
            ${badge(session.status)}
          </header>
          <div class="small muted mono">${escapeHtml(session.launch_command || "")}</div>
          <div class="small muted" style="margin-top:8px;">
            task=${escapeHtml(session.current_task_id ?? "-")} · pid=${escapeHtml(session.process_id ?? "-")}
          </div>
        </article>
      `).join("") || `<div class="muted">No sessions yet.</div>`;

      document.querySelectorAll("[data-session-id]").forEach(node => {
        node.addEventListener("click", async () => {
          selectedSessionId = Number(node.getAttribute("data-session-id"));
          await loadLogs(selectedSessionId);
          startStream();
        });
      });

      document.getElementById("events").innerHTML = (data.recent_events || []).map(event => `
        <article class="card">
          <header>
            <strong>${escapeHtml(event.kind)}</strong>
            <span class="small muted mono">${escapeHtml(event.at)}</span>
          </header>
          <div class="small">${escapeHtml(event.message)}</div>
        </article>
      `).join("") || `<div class="muted">No recent events.</div>`;
    }

    async function loadLogs(sessionId) {
      const state = encodeURIComponent(stateInput.value.trim() || ".qlang/supervisor.json");
      const tail = encodeURIComponent(tailInput.value);
      const response = await fetch(`/api/supervisor/logs?state=${state}&session=${sessionId}&tail=${tail}`);
      const data = await response.json();
      document.getElementById("logMeta").textContent =
        `session=${data.session_id ?? "-"} agent=${data.agent ?? "-"} task=${data.task_id ?? "-"}`;
      document.getElementById("stdoutLog").textContent = (data.stdout_tail || []).join("\n");
      document.getElementById("stderrLog").textContent = (data.stderr_tail || []).join("\n");
    }

    function startStream() {
      if (eventSource) {
        eventSource.close();
      }
      const state = encodeURIComponent(stateInput.value.trim() || ".qlang/supervisor.json");
      const tail = encodeURIComponent(tailInput.value);
      const session = selectedSessionId ? `&session=${selectedSessionId}` : "";
      eventSource = new EventSource(`/api/supervisor/stream?state=${state}&tail=${tail}${session}`);
      eventSource.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.state) {
            renderState(payload.state);
          }
          if (payload.logs && !payload.logs.error) {
            document.getElementById("logMeta").textContent =
              `session=${payload.logs.session_id ?? "-"} agent=${payload.logs.agent ?? "-"} task=${payload.logs.task_id ?? "-"}`;
            document.getElementById("stdoutLog").textContent = (payload.logs.stdout_tail || []).join("\n");
            document.getElementById("stderrLog").textContent = (payload.logs.stderr_tail || []).join("\n");
          }
        } catch (error) {
          actionResult.textContent = `stream parse error: ${error}`;
        }
      };
      eventSource.onerror = () => {
        actionResult.textContent = "supervisor stream disconnected";
      };
    }

    async function postJson(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      return await response.json();
    }

    async function loadPresets() {
      const response = await fetch("/api/supervisor/presets");
      const presets = await response.json();
      document.getElementById("presets").innerHTML = (presets || []).map(preset => `
        <article class="card">
          <header>
            <strong>${escapeHtml(preset.name)}</strong>
            <span class="badge ${preset.available ? "done" : "failed"}">${preset.available ? "available" : "missing"}</span>
          </header>
          <div class="small muted">role=${escapeHtml(preset.role || "-")}</div>
          <div class="small muted mono">${escapeHtml([preset.command].concat(preset.args || []).join(" "))}</div>
          <div class="small muted mono" style="margin-top:6px;">${escapeHtml(preset.resolved_path || "not found in PATH")}</div>
          <div class="small" style="margin-top:8px;">${escapeHtml(preset.description || "")}</div>
          <div class="small muted" style="margin-top:8px;">capabilities=${escapeHtml((preset.capabilities || []).join(", "))}</div>
          <div class="small muted" style="margin-top:4px;">recommended=${escapeHtml((preset.recommended_for || []).join(", "))}</div>
          <div style="display:flex; gap:8px; margin-top:10px;">
            <button type="button" onclick="installPreset('${preset.key}')" ${preset.available ? "" : "disabled"}>Install</button>
            <button type="button" onclick="usePreset('${preset.key}', '${preset.role}', '${preset.default_next_action || ""}')" ${preset.available ? "" : "disabled"}>Use For Task</button>
          </div>
        </article>
      `).join("");
    }

    async function installPreset(preset) {
      const result = await postJson("/api/supervisor/install-preset", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        preset
      });
      actionResult.textContent = result.error || `${result.status}: ${result.agent}`;
      await loadState();
    }

    async function usePreset(presetKey, role, nextAction) {
      document.getElementById("taskAgent").value = presetKey;
      document.getElementById("taskNextAction").value = nextAction || "start work";
      const title = document.getElementById("taskTitle");
      if (!title.value.trim()) {
        title.value = `Task for ${presetKey}`;
      }
      const goal = document.getElementById("taskGoal");
      if (!goal.value.trim()) {
        goal.value = `Run the ${role} workflow through ${presetKey}`;
      }
      actionResult.textContent = `preset selected: ${presetKey} (${role})`;
    }

    function prepareQuick(presetKey, role, nextAction, title) {
      document.getElementById("taskAgent").value = presetKey;
      document.getElementById("taskNextAction").value = nextAction || "start work";
      document.getElementById("taskTitle").value = title;
      if (!document.getElementById("taskGoal").value.trim()) {
        document.getElementById("taskGoal").value = `Run the ${role} workflow through ${presetKey}`;
      }
      actionResult.textContent = `quick route selected: ${presetKey}`;
    }

    async function dispatchQuick(presetKey, role, nextAction, title) {
      const existingGoal = document.getElementById("taskGoal").value.trim();
      const goal = existingGoal || `Run the ${role} workflow through ${presetKey}`;
      const result = await postJson("/api/supervisor/dispatch", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        preset: presetKey,
        title,
        goal,
        next_action: nextAction,
        start_now: true,
        spawn: true
      });
      actionResult.textContent = result.error || `dispatched ${presetKey} task ${result.task_id}`;
      if (!result.error) {
        document.getElementById("taskAgent").value = presetKey;
        document.getElementById("taskTitle").value = title;
        document.getElementById("taskGoal").value = goal;
        document.getElementById("taskNextAction").value = nextAction || "start work";
        await loadState();
      }
    }

    async function suggestAgentForTask() {
      const result = await postJson("/api/supervisor/suggest-agent", {
        title: document.getElementById("taskTitle").value.trim(),
        goal: document.getElementById("taskGoal").value.trim(),
        route: document.getElementById("taskAgent").value.trim(),
        risk: document.getElementById("taskNextAction").value.trim()
      });
      if (result.error) {
        actionResult.textContent = result.error;
        return;
      }
      usePreset(result.preset, result.role, result.default_next_action);
      actionResult.textContent = `suggested ${result.preset} (${result.role})`;
    }

    async function refreshDaemonStatus() {
      const response = await fetch("/api/supervisor/daemon/status");
      const status = await response.json();
      const statePath = status.state_path || stateInput.value.trim() || ".qlang/supervisor.json";
      document.getElementById("daemonStatus").textContent =
        `Daemon: ${status.running ? "running" : "stopped"} · pid=${status.process_id ?? "-"} · state=${statePath} · interval=${status.interval_ms ?? "-"}ms`;
    }

    async function startDaemon() {
      const result = await postJson("/api/supervisor/daemon/start", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        spawn: true,
        interval_ms: 2000
      });
      actionResult.textContent = result.error || `daemon started pid=${result.process_id}`;
      await refreshDaemonStatus();
    }

    async function stopDaemon() {
      const result = await postJson("/api/supervisor/daemon/stop", {});
      actionResult.textContent = result.error || `${result.status || "stopped"} pid=${result.process_id ?? "-"}`;
      await refreshDaemonStatus();
    }

    async function runAction(action) {
      const result = await postJson("/api/supervisor/action", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        action
      });
      actionResult.textContent = result.error || result.status || JSON.stringify(result);
      await loadState();
    }

    async function addAgent() {
      const args = document.getElementById("agentArgs").value
        .split("||")
        .map(value => value.trim())
        .filter(Boolean);
      const result = await postJson("/api/supervisor/agent", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        name: document.getElementById("agentName").value.trim(),
        kind: document.getElementById("agentKind").value.trim() || "Custom",
        command: document.getElementById("agentCommand").value.trim(),
        args
      });
      actionResult.textContent = result.error || `${result.status}: ${result.agent || ""}`;
      await loadState();
    }

    async function addTask() {
      const result = await postJson("/api/supervisor/task", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        title: document.getElementById("taskTitle").value.trim(),
        goal: document.getElementById("taskGoal").value.trim(),
        assigned_agent: document.getElementById("taskAgent").value.trim() || null,
        next_action: document.getElementById("taskNextAction").value.trim() || "start work"
      });
      actionResult.textContent = result.error || `${result.status}: task ${result.task_id || ""}`;
      await loadState();
    }

    function splitField(id) {
      return document.getElementById(id).value
        .split("||")
        .map(value => value.trim())
        .filter(Boolean);
    }

    function handoverPayload() {
      return {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        task_id: Number(document.getElementById("handoverTaskId").value) || null,
        input: document.getElementById("handoverInput").value.trim() || null,
        output: document.getElementById("handoverOutput").value.trim() || "handoff.qlms",
        from: document.getElementById("handoverFrom").value.trim(),
        to: document.getElementById("handoverTo").value.trim(),
        phase: document.getElementById("handoverPhase").value.trim() || "analyze",
        summary: document.getElementById("handoverSummary").value.trim(),
        request: document.getElementById("handoverRequest").value.trim(),
        next_action: document.getElementById("handoverNextAction").value.trim(),
        files: splitField("handoverFiles"),
        evidence: splitField("handoverEvidence"),
        risks: splitField("handoverRisks"),
        proposed_changes: splitField("handoverChanges"),
        tests: splitField("handoverTests"),
        notes: splitField("handoverNotes")
      };
    }

    async function createHandover() {
      const result = await postJson("/api/supervisor/handover/create", handoverPayload());
      actionResult.textContent = result.error || `handover written: ${result.written}`;
      if (!result.error) {
        document.getElementById("handoverOutput").value = result.written;
        await loadState();
      }
    }

    async function replyHandover() {
      const payload = handoverPayload();
      const result = await postJson("/api/supervisor/handover/reply", payload);
      actionResult.textContent = result.error || `handover reply written: ${result.written}`;
      if (!result.error) {
        document.getElementById("handoverOutput").value = result.written;
        await loadState();
      }
    }

    async function showHandover() {
      const input = encodeURIComponent(document.getElementById("handoverInput").value.trim() || document.getElementById("handoverOutput").value.trim());
      const response = await fetch(`/api/supervisor/handover/show?input=${input}`);
      const result = await response.json();
      document.getElementById("handoverView").textContent = JSON.stringify(result, null, 2);
      actionResult.textContent = result.error || "handover loaded";
    }

    async function completeTask(taskId) {
      const note = window.prompt("Completion note", "Task completed via cockpit");
      const result = await postJson("/api/supervisor/task-action", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        task_id: taskId,
        action: "complete",
        note: note || null
      });
      actionResult.textContent = result.error || `${result.action}: task ${result.task_id}`;
      await loadState();
    }

    async function failTask(taskId) {
      const note = window.prompt("Failure note", "Task failed via cockpit");
      const result = await postJson("/api/supervisor/task-action", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        task_id: taskId,
        action: "fail",
        note: note || null
      });
      actionResult.textContent = result.error || `${result.action}: task ${result.task_id}`;
      await loadState();
    }

    async function handoverTask(taskId) {
      const path = window.prompt("Handover path", "handoff.qlms");
      if (!path) return;
      const result = await postJson("/api/supervisor/task-action", {
        state: stateInput.value.trim() || ".qlang/supervisor.json",
        task_id: taskId,
        action: "handover",
        path
      });
      actionResult.textContent = result.error || `${result.action}: task ${result.task_id}`;
      await loadState();
    }

    refreshBtn.addEventListener("click", loadState);
    stateInput.addEventListener("change", startStream);
    tailInput.addEventListener("change", startStream);
    window.addEventListener("beforeunload", () => { if (eventSource) eventSource.close(); });
    startStream();
    loadState();
  </script>
</body>
</html>
"#;
