use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorState {
    pub version: String,
    pub permission_profile: PermissionProfile,
    pub agents: Vec<AgentSpec>,
    pub sessions: Vec<AgentSession>,
    pub tasks: Vec<TaskRecord>,
    pub event_log: Vec<SupervisorEvent>,
    pub next_task_id: u64,
    pub next_session_id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionProfile {
    pub mode: String,
    pub allow_edit: bool,
    pub allow_execute: bool,
    pub allow_network: bool,
    pub allow_sensitive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    pub name: String,
    pub kind: AgentKind,
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: Option<String>,
    pub plugins: Vec<String>,
    pub skills: Vec<String>,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentKind {
    ClaudeCode,
    Codex,
    Gemini,
    Kimi,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionStatus {
    Idle,
    Running,
    Waiting,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskStatus {
    Queued,
    Running,
    Waiting,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: u64,
    pub agent_name: String,
    pub tab_label: String,
    pub status: SessionStatus,
    pub current_task_id: Option<u64>,
    pub started_at: String,
    pub last_update: String,
    pub last_handover_path: Option<String>,
    pub launch_command: String,
    pub process_id: Option<u32>,
    pub exit_code: Option<i32>,
    pub stdout_log_path: Option<String>,
    pub stderr_log_path: Option<String>,
    pub prompt_log_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: u64,
    pub title: String,
    pub goal: String,
    pub assigned_agent: Option<String>,
    pub status: TaskStatus,
    pub depends_on: Vec<u64>,
    pub created_at: String,
    pub updated_at: String,
    pub next_action: String,
    pub latest_handover_path: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorEvent {
    pub at: String,
    pub kind: String,
    pub message: String,
    pub task_id: Option<u64>,
    pub session_id: Option<u64>,
}

pub fn handle_supervisor(args: &[String]) {
    let Some(subcommand) = args.first().map(|s| s.as_str()) else {
        print_supervisor_usage();
        return;
    };

    match subcommand {
        "init" => cmd_init(&args[1..]),
        "add-agent" => cmd_add_agent(&args[1..]),
        "enqueue" => cmd_enqueue(&args[1..]),
        "poll" => cmd_poll(&args[1..]),
        "cycle" => cmd_cycle(&args[1..]),
        "daemon" => cmd_daemon(&args[1..]),
        "tick" => cmd_tick(&args[1..]),
        "complete" => cmd_complete(&args[1..]),
        "fail" => cmd_fail(&args[1..]),
        "handover" => cmd_handover(&args[1..]),
        "logs" => cmd_logs(&args[1..]),
        "show" => cmd_show(&args[1..]),
        "help" | "-h" | "--help" => print_supervisor_usage(),
        other => fail(2, &format!("unknown supervisor command: {other}")),
    }
}

pub fn handle_run_agent(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let agent_name = arg_value(args, "--agent").unwrap_or_else(|| fail(2, "--agent is required"));
    let task_id: Option<u64> = arg_value(args, "--task").and_then(|v| v.parse().ok());
    let tab_label = arg_value(args, "--tab").unwrap_or_else(|| agent_name.clone());
    let spawn = flag_present(args, "--spawn");

    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let agent = state
        .agents
        .iter()
        .find(|candidate| candidate.name == agent_name)
        .cloned()
        .unwrap_or_else(|| fail(2, &format!("agent '{agent_name}' is not registered")));

    let session_id = state.next_session_id;
    state.next_session_id += 1;
    let now = now_string();
    let launch_command = render_launch_command(&agent);
    let (process_id, exit_code, session_status, stdout_log_path, stderr_log_path, prompt_log_path) = if spawn {
        match spawn_agent_process(&state_path, session_id, &agent) {
            Ok(spawned) => (
                Some(spawned.process_id),
                None,
                SessionStatus::Running,
                Some(spawned.stdout_log_path),
                Some(spawned.stderr_log_path),
                Some(spawned.prompt_log_path),
            ),
            Err(err) => {
                eprintln!("[run-agent] spawn failed: {err}");
                (None, Some(-1), SessionStatus::Failed, None, None, None)
            }
        }
    } else {
        let (_, _, prompt_log_path) = session_io_paths(&state_path, session_id);
        let _ = ensure_parent_dir(&prompt_log_path);
        let _ = File::create(&prompt_log_path);
        (
            None,
            None,
            SessionStatus::Running,
            None,
            None,
            Some(prompt_log_path.display().to_string()),
        )
    };
    state.sessions.push(AgentSession {
        id: session_id,
        agent_name: agent.name.clone(),
        tab_label,
        status: session_status.clone(),
        current_task_id: task_id,
        started_at: now.clone(),
        last_update: now.clone(),
        last_handover_path: None,
        launch_command: launch_command.clone(),
        process_id,
        exit_code,
        stdout_log_path,
        stderr_log_path,
        prompt_log_path,
    });

    if let Some(task_id) = task_id {
        if let Some(task) = state.tasks.iter_mut().find(|task| task.id == task_id) {
            task.status = if session_status == SessionStatus::Failed {
                TaskStatus::Failed
            } else {
                TaskStatus::Running
            };
            task.updated_at = now.clone();
            if task.assigned_agent.is_none() {
                task.assigned_agent = Some(agent.name.clone());
            }
        }
    }

    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "run-agent".into(),
        message: format!("started agent '{}' with command '{}'", agent.name, launch_command),
        task_id,
        session_id: Some(session_id),
    });
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "session_id": session_id,
            "agent": agent.name,
            "launch_command": launch_command,
            "task_id": task_id,
            "spawned": spawn,
            "process_id": process_id
        }))
        .expect("serialize run-agent output")
    );
}

fn cmd_init(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let mode = arg_value(args, "--mode").unwrap_or_else(|| "continuous".into());
    let state = SupervisorState {
        version: "supervisor/v1".into(),
        permission_profile: PermissionProfile {
            mode,
            allow_edit: true,
            allow_execute: true,
            allow_network: true,
            allow_sensitive: false,
        },
        agents: Vec::new(),
        sessions: Vec::new(),
        tasks: Vec::new(),
        event_log: vec![SupervisorEvent {
            at: now_string(),
            kind: "init".into(),
            message: "initialized supervisor state".into(),
            task_id: None,
            session_id: None,
        }],
        next_task_id: 1,
        next_session_id: 1,
    };
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "mode": state.permission_profile.mode,
            "status": "initialized"
        }))
        .expect("serialize init output")
    );
}

fn cmd_add_agent(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let name = arg_value(args, "--name").unwrap_or_else(|| fail(2, "--name is required"));
    let kind = parse_agent_kind(&arg_value(args, "--kind").unwrap_or_else(|| "custom".into()));
    let command = arg_value(args, "--command").unwrap_or_else(|| fail(2, "--command is required"));
    let args_list = collect_values(args, "--arg");
    let plugins = collect_values(args, "--plugin");
    let skills = collect_values(args, "--skill");
    let labels = collect_values(args, "--label");
    let working_dir = arg_value(args, "--cwd");

    if state.agents.iter().any(|agent| agent.name == name) {
        fail(2, &format!("agent '{name}' already exists"));
    }

    state.agents.push(AgentSpec {
        name: name.clone(),
        kind,
        command: command.clone(),
        args: args_list,
        working_dir,
        plugins,
        skills,
        labels,
    });
    state.event_log.push(SupervisorEvent {
        at: now_string(),
        kind: "add-agent".into(),
        message: format!("registered agent '{name}'"),
        task_id: None,
        session_id: None,
    });
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "agent": name,
            "status": "registered"
        }))
        .expect("serialize add-agent output")
    );
}

fn cmd_enqueue(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let title = arg_value(args, "--title").unwrap_or_else(|| fail(2, "--title is required"));
    let goal = arg_value(args, "--goal").unwrap_or_else(|| fail(2, "--goal is required"));
    let assigned_agent = arg_value(args, "--agent");
    let next_action = arg_value(args, "--next-action").unwrap_or_else(|| "start work".into());
    let depends_on = collect_values(args, "--depends-on")
        .into_iter()
        .filter_map(|value| value.parse::<u64>().ok())
        .collect::<Vec<_>>();
    let notes = collect_values(args, "--note");
    let id = state.next_task_id;
    state.next_task_id += 1;
    let now = now_string();
    state.tasks.push(TaskRecord {
        id,
        title,
        goal,
        assigned_agent,
        status: TaskStatus::Queued,
        depends_on,
        created_at: now.clone(),
        updated_at: now.clone(),
        next_action,
        latest_handover_path: None,
        notes,
    });
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "enqueue".into(),
        message: format!("enqueued task #{id}"),
        task_id: Some(id),
        session_id: None,
    });
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "task_id": id,
            "status": "queued"
        }))
        .expect("serialize enqueue output")
    );
}

fn cmd_tick(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let spawn = flag_present(args, "--spawn");
    let now = now_string();
    let done_ids = state
        .tasks
        .iter()
        .filter(|task| task.status == TaskStatus::Done)
        .map(|task| task.id)
        .collect::<Vec<_>>();

    let next_task_index = state.tasks.iter().position(|task| {
        task.status == TaskStatus::Queued
            && task
                .depends_on
                .iter()
                .all(|dependency| done_ids.contains(dependency))
    });

    let Some(task_index) = next_task_index else {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "state": state_path.display().to_string(),
                "status": "idle",
                "message": "no runnable tasks"
            }))
            .expect("serialize tick idle output")
        );
        return;
    };

    let task_id = state.tasks[task_index].id;
    let assigned_agent = state.tasks[task_index]
        .assigned_agent
        .clone()
        .or_else(|| state.agents.first().map(|agent| agent.name.clone()))
        .unwrap_or_else(|| fail(2, "no agents registered"));
    state.tasks[task_index].status = TaskStatus::Running;
    state.tasks[task_index].updated_at = now.clone();
    state.tasks[task_index].assigned_agent = Some(assigned_agent.clone());

    let session_id = state.next_session_id;
    state.next_session_id += 1;
    let agent = state
        .agents
        .iter()
        .find(|agent| agent.name == assigned_agent)
        .cloned()
        .unwrap_or_else(|| fail(2, &format!("assigned agent '{}' is missing", assigned_agent)));
    let launch_command = render_launch_command(&agent);
    let (process_id, exit_code, session_status, stdout_log_path, stderr_log_path, prompt_log_path) = if spawn {
        match spawn_agent_process(&state_path, session_id, &agent) {
            Ok(spawned) => (
                Some(spawned.process_id),
                None,
                SessionStatus::Running,
                Some(spawned.stdout_log_path),
                Some(spawned.stderr_log_path),
                Some(spawned.prompt_log_path),
            ),
            Err(err) => {
                eprintln!("[supervisor tick] spawn failed: {err}");
                (None, Some(-1), SessionStatus::Failed, None, None, None)
            }
        }
    } else {
        let (_, _, prompt_log_path) = session_io_paths(&state_path, session_id);
        let _ = ensure_parent_dir(&prompt_log_path);
        let _ = File::create(&prompt_log_path);
        (
            None,
            None,
            SessionStatus::Running,
            None,
            None,
            Some(prompt_log_path.display().to_string()),
        )
    };
    state.sessions.push(AgentSession {
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
        prompt_log_path,
    });
    if session_status == SessionStatus::Failed {
        state.tasks[task_index].status = TaskStatus::Failed;
    }
    state.event_log.push(SupervisorEvent {
        at: now,
        kind: "tick".into(),
        message: format!("scheduled task #{task_id} on agent '{assigned_agent}'"),
        task_id: Some(task_id),
        session_id: Some(session_id),
    });
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "status": "scheduled",
            "task_id": task_id,
            "agent": assigned_agent,
            "session_id": session_id,
            "launch_command": launch_command,
            "spawned": spawn,
            "process_id": process_id
        }))
        .expect("serialize tick output")
    );
}

fn cmd_poll(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let now = now_string();
    let mut updates = 0usize;
    let mut completed_task_ids = Vec::new();

    for session in &mut state.sessions {
        if session.status != SessionStatus::Running {
            continue;
        }
        let Some(pid) = session.process_id else {
            continue;
        };
        match is_process_running(pid) {
            Ok(true) => {}
            Ok(false) => {
                session.status = SessionStatus::Done;
                session.last_update = now.clone();
                session.exit_code = Some(0);
                if let Some(task_id) = session.current_task_id {
                    completed_task_ids.push(task_id);
                }
                updates += 1;
            }
            Err(err) => {
                session.status = SessionStatus::Waiting;
                session.last_update = now.clone();
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
        if completed_task_ids.contains(&task.id) && task.status == TaskStatus::Running {
            task.status = TaskStatus::Done;
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
        save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
    }

    let snapshot = build_cockpit_snapshot(&state, &state_path);
    println!(
        "{}",
        serde_json::to_string_pretty(&snapshot).expect("serialize poll output")
    );
}

fn cmd_cycle(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let spawn = flag_present(args, "--spawn");
    poll_state_file(&state_path);
    let tick_args = if spawn {
        vec![
            "--state".to_string(),
            state_path.display().to_string(),
            "--spawn".to_string(),
        ]
    } else {
        vec!["--state".to_string(), state_path.display().to_string()]
    };
    cmd_tick(&tick_args);
}

fn cmd_daemon(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let spawn = flag_present(args, "--spawn");
    let once = flag_present(args, "--once");
    let interval_ms: u64 = arg_value(args, "--interval-ms")
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000);
    let max_cycles: Option<u64> = arg_value(args, "--max-cycles").and_then(|value| value.parse().ok());

    let mut cycles = 0u64;
    loop {
        poll_state_file(&state_path);
        let tick_args = if spawn {
            vec![
                "--state".to_string(),
                state_path.display().to_string(),
                "--spawn".to_string(),
            ]
        } else {
            vec!["--state".to_string(), state_path.display().to_string()]
        };
        cmd_tick(&tick_args);
        cycles += 1;

        let state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
        let summary = serde_json::json!({
            "state": state_path.display().to_string(),
            "mode": "daemon",
            "cycle": cycles,
            "spawn": spawn,
            "queued_tasks": state.tasks.iter().filter(|task| task.status == TaskStatus::Queued).count(),
            "running_tasks": state.tasks.iter().filter(|task| task.status == TaskStatus::Running).count(),
            "done_tasks": state.tasks.iter().filter(|task| task.status == TaskStatus::Done).count(),
            "failed_tasks": state.tasks.iter().filter(|task| task.status == TaskStatus::Failed).count()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).expect("serialize daemon summary")
        );

        if once || max_cycles.map(|limit| cycles >= limit).unwrap_or(false) {
            break;
        }
        thread::sleep(Duration::from_millis(interval_ms));
    }
}

fn cmd_complete(args: &[String]) {
    transition_task(args, TaskStatus::Done, SessionStatus::Done, "complete");
}

fn cmd_fail(args: &[String]) {
    transition_task(args, TaskStatus::Failed, SessionStatus::Failed, "fail");
}

fn transition_task(args: &[String], task_status: TaskStatus, session_status: SessionStatus, event_kind: &str) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let task_id: u64 = arg_value(args, "--task")
        .unwrap_or_else(|| fail(2, "--task is required"))
        .parse()
        .unwrap_or_else(|_| fail(2, "--task must be a valid integer"));
    let note = arg_value(args, "--note");
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let now = now_string();

    let task = state
        .tasks
        .iter_mut()
        .find(|task| task.id == task_id)
        .unwrap_or_else(|| fail(2, &format!("task #{task_id} not found")));
    task.status = task_status.clone();
    task.updated_at = now.clone();
    if let Some(note) = &note {
        task.notes.push(note.clone());
    }

    if let Some(session) = state
        .sessions
        .iter_mut()
        .find(|session| session.current_task_id == Some(task_id) && session.status == SessionStatus::Running)
    {
        let failed = session_status == SessionStatus::Failed;
        session.status = session_status;
        session.last_update = now.clone();
        session.exit_code = if failed { Some(1) } else { Some(0) };
    }

    state.event_log.push(SupervisorEvent {
        at: now,
        kind: event_kind.into(),
        message: format!("task #{task_id} -> {:?}", task_status),
        task_id: Some(task_id),
        session_id: None,
    });
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
}

fn cmd_handover(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let task_id: u64 = arg_value(args, "--task")
        .unwrap_or_else(|| fail(2, "--task is required"))
        .parse()
        .unwrap_or_else(|_| fail(2, "--task must be a valid integer"));
    let path = arg_value(args, "--path").unwrap_or_else(|| fail(2, "--path is required"));
    let mut state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let now = now_string();
    let task = state
        .tasks
        .iter_mut()
        .find(|task| task.id == task_id)
        .unwrap_or_else(|| fail(2, &format!("task #{task_id} not found")));
    task.latest_handover_path = Some(path.clone());
    task.updated_at = now.clone();
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
    save_state(&state_path, &state).unwrap_or_else(|e| fail(3, &e));
}

fn cmd_show(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let pretty = !args.iter().any(|arg| arg == "--json");
    let snapshot = build_cockpit_snapshot(&state, &state_path);
    let rendered = if pretty {
        serde_json::to_string_pretty(&snapshot)
    } else {
        serde_json::to_string(&snapshot)
    }
    .expect("serialize supervisor snapshot");
    println!("{rendered}");
}

fn cmd_logs(args: &[String]) {
    let state_path = PathBuf::from(arg_value(args, "--state").unwrap_or_else(|| ".qlang/supervisor.json".into()));
    let state = load_state(&state_path).unwrap_or_else(|e| fail(2, &e));
    let tail: usize = arg_value(args, "--tail")
        .and_then(|value| value.parse().ok())
        .unwrap_or(40);
    let session = if let Some(id) = arg_value(args, "--session").and_then(|value| value.parse::<u64>().ok()) {
        state.sessions.iter().find(|session| session.id == id)
    } else if let Some(task_id) = arg_value(args, "--task").and_then(|value| value.parse::<u64>().ok()) {
        state.sessions.iter().rev().find(|session| session.current_task_id == Some(task_id))
    } else {
        state.sessions.last()
    }
    .unwrap_or_else(|| fail(2, "no matching session found"));

    let stdout = session
        .stdout_log_path
        .as_ref()
        .map(|path| tail_text_file(Path::new(path), tail))
        .transpose()
        .unwrap_or_else(|e| fail(3, &e));
    let stderr = session
        .stderr_log_path
        .as_ref()
        .map(|path| tail_text_file(Path::new(path), tail))
        .transpose()
        .unwrap_or_else(|e| fail(3, &e));

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "state": state_path.display().to_string(),
            "session_id": session.id,
            "agent": session.agent_name,
            "task_id": session.current_task_id,
            "stdout_log_path": session.stdout_log_path,
            "stderr_log_path": session.stderr_log_path,
            "stdout_tail": stdout,
            "stderr_tail": stderr
        }))
        .expect("serialize logs output")
    );
}

#[derive(Serialize)]
struct CockpitSnapshot {
    state_path: String,
    permission_profile: PermissionProfile,
    agent_count: usize,
    session_count: usize,
    queued_tasks: usize,
    running_tasks: usize,
    waiting_tasks: usize,
    done_tasks: usize,
    failed_tasks: usize,
    agents: Vec<AgentSpec>,
    sessions: Vec<AgentSession>,
    tasks: Vec<TaskRecord>,
    recent_events: Vec<SupervisorEvent>,
}

fn build_cockpit_snapshot(state: &SupervisorState, path: &Path) -> CockpitSnapshot {
    let mut counts = HashMap::new();
    for task in &state.tasks {
        *counts.entry(format!("{:?}", task.status)).or_insert(0usize) += 1;
    }
    CockpitSnapshot {
        state_path: path.display().to_string(),
        permission_profile: state.permission_profile.clone(),
        agent_count: state.agents.len(),
        session_count: state.sessions.len(),
        queued_tasks: *counts.get("Queued").unwrap_or(&0),
        running_tasks: *counts.get("Running").unwrap_or(&0),
        waiting_tasks: *counts.get("Waiting").unwrap_or(&0),
        done_tasks: *counts.get("Done").unwrap_or(&0),
        failed_tasks: *counts.get("Failed").unwrap_or(&0),
        agents: state.agents.clone(),
        sessions: state.sessions.clone(),
        tasks: state.tasks.clone(),
        recent_events: state.event_log.iter().rev().take(20).cloned().collect(),
    }
}

struct SpawnedProcess {
    process_id: u32,
    stdout_log_path: String,
    stderr_log_path: String,
    prompt_log_path: String,
}

fn spawn_agent_process(state_path: &Path, session_id: u64, agent: &AgentSpec) -> Result<SpawnedProcess, String> {
    let (program, args) = resolve_launch(agent)?;
    let (stdout_log_path, stderr_log_path, prompt_log_path) = session_io_paths(state_path, session_id);
    ensure_parent_dir(&stdout_log_path)?;
    let stdout_file = File::create(&stdout_log_path)
        .map_err(|e| format!("failed to create stdout log '{}': {e}", stdout_log_path.display()))?;
    let stderr_file = File::create(&stderr_log_path)
        .map_err(|e| format!("failed to create stderr log '{}': {e}", stderr_log_path.display()))?;
    File::create(&prompt_log_path)
        .map_err(|e| format!("failed to create prompt log '{}': {e}", prompt_log_path.display()))?;
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
        prompt_log_path: prompt_log_path.display().to_string(),
    })
}

fn resolve_launch(agent: &AgentSpec) -> Result<(String, Vec<String>), String> {
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

fn is_process_running(pid: u32) -> Result<bool, String> {
    if cfg!(target_os = "windows") {
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
    } else {
        let status = Command::new("sh")
            .args(["-lc", &format!("kill -0 {pid} >/dev/null 2>&1")])
            .status()
            .map_err(|e| format!("kill -0 failed: {e}"))?;
        Ok(status.success())
    }
}

fn poll_state_file(state_path: &Path) {
    let mut state = load_state(state_path).unwrap_or_else(|e| fail(2, &e));
    let now = now_string();
    let mut updates = 0usize;
    let mut completed_task_ids = Vec::new();

    for session in &mut state.sessions {
        if session.status != SessionStatus::Running {
            continue;
        }
        let Some(pid) = session.process_id else {
            continue;
        };
        match is_process_running(pid) {
            Ok(true) => {}
            Ok(false) => {
                session.status = SessionStatus::Done;
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
        if completed_task_ids.contains(&task.id) && task.status == TaskStatus::Running {
            task.status = TaskStatus::Done;
            task.updated_at = now.clone();
            updates += 1;
        }
    }

    if updates > 0 {
        state.event_log.push(SupervisorEvent {
            at: now,
            kind: "poll".into(),
            message: format!("applied {} session/task updates", updates),
            task_id: None,
            session_id: None,
        });
        save_state(state_path, &state).unwrap_or_else(|e| fail(3, &e));
    }
}

fn load_state(path: &Path) -> Result<SupervisorState, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read supervisor state '{}': {e}", path.display()))?;
    serde_json::from_str(&raw)
        .map_err(|e| format!("failed to parse supervisor state '{}': {e}", path.display()))
}

fn save_state(path: &Path, state: &SupervisorState) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create supervisor directory '{}': {e}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| format!("failed to serialize supervisor state: {e}"))?;
    fs::write(path, json).map_err(|e| format!("failed to write supervisor state '{}': {e}", path.display()))
}

fn parse_agent_kind(value: &str) -> AgentKind {
    match value {
        "claude" | "claude-code" => AgentKind::ClaudeCode,
        "codex" => AgentKind::Codex,
        "gemini" => AgentKind::Gemini,
        "kimi" => AgentKind::Kimi,
        _ => AgentKind::Custom,
    }
}

fn render_launch_command(agent: &AgentSpec) -> String {
    let mut parts = vec![agent.command.clone()];
    parts.extend(agent.args.clone());
    parts.join(" ")
}

fn session_io_paths(state_path: &Path, session_id: u64) -> (PathBuf, PathBuf, PathBuf) {
    let base_dir = state_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".qlang"));
    let session_dir = base_dir.join("logs").join(format!("session-{session_id}"));
    (
        session_dir.join("stdout.log"),
        session_dir.join("stderr.log"),
        session_dir.join("prompt.log"),
    )
}

fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create directory '{}': {e}", parent.display()))?;
    }
    Ok(())
}

fn tail_text_file(path: &Path, tail: usize) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("failed to read log '{}': {e}", path.display()))?;
    let lines = raw.lines().map(|line| line.to_string()).collect::<Vec<_>>();
    let start = lines.len().saturating_sub(tail);
    Ok(lines[start..].to_vec())
}

fn collect_values(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                values.push(value.clone());
                idx += 1;
            }
        }
        idx += 1;
    }
    values
}

fn flag_present(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

fn now_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    secs.to_string()
}

fn print_supervisor_usage() {
    println!("QLANG Supervisor");
    println!();
    println!("  qlang supervisor init --state .qlang/supervisor.json");
    println!("  qlang supervisor add-agent --state .qlang/supervisor.json --name claude --kind claude-code --command claude --arg code");
    println!("  qlang supervisor enqueue --state .qlang/supervisor.json --title \"Inspect parser panic\" --goal \"Find the root cause\" --agent claude");
    println!("  qlang supervisor tick --state .qlang/supervisor.json [--spawn]");
    println!("  qlang supervisor poll --state .qlang/supervisor.json");
    println!("  qlang supervisor cycle --state .qlang/supervisor.json [--spawn]");
    println!("  qlang supervisor daemon --state .qlang/supervisor.json [--spawn] [--interval-ms 2000]");
    println!("  qlang supervisor handover --state .qlang/supervisor.json --task 1 --path handoff.qlms");
    println!("  qlang supervisor logs --state .qlang/supervisor.json --session 1 [--tail 50]");
    println!("  qlang supervisor complete --state .qlang/supervisor.json --task 1 --note \"patch merged\"");
    println!("  qlang supervisor show --state .qlang/supervisor.json");
    println!("  qlang run-agent --state .qlang/supervisor.json --agent claude --task 1 [--spawn]");
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}
