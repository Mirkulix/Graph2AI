use axum::{
    extract::{Path, State},
    Json,
};
use qlang_agent::protocol::{self, AgentId, Capability, MessageIntent};
use qo_agents::{ExecutionGraph, Goal};
use qo_evolution::SystemStats;
use qo_memory::graph_builders;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::AppState;

fn serialize_goal(goal: &Goal) -> Option<String> {
    serde_json::to_string(goal).ok()
}

#[derive(Deserialize)]
pub struct CreateGoalRequest {
    pub description: String,
}

#[derive(Serialize)]
pub struct CreateGoalResponse {
    pub goal: Goal,
    pub message: String,
}

pub async fn list_goals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Goal>>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goals = registry.list_goals().to_vec();
    Ok(Json(goals))
}

pub async fn create_goal(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateGoalRequest>,
) -> Result<Json<CreateGoalResponse>, (axum::http::StatusCode, String)> {
    // Create the goal and get its id
    let goal_id = {
        let mut registry = state.agents.lock().await;
        let goal = registry.create_goal(req.description.clone());
        goal.id
    };

    // Retrieve the newly created goal for the response
    let goal = {
        let registry = state.agents.lock().await;
        registry
            .get_goal(goal_id)
            .cloned()
            .ok_or_else(|| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Goal not found after creation".to_string()))?
    };

    // Log action history
    if let Err(e) = state.store.log_action("goal_created", &req.description, "") {
        tracing::warn!("failed to log goal_created action: {e}");
    }

    // Persist goal to store
    if let Some(json) = serialize_goal(&goal) {
        if let Err(e) = state.store.save_goal(goal.id, &json) {
            tracing::warn!("failed to persist goal {}: {e}", goal.id);
        }
    }

    // Emit activity: goal created
    state.stream.publish_activity(
        format!("Neues Ziel erstellt: {}", req.description),
        None,
        "info",
    );

    // Spawn background task to execute the goal
    let state_clone = state.clone();
    let description = req.description.clone();
    tokio::spawn(async move {
        execute_goal_background(state_clone, goal_id, description).await;
    });

    Ok(Json(CreateGoalResponse {
        goal,
        message: "Ziel erstellt und wird im Hintergrund ausgeführt.".to_string(),
    }))
}

pub async fn execute_goal_background(
    state: Arc<AppState>,
    goal_id: u64,
    description: String,
) {
    let goal_start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    state.stream.publish_activity(
        "CEO analysiert Ziel...",
        Some("CEO".to_string()),
        "progress",
    );

    // Send QLMS message: CEO starts goal decomposition via MessageBus
    {
        let bus = &state.message_bus;
        let ceo_graph = qo_agents::executor::build_ceo_decompose_graph("goal_start");
        let mut inputs = HashMap::new();
        inputs.insert(
            "goal".to_string(),
            qlang_core::tensor::TensorData::from_string(&description),
        );
        let msg = qlang_agent::protocol::GraphMessage {
            id: protocol::next_msg_id(),
            from: AgentId { name: "ceo".into(), capabilities: vec![Capability::Execute] },
            to: AgentId { name: "researcher".into(), capabilities: vec![Capability::Execute] },
            graph: ceo_graph,
            inputs,
            intent: MessageIntent::Execute,
            in_reply_to: None,
            signature: None, signer_pubkey: None, graph_hash: None,
        };
        let status = bus.send(msg).await;
        tracing::info!("MessageBus: CEO → team (goal_start): {:?}", status);
    }

    // We need to execute the goal step by step with activity events.
    // First, do the decomposition phase — pass quantum state for strategy steering.
    let decompose_result = {
        let mut registry = state.agents.lock().await;
        let quantum = state.quantum.lock().await;
        registry.execute_goal_decompose(goal_id, &*state.llm, Some(&*quantum)).await
    };

    let chosen_strategy = match decompose_result {
        Err(e) => {
            state.stream.publish_activity(
                format!("CEO ✗ Fehler bei Dekomposition: {}", e),
                Some("CEO".to_string()),
                "error",
            );
            return;
        }
        Ok((subtask_count, strategy)) => {
            state.stream.publish_activity(
                format!("CEO hat {} Teilaufgaben erstellt [Strategie: {}]", subtask_count, strategy),
                Some("CEO".to_string()),
                "success",
            );
            strategy
        }
    };

    // Execute all subtasks in parallel — releases the lock before spawning
    state.stream.publish_activity(
        "Agenten arbeiten parallel an Teilaufgaben...".to_string(),
        None,
        "progress",
    );

    let parallel_outcomes = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_subtasks_parallel(goal_id, state.llm.clone()).await
    };

    // Publish per-subtask activity events, send QLMS result messages, and build results for graph
    let mut subtask_results: Vec<(String, String, bool, u64)> = Vec::new();
    for (_, agent_name, task_desc, succeeded, duration_ms) in &parallel_outcomes {
        if *succeeded {
            state.stream.publish_activity(
                format!("{} \u{2713} Aufgabe erledigt", agent_name),
                Some(agent_name.clone()),
                "success",
            );
        } else {
            state.stream.publish_activity(
                format!("{} \u{2717} Teilaufgabe fehlgeschlagen", agent_name),
                Some(agent_name.clone()),
                "error",
            );
        }
        // Send QLMS result message: Agent → CEO via MessageBus
        {
            let result_graph = qo_agents::executor::build_agent_task_graph(&agent_name.to_lowercase());
            let mut inputs = HashMap::new();
            let status_text = if *succeeded { "completed" } else { "failed" };
            inputs.insert(
                "task".to_string(),
                qlang_core::tensor::TensorData::from_string(&format!("{}: {}", status_text, task_desc)),
            );
            let msg = qlang_agent::protocol::GraphMessage {
                id: protocol::next_msg_id(),
                from: AgentId { name: agent_name.to_lowercase(), capabilities: vec![Capability::Execute] },
                to: AgentId { name: "ceo".into(), capabilities: vec![Capability::Execute] },
                graph: result_graph,
                inputs,
                intent: MessageIntent::Result { original_message_id: 0 },
                in_reply_to: None,
                signature: None, signer_pubkey: None, graph_hash: None,
            };
            let _ = state.message_bus.send(msg).await;
        }
        subtask_results.push((agent_name.clone(), task_desc.clone(), *succeeded, *duration_ms));
    }

    // CEO summary
    state.stream.publish_activity(
        "CEO fasst Ergebnisse zusammen...",
        Some("CEO".to_string()),
        "progress",
    );

    let summary_result = {
        let mut registry = state.agents.lock().await;
        registry.execute_goal_summarize(goal_id, &*state.llm).await
    };

    match summary_result {
        Ok(summary) => {
            let short = if summary.len() > 80 {
                format!("{}...", &summary[..80])
            } else {
                summary.clone()
            };
            state.stream.publish_activity(
                format!("Ziel abgeschlossen: {}", short),
                Some("CEO".to_string()),
                "success",
            );
            if let Err(e) = state.store.log_action("goal_completed", &description, &short) {
                tracing::warn!("failed to log goal_completed action: {e}");
            }
            // Remember goal and result in long-term memory
            {
                let mut mem = state.memory.lock().await;
                let mem_key = format!("goal_{}", goal_id);
                let memory_text = format!("Ziel: {}\nErgebnis: {}", description, short);
                mem.remember(mem_key.clone(), &memory_text, &state.store);
                let _ = state.store.set(&mem_key, &memory_text);
            }
        }
        Err(e) => {
            state.stream.publish_activity(
                format!("CEO ✗ Fehler bei Zusammenfassung: {}", e),
                Some("CEO".to_string()),
                "error",
            );
            tracing::warn!("Goal {} summary failed: {}", goal_id, e);
            if let Err(log_err) = state.store.log_action("goal_failed", &description, &e.to_string()) {
                tracing::warn!("failed to log goal_failed action: {log_err}");
            }
        }
    }

    // Update agent stats
    {
        let mut registry = state.agents.lock().await;
        registry.finalize_goal(goal_id);
    }

    // Evolve quantum state based on goal outcome
    {
        let goal_succeeded = {
            let registry = state.agents.lock().await;
            registry.get_goal(goal_id)
                .map(|g| g.subtasks.iter().all(|st| st.status == qo_agents::GoalStatus::Completed))
                .unwrap_or(false)
        };
        let strategy_idx = qo_agents::executor::strategy_index(&chosen_strategy);
        let mut quantum = state.quantum.lock().await;
        quantum.evolve(strategy_idx, goal_succeeded, 0.1);
        tracing::info!(
            "Quantum evolved: strategy='{}' idx={} success={} gen={}",
            chosen_strategy, strategy_idx, goal_succeeded, quantum.generation
        );
        // Persist updated quantum state
        if let Ok(json) = serde_json::to_string(&*quantum) {
            if let Err(e) = state.store.save_quantum_state(&json) {
                tracing::warn!("failed to persist quantum state: {e}");
            }
        }
    }

    // Mini-evolution: pattern analysis after goal completion
    {
        let (stats, _cs_energy) = {
            let agents = state.agents.lock().await;
            let cs = state.consciousness.lock().await;
            let agent_list = agents.list_agents();
            let active = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Active).count() as u8;
            let idle = agent_list.iter().filter(|a| a.status == qo_agents::AgentStatus::Idle).count() as u8;
            let completed: u32 = agent_list.iter().map(|a| a.tasks_completed).sum();
            let failed: u32 = agent_list.iter().map(|a| a.tasks_failed).sum();
            (SystemStats {
                total_tasks: completed + failed,
                tasks_completed: completed,
                tasks_failed: failed,
                agents_active: active,
                agents_idle: idle,
                avg_energy: cs.energy,
                completed_streak: 0,
            }, cs.energy)
        };
        let new_patterns = {
            let mut patterns = state.patterns.lock().await;
            patterns.analyze(&stats)
                .iter()
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        };
        if !new_patterns.is_empty() {
            tracing::info!("Post-goal mini-evolution: {} new patterns", new_patterns.len());
        }
    }

    // Persist finalized goal and updated agent stats
    {
        let registry = state.agents.lock().await;
        if let Some(goal) = registry.get_goal(goal_id) {
            if let Some(json) = serialize_goal(goal) {
                if let Err(e) = state.store.save_goal(goal_id, &json) {
                    tracing::warn!("failed to persist finalized goal {}: {e}", goal_id);
                }
            }
        }
        for agent in registry.list_agents() {
            if let Ok(json) = serde_json::to_string(&agent) {
                let role_str = format!("{:?}", agent.role);
                if let Err(e) = state.store.save_agent_stats(&role_str, &json) {
                    tracing::warn!("failed to persist agent stats for {:?}: {e}", agent.role);
                }
            }
        }
    }

    // Build and store QLANG graph for this goal execution — both JSON metadata AND real QLBG binary
    {
        let total_duration_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
            - goal_start_ms;
        let graph = graph_builders::build_goal_graph(&description, &subtask_results, total_duration_ms);
        let store_id = match state.graph_store.store(&graph) {
            Ok(id) => id,
            Err(e) => { tracing::warn!("failed to store goal graph metadata: {e}"); 0 }
        };

        // Build a REAL QLANG graph for this goal and store as QLBG binary
        let mut qlang_graph = qlang_core::graph::Graph::new(format!("goal_{}", goal_id));
        let str_type = qlang_core::tensor::TensorType::new(
            qlang_core::tensor::Dtype::Utf8,
            qlang_core::tensor::Shape::scalar(),
        );
        let input = qlang_graph.add_node(
            qlang_core::ops::Op::Input { name: "goal".into() },
            vec![], vec![str_type.clone()],
        );
        // Add a node per subtask agent
        let mut prev_node = input;
        for (agent_name, _task_desc, succeeded, _dur) in &subtask_results {
            let model = if *succeeded { "completed" } else { "failed" };
            let node = qlang_graph.add_node(
                qlang_core::ops::Op::OllamaChat { model: format!("agent_{agent_name}_{model}") },
                vec![str_type.clone()], vec![str_type.clone()],
            );
            qlang_graph.add_edge(prev_node, 0, node, 0, str_type.clone());
            prev_node = node;
        }
        let output = qlang_graph.add_node(
            qlang_core::ops::Op::Output { name: "result".into() },
            vec![str_type.clone()], vec![],
        );
        qlang_graph.add_edge(prev_node, 0, output, 0, str_type.clone());

        let binary = qlang_core::binary::to_binary(&qlang_graph);
        tracing::info!(
            "Goal #{} QLANG graph: {} nodes, {} edges, {} bytes .qlbg",
            goal_id, qlang_graph.nodes.len(), qlang_graph.edges.len(), binary.len()
        );
        if let Err(e) = state.graph_store.store_binary_graph(store_id, &binary) {
            tracing::warn!("failed to store goal QLBG binary: {e}");
        }
    }

    state.stream.publish_activity(
        format!("Ziel #{} vollständig abgeschlossen", goal_id),
        None,
        "info",
    );

    // Publish updated description for activity
    let _ = description;
}

pub async fn get_goal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<Goal>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goal = registry.get_goal(id).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Goal {} not found", id),
        )
    })?;
    Ok(Json(goal.clone()))
}

pub async fn get_goal_graph(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ExecutionGraph>, (axum::http::StatusCode, String)> {
    let registry = state.agents.lock().await;
    let goal = registry.get_goal(id).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Goal {} not found", id),
        )
    })?;
    let graph = goal.execution_graph.clone().ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            "No execution graph available yet".to_string(),
        )
    })?;
    Ok(Json(graph))
}
