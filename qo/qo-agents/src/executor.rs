use crate::{
    agent::AgentRole,
    goal::{ExecutionGraph, Goal, GoalStatus, GraphEdge, GraphNode, SubTask},
    llm_node,
};
use qo_llm::LlmRouter;
use qo_evolution::QuantumState;
use qo_simulation::{predict, Scenario, Simulator, Strategy};
use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType};
use qlang_agent::bus::MessageBus;
use qlang_agent::protocol::{self, AgentId, Capability, GraphMessage, MessageIntent};
use std::collections::HashMap;
use std::sync::Arc;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================
// Simulation helper — runs before CEO decomposition
// ============================================================

/// Run a fast Monte-Carlo simulation across 4 standard strategies.
/// Returns (recommended_strategy_name, confidence_0_to_1).
pub fn simulate_before_execution(goal_description: &str) -> (String, f32) {
    let mut scenario = Scenario::new(1, goal_description.to_string());

    scenario.add_strategy(Strategy {
        name: "Direkte Ausführung".into(),
        agents_involved: vec!["Researcher".into()],
        steps: vec!["Recherchiere und liefere Ergebnis".into()],
        estimated_cost: 0.0,
        estimated_duration_ms: 500,
    });
    scenario.add_strategy(Strategy {
        name: "Dekomposition + Delegation".into(),
        agents_involved: vec!["CEO".into(), "Researcher".into(), "Developer".into()],
        steps: vec![
            "Zerlege".into(),
            "Recherchiere".into(),
            "Implementiere".into(),
            "Verifiziere".into(),
        ],
        estimated_cost: 0.0,
        estimated_duration_ms: 2000,
    });
    scenario.add_strategy(Strategy {
        name: "Recherche zuerst".into(),
        agents_involved: vec!["Researcher".into(), "Strategist".into()],
        steps: vec!["Recherchiere".into(), "Analysiere und plane".into()],
        estimated_cost: 0.0,
        estimated_duration_ms: 1000,
    });
    scenario.add_strategy(Strategy {
        name: "Kreative Lösung".into(),
        agents_involved: vec!["Artisan".into(), "Researcher".into()],
        steps: vec!["Brainstorme".into(), "Recherchiere".into()],
        estimated_cost: 0.0,
        estimated_duration_ms: 800,
    });

    let mut sim = Simulator::new(30);
    let results = sim.simulate(&scenario);
    let prediction = predict(&scenario, &results);

    (prediction.recommended_name.clone(), prediction.confidence)
}

/// Map a strategy name to the CEO decomposition instruction.
pub fn strategy_instruction(chosen_strategy: &str) -> &'static str {
    match chosen_strategy {
        "Direkte Ausführung" => {
            "Delegiere diese Aufgabe an EINEN Agenten (Researcher). Keine Zerlegung nötig."
        }
        "Recherche zuerst" => {
            "Beginne mit Recherche (Researcher), dann Analyse (Strategist). Maximal 2 Schritte."
        }
        "Kreative Lösung" => {
            "Nutze den Artisan für einen kreativen Ansatz, unterstützt durch Researcher."
        }
        _ => "Zerlege in 2-4 Teilaufgaben und delegiere an passende Agenten.",
    }
}

/// Return strategy index (0-3) for the 4 standard strategies.
pub fn strategy_index(name: &str) -> usize {
    match name {
        "Direkte Ausführung" => 0,
        "Dekomposition + Delegation" => 1,
        "Recherche zuerst" => 2,
        "Kreative Lösung" => 3,
        _ => 1,
    }
}

// ============================================================
// QLANG Graph Builders — produce real Graph structs
// ============================================================

/// Build a CEO decomposition graph (Input → Output, Utf8 scalars).
pub fn build_ceo_decompose_graph(label: &str) -> Graph {
    let mut g = Graph::new(format!("ceo_decompose_{}", label));
    let utf8_scalar = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input = g.add_node(
        Op::Input { name: "goal".into() },
        vec![],
        vec![utf8_scalar.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "plan".into() },
        vec![utf8_scalar.clone()],
        vec![],
    );

    g.add_edge(input, 0, output, 0, utf8_scalar);
    g
}

/// Build an agent task graph (Input → Output, Utf8 scalars).
pub fn build_agent_task_graph(agent_name: &str) -> Graph {
    let mut g = Graph::new(format!("agent_{}", agent_name));
    let utf8_scalar = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input = g.add_node(
        Op::Input { name: "task".into() },
        vec![],
        vec![utf8_scalar.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "result".into() },
        vec![utf8_scalar.clone()],
        vec![],
    );

    g.add_edge(input, 0, output, 0, utf8_scalar);
    g
}

/// Build a GraphMessage wrapping an agent-to-agent task dispatch.
pub fn build_agent_message(
    from_role: &str,
    to_role: &str,
    graph: Graph,
    task_text: &str,
) -> GraphMessage {
    use qlang_core::tensor::TensorData;
    let mut inputs = HashMap::new();
    inputs.insert("task".to_string(), TensorData::from_string(task_text));

    GraphMessage {
        id: now_ms(),
        from: AgentId {
            name: from_role.into(),
            capabilities: vec![Capability::Execute],
        },
        to: AgentId {
            name: to_role.into(),
            capabilities: vec![Capability::Execute],
        },
        graph,
        inputs,
        intent: MessageIntent::Execute,
        in_reply_to: None,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    }
}

// ============================================================
// ExecutionGraphData — real QLANG graph metadata for one run
// ============================================================

/// Tracks the real QLANG graphs built and executed during a goal run.
pub struct ExecutionGraphData {
    /// (name, node_count, binary_size_bytes)
    pub graphs: Vec<(String, usize, usize)>,
    /// (name, duration_ms, success)
    pub executions: Vec<(String, u64, bool)>,
    pub total_binary_bytes: usize,
}

impl ExecutionGraphData {
    pub fn new() -> Self {
        Self {
            graphs: Vec::new(),
            executions: Vec::new(),
            total_binary_bytes: 0,
        }
    }

    pub fn add_graph(&mut self, name: &str, graph: &Graph, binary_size: usize) {
        self.graphs
            .push((name.to_string(), graph.nodes.len(), binary_size));
        self.total_binary_bytes += binary_size;
    }

    pub fn record_execution(&mut self, name: &str, duration_ms: u64, success: bool) {
        self.executions
            .push((name.to_string(), duration_ms, success));
    }
}

impl Default for ExecutionGraphData {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// execute_goal — builds + serializes real QLANG graphs,
// uses QLMS protocol messages, then calls LLM for reasoning.
// ============================================================

/// Execute a goal: CEO decomposes it, each subtask is handled by the
/// assigned agent. Real QLANG graphs are built, serialised to binary
/// (.qlbg), and QLMS GraphMessages are sent between agents via the MessageBus.
pub async fn execute_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (graph_data, chosen_strategy, _succeeded) = execute_goal_qlang(llm, goal, None, None).await?;

    // Log QLANG graph stats
    tracing::info!(
        "QLANG execution complete: strategy='{}', {} graphs ({} bytes total binary), {} executions",
        chosen_strategy,
        graph_data.graphs.len(),
        graph_data.total_binary_bytes,
        graph_data.executions.len(),
    );
    for (name, nodes, bytes) in &graph_data.graphs {
        tracing::debug!("  graph '{}': {} nodes, {} bytes (QLBG)", name, nodes, bytes);
    }

    Ok(())
}

/// Core implementation: builds real QLANG graphs and executes them.
/// Returns (graph_data, chosen_strategy_name, goal_succeeded).
///
/// When a `MessageBus` is provided, all agent-to-agent communication
/// flows through it as real QLMS GraphMessages. Without a bus, messages
/// are built but not routed (legacy behavior).
pub async fn execute_goal_qlang(
    llm: &LlmRouter,
    goal: &mut Goal,
    quantum_state: Option<&QuantumState>,
    bus: Option<Arc<MessageBus>>,
) -> Result<(ExecutionGraphData, String, bool), Box<dyn std::error::Error + Send + Sync>> {
    goal.status = GoalStatus::InProgress;
    let mut graph_data = ExecutionGraphData::new();

    // Build initial display graph (for UI)
    let mut nodes: Vec<GraphNode> = vec![
        GraphNode {
            id: "input".to_string(),
            label: format!(
                "Ziel: {}",
                &goal.description.chars().take(40).collect::<String>()
            ),
            node_type: "input".to_string(),
            agent: None,
            status: "completed".to_string(),
            duration_ms: None,
        },
        GraphNode {
            id: "ceo_decompose".to_string(),
            label: "CEO: Dekomposition".to_string(),
            node_type: "llm".to_string(),
            agent: Some("CEO".to_string()),
            status: "pending".to_string(),
            duration_ms: None,
        },
    ];
    let mut edges: Vec<GraphEdge> = vec![GraphEdge {
        from: "input".to_string(),
        to: "ceo_decompose".to_string(),
        data_type: "goal".to_string(),
    }];

    // --- Step 1: Build + serialise CEO decomposition graph ---
    let ceo_graph = build_ceo_decompose_graph("goal_decompose");
    let ceo_bytes = binary::to_binary(&ceo_graph);
    graph_data.add_graph("ceo_decompose", &ceo_graph, ceo_bytes.len());

    // Send QLMS message CEO → team via MessageBus
    let ceo_msg = build_agent_message("ceo", "researcher", ceo_graph.clone(), &goal.description);
    if let Some(ref bus) = bus {
        let status = bus.send(ceo_msg).await;
        tracing::debug!("CEO broadcast: {:?}", status);
    }

    // --- Step 2a: Run simulation to predict best strategy ---
    let (sim_strategy, sim_confidence) = simulate_before_execution(&goal.description);
    tracing::info!(
        "Simulation recommended '{}' (confidence: {:.0}%)",
        sim_strategy,
        sim_confidence * 100.0
    );

    // --- Step 2b: Get quantum recommendation ---
    let (quantum_strategy, quantum_confidence) = if let Some(ref qs) = quantum_state {
        let confidence = qs.purity() as f32;
        match qs.measure() {
            Some((_, name)) => (name.to_string(), confidence),
            None => ("Dekomposition + Delegation".to_string(), 0.0f32),
        }
    } else {
        ("Dekomposition + Delegation".to_string(), 0.0f32)
    };
    tracing::info!(
        "Quantum recommended '{}' (purity/confidence: {:.2})",
        quantum_strategy,
        quantum_confidence
    );

    // Prefer quantum when it has high purity (confident), else use simulation
    let chosen_strategy = if quantum_confidence > 0.6 {
        quantum_strategy.clone()
    } else {
        sim_strategy.clone()
    };
    tracing::info!("Chosen strategy: '{}'", chosen_strategy);

    // --- Step 2c: CEO reasoning via LLM using chosen strategy ---
    let instruction = strategy_instruction(&chosen_strategy);
    let decomposition_prompt = format!(
        "Strategie: {chosen_strategy} (Konfidenz: {sim_confidence:.0}%)\n\
        Anweisung: {instruction}\n\n\
        Ziel: {}\n\n\
        Verfügbare Agenten: Researcher, Developer, Strategist, Artisan.\n\
        Antworte im Format:\n1. [Agent] Aufgabe",
        goal.description
    );
    let t0 = now_ms();
    let decomposition =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Goal-Dekomposition", &decomposition_prompt)
            .await?;
    let decompose_ms = now_ms() - t0;
    graph_data.record_execution("ceo_decompose", decompose_ms, true);

    // Mark CEO node done
    if let Some(n) = nodes.iter_mut().find(|n| n.id == "ceo_decompose") {
        n.status = "completed".to_string();
        n.duration_ms = Some(decompose_ms);
    }

    // --- Step 3: Parse subtasks ---
    goal.subtasks = parse_subtasks(&decomposition);
    if goal.subtasks.is_empty() {
        goal.subtasks.push(SubTask {
            description: goal.description.clone(),
            assigned_to: AgentRole::Researcher,
            status: GoalStatus::Pending,
            result: None,
        });
    }

    let subtask_count = goal.subtasks.len();

    // Add subtask nodes to display graph
    for i in 0..subtask_count {
        let agent_name = goal.subtasks[i].assigned_to.name().to_string();
        let desc = goal.subtasks[i]
            .description
            .chars()
            .take(40)
            .collect::<String>();
        let node_id = format!("subtask_{}", i);
        nodes.push(GraphNode {
            id: node_id.clone(),
            label: format!("{}: {}", agent_name, desc),
            node_type: "llm".to_string(),
            agent: Some(agent_name),
            status: "pending".to_string(),
            duration_ms: None,
        });
        edges.push(GraphEdge {
            from: "ceo_decompose".to_string(),
            to: node_id,
            data_type: "subtask".to_string(),
        });
    }

    // Summary + output nodes
    nodes.push(GraphNode {
        id: "ceo_summary".to_string(),
        label: "CEO: Zusammenfassung".to_string(),
        node_type: "llm".to_string(),
        agent: Some("CEO".to_string()),
        status: "pending".to_string(),
        duration_ms: None,
    });
    nodes.push(GraphNode {
        id: "output".to_string(),
        label: "Ergebnis".to_string(),
        node_type: "output".to_string(),
        agent: None,
        status: "pending".to_string(),
        duration_ms: None,
    });
    edges.push(GraphEdge {
        from: "ceo_summary".to_string(),
        to: "output".to_string(),
        data_type: "result".to_string(),
    });

    // --- Step 4: Build + execute real QLANG graphs for each subtask ---
    for i in 0..subtask_count {
        let agent_role = goal.subtasks[i].assigned_to;
        let agent_name_lower = agent_role.name().to_lowercase();
        let graph_name = format!("agent_{}", agent_name_lower);

        let agent_graph = build_agent_task_graph(&agent_name_lower);
        let agent_bytes = binary::to_binary(&agent_graph);
        graph_data.add_graph(&graph_name, &agent_graph, agent_bytes.len());

        // QLMS: CEO dispatches task to agent via GraphMessage through MessageBus
        let task_text = goal.subtasks[i].description.clone();
        let dispatch_msg = build_agent_message("ceo", &agent_name_lower, agent_graph, &task_text);
        if let Some(ref bus) = bus {
            let status = bus.send(dispatch_msg).await;
            tracing::debug!("CEO → {}: {:?}", agent_name_lower, status);
        }

        goal.subtasks[i].status = GoalStatus::InProgress;
        let goal_desc = goal.description.clone();

        let values = qo_values::ValueScores::default();
        let t_sub = now_ms();
        match llm_node::agent_execute_with_tools(llm, agent_role, &goal_desc, &task_text, &values).await {
            Ok(result) => {
                let sub_ms = now_ms() - t_sub;
                goal.subtasks[i].result = Some(result.clone());
                goal.subtasks[i].status = GoalStatus::Completed;
                graph_data.record_execution(&graph_name, sub_ms, true);

                // Send result back to CEO via MessageBus
                if let Some(ref bus) = bus {
                    let result_graph = build_agent_task_graph(&agent_name_lower);
                    let mut result_inputs = HashMap::new();
                    result_inputs.insert("task".to_string(), qlang_core::tensor::TensorData::from_string(&result));
                    let reply = GraphMessage {
                        id: protocol::next_msg_id(),
                        from: AgentId { name: agent_name_lower.clone(), capabilities: vec![Capability::Execute] },
                        to: AgentId { name: "ceo".into(), capabilities: vec![Capability::Execute] },
                        graph: result_graph,
                        inputs: result_inputs,
                        intent: MessageIntent::Result { original_message_id: 0 },
                        in_reply_to: None,
                        signature: None,
                        signer_pubkey: None,
                        graph_hash: None,
                    };
                    bus.send(reply).await;
                }

                let node_id = format!("subtask_{}", i);
                if let Some(n) = nodes.iter_mut().find(|n| n.id == node_id) {
                    n.status = "completed".to_string();
                    n.duration_ms = Some(sub_ms);
                }
            }
            Err(e) => {
                let sub_ms = now_ms() - t_sub;
                goal.subtasks[i].result = Some(format!("Fehler: {e}"));
                goal.subtasks[i].status = GoalStatus::Failed;
                graph_data.record_execution(&graph_name, sub_ms, false);

                let node_id = format!("subtask_{}", i);
                if let Some(n) = nodes.iter_mut().find(|n| n.id == node_id) {
                    n.status = "failed".to_string();
                    n.duration_ms = Some(sub_ms);
                }
            }
        }

        edges.push(GraphEdge {
            from: format!("subtask_{}", i),
            to: "ceo_summary".to_string(),
            data_type: "result".to_string(),
        });
    }

    // --- Step 5: CEO summary graph ---
    let summary_graph = build_ceo_decompose_graph("summary");
    let summary_bytes = binary::to_binary(&summary_graph);
    graph_data.add_graph("ceo_summary", &summary_graph, summary_bytes.len());

    let all_results: Vec<String> = goal
        .subtasks
        .iter()
        .filter_map(|st| {
            st.result
                .as_ref()
                .map(|r| format!("{}: {}", st.assigned_to.name(), r))
        })
        .collect();

    let summary_prompt = format!(
        "Fasse die Ergebnisse der Teilaufgaben zusammen und gib ein Gesamtergebnis.\n\n\
        Ziel: {}\n\n\
        Ergebnisse:\n{}",
        goal.description,
        all_results.join("\n\n")
    );

    let t_sum = now_ms();
    let summary =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Zusammenfassung", &summary_prompt).await?;
    let sum_ms = now_ms() - t_sum;
    graph_data.record_execution("ceo_summary", sum_ms, true);

    if let Some(n) = nodes.iter_mut().find(|n| n.id == "ceo_summary") {
        n.status = "completed".to_string();
        n.duration_ms = Some(sum_ms);
    }
    if let Some(n) = nodes.iter_mut().find(|n| n.id == "output") {
        n.status = "completed".to_string();
    }

    goal.result = Some(summary);
    goal.status = GoalStatus::Completed;
    goal.execution_graph = Some(ExecutionGraph { nodes, edges });

    let goal_succeeded = goal.subtasks.iter().all(|st| st.status == GoalStatus::Completed);
    Ok((graph_data, chosen_strategy, goal_succeeded))
}

// ============================================================
// Legacy entry points (kept for backward compat)
// ============================================================

/// Decomposes a goal using simulation + optional quantum steering.
/// Returns the chosen strategy name.
pub async fn decompose_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
    quantum_state: Option<&QuantumState>,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    goal.status = GoalStatus::InProgress;

    // Run simulation to predict best strategy
    let (sim_strategy, sim_confidence) = simulate_before_execution(&goal.description);
    tracing::info!(
        "Simulation recommended '{}' (confidence: {:.0}%)",
        sim_strategy,
        sim_confidence * 100.0
    );

    // Get quantum recommendation
    let (quantum_strategy, quantum_confidence) = if let Some(ref qs) = quantum_state {
        let confidence = qs.purity() as f32;
        match qs.measure() {
            Some((_, name)) => (name.to_string(), confidence),
            None => ("Dekomposition + Delegation".to_string(), 0.0f32),
        }
    } else {
        ("Dekomposition + Delegation".to_string(), 0.0f32)
    };
    tracing::info!(
        "Quantum recommended '{}' (purity/confidence: {:.2})",
        quantum_strategy,
        quantum_confidence
    );

    let chosen_strategy = if quantum_confidence > 0.6 {
        quantum_strategy
    } else {
        sim_strategy
    };
    tracing::info!("Chosen strategy for decomposition: '{}'", chosen_strategy);

    let instruction = strategy_instruction(&chosen_strategy);
    let decomposition_prompt = format!(
        "Strategie: {chosen_strategy} (Konfidenz: {sim_confidence:.0}%)\n\
        Anweisung: {instruction}\n\n\
        Ziel: {}\n\n\
        Verfügbare Agenten: Researcher, Developer, Strategist, Artisan.\n\
        Antworte im Format:\n1. [Agent] Aufgabe",
        goal.description
    );

    let decomposition = llm_node::llm_reason(
        llm,
        AgentRole::Ceo,
        "Goal-Dekomposition",
        &decomposition_prompt,
    )
    .await?;

    goal.subtasks = parse_subtasks(&decomposition);

    if goal.subtasks.is_empty() {
        goal.subtasks.push(SubTask {
            description: goal.description.clone(),
            assigned_to: AgentRole::Researcher,
            status: GoalStatus::Pending,
            result: None,
        });
    }

    Ok(chosen_strategy)
}

pub async fn execute_subtask(
    llm: &LlmRouter,
    goal: &mut Goal,
    index: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let subtask = goal
        .subtasks
        .get_mut(index)
        .ok_or("Subtask index out of bounds")?;
    subtask.status = GoalStatus::InProgress;

    let goal_desc = goal.description.clone();
    let subtask_desc = goal.subtasks[index].description.clone();
    let assigned_to = goal.subtasks[index].assigned_to;

    let values = qo_values::ValueScores::default();
    match llm_node::agent_execute_with_tools(llm, assigned_to, &goal_desc, &subtask_desc, &values).await {
        Ok(result) => {
            goal.subtasks[index].result = Some(result.clone());
            goal.subtasks[index].status = GoalStatus::Completed;
            Ok(result)
        }
        Err(e) => {
            let err_msg = format!("Fehler: {e}");
            goal.subtasks[index].result = Some(err_msg.clone());
            goal.subtasks[index].status = GoalStatus::Failed;
            Err(e)
        }
    }
}

pub async fn summarize_goal(
    llm: &LlmRouter,
    goal: &mut Goal,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let all_results: Vec<String> = goal
        .subtasks
        .iter()
        .filter_map(|s| s.result.as_ref().map(|r| format!("{}: {}", s.assigned_to.name(), r)))
        .collect();

    let summary_prompt = format!(
        "Fasse die Ergebnisse der Teilaufgaben zusammen und gib ein Gesamtergebnis.\n\n\
        Ziel: {}\n\n\
        Ergebnisse:\n{}",
        goal.description,
        all_results.join("\n\n")
    );

    let summary =
        llm_node::llm_reason(llm, AgentRole::Ceo, "Zusammenfassung", &summary_prompt).await?;
    goal.result = Some(summary.clone());
    goal.status = GoalStatus::Completed;

    Ok(summary)
}

// ============================================================
// Subtask parser (unchanged)
// ============================================================

fn parse_subtasks(response: &str) -> Vec<SubTask> {
    let mut subtasks = Vec::new();
    for line in response.lines() {
        let line = line.trim();
        let (role, desc) = if let Some(rest) =
            line.strip_prefix(|c: char| c.is_ascii_digit() || c == '-' || c == '*')
        {
            let rest = rest.trim_start_matches(['.', ')', ' ']);
            parse_role_and_desc(rest)
        } else {
            parse_role_and_desc(line)
        };

        if let Some(role) = role {
            if !desc.is_empty() {
                subtasks.push(SubTask {
                    description: desc,
                    assigned_to: role,
                    status: GoalStatus::Pending,
                    result: None,
                });
            }
        }
    }
    subtasks
}

fn parse_role_and_desc(s: &str) -> (Option<AgentRole>, String) {
    let s = s.trim();
    if s.starts_with('[') {
        if let Some(end) = s.find(']') {
            let role_str = &s[1..end];
            let desc = s[end + 1..].trim().to_string();
            return (match_role(role_str), desc);
        }
    }
    for role in AgentRole::ALL {
        let name = role.name();
        if s.starts_with(name) {
            let rest = s[name.len()..].trim_start_matches([':', '-', ' ']);
            return (Some(role), rest.trim().to_string());
        }
    }
    (None, s.to_string())
}

fn match_role(s: &str) -> Option<AgentRole> {
    let s_lower = s.to_lowercase();
    match s_lower.as_str() {
        "ceo" => Some(AgentRole::Ceo),
        "researcher" | "recherche" | "analyse" => Some(AgentRole::Researcher),
        "developer" | "entwickler" | "code" | "technik" => Some(AgentRole::Developer),
        "guardian" | "wächter" | "werte" => Some(AgentRole::Guardian),
        "strategist" | "strategie" | "planung" => Some(AgentRole::Strategist),
        "artisan" | "kreativ" | "design" => Some(AgentRole::Artisan),
        _ => None,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::binary;

    // --- Original parser tests (all preserved) ---

    #[test]
    fn parse_subtasks_bracket_format() {
        let input =
            "1. [Researcher] Analysiere den Markt\n2. [Developer] Baue den Prototyp\n3. [Strategist] Erstelle die Roadmap";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 3);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Developer);
        assert_eq!(tasks[2].assigned_to, AgentRole::Strategist);
    }

    #[test]
    fn parse_subtasks_colon_format() {
        let input = "- Researcher: Recherchiere Trends\n- Artisan: Erstelle Design";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Artisan);
    }

    #[test]
    fn parse_subtasks_german_names() {
        let input = "1. [Recherche] Finde Informationen\n2. [Kreativ] Gestalte Logo";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Artisan);
    }

    #[test]
    fn test_parse_empty_response() {
        let tasks = parse_subtasks("");
        assert_eq!(tasks.len(), 0);
    }

    #[test]
    fn test_parse_mixed_formats() {
        let input = "1. [Researcher] Analysiere den Markt\n- Developer: Baue den Prototyp";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Developer);
    }

    #[test]
    fn test_parse_numbered_without_brackets() {
        let input = "1. Researcher: do X\n2. Strategist: do Y";
        let tasks = parse_subtasks(input);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].assigned_to, AgentRole::Researcher);
        assert_eq!(tasks[1].assigned_to, AgentRole::Strategist);
    }

    #[test]
    fn parse_subtasks_handles_real_ceo_output() {
        let response = "Hier ist mein Plan:\n\n\
            1. [Researcher] Recherchiere aktuelle Trends im KI-Markt\n\
            2. [Developer] Erstelle einen Prototyp der API\n\
            3. [Strategist] Entwickle eine Go-to-Market Strategie\n\n\
            Diese drei Schritte sollten das Ziel erreichen.";
        let tasks = parse_subtasks(response);
        assert_eq!(tasks.len(), 3);
        assert!(tasks[0].description.contains("Recherchiere"));
        assert!(tasks[1].description.contains("Prototyp"));
        assert!(tasks[2].description.contains("Strategie"));
    }

    // --- New QLANG graph tests ---

    #[test]
    fn builds_real_qlang_graph() {
        let g = build_ceo_decompose_graph("Test goal");

        // Must have input + output nodes
        assert!(g.nodes.len() >= 2, "graph must have at least 2 nodes");
        assert!(!g.edges.is_empty(), "graph must have at least one edge");

        // Verify binary serialisation
        let bytes = binary::to_binary(&g);
        assert!(!bytes.is_empty(), "binary must not be empty");

        // Check QLBG magic bytes
        assert_eq!(
            &bytes[0..4],
            &[0x51, 0x4C, 0x42, 0x47],
            "binary must start with QLBG magic"
        );

        // Round-trip
        let restored = binary::from_binary(&bytes).expect("must deserialise");
        assert_eq!(
            restored.nodes.len(),
            g.nodes.len(),
            "node count must survive round-trip"
        );
        assert_eq!(
            restored.edges.len(),
            g.edges.len(),
            "edge count must survive round-trip"
        );

        println!(
            "QLBG binary: {} bytes, {} nodes, {} edges",
            bytes.len(),
            g.nodes.len(),
            g.edges.len()
        );
    }

    #[test]
    fn builds_real_agent_task_graph() {
        let g = build_agent_task_graph("researcher");
        assert!(g.nodes.len() >= 2);
        assert!(!g.edges.is_empty());

        let bytes = binary::to_binary(&g);
        assert_eq!(&bytes[0..4], &[0x51, 0x4C, 0x42, 0x47]);
        println!("agent graph binary: {} bytes", bytes.len());
    }

    #[test]
    fn agent_protocol_message() {
        let g = build_agent_task_graph("researcher");
        let msg = build_agent_message("ceo", "researcher", g, "Research AI trends");

        assert_eq!(msg.from.name, "ceo");
        assert_eq!(msg.to.name, "researcher");
        assert!(matches!(msg.intent, MessageIntent::Execute));
        assert!(msg.inputs.contains_key("task"));
        assert_eq!(
            msg.inputs["task"].as_string().unwrap(),
            "Research AI trends"
        );

        println!(
            "GraphMessage: {} → {}, intent={:?}",
            msg.from.name, msg.to.name, msg.intent
        );
    }

    #[test]
    fn execution_graph_data_tracks_correctly() {
        let mut data = ExecutionGraphData::new();
        let g1 = build_ceo_decompose_graph("decompose");
        let b1 = binary::to_binary(&g1);
        let b1_len = b1.len();
        data.add_graph("ceo_decompose", &g1, b1_len);

        let g2 = build_agent_task_graph("researcher");
        let b2 = binary::to_binary(&g2);
        let b2_len = b2.len();
        data.add_graph("agent_researcher", &g2, b2_len);

        data.record_execution("ceo_decompose", 120, true);
        data.record_execution("agent_researcher", 340, true);

        assert_eq!(data.graphs.len(), 2);
        assert_eq!(data.executions.len(), 2);
        assert_eq!(data.total_binary_bytes, b1_len + b2_len);

        println!(
            "Total binary: {} bytes across {} graphs",
            data.total_binary_bytes,
            data.graphs.len()
        );
    }

    // --- Simulation + strategy tests ---

    #[test]
    fn simulation_runs_before_execution() {
        let (strategy, confidence) = simulate_before_execution("Recherchiere die Vorteile von Rust");
        assert!(!strategy.is_empty(), "strategy name must not be empty");
        assert!(confidence > 0.0 && confidence <= 1.0, "confidence must be in (0, 1]");
    }

    #[test]
    fn strategy_instruction_maps_correctly() {
        let strategies = [
            "Direkte Ausführung",
            "Recherche zuerst",
            "Kreative Lösung",
            "Dekomposition + Delegation",
        ];
        let instructions: Vec<&str> = strategies
            .iter()
            .map(|s| strategy_instruction(s))
            .collect();
        // All must be non-empty
        for instr in &instructions {
            assert!(!instr.is_empty());
        }
        // "Direkte Ausführung" must mention EINEN
        assert!(instructions[0].contains("EINEN"));
        // "Recherche zuerst" must mention Recherche
        assert!(instructions[1].contains("Recherche"));
        // "Kreative Lösung" must mention Artisan
        assert!(instructions[2].contains("Artisan"));
        // Default must mention Zerlege
        assert!(instructions[3].contains("Zerlege"));
    }

    #[test]
    fn strategy_index_maps_correctly() {
        assert_eq!(strategy_index("Direkte Ausführung"), 0);
        assert_eq!(strategy_index("Dekomposition + Delegation"), 1);
        assert_eq!(strategy_index("Recherche zuerst"), 2);
        assert_eq!(strategy_index("Kreative Lösung"), 3);
        assert_eq!(strategy_index("unknown"), 1); // default fallback
    }
}
