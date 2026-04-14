use crate::graph_store::*;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Build a graph for a chat interaction
pub fn build_chat_graph(
    message: &str,
    response: &str,
    tier: &str,
    duration_ms: u64,
) -> StoredGraph {
    let title_len = message.len().min(50);
    StoredGraph {
        id: 0, // assigned by store
        timestamp: now_secs(),
        graph_type: GraphType::Chat,
        title: format!("Chat: {}", &message[..title_len]),
        nodes: vec![
            QlangNode {
                id: "input".into(),
                op: "input".into(),
                node_type: NodeType::Input,
                label: "User Input".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: None,
                input_type: None,
                output_type: Some("String".into()),
            },
            QlangNode {
                id: "values_check".into(),
                op: "values_check".into(),
                node_type: NodeType::Values,
                label: "Werte-Check".into(),
                agent: Some("Guardian".into()),
                status: NodeStatus::Completed,
                duration_ms: Some(0),
                input_type: Some("String".into()),
                output_type: Some("bool".into()),
            },
            QlangNode {
                id: "llm_route".into(),
                op: "complexity_score".into(),
                node_type: NodeType::Deterministic,
                label: "LLM Router".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: Some(0),
                input_type: Some("String".into()),
                output_type: Some("Tier".into()),
            },
            QlangNode {
                id: "llm_call".into(),
                op: "llm_reason".into(),
                node_type: NodeType::Llm,
                label: format!("LLM ({})", tier),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: Some(duration_ms),
                input_type: Some("Vec<Message>".into()),
                output_type: Some("String".into()),
            },
            QlangNode {
                id: "output".into(),
                op: "output".into(),
                node_type: NodeType::Output,
                label: "Response".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: None,
                input_type: Some("String".into()),
                output_type: None,
            },
        ],
        edges: vec![
            QlangEdge {
                from: "input".into(),
                to: "values_check".into(),
                data_type: "text".into(),
            },
            QlangEdge {
                from: "values_check".into(),
                to: "llm_route".into(),
                data_type: "text".into(),
            },
            QlangEdge {
                from: "llm_route".into(),
                to: "llm_call".into(),
                data_type: "text".into(),
            },
            QlangEdge {
                from: "llm_call".into(),
                to: "output".into(),
                data_type: "text".into(),
            },
        ],
        metadata: GraphMetadata {
            total_duration_ms: Some(duration_ms),
            llm_tier: Some(tier.into()),
            tokens_estimated: Some(((message.len() + response.len()) / 4) as u64),
            cost_usd: None,
        },
    }
}

/// Build a graph for a goal execution with subtasks.
/// subtasks: (agent, description, success, duration_ms)
pub fn build_goal_graph(
    description: &str,
    subtasks: &[(String, String, bool, u64)],
    total_duration_ms: u64,
) -> StoredGraph {
    let title_len = description.len().min(40);
    let mut nodes = vec![
        QlangNode {
            id: "input".into(),
            op: "input".into(),
            node_type: NodeType::Input,
            label: format!("Ziel: {}", &description[..title_len]),
            agent: None,
            status: NodeStatus::Completed,
            duration_ms: None,
            input_type: None,
            output_type: Some("String".into()),
        },
        QlangNode {
            id: "memory_search".into(),
            op: "hnsw_search".into(),
            node_type: NodeType::Memory,
            label: "Memory Search".into(),
            agent: None,
            status: NodeStatus::Completed,
            duration_ms: Some(1),
            input_type: Some("String".into()),
            output_type: Some("Vec<Memory>".into()),
        },
        QlangNode {
            id: "ceo_decompose".into(),
            op: "llm_reason".into(),
            node_type: NodeType::Llm,
            label: "CEO: Dekomposition".into(),
            agent: Some("CEO".into()),
            status: NodeStatus::Completed,
            duration_ms: None,
            input_type: Some("String".into()),
            output_type: Some("Vec<SubTask>".into()),
        },
    ];

    let mut edges = vec![
        QlangEdge {
            from: "input".into(),
            to: "memory_search".into(),
            data_type: "text".into(),
        },
        QlangEdge {
            from: "memory_search".into(),
            to: "ceo_decompose".into(),
            data_type: "context".into(),
        },
    ];

    for (i, (agent, desc, success, dur)) in subtasks.iter().enumerate() {
        let id = format!("subtask_{}", i);
        let label_len = desc.len().min(30);
        nodes.push(QlangNode {
            id: id.clone(),
            op: "llm_reason".into(),
            node_type: NodeType::Llm,
            label: format!("{}: {}", agent, &desc[..label_len]),
            agent: Some(agent.clone()),
            status: if *success {
                NodeStatus::Completed
            } else {
                NodeStatus::Failed
            },
            duration_ms: Some(*dur),
            input_type: Some("SubTask".into()),
            output_type: Some("String".into()),
        });
        edges.push(QlangEdge {
            from: "ceo_decompose".into(),
            to: id.clone(),
            data_type: "subtask".into(),
        });
        edges.push(QlangEdge {
            from: id,
            to: "ceo_summary".into(),
            data_type: "result".into(),
        });
    }

    nodes.push(QlangNode {
        id: "ceo_summary".into(),
        op: "llm_reason".into(),
        node_type: NodeType::Llm,
        label: "CEO: Zusammenfassung".into(),
        agent: Some("CEO".into()),
        status: NodeStatus::Completed,
        duration_ms: None,
        input_type: Some("Vec<Result>".into()),
        output_type: Some("String".into()),
    });
    nodes.push(QlangNode {
        id: "output".into(),
        op: "output".into(),
        node_type: NodeType::Output,
        label: "Ergebnis".into(),
        agent: None,
        status: NodeStatus::Completed,
        duration_ms: None,
        input_type: Some("String".into()),
        output_type: None,
    });
    edges.push(QlangEdge {
        from: "ceo_summary".into(),
        to: "output".into(),
        data_type: "result".into(),
    });

    let title_len2 = description.len().min(50);
    StoredGraph {
        id: 0,
        timestamp: now_secs(),
        graph_type: GraphType::GoalExecution,
        title: format!("Goal: {}", &description[..title_len2]),
        nodes,
        edges,
        metadata: GraphMetadata {
            total_duration_ms: Some(total_duration_ms),
            llm_tier: Some("groq".into()),
            tokens_estimated: None,
            cost_usd: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_graph_has_correct_structure() {
        let g = build_chat_graph("hello", "world", "groq", 100);
        assert_eq!(g.nodes.len(), 5);
        assert_eq!(g.edges.len(), 4);
        assert_eq!(g.graph_type as u8, GraphType::Chat as u8);
        assert!(g.metadata.total_duration_ms.is_some());
    }

    #[test]
    fn goal_graph_has_correct_structure() {
        let subtasks = vec![
            ("Researcher".into(), "Do research".into(), true, 500),
            ("Developer".into(), "Build it".into(), false, 300),
        ];
        let g = build_goal_graph("Test goal", &subtasks, 1000);
        assert!(g.nodes.len() >= 5); // input + memory + ceo + 2 subtasks + summary + output
        assert!(!g.edges.is_empty());
        // Check subtask nodes exist
        assert!(g.nodes.iter().any(|n| n.id == "subtask_0"));
        assert!(g.nodes.iter().any(|n| n.id == "subtask_1"));
    }

    #[test]
    fn evolution_graph_structure() {
        let g = build_evolution_graph(3, 2);
        assert_eq!(g.nodes.len(), 5);
        assert!(g.title.contains("3 Patterns"));
    }
}

/// Build a graph for an evolution analysis cycle
pub fn build_evolution_graph(patterns_found: usize, proposals_created: usize) -> StoredGraph {
    StoredGraph {
        id: 0,
        timestamp: now_secs(),
        graph_type: GraphType::Evolution,
        title: format!(
            "Evolution: {} Patterns, {} Proposals",
            patterns_found, proposals_created
        ),
        nodes: vec![
            QlangNode {
                id: "input".into(),
                op: "input".into(),
                node_type: NodeType::Input,
                label: "System Stats".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: None,
                input_type: None,
                output_type: Some("SystemStats".into()),
            },
            QlangNode {
                id: "analyze".into(),
                op: "pattern_detect".into(),
                node_type: NodeType::Deterministic,
                label: "Pattern-Analyse".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: Some(1),
                input_type: Some("SystemStats".into()),
                output_type: Some("Vec<Pattern>".into()),
            },
            QlangNode {
                id: "propose".into(),
                op: "proposal_generate".into(),
                node_type: NodeType::Deterministic,
                label: "Proposal-Engine".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: Some(0),
                input_type: Some("Vec<Pattern>".into()),
                output_type: Some("Vec<Proposal>".into()),
            },
            QlangNode {
                id: "quantum".into(),
                op: "quantum_evolve".into(),
                node_type: NodeType::Deterministic,
                label: "Quantum State Update".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: Some(0),
                input_type: Some("Vec<Pattern>".into()),
                output_type: Some("QuantumState".into()),
            },
            QlangNode {
                id: "output".into(),
                op: "output".into(),
                node_type: NodeType::Output,
                label: "Evolution Result".into(),
                agent: None,
                status: NodeStatus::Completed,
                duration_ms: None,
                input_type: Some("EvolutionResult".into()),
                output_type: None,
            },
        ],
        edges: vec![
            QlangEdge {
                from: "input".into(),
                to: "analyze".into(),
                data_type: "stats".into(),
            },
            QlangEdge {
                from: "analyze".into(),
                to: "propose".into(),
                data_type: "patterns".into(),
            },
            QlangEdge {
                from: "analyze".into(),
                to: "quantum".into(),
                data_type: "patterns".into(),
            },
            QlangEdge {
                from: "propose".into(),
                to: "output".into(),
                data_type: "proposals".into(),
            },
            QlangEdge {
                from: "quantum".into(),
                to: "output".into(),
                data_type: "quantum_state".into(),
            },
        ],
        metadata: GraphMetadata {
            total_duration_ms: Some(1),
            llm_tier: None,
            tokens_estimated: None,
            cost_usd: None,
        },
    }
}
