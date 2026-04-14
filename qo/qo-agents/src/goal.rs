use serde::{Deserialize, Serialize};
use crate::agent::AgentRole;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub description: String,
    pub assigned_to: AgentRole,
    pub status: GoalStatus,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub agent: Option<String>,
    pub status: String,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: u64,
    pub description: String,
    pub status: GoalStatus,
    pub subtasks: Vec<SubTask>,
    pub result: Option<String>,
    pub created_at: u64,
    pub execution_graph: Option<ExecutionGraph>,
}

impl Goal {
    pub fn new(id: u64, description: String) -> Self {
        Self {
            id,
            description,
            status: GoalStatus::Pending,
            subtasks: Vec::new(),
            result: None,
            created_at: now_secs(),
            execution_graph: None,
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
