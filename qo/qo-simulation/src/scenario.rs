use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub id: u64,
    pub description: String,
    pub strategies: Vec<Strategy>,
    pub constraints: Vec<String>,
    pub simulations_run: u32,
    pub best_strategy: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub name: String,
    pub agents_involved: Vec<String>,  // agent roles
    pub steps: Vec<String>,
    pub estimated_cost: f64,
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub strategy_index: usize,
    pub success: bool,
    pub score: f32,          // 0.0 - 1.0
    pub duration_ms: u64,
    pub cost: f64,
    pub value_alignment: f32, // how well aligned with 5 values
    pub risks: Vec<String>,
    pub benefits: Vec<String>,
}

impl Scenario {
    pub fn new(id: u64, description: String) -> Self {
        Self {
            id,
            description,
            strategies: Vec::new(),
            constraints: Vec::new(),
            simulations_run: 0,
            best_strategy: None,
        }
    }

    pub fn add_strategy(&mut self, strategy: Strategy) {
        self.strategies.push(strategy);
    }
}
