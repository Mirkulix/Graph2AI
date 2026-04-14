use axum::{extract::State, Json};
use qo_simulation::{predict, Prediction, Scenario, Simulator, Strategy};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct RunSimulationRequest {
    pub description: String,
    #[serde(default = "default_num_simulations")]
    pub num_simulations: u32,
}

fn default_num_simulations() -> u32 {
    20
}

#[derive(Debug, Serialize)]
pub struct StrategyInfo {
    pub name: String,
    pub agents: Vec<String>,
    pub steps: u32,
    pub estimated_cost: f64,
    pub estimated_duration_ms: u64,
}

pub async fn run_simulation(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<RunSimulationRequest>,
) -> Json<Prediction> {
    let mut scenario = Scenario::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        req.description,
    );

    scenario.add_strategy(Strategy {
        name: "Direkte Ausführung".into(),
        agents_involved: vec!["Researcher".into()],
        steps: vec!["Direkt ausführen".into()],
        estimated_cost: 0.0,
        estimated_duration_ms: 500,
    });

    scenario.add_strategy(Strategy {
        name: "Dekomposition + Delegation".into(),
        agents_involved: vec!["CEO".into(), "Researcher".into(), "Developer".into()],
        steps: vec![
            "Ziel analysieren".into(),
            "Aufgaben delegieren".into(),
            "Ausführen".into(),
            "Ergebnis prüfen".into(),
        ],
        estimated_cost: 0.01,
        estimated_duration_ms: 3000,
    });

    scenario.add_strategy(Strategy {
        name: "Recherche zuerst".into(),
        agents_involved: vec!["Researcher".into(), "Strategist".into()],
        steps: vec!["Recherchieren".into(), "Strategie entwickeln".into()],
        estimated_cost: 0.003,
        estimated_duration_ms: 1500,
    });

    scenario.add_strategy(Strategy {
        name: "Kreative Lösung".into(),
        agents_involved: vec!["Artisan".into(), "Researcher".into()],
        steps: vec!["Kreativ brainstormen".into(), "Umsetzen".into()],
        estimated_cost: 0.005,
        estimated_duration_ms: 2000,
    });

    let mut simulator = Simulator::new(req.num_simulations);
    let results = simulator.simulate(&scenario);
    let prediction = predict(&scenario, &results);

    Json(prediction)
}

pub async fn list_strategies() -> Json<Vec<StrategyInfo>> {
    Json(vec![
        StrategyInfo {
            name: "Direkte Ausführung".into(),
            agents: vec!["Researcher".into()],
            steps: 1,
            estimated_cost: 0.0,
            estimated_duration_ms: 500,
        },
        StrategyInfo {
            name: "Dekomposition + Delegation".into(),
            agents: vec!["CEO".into(), "Researcher".into(), "Developer".into()],
            steps: 4,
            estimated_cost: 0.01,
            estimated_duration_ms: 3000,
        },
        StrategyInfo {
            name: "Recherche zuerst".into(),
            agents: vec!["Researcher".into(), "Strategist".into()],
            steps: 2,
            estimated_cost: 0.003,
            estimated_duration_ms: 1500,
        },
        StrategyInfo {
            name: "Kreative Lösung".into(),
            agents: vec!["Artisan".into(), "Researcher".into()],
            steps: 2,
            estimated_cost: 0.005,
            estimated_duration_ms: 2000,
        },
    ])
}
