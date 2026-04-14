use crate::scenario::{Scenario, SimulationResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub scenario_id: u64,
    pub recommended_strategy: usize,
    pub recommended_name: String,
    pub confidence: f32,
    pub strategy_scores: Vec<StrategyScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyScore {
    pub strategy_index: usize,
    pub name: String,
    pub avg_score: f32,
    pub success_rate: f32,
    pub avg_value_alignment: f32,
    pub avg_duration_ms: u64,
    pub avg_cost: f64,
    pub top_risks: Vec<String>,
    pub top_benefits: Vec<String>,
}

pub fn predict(scenario: &Scenario, results: &[Vec<SimulationResult>]) -> Prediction {
    let mut strategy_scores = Vec::new();

    for (idx, strategy_results) in results.iter().enumerate() {
        let n = strategy_results.len() as f32;
        if n == 0.0 { continue; }

        let avg_score = strategy_results.iter().map(|r| r.score).sum::<f32>() / n;
        let success_rate = strategy_results.iter().filter(|r| r.success).count() as f32 / n;
        let avg_alignment = strategy_results.iter().map(|r| r.value_alignment).sum::<f32>() / n;
        let avg_duration = strategy_results.iter().map(|r| r.duration_ms).sum::<u64>() / n as u64;
        let avg_cost = strategy_results.iter().map(|r| r.cost).sum::<f64>() / n as f64;

        // Collect unique risks and benefits
        let mut risks: Vec<String> = strategy_results.iter().flat_map(|r| r.risks.clone()).collect();
        risks.sort();
        risks.dedup();
        let mut benefits: Vec<String> = strategy_results.iter().flat_map(|r| r.benefits.clone()).collect();
        benefits.sort();
        benefits.dedup();

        let name = scenario.strategies.get(idx).map(|s| s.name.clone()).unwrap_or_default();

        strategy_scores.push(StrategyScore {
            strategy_index: idx,
            name,
            avg_score,
            success_rate,
            avg_value_alignment: avg_alignment,
            avg_duration_ms: avg_duration,
            avg_cost,
            top_risks: risks.into_iter().take(3).collect(),
            top_benefits: benefits.into_iter().take(3).collect(),
        });
    }

    // Best = highest combined score (70% success_rate + 30% value_alignment)
    let best_idx = strategy_scores.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let score_a = a.success_rate * 0.7 + a.avg_value_alignment * 0.3;
            let score_b = b.success_rate * 0.7 + b.avg_value_alignment * 0.3;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let confidence = strategy_scores.get(best_idx).map(|s| s.success_rate).unwrap_or(0.0);
    let recommended_name = strategy_scores.get(best_idx).map(|s| s.name.clone()).unwrap_or_default();

    Prediction {
        scenario_id: scenario.id,
        recommended_strategy: best_idx,
        recommended_name,
        confidence,
        strategy_scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::*;
    use crate::simulator::Simulator;

    #[test]
    fn predict_returns_valid_recommendation() {
        let mut s = Scenario::new(1, "Test".into());
        s.add_strategy(Strategy {
            name: "A".into(), agents_involved: vec!["R".into()],
            steps: vec!["do".into()], estimated_cost: 0.0, estimated_duration_ms: 100,
        });
        let mut sim = Simulator::new(20);
        let results = sim.simulate(&s);
        let pred = predict(&s, &results);
        assert_eq!(pred.scenario_id, 1);
        assert_eq!(pred.strategy_scores.len(), 1);
        assert!(pred.confidence > 0.0);
    }

    #[test]
    fn predict_multiple_strategies() {
        let mut s = Scenario::new(2, "Multi".into());
        s.add_strategy(Strategy {
            name: "Fast".into(), agents_involved: vec!["R".into()],
            steps: vec!["go".into()], estimated_cost: 0.0, estimated_duration_ms: 100,
        });
        s.add_strategy(Strategy {
            name: "Thorough".into(), agents_involved: vec!["R".into(), "D".into()],
            steps: vec!["plan".into(), "do".into(), "check".into()],
            estimated_cost: 0.005, estimated_duration_ms: 1000,
        });
        let mut sim = Simulator::new(50);
        let results = sim.simulate(&s);
        let pred = predict(&s, &results);
        assert_eq!(pred.strategy_scores.len(), 2);
        assert!(!pred.recommended_name.is_empty());
    }

    #[test]
    fn test_empty_results() {
        // Scenario with a strategy but zero simulations run
        let mut s = Scenario::new(3, "Empty test".into());
        s.add_strategy(Strategy {
            name: "NoRun".into(), agents_involved: vec!["R".into()],
            steps: vec!["do".into()], estimated_cost: 0.0, estimated_duration_ms: 100,
        });
        // Pass empty results (num_simulations=0)
        let results: Vec<Vec<SimulationResult>> = vec![vec![]];
        let pred = predict(&s, &results);
        // predict skips strategies with n=0, so strategy_scores may be empty
        assert_eq!(pred.scenario_id, 3);
        // recommended_strategy defaults to 0, confidence defaults to 0.0
        assert_eq!(pred.confidence, 0.0);
    }
}
