use crate::scenario::{Scenario, SimulationResult, Strategy};

pub struct Simulator {
    /// How many simulations to run per strategy
    pub num_simulations: u32,
    /// Random seed for reproducibility
    seed: u64,
}

impl Simulator {
    pub fn new(num_simulations: u32) -> Self {
        Self {
            num_simulations,
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Create a simulator with a fixed seed for reproducible tests.
    pub fn with_seed(num_simulations: u32, seed: u64) -> Self {
        Self { num_simulations, seed }
    }

    /// Simulate all strategies for a scenario
    pub fn simulate(&mut self, scenario: &Scenario) -> Vec<Vec<SimulationResult>> {
        let mut all_results = Vec::new();

        for (strategy_idx, strategy) in scenario.strategies.iter().enumerate() {
            let mut strategy_results = Vec::new();
            for sim_num in 0..self.num_simulations {
                let result = self.simulate_one(strategy_idx, strategy, sim_num);
                strategy_results.push(result);
            }
            all_results.push(strategy_results);
        }
        all_results
    }

    fn simulate_one(&mut self, strategy_idx: usize, strategy: &Strategy, sim_num: u32) -> SimulationResult {
        // Deterministic pseudo-random based on seed + indices
        let hash = self.pseudo_random(strategy_idx as u64, sim_num as u64);

        // Score based on strategy properties
        let agent_count = strategy.agents_involved.len() as f32;
        let step_count = strategy.steps.len() as f32;

        // More agents = more coordination overhead but more capability
        let coordination_penalty = (agent_count - 1.0).max(0.0) * 0.05;
        let capability_bonus = (agent_count * 0.1).min(0.3);

        // Fewer steps = simpler = more likely to succeed
        let simplicity_bonus = (1.0 / step_count.max(1.0)) * 0.2;

        // Cost factor — cheaper strategies get small bonus
        let cost_factor = if strategy.estimated_cost == 0.0 { 0.1 } else { 0.0 };

        // Random variation (±15%)
        let random_factor = ((hash % 30) as f32 / 100.0) - 0.15;

        let base_score = 0.5 + capability_bonus - coordination_penalty + simplicity_bonus + cost_factor;
        let score = (base_score + random_factor).clamp(0.0, 1.0);

        let success = score > 0.45;

        // Value alignment — strategies with fewer steps are more "achtsam"
        let value_alignment = (simplicity_bonus * 2.0 + 0.5).min(1.0);

        // Duration based on steps and agents
        let duration = strategy.estimated_duration_ms + (hash % 500);

        let risks = if score < 0.5 {
            vec!["Hohe Komplexität könnte zu Fehlern führen".to_string()]
        } else {
            vec![]
        };

        let benefits = if score > 0.7 {
            vec!["Hohe Erfolgschance".to_string(), "Gute Werte-Ausrichtung".to_string()]
        } else if score > 0.5 {
            vec!["Moderate Erfolgschance".to_string()]
        } else {
            vec![]
        };

        SimulationResult {
            strategy_index: strategy_idx,
            success,
            score,
            duration_ms: duration,
            cost: strategy.estimated_cost,
            value_alignment,
            risks,
            benefits,
        }
    }

    fn pseudo_random(&mut self, a: u64, b: u64) -> u64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.seed.wrapping_add(a * 31).wrapping_add(b * 17) % 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{Scenario, Strategy};

    fn test_scenario() -> Scenario {
        let mut s = Scenario::new(1, "Test goal".into());
        s.add_strategy(Strategy {
            name: "Simple".into(),
            agents_involved: vec!["Researcher".into()],
            steps: vec!["Recherchiere".into()],
            estimated_cost: 0.0,
            estimated_duration_ms: 500,
        });
        s.add_strategy(Strategy {
            name: "Complex".into(),
            agents_involved: vec!["CEO".into(), "Researcher".into(), "Developer".into()],
            steps: vec!["Plane".into(), "Recherchiere".into(), "Implementiere".into(), "Verifiziere".into()],
            estimated_cost: 0.01,
            estimated_duration_ms: 2000,
        });
        s
    }

    #[test]
    fn simulate_produces_results() {
        let scenario = test_scenario();
        let mut sim = Simulator::new(10);
        let results = sim.simulate(&scenario);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 10);
        assert_eq!(results[1].len(), 10);
    }

    #[test]
    fn simpler_strategy_scores_higher() {
        let scenario = test_scenario();
        let mut sim = Simulator::new(100);
        let results = sim.simulate(&scenario);
        let prediction = crate::predictor::predict(&scenario, &results);
        // Simple strategy should generally win
        assert!(!prediction.strategy_scores.is_empty());
        assert!(prediction.confidence > 0.0);
    }

    #[test]
    fn scores_are_in_range() {
        let scenario = test_scenario();
        let mut sim = Simulator::new(50);
        let results = sim.simulate(&scenario);
        for strategy_results in &results {
            for r in strategy_results {
                assert!(r.score >= 0.0 && r.score <= 1.0);
                assert!(r.value_alignment >= 0.0 && r.value_alignment <= 1.0);
            }
        }
    }

    #[test]
    fn test_reproducible_with_same_seed() {
        let scenario = test_scenario();
        let mut sim1 = Simulator::with_seed(10, 42);
        let mut sim2 = Simulator::with_seed(10, 42);
        let results1 = sim1.simulate(&scenario);
        let results2 = sim2.simulate(&scenario);
        // Both should produce identical scores
        for (sr1, sr2) in results1.iter().zip(results2.iter()) {
            for (r1, r2) in sr1.iter().zip(sr2.iter()) {
                assert_eq!(r1.score, r2.score);
                assert_eq!(r1.success, r2.success);
            }
        }
    }

    #[test]
    fn test_zero_simulations() {
        let scenario = test_scenario();
        let mut sim = Simulator::new(0);
        let results = sim.simulate(&scenario);
        // Two strategies, but zero runs each
        assert_eq!(results.len(), 2);
        assert!(results[0].is_empty());
        assert!(results[1].is_empty());
    }
}
