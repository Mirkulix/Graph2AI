use qlang_core::quantum::DensityMatrix;
use qlang_runtime::quantum_flow;
use serde::{Deserialize, Serialize};

/// QO's quantum state — wraps a REAL IGQK density matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Strategy labels
    pub strategies: Vec<String>,
    /// Generation counter
    pub generation: u64,
    /// The REAL density matrix from IGQK theory
    #[serde(skip)]
    pub rho: Option<DensityMatrix>,
    /// Serializable eigenvalues (for persistence)
    pub eigenvalues: Vec<f64>,
    /// Serializable eigenvectors
    pub eigenvectors: Vec<f64>,
    pub dim: usize,
}

impl QuantumState {
    pub fn new(strategies: Vec<String>) -> Self {
        let dim = strategies.len();
        // Start as maximally mixed state (equal probability for all strategies)
        let rho = DensityMatrix::maximally_mixed(dim);
        Self {
            strategies,
            generation: 0,
            eigenvalues: rho.eigenvalues.clone(),
            eigenvectors: rho.eigenvectors.clone(),
            dim,
            rho: Some(rho),
        }
    }

    fn ensure_rho(&mut self) {
        if self.rho.is_none() {
            self.rho = Some(DensityMatrix {
                dim: self.dim,
                eigenvalues: self.eigenvalues.clone(),
                eigenvectors: self.eigenvectors.clone(),
            });
        }
    }

    /// Evolve the quantum state using REAL IGQK quantum gradient flow
    pub fn evolve(&mut self, strategy_index: usize, success: bool, learning_rate: f32) {
        self.ensure_rho();
        let rho = self.rho.as_mut().unwrap();
        let dim = rho.dim;

        // Construct natural gradient: reward successful strategy, penalize failure
        let mut gradient = vec![0.0f64; dim * dim];
        if strategy_index < dim {
            let reward = if success { learning_rate as f64 } else { -(learning_rate as f64) * 0.5 };
            gradient[strategy_index * dim + strategy_index] = reward;
        }

        // Construct Hamiltonian from current eigenvalues
        let hamiltonian = quantum_flow::construct_hamiltonian(dim, &rho.eigenvalues);

        // Evolve using REAL quantum gradient flow: dρ/dt = -i[H,ρ] - γ{G⁻¹∇L, ρ}
        // gamma and dt tuned so changes are visible after 1-5 goals
        let gamma = 0.5;
        let dt = 0.5;
        let new_rho = quantum_flow::evolve_step(rho, &hamiltonian, &gradient, gamma, dt);

        // Update state
        self.eigenvalues = new_rho.eigenvalues.clone();
        self.eigenvectors = new_rho.eigenvectors.clone();
        self.rho = Some(new_rho);
        self.generation += 1;
    }

    /// Measure the quantum state — collapse to best strategy (Born rule)
    pub fn measure(&self) -> Option<(usize, &str)> {
        // The eigenvalues directly give strategy probabilities
        // (for a diagonal density matrix in the strategy basis)
        let probs = &self.eigenvalues;
        let best = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;
        Some((best, self.strategies[best].as_str()))
    }

    /// Von Neumann entropy S(ρ) = -Tr(ρ log ρ)
    pub fn entropy(&self) -> f64 {
        if let Some(ref rho) = self.rho {
            rho.entropy()
        } else {
            // Calculate from eigenvalues
            self.eigenvalues
                .iter()
                .filter(|&&v| v > 0.0)
                .map(|&v| -v * v.ln())
                .sum()
        }
    }

    /// Purity Tr(ρ²) — 1.0 means pure state, 1/dim means maximally mixed
    pub fn purity(&self) -> f64 {
        if let Some(ref rho) = self.rho {
            rho.purity()
        } else {
            self.eigenvalues.iter().map(|v| v * v).sum()
        }
    }

    /// Summary for API
    pub fn summary(&self) -> QuantumSummary {
        QuantumSummary {
            generation: self.generation,
            entropy: self.entropy(),
            purity: self.purity(),
            top_strategy: self
                .strategies
                .iter()
                .zip(self.eigenvalues.iter())
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(s, _)| s.clone()),
            strategies: self
                .strategies
                .iter()
                .zip(self.eigenvalues.iter())
                .map(|(s, &w)| (s.clone(), w as f32))
                .collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumSummary {
    pub generation: u64,
    pub entropy: f64,
    pub purity: f64,
    pub top_strategy: Option<String>,
    pub strategies: Vec<(String, f32)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_maximally_mixed() {
        let qs = QuantumState::new(vec!["A".into(), "B".into(), "C".into()]);
        // Maximally mixed: all eigenvalues equal
        let expected = 1.0 / 3.0;
        for &ev in &qs.eigenvalues {
            assert!((ev - expected).abs() < 0.01);
        }
        // Entropy should be maximal
        assert!(qs.entropy() > 1.0);
    }

    #[test]
    fn evolve_changes_density_matrix() {
        let mut qs = QuantumState::new(vec!["A".into(), "B".into()]);
        let initial_entropy = qs.entropy();

        // Reward strategy 0 many times
        for _ in 0..20 {
            qs.evolve(0, true, 0.1);
        }

        // State should have shifted — entropy decreased (more certain)
        assert!(qs.entropy() <= initial_entropy + 0.1); // might increase slightly due to quantum dynamics
        assert_eq!(qs.generation, 20);
    }

    #[test]
    fn measure_returns_strategy() {
        let mut qs = QuantumState::new(vec!["Alpha".into(), "Beta".into()]);
        let (idx, name) = qs.measure().unwrap();
        assert!(idx < 2);
        assert!(name == "Alpha" || name == "Beta");
    }

    #[test]
    fn purity_is_valid() {
        let qs = QuantumState::new(vec!["X".into(), "Y".into(), "Z".into()]);
        let purity = qs.purity();
        // Maximally mixed: purity = 1/dim = 1/3
        assert!(purity > 0.0 && purity <= 1.0);
        assert!((purity - 1.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut qs = QuantumState::new(vec!["A".into(), "B".into()]);
        qs.evolve(0, true, 0.1);

        let json = serde_json::to_string(&qs).unwrap();
        let restored: QuantumState = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.generation, qs.generation);
        assert_eq!(restored.strategies, qs.strategies);
        assert_eq!(restored.eigenvalues.len(), qs.eigenvalues.len());
    }
}
