use serde::{Deserialize, Serialize};
use std::fmt;

/// A density matrix ρ representing a quantum state.
///
/// This is the core IGQK data type. It represents probabilistic
/// superposition of neural network weight configurations.
///
/// Properties enforced:
/// - ρ ≥ 0 (positive semidefinite)
/// - Tr(ρ) = 1 (normalized)
/// - ρ = ρ† (Hermitian)
///
/// Stored as a d×d complex matrix where d is the Hilbert space dimension.
/// For efficiency, we use low-rank representation: ρ ≈ Σ pᵢ |ψᵢ⟩⟨ψᵢ|
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityMatrix {
    /// Dimension of the Hilbert space
    pub dim: usize,
    /// Eigenvalues (probabilities), sorted descending. Sum = 1.0
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (pure states), stored as flat [dim × rank] matrix
    /// Column i is the eigenvector for eigenvalue i
    pub eigenvectors: Vec<f64>,
}

impl DensityMatrix {
    /// Create a pure state |ψ⟩⟨ψ| (rank 1 density matrix).
    /// The state vector must have length `dim`.
    pub fn pure_state(state: &[f64]) -> Self {
        let dim = state.len();
        let norm_sq: f64 = state.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        let normalized: Vec<f64> = state.iter().map(|x| x / norm).collect();

        Self {
            dim,
            eigenvalues: vec![1.0],
            eigenvectors: normalized,
        }
    }

    /// Create a maximally mixed state ρ = I/d (maximum uncertainty).
    pub fn maximally_mixed(dim: usize) -> Self {
        let p = 1.0 / dim as f64;
        let eigenvalues = vec![p; dim];

        // Eigenvectors = identity matrix columns
        let mut eigenvectors = vec![0.0; dim * dim];
        for i in 0..dim {
            eigenvectors[i * dim + i] = 1.0;
        }

        Self {
            dim,
            eigenvalues,
            eigenvectors,
        }
    }

    /// Rank of the density matrix (number of non-zero eigenvalues).
    pub fn rank(&self) -> usize {
        self.eigenvalues.iter().filter(|&&p| p > 1e-12).count()
    }

    /// Trace: must always be 1.0 for valid density matrix.
    pub fn trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }

    /// Von Neumann entropy: S(ρ) = -Tr(ρ log ρ) = -Σ pᵢ log(pᵢ)
    /// Measures quantum uncertainty. S = 0 for pure state, S = log(d) for maximally mixed.
    pub fn entropy(&self) -> f64 {
        self.eigenvalues
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Purity: Tr(ρ²) = Σ pᵢ². Equal to 1 for pure state, 1/d for maximally mixed.
    pub fn purity(&self) -> f64 {
        self.eigenvalues.iter().map(|p| p * p).sum()
    }

    /// Check if this is a valid density matrix.
    pub fn is_valid(&self) -> bool {
        let trace_ok = (self.trace() - 1.0).abs() < 1e-10;
        let eigenvalues_ok = self.eigenvalues.iter().all(|&p| p >= -1e-12);
        let dim_ok = self.eigenvectors.len() == self.dim * self.eigenvalues.len();
        trace_ok && eigenvalues_ok && dim_ok
    }

    /// Renormalize the density matrix so Tr(ρ) = 1.
    pub fn renormalize(&mut self) {
        let trace = self.trace();
        if trace > 1e-15 {
            for p in &mut self.eigenvalues {
                *p /= trace;
            }
        }
    }

    /// Perform quantum measurement with projective operators.
    /// Returns probability distribution over measurement outcomes.
    ///
    /// Given measurement operators {Mw}, returns P(w) = Tr(ρ Mw) for each w.
    /// This implements the Born rule from IGQK theory.
    pub fn measure(&self, measurement_operators: &[Vec<f64>]) -> Vec<f64> {
        let mut probabilities = Vec::with_capacity(measurement_operators.len());

        for operator in measurement_operators {
            // P(w) = Tr(ρ Mw) = Σᵢ pᵢ ⟨ψᵢ|Mw|ψᵢ⟩
            let mut prob = 0.0;
            for (k, &pk) in self.eigenvalues.iter().enumerate() {
                // Extract eigenvector k
                let psi_start = k * self.dim;
                let psi = &self.eigenvectors[psi_start..psi_start + self.dim];

                // ⟨ψ|M|ψ⟩ = Σᵢⱼ ψᵢ* Mᵢⱼ ψⱼ
                // For real case (no complex numbers yet):
                let mut expectation = 0.0;
                for i in 0..self.dim {
                    for j in 0..self.dim {
                        expectation += psi[i] * operator[i * self.dim + j] * psi[j];
                    }
                }
                prob += pk * expectation;
            }
            probabilities.push(prob.max(0.0)); // clamp numerical noise
        }

        // Normalize
        let total: f64 = probabilities.iter().sum();
        if total > 1e-15 {
            for p in &mut probabilities {
                *p /= total;
            }
        }

        probabilities
    }
}

impl fmt::Display for DensityMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ρ(dim={}, rank={}, S={:.4}, Tr={:.6})",
            self.dim,
            self.rank(),
            self.entropy(),
            self.trace()
        )
    }
}

/// Parameters for quantum gradient flow evolution.
///
/// Implements: dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionParams {
    /// Quantum uncertainty parameter (ℏ in IGQK theory)
    pub hbar: f64,
    /// Damping parameter γ (balances quantum exploration vs gradient descent)
    pub gamma: f64,
    /// Time step for numerical integration
    pub dt: f64,
    /// Number of evolution steps
    pub steps: usize,
}

impl Default for EvolutionParams {
    fn default() -> Self {
        Self {
            hbar: 0.1,
            gamma: 0.01,
            dt: 0.001,
            steps: 1000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_state_properties() {
        let psi = vec![1.0, 0.0, 0.0]; // |0⟩
        let rho = DensityMatrix::pure_state(&psi);

        assert!(rho.is_valid());
        assert_eq!(rho.rank(), 1);
        assert!((rho.trace() - 1.0).abs() < 1e-10);
        assert!((rho.purity() - 1.0).abs() < 1e-10);
        assert!(rho.entropy().abs() < 1e-10); // pure state has zero entropy
    }

    #[test]
    fn mixed_state_properties() {
        let rho = DensityMatrix::maximally_mixed(4);

        assert!(rho.is_valid());
        assert_eq!(rho.rank(), 4);
        assert!((rho.trace() - 1.0).abs() < 1e-10);
        assert!((rho.purity() - 0.25).abs() < 1e-10); // 1/d = 1/4
        assert!((rho.entropy() - 4.0_f64.ln()).abs() < 1e-10); // log(d)
    }

    #[test]
    fn renormalize_works() {
        let mut rho = DensityMatrix {
            dim: 2,
            eigenvalues: vec![0.3, 0.3], // Tr = 0.6, not 1.0
            eigenvectors: vec![1.0, 0.0, 0.0, 1.0],
        };

        assert!((rho.trace() - 1.0).abs() > 1e-10);
        rho.renormalize();
        assert!((rho.trace() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn measurement_born_rule() {
        // State: 50/50 superposition of |0⟩ and |1⟩
        let rho = DensityMatrix {
            dim: 2,
            eigenvalues: vec![0.5, 0.5],
            eigenvectors: vec![1.0, 0.0, 0.0, 1.0],
        };

        // Measurement operators: project onto |0⟩ and |1⟩
        let m0 = vec![1.0, 0.0, 0.0, 0.0]; // |0⟩⟨0|
        let m1 = vec![0.0, 0.0, 0.0, 1.0]; // |1⟩⟨1|

        let probs = rho.measure(&[m0, m1]);

        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn display_density_matrix() {
        let rho = DensityMatrix::maximally_mixed(4);
        let s = format!("{rho}");
        assert!(s.contains("dim=4"));
        assert!(s.contains("rank=4"));
    }
}
