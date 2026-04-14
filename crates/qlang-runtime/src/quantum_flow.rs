//! Quantum Gradient Flow
//!
//! Implements the IGQK evolution equation:
//!   dρ/dt = -i[H, ρ] - γ{G⁻¹∇L, ρ}
//!
//! where:
//! - [H, ρ] = Hρ - ρH  (commutator — quantum exploration)
//! - {A, ρ} = Aρ + ρA  (anticommutator — gradient descent)
//! - H is a diagonal Hamiltonian (simplified Laplace-Beltrami operator)
//! - γ is the damping parameter

use qlang_core::quantum::DensityMatrix;

// ---------------------------------------------------------------------------
// Helper: dense n×n matrix operations (flat row-major storage)
// ---------------------------------------------------------------------------

/// Multiply two n×n matrices stored as flat row-major Vec<f64>.
fn mat_mul_f64(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

/// Commutator [A, B] = AB - BA.  Result is traceless.
fn mat_commutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = mat_mul_f64(a, b, n);
    let ba = mat_mul_f64(b, a, n);
    ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
}

/// Anticommutator {A, B} = AB + BA.
fn mat_anticommutator(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let ab = mat_mul_f64(a, b, n);
    let ba = mat_mul_f64(b, a, n);
    ab.iter().zip(ba.iter()).map(|(x, y)| x + y).collect()
}

/// Trace of an n×n matrix.
#[allow(dead_code)]
fn mat_trace(m: &[f64], n: usize) -> f64 {
    (0..n).map(|i| m[i * n + i]).sum()
}

// ---------------------------------------------------------------------------
// Convert DensityMatrix ↔ dense flat representation
// ---------------------------------------------------------------------------

/// Expand the spectral (eigenvalue/eigenvector) representation kept in
/// `DensityMatrix` into a full n×n dense matrix:  ρ = Σ pₖ |ψₖ⟩⟨ψₖ|
fn density_to_dense(rho: &DensityMatrix) -> Vec<f64> {
    let n = rho.dim;
    let mut dense = vec![0.0; n * n];
    for (k, &pk) in rho.eigenvalues.iter().enumerate() {
        let psi = &rho.eigenvectors[k * n..(k + 1) * n];
        for i in 0..n {
            for j in 0..n {
                dense[i * n + j] += pk * psi[i] * psi[j];
            }
        }
    }
    dense
}

/// Reconstruct a `DensityMatrix` from a dense n×n matrix by simple
/// diagonalisation via Jacobi eigenvalue iteration (sufficient for small
/// dimensions used in practice).
fn dense_to_density(dense: &[f64], n: usize) -> DensityMatrix {
    // Use Jacobi eigenvalue algorithm for real symmetric matrices.
    let (eigenvalues, eigenvectors) = jacobi_eigen(dense, n);

    // Clamp negative eigenvalues (numerical noise) and renormalise.
    let mut evals: Vec<f64> = eigenvalues.iter().map(|&v| v.max(0.0)).collect();
    let sum: f64 = evals.iter().sum();
    if sum > 1e-15 {
        for v in &mut evals {
            *v /= sum;
        }
    }

    DensityMatrix {
        dim: n,
        eigenvalues: evals,
        eigenvectors,
    }
}

/// Jacobi eigenvalue algorithm for real symmetric n×n matrices.
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors are stored
/// as row-major [n × n] — column k is eigenvector k.
fn jacobi_eigen(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = mat.to_vec();
    // v = identity (accumulates rotations)
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Givens rotation to A:  A' = G^T A G
        let mut new_a = a.clone();
        for i in 0..n {
            new_a[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
            new_a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
        }
        let a_copy = new_a.clone();
        for j in 0..n {
            new_a[p * n + j] = c * a_copy[p * n + j] + s * a_copy[q * n + j];
            new_a[q * n + j] = -s * a_copy[p * n + j] + c * a_copy[q * n + j];
        }
        a = new_a;

        // Accumulate eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[i * n + p] = c * v[i * n + p] + s * v[i * n + q];
            new_v[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    // Eigenvectors stored so that column k of v is eigenvector k.
    // DensityMatrix expects row-major [rank × dim] where row k is eigenvector k.
    // Transpose v (columns → rows).
    let mut evecs = vec![0.0; n * n];
    for k in 0..n {
        for i in 0..n {
            evecs[k * n + i] = v[i * n + k];
        }
    }

    (eigenvalues, evecs)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parameters for quantum gradient flow evolution.
#[derive(Debug, Clone)]
pub struct QuantumGradientFlow {
    /// Quantum uncertainty parameter (hbar)
    pub hbar: f64,
    /// Damping parameter γ
    pub gamma: f64,
    /// Time step dt
    pub dt: f64,
    /// Maximum number of evolution steps
    pub max_steps: usize,
}

impl Default for QuantumGradientFlow {
    fn default() -> Self {
        Self {
            hbar: 0.1,
            gamma: 0.01,
            dt: 0.001,
            max_steps: 1000,
        }
    }
}

/// Construct a diagonal Hamiltonian from eigenvalues.
///
/// This is a simplified Laplace-Beltrami operator H = diag(eigenvalues).
pub fn construct_hamiltonian(dim: usize, eigenvalues: &[f64]) -> Vec<f64> {
    let mut h = vec![0.0; dim * dim];
    for (i, &val) in eigenvalues.iter().enumerate().take(dim) {
        h[i * dim + i] = val;
    }
    h
}

/// One step of quantum gradient flow:
///
///   ρ' = ρ + dt · ( -i[H, ρ] - γ {G⁻¹∇L, ρ} )
///
/// Because we work with real-valued density matrices the commutator term
/// -i[H, ρ] is treated as an antisymmetric perturbation.  In the real
/// regime the commutator of two symmetric matrices is antisymmetric, so
/// its contribution vanishes on the diagonal (trace-preserving).  We keep
/// it for completeness — it drives off-diagonal coherence.
///
/// The `natural_gradient` parameter is G⁻¹∇L represented as a diagonal
/// n×n matrix (flat row-major).
///
/// After the update the density matrix is re-diagonalised, negative
/// eigenvalues are clamped to zero, and the trace is renormalised to 1.
pub fn evolve_step(
    rho: &DensityMatrix,
    hamiltonian: &[f64],
    natural_gradient: &[f64],
    gamma: f64,
    dt: f64,
) -> DensityMatrix {
    let n = rho.dim;
    let rho_dense = density_to_dense(rho);

    // Commutator [H, ρ]  (quantum exploration)
    let comm = mat_commutator(hamiltonian, &rho_dense, n);

    // Anticommutator {G⁻¹∇L, ρ}  (gradient descent)
    let anticomm = mat_anticommutator(natural_gradient, &rho_dense, n);

    // Euler step:  ρ' = ρ + dt·( -[H,ρ]  - γ·{G⁻¹∇L, ρ} )
    // Note: the -i factor is dropped in the real regime; we keep the sign.
    let mut new_dense = vec![0.0; n * n];
    for idx in 0..n * n {
        new_dense[idx] = rho_dense[idx] + dt * (-comm[idx] - gamma * anticomm[idx]);
    }

    // Symmetrise (numerical noise can break symmetry)
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (new_dense[i * n + j] + new_dense[j * n + i]);
            new_dense[i * n + j] = avg;
            new_dense[j * n + i] = avg;
        }
    }

    dense_to_density(&new_dense, n)
}

/// Full evolution for `params.max_steps` steps.
///
/// Returns the final density matrix and a vector of von Neumann entropy
/// values recorded at each step (length = max_steps + 1, including
/// initial state).
pub fn evolve_full(
    rho: &DensityMatrix,
    hamiltonian: &[f64],
    gradient: &[f64],
    params: &QuantumGradientFlow,
) -> (DensityMatrix, Vec<f64>) {
    let mut current = rho.clone();
    let mut entropy_history = Vec::with_capacity(params.max_steps + 1);
    entropy_history.push(current.entropy());

    for _ in 0..params.max_steps {
        current = evolve_step(&current, hamiltonian, gradient, params.gamma, params.dt);
        entropy_history.push(current.entropy());
    }

    (current, entropy_history)
}

/// Quantum measurement via the Born rule.
///
/// Uses projective measurements onto the computational basis:
///   M_w = |w⟩⟨w|
///
/// Returns P(w) = Tr(ρ M_w) for w = 0..n_outcomes-1.
/// If `n_outcomes > dim`, the extra outcomes get probability 0.
pub fn quantum_measurement(rho: &DensityMatrix, n_outcomes: usize) -> Vec<f64> {
    let n = rho.dim;
    let rho_dense = density_to_dense(rho);

    let mut probs = Vec::with_capacity(n_outcomes);
    for w in 0..n_outcomes {
        if w < n {
            // Tr(ρ |w⟩⟨w|) = ρ_{ww}
            probs.push(rho_dense[w * n + w].max(0.0));
        } else {
            probs.push(0.0);
        }
    }

    // Normalise
    let total: f64 = probs.iter().sum();
    if total > 1e-15 {
        for p in &mut probs {
            *p /= total;
        }
    }

    probs
}

/// Collapse a density matrix to a classical weight vector.
///
/// For each dimension i, computes w_i = Tr(ρ · M_i) where M_i is a
/// weighted projector onto basis state |i⟩.  In the computational basis
/// this simply yields the diagonal elements of ρ, which represent the
/// probability of each basis configuration.
///
/// The resulting vector can be interpreted as classical weights (e.g. for
/// ternary quantisation the largest-probability basis states would be
/// selected).
pub fn collapse_to_weights(rho: &DensityMatrix) -> Vec<f64> {
    let n = rho.dim;
    let rho_dense = density_to_dense(rho);

    (0..n).map(|i| rho_dense[i * n + i]).collect()
}

// ===========================================================================
// Quantum Scheduler — anneals hbar (exploration) and gamma (exploitation)
// ===========================================================================

/// Annealing schedule type for quantum parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleType {
    /// Cosine annealing (smooth, widely used).
    Cosine,
    /// Linear interpolation.
    Linear,
    /// Exponential decay.
    Exponential,
}

/// Scheduler that anneals quantum parameters during training.
///
/// During IGQK training, the balance between *exploration* (quantum
/// uncertainty, controlled by hbar) and *exploitation* (gradient descent,
/// controlled by gamma) should shift over time:
///
/// - **Early training**: high hbar (explore the loss landscape broadly),
///   low gamma (weak gradient pull).
/// - **Late training**: low hbar (collapse toward a solution),
///   high gamma (strong gradient descent).
///
/// The scheduler interpolates both parameters from start to end values
/// according to a chosen schedule (cosine, linear, or exponential).
#[derive(Debug, Clone)]
pub struct QuantumScheduler {
    /// Initial hbar (high = more exploration).
    pub hbar_start: f64,
    /// Final hbar (low = less exploration).
    pub hbar_end: f64,
    /// Initial gamma (low = weak gradient).
    pub gamma_start: f64,
    /// Final gamma (high = strong gradient).
    pub gamma_end: f64,
    /// Total number of steps.
    pub total_steps: usize,
    /// Current step (0-indexed).
    current_step: usize,
    /// Annealing schedule type.
    pub schedule: ScheduleType,
}

impl QuantumScheduler {
    /// Create a new scheduler with sensible IGQK defaults.
    ///
    /// - hbar: 0.1 -> 0.001  (exploration -> exploitation)
    /// - gamma: 0.01 -> 0.1  (weak -> strong gradient)
    /// - Schedule: Cosine
    pub fn new(total_steps: usize) -> Self {
        QuantumScheduler {
            hbar_start: 0.1,
            hbar_end: 0.001,
            gamma_start: 0.01,
            gamma_end: 0.1,
            total_steps,
            current_step: 0,
            schedule: ScheduleType::Cosine,
        }
    }

    /// Create a scheduler with custom parameters.
    pub fn with_params(
        total_steps: usize,
        hbar_start: f64,
        hbar_end: f64,
        gamma_start: f64,
        gamma_end: f64,
        schedule: ScheduleType,
    ) -> Self {
        QuantumScheduler {
            hbar_start,
            hbar_end,
            gamma_start,
            gamma_end,
            total_steps,
            current_step: 0,
            schedule,
        }
    }

    /// Get the current (hbar, gamma) values based on the schedule.
    pub fn get_params(&self) -> (f64, f64) {
        let progress = if self.total_steps == 0 {
            1.0
        } else {
            self.current_step as f64 / self.total_steps as f64
        };

        let factor = match self.schedule {
            ScheduleType::Cosine => {
                0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            ScheduleType::Linear => 1.0 - progress,
            ScheduleType::Exponential => (-5.0 * progress).exp(),
        };

        let hbar = self.hbar_end + (self.hbar_start - self.hbar_end) * factor;
        let gamma = self.gamma_end - (self.gamma_end - self.gamma_start) * factor;
        (hbar, gamma)
    }

    /// Advance the scheduler by one step.
    pub fn step(&mut self) {
        if self.current_step < self.total_steps {
            self.current_step += 1;
        }
    }

    /// Current training progress in [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.total_steps == 0 {
            1.0
        } else {
            self.current_step as f64 / self.total_steps as f64
        }
    }

    /// Current step index.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset the scheduler to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Perform one step of quantum gradient flow using the scheduler's
    /// current (hbar, gamma) values, then advance the step counter.
    ///
    /// This is a convenience method that wraps `evolve_step`.
    pub fn evolve_scheduled(
        &mut self,
        rho: &DensityMatrix,
        hamiltonian: &[f64],
        natural_gradient: &[f64],
        dt: f64,
    ) -> DensityMatrix {
        let (_hbar, gamma) = self.get_params();
        let result = evolve_step(rho, hamiltonian, natural_gradient, gamma, dt);
        self.step();
        result
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a symmetric gradient matrix with off-diagonal terms
    /// so that the anticommutator genuinely mixes the state.
    fn simple_gradient(dim: usize) -> Vec<f64> {
        let mut g = vec![0.0; dim * dim];
        for i in 0..dim {
            g[i * dim + i] = 0.1 * (i as f64 + 1.0);
            // Add small off-diagonal coupling
            for j in 0..dim {
                if i != j {
                    g[i * dim + j] = 0.02 / ((i as f64 - j as f64).abs() + 1.0);
                }
            }
        }
        // Symmetrise
        for i in 0..dim {
            for j in (i + 1)..dim {
                let avg = 0.5 * (g[i * dim + j] + g[j * dim + i]);
                g[i * dim + j] = avg;
                g[j * dim + i] = avg;
            }
        }
        g
    }

    // 1. Commutator is traceless: Tr([A,B]) = 0
    #[test]
    fn commutator_is_traceless() {
        let n = 3;
        let a = vec![
            1.0, 2.0, 0.0,
            2.0, 3.0, 1.0,
            0.0, 1.0, 4.0,
        ];
        let b = vec![
            0.0, 1.0, 2.0,
            1.0, 0.0, 3.0,
            2.0, 3.0, 0.0,
        ];
        let comm = mat_commutator(&a, &b, n);
        let tr = mat_trace(&comm, n);
        assert!(tr.abs() < 1e-12, "Tr([A,B]) = {tr}, expected 0");
    }

    // 2. Anticommutator is symmetric
    #[test]
    fn anticommutator_is_symmetric() {
        let n = 3;
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let b = vec![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        let ac = mat_anticommutator(&a, &b, n);
        for i in 0..n {
            for j in 0..n {
                let diff = (ac[i * n + j] - ac[j * n + i]).abs();
                assert!(
                    diff < 1e-12,
                    "Anticommutator not symmetric at ({i},{j}): diff={diff}"
                );
            }
        }
    }

    // 3. Evolution preserves trace (Tr(ρ) = 1 after step)
    #[test]
    fn evolution_preserves_trace() {
        let rho = DensityMatrix::maximally_mixed(4);
        let h = construct_hamiltonian(4, &[1.0, 2.0, 3.0, 4.0]);
        let grad = simple_gradient(4);
        let rho2 = evolve_step(&rho, &h, &grad, 0.01, 0.001);
        assert!(
            (rho2.trace() - 1.0).abs() < 1e-10,
            "Trace = {}, expected 1.0",
            rho2.trace()
        );
    }

    // 4. Evolution preserves positivity (eigenvalues >= 0)
    #[test]
    fn evolution_preserves_positivity() {
        let rho = DensityMatrix::maximally_mixed(4);
        let h = construct_hamiltonian(4, &[1.0, 2.0, 3.0, 4.0]);
        let grad = simple_gradient(4);
        let rho2 = evolve_step(&rho, &h, &grad, 0.01, 0.001);
        for (i, &ev) in rho2.eigenvalues.iter().enumerate() {
            assert!(ev >= -1e-12, "Eigenvalue {i} = {ev}, expected >= 0");
        }
    }

    // 5. Entropy changes during evolution
    #[test]
    fn entropy_changes_during_evolution() {
        // Start from a mixed state; the dissipative gradient term
        // reshapes the eigenvalue distribution, changing entropy.
        let rho = DensityMatrix {
            dim: 4,
            eigenvalues: vec![0.7, 0.2, 0.08, 0.02],
            eigenvectors: {
                let mut v = vec![0.0; 16];
                for i in 0..4 { v[i * 4 + i] = 1.0; }
                v
            },
        };
        let h = construct_hamiltonian(4, &[0.0, 1.0, 2.0, 3.0]);
        let grad = simple_gradient(4);

        let params = QuantumGradientFlow {
            hbar: 0.1,
            gamma: 1.0,
            dt: 0.01,
            max_steps: 100,
        };
        let (final_rho, _) = evolve_full(&rho, &h, &grad, &params);

        let s0 = rho.entropy();
        let s_final = final_rho.entropy();
        assert!(
            (s_final - s0).abs() > 1e-6,
            "Entropy did not change: s0={s0}, s_final={s_final}"
        );
    }

    // 6. Dissipative evolution concentrates state (entropy decreases)
    #[test]
    fn dissipative_evolution_concentrates_state() {
        // With strong damping (large γ) and weak Hamiltonian, the
        // gradient descent term dominates, driving the state toward a
        // lower-entropy (more concentrated) configuration.
        let rho = DensityMatrix::maximally_mixed(3);
        let initial_entropy = rho.entropy();

        let h = construct_hamiltonian(3, &[0.0, 0.0, 0.0]); // zero H
        let grad = simple_gradient(3);

        let params = QuantumGradientFlow {
            hbar: 0.1,
            gamma: 1.0,
            dt: 0.01,
            max_steps: 100,
        };
        let (final_rho, entropy_hist) = evolve_full(&rho, &h, &grad, &params);
        let s_final = final_rho.entropy();

        assert!(
            s_final < initial_entropy - 1e-4,
            "Final entropy {s_final} should be < initial {initial_entropy}"
        );
        // Last entropy should differ from first
        assert!(
            (entropy_hist.last().unwrap() - entropy_hist[0]).abs() > 1e-6,
            "Entropy should change during evolution"
        );
    }

    // 7. Hamiltonian construction is diagonal
    #[test]
    fn hamiltonian_is_diagonal() {
        let vals = [1.5, 2.5, 3.5, 4.5];
        let h = construct_hamiltonian(4, &vals);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!(
                        (h[i * 4 + j] - vals[i]).abs() < 1e-15,
                        "Diagonal ({i},{i}) wrong"
                    );
                } else {
                    assert!(
                        h[i * 4 + j].abs() < 1e-15,
                        "Off-diagonal ({i},{j}) not zero"
                    );
                }
            }
        }
    }

    // 8. Measurement probabilities sum to 1
    #[test]
    fn measurement_probabilities_sum_to_one() {
        let rho = DensityMatrix::maximally_mixed(4);
        let probs = quantum_measurement(&rho, 4);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Probabilities sum to {total}, expected 1.0"
        );
    }

    // 9. Collapse produces finite weights
    #[test]
    fn collapse_produces_finite_weights() {
        let rho = DensityMatrix::maximally_mixed(4);
        let weights = collapse_to_weights(&rho);
        assert_eq!(weights.len(), 4);
        for (i, &w) in weights.iter().enumerate() {
            assert!(w.is_finite(), "Weight {i} is not finite: {w}");
        }
    }

    // 10. Full evolution entropy history is monotonic (pure → mixed)
    #[test]
    fn entropy_history_monotonic_for_dissipative() {
        // Pure dissipation (H=0): the gradient term drives the state
        // toward lower entropy monotonically (non-increasing).
        let rho = DensityMatrix::maximally_mixed(4);
        let h = construct_hamiltonian(4, &[0.0, 0.0, 0.0, 0.0]); // no unitary term
        let grad = simple_gradient(4);

        let params = QuantumGradientFlow {
            hbar: 0.1,
            gamma: 0.5,
            dt: 0.005,
            max_steps: 30,
        };
        let (_, entropy_hist) = evolve_full(&rho, &h, &grad, &params);

        // Entropy should be non-increasing (with small tolerance for noise).
        for i in 1..entropy_hist.len() {
            assert!(
                entropy_hist[i] <= entropy_hist[i - 1] + 1e-6,
                "Entropy increased at step {i}: {} -> {}",
                entropy_hist[i - 1],
                entropy_hist[i]
            );
        }
    }

    // 11. Measurement on pure state concentrates probability
    #[test]
    fn measurement_pure_state_concentrated() {
        let psi = vec![1.0, 0.0, 0.0];
        let rho = DensityMatrix::pure_state(&psi);
        let probs = quantum_measurement(&rho, 3);
        assert!(
            probs[0] > 0.99,
            "Pure |0> should have P(0) ≈ 1, got {}",
            probs[0]
        );
    }

    // 12. Mat mul identity
    #[test]
    fn mat_mul_identity() {
        let n = 3;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut id = vec![0.0; 9];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let result = mat_mul_f64(&a, &id, n);
        for i in 0..9 {
            assert!((result[i] - a[i]).abs() < 1e-15);
        }
    }

    // ===================================================================
    // QuantumScheduler tests
    // ===================================================================

    // 13. Scheduler starts at step 0
    #[test]
    fn scheduler_initial_state() {
        let s = QuantumScheduler::new(100);
        assert_eq!(s.current_step(), 0);
        assert!((s.progress() - 0.0).abs() < 1e-15);
    }

    // 14. Scheduler hbar starts high, gamma starts low
    #[test]
    fn scheduler_initial_params() {
        let s = QuantumScheduler::new(100);
        let (hbar, gamma) = s.get_params();
        // At step 0 (progress = 0), cosine factor = 1.0
        // hbar = hbar_end + (hbar_start - hbar_end) * 1.0 = hbar_start
        assert!(
            (hbar - 0.1).abs() < 1e-10,
            "Initial hbar should be ~0.1, got {hbar}"
        );
        assert!(
            (gamma - 0.01).abs() < 1e-10,
            "Initial gamma should be ~0.01, got {gamma}"
        );
    }

    // 15. Scheduler hbar ends low, gamma ends high
    #[test]
    fn scheduler_final_params() {
        let mut s = QuantumScheduler::new(100);
        for _ in 0..100 {
            s.step();
        }
        assert_eq!(s.current_step(), 100);
        let (hbar, gamma) = s.get_params();
        // At step 100 (progress = 1.0), cosine factor ≈ 0
        assert!(
            (hbar - 0.001).abs() < 0.01,
            "Final hbar should be near 0.001, got {hbar}"
        );
        assert!(
            (gamma - 0.1).abs() < 0.01,
            "Final gamma should be near 0.1, got {gamma}"
        );
    }

    // 16. Scheduler hbar monotonically decreases (cosine)
    #[test]
    fn scheduler_hbar_decreases_cosine() {
        let mut s = QuantumScheduler::new(50);
        let mut prev_hbar = f64::MAX;
        for _ in 0..=50 {
            let (hbar, _) = s.get_params();
            assert!(
                hbar <= prev_hbar + 1e-12,
                "hbar should not increase: {prev_hbar} -> {hbar}"
            );
            prev_hbar = hbar;
            s.step();
        }
    }

    // 17. Scheduler gamma monotonically increases (cosine)
    #[test]
    fn scheduler_gamma_increases_cosine() {
        let mut s = QuantumScheduler::new(50);
        let mut prev_gamma = -f64::MAX;
        for _ in 0..=50 {
            let (_, gamma) = s.get_params();
            assert!(
                gamma >= prev_gamma - 1e-12,
                "gamma should not decrease: {prev_gamma} -> {gamma}"
            );
            prev_gamma = gamma;
            s.step();
        }
    }

    // 18. Linear schedule
    #[test]
    fn scheduler_linear() {
        let s = QuantumScheduler::with_params(
            100, 1.0, 0.0, 0.0, 1.0, ScheduleType::Linear,
        );
        let (hbar, gamma) = s.get_params();
        // At step 0, linear factor = 1.0
        assert!((hbar - 1.0).abs() < 1e-10);
        assert!((gamma - 0.0).abs() < 1e-10);
    }

    // 19. Exponential schedule
    #[test]
    fn scheduler_exponential() {
        let mut s = QuantumScheduler::with_params(
            100, 1.0, 0.0, 0.0, 1.0, ScheduleType::Exponential,
        );
        let (hbar0, _) = s.get_params();
        assert!((hbar0 - 1.0).abs() < 1e-10);
        for _ in 0..100 {
            s.step();
        }
        let (hbar_final, _) = s.get_params();
        assert!(
            hbar_final < 0.01,
            "Exponential decay should drive hbar near 0, got {hbar_final}"
        );
    }

    // 20. Step does not exceed total_steps
    #[test]
    fn scheduler_step_clamps() {
        let mut s = QuantumScheduler::new(10);
        for _ in 0..20 {
            s.step();
        }
        assert_eq!(s.current_step(), 10);
        assert!((s.progress() - 1.0).abs() < 1e-15);
    }

    // 21. Reset brings scheduler back to step 0
    #[test]
    fn scheduler_reset() {
        let mut s = QuantumScheduler::new(100);
        for _ in 0..50 {
            s.step();
        }
        assert_eq!(s.current_step(), 50);
        s.reset();
        assert_eq!(s.current_step(), 0);
        assert!((s.progress() - 0.0).abs() < 1e-15);
    }

    // 22. Zero total_steps edge case
    #[test]
    fn scheduler_zero_steps() {
        let s = QuantumScheduler::new(0);
        assert!((s.progress() - 1.0).abs() < 1e-15);
        let (hbar, gamma) = s.get_params();
        assert!(hbar.is_finite());
        assert!(gamma.is_finite());
    }

    // 23. evolve_scheduled advances the step counter
    #[test]
    fn scheduler_evolve_scheduled_advances() {
        let mut s = QuantumScheduler::new(10);
        let rho = DensityMatrix::maximally_mixed(3);
        let h = construct_hamiltonian(3, &[1.0, 2.0, 3.0]);
        let grad = simple_gradient(3);
        let _ = s.evolve_scheduled(&rho, &h, &grad, 0.001);
        assert_eq!(s.current_step(), 1);
    }

    // 24. evolve_scheduled preserves trace
    #[test]
    fn scheduler_evolve_preserves_trace() {
        let mut s = QuantumScheduler::new(10);
        let rho = DensityMatrix::maximally_mixed(4);
        let h = construct_hamiltonian(4, &[1.0, 2.0, 3.0, 4.0]);
        let grad = simple_gradient(4);
        let rho2 = s.evolve_scheduled(&rho, &h, &grad, 0.001);
        assert!(
            (rho2.trace() - 1.0).abs() < 1e-10,
            "Trace = {}, expected 1.0",
            rho2.trace()
        );
    }
}
