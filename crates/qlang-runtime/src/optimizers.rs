//! Optimizers and gradient utilities for the QLANG autograd system.
//!
//! Provides:
//! - `Optimizer` trait for parameter updates
//! - `Sgd` — Stochastic Gradient Descent with optional momentum
//! - `Adam` — Adaptive Moment Estimation
//! - Gradient clipping (L2 norm and value)
//! - Learning rate schedules

/// Trait for parameter optimizers.
pub trait Optimizer {
    /// Update `params` in-place given their `grads`.
    fn step(&mut self, params: &mut [f32], grads: &[f32]);
}

// ---------------------------------------------------------------------------
// SGD
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent with optional momentum.
pub struct Sgd {
    pub learning_rate: f32,
    pub momentum: f32,
    velocity: Vec<f32>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    ///
    /// * `learning_rate` — step size
    /// * `momentum` — momentum factor (0.0 = no momentum)
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());

        // Lazily initialise velocity
        if self.velocity.len() != params.len() {
            self.velocity = vec![0.0; params.len()];
        }

        for i in 0..params.len() {
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i];
            params[i] -= self.learning_rate * self.velocity[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// Adam optimizer (Kingma & Ba, 2014).
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// First moment estimates.
    m: Vec<f32>,
    /// Second moment estimates.
    v: Vec<f32>,
    /// Timestep counter (for bias correction).
    t: u64,
}

impl Adam {
    /// Create a new Adam optimizer with default hyper-parameters.
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Create a new Adam optimizer with full control over hyper-parameters.
    pub fn with_params(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());

        // Lazily initialise moment vectors
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            let g = grads[i];

            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Update parameters
            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

/// Clip gradients by L2 norm.
///
/// If the L2 norm of `grads` exceeds `max_norm`, all elements are scaled
/// down so that the norm equals `max_norm`.
pub fn clip_gradients(grads: &mut [f32], max_norm: f32) {
    let norm_sq: f32 = grads.iter().map(|g| g * g).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
}

/// Clip each gradient element to the range `[-max_value, max_value]`.
pub fn clip_gradients_value(grads: &mut [f32], max_value: f32) {
    for g in grads.iter_mut() {
        *g = g.clamp(-max_value, max_value);
    }
}

// ---------------------------------------------------------------------------
// Learning-rate schedules
// ---------------------------------------------------------------------------

/// Learning rate schedule variants.
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Constant learning rate.
    Constant,
    /// Multiply by `gamma` every `step_size` steps.
    StepDecay {
        step_size: u64,
        gamma: f32,
    },
    /// Cosine annealing from `base_lr` to `min_lr`.
    CosineAnnealing {
        min_lr: f32,
    },
    /// Linear warmup from 0 to `base_lr` over `warmup_steps`, then constant.
    LinearWarmup {
        warmup_steps: u64,
    },
}

/// Compute the learning rate for a given `schedule` at `step`.
///
/// * `schedule` — the schedule variant
/// * `step` — current training step (0-indexed)
/// * `total_steps` — total number of training steps
/// * `base_lr` — base learning rate
pub fn get_lr(schedule: &LearningRateSchedule, step: u64, total_steps: u64, base_lr: f32) -> f32 {
    match schedule {
        LearningRateSchedule::Constant => base_lr,

        LearningRateSchedule::StepDecay { step_size, gamma } => {
            let n_decays = step / step_size;
            base_lr * gamma.powi(n_decays as i32)
        }

        LearningRateSchedule::CosineAnnealing { min_lr } => {
            if total_steps == 0 {
                return base_lr;
            }
            let progress = (step as f32) / (total_steps as f32);
            let cosine = (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0;
            min_lr + (base_lr - min_lr) * cosine
        }

        LearningRateSchedule::LinearWarmup { warmup_steps } => {
            if *warmup_steps == 0 || step >= *warmup_steps {
                base_lr
            } else {
                base_lr * (step as f32) / (*warmup_steps as f32)
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SGD tests
    // -----------------------------------------------------------------------

    #[test]
    fn sgd_basic_step() {
        let mut opt = Sgd::new(0.1, 0.0);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.5, 1.0, -0.5];
        opt.step(&mut params, &grads);
        // params -= 0.1 * grads
        assert!((params[0] - 0.95).abs() < 1e-6);
        assert!((params[1] - 1.9).abs() < 1e-6);
        assert!((params[2] - 3.05).abs() < 1e-6);
    }

    #[test]
    fn sgd_with_momentum() {
        let mut opt = Sgd::new(0.1, 0.9);
        let mut params = vec![1.0];
        let grads = vec![1.0];

        // Step 1: v = 0.9*0 + 1.0 = 1.0, params = 1.0 - 0.1*1.0 = 0.9
        opt.step(&mut params, &grads);
        assert!((params[0] - 0.9).abs() < 1e-6);

        // Step 2: v = 0.9*1.0 + 1.0 = 1.9, params = 0.9 - 0.1*1.9 = 0.71
        opt.step(&mut params, &grads);
        assert!((params[0] - 0.71).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Adam tests
    // -----------------------------------------------------------------------

    #[test]
    fn adam_basic_step() {
        let mut opt = Adam::new(0.001);
        let mut params = vec![0.5, -0.3];
        let grads = vec![1.0, -1.0];

        opt.step(&mut params, &grads);

        // After one step, params should have moved
        assert!(params[0] < 0.5, "param should have decreased");
        assert!(params[1] > -0.3, "param should have increased");
    }

    #[test]
    fn adam_converges_on_quadratic() {
        // Minimise f(x) = x^2, gradient = 2x
        let mut opt = Adam::new(0.1);
        let mut params = vec![5.0];

        for _ in 0..200 {
            let grads = vec![2.0 * params[0]];
            opt.step(&mut params, &grads);
        }

        assert!(
            params[0].abs() < 0.05,
            "Adam should converge near 0, got {}",
            params[0]
        );
    }

    #[test]
    fn adam_converges_faster_than_sgd() {
        // Minimise f(x) = x^2 starting from x=5
        let steps = 100;
        let start = 5.0;

        // SGD
        let mut sgd = Sgd::new(0.01, 0.0);
        let mut sgd_params = vec![start];
        for _ in 0..steps {
            let grads = vec![2.0 * sgd_params[0]];
            sgd.step(&mut sgd_params, &grads);
        }

        // Adam
        let mut adam = Adam::new(0.1);
        let mut adam_params = vec![start];
        for _ in 0..steps {
            let grads = vec![2.0 * adam_params[0]];
            adam.step(&mut adam_params, &grads);
        }

        assert!(
            adam_params[0].abs() < sgd_params[0].abs(),
            "Adam ({}) should be closer to optimum than SGD ({})",
            adam_params[0].abs(),
            sgd_params[0].abs()
        );
    }

    #[test]
    fn adam_with_custom_params() {
        let mut opt = Adam::with_params(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![1.0];
        let grads = vec![1.0];
        opt.step(&mut params, &grads);
        assert!(params[0] < 1.0);
    }

    // -----------------------------------------------------------------------
    // Gradient clipping tests
    // -----------------------------------------------------------------------

    #[test]
    fn clip_gradients_no_clip_needed() {
        let mut grads = vec![1.0, 2.0];
        let norm_before: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        clip_gradients(&mut grads, 10.0);
        let norm_after: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-6, "Should not clip");
    }

    #[test]
    fn clip_gradients_l2_clips() {
        let mut grads = vec![3.0, 4.0]; // norm = 5.0
        clip_gradients(&mut grads, 2.5);
        let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(
            (norm - 2.5).abs() < 1e-5,
            "Norm should be 2.5, got {}",
            norm
        );
        // Direction preserved
        assert!((grads[0] / grads[1] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn clip_gradients_value_clips() {
        let mut grads = vec![5.0, -3.0, 0.5, -10.0];
        clip_gradients_value(&mut grads, 2.0);
        assert_eq!(grads, vec![2.0, -2.0, 0.5, -2.0]);
    }

    #[test]
    fn clip_gradients_value_no_clip_needed() {
        let mut grads = vec![0.1, -0.2, 0.0];
        let original = grads.clone();
        clip_gradients_value(&mut grads, 1.0);
        assert_eq!(grads, original);
    }

    // -----------------------------------------------------------------------
    // Learning rate schedule tests
    // -----------------------------------------------------------------------

    #[test]
    fn lr_constant() {
        let schedule = LearningRateSchedule::Constant;
        assert!((get_lr(&schedule, 0, 100, 0.01) - 0.01).abs() < 1e-8);
        assert!((get_lr(&schedule, 50, 100, 0.01) - 0.01).abs() < 1e-8);
        assert!((get_lr(&schedule, 99, 100, 0.01) - 0.01).abs() < 1e-8);
    }

    #[test]
    fn lr_step_decay() {
        let schedule = LearningRateSchedule::StepDecay {
            step_size: 10,
            gamma: 0.5,
        };
        let base_lr = 0.1;

        // Step 0..9: 0.1
        assert!((get_lr(&schedule, 0, 100, base_lr) - 0.1).abs() < 1e-6);
        assert!((get_lr(&schedule, 9, 100, base_lr) - 0.1).abs() < 1e-6);
        // Step 10..19: 0.05
        assert!((get_lr(&schedule, 10, 100, base_lr) - 0.05).abs() < 1e-6);
        // Step 20..29: 0.025
        assert!((get_lr(&schedule, 20, 100, base_lr) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn lr_cosine_annealing() {
        let schedule = LearningRateSchedule::CosineAnnealing { min_lr: 0.0 };
        let base_lr = 0.1;
        let total = 100;

        // Step 0: should be base_lr
        let lr0 = get_lr(&schedule, 0, total, base_lr);
        assert!((lr0 - base_lr).abs() < 1e-6, "at step 0, lr={}", lr0);

        // Step total: should be min_lr
        let lr_end = get_lr(&schedule, total, total, base_lr);
        assert!(lr_end.abs() < 1e-6, "at final step, lr={}", lr_end);

        // Step total/2: should be midpoint
        let lr_mid = get_lr(&schedule, total / 2, total, base_lr);
        assert!(
            (lr_mid - base_lr / 2.0).abs() < 1e-4,
            "at midpoint, lr={}",
            lr_mid
        );

        // Monotonically decreasing
        let lr_25 = get_lr(&schedule, 25, total, base_lr);
        let lr_75 = get_lr(&schedule, 75, total, base_lr);
        assert!(lr_25 > lr_75, "lr@25={} should > lr@75={}", lr_25, lr_75);
    }

    #[test]
    fn lr_linear_warmup() {
        let schedule = LearningRateSchedule::LinearWarmup { warmup_steps: 10 };
        let base_lr = 0.1;

        // Step 0: 0
        assert!((get_lr(&schedule, 0, 100, base_lr)).abs() < 1e-8);
        // Step 5: half
        assert!((get_lr(&schedule, 5, 100, base_lr) - 0.05).abs() < 1e-6);
        // Step 10: full (warmup done)
        assert!((get_lr(&schedule, 10, 100, base_lr) - 0.1).abs() < 1e-6);
        // Step 50: still full
        assert!((get_lr(&schedule, 50, 100, base_lr) - 0.1).abs() < 1e-6);
    }
}
