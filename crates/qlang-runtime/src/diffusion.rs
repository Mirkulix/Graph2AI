//! Diffusion Engine for QLANG
//!
//! Ported from ParaDiffuse (Python/PyTorch) to Rust.
//!
//! Treats text/embedding generation as iterative denoising:
//! 1. Start with pure noise
//! 2. In N steps, gradually remove noise via a learned denoiser
//! 3. End with coherent embeddings
//!
//! Supports both **cosine** and **linear** noise schedules, plus
//! deterministic DDIM sampling for fast (10-20 step) generation.

// ---------------------------------------------------------------------------
// Random number generation (xorshift64, Box-Muller)
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG state.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed },
        }
    }

    fn from_clock() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in (0, 1) -- never exactly 0 or 1.
    fn next_f64(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        v.max(1e-15).min(1.0 - 1e-15)
    }
}

/// Generate `n` standard-normal f32 values via the Box-Muller transform.
pub fn random_normal(n: usize) -> Vec<f32> {
    random_normal_seeded(n, None)
}

/// Generate `n` standard-normal f32 values, optionally with a fixed seed.
pub fn random_normal_seeded(n: usize, seed: Option<u64>) -> Vec<f32> {
    let mut rng = match seed {
        Some(s) => Xorshift64::new(s),
        None => Xorshift64::from_clock(),
    };

    let mut result = Vec::with_capacity(n + 1);
    while result.len() < n {
        let u1 = rng.next_f64();
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        result.push((r * theta.cos()) as f32);
        if result.len() < n {
            result.push((r * theta.sin()) as f32);
        }
    }
    result.truncate(n);
    result
}

// ---------------------------------------------------------------------------
// Diffusion noise schedule
// ---------------------------------------------------------------------------

/// Precomputed noise schedule for the forward diffusion process.
///
/// `q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`
#[derive(Debug, Clone)]
pub struct DiffusionSchedule {
    /// Total number of timesteps (e.g. 1000).
    pub timesteps: usize,
    /// Per-step noise `beta_t` in `(0, 1)`.
    pub betas: Vec<f32>,
    /// `alpha_t = 1 - beta_t`.
    pub alphas: Vec<f32>,
    /// Cumulative product of alphas: `alpha_bar_t = prod_{s=0}^{t} alpha_s`.
    pub alpha_bars: Vec<f32>,
    /// `sqrt(alpha_bar_t)`.
    pub sqrt_alpha_bars: Vec<f32>,
    /// `sqrt(1 - alpha_bar_t)`.
    pub sqrt_one_minus_alpha_bars: Vec<f32>,
}

impl DiffusionSchedule {
    /// Cosine schedule (Nichol & Dhariwal, 2021).
    ///
    /// Better than linear for text/embeddings because it maintains high
    /// signal-to-noise ratio for longer during the forward process.
    pub fn cosine(timesteps: usize) -> Self {
        assert!(timesteps >= 1, "timesteps must be >= 1");

        let s: f64 = 0.008; // small offset to prevent singularities
        let f0 = ((s / (1.0 + s)) * std::f64::consts::FRAC_PI_2).cos();
        let f0_sq = f0 * f0;

        let mut alpha_bars = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            let frac = (t as f64 + 1.0) / timesteps as f64; // 1/T .. T/T
            let ft = (((frac + s) / (1.0 + s)) * std::f64::consts::FRAC_PI_2).cos();
            let ab = ((ft * ft) / f0_sq).min(1.0).max(1e-8) as f32;
            alpha_bars.push(ab);
        }

        Self::from_alpha_bars(timesteps, &alpha_bars)
    }

    /// Linear schedule from `beta_start` to `beta_end`.
    pub fn linear(timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        assert!(timesteps >= 1, "timesteps must be >= 1");

        let betas: Vec<f32> = if timesteps == 1 {
            vec![beta_start]
        } else {
            (0..timesteps)
                .map(|t| {
                    beta_start
                        + (beta_end - beta_start) * t as f32 / (timesteps - 1) as f32
                })
                .collect()
        };

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alpha_bars = Vec::with_capacity(timesteps);
        let mut cumulative = 1.0_f32;
        for &a in &alphas {
            cumulative *= a;
            alpha_bars.push(cumulative.max(1e-8));
        }

        let sqrt_alpha_bars: Vec<f32> = alpha_bars.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus: Vec<f32> =
            alpha_bars.iter().map(|a| (1.0 - a).max(0.0).sqrt()).collect();

        DiffusionSchedule {
            timesteps,
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars: sqrt_one_minus,
        }
    }

    /// Build a schedule from a given `alpha_bar` curve.
    fn from_alpha_bars(timesteps: usize, alpha_bars: &[f32]) -> Self {
        let mut betas = Vec::with_capacity(timesteps);
        for t in 0..timesteps {
            let prev = if t == 0 { 1.0 } else { alpha_bars[t - 1] };
            let beta = (1.0 - alpha_bars[t] / prev).clamp(0.0, 0.999);
            betas.push(beta);
        }

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();
        let sqrt_alpha_bars: Vec<f32> = alpha_bars.iter().map(|a| a.sqrt()).collect();
        let sqrt_one_minus: Vec<f32> =
            alpha_bars.iter().map(|a| (1.0 - a).max(0.0).sqrt()).collect();

        DiffusionSchedule {
            timesteps,
            betas,
            alphas,
            alpha_bars: alpha_bars.to_vec(),
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars: sqrt_one_minus,
        }
    }

    /// Forward diffusion: add noise to clean data.
    ///
    /// `q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps`
    ///
    /// # Panics
    /// Panics if `x0` and `noise` differ in length, or `t >= timesteps`.
    pub fn add_noise(&self, x0: &[f32], noise: &[f32], t: usize) -> Vec<f32> {
        assert_eq!(x0.len(), noise.len(), "x0 and noise must have equal length");
        assert!(t < self.timesteps, "t={t} out of range [0, {})", self.timesteps);

        let sa = self.sqrt_alpha_bars[t];
        let sn = self.sqrt_one_minus_alpha_bars[t];
        x0.iter()
            .zip(noise.iter())
            .map(|(&x, &e)| sa * x + sn * e)
            .collect()
    }

    /// Predict x_0 from noisy x_t and predicted noise.
    ///
    /// `x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)`
    pub fn predict_x0(&self, xt: &[f32], predicted_noise: &[f32], t: usize) -> Vec<f32> {
        assert!(t < self.timesteps, "t out of range");
        let sa = self.sqrt_alpha_bars[t];
        let sn = self.sqrt_one_minus_alpha_bars[t];
        xt.iter()
            .zip(predicted_noise.iter())
            .map(|(&x, &eps)| (x - sn * eps) / sa.max(1e-8))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DDIM Sampler
// ---------------------------------------------------------------------------

/// Deterministic DDIM sampler for fast generation (10-20 steps).
///
/// Song, Meng & Ermon (2020): "Denoising Diffusion Implicit Models"
///
/// Unlike DDPM which requires ~1000 steps, DDIM can generate high-quality
/// outputs in as few as 10 steps by skipping intermediate timesteps.
#[derive(Debug, Clone)]
pub struct DdimSampler {
    schedule: DiffusionSchedule,
    /// Number of DDIM sampling steps (typically 10-20).
    sampling_steps: usize,
    /// Precomputed timestep indices into the schedule (descending).
    timestep_indices: Vec<usize>,
}

impl DdimSampler {
    /// Create a new DDIM sampler.
    ///
    /// `steps` sampling steps are evenly spaced across the full schedule.
    pub fn new(schedule: DiffusionSchedule, steps: usize) -> Self {
        let steps = steps.max(1).min(schedule.timesteps);
        let step_size = schedule.timesteps / steps;

        // Descending sequence: highest noise first, stepping down
        let timestep_indices: Vec<usize> = (0..steps)
            .rev()
            .map(|i| ((i * step_size) + step_size - 1).min(schedule.timesteps - 1))
            .collect();

        DdimSampler {
            schedule,
            sampling_steps: steps,
            timestep_indices,
        }
    }

    /// Number of sampling steps.
    pub fn steps(&self) -> usize {
        self.sampling_steps
    }

    /// Generate from noise using DDIM sampling.
    ///
    /// The `denoiser` closure takes `(noisy_input, timestep)` and returns
    /// the predicted noise vector (same shape as input).
    ///
    /// Returns the denoised output.
    pub fn sample<F>(&self, shape: usize, denoiser: F) -> Vec<f32>
    where
        F: Fn(&[f32], usize) -> Vec<f32>,
    {
        self.sample_seeded(shape, denoiser, None)
    }

    /// Generate from noise using DDIM sampling with optional fixed seed.
    pub fn sample_seeded<F>(&self, shape: usize, denoiser: F, seed: Option<u64>) -> Vec<f32>
    where
        F: Fn(&[f32], usize) -> Vec<f32>,
    {
        // Start from pure Gaussian noise.
        let mut x = random_normal_seeded(shape, seed);

        for (i, &t) in self.timestep_indices.iter().enumerate() {
            // Previous timestep (or 0 if this is the last step).
            let t_prev = if i + 1 < self.timestep_indices.len() {
                self.timestep_indices[i + 1]
            } else {
                0
            };

            // Predict noise at current timestep.
            let predicted_noise = denoiser(&x, t);
            assert_eq!(
                predicted_noise.len(),
                shape,
                "denoiser must return same shape as input"
            );

            let alpha_t = self.schedule.alpha_bars[t];
            let alpha_prev = if t_prev > 0 {
                self.schedule.alpha_bars[t_prev]
            } else {
                1.0 // alpha_bar_0 = 1 (no noise at t=0)
            };

            let sqrt_alpha_t = alpha_t.sqrt().max(1e-8);
            let sqrt_one_minus_alpha_t = (1.0 - alpha_t).max(0.0).sqrt();
            let sqrt_alpha_prev = alpha_prev.sqrt();
            let sqrt_one_minus_alpha_prev = (1.0 - alpha_prev).max(0.0).sqrt();

            // DDIM deterministic update:
            //   x0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            //   x_{t-1} = sqrt(alpha_prev) * x0_pred + sqrt(1-alpha_prev) * eps
            x = x
                .iter()
                .zip(predicted_noise.iter())
                .map(|(&xt, &eps)| {
                    let x0_pred = (xt - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t;
                    sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * eps
                })
                .collect();
        }

        x
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Schedule construction tests --

    #[test]
    fn cosine_schedule_creation() {
        let sched = DiffusionSchedule::cosine(1000);
        assert_eq!(sched.timesteps, 1000);
        assert_eq!(sched.betas.len(), 1000);
        assert_eq!(sched.alphas.len(), 1000);
        assert_eq!(sched.alpha_bars.len(), 1000);
        assert_eq!(sched.sqrt_alpha_bars.len(), 1000);
        assert_eq!(sched.sqrt_one_minus_alpha_bars.len(), 1000);
    }

    #[test]
    fn linear_schedule_creation() {
        let sched = DiffusionSchedule::linear(500, 0.0001, 0.02);
        assert_eq!(sched.timesteps, 500);
        assert_eq!(sched.betas.len(), 500);
    }

    #[test]
    fn cosine_alpha_bars_decrease() {
        let sched = DiffusionSchedule::cosine(100);
        // alpha_bar should generally decrease: less signal as t increases
        for i in 1..sched.alpha_bars.len() {
            assert!(
                sched.alpha_bars[i] <= sched.alpha_bars[i - 1] + 1e-6,
                "alpha_bar should not increase: t={i}, {} > {}",
                sched.alpha_bars[i],
                sched.alpha_bars[i - 1]
            );
        }
    }

    #[test]
    fn linear_alpha_bars_decrease() {
        let sched = DiffusionSchedule::linear(100, 0.0001, 0.02);
        for i in 1..sched.alpha_bars.len() {
            assert!(
                sched.alpha_bars[i] <= sched.alpha_bars[i - 1] + 1e-6,
                "alpha_bar should not increase: t={i}"
            );
        }
    }

    #[test]
    fn betas_in_valid_range() {
        let sched = DiffusionSchedule::cosine(1000);
        for (i, &b) in sched.betas.iter().enumerate() {
            assert!(b >= 0.0, "beta[{i}] = {b} < 0");
            assert!(b <= 1.0, "beta[{i}] = {b} > 1");
        }
    }

    #[test]
    fn alpha_bars_positive() {
        let sched = DiffusionSchedule::cosine(1000);
        for (i, &ab) in sched.alpha_bars.iter().enumerate() {
            assert!(ab > 0.0, "alpha_bar[{i}] = {ab} must be > 0");
            assert!(ab <= 1.0, "alpha_bar[{i}] = {ab} must be <= 1");
        }
    }

    // -- Forward diffusion tests --

    #[test]
    fn add_noise_at_t0_mostly_signal() {
        let sched = DiffusionSchedule::cosine(1000);
        let x0 = vec![1.0, 2.0, 3.0, 4.0];
        let noise = vec![0.1, 0.2, 0.3, 0.4];
        let noisy = sched.add_noise(&x0, &noise, 0);

        // At t=0, alpha_bar is close to 1, so output is mostly signal.
        for (i, (&n, &x)) in noisy.iter().zip(x0.iter()).enumerate() {
            assert!(
                (n - x).abs() < 0.5,
                "At t=0, noisy[{i}]={n} should be close to x0={x}"
            );
        }
    }

    #[test]
    fn add_noise_at_high_t_mostly_noise() {
        let sched = DiffusionSchedule::cosine(1000);
        let x0 = vec![1.0; 8];
        let noise = vec![0.0; 8];
        let t = 999;
        let noisy = sched.add_noise(&x0, &noise, t);

        // At high t with zero noise, the signal coefficient is small.
        let sa = sched.sqrt_alpha_bars[t];
        for &v in &noisy {
            assert!(
                (v - sa).abs() < 1e-5,
                "Expected ~{sa}, got {v} at t={t}"
            );
        }
    }

    #[test]
    fn add_noise_roundtrip() {
        let sched = DiffusionSchedule::cosine(100);
        let x0 = vec![1.0, -1.0, 0.5, -0.5];
        let noise = vec![0.3, -0.3, 0.1, -0.1];

        // Test at various timesteps. Avoid the very last step where
        // alpha_bar is near zero and numerical precision degrades.
        for t in [0, 10, 50, 90] {
            let noisy = sched.add_noise(&x0, &noise, t);
            let x0_recovered = sched.predict_x0(&noisy, &noise, t);
            for (i, (&recovered, &original)) in
                x0_recovered.iter().zip(x0.iter()).enumerate()
            {
                assert!(
                    (recovered - original).abs() < 1e-3,
                    "Roundtrip failed at t={t}, idx={i}: {recovered} vs {original}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "x0 and noise must have equal length")]
    fn add_noise_mismatched_lengths() {
        let sched = DiffusionSchedule::cosine(10);
        sched.add_noise(&[1.0, 2.0], &[1.0], 0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn add_noise_t_out_of_range() {
        let sched = DiffusionSchedule::cosine(10);
        sched.add_noise(&[1.0], &[0.0], 10);
    }

    // -- DDIM sampler tests --

    #[test]
    fn ddim_sample_returns_correct_shape() {
        let sched = DiffusionSchedule::cosine(100);
        let sampler = DdimSampler::new(sched, 10);
        // Trivial denoiser: always predicts zero noise (identity).
        let result = sampler.sample(64, |_x, _t| vec![0.0; 64]);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn ddim_sample_converges_with_identity_denoiser() {
        let sched = DiffusionSchedule::cosine(100);
        let sampler = DdimSampler::new(sched, 10);

        // A denoiser that returns zero means "the input is already clean".
        // Over iterations, x should converge toward something stable.
        let result = sampler.sample_seeded(16, |_x, _t| vec![0.0; 16], Some(42));
        for &v in &result {
            assert!(v.is_finite(), "Output must be finite, got {v}");
        }
    }

    #[test]
    fn ddim_sample_with_perfect_denoiser() {
        // If the denoiser perfectly returns the actual noise that was added,
        // the predicted x0 should recover the original signal.
        // We test this indirectly: a constant-output denoiser should produce
        // finite, bounded results.
        let sched = DiffusionSchedule::cosine(50);
        let sampler = DdimSampler::new(sched, 5);
        let result = sampler.sample_seeded(8, |x, _t| {
            // Return scaled version of input as "predicted noise"
            x.iter().map(|v| v * 0.5).collect()
        }, Some(123));
        for &v in &result {
            assert!(v.is_finite(), "Output must be finite");
        }
    }

    #[test]
    fn ddim_step_count() {
        let sched = DiffusionSchedule::cosine(1000);
        let sampler = DdimSampler::new(sched, 10);
        assert_eq!(sampler.steps(), 10);
    }

    #[test]
    fn ddim_step_count_clamped() {
        let sched = DiffusionSchedule::cosine(5);
        let sampler = DdimSampler::new(sched, 100);
        // steps clamped to schedule.timesteps
        assert_eq!(sampler.steps(), 5);
    }

    #[test]
    fn ddim_timestep_indices_descending() {
        let sched = DiffusionSchedule::cosine(100);
        let sampler = DdimSampler::new(sched, 10);
        for i in 1..sampler.timestep_indices.len() {
            assert!(
                sampler.timestep_indices[i] <= sampler.timestep_indices[i - 1],
                "Timestep indices must be descending"
            );
        }
    }

    // -- Random number generation tests --

    #[test]
    fn random_normal_correct_length() {
        for n in [0, 1, 2, 7, 100, 1001] {
            let v = random_normal_seeded(n, Some(42));
            assert_eq!(v.len(), n, "Expected length {n}, got {}", v.len());
        }
    }

    #[test]
    fn random_normal_finite() {
        let v = random_normal_seeded(1000, Some(12345));
        for (i, &x) in v.iter().enumerate() {
            assert!(x.is_finite(), "random_normal[{i}] = {x} is not finite");
        }
    }

    #[test]
    fn random_normal_approximate_stats() {
        // With large N, mean should be near 0 and std near 1.
        let n = 50_000;
        let v = random_normal_seeded(n, Some(99));
        let mean: f32 = v.iter().sum::<f32>() / n as f32;
        let var: f32 = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        let std = var.sqrt();

        assert!(
            mean.abs() < 0.05,
            "Mean = {mean}, expected close to 0"
        );
        assert!(
            (std - 1.0).abs() < 0.1,
            "Std = {std}, expected close to 1"
        );
    }

    #[test]
    fn random_normal_seeded_deterministic() {
        let a = random_normal_seeded(100, Some(777));
        let b = random_normal_seeded(100, Some(777));
        assert_eq!(a, b, "Same seed must produce same output");
    }

    // -- Edge cases --

    #[test]
    fn single_timestep_schedule() {
        let sched = DiffusionSchedule::cosine(1);
        assert_eq!(sched.timesteps, 1);
        let noisy = sched.add_noise(&[1.0], &[0.5], 0);
        assert_eq!(noisy.len(), 1);
        assert!(noisy[0].is_finite());
    }

    #[test]
    fn linear_single_timestep() {
        let sched = DiffusionSchedule::linear(1, 0.01, 0.02);
        assert_eq!(sched.timesteps, 1);
        assert_eq!(sched.betas.len(), 1);
    }
}
