//! Mutation Engine — creates offspring specialists via weight perturbation.
//!
//! Provides deterministic mutation for ternary weights {-1, 0, +1} and f32
//! shadow weights used by FFNetwork. Operates on flat slices so callers can
//! feed weights from either `TernaryBrain` (i8) or `FFNetwork` (f32).
//!
//! All randomness flows through a seeded `XorShift64`, so the same seed always
//! produces the same mutation — essential for reproducible Organism evolution.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Parallel threshold. Vectors longer than this run via rayon.
const PAR_THRESHOLD: usize = 10_000;

/// Mutation modes supported by the engine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MutationMode {
    /// Single-parent point mutation: random flips within {-1, 0, +1}.
    PointMutation,
    /// Two-parent crossover at random cut points.
    Crossover,
    /// Point mutation plus layer-level duplication / zeroing bursts.
    Structural,
}

/// Configuration for a mutation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Probability per weight of a sign flip.
    pub flip_rate: f32,
    /// Probability per weight of being forced to 0 (pruning).
    pub zero_rate: f32,
    /// When a non-zero weight is flipped, probability it stays non-zero
    /// (vs. becoming 0). Higher = denser networks.
    pub expand_rate: f32,
    /// Probability that a crossover operation picks parent B's weight.
    pub crossover_rate: f32,
    pub mode: MutationMode,
    pub seed: u64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            flip_rate: 0.01,
            zero_rate: 0.005,
            expand_rate: 0.95,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 42,
        }
    }
}

/// Deterministic XorShift64 RNG — no external dependency.
#[derive(Debug, Clone)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        // Avoid the degenerate zero state.
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform float in [0.0, 1.0).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Use top 24 bits for mantissa-sized float.
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        (bits as f32) / (1u32 << 24) as f32
    }

    /// Standard normal via Box–Muller (two f32s consumed, one returned).
    #[inline]
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// Telemetry for a single mutation pass.
#[derive(Debug, Clone, Serialize)]
pub struct MutationStats {
    pub total_weights: usize,
    pub changed: usize,
    pub flipped_sign: usize,
    pub zeroed: usize,
    pub activated: usize, // was 0, became ±1
}

impl MutationStats {
    fn empty(total: usize) -> Self {
        Self {
            total_weights: total,
            changed: 0,
            flipped_sign: 0,
            zeroed: 0,
            activated: 0,
        }
    }
}

/// Core per-weight mutation rule for ternary values.
/// Given the old value and two uniform draws, produce the new value and
/// (optionally) what kind of change happened.
#[inline]
fn mutate_one_ternary(old: i8, r_flip: f32, r_dir: f32, config: &MutationConfig) -> i8 {
    // Zeroing pass — pruning.
    if r_flip < config.zero_rate {
        return 0;
    }
    // Flip pass.
    if r_flip < config.zero_rate + config.flip_rate {
        if old == 0 {
            // Activate: pick ±1.
            return if r_dir < 0.5 { -1 } else { 1 };
        }
        // Non-zero: either stay non-zero with flipped sign, or collapse to 0.
        if r_dir < config.expand_rate {
            return -old;
        } else {
            return 0;
        }
    }
    old
}

/// Mutate a ternary i8 slice in place. Returns number of changed weights.
pub fn mutate_ternary_i8(weights: &mut [i8], config: &MutationConfig) -> usize {
    mutate_with_stats(weights, config).changed
}

/// Mutate ternary weights and collect detailed statistics.
pub fn mutate_with_stats(weights: &mut [i8], config: &MutationConfig) -> MutationStats {
    let total = weights.len();
    if total == 0 {
        return MutationStats::empty(0);
    }

    // We need determinism across sequential and parallel paths, so we draw
    // all random numbers up-front from a single RNG stream.
    let mut rng = XorShift64::new(config.seed);
    let mut flips = vec![0f32; total];
    let mut dirs = vec![0f32; total];
    for i in 0..total {
        flips[i] = rng.next_f32();
        dirs[i] = rng.next_f32();
    }

    let apply = |w: &mut i8, rf: f32, rd: f32| -> (bool, bool, bool, bool) {
        let old = *w;
        let new = mutate_one_ternary(old, rf, rd, config);
        if new == old {
            return (false, false, false, false);
        }
        *w = new;
        let flipped_sign = old != 0 && new != 0 && old != new;
        let zeroed = old != 0 && new == 0;
        let activated = old == 0 && new != 0;
        (true, flipped_sign, zeroed, activated)
    };

    if total > PAR_THRESHOLD {
        let stats: (usize, usize, usize, usize) = weights
            .par_iter_mut()
            .zip(flips.par_iter())
            .zip(dirs.par_iter())
            .map(|((w, &rf), &rd)| {
                let (ch, fs, zr, ac) = apply(w, rf, rd);
                (ch as usize, fs as usize, zr as usize, ac as usize)
            })
            .reduce(|| (0, 0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
        MutationStats {
            total_weights: total,
            changed: stats.0,
            flipped_sign: stats.1,
            zeroed: stats.2,
            activated: stats.3,
        }
    } else {
        let mut stats = MutationStats::empty(total);
        for i in 0..total {
            let (ch, fs, zr, ac) = apply(&mut weights[i], flips[i], dirs[i]);
            if ch {
                stats.changed += 1;
            }
            if fs {
                stats.flipped_sign += 1;
            }
            if zr {
                stats.zeroed += 1;
            }
            if ac {
                stats.activated += 1;
            }
        }
        stats
    }
}

/// Mutate f32 shadow weights. `stddev` controls the magnitude of Gaussian
/// perturbations applied to a fraction of weights determined by `flip_rate`.
/// Also applies pruning via `zero_rate`. Returns number of changed entries.
pub fn mutate_shadow_f32(weights: &mut [f32], config: &MutationConfig, stddev: f32) -> usize {
    let total = weights.len();
    if total == 0 {
        return 0;
    }
    let mut rng = XorShift64::new(config.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut changes = 0usize;
    for w in weights.iter_mut() {
        let r = rng.next_f32();
        if r < config.zero_rate {
            if *w != 0.0 {
                *w = 0.0;
                changes += 1;
            }
        } else if r < config.zero_rate + config.flip_rate {
            let delta = rng.next_gauss() * stddev;
            *w += delta;
            changes += 1;
        }
    }
    changes
}

/// Crossover two ternary parents into a child of the same length.
/// `crossover_rate` in config tilts the mix toward parent B.
pub fn crossover_ternary_i8(
    parent_a: &[i8],
    parent_b: &[i8],
    config: &MutationConfig,
) -> Vec<i8> {
    let n = parent_a.len().min(parent_b.len());
    let mut child = Vec::with_capacity(n);
    let mut rng = XorShift64::new(config.seed ^ 0xC0DE_F00D_DEAD_BEEF);
    for i in 0..n {
        let pick_b = rng.next_f32() < config.crossover_rate;
        child.push(if pick_b { parent_b[i] } else { parent_a[i] });
    }
    // Apply a light point-mutation pass on top so offspring differ even when
    // parents are identical.
    let mut post_cfg = config.clone();
    post_cfg.seed = config.seed ^ 0x5EED_CAFE;
    mutate_ternary_i8(&mut child, &post_cfg);
    child
}

/// Crossover two f32 parents into a child (same length, truncated to min).
pub fn crossover_shadow_f32(
    parent_a: &[f32],
    parent_b: &[f32],
    config: &MutationConfig,
) -> Vec<f32> {
    let n = parent_a.len().min(parent_b.len());
    let mut child = Vec::with_capacity(n);
    let mut rng = XorShift64::new(config.seed ^ 0xC0DE_BABE_FACE_0001);
    for i in 0..n {
        let pick_b = rng.next_f32() < config.crossover_rate;
        child.push(if pick_b { parent_b[i] } else { parent_a[i] });
    }
    child
}

/// Higher-level: clone the weights of a `TernaryBrain` specialist (already
/// flattened to f32 via `TernaryBrain::all_weights_f32()`), apply mutation,
/// and return the mutated f32 vector suitable for re-quantization.
///
/// Values are first rounded to nearest ternary in {-1, 0, +1}, mutated in
/// ternary space, then returned as f32. Bias / non-ternary magnitudes are
/// preserved when `|w| > 1.5` by routing them through the shadow path.
pub fn mutate_specialist(original_weights: &[f32], config: &MutationConfig) -> Vec<f32> {
    let n = original_weights.len();
    let mut ternary = Vec::with_capacity(n);
    let mut is_shadow = Vec::with_capacity(n);
    for &w in original_weights {
        if w.abs() > 1.5 {
            // Treat as shadow (bias etc.) — keep magnitude.
            ternary.push(0);
            is_shadow.push(true);
        } else {
            let t = if w > 0.5 {
                1i8
            } else if w < -0.5 {
                -1
            } else {
                0
            };
            ternary.push(t);
            is_shadow.push(false);
        }
    }

    match config.mode {
        MutationMode::PointMutation | MutationMode::Crossover => {
            mutate_ternary_i8(&mut ternary, config);
        }
        MutationMode::Structural => {
            mutate_ternary_i8(&mut ternary, config);
            // Structural burst: duplicate a contiguous region.
            if n >= 8 {
                let mut rng = XorShift64::new(config.seed ^ 0xBADC_0DE);
                let src = (rng.next_u64() as usize) % (n / 2);
                let dst = n / 2 + (rng.next_u64() as usize) % (n / 2);
                let span = (n / 16).max(1);
                let end_src = (src + span).min(n);
                let end_dst = (dst + span).min(n);
                let copy_len = (end_src - src).min(end_dst - dst);
                for i in 0..copy_len {
                    ternary[dst + i] = ternary[src + i];
                }
            }
        }
    }

    // Also perturb shadow-magnitude weights via Gaussian.
    let mut shadow_vals: Vec<f32> = original_weights
        .iter()
        .zip(&is_shadow)
        .map(|(w, s)| if *s { *w } else { 0.0 })
        .collect();
    mutate_shadow_f32(&mut shadow_vals, config, 0.05);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if is_shadow[i] {
            out.push(shadow_vals[i]);
        } else {
            out.push(ternary[i] as f32);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_mutation_rate() {
        // 100K weights, flip_rate 0.01, zero_rate 0.005 → ~1.5% changes expected.
        let mut weights = vec![1i8; 100_000];
        let cfg = MutationConfig {
            flip_rate: 0.01,
            zero_rate: 0.005,
            expand_rate: 0.95,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 1234,
        };
        let changed = mutate_ternary_i8(&mut weights, &cfg);
        let rate = changed as f32 / weights.len() as f32;
        // Expected ~0.015, allow wide tolerance for statistical noise.
        assert!(
            rate > 0.010 && rate < 0.022,
            "mutation rate {} not in [0.010, 0.022]",
            rate
        );
        println!("point mutation: {} changes out of {} ({:.4})", changed, weights.len(), rate);
    }

    #[test]
    fn test_crossover_mixes_parents() {
        let parent_a = vec![1i8; 1000];
        let parent_b = vec![-1i8; 1000];
        let cfg = MutationConfig {
            // Disable post-mutation so we measure pure crossover.
            flip_rate: 0.0,
            zero_rate: 0.0,
            expand_rate: 1.0,
            crossover_rate: 0.5,
            mode: MutationMode::Crossover,
            seed: 77,
        };
        let child = crossover_ternary_i8(&parent_a, &parent_b, &cfg);
        let from_a = child.iter().filter(|&&v| v == 1).count();
        let from_b = child.iter().filter(|&&v| v == -1).count();
        assert!(from_a > 300 && from_a < 700, "from_a = {}", from_a);
        assert!(from_b > 300 && from_b < 700, "from_b = {}", from_b);
        assert_eq!(from_a + from_b, 1000);
        println!("crossover: {} from A, {} from B", from_a, from_b);
    }

    #[test]
    fn test_ternary_constraint() {
        let mut weights: Vec<i8> = (0..10_000).map(|i| ((i % 3) as i8) - 1).collect();
        let cfg = MutationConfig {
            flip_rate: 0.1,
            zero_rate: 0.05,
            expand_rate: 0.9,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 99,
        };
        mutate_ternary_i8(&mut weights, &cfg);
        for &w in &weights {
            assert!(w == -1 || w == 0 || w == 1, "invalid ternary {}", w);
        }
    }

    #[test]
    fn test_determinism() {
        let base: Vec<i8> = (0..5000).map(|i| ((i % 3) as i8) - 1).collect();
        let cfg = MutationConfig {
            flip_rate: 0.05,
            zero_rate: 0.01,
            expand_rate: 0.9,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 31415,
        };
        let mut a = base.clone();
        let mut b = base.clone();
        mutate_ternary_i8(&mut a, &cfg);
        mutate_ternary_i8(&mut b, &cfg);
        assert_eq!(a, b, "same seed must yield identical results");

        let cfg2 = MutationConfig { seed: 31416, ..cfg };
        let mut c = base.clone();
        mutate_ternary_i8(&mut c, &cfg2);
        assert_ne!(a, c, "different seeds must diverge");
    }

    #[test]
    fn test_stats_correctness() {
        let original: Vec<i8> = (0..2000).map(|i| ((i % 3) as i8) - 1).collect();
        let mut mutated = original.clone();
        let cfg = MutationConfig {
            flip_rate: 0.2,
            zero_rate: 0.1,
            expand_rate: 0.8,
            crossover_rate: 0.5,
            mode: MutationMode::PointMutation,
            seed: 2024,
        };
        let stats = mutate_with_stats(&mut mutated, &cfg);

        let mut actual_changed = 0;
        let mut actual_flipped = 0;
        let mut actual_zeroed = 0;
        let mut actual_activated = 0;
        for (o, m) in original.iter().zip(mutated.iter()) {
            if o != m {
                actual_changed += 1;
                if *o != 0 && *m != 0 && o != m {
                    actual_flipped += 1;
                }
                if *o != 0 && *m == 0 {
                    actual_zeroed += 1;
                }
                if *o == 0 && *m != 0 {
                    actual_activated += 1;
                }
            }
        }
        assert_eq!(stats.total_weights, original.len());
        assert_eq!(stats.changed, actual_changed);
        assert_eq!(stats.flipped_sign, actual_flipped);
        assert_eq!(stats.zeroed, actual_zeroed);
        assert_eq!(stats.activated, actual_activated);
        println!("stats: {:?}", stats);
    }

    #[test]
    fn test_shadow_f32_mutation() {
        let mut weights = vec![0.5f32; 10_000];
        let cfg = MutationConfig {
            flip_rate: 0.1,
            zero_rate: 0.02,
            ..Default::default()
        };
        let changed = mutate_shadow_f32(&mut weights, &cfg, 0.1);
        assert!(changed > 500 && changed < 2000, "changed = {}", changed);
    }

    #[test]
    fn test_mutate_specialist_preserves_ternary_grid() {
        let original: Vec<f32> = (0..1000)
            .map(|i| match i % 3 {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let cfg = MutationConfig::default();
        let out = mutate_specialist(&original, &cfg);
        assert_eq!(out.len(), original.len());
        for v in &out {
            assert!(
                *v == -1.0 || *v == 0.0 || *v == 1.0,
                "specialist weight {} not ternary",
                v
            );
        }
    }
}
