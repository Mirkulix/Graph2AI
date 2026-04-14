//! Lightweight Vision Transformer for feature extraction.
//!
//! Pipeline: 32x32x3 image → 4x4 patches → linear projection → self-attention → features
//!
//! The weights are FIXED (random init, not trained). The transformer's attention
//! mechanism captures spatial relationships between patches — this is the feature
//! extraction that TernaryBrain needs for CIFAR-10.
//!
//! ViT architecture:
//! - Patch size: 4x4 → 64 patches of 48 dims (4*4*3)
//! - Projection: 48 → d_model
//! - Self-attention: 1 layer, n_heads heads
//! - Output: average of all patch embeddings → d_model features

use rayon::prelude::*;

/// A frozen (non-trainable) Vision Transformer for feature extraction.
pub struct VisionTransformer {
    /// Patch projection: [d_model, patch_dim]
    w_proj: Vec<f32>,
    b_proj: Vec<f32>,
    /// Query weights: [d_model, d_model]
    w_q: Vec<f32>,
    /// Key weights: [d_model, d_model]
    w_k: Vec<f32>,
    /// Value weights: [d_model, d_model]
    w_v: Vec<f32>,
    /// Output projection: [d_model, d_model]
    w_o: Vec<f32>,
    /// Positional encoding: [n_patches, d_model]
    pos_enc: Vec<f32>,
    /// FFN weights
    w_ff1: Vec<f32>,
    w_ff2: Vec<f32>,

    pub d_model: usize,
    pub n_heads: usize,
    pub n_patches: usize,
    pub patch_dim: usize,
    pub d_ff: usize,
}

impl VisionTransformer {
    /// Create with deterministic random weights.
    pub fn new(d_model: usize, n_heads: usize, image_size: usize, patch_size: usize, channels: usize) -> Self {
        let n_patches = (image_size / patch_size) * (image_size / patch_size);
        let patch_dim = patch_size * patch_size * channels;
        let d_ff = d_model * 2;

        let mut seed = 31337u64;
        let mut rand = |scale: f32| -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / u32::MAX as f64 - 0.5) as f32 * 2.0 * scale
        };

        let proj_scale = (2.0 / (patch_dim + d_model) as f64).sqrt() as f32;
        let attn_scale = (2.0 / (d_model * 2) as f64).sqrt() as f32;
        let ff_scale = (2.0 / (d_model + d_ff) as f64).sqrt() as f32;

        let w_proj: Vec<f32> = (0..d_model * patch_dim).map(|_| rand(proj_scale)).collect();
        let b_proj = vec![0.0f32; d_model];
        let w_q: Vec<f32> = (0..d_model * d_model).map(|_| rand(attn_scale)).collect();
        let w_k: Vec<f32> = (0..d_model * d_model).map(|_| rand(attn_scale)).collect();
        let w_v: Vec<f32> = (0..d_model * d_model).map(|_| rand(attn_scale)).collect();
        let w_o: Vec<f32> = (0..d_model * d_model).map(|_| rand(attn_scale)).collect();
        let w_ff1: Vec<f32> = (0..d_model * d_ff).map(|_| rand(ff_scale)).collect();
        let w_ff2: Vec<f32> = (0..d_ff * d_model).map(|_| rand(ff_scale)).collect();

        // Sinusoidal positional encoding
        let mut pos_enc = vec![0.0f32; n_patches * d_model];
        for pos in 0..n_patches {
            for i in 0..d_model {
                let angle = pos as f32 / 10000.0f32.powf(2.0 * (i / 2) as f32 / d_model as f32);
                pos_enc[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            w_proj, b_proj, w_q, w_k, w_v, w_o, pos_enc, w_ff1, w_ff2,
            d_model, n_heads, n_patches, patch_dim, d_ff,
        }
    }

    /// Extract patches from a CIFAR-10 image [C*H*W] → [n_patches, patch_dim].
    fn patchify(&self, image: &[f32], image_size: usize, patch_size: usize, channels: usize) -> Vec<f32> {
        let patches_per_row = image_size / patch_size;
        let n_patches = patches_per_row * patches_per_row;
        let mut patches = vec![0.0f32; n_patches * self.patch_dim];

        for py in 0..patches_per_row {
            for px in 0..patches_per_row {
                let patch_idx = py * patches_per_row + px;
                for c in 0..channels {
                    for dy in 0..patch_size {
                        for dx in 0..patch_size {
                            let y = py * patch_size + dy;
                            let x = px * patch_size + dx;
                            let src = c * image_size * image_size + y * image_size + x;
                            let dst = patch_idx * self.patch_dim + c * patch_size * patch_size + dy * patch_size + dx;
                            patches[dst] = image[src];
                        }
                    }
                }
            }
        }
        patches
    }

    /// Linear projection: [n_patches, patch_dim] → [n_patches, d_model].
    fn project(&self, patches: &[f32]) -> Vec<f32> {
        let mut projected = vec![0.0f32; self.n_patches * self.d_model];
        for p in 0..self.n_patches {
            for j in 0..self.d_model {
                let mut sum = self.b_proj[j];
                for k in 0..self.patch_dim {
                    sum += patches[p * self.patch_dim + k] * self.w_proj[j * self.patch_dim + k];
                }
                projected[p * self.d_model + j] = sum;
            }
        }
        // Add positional encoding
        for i in 0..self.n_patches * self.d_model {
            projected[i] += self.pos_enc[i];
        }
        projected
    }

    /// Self-attention: [n_patches, d_model] → [n_patches, d_model].
    fn self_attention(&self, x: &[f32]) -> Vec<f32> {
        let n = self.n_patches;
        let d = self.d_model;
        let h = self.n_heads;
        let dk = d / h;
        let scale = 1.0 / (dk as f32).sqrt();

        // Q, K, V projections
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];

        for p in 0..n {
            for j in 0..d {
                let mut sq = 0.0f32;
                let mut sk = 0.0f32;
                let mut sv = 0.0f32;
                for i in 0..d {
                    let xi = x[p * d + i];
                    sq += xi * self.w_q[j * d + i];
                    sk += xi * self.w_k[j * d + i];
                    sv += xi * self.w_v[j * d + i];
                }
                q[p * d + j] = sq;
                k[p * d + j] = sk;
                v[p * d + j] = sv;
            }
        }

        // Attention: softmax(Q·K^T / sqrt(dk)) · V — per head
        let mut attn_out = vec![0.0f32; n * d];
        for head in 0..h {
            let offset = head * dk;
            for i in 0..n {
                // Compute attention scores for patch i
                let mut scores = vec![0.0f32; n];
                for j in 0..n {
                    let mut dot = 0.0f32;
                    for k_idx in 0..dk {
                        dot += q[i * d + offset + k_idx] * k[j * d + offset + k_idx];
                    }
                    scores[j] = dot * scale;
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
                for s in &mut scores { *s /= sum_exp; }

                // Weighted sum of values
                for k_idx in 0..dk {
                    let mut weighted = 0.0f32;
                    for j in 0..n {
                        weighted += scores[j] * v[j * d + offset + k_idx];
                    }
                    attn_out[i * d + offset + k_idx] = weighted;
                }
            }
        }

        // Output projection
        let mut out = vec![0.0f32; n * d];
        for p in 0..n {
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += attn_out[p * d + i] * self.w_o[j * d + i];
                }
                out[p * d + j] = x[p * d + j] + sum; // residual connection
            }
        }
        out
    }

    /// FFN: [n, d] → [n, d_ff] → ReLU → [n, d] + residual.
    fn ffn(&self, x: &[f32]) -> Vec<f32> {
        let n = self.n_patches;
        let d = self.d_model;

        let mut hidden = vec![0.0f32; n * self.d_ff];
        for p in 0..n {
            for j in 0..self.d_ff {
                let mut sum = 0.0f32;
                for i in 0..d {
                    sum += x[p * d + i] * self.w_ff1[j * d + i];
                }
                hidden[p * self.d_ff + j] = sum.max(0.0); // ReLU
            }
        }

        let mut out = vec![0.0f32; n * d];
        for p in 0..n {
            for j in 0..d {
                let mut sum = 0.0f32;
                for i in 0..self.d_ff {
                    sum += hidden[p * self.d_ff + i] * self.w_ff2[j * self.d_ff + i];
                }
                out[p * d + j] = x[p * d + j] + sum; // residual
            }
        }
        out
    }

    /// Full forward: image → patches → project → attention → FFN → avg pool → features.
    pub fn extract_one(&self, image: &[f32]) -> Vec<f32> {
        let patches = self.patchify(image, 32, 4, 3);
        let projected = self.project(&patches);
        let attended = self.self_attention(&projected);
        let ffn_out = self.ffn(&attended);

        // Global average pooling over patches → [d_model]
        let mut features = vec![0.0f32; self.d_model];
        for p in 0..self.n_patches {
            for j in 0..self.d_model {
                features[j] += ffn_out[p * self.d_model + j];
            }
        }
        for j in 0..self.d_model {
            features[j] /= self.n_patches as f32;
        }
        features
    }

    /// Batch extraction (parallel).
    pub fn extract_batch(&self, images: &[f32], n_samples: usize) -> Vec<f32> {
        let img_dim = 3072; // 32*32*3
        let d = self.d_model;

        let all: Vec<Vec<f32>> = (0..n_samples)
            .into_par_iter()
            .map(|i| self.extract_one(&images[i * img_dim..(i + 1) * img_dim]))
            .collect();

        let mut flat = Vec::with_capacity(n_samples * d);
        for f in all { flat.extend_from_slice(&f); }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vit_basic() {
        let vit = VisionTransformer::new(32, 4, 32, 4, 3);
        assert_eq!(vit.n_patches, 64); // 8x8 patches
        assert_eq!(vit.patch_dim, 48); // 4*4*3

        let image = vec![0.5f32; 3072];
        let features = vit.extract_one(&image);
        assert_eq!(features.len(), 32);
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn vit_batch_parallel() {
        let vit = VisionTransformer::new(32, 4, 32, 4, 3);
        let images = vec![0.5f32; 100 * 3072];
        let features = vit.extract_batch(&images, 100);
        assert_eq!(features.len(), 100 * 32);
    }

    #[test]
    fn vit_different_images_different_features() {
        let vit = VisionTransformer::new(32, 4, 32, 4, 3);
        let mut img_a = vec![0.0f32; 3072];
        let mut img_b = vec![0.0f32; 3072];
        for i in 0..1024 { img_a[i] = 1.0; } // red image
        for i in 2048..3072 { img_b[i] = 1.0; } // blue image

        let feat_a = vit.extract_one(&img_a);
        let feat_b = vit.extract_one(&img_b);
        assert_ne!(feat_a, feat_b);
    }
}
