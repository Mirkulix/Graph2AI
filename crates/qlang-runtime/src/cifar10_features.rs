//! CIFAR-10 Feature Extraction — transform raw pixels into classifiable features.
//!
//! Raw 32x32x3 pixels → compact feature vector that TernaryBrain can classify.
//!
//! Features extracted (all parallel via rayon):
//! 1. Spatial color pooling: average RGB per 4x4 grid region (16*3 = 48)
//! 2. Fine spatial pooling: average RGB per 8x8 grid region (64*3 = 192)
//! 3. Color histograms: per-quadrant RGB histogram (4*3*8 = 96)
//! 4. Edge features: horizontal + vertical gradients per 4x4 region (16*2 = 32)
//! 5. Intensity features: grayscale mean/std per 4x4 region (16*2 = 32)
//!
//! Total: 400 features (vs 3072 raw pixels)

use rayon::prelude::*;

const W: usize = 32;
const H: usize = 32;
#[allow(dead_code)] // Channel count constant, kept for reference with W/H
const C: usize = 3;

/// Extract features from a single CIFAR-10 image [3072] → [FEATURE_DIM].
pub fn extract_one(image: &[f32]) -> Vec<f32> {
    let mut features = Vec::with_capacity(400);

    // 1. Coarse spatial color pooling: 4x4 grid → 48 features
    for gy in 0..4 {
        for gx in 0..4 {
            let mut rgb = [0.0f32; 3];
            let mut count = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = gy * 8 + dy;
                    let x = gx * 8 + dx;
                    if y < H && x < W {
                        for c in 0..3 {
                            rgb[c] += image[c * 1024 + y * W + x];
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                for c in 0..3 { features.push(rgb[c] / count); }
            } else {
                features.extend_from_slice(&[0.0; 3]);
            }
        }
    }

    // 2. Fine spatial color pooling: 8x8 grid → 192 features
    for gy in 0..8 {
        for gx in 0..8 {
            let mut rgb = [0.0f32; 3];
            let mut count = 0.0f32;
            for dy in 0..4 {
                for dx in 0..4 {
                    let y = gy * 4 + dy;
                    let x = gx * 4 + dx;
                    if y < H && x < W {
                        for c in 0..3 {
                            rgb[c] += image[c * 1024 + y * W + x];
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                for c in 0..3 { features.push(rgb[c] / count); }
            } else {
                features.extend_from_slice(&[0.0; 3]);
            }
        }
    }

    // 3. Color histograms per quadrant: 4 quadrants × 3 channels × 8 bins = 96
    for qy in 0..2 {
        for qx in 0..2 {
            let mut hist = [0.0f32; 24]; // 3 channels × 8 bins
            let mut count = 0.0f32;
            for dy in 0..16 {
                for dx in 0..16 {
                    let y = qy * 16 + dy;
                    let x = qx * 16 + dx;
                    if y < H && x < W {
                        for c in 0..3 {
                            let val = image[c * 1024 + y * W + x];
                            let bin = (val * 7.99).min(7.0) as usize;
                            hist[c * 8 + bin] += 1.0;
                        }
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                for h in &mut hist { *h /= count; }
            }
            features.extend_from_slice(&hist);
        }
    }

    // 4. Edge features: horizontal + vertical gradient magnitude per 4x4 grid = 32
    let gray: Vec<f32> = (0..1024).map(|i| {
        (image[i] + image[1024 + i] + image[2048 + i]) / 3.0
    }).collect();

    for gy in 0..4 {
        for gx in 0..4 {
            let mut h_edge = 0.0f32;
            let mut v_edge = 0.0f32;
            let mut count = 0.0f32;
            for dy in 1..7 {
                for dx in 1..7 {
                    let y = gy * 8 + dy;
                    let x = gx * 8 + dx;
                    if y < H - 1 && x < W - 1 {
                        h_edge += (gray[y * W + x + 1] - gray[y * W + x - 1]).abs();
                        v_edge += (gray[(y + 1) * W + x] - gray[(y - 1) * W + x]).abs();
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                features.push(h_edge / count);
                features.push(v_edge / count);
            } else {
                features.extend_from_slice(&[0.0; 2]);
            }
        }
    }

    // 5. Intensity statistics per 4x4 grid: mean + std = 32
    for gy in 0..4 {
        for gx in 0..4 {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut count = 0.0f32;
            for dy in 0..8 {
                for dx in 0..8 {
                    let y = gy * 8 + dy;
                    let x = gx * 8 + dx;
                    if y < H && x < W {
                        let v = gray[y * W + x];
                        sum += v;
                        sum_sq += v * v;
                        count += 1.0;
                    }
                }
            }
            if count > 0.0 {
                let mean = sum / count;
                let var = (sum_sq / count - mean * mean).max(0.0);
                features.push(mean);
                features.push(var.sqrt());
            } else {
                features.extend_from_slice(&[0.0; 2]);
            }
        }
    }

    features
}

/// Feature dimension.
pub fn feature_dim() -> usize {
    48 + 192 + 96 + 32 + 32 // = 400
}

/// Extract features for all images in parallel.
pub fn extract_batch(images: &[f32], n_samples: usize) -> Vec<f32> {
    let dim = 3072;
    let feat_dim = feature_dim();

    let all_features: Vec<Vec<f32>> = (0..n_samples)
        .into_par_iter()
        .map(|i| extract_one(&images[i * dim..(i + 1) * dim]))
        .collect();

    let mut flat = Vec::with_capacity(n_samples * feat_dim);
    for f in all_features {
        flat.extend_from_slice(&f);
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_extraction_basic() {
        let image = vec![0.5f32; 3072];
        let features = extract_one(&image);
        assert_eq!(features.len(), feature_dim());
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn feature_extraction_batch() {
        let images = vec![0.5f32; 100 * 3072];
        let features = extract_batch(&images, 100);
        assert_eq!(features.len(), 100 * feature_dim());
    }

    #[test]
    fn features_differ_for_different_images() {
        let mut img_a = vec![0.0f32; 3072];
        let mut img_b = vec![0.0f32; 3072];
        // Image A: bright top-left
        for y in 0..16 { for x in 0..16 { img_a[y * 32 + x] = 1.0; } }
        // Image B: bright bottom-right
        for y in 16..32 { for x in 16..32 { img_b[y * 32 + x] = 1.0; } }

        let feat_a = extract_one(&img_a);
        let feat_b = extract_one(&img_b);
        assert_ne!(feat_a, feat_b, "Different images must produce different features");
    }
}
