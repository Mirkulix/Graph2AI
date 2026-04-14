//! Random Convolutional Features — proven approach for gradient-free image classification.
//!
//! Pipeline:
//! 1. Apply N random convolutional filters (3x3, 5x5) to image
//! 2. ReLU activation
//! 3. Global average pooling per filter → N features
//! 4. Classify with TernaryBrain
//!
//! This is NOT learned — filters are random but fixed. The structure
//! of convolution + pooling extracts useful spatial features that
//! raw pixels or simple statistics cannot.
//!
//! Expected: 40-55% on CIFAR-10 (vs 29% with pixel statistics).

use rayon::prelude::*;

/// A bank of random convolutional filters.
pub struct RandomConvBank {
    /// Filters: [n_filters, n_channels, kh, kw]
    filters: Vec<Vec<f32>>,
    /// Biases per filter
    biases: Vec<f32>,
    /// Filter spatial size
    kernel_size: usize,
    /// Number of input channels
    in_channels: usize,
    /// Number of filters
    n_filters: usize,
    /// Image width/height
    image_size: usize,
}

impl RandomConvBank {
    /// Create a bank of random convolutional filters.
    ///
    /// Uses deterministic pseudo-random initialization.
    pub fn new(n_filters: usize, in_channels: usize, kernel_size: usize, image_size: usize, seed: u64) -> Self {
        let mut rng = seed;
        let filter_size = in_channels * kernel_size * kernel_size;

        let mut filters = Vec::with_capacity(n_filters);
        let mut biases = Vec::with_capacity(n_filters);

        let scale = (2.0 / filter_size as f64).sqrt() as f32;

        for _ in 0..n_filters {
            let mut filter = Vec::with_capacity(filter_size);
            for _ in 0..filter_size {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((rng >> 32) as f64 / u32::MAX as f64 - 0.5) as f32 * 2.0 * scale;
                filter.push(val);
            }
            filters.push(filter);

            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            biases.push(0.01); // small positive bias to help ReLU pass signal
        }

        Self { filters, biases, kernel_size, in_channels, n_filters, image_size }
    }

    /// Apply all filters to one image and return feature vector.
    ///
    /// For each filter: convolve → global max pooling → 1 scalar.
    /// Returns [n_filters] features.
    fn extract_one(&self, image: &[f32]) -> Vec<f32> {
        let h = self.image_size;
        let w = self.image_size;
        let c = self.in_channels;
        let k = self.kernel_size;
        let pad = k / 2;

        let mut features = Vec::with_capacity(self.n_filters);

        for f_idx in 0..self.n_filters {
            let filter = &self.filters[f_idx];
            let bias = self.biases[f_idx];

            // Compute full convolution map → global max pool
            let mut max_val = f32::NEG_INFINITY;

            for oy in 0..h {
                for ox in 0..w {
                    let mut conv_sum = bias;
                    for ch in 0..c {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = oy as isize + ky as isize - pad as isize;
                                let ix = ox as isize + kx as isize - pad as isize;
                                if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                    let pixel = image[ch * h * w + iy as usize * w + ix as usize];
                                    let weight = filter[ch * k * k + ky * k + kx];
                                    conv_sum += pixel * weight;
                                }
                            }
                        }
                    }

                    if conv_sum > max_val {
                        max_val = conv_sum;
                    }
                }
            }

            features.push(if max_val > f32::NEG_INFINITY { max_val } else { 0.0 });
        }

        features
    }

    /// Feature dimension = n_filters (global average pooling).
    pub fn feature_dim(&self) -> usize {
        self.n_filters
    }

    /// Extract features for all images in parallel.
    pub fn extract_batch(&self, images: &[f32], n_samples: usize) -> Vec<f32> {
        let img_size = self.image_size * self.image_size * self.in_channels;
        let n_filters = self.n_filters;

        let all: Vec<Vec<f32>> = (0..n_samples)
            .into_par_iter()
            .map(|i| self.extract_one(&images[i * img_size..(i + 1) * img_size]))
            .collect();

        let mut flat = Vec::with_capacity(n_samples * n_filters);
        for f in all {
            flat.extend_from_slice(&f);
        }
        flat
    }

}

/// Create a multi-scale random feature bank.
///
/// Combines 3x3 and 5x5 filters for multi-resolution features.
pub struct MultiScaleConvBank {
    bank_3x3: RandomConvBank,
    bank_5x5: RandomConvBank,
}

impl MultiScaleConvBank {
    pub fn new(n_filters_3x3: usize, n_filters_5x5: usize, in_channels: usize, image_size: usize) -> Self {
        Self {
            bank_3x3: RandomConvBank::new(n_filters_3x3, in_channels, 3, image_size, 42),
            bank_5x5: RandomConvBank::new(n_filters_5x5, in_channels, 5, image_size, 137),
        }
    }

    pub fn extract_batch(&self, images: &[f32], n_samples: usize) -> Vec<f32> {
        let feat_3 = self.bank_3x3.extract_batch(images, n_samples);
        let feat_5 = self.bank_5x5.extract_batch(images, n_samples);
        let dim_3 = self.bank_3x3.feature_dim();
        let dim_5 = self.bank_5x5.feature_dim();
        let total_dim = dim_3 + dim_5;

        let mut combined = Vec::with_capacity(n_samples * total_dim);
        for i in 0..n_samples {
            combined.extend_from_slice(&feat_3[i * dim_3..(i + 1) * dim_3]);
            combined.extend_from_slice(&feat_5[i * dim_5..(i + 1) * dim_5]);
        }
        combined
    }

    pub fn feature_dim(&self) -> usize {
        self.bank_3x3.feature_dim() + self.bank_5x5.feature_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_conv_basic() {
        let bank = RandomConvBank::new(16, 3, 3, 32, 42);
        let image = vec![0.5f32; 3072];
        let features = bank.extract_one(&image);
        assert_eq!(features.len(), 16);
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn random_conv_parallel_batch() {
        let bank = RandomConvBank::new(32, 3, 3, 32, 42);
        let images = vec![0.5f32; 100 * 3072];
        let features = bank.extract_batch(&images, 100);
        assert_eq!(features.len(), 100 * 32);
    }

    #[test]
    fn multiscale_features() {
        let bank = MultiScaleConvBank::new(64, 32, 3, 32);
        assert_eq!(bank.feature_dim(), 96);
        let images = vec![0.5f32; 10 * 3072];
        let features = bank.extract_batch(&images, 10);
        assert_eq!(features.len(), 10 * 96);
    }

    #[test]
    fn features_differ_for_different_images() {
        let bank = RandomConvBank::new(32, 3, 3, 32, 42);
        let mut img_a = vec![0.0f32; 3072];
        let mut img_b = vec![0.0f32; 3072];
        // Bright top-left in red channel
        for y in 0..16 { for x in 0..16 { img_a[y * 32 + x] = 1.0; } }
        // Bright bottom-right in blue channel
        for y in 16..32 { for x in 16..32 { img_b[2048 + y * 32 + x] = 1.0; } }

        let feat_a = bank.extract_one(&img_a);
        let feat_b = bank.extract_one(&img_b);
        assert_ne!(feat_a, feat_b);
    }
}
