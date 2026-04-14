//! Pretrained ResNet-18 feature extractor via candle + HuggingFace Hub.
//!
//! Downloads microsoft/resnet-18 (or similar), runs CIFAR-10 images through it,
//! extracts the 512-dim feature vector BEFORE the classification head.
//! TernaryBrain then classifies these features.

use candle_core::{Device, Tensor, DType};
use candle_nn::{Func, Module, VarBuilder};
use candle_transformers::models::resnet;
use std::sync::OnceLock;

static RESNET: OnceLock<ResNetExtractor> = OnceLock::new();

pub struct ResNetExtractor {
    model: Func<'static>,
    device: Device,
}

impl ResNetExtractor {
    /// Load pretrained ResNet-18 from HuggingFace Hub.
    pub fn load() -> Result<Self, String> {
        let device = Device::Cpu;
        tracing::info!("Loading pretrained ResNet-18...");

        // Try local converted file first, then HuggingFace
        let weights_paths = [
            "data/resnet18_pytorch.safetensors",
            "../data/resnet18_pytorch.safetensors",
            "/home/mirkulix/neoqlang/qlang/data/resnet18_pytorch.safetensors",
        ];

        let weights_path = weights_paths.iter()
            .find(|p| std::path::Path::new(p).exists())
            .map(|p| std::path::PathBuf::from(p))
            .ok_or_else(|| "ResNet-18 weights not found. Run the conversion script first.".to_string())?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                .map_err(|e| format!("Load weights: {e}"))?
        };

        // ResNet-18 without final classification layer → 512-dim features
        let model = resnet::resnet18_no_final_layer(vb)
            .map_err(|e| format!("Build model: {e}"))?;

        Ok(Self { model, device })
    }

    /// Extract features from a single CIFAR-10 image [3072] → [512].
    ///
    /// CIFAR-10 images are 32x32x3, ResNet expects 224x224x3.
    /// We upsample via simple nearest-neighbor interpolation.
    pub fn extract_one(&self, image: &[f32]) -> Result<Vec<f32>, String> {
        // Convert [C*H*W] flat → Tensor [1, 3, 32, 32]
        let tensor = Tensor::from_vec(image.to_vec(), &[3, 32, 32], &self.device)
            .map_err(|e| format!("tensor: {e}"))?;

        // Upsample 32x32 → 224x224 via interpolation
        let upsampled = tensor.unsqueeze(0)
            .map_err(|e| format!("unsqueeze: {e}"))?
            .upsample_nearest2d(224, 224)
            .map_err(|e| format!("upsample: {e}"))?;

        // ImageNet normalization
        let mean = Tensor::new(&[0.485f32, 0.456, 0.406], &self.device)
            .map_err(|e| format!("mean: {e}"))?
            .reshape(&[1, 3, 1, 1])
            .map_err(|e| format!("reshape mean: {e}"))?;
        let std = Tensor::new(&[0.229f32, 0.224, 0.225], &self.device)
            .map_err(|e| format!("std: {e}"))?
            .reshape(&[1, 3, 1, 1])
            .map_err(|e| format!("reshape std: {e}"))?;

        let normalized = upsampled.broadcast_sub(&mean)
            .map_err(|e| format!("sub: {e}"))?
            .broadcast_div(&std)
            .map_err(|e| format!("div: {e}"))?;

        // Forward through ResNet-18 (no final layer) → 512-dim features
        let features_tensor = self.model.forward(&normalized)
            .map_err(|e| format!("forward: {e}"))?;

        // Flatten: [1, 512, 1, 1] → [512]
        let flat = features_tensor.flatten_all()
            .map_err(|e| format!("flatten: {e}"))?;

        let features: Vec<f32> = flat.to_vec1()
            .map_err(|e| format!("to_vec: {e}"))?;

        Ok(features)
    }

    /// Feature dimension (ResNet-18 penultimate = 512).
    pub fn feature_dim(&self) -> usize {
        512
    }

    /// Extract features for multiple images (sequential — model is not thread-safe).
    pub fn extract_batch(&self, images: &[f32], n_samples: usize) -> Result<Vec<f32>, String> {
        let mut all = Vec::with_capacity(n_samples * 1000);
        for i in 0..n_samples {
            let features = self.extract_one(&images[i * 3072..(i + 1) * 3072])?;
            all.extend_from_slice(&features);
            if i % 500 == 0 && i > 0 {
                tracing::info!("Extracted {}/{} images", i, n_samples);
            }
        }
        Ok(all)
    }
}

/// Get or load the pretrained ResNet (lazy singleton).
pub fn get_resnet() -> Result<&'static ResNetExtractor, String> {
    if let Some(model) = RESNET.get() {
        return Ok(model);
    }
    let model = ResNetExtractor::load()?;
    let _ = RESNET.set(model);
    RESNET.get().ok_or_else(|| "Failed to cache ResNet".to_string())
}
