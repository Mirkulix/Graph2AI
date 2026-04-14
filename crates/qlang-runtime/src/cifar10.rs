//! CIFAR-10 data loader — 32x32 RGB images, 10 classes.
//!
//! Binary format per file:
//!   10000 x (1 byte label + 3072 bytes image)
//!   Image: 1024 R + 1024 G + 1024 B (row-major)

use std::path::Path;

pub struct Cifar10Data {
    pub train_images: Vec<f32>,  // [n_train, 3072] normalized 0..1
    pub train_labels: Vec<u8>,   // [n_train]
    pub test_images: Vec<f32>,   // [n_test, 3072]
    pub test_labels: Vec<u8>,    // [n_test]
    pub n_train: usize,
    pub n_test: usize,
    pub image_dim: usize,        // 3072
    pub n_classes: usize,        // 10
}

impl Cifar10Data {
    /// Load CIFAR-10 from the binary batch files directory.
    pub fn load(dir: &str) -> Result<Self, String> {
        let base = Path::new(dir);

        // Try both direct and nested paths
        let batch_dir = if base.join("data_batch_1.bin").exists() {
            base.to_path_buf()
        } else if base.join("cifar-10-batches-bin/data_batch_1.bin").exists() {
            base.join("cifar-10-batches-bin")
        } else {
            return Err(format!("CIFAR-10 not found in {}", dir));
        };

        // Load training batches
        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();
        for i in 1..=5 {
            let path = batch_dir.join(format!("data_batch_{}.bin", i));
            let (imgs, lbls) = load_batch(&path)?;
            train_images.extend_from_slice(&imgs);
            train_labels.extend_from_slice(&lbls);
        }

        // Load test batch
        let test_path = batch_dir.join("test_batch.bin");
        let (test_images, test_labels) = load_batch(&test_path)?;

        let n_train = train_labels.len();
        let n_test = test_labels.len();

        Ok(Cifar10Data {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_dim: 3072,
            n_classes: 10,
        })
    }
}

/// Load a single CIFAR-10 binary batch file.
/// Format: 10000 x (1 byte label + 3072 bytes pixel data)
fn load_batch(path: &Path) -> Result<(Vec<f32>, Vec<u8>), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;

    let record_size = 1 + 3072; // label + image
    let n_samples = data.len() / record_size;

    let mut images = Vec::with_capacity(n_samples * 3072);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let offset = i * record_size;
        labels.push(data[offset]);

        // Normalize pixels to [0, 1]
        for j in 0..3072 {
            images.push(data[offset + 1 + j] as f32 / 255.0);
        }
    }

    Ok((images, labels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_cifar10() {
        let paths = [
            "data/cifar10",
            "../data/cifar10",
            "/home/mirkulix/neoqlang/qlang/data/cifar10",
        ];

        let mut data = None;
        for p in &paths {
            if let Ok(d) = Cifar10Data::load(p) {
                data = Some(d);
                break;
            }
        }

        let data = match data {
            Some(d) => d,
            None => { println!("CIFAR-10 not found, skipping test"); return; }
        };

        assert_eq!(data.n_train, 50000);
        assert_eq!(data.n_test, 10000);
        assert_eq!(data.image_dim, 3072);
        assert_eq!(data.n_classes, 10);
        assert_eq!(data.train_images.len(), 50000 * 3072);
        assert_eq!(data.test_images.len(), 10000 * 3072);

        // Labels should be 0-9
        assert!(data.train_labels.iter().all(|&l| l < 10));
        assert!(data.test_labels.iter().all(|&l| l < 10));

        // Pixels should be [0, 1]
        assert!(data.train_images.iter().all(|&p| p >= 0.0 && p <= 1.0));

        println!("CIFAR-10 loaded: {} train, {} test", data.n_train, data.n_test);
    }
}
