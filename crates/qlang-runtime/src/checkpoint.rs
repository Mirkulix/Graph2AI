//! Model Checkpoint — Save and load trained model weights.
//!
//! Format (.qlm = QLANG Model):
//!   - Graph structure (as .qlg binary)
//!   - Weight tensors (raw f32 data)
//!   - Training metadata (loss, accuracy, epochs)
//!   - IGQK compression state

use qlang_core::graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model architecture as QLANG graph
    pub graph: Graph,
    /// Named weight tensors
    pub weights: HashMap<String, WeightTensor>,
    /// Training metadata
    pub metadata: TrainingMetadata,
    /// IGQK compression state (if compressed)
    pub compression: Option<CompressionState>,
}

/// A weight tensor with shape and data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String, // "f32", "ternary"
    pub data: Vec<u8>, // raw bytes
}

impl WeightTensor {
    pub fn from_f32(name: &str, shape: Vec<usize>, data: &[f32]) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.into(),
            shape,
            dtype: "f32".into(),
            data: bytes,
        }
    }

    pub fn as_f32(&self) -> Option<Vec<f32>> {
        if self.dtype != "f32" { return None; }
        Some(self.data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Training metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub epochs_trained: usize,
    pub final_loss: f32,
    pub final_accuracy: f32,
    pub learning_rate: f32,
    pub optimizer: String,
    pub training_time_ms: u64,
    pub framework: String, // always "qlang"
    pub framework_version: String,
}

impl Default for TrainingMetadata {
    fn default() -> Self {
        Self {
            epochs_trained: 0,
            final_loss: f32::NAN,
            final_accuracy: 0.0,
            learning_rate: 0.01,
            optimizer: "sgd".into(),
            training_time_ms: 0,
            framework: "qlang".into(),
            framework_version: "0.4".into(),
        }
    }
}

/// IGQK compression state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionState {
    pub method: String, // "ternary", "lowrank", "sparse"
    pub compression_ratio: f32,
    pub distortion: f32,
    pub accuracy_before: f32,
    pub accuracy_after: f32,
}

impl Checkpoint {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            weights: HashMap::new(),
            metadata: TrainingMetadata::default(),
            compression: None,
        }
    }

    /// Add a weight tensor.
    pub fn add_weight(&mut self, tensor: WeightTensor) {
        self.weights.insert(tensor.name.clone(), tensor);
    }

    /// Total parameter count.
    pub fn param_count(&self) -> usize {
        self.weights.values().map(|w| w.numel()).sum()
    }

    /// Total size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.weights.values().map(|w| w.size_bytes()).sum()
    }

    /// Save to JSON file.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from JSON file.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let checkpoint: Self = serde_json::from_str(&json)?;
        Ok(checkpoint)
    }

    /// Save in compact binary format (.qlm).
    pub fn save_binary(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let json_bytes = serde_json::to_vec(self)?;
        let mut buf = Vec::with_capacity(12 + json_bytes.len());

        // Header: "QLMD" (QLANG Model Data)
        buf.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x44]);
        // Version
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Payload size
        buf.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        // Payload
        buf.extend_from_slice(&json_bytes);

        std::fs::write(path, &buf)?;
        Ok(buf.len())
    }

    /// Load from binary format (.qlm).
    pub fn load_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        if data.len() < 12 {
            return Err("File too short".into());
        }
        if &data[0..4] != b"QLMD" {
            return Err("Invalid magic bytes".into());
        }
        let payload = &data[12..];
        let checkpoint: Self = serde_json::from_slice(payload)?;
        Ok(checkpoint)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Model: {}\n", self.graph.id));
        s.push_str(&format!("Parameters: {} ({:.1} KB)\n", self.param_count(),
            self.total_bytes() as f64 / 1024.0));
        s.push_str(&format!("Graph: {} nodes, {} edges\n", self.graph.nodes.len(), self.graph.edges.len()));
        s.push_str(&format!("Training: {} epochs, loss={:.4}, acc={:.1}%\n",
            self.metadata.epochs_trained, self.metadata.final_loss, self.metadata.final_accuracy * 100.0));
        if let Some(comp) = &self.compression {
            s.push_str(&format!("Compression: {} ({:.1}x), acc: {:.1}%→{:.1}%\n",
                comp.method, comp.compression_ratio, comp.accuracy_before * 100.0, comp.accuracy_after * 100.0));
        }

        s.push_str("\nWeights:\n");
        for (name, tensor) in &self.weights {
            s.push_str(&format!("  {}: {:?} {} ({} bytes)\n",
                name, tensor.shape, tensor.dtype, tensor.size_bytes()));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;

    #[test]
    fn checkpoint_basics() {
        let mut ckpt = Checkpoint::new(Graph::new("test_model"));
        ckpt.add_weight(WeightTensor::from_f32("W1", vec![784, 128], &vec![0.1; 784 * 128]));
        ckpt.add_weight(WeightTensor::from_f32("W2", vec![128, 10], &vec![0.1; 128 * 10]));

        assert_eq!(ckpt.param_count(), 784 * 128 + 128 * 10);
        assert_eq!(ckpt.total_bytes(), (784 * 128 + 128 * 10) * 4);
    }

    #[test]
    fn weight_tensor_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = WeightTensor::from_f32("test", vec![2, 2], &data);
        assert_eq!(tensor.as_f32().unwrap(), data);
    }

    #[test]
    fn checkpoint_binary_roundtrip() {
        let mut ckpt = Checkpoint::new(Graph::new("binary_test"));
        ckpt.add_weight(WeightTensor::from_f32("w", vec![4], &[1.0, 2.0, 3.0, 4.0]));
        ckpt.metadata.epochs_trained = 10;
        ckpt.metadata.final_loss = 0.05;

        let path = "/tmp/qlang_test_checkpoint.qlm";
        let size = ckpt.save_binary(path).unwrap();
        assert!(size > 0);

        let loaded = Checkpoint::load_binary(path).unwrap();
        assert_eq!(loaded.graph.id, "binary_test");
        assert_eq!(loaded.metadata.epochs_trained, 10);
        assert_eq!(loaded.weights["w"].as_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_summary() {
        let mut ckpt = Checkpoint::new(Graph::new("summary_test"));
        ckpt.add_weight(WeightTensor::from_f32("W1", vec![64, 32], &vec![0.0; 64 * 32]));
        ckpt.metadata.epochs_trained = 50;
        // Test fixture value — does NOT represent a real trained model.
        ckpt.metadata.final_accuracy = 0.95;

        let summary = ckpt.summary();
        assert!(summary.contains("summary_test"));
        assert!(summary.contains("50 epochs"));
        assert!(summary.contains("95.0%"));
    }
}
