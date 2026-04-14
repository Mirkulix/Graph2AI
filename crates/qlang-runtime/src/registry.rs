//! Model Registry — Store and version trained models.
//!
//! Provides a local model registry that tracks trained models with metadata.
//! Models are stored in `~/.qlang/models/{name}/{version}/` with:
//! - `model.qlm` — the serialized checkpoint
//! - `metadata.json` — name, version, metrics, timestamps

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::checkpoint::Checkpoint;

/// Metadata about a registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub name: String,
    pub version: String,
    pub created_at: String,
    pub param_count: u64,
    pub accuracy: Option<f64>,
    pub loss: Option<f64>,
    pub compressed: bool,
    pub compression_ratio: Option<f64>,
    pub tags: Vec<String>,
    pub description: String,
}

/// The model registry.
#[derive(Debug)]
pub struct Registry {
    root: PathBuf,
}

/// Errors from registry operations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(String),
    #[error("model not found: {0} v{1}")]
    NotFound(String, String),
    #[error("model already exists: {0} v{1}")]
    AlreadyExists(String, String),
}

impl From<serde_json::Error> for RegistryError {
    fn from(e: serde_json::Error) -> Self {
        RegistryError::Serde(e.to_string())
    }
}

impl Registry {
    /// Create a new registry at the default location (~/.qlang/models).
    pub fn new() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        Self {
            root: PathBuf::from(home).join(".qlang").join("models"),
        }
    }

    /// Create a registry at a custom path.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Path for a specific model version.
    fn model_dir(&self, name: &str, version: &str) -> PathBuf {
        self.root.join(name).join(version)
    }

    /// Save a model to the registry.
    pub fn save(
        &self,
        entry: &ModelEntry,
        checkpoint: &Checkpoint,
    ) -> Result<PathBuf, RegistryError> {
        let dir = self.model_dir(&entry.name, &entry.version);
        if dir.exists() {
            return Err(RegistryError::AlreadyExists(
                entry.name.clone(),
                entry.version.clone(),
            ));
        }
        std::fs::create_dir_all(&dir)?;

        // Save metadata
        let meta_path = dir.join("metadata.json");
        let meta_json = serde_json::to_string_pretty(entry)?;
        std::fs::write(&meta_path, meta_json)?;

        // Save checkpoint
        let model_path = dir.join("model.json");
        let model_json = serde_json::to_string(checkpoint)?;
        std::fs::write(&model_path, model_json)?;

        Ok(dir)
    }

    /// Load a model from the registry.
    pub fn load(&self, name: &str, version: &str) -> Result<(ModelEntry, Checkpoint), RegistryError> {
        let dir = self.model_dir(name, version);
        if !dir.exists() {
            return Err(RegistryError::NotFound(name.into(), version.into()));
        }

        let meta_json = std::fs::read_to_string(dir.join("metadata.json"))?;
        let entry: ModelEntry = serde_json::from_str(&meta_json)?;

        let model_json = std::fs::read_to_string(dir.join("model.json"))?;
        let checkpoint: Checkpoint = serde_json::from_str(&model_json)?;

        Ok((entry, checkpoint))
    }

    /// List all registered models.
    pub fn list(&self) -> Result<Vec<ModelEntry>, RegistryError> {
        let mut entries = Vec::new();
        if !self.root.exists() {
            return Ok(entries);
        }

        for name_dir in std::fs::read_dir(&self.root)? {
            let name_dir = name_dir?;
            if !name_dir.file_type()?.is_dir() {
                continue;
            }
            for ver_dir in std::fs::read_dir(name_dir.path())? {
                let ver_dir = ver_dir?;
                if !ver_dir.file_type()?.is_dir() {
                    continue;
                }
                let meta_path = ver_dir.path().join("metadata.json");
                if meta_path.exists() {
                    let json = std::fs::read_to_string(&meta_path)?;
                    if let Ok(entry) = serde_json::from_str::<ModelEntry>(&json) {
                        entries.push(entry);
                    }
                }
            }
        }

        entries.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
        Ok(entries)
    }

    /// Delete a model from the registry.
    pub fn delete(&self, name: &str, version: &str) -> Result<(), RegistryError> {
        let dir = self.model_dir(name, version);
        if !dir.exists() {
            return Err(RegistryError::NotFound(name.into(), version.into()));
        }
        std::fs::remove_dir_all(&dir)?;

        // Clean up empty parent directory
        let parent = self.root.join(name);
        if parent.exists() && parent.read_dir()?.next().is_none() {
            std::fs::remove_dir(&parent)?;
        }
        Ok(())
    }

    /// Compare two versions of a model.
    pub fn compare(
        &self,
        name: &str,
        version_a: &str,
        version_b: &str,
    ) -> Result<ModelComparison, RegistryError> {
        let (a, _) = self.load(name, version_a)?;
        let (b, _) = self.load(name, version_b)?;

        Ok(ModelComparison {
            name: name.into(),
            version_a: version_a.into(),
            version_b: version_b.into(),
            accuracy_delta: match (a.accuracy, b.accuracy) {
                (Some(aa), Some(bb)) => Some(bb - aa),
                _ => None,
            },
            loss_delta: match (a.loss, b.loss) {
                (Some(la), Some(lb)) => Some(lb - la),
                _ => None,
            },
            param_count_delta: b.param_count as i64 - a.param_count as i64,
            compression_ratio_delta: match (a.compression_ratio, b.compression_ratio) {
                (Some(ca), Some(cb)) => Some(cb - ca),
                _ => None,
            },
        })
    }
}

/// Result of comparing two model versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub name: String,
    pub version_a: String,
    pub version_b: String,
    pub accuracy_delta: Option<f64>,
    pub loss_delta: Option<f64>,
    pub param_count_delta: i64,
    pub compression_ratio_delta: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::checkpoint::{Checkpoint, CompressionState, TrainingMetadata};
    use qlang_core::graph::Graph;

    fn test_entry(name: &str, version: &str) -> ModelEntry {
        ModelEntry {
            name: name.into(),
            version: version.into(),
            created_at: "2026-04-03T00:00:00Z".into(),
            param_count: 1000,
            accuracy: Some(0.95),
            loss: Some(0.05),
            compressed: false,
            compression_ratio: None,
            tags: vec!["test".into()],
            description: "Test model".into(),
        }
    }

    fn test_checkpoint() -> Checkpoint {
        Checkpoint {
            graph: Graph::new("test"),
            weights: HashMap::new(),
            metadata: TrainingMetadata {
                epochs_trained: 10,
                final_loss: 0.05,
                final_accuracy: 0.95,
                learning_rate: 0.01,
                optimizer: "sgd".into(),
                training_time_ms: 1000,
                framework: "qlang".into(),
                framework_version: "0.1.0".into(),
            },
            compression: Some(CompressionState {
                method: "none".into(),
                compression_ratio: 1.0,
                distortion: 0.0,
                accuracy_before: 0.95,
                accuracy_after: 0.95,
            }),
        }
    }

    #[test]
    fn test_save_and_load() {
        let dir = std::env::temp_dir().join("qlang_registry_test_save");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = Registry::with_root(&dir);
        let entry = test_entry("mymodel", "1.0.0");
        let ckpt = test_checkpoint();

        let path = reg.save(&entry, &ckpt).unwrap();
        assert!(path.exists());

        let (loaded_entry, _loaded_ckpt) = reg.load("mymodel", "1.0.0").unwrap();
        assert_eq!(loaded_entry.name, "mymodel");
        assert_eq!(loaded_entry.accuracy, Some(0.95));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_list_models() {
        let dir = std::env::temp_dir().join("qlang_registry_test_list");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = Registry::with_root(&dir);
        let ckpt = test_checkpoint();

        reg.save(&test_entry("model_a", "1.0"), &ckpt).unwrap();
        reg.save(&test_entry("model_a", "2.0"), &ckpt).unwrap();
        reg.save(&test_entry("model_b", "1.0"), &ckpt).unwrap();

        let list = reg.list().unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].name, "model_a");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_delete_model() {
        let dir = std::env::temp_dir().join("qlang_registry_test_del");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = Registry::with_root(&dir);
        reg.save(&test_entry("delme", "1.0"), &test_checkpoint())
            .unwrap();
        assert!(reg.load("delme", "1.0").is_ok());

        reg.delete("delme", "1.0").unwrap();
        assert!(reg.load("delme", "1.0").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compare_versions() {
        let dir = std::env::temp_dir().join("qlang_registry_test_cmp");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = Registry::with_root(&dir);
        let ckpt = test_checkpoint();

        let mut entry_v1 = test_entry("cmpmodel", "1.0");
        entry_v1.accuracy = Some(0.90);
        entry_v1.loss = Some(0.10);

        let mut entry_v2 = test_entry("cmpmodel", "2.0");
        entry_v2.accuracy = Some(0.95);
        entry_v2.loss = Some(0.05);

        reg.save(&entry_v1, &ckpt).unwrap();
        reg.save(&entry_v2, &ckpt).unwrap();

        let cmp = reg.compare("cmpmodel", "1.0", "2.0").unwrap();
        assert!((cmp.accuracy_delta.unwrap() - 0.05).abs() < 1e-10);
        assert!((cmp.loss_delta.unwrap() - (-0.05)).abs() < 1e-10);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_not_found() {
        let reg = Registry::with_root("/tmp/qlang_registry_test_nf");
        assert!(reg.load("nonexistent", "1.0").is_err());
    }

    #[test]
    fn test_duplicate_save_fails() {
        let dir = std::env::temp_dir().join("qlang_registry_test_dup");
        let _ = std::fs::remove_dir_all(&dir);

        let reg = Registry::with_root(&dir);
        let ckpt = test_checkpoint();
        reg.save(&test_entry("dup", "1.0"), &ckpt).unwrap();
        assert!(reg.save(&test_entry("dup", "1.0"), &ckpt).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
