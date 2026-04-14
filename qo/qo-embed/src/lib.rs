//! QO Embedding Model — all-MiniLM-L6-v2 in pure Rust
//!
//! Uses candle (HuggingFace's Rust ML framework) to run a real
//! sentence embedding model. No Python, no Ollama, no external server.
//!
//! Model: sentence-transformers/all-MiniLM-L6-v2
//! - 22MB parameters
//! - 384-dimensional embeddings
//! - ~5ms per sentence on CPU
//! - Understands semantic similarity

use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

/// The embedding model — loaded once, used forever
static MODEL: OnceLock<EmbeddingModel> = OnceLock::new();

pub struct EmbeddingModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

/// Get or load the embedding model (lazy singleton)
pub fn get_model() -> Result<&'static EmbeddingModel, String> {
    if let Some(model) = MODEL.get() {
        return Ok(model);
    }
    tracing::info!("Loading all-MiniLM-L6-v2 embedding model...");
    let start = std::time::Instant::now();
    let model = EmbeddingModel::load()?;
    tracing::info!("Embedding model loaded in {:?}", start.elapsed());
    let _ = MODEL.set(model);
    MODEL.get().ok_or_else(|| "Failed to cache model".to_string())
}

impl EmbeddingModel {
    /// Load all-MiniLM-L6-v2 from HuggingFace Hub (downloads once, caches locally)
    fn load() -> Result<Self, String> {
        let device = Device::Cpu;
        let repo_id = "sentence-transformers/all-MiniLM-L6-v2";

        // Download model files from HuggingFace
        let api = Api::new().map_err(|e| format!("HF API error: {e}"))?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        let config_path = repo.get("config.json").map_err(|e| format!("Config download: {e}"))?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| format!("Tokenizer download: {e}"))?;
        let weights_path = repo.get("model.safetensors").map_err(|e| format!("Weights download: {e}"))?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("Read config: {e}"))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Parse config: {e}"))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Load tokenizer: {e}"))?;

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .map_err(|e| format!("Load weights: {e}"))?
        };

        let model = BertModel::load(vb, &config)
            .map_err(|e| format!("Build model: {e}"))?;

        Ok(Self { model, tokenizer, device })
    }

    /// Embed a single text into a 384-dim vector
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| format!("Tokenize: {e}"))?;

        let token_ids = encoding.get_ids().to_vec();
        let attention_mask = encoding.get_attention_mask().to_vec();

        let token_ids = Tensor::new(vec![token_ids], &self.device)
            .map_err(|e| format!("Token tensor: {e}"))?;
        let attention_mask_tensor = Tensor::new(vec![attention_mask.clone()], &self.device)
            .map_err(|e| format!("Mask tensor: {e}"))?;
        let token_type_ids = token_ids.zeros_like()
            .map_err(|e| format!("Type IDs: {e}"))?;

        // Forward pass
        let output = self.model.forward(&token_ids, &token_type_ids, Some(&attention_mask_tensor))
            .map_err(|e| format!("Forward: {e}"))?;

        // Mean pooling: average all token embeddings (masked by attention)
        // output shape: [1, seq_len, hidden_dim]
        // attention_mask shape: [1, seq_len]
        let (_batch, seq_len, hidden_dim) = output.dims3()
            .map_err(|e| format!("Dims: {e}"))?;

        // Expand mask to [1, seq_len, hidden_dim]
        let mask = attention_mask_tensor
            .to_dtype(DType::F32)
            .map_err(|e| format!("Dtype: {e}"))?
            .unsqueeze(2)
            .map_err(|e| format!("Unsqueeze: {e}"))?
            .expand(&[1, seq_len, hidden_dim])
            .map_err(|e| format!("Expand: {e}"))?;

        let masked = (&output * &mask)
            .map_err(|e| format!("Mask multiply: {e}"))?;

        // Sum over seq_len dimension → [1, hidden_dim]
        let sum = masked.sum(1)
            .map_err(|e| format!("Sum: {e}"))?;
        let count = mask.sum(1)
            .map_err(|e| format!("Count: {e}"))?
            .clamp(1e-9, f64::MAX)
            .map_err(|e| format!("Clamp: {e}"))?;
        let pooled = (&sum / &count)
            .map_err(|e| format!("Divide: {e}"))?;

        // L2 normalize → [1, hidden_dim]
        let norm = pooled.sqr()
            .map_err(|e| format!("Sqr: {e}"))?
            .sum(1)
            .map_err(|e| format!("Sum norm: {e}"))?
            .sqrt()
            .map_err(|e| format!("Sqrt: {e}"))?
            .clamp(1e-12, f64::MAX)
            .map_err(|e| format!("Clamp norm: {e}"))?;
        let normalized = pooled.broadcast_div(&norm)
            .map_err(|e| format!("Normalize: {e}"))?;

        // To Vec<f32> → [hidden_dim]
        let vec: Vec<f32> = normalized.squeeze(0)
            .map_err(|e| format!("Squeeze: {e}"))?
            .to_vec1()
            .map_err(|e| format!("To vec: {e}"))?;

        Ok(vec)
    }

    /// Embed multiple texts (batched for efficiency)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() { return 0.0; }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
        dot / (norm_a * norm_b)
    }

    /// Output dimension
    pub fn dimension(&self) -> usize {
        384
    }
}

/// Quick embed function — uses the cached model
pub fn embed(text: &str) -> Result<Vec<f32>, String> {
    get_model()?.embed(text)
}

/// Quick similarity function
pub fn similarity(a: &str, b: &str) -> Result<f32, String> {
    let model = get_model()?;
    let va = model.embed(a)?;
    let vb = model.embed(b)?;
    Ok(EmbeddingModel::cosine_similarity(&va, &vb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loads_and_embeds() {
        let model = EmbeddingModel::load().expect("Model should load");
        let vec = model.embed("Hello World").expect("Should embed");
        assert_eq!(vec.len(), 384, "Should be 384-dim");

        // Check it's normalized (L2 norm ≈ 1.0)
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "Should be normalized, got {norm}");
    }

    #[test]
    fn semantic_similarity_works() {
        let model = EmbeddingModel::load().expect("Model should load");
        let dog = model.embed("The dog runs in the park").unwrap();
        let cat = model.embed("The cat plays in the garden").unwrap();
        let math = model.embed("Calculate the determinant of a matrix").unwrap();

        let sim_dog_cat = EmbeddingModel::cosine_similarity(&dog, &cat);
        let sim_dog_math = EmbeddingModel::cosine_similarity(&dog, &math);

        println!("dog-cat: {sim_dog_cat:.4}");
        println!("dog-math: {sim_dog_math:.4}");

        assert!(sim_dog_cat > sim_dog_math,
            "dog-cat ({sim_dog_cat}) should be > dog-math ({sim_dog_math})");
    }

    #[test]
    fn german_similarity() {
        let model = EmbeddingModel::load().expect("Model should load");
        let rust = model.embed("Rust ist eine Programmiersprache").unwrap();
        let python = model.embed("Python ist eine Programmiersprache").unwrap();
        let kochen = model.embed("Ich koche Pasta mit Tomaten").unwrap();

        let sim_rust_python = EmbeddingModel::cosine_similarity(&rust, &python);
        let sim_rust_kochen = EmbeddingModel::cosine_similarity(&rust, &kochen);

        println!("rust-python: {sim_rust_python:.4}");
        println!("rust-kochen: {sim_rust_kochen:.4}");

        assert!(sim_rust_python > sim_rust_kochen,
            "rust-python ({sim_rust_python}) should be > rust-kochen ({sim_rust_kochen})");
    }
}
pub mod embed_data;
pub mod vision;
pub mod vision_resnet;
