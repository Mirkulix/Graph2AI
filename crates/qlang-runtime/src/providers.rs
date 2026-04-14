//! QLANG Provider Registry — discovers and manages LLM providers.
//!
//! This module provides a unified interface to multiple LLM backends
//! (local Ollama, OpenAI, Anthropic, etc.) with automatic discovery,
//! cost tracking, and data-class-aware routing.

use std::collections::HashMap;
use std::time::Instant;
use std::path::Path;
use std::fs;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Core enums
// ---------------------------------------------------------------------------

/// Security classification for data being sent to a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataClass {
    /// Public data — can be sent anywhere
    Public,
    /// Internal data — prefer local, allow trusted cloud
    Internal,
    /// Confidential — local only
    Confidential,
}

/// What a model can do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    TextGeneration,
    CodeGeneration,
    Summarization,
    Embedding,
    ImageGeneration,
    Classification,
    Custom(String),
}

/// How much we trust a provider with sensitive data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// Runs on user's machine
    Local,
    /// Cloud provider with data-processing agreement
    Trusted,
    /// Public cloud, no special agreement
    Untrusted,
}

/// Preference for model selection.
#[derive(Debug, Clone)]
pub enum SelectionPreference {
    /// Prefer a local model
    Local,
    /// Pick the cheapest that can do the job
    Cheapest,
    /// Pick the fastest
    Fastest,
    /// Pick the highest quality
    BestQuality,
}

// ---------------------------------------------------------------------------
// Model / Provider config
// ---------------------------------------------------------------------------

/// Configuration for a single model exposed by a provider.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub display_name: String,
    pub capabilities: Vec<Capability>,
    pub cost_per_million_tokens: f64,
    pub context_window: usize,
    pub quality_score: f32, // 0.0-1.0 subjective quality ranking
}

/// A provider that can serve LLM requests.
#[derive(Debug, Clone)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub trust_level: TrustLevel,
    pub models: Vec<ModelConfig>,
    pub base_url: String,
    pub api_key: Option<String>,
    pub available: bool,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// Response from an LLM call.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub provider_id: String,
    pub model_id: String,
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cost: f64,
    pub latency_ms: u64,
}

// ---------------------------------------------------------------------------
// Cost tracking
// ---------------------------------------------------------------------------

/// Tracks cumulative spending across providers.
#[derive(Debug, Clone)]
pub struct CostTracker {
    pub total_spent: f64,
    pub budget_limit: f64,
    pub by_provider: HashMap<String, f64>,
    pub request_count: usize,
}

impl CostTracker {
    pub fn new(budget: f64) -> Self {
        Self {
            total_spent: 0.0,
            budget_limit: budget,
            by_provider: HashMap::new(),
            request_count: 0,
        }
    }

    pub fn record(&mut self, provider_id: &str, cost: f64) {
        self.total_spent += cost;
        *self.by_provider.entry(provider_id.to_string()).or_insert(0.0) += cost;
        self.request_count += 1;
    }

    pub fn remaining(&self) -> f64 {
        (self.budget_limit - self.total_spent).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Central registry that discovers providers and routes requests.
pub struct ProviderRegistry {
    pub providers: Vec<Provider>,
    pub cost_tracker: CostTracker,
}

impl ProviderRegistry {
    pub fn new(budget: f64) -> Self {
        Self {
            providers: Vec::new(),
            cost_tracker: CostTracker::new(budget),
        }
    }

    /// Discover all available providers (Ollama, env-configured cloud, etc.)
    pub fn discover_all(&mut self) {
        // Always try local Ollama
        self.discover_ollama();

        // Cloud providers from environment
        self.discover_openai();
        self.discover_anthropic();
        self.discover_google();
        self.discover_azure_openai();
        self.discover_cohere();
        self.discover_mistral();
        self.discover_groq();
        self.discover_huggingface();
        self.discover_together();
        self.discover_perplexity();
        self.discover_bedrock();
        self.discover_xai();
    }

    fn discover_ollama(&mut self) {
        let client = crate::ollama::OllamaClient::from_env();
        match client.list_models() {
            Ok(models) => {
                let model_configs: Vec<ModelConfig> = models
                    .iter()
                    .map(|m| ModelConfig {
                        model_id: m.clone(),
                        display_name: m.clone(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 0.0, // local = free
                        context_window: 8192,
                        quality_score: 0.6,
                    })
                    .collect();

                if !model_configs.is_empty() {
                    self.providers.push(Provider {
                        id: "ollama".into(),
                        name: "Ollama (local)".into(),
                        trust_level: TrustLevel::Local,
                        models: model_configs,
                        base_url: format!(
                            "http://{}:{}",
                            std::env::var("QLANG_OLLAMA_HOST")
                                .unwrap_or_else(|_| "127.0.0.1".into()),
                            std::env::var("QLANG_OLLAMA_PORT")
                                .unwrap_or_else(|_| "11434".into()),
                        ),
                        api_key: None,
                        available: true,
                    });
                }
            }
            Err(_) => {
                eprintln!("[providers] Ollama not available");
            }
        }
    }

    fn discover_openai(&mut self) {
        let key = std::env::var("OPENAI_API_KEY").ok()
            .or_else(|| read_api_key_from_config("openai"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "openai".into(),
                name: "OpenAI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "gpt-4o".into(),
                        display_name: "GPT-4o".into(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 5.0,
                        context_window: 128_000,
                        quality_score: 0.95,
                    },
                    ModelConfig {
                        model_id: "gpt-4o-mini".into(),
                        display_name: "GPT-4o Mini".into(),
                        capabilities: vec![
                            Capability::TextGeneration,
                            Capability::CodeGeneration,
                            Capability::Summarization,
                        ],
                        cost_per_million_tokens: 0.15,
                        context_window: 128_000,
                        quality_score: 0.80,
                    },
                ],
                base_url: "https://api.openai.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_anthropic(&mut self) {
        let key = std::env::var("ANTHROPIC_API_KEY").ok()
            .or_else(|| read_api_key_from_config("anthropic"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "anthropic".into(),
                name: "Anthropic".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![ModelConfig {
                    model_id: "claude-sonnet-4-20250514".into(),
                    display_name: "Claude Sonnet 4".into(),
                    capabilities: vec![
                        Capability::TextGeneration,
                        Capability::CodeGeneration,
                        Capability::Summarization,
                    ],
                    cost_per_million_tokens: 3.0,
                    context_window: 200_000,
                    quality_score: 0.95,
                }],
                base_url: "https://api.anthropic.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_google(&mut self) {
        let key = std::env::var("GOOGLE_API_KEY").ok()
            .or_else(|| read_api_key_from_config("google"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "google".into(),
                name: "Google Gemini".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "gemini-1.5-pro".into(),
                        display_name: "Gemini 1.5 Pro".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 5.0,
                        context_window: 1_000_000,
                        quality_score: 0.90,
                    },
                    ModelConfig {
                        model_id: "gemini-1.5-flash".into(),
                        display_name: "Gemini 1.5 Flash".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 0.5,
                        context_window: 1_000_000,
                        quality_score: 0.80,
                    },
                ],
                base_url: "https://generativelanguage.googleapis.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_azure_openai(&mut self) {
        let key = std::env::var("AZURE_OPENAI_API_KEY").ok()
            .or_else(|| read_api_key_from_config("azure-openai"));
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").ok();
        if let (Some(key), Some(ep)) = (key, endpoint) {
            self.providers.push(Provider {
                id: "azure-openai".into(),
                name: "Azure OpenAI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "gpt-4o".into(),
                        display_name: "GPT-4o".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 5.0,
                        context_window: 128_000,
                        quality_score: 0.95,
                    },
                    ModelConfig {
                        model_id: "gpt-4o-mini".into(),
                        display_name: "GPT-4o Mini".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 0.15,
                        context_window: 128_000,
                        quality_score: 0.80,
                    },
                ],
                base_url: ep,
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_cohere(&mut self) {
        let key = std::env::var("COHERE_API_KEY").ok()
            .or_else(|| read_api_key_from_config("cohere"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "cohere".into(),
                name: "Cohere".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "command".into(),
                        display_name: "Command".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::Summarization],
                        cost_per_million_tokens: 0.5,
                        context_window: 128_000,
                        quality_score: 0.85,
                    },
                    ModelConfig {
                        model_id: "command-light".into(),
                        display_name: "Command Light".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::Summarization],
                        cost_per_million_tokens: 0.1,
                        context_window: 128_000,
                        quality_score: 0.75,
                    },
                ],
                base_url: "https://api.cohere.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_mistral(&mut self) {
        let key = std::env::var("MISTRAL_API_KEY").ok()
            .or_else(|| read_api_key_from_config("mistral"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "mistral".into(),
                name: "Mistral AI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "mistral-large-latest".into(),
                        display_name: "Mistral Large".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 2.0,
                        context_window: 32_000,
                        quality_score: 0.90,
                    },
                    ModelConfig {
                        model_id: "mistral-small-latest".into(),
                        display_name: "Mistral Small".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::Summarization],
                        cost_per_million_tokens: 0.3,
                        context_window: 16_000,
                        quality_score: 0.80,
                    },
                ],
                base_url: "https://api.mistral.ai".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_groq(&mut self) {
        let key = std::env::var("GROQ_API_KEY").ok()
            .or_else(|| read_api_key_from_config("groq"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "groq".into(),
                name: "Groq".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "llama3-70b-8192".into(),
                        display_name: "Llama 3 70B".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration],
                        cost_per_million_tokens: 0.2,
                        context_window: 8_192,
                        quality_score: 0.88,
                    },
                    ModelConfig {
                        model_id: "mixtral-8x7b-32768".into(),
                        display_name: "Mixtral 8x7B".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration],
                        cost_per_million_tokens: 0.15,
                        context_window: 32_768,
                        quality_score: 0.85,
                    },
                ],
                base_url: "https://api.groq.com".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_huggingface(&mut self) {
        let key = std::env::var("HUGGINGFACE_API_KEY").ok()
            .or_else(|| read_api_key_from_config("huggingface"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "huggingface".into(),
                name: "HuggingFace Inference".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "meta-llama/Meta-Llama-3-8B-Instruct".into(),
                        display_name: "Llama 3 8B Instruct".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration],
                        cost_per_million_tokens: 0.0,
                        context_window: 8_192,
                        quality_score: 0.80,
                    },
                ],
                base_url: "https://api-inference.huggingface.co".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_together(&mut self) {
        let key = std::env::var("TOGETHER_API_KEY").ok()
            .or_else(|| read_api_key_from_config("together"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "together".into(),
                name: "Together AI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo".into(),
                        display_name: "Llama 3.1 70B Turbo".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration],
                        cost_per_million_tokens: 0.25,
                        context_window: 32_000,
                        quality_score: 0.88,
                    },
                ],
                base_url: "https://api.together.xyz".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_perplexity(&mut self) {
        let key = std::env::var("PERPLEXITY_API_KEY").ok()
            .or_else(|| read_api_key_from_config("perplexity"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "perplexity".into(),
                name: "Perplexity".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "llama-3.1-sonar-large-32k-online".into(),
                        display_name: "Sonar Large 32k Online".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::Summarization],
                        cost_per_million_tokens: 1.0,
                        context_window: 32_000,
                        quality_score: 0.88,
                    },
                ],
                base_url: "https://api.perplexity.ai".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    fn discover_bedrock(&mut self) {
        let region = std::env::var("AWS_REGION").ok();
        let access = std::env::var("AWS_ACCESS_KEY_ID").ok();
        let secret = std::env::var("AWS_SECRET_ACCESS_KEY").ok();
        if region.is_some() && access.is_some() && secret.is_some() {
            self.providers.push(Provider {
                id: "aws-bedrock".into(),
                name: "AWS Bedrock".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "anthropic.claude-3-sonnet-20240229-v1:0".into(),
                        display_name: "Claude 3 Sonnet (Bedrock)".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 3.0,
                        context_window: 200_000,
                        quality_score: 0.95,
                    },
                    ModelConfig {
                        model_id: "meta.llama3-70b-instruct-v1:0".into(),
                        display_name: "Llama 3 70B Instruct (Bedrock)".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration],
                        cost_per_million_tokens: 1.0,
                        context_window: 8_192,
                        quality_score: 0.88,
                    },
                ],
                base_url: "https://bedrock.amazonaws.com".into(),
                api_key: None,
                available: true,
            });
        }
    }

    fn discover_xai(&mut self) {
        let key = std::env::var("XAI_API_KEY").ok()
            .or_else(|| read_api_key_from_config("xai"));
        if let Some(key) = key {
            self.providers.push(Provider {
                id: "xai".into(),
                name: "xAI".into(),
                trust_level: TrustLevel::Trusted,
                models: vec![
                    ModelConfig {
                        model_id: "grok-2".into(),
                        display_name: "Grok‑2".into(),
                        capabilities: vec![Capability::TextGeneration, Capability::CodeGeneration, Capability::Summarization],
                        cost_per_million_tokens: 2.0,
                        context_window: 128_000,
                        quality_score: 0.90,
                    },
                ],
                base_url: "https://api.x.ai".into(),
                api_key: Some(key),
                available: true,
            });
        }
    }

    /// List all known models: (provider_id, model_id, display_name, cost)
    pub fn list_all_models(&self) -> Vec<(String, String, String, f64)> {
        let mut out = Vec::new();
        for p in &self.providers {
            for m in &p.models {
                out.push((
                    p.id.clone(),
                    m.model_id.clone(),
                    m.display_name.clone(),
                    m.cost_per_million_tokens,
                ));
            }
        }
        out
    }

    /// Select the best model matching requirements.
    pub fn select_model(
        &self,
        capability: &Capability,
        data_class: DataClass,
        pref: SelectionPreference,
    ) -> Option<(&Provider, &ModelConfig)> {
        let mut candidates: Vec<(&Provider, &ModelConfig)> = Vec::new();

        for provider in &self.providers {
            if !provider.available {
                continue;
            }
            // Data-class filter
            match data_class {
                DataClass::Confidential => {
                    if provider.trust_level != TrustLevel::Local {
                        continue;
                    }
                }
                DataClass::Internal => {
                    if provider.trust_level == TrustLevel::Untrusted {
                        continue;
                    }
                }
                DataClass::Public => {}
            }
            for model in &provider.models {
                if model.capabilities.contains(capability) {
                    candidates.push((provider, model));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        match pref {
            SelectionPreference::Local => candidates
                .iter()
                .find(|(p, _)| p.trust_level == TrustLevel::Local)
                .or(candidates.first())
                .copied(),
            SelectionPreference::Cheapest => {
                candidates.sort_by(|a, b| {
                    a.1.cost_per_million_tokens
                        .partial_cmp(&b.1.cost_per_million_tokens)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first().copied()
            }
            SelectionPreference::Fastest => {
                // Prefer local (no network latency), then cheapest (smaller = faster)
                candidates.sort_by(|a, b| {
                    let a_local = if a.0.trust_level == TrustLevel::Local {
                        0
                    } else {
                        1
                    };
                    let b_local = if b.0.trust_level == TrustLevel::Local {
                        0
                    } else {
                        1
                    };
                    a_local.cmp(&b_local).then(
                        a.1.cost_per_million_tokens
                            .partial_cmp(&b.1.cost_per_million_tokens)
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
                });
                candidates.first().copied()
            }
            SelectionPreference::BestQuality => {
                candidates.sort_by(|a, b| {
                    b.1.quality_score
                        .partial_cmp(&a.1.quality_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first().copied()
            }
        }
    }

    /// Generate text using the best matching model.
    pub fn generate(
        &mut self,
        prompt: &str,
        capability: &Capability,
        data_class: DataClass,
        pref: SelectionPreference,
        _system: Option<&str>,
    ) -> Result<LlmResponse, String> {
        let (provider, model) = self
            .select_model(capability, data_class, pref)
            .ok_or_else(|| "No suitable model found".to_string())?;

        let provider_id = provider.id.clone();
        let model_id = model.model_id.clone();
        let cost_per_m = model.cost_per_million_tokens;

        let start = Instant::now();

        // Route to the right backend
        let text = match provider_id.as_str() {
            "ollama" => {
                let client = crate::ollama::OllamaClient::from_env();
                client
                    .generate(&model_id, prompt, None)
                    .map_err(|e| format!("Ollama error: {}", e))?
            }
            "openai" => {
                if let Some(client) = crate::openai_client::OpenAiClient::from_env() {
                    client.generate(&model_id, prompt)
                        .map_err(|e| format!("OpenAI error: {}", e))?
                } else {
                    return Err("OpenAI API key not configured (set OPENAI_API_KEY)".into());
                }
            }
            "anthropic" => {
                if let Some(client) = crate::anthropic_client::AnthropicClient::from_env() {
                    let msgs = vec![crate::ollama::ChatMessage::user(prompt)];
                    client.chat(&model_id, &msgs)
                        .map_err(|e| format!("Anthropic error: {}", e))?
                } else {
                    return Err("Anthropic API key not configured (set ANTHROPIC_API_KEY)".into());
                }
            }
            "gemini" => {
                if let Some(client) = crate::gemini_client::GeminiClient::from_env() {
                    client.generate(&model_id, prompt)
                        .map_err(|e| format!("Gemini error: {}", e))?
                } else {
                    return Err("Gemini API key not configured (set GEMINI_API_KEY)".into());
                }
            }
            "groq" => {
                if let Some(client) = crate::groq_client::GroqClient::from_env() {
                    let msgs = vec![crate::ollama::ChatMessage::user(prompt)];
                    client.chat(&model_id, &msgs)
                        .map_err(|e| format!("Groq error: {}", e))?
                } else {
                    return Err("Groq API key not configured (set GROQ_API_KEY)".into());
                }
            }
            _ => {
                let client = crate::ollama::OllamaClient::from_env();
                match client.generate(&model_id, prompt, None) {
                    Ok(t) => t,
                    Err(_) => {
                        return Err(format!(
                            "Cloud provider {} not yet implemented for direct calls",
                            provider_id
                        ));
                    }
                }
            }
        };

        let latency_ms = start.elapsed().as_millis() as u64;

        // Rough token estimate (4 chars per token)
        let input_tokens = prompt.len() / 4;
        let output_tokens = text.len() / 4;
        let total_tokens = input_tokens + output_tokens;
        let cost = (total_tokens as f64 / 1_000_000.0) * cost_per_m;

        self.cost_tracker.record(&provider_id, cost);

        Ok(LlmResponse {
            provider_id,
            model_id,
            text,
            input_tokens,
            output_tokens,
            cost,
            latency_ms,
        })
    }

    /// Human-readable cost summary.
    pub fn cost_summary(&self) -> String {
        let ct = &self.cost_tracker;
        let mut s = format!(
            "Total: ${:.4} / ${:.2} ({} requests)",
            ct.total_spent, ct.budget_limit, ct.request_count
        );
        for (pid, cost) in &ct.by_provider {
            s.push_str(&format!("\n  {}: ${:.4}", pid, cost));
        }
        s
    }
}

/// Read an API key for a given provider from config file:
/// workspace_root/data/config/providers.json
fn read_api_key_from_config(provider_id: &str) -> Option<String> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest).parent()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."));
    let config_path = workspace_root.join("data").join("config").join("providers.json");
    let content = fs::read_to_string(&config_path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    json.get(provider_id)
        .and_then(|p| p.get("api_key"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}
