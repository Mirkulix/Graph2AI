use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub api_key: String,
    pub base_url: Option<String>,
    pub model: String,
    pub enabled: bool,
    pub tier: u8,
    pub cost_per_1k_tokens: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ProviderType {
    Groq,
    OpenAI,
    Anthropic,
    DeepSeek,
    Gemini,
    Ollama,
    OpenRouter,
    Mistral,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTemplate {
    pub id: &'static str,
    pub name: &'static str,
    pub provider_type: ProviderType,
    pub base_url: &'static str,
    pub models: Vec<ModelOption>,
    pub tier: u8,
    pub free: bool,
    pub description: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOption {
    pub id: &'static str,
    pub name: &'static str,
    pub cost_per_1k: f64,
    pub recommended: bool,
}

/// Pre-configured templates for popular providers
pub fn provider_templates() -> Vec<ProviderTemplate> {
    vec![
        ProviderTemplate {
            id: "groq",
            name: "Groq",
            provider_type: ProviderType::Groq,
            base_url: "https://api.groq.com/openai/v1",
            models: vec![
                ModelOption { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B", cost_per_1k: 0.0, recommended: true },
                ModelOption { id: "llama-3.1-8b-instant", name: "Llama 3.1 8B", cost_per_1k: 0.0, recommended: false },
                ModelOption { id: "gemma2-9b-it", name: "Gemma 2 9B", cost_per_1k: 0.0, recommended: false },
            ],
            tier: 2,
            free: true,
            description: "Schnellste Inference — 300 tok/s, kostenloser Tier",
        },
        ProviderTemplate {
            id: "deepseek",
            name: "DeepSeek",
            provider_type: ProviderType::DeepSeek,
            base_url: "https://api.deepseek.com/v1",
            models: vec![
                ModelOption { id: "deepseek-chat", name: "DeepSeek V3", cost_per_1k: 0.001, recommended: true },
                ModelOption { id: "deepseek-reasoner", name: "DeepSeek R1", cost_per_1k: 0.004, recommended: false },
            ],
            tier: 3,
            free: false,
            description: "Günstigster Paid-Provider — $0.001/1K Tokens",
        },
        ProviderTemplate {
            id: "openrouter",
            name: "OpenRouter",
            provider_type: ProviderType::OpenRouter,
            base_url: "https://openrouter.ai/api/v1",
            models: vec![
                ModelOption { id: "meta-llama/llama-3.3-70b-instruct:free", name: "Llama 3.3 70B (Free)", cost_per_1k: 0.0, recommended: true },
                ModelOption { id: "google/gemini-2.0-flash-exp:free", name: "Gemini 2.0 Flash (Free)", cost_per_1k: 0.0, recommended: false },
                ModelOption { id: "anthropic/claude-sonnet-4", name: "Claude Sonnet 4", cost_per_1k: 0.012, recommended: false },
            ],
            tier: 2,
            free: true,
            description: "Viele kostenlose Modelle — ein API Key für alles",
        },
        ProviderTemplate {
            id: "anthropic",
            name: "Anthropic (Claude)",
            provider_type: ProviderType::Anthropic,
            base_url: "https://api.anthropic.com/v1",
            models: vec![
                ModelOption { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", cost_per_1k: 0.012, recommended: true },
                ModelOption { id: "claude-haiku-4-5-20251001", name: "Claude Haiku 4.5", cost_per_1k: 0.003, recommended: false },
            ],
            tier: 3,
            free: false,
            description: "Höchste Qualität — für komplexe Reasoning-Tasks",
        },
        ProviderTemplate {
            id: "gemini",
            name: "Google Gemini",
            provider_type: ProviderType::Gemini,
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
            models: vec![
                ModelOption { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", cost_per_1k: 0.0, recommended: true },
                ModelOption { id: "gemini-2.5-pro-preview-06-05", name: "Gemini 2.5 Pro", cost_per_1k: 0.005, recommended: false },
            ],
            tier: 2,
            free: true,
            description: "Google AI — Flash ist kostenlos, Pro für komplexe Tasks",
        },
        ProviderTemplate {
            id: "ollama",
            name: "Ollama (Lokal)",
            provider_type: ProviderType::Ollama,
            base_url: "http://localhost:11434/v1",
            models: vec![
                ModelOption { id: "llama3.2:3b", name: "Llama 3.2 3B", cost_per_1k: 0.0, recommended: true },
                ModelOption { id: "qwen2.5:7b", name: "Qwen 2.5 7B", cost_per_1k: 0.0, recommended: false },
                ModelOption { id: "gemma2:9b", name: "Gemma 2 9B", cost_per_1k: 0.0, recommended: false },
            ],
            tier: 1,
            free: true,
            description: "Vollständig lokal — kein API Key nötig, $0 Kosten",
        },
        ProviderTemplate {
            id: "mistral",
            name: "Mistral AI",
            provider_type: ProviderType::Mistral,
            base_url: "https://api.mistral.ai/v1",
            models: vec![
                ModelOption { id: "mistral-small-latest", name: "Mistral Small", cost_per_1k: 0.001, recommended: true },
                ModelOption { id: "mistral-large-latest", name: "Mistral Large", cost_per_1k: 0.008, recommended: false },
            ],
            tier: 3,
            free: false,
            description: "Europäischer Provider — gut und günstig",
        },
        ProviderTemplate {
            id: "openai",
            name: "OpenAI",
            provider_type: ProviderType::OpenAI,
            base_url: "https://api.openai.com/v1",
            models: vec![
                ModelOption { id: "gpt-4o-mini", name: "GPT-4o Mini", cost_per_1k: 0.0006, recommended: true },
                ModelOption { id: "gpt-4o", name: "GPT-4o", cost_per_1k: 0.01, recommended: false },
            ],
            tier: 3,
            free: false,
            description: "OpenAI — breit unterstützt",
        },
    ]
}
