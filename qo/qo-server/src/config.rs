use qo_agents::AgentRole;
use qo_llm::Tier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct LlmRoutingConfig {
    pub routes: HashMap<AgentRole, Tier>,
    pub fallback: Tier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawLlmRoutingConfig {
    #[serde(default)]
    routes: HashMap<AgentRole, String>,
    #[serde(default = "default_fallback_name")]
    fallback: String,
}

fn default_fallback() -> Tier {
    Tier::Groq
}

fn default_fallback_name() -> String {
    tier_name(default_fallback()).to_string()
}

fn tier_name(tier: Tier) -> &'static str {
    match tier {
        Tier::Local => "local",
        Tier::Groq => "groq",
        Tier::Cloud => "cloud",
    }
}

fn parse_tier(name: &str) -> Option<Tier> {
    match name.trim().to_ascii_lowercase().as_str() {
        "local" | "ollama" => Some(Tier::Local),
        "groq" => Some(Tier::Groq),
        "cloud" | "deepseek" | "anthropic" | "claude" | "openai" | "gemini" => {
            Some(Tier::Cloud)
        }
        _ => None,
    }
}

impl Default for LlmRoutingConfig {
    fn default() -> Self {
        let mut routes = HashMap::new();
        // Match the original defaults from AgentRole::preferred_provider()
        routes.insert(AgentRole::Ceo, Tier::Cloud);
        routes.insert(AgentRole::Researcher, Tier::Cloud);
        routes.insert(AgentRole::Developer, Tier::Cloud);
        routes.insert(AgentRole::Strategist, Tier::Cloud);
        routes.insert(AgentRole::Artisan, Tier::Cloud);
        routes.insert(AgentRole::Guardian, Tier::Groq);

        Self {
            routes,
            fallback: Tier::Groq,
        }
    }
}

impl LlmRoutingConfig {
    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        if path.as_ref().exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(raw) = toml::from_str::<RawLlmRoutingConfig>(&contents) {
                    let routes = raw
                        .routes
                        .into_iter()
                        .filter_map(|(role, tier)| parse_tier(&tier).map(|parsed| (role, parsed)))
                        .collect();

                    return Self {
                        routes,
                        fallback: parse_tier(&raw.fallback).unwrap_or_else(default_fallback),
                    };
                }
            }
        }
        Self::default()
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let raw = RawLlmRoutingConfig {
            routes: self
                .routes
                .iter()
                .map(|(role, tier)| (*role, tier_name(*tier).to_string()))
                .collect(),
            fallback: tier_name(self.fallback).to_string(),
        };
        let contents = toml::to_string_pretty(&raw).map_err(|e| e.to_string())?;
        if let Some(parent) = path.as_ref().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, contents).map_err(|e| e.to_string())
    }

    pub fn get_tier(&self, role: &AgentRole) -> Tier {
        self.routes.get(role).copied().unwrap_or(self.fallback)
    }
}
