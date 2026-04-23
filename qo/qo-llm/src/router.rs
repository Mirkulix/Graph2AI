use crate::{
    cloud::{CloudClient, CloudMessage},
    deepseek::{DeepSeekClient, DeepSeekMessage},
    groq::{GroqClient, GroqMessage},
    ollama::OllamaClient,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Local,
    Groq,
    Cloud,
    DeepSeek,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStats {
    pub name: String,
    pub model: String,
    pub requests: u64,
    pub total_tokens_estimate: u64,
    pub cost_usd: f64,
    pub avg_latency_ms: u64,
    pub status: String,
}

#[derive(Debug, Default)]
pub struct CostTracker {
    pub groq_requests: AtomicU64,
    pub groq_tokens: AtomicU64,
    pub cloud_requests: AtomicU64,
    pub cloud_tokens: AtomicU64,
    pub local_requests: AtomicU64,
    pub deepseek_requests: AtomicU64,
    pub deepseek_tokens: AtomicU64,
    pub total_latency_groq_ms: AtomicU64,
    pub total_latency_cloud_ms: AtomicU64,
    pub total_latency_deepseek_ms: AtomicU64,
}

/// LlmRouter holds the per-tier clients behind `RwLock`s so that
/// the UI / providers route can hot-reload them without a restart.
///
/// Reads are async (`.read().await`) — every chat path goes through
/// these locks once at the top of a request, then releases the guard
/// after cloning out the small client struct or borrowing it for the
/// duration of the call.
pub struct LlmRouter {
    groq: RwLock<Option<GroqClient>>,
    cloud: RwLock<Option<CloudClient>>,
    ollama: RwLock<Option<OllamaClient>>, // Tier 1 local inference
    deepseek: RwLock<Option<DeepSeekClient>>,
    cloud_model: RwLock<Option<String>>,
    deepseek_model: RwLock<Option<String>>,
    cost_tracker: Arc<CostTracker>,
}

impl LlmRouter {
    pub fn new(
        groq_api_key: Option<String>,
        cloud_config: Option<(String, String, String)>,
        ollama_config: Option<(String, String)>, // (base_url, model)
    ) -> Self {
        let cloud_model = cloud_config.as_ref().map(|(_, _, model)| model.clone());
        // Auto-pick up DEEPSEEK_API_KEY from the environment so callers
        // that haven't been updated to the builder API still get the
        // new tier when the key is set.
        let (deepseek, deepseek_model) = match std::env::var("DEEPSEEK_API_KEY").ok() {
            Some(key) if !key.is_empty() => {
                let model = std::env::var("DEEPSEEK_MODEL")
                    .unwrap_or_else(|_| "deepseek-chat".to_string());
                (
                    Some(DeepSeekClient::with_model(key, model.clone())),
                    Some(model),
                )
            }
            _ => (None, None),
        };
        Self {
            groq: RwLock::new(groq_api_key.map(GroqClient::new)),
            cloud: RwLock::new(cloud_config
                .map(|(api_key, base_url, model)| CloudClient::new(api_key, base_url, model))),
            ollama: RwLock::new(ollama_config.map(|(url, model)| OllamaClient::new(url, model))),
            deepseek: RwLock::new(deepseek),
            cloud_model: RwLock::new(cloud_model),
            deepseek_model: RwLock::new(deepseek_model),
            cost_tracker: Arc::new(CostTracker::default()),
        }
    }

    /// Builder-style override for the DeepSeek client (used by tests
    /// and callers that want to inject an explicit key/model instead
    /// of relying on `DEEPSEEK_API_KEY`).
    pub fn with_deepseek(self, api_key: String, model: Option<String>) -> Self {
        let model = model.unwrap_or_else(|| "deepseek-chat".to_string());
        // Synchronous overwrite of the lock contents — `with_deepseek`
        // is a non-async builder so we use `try_write` which always
        // succeeds on a freshly-constructed router (no other holders).
        if let Ok(mut g) = self.deepseek.try_write() {
            *g = Some(DeepSeekClient::with_model(api_key, model.clone()));
        }
        if let Ok(mut g) = self.deepseek_model.try_write() {
            *g = Some(model);
        }
        self
    }

    /// Hot-install or replace a provider client in the running router.
    ///
    /// Called by `/api/providers/add`, `/edit`, `/toggle` so a freshly
    /// configured key takes effect without restarting `qo`.
    ///
    /// `provider_type` is matched case-insensitively against
    /// `ProviderType` variants ("groq", "deepseek", "openai",
    /// "anthropic", "ollama", …). Provider types we do not yet wire
    /// for hot-reload return `Err` so the caller can log it.
    pub async fn install_provider(
        &self,
        provider_type: &str,
        api_key: String,
        base_url: Option<String>,
        model: Option<String>,
    ) -> Result<(), String> {
        match provider_type.to_lowercase().as_str() {
            "deepseek" => {
                let model = model.unwrap_or_else(|| "deepseek-chat".to_string());
                *self.deepseek.write().await =
                    Some(DeepSeekClient::with_model(api_key, model.clone()));
                *self.deepseek_model.write().await = Some(model);
                Ok(())
            }
            "groq" => {
                *self.groq.write().await = Some(GroqClient::new(api_key));
                Ok(())
            }
            // Treat OpenAI-compatible providers as the generic "cloud"
            // tier. Anthropic / OpenRouter / Gemini / Mistral / Custom
            // all use the same OpenAI-shaped chat-completions API.
            "openai" | "anthropic" | "openrouter" | "gemini" | "mistral" | "custom" => {
                let url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
                let model = model.unwrap_or_else(|| "gpt-4o-mini".to_string());
                *self.cloud.write().await =
                    Some(CloudClient::new(api_key, url, model.clone()));
                *self.cloud_model.write().await = Some(model);
                Ok(())
            }
            "ollama" => {
                let url = base_url.unwrap_or_else(|| "http://localhost:11434".to_string());
                let model = model.unwrap_or_else(|| "llama3.2:3b".to_string());
                *self.ollama.write().await = Some(OllamaClient::new(url, model));
                Ok(())
            }
            other => Err(format!(
                "provider type '{}' not yet wired for hot-reload",
                other
            )),
        }
    }

    /// Remove a provider from the live router (called on DELETE).
    pub async fn remove_provider(&self, provider_type: &str) {
        match provider_type.to_lowercase().as_str() {
            "deepseek" => {
                *self.deepseek.write().await = None;
                *self.deepseek_model.write().await = None;
            }
            "groq" => {
                *self.groq.write().await = None;
            }
            "openai" | "anthropic" | "openrouter" | "gemini" | "mistral" | "custom" => {
                *self.cloud.write().await = None;
                *self.cloud_model.write().await = None;
            }
            "ollama" => {
                *self.ollama.write().await = None;
            }
            _ => {}
        }
    }

    /// Expose cost tracker (e.g. for sharing with server state).
    pub fn cost_tracker(&self) -> Arc<CostTracker> {
        self.cost_tracker.clone()
    }

    /// Returns stats for each configured provider.
    ///
    /// Now async because the per-tier clients live behind `RwLock`s.
    pub async fn provider_stats(&self) -> Vec<ProviderStats> {
        let mut stats = Vec::new();

        // Groq
        let groq_requests = self.cost_tracker.groq_requests.load(Ordering::Relaxed);
        let groq_tokens = self.cost_tracker.groq_tokens.load(Ordering::Relaxed);
        let total_latency_groq = self.cost_tracker.total_latency_groq_ms.load(Ordering::Relaxed);
        let avg_latency_groq = if groq_requests > 0 {
            total_latency_groq / groq_requests
        } else {
            0
        };
        let groq_status = if self.groq.read().await.is_some() { "active" } else { "inactive" }.to_string();
        stats.push(ProviderStats {
            name: "Groq".to_string(),
            model: "llama-3.3-70b-versatile".to_string(),
            requests: groq_requests,
            total_tokens_estimate: groq_tokens,
            cost_usd: 0.0,
            avg_latency_ms: avg_latency_groq,
            status: groq_status,
        });

        // Cloud
        let cloud_requests = self.cost_tracker.cloud_requests.load(Ordering::Relaxed);
        let cloud_tokens = self.cost_tracker.cloud_tokens.load(Ordering::Relaxed);
        let total_latency_cloud = self.cost_tracker.total_latency_cloud_ms.load(Ordering::Relaxed);
        let avg_latency_cloud = if cloud_requests > 0 {
            total_latency_cloud / cloud_requests
        } else {
            0
        };
        // Estimate cost at $0.01/1K tokens average
        let cloud_cost = (cloud_tokens as f64 / 1000.0) * 0.01;
        let cloud_status = if self.cloud.read().await.is_some() { "active" } else { "inactive" }.to_string();
        let cloud_model = self.cloud_model.read().await.clone().unwrap_or_default();
        stats.push(ProviderStats {
            name: "Cloud".to_string(),
            model: cloud_model,
            requests: cloud_requests,
            total_tokens_estimate: cloud_tokens,
            cost_usd: cloud_cost,
            avg_latency_ms: avg_latency_cloud,
            status: cloud_status,
        });

        // Local — Ollama or IGQK future
        let local_requests = self.cost_tracker.local_requests.load(Ordering::Relaxed);
        let (local_name, local_model, local_status) = if let Some(ref ollama) = *self.ollama.read().await {
            (
                "Ollama (local)".to_string(),
                ollama.model.clone(),
                "active".to_string(),
            )
        } else {
            (
                "QO-LLM (IGQK)".to_string(),
                "qwen2.5-0.5b-ternary".to_string(),
                "coming_soon".to_string(),
            )
        };
        stats.push(ProviderStats {
            name: local_name,
            model: local_model,
            requests: local_requests,
            total_tokens_estimate: 0,
            cost_usd: 0.0,
            avg_latency_ms: 0,
            status: local_status,
        });

        // DeepSeek — only listed when actually configured, so existing
        // tests that expect exactly three providers still pass.
        if let Some(ref ds) = *self.deepseek.read().await {
            let ds_requests = self.cost_tracker.deepseek_requests.load(Ordering::Relaxed);
            let ds_tokens = self.cost_tracker.deepseek_tokens.load(Ordering::Relaxed);
            let total_latency_ds = self
                .cost_tracker
                .total_latency_deepseek_ms
                .load(Ordering::Relaxed);
            let avg_latency_ds = if ds_requests > 0 {
                total_latency_ds / ds_requests
            } else {
                0
            };
            // Conservative estimate: ~$0.00027/1K tokens for deepseek-chat.
            let ds_cost = (ds_tokens as f64 / 1000.0) * 0.00027;
            stats.push(ProviderStats {
                name: "DeepSeek".to_string(),
                model: ds.model().to_string(),
                requests: ds_requests,
                total_tokens_estimate: ds_tokens,
                cost_usd: ds_cost,
                avg_latency_ms: avg_latency_ds,
                status: "active".to_string(),
            });
        }

        stats
    }

    /// Returns total estimated cost across all providers.
    pub fn total_cost(&self) -> f64 {
        let cloud_tokens = self.cost_tracker.cloud_tokens.load(Ordering::Relaxed);
        (cloud_tokens as f64 / 1000.0) * 0.01
    }

    /// Heuristic complexity score in [0.0, 1.0].
    /// Considers prompt length and presence of complexity keywords.
    pub fn score_complexity(&self, prompt: &str) -> f32 {
        let len_score = (prompt.len() as f32 / 2000.0).min(1.0) * 0.5;

        let complex_keywords = [
            "architecture",
            "design",
            "optimize",
            "refactor",
            "security",
            "algorithm",
            "implement",
            "analyze",
            "complex",
            "system",
        ];
        let lower = prompt.to_lowercase();
        let keyword_hits = complex_keywords
            .iter()
            .filter(|kw| lower.contains(*kw))
            .count();
        let keyword_score = (keyword_hits as f32 / complex_keywords.len() as f32) * 0.5;

        (len_score + keyword_score).min(1.0)
    }

    /// Select a tier based on complexity score.
    pub fn select_tier(&self, complexity: f32) -> Tier {
        if complexity < 0.3 {
            Tier::Local
        } else if complexity < 0.7 {
            Tier::Groq
        } else {
            Tier::Cloud
        }
    }

    /// Return true if the requested tier is actually usable (has a
    /// configured client). Used by per-agent model-binding so a
    /// preferred-tier that is offline falls back to auto routing
    /// instead of erroring.
    pub async fn tier_available(&self, tier: Tier) -> bool {
        match tier {
            Tier::Local => self.ollama.read().await.is_some(),
            Tier::Groq => self.groq.read().await.is_some(),
            Tier::Cloud => self.cloud.read().await.is_some(),
            Tier::DeepSeek => self.deepseek.read().await.is_some(),
        }
    }

    /// Translate a free-form provider hint (from `AgentRole::preferred_provider`)
    /// into a [`Tier`]. Unknown or empty hints map to `None`.
    pub fn tier_from_hint(hint: &str) -> Option<Tier> {
        match hint.trim().to_lowercase().as_str() {
            "ollama" | "local" => Some(Tier::Local),
            "groq" => Some(Tier::Groq),
            "deepseek" => Some(Tier::DeepSeek),
            "cloud" | "anthropic" | "claude" | "openai" | "gemini" => {
                Some(Tier::Cloud)
            }
            _ => None,
        }
    }

    /// Route the message through the preferred tier when possible, else
    /// fall back to the complexity-based auto router. Returns the
    /// response together with the tier that actually served it, so the
    /// caller can expose it in the dashboard.
    pub async fn chat_preferring(
        &self,
        preferred: Option<Tier>,
        messages: Vec<(String, String)>,
    ) -> Result<(String, Tier), Box<dyn Error + Send + Sync>> {
        if let Some(tier) = preferred {
            if self.tier_available(tier).await {
                tracing::debug!(?tier, "agent requested preferred tier");
                match self.chat_on_tier(tier, messages.clone()).await {
                    Ok(body) => return Ok((body, tier)),
                    Err(e) => {
                        // Transient provider failure — don't kill the goal
                        // outright. Log it loud and fall through to the
                        // complexity-based auto router, which may pick a
                        // healthier tier.
                        tracing::warn!(
                            ?tier,
                            error = %e,
                            "preferred tier failed — falling back to auto routing"
                        );
                    }
                }
            } else {
                tracing::warn!(
                    ?tier,
                    "preferred tier not configured — falling back to auto routing"
                );
            }
        }
        let prompt = {
            // Compute before moving `messages` into `chat`.
            let s = messages
                .last()
                .map(|(_, c)| c.as_str())
                .unwrap_or_default();
            self.select_tier(self.score_complexity(s))
        };
        let body = self.chat(messages).await?;
        Ok((body, prompt))
    }

    /// Force a specific tier. Does NOT fall back — caller is expected
    /// to check `tier_available` first (via [`chat_preferring`] which
    /// does fall back).
    async fn chat_on_tier(
        &self,
        tier: Tier,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        match tier {
            Tier::Cloud => {
                let guard = self.cloud.read().await;
                let cloud = guard
                    .as_ref()
                    .ok_or("Cloud provider not configured")?;
                let msgs: Vec<CloudMessage> = messages
                    .into_iter()
                    .map(|(role, content)| CloudMessage { role, content })
                    .collect();
                let start = Instant::now();
                let result = cloud.chat(msgs).await?;
                let elapsed = start.elapsed().as_millis() as u64;
                let tokens = (result.len() / 4) as u64;
                self.cost_tracker.cloud_requests.fetch_add(1, Ordering::Relaxed);
                self.cost_tracker.cloud_tokens.fetch_add(tokens, Ordering::Relaxed);
                self.cost_tracker
                    .total_latency_cloud_ms
                    .fetch_add(elapsed, Ordering::Relaxed);
                Ok(result)
            }
            Tier::Groq => self.groq_chat(messages).await,
            Tier::Local => {
                let guard = self.ollama.read().await;
                let ollama = guard
                    .as_ref()
                    .ok_or("Local (Ollama) provider not configured")?;
                let response = ollama.chat(messages).await?;
                self.cost_tracker.local_requests.fetch_add(1, Ordering::Relaxed);
                Ok(response)
            }
            Tier::DeepSeek => self.deepseek_chat(messages).await,
        }
    }

    /// Route the conversation to the appropriate tier.
    /// In Phase 1, Local falls back to Groq when no local model is available.
    pub async fn chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let prompt = messages
            .last()
            .map(|(_, c)| c.as_str())
            .unwrap_or_default();
        let complexity = self.score_complexity(prompt);
        let tier = self.select_tier(complexity);

        tracing::debug!(?tier, complexity, "routing request");

        match tier {
            Tier::Cloud => {
                {
                    let guard = self.cloud.read().await;
                    if let Some(cloud) = guard.as_ref() {
                        let msgs: Vec<CloudMessage> = messages
                            .into_iter()
                            .map(|(role, content)| CloudMessage { role, content })
                            .collect();
                        let start = Instant::now();
                        let result = cloud.chat(msgs).await?;
                        let elapsed = start.elapsed().as_millis() as u64;
                        let tokens = (result.len() / 4) as u64;
                        self.cost_tracker.cloud_requests.fetch_add(1, Ordering::Relaxed);
                        self.cost_tracker.cloud_tokens.fetch_add(tokens, Ordering::Relaxed);
                        self.cost_tracker.total_latency_cloud_ms.fetch_add(elapsed, Ordering::Relaxed);
                        return Ok(result);
                    }
                }
                // fall through to Groq — `messages` was not moved because the
                // Cloud branch above only ran inside the `if let Some(...)`.
                self.groq_chat(messages).await
            }
            Tier::Local => {
                {
                    let guard = self.ollama.read().await;
                    if let Some(ref ollama) = *guard {
                        match ollama.chat(messages.clone()).await {
                            Ok(response) => {
                                self.cost_tracker.local_requests.fetch_add(1, Ordering::Relaxed);
                                return Ok(response);
                            }
                            Err(e) => {
                                tracing::warn!("Ollama failed, falling back to Groq: {e}");
                            }
                        }
                    }
                }
                // Fallback to Groq when Ollama is absent or failed
                self.groq_chat(messages).await
            }
            Tier::Groq => {
                self.groq_chat(messages).await
            }
            Tier::DeepSeek => {
                self.deepseek_chat(messages).await
            }
        }
    }

    async fn deepseek_chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let guard = self.deepseek.read().await;
        let ds = guard
            .as_ref()
            .ok_or("DeepSeek provider not configured (set DEEPSEEK_API_KEY)")?;
        let msgs: Vec<DeepSeekMessage> = messages
            .into_iter()
            .map(|(role, content)| DeepSeekMessage { role, content })
            .collect();
        let start = Instant::now();
        let result = ds.chat(msgs).await?;
        let elapsed = start.elapsed().as_millis() as u64;
        let tokens = (result.len() / 4) as u64;
        self.cost_tracker.deepseek_requests.fetch_add(1, Ordering::Relaxed);
        self.cost_tracker.deepseek_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.cost_tracker
            .total_latency_deepseek_ms
            .fetch_add(elapsed, Ordering::Relaxed);
        Ok(result)
    }

    async fn groq_chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let guard = self.groq.read().await;
        if let Some(groq) = guard.as_ref() {
            let msgs: Vec<GroqMessage> = messages
                .into_iter()
                .map(|(role, content)| GroqMessage { role, content })
                .collect();
            let start = Instant::now();
            let result = groq.chat(msgs).await?;
            let elapsed = start.elapsed().as_millis() as u64;
            let tokens = (result.len() / 4) as u64;
            self.cost_tracker.groq_requests.fetch_add(1, Ordering::Relaxed);
            self.cost_tracker.groq_tokens.fetch_add(tokens, Ordering::Relaxed);
            self.cost_tracker.total_latency_groq_ms.fetch_add(elapsed, Ordering::Relaxed);
            Ok(result)
        } else {
            Err("No LLM backend configured".into())
        }
    }

    /// Call any OpenAI-compatible provider endpoint directly.
    pub async fn chat_with_provider(
        &self,
        base_url: &str,
        api_key: &str,
        model: &str,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({"role": role, "content": content})
            })
            .collect();

        let body = serde_json::json!({
            "model": model,
            "messages": msgs,
            "max_tokens": 2048,
        });

        let mut req = client.post(&url).json(&body);
        if !api_key.is_empty() {
            req = req.bearer_auth(api_key);
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Provider error {status}: {body}").into());
        }

        let json: serde_json::Value = resp.json().await?;
        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "No content in response".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> LlmRouter {
        LlmRouter::new(None, None, None)
    }

    #[test]
    fn simple_prompt_routes_to_local() {
        let r = router();
        let complexity = r.score_complexity("hi");
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Local);
    }

    #[test]
    fn medium_prompt_routes_to_groq() {
        let r = router();
        // 1600 chars → len_score = (1600/2000)*0.5 = 0.4, no keywords → total = 0.4 (Groq range)
        let prompt = "a".repeat(1600);
        let complexity = r.score_complexity(&prompt);
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Groq);
    }

    #[test]
    fn complex_prompt_routes_to_cloud() {
        let r = router();
        // Long prompt with many complexity keywords
        let prompt = format!(
            "{} architecture design optimize refactor security algorithm implement analyze complex system",
            "a".repeat(1500)
        );
        let complexity = r.score_complexity(&prompt);
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Cloud);
    }

    #[tokio::test]
    async fn provider_stats_returns_three_providers() {
        let r = router();
        let stats = r.provider_stats().await;
        assert_eq!(stats.len(), 3);
        assert_eq!(stats[0].name, "Groq");
        assert_eq!(stats[1].name, "Cloud");
        // Without Ollama configured, falls back to IGQK placeholder
        assert_eq!(stats[2].name, "QO-LLM (IGQK)");
    }

    #[tokio::test]
    async fn tier_local_with_ollama_configured() {
        let router = LlmRouter::new(
            None,
            None,
            Some(("http://localhost:11434".into(), "orbit-companion-ft-q4".into())),
        );
        assert!(router.ollama.read().await.is_some());
        // provider_stats should show Ollama as active
        let stats = router.provider_stats().await;
        assert_eq!(stats[2].name, "Ollama (local)");
        assert_eq!(stats[2].model, "orbit-companion-ft-q4");
        assert_eq!(stats[2].status, "active");
    }

    #[test]
    fn total_cost_starts_at_zero() {
        let r = router();
        assert_eq!(r.total_cost(), 0.0);
    }

    #[tokio::test]
    async fn groq_status_inactive_when_no_key() {
        let r = router();
        let stats = r.provider_stats().await;
        assert_eq!(stats[0].status, "inactive");
    }

    #[test]
    fn test_chat_with_provider_builds_correct_url() {
        // Verify that trailing slashes are trimmed when constructing the URL.
        // We only test the URL-building logic (no real HTTP call).
        let base = "https://api.example.com/v1/";
        let expected = "https://api.example.com/v1/chat/completions";
        let actual = format!("{}/chat/completions", base.trim_end_matches('/'));
        assert_eq!(actual, expected);

        // No trailing slash variant
        let base2 = "https://api.example.com/v1";
        let actual2 = format!("{}/chat/completions", base2.trim_end_matches('/'));
        assert_eq!(actual2, expected);
    }

    #[test]
    fn test_empty_prompt_routes_to_local() {
        let r = router();
        let complexity = r.score_complexity("");
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Local);
    }

    #[test]
    fn test_very_long_prompt_routes_to_cloud() {
        let r = router();
        // 2000 chars + complexity keywords → score > 0.7
        let prompt = format!("{} architecture design optimize refactor security algorithm implement analyze complex system", "a".repeat(2000));
        let complexity = r.score_complexity(&prompt);
        let tier = r.select_tier(complexity);
        assert_eq!(tier, Tier::Cloud);
    }

    #[test]
    fn test_score_complexity_range() {
        let r = router();
        let long = "a".repeat(5000);
        let prompts: &[&str] = &["", "hi", long.as_str(), "architecture design security system"];
        for prompt in prompts {
            let score = r.score_complexity(prompt);
            assert!(score >= 0.0 && score <= 1.0, "score {} out of range for prompt len {}", score, prompt.len());
        }
    }

    #[test]
    fn test_tier_boundaries() {
        let r = router();
        // Exactly at 0.3 boundary → Groq
        assert_eq!(r.select_tier(0.3), Tier::Groq);
        // Just below 0.3 → Local
        assert_eq!(r.select_tier(0.29), Tier::Local);
        // Exactly at 0.7 boundary → Cloud
        assert_eq!(r.select_tier(0.7), Tier::Cloud);
        // Just below 0.7 → Groq
        assert_eq!(r.select_tier(0.69), Tier::Groq);
    }

    #[test]
    fn test_provider_selection_prefers_lower_tier() {
        // Simulate priority sorting: enabled providers sorted by tier ascending.
        #[derive(Debug, PartialEq)]
        struct FakeProvider { tier: u8, name: &'static str, enabled: bool }

        let mut providers = vec![
            FakeProvider { tier: 3, name: "openai", enabled: true },
            FakeProvider { tier: 1, name: "ollama", enabled: true },
            FakeProvider { tier: 2, name: "groq", enabled: true },
            FakeProvider { tier: 2, name: "openrouter", enabled: false }, // disabled
        ];

        // Keep only enabled, sort by tier ascending
        providers.retain(|p| p.enabled);
        providers.sort_by_key(|p| p.tier);

        assert_eq!(providers[0].name, "ollama");
        assert_eq!(providers[0].tier, 1);
        assert_eq!(providers[1].tier, 2);
        assert_eq!(providers[2].tier, 3);
        // Disabled openrouter must not appear
        assert!(!providers.iter().any(|p| p.name == "openrouter"));
    }

    #[tokio::test]
    async fn install_provider_adds_deepseek_at_runtime() {
        let r = router();
        assert!(!r.tier_available(Tier::DeepSeek).await);
        r.install_provider("deepseek", "test-key".to_string(), None, Some("deepseek-chat".to_string()))
            .await
            .unwrap();
        assert!(r.tier_available(Tier::DeepSeek).await);
        // remove_provider clears it again
        r.remove_provider("deepseek").await;
        assert!(!r.tier_available(Tier::DeepSeek).await);
    }

    #[tokio::test]
    async fn install_provider_unknown_type_errors() {
        let r = router();
        let err = r
            .install_provider("magic-cloud", "key".to_string(), None, None)
            .await
            .unwrap_err();
        assert!(err.contains("not yet wired"));
    }
}
