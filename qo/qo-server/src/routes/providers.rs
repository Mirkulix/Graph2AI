use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::AppState;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProvidersResponse {
    pub providers: Vec<qo_llm::ProviderStats>,
    pub total_cost_usd: f64,
    pub total_requests: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CostSummaryResponse {
    pub total_cost_usd: f64,
    pub cloud_cost_usd: f64,
    pub groq_cost_usd: f64,
    pub local_cost_usd: f64,
    pub total_requests: u64,
    pub savings_vs_all_cloud_usd: f64,
}

pub async fn list_providers(State(state): State<Arc<AppState>>) -> Json<ProvidersResponse> {
    let providers = state.llm.provider_stats();
    let total_cost = state.llm.total_cost();
    let total_requests: u64 = providers.iter().map(|p| p.requests).sum();

    Json(ProvidersResponse {
        providers,
        total_cost_usd: total_cost,
        total_requests,
    })
}

pub async fn cost_summary(State(state): State<Arc<AppState>>) -> Json<CostSummaryResponse> {
    let providers = state.llm.provider_stats();

    let groq_cost = providers
        .iter()
        .find(|p| p.name == "Groq")
        .map(|p| p.cost_usd)
        .unwrap_or(0.0);
    let cloud_cost = providers
        .iter()
        .find(|p| p.name == "Cloud")
        .map(|p| p.cost_usd)
        .unwrap_or(0.0);
    let local_cost = 0.0f64;

    let total_cost = groq_cost + cloud_cost + local_cost;
    let total_requests: u64 = providers.iter().map(|p| p.requests).sum();

    // Savings: what would have been spent if all requests used Claude at $0.01/1K tokens avg
    // Estimate 300 tokens per request on average
    let total_tokens_estimate: u64 = providers.iter().map(|p| p.total_tokens_estimate).sum();
    let hypothetical_cloud_cost = (total_tokens_estimate as f64 / 1000.0) * 0.01;
    let savings = hypothetical_cloud_cost - total_cost;

    Json(CostSummaryResponse {
        total_cost_usd: total_cost,
        cloud_cost_usd: cloud_cost,
        groq_cost_usd: groq_cost,
        local_cost_usd: local_cost,
        total_requests,
        savings_vs_all_cloud_usd: savings.max(0.0),
    })
}

// ─── Provider Management ───────────────────────────────────────────────────

pub async fn list_templates() -> Json<Vec<qo_llm::config::ProviderTemplate>> {
    Json(qo_llm::provider_templates())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfiguredProviderResponse {
    pub id: String,
    pub name: String,
    pub provider_type: qo_llm::ProviderType,
    pub model: String,
    pub base_url: Option<String>,
    pub enabled: bool,
    pub tier: u8,
    pub cost_per_1k_tokens: f64,
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub tokens: u64,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub avg_latency_ms: u64,
    #[serde(default)]
    pub source: String, // "env" or "db"
}

pub async fn list_configured(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ConfiguredProviderResponse>>, StatusCode> {
    let mut result = Vec::new();

    // Include runtime providers (from env vars) as editable entries
    let runtime_stats = state.llm.provider_stats();
    for rs in &runtime_stats {
        // Skip "coming soon" providers
        if rs.status == "coming_soon" {
            continue;
        }
        // Check if this runtime provider has been overridden in redb
        let id = rs.name.to_lowercase().replace(|c: char| !c.is_ascii_alphanumeric(), "-");
        let has_override = state.store.get_provider(&id).ok().flatten().is_some();
        if has_override {
            continue; // will be shown from redb below
        }
        result.push(ConfiguredProviderResponse {
            id,
            name: rs.name.clone(),
            provider_type: if rs.name == "Groq" { qo_llm::ProviderType::Groq } else { qo_llm::ProviderType::Custom },
            model: rs.model.clone(),
            base_url: None,
            enabled: rs.status == "active",
            tier: if rs.name == "Groq" { 2 } else { 3 },
            cost_per_1k_tokens: rs.cost_usd / (rs.total_tokens_estimate.max(1) as f64 / 1000.0),
            requests: rs.requests,
            tokens: rs.total_tokens_estimate,
            cost_usd: rs.cost_usd,
            avg_latency_ms: rs.avg_latency_ms,
            source: "env".to_string(),
        });
    }

    // User-configured providers from redb
    let providers = state
        .store
        .list_providers()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    for (_, json) in providers {
        if let Ok(cfg) = serde_json::from_str::<qo_llm::ProviderConfig>(&json) {
            result.push(ConfiguredProviderResponse {
                id: cfg.id,
                name: cfg.name,
                provider_type: cfg.provider_type,
                model: cfg.model,
                base_url: cfg.base_url,
                enabled: cfg.enabled,
                tier: cfg.tier,
                cost_per_1k_tokens: cfg.cost_per_1k_tokens,
                requests: 0,
                tokens: 0,
                cost_usd: 0.0,
                avg_latency_ms: 0,
                source: "db".to_string(),
            });
        }
    }
    Ok(Json(result))
}

#[derive(Debug, Deserialize)]
pub struct AddProviderRequest {
    pub template_id: String,
    pub api_key: String,
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct AddProviderResponse {
    pub id: String,
    pub success: bool,
}

pub async fn add_provider(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddProviderRequest>,
) -> Result<Json<AddProviderResponse>, StatusCode> {
    // Find the template
    let templates = qo_llm::provider_templates();
    let template = templates
        .iter()
        .find(|t| t.id == req.template_id)
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Find the model's cost
    let cost_per_1k = template
        .models
        .iter()
        .find(|m| m.id == req.model)
        .map(|m| m.cost_per_1k)
        .unwrap_or(0.0);

    let config = qo_llm::ProviderConfig {
        id: template.id.to_string(),
        name: template.name.to_string(),
        provider_type: template.provider_type,
        api_key: req.api_key,
        base_url: Some(template.base_url.to_string()),
        model: req.model,
        enabled: true,
        tier: template.tier,
        cost_per_1k_tokens: cost_per_1k,
    };

    let json = serde_json::to_string(&config).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .store
        .save_provider(&config.id, &json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(AddProviderResponse {
        id: config.id,
        success: true,
    }))
}

#[derive(Debug, Deserialize)]
pub struct TestProviderRequest {
    pub id: String,
}

#[derive(Debug, Serialize)]
pub struct TestProviderResponse {
    pub success: bool,
    pub latency_ms: u64,
    pub message: String,
}

pub async fn test_provider(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TestProviderRequest>,
) -> Result<Json<TestProviderResponse>, StatusCode> {
    // Try redb first, then fallback to env-based runtime providers
    let (base_url, model, api_key) = if let Some(json) = state
        .store
        .get_provider(&req.id)
        .ok()
        .flatten()
    {
        let config: qo_llm::ProviderConfig =
            serde_json::from_str(&json).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        (
            config.base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            config.model,
            config.api_key,
        )
    } else {
        // Fallback: env-based providers
        match req.id.as_str() {
            "groq" => (
                "https://api.groq.com/openai/v1".to_string(),
                "llama-3.3-70b-versatile".to_string(),
                std::env::var("GROQ_API_KEY").unwrap_or_default(),
            ),
            "cloud" => (
                std::env::var("CLOUD_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                std::env::var("CLOUD_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
                std::env::var("ANTHROPIC_API_KEY")
                    .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
                    .unwrap_or_default(),
            ),
            _ => return Ok(Json(TestProviderResponse {
                success: false,
                latency_ms: 0,
                message: format!("Provider '{}' nicht gefunden", req.id),
            })),
        }
    };

    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Sage einfach 'OK'"}],
        "max_tokens": 10,
    });

    let start = Instant::now();

    // Don't send Authorization header if api_key is empty (e.g. Ollama)
    let mut request = client.post(&url).json(&body);
    if !api_key.is_empty() {
        request = request.bearer_auth(&api_key);
    }
    let result = request.send().await;

    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(resp) if resp.status().is_success() => {
            let json_resp: serde_json::Value = resp.json().await.unwrap_or_default();
            let content = json_resp["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("OK")
                .to_string();
            Ok(Json(TestProviderResponse {
                success: true,
                latency_ms: elapsed,
                message: content,
            }))
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            Ok(Json(TestProviderResponse {
                success: false,
                latency_ms: elapsed,
                message: format!("HTTP {status}"),
            }))
        }
        Err(e) => Ok(Json(TestProviderResponse {
            success: false,
            latency_ms: elapsed,
            message: e.to_string(),
        })),
    }
}

pub async fn toggle_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let json = state
        .store
        .get_provider(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let mut config: qo_llm::ProviderConfig =
        serde_json::from_str(&json).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    config.enabled = !config.enabled;

    let updated_json =
        serde_json::to_string(&config).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .store
        .save_provider(&id, &updated_json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({ "id": id, "enabled": config.enabled })))
}

#[derive(Debug, Deserialize)]
pub struct UpdateProviderRequest {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub enabled: Option<bool>,
}

pub async fn update_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateProviderRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let json = state
        .store
        .get_provider(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let mut config: qo_llm::ProviderConfig =
        serde_json::from_str(&json).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if let Some(api_key) = req.api_key {
        config.api_key = api_key;
    }
    if let Some(model) = &req.model {
        config.model = model.clone();
        // Update cost from template
        let templates = qo_llm::provider_templates();
        if let Some(template) = templates.iter().find(|t| t.id == id) {
            if let Some(m) = template.models.iter().find(|m| m.id == model.as_str()) {
                config.cost_per_1k_tokens = m.cost_per_1k;
            }
        }
    }
    if let Some(enabled) = req.enabled {
        config.enabled = enabled;
    }

    let updated_json =
        serde_json::to_string(&config).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    state
        .store
        .save_provider(&id, &updated_json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(serde_json::json!({
        "id": id,
        "name": config.name,
        "model": config.model,
        "enabled": config.enabled,
        "updated": true
    })))
}

pub async fn delete_provider(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    state
        .store
        .delete_provider(&id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(StatusCode::NO_CONTENT)
}
