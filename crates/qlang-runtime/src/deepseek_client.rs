use serde::{Deserialize, Serialize};
use crate::cloud_http::{CloudClient, CloudError};

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekError {
    #[error("HTTP error: {0}")]
    Http(#[from] CloudError),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("API error: {0}")]
    Api(String),
}

/// DeepSeek client — OpenAI-compatible HTTP API.
///
/// Base URL: `https://api.deepseek.com/v1`. Uses bearer-token auth via
/// the `DEEPSEEK_API_KEY` env var. Same TLS plumbing as `OpenAiClient`,
/// so no extra dependencies are pulled in.
#[derive(Debug, Clone)]
pub struct DeepSeekClient {
    api_key: String,
    client: CloudClient,
}

impl DeepSeekClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok()?;
        Some(Self {
            api_key,
            client: CloudClient::new("api.deepseek.com", 443).with_tls(true).with_timeout(120),
        })
    }

    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: CloudClient::new("api.deepseek.com", 443).with_tls(true).with_timeout(120),
        }
    }

    /// Single-prompt completion. Wraps the prompt as a single `user`
    /// message and dispatches to `/v1/chat/completions` (DeepSeek does
    /// not expose a legacy `/v1/completions` endpoint).
    pub fn generate(&self, model: &str, prompt: &str) -> Result<String, DeepSeekError> {
        let model = if model.is_empty() { "deepseek-chat" } else { model };
        let msgs = vec![crate::ollama::ChatMessage::user(prompt)];
        self.chat(model, &msgs)
    }

    pub fn chat(&self, model: &str, messages: &[crate::ollama::ChatMessage]) -> Result<String, DeepSeekError> {
        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<Message>,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(Serialize)]
        struct Message {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct ChatResp {
            choices: Vec<ChatChoice>,
        }

        #[derive(Deserialize)]
        struct ChatChoice {
            message: ChatMsg,
        }

        #[derive(Deserialize)]
        struct ChatMsg {
            content: String,
        }

        let model = if model.is_empty() { "deepseek-chat" } else { model };

        let chat_messages: Vec<Message> = messages.iter().map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone(),
        }).collect();

        let body = serde_json::to_string(&ChatRequest {
            model: model.to_string(),
            messages: chat_messages,
            max_tokens: 1024,
            temperature: 0.7,
        })?;

        let auth = format!("Bearer {}", self.api_key);
        let response = self.client.post_tls(
            "/v1/chat/completions",
            &body,
            &[("Authorization", &auth)],
            "api.deepseek.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(DeepSeekError::Api(err));
        }

        let parsed: ChatResp = serde_json::from_str(&response.body)
            .map_err(|e| DeepSeekError::Api(format!("JSON parse error: {}", e)))?;

        Ok(parsed.choices.into_iter().next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }

    pub fn is_configured() -> bool {
        std::env::var("DEEPSEEK_API_KEY").is_ok()
    }
}
