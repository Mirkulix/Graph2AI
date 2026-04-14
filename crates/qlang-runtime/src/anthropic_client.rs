use serde::{Deserialize, Serialize};
use crate::cloud_http::{CloudClient, CloudError};

#[derive(Debug, thiserror::Error)]
pub enum AnthropicError {
    #[error("HTTP error: {0}")]
    Http(#[from] CloudError),
    #[error("API error: {0}")]
    Api(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct AnthropicClient {
    api_key: String,
    client: CloudClient,
}

impl AnthropicClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok()?;
        Some(Self {
            api_key,
            client: CloudClient::new("api.anthropic.com", 443).with_timeout(120),
        })
    }

    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: CloudClient::new("api.anthropic.com", 443).with_timeout(120),
        }
    }

    pub fn chat(&self, model: &str, messages: &[crate::ollama::ChatMessage]) -> Result<String, AnthropicError> {
        #[derive(Serialize)]
        struct Request {
            model: String,
            messages: Vec<Msg>,
            max_tokens: u32,
        }

        #[derive(Serialize)]
        struct Msg {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct Response {
            content: Vec<ContentBlock>,
        }

        #[derive(Deserialize)]
        #[serde(tag = "type")]
        enum ContentBlock {
            #[serde(rename = "text")]
            Text { text: String },
            #[serde(other)]
            Other,
        }

        let msgs: Vec<Msg> = messages.iter().map(|m| Msg {
            role: m.role.clone(),
            content: m.content.clone(),
        }).collect();

        let body = serde_json::to_string(&Request {
            model: model.to_string(),
            messages: msgs,
            max_tokens: 1024,
        })?;

        let auth = format!("Bearer {}", self.api_key);
        let response = self.client.post_tls(
            "/v1/messages",
            &body,
            &[("Authorization", &auth), ("anthropic-version", "2023-06-01")],
            "api.anthropic.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(AnthropicError::Api(err));
        }

        let parsed: Response = serde_json::from_str(&response.body)
            .map_err(|e| AnthropicError::Api(format!("JSON parse error: {}", e)))?;

        for block in parsed.content {
            if let ContentBlock::Text { text } = block {
                return Ok(text);
            }
        }
        Ok(String::new())
    }

    pub fn is_configured() -> bool {
        std::env::var("ANTHROPIC_API_KEY").is_ok()
    }
}
