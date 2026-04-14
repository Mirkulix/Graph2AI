use serde::{Deserialize, Serialize};
use crate::cloud_http::{CloudClient, CloudError};

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("HTTP error: {0}")]
    Http(#[from] CloudError),
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("API error: {0}")]
    Api(String),
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    api_key: String,
    _organization: Option<String>,
    client: CloudClient,
}

impl OpenAiClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").ok()?;
        let _organization = std::env::var("OPENAI_ORG_ID").ok();
        Some(Self {
            api_key,
            _organization,
            client: CloudClient::new("api.openai.com", 443).with_tls(true).with_timeout(120),
        })
    }

    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            _organization: None,
            client: CloudClient::new("api.openai.com", 443).with_tls(true).with_timeout(120),
        }
    }

    pub fn generate(&self, model: &str, prompt: &str) -> Result<String, OpenAiError> {
        #[derive(Serialize)]
        struct Request {
            model: String,
            prompt: String,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            text: String,
        }

        let body = serde_json::to_string(&Request {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: 1024,
            temperature: 0.7,
        })?;

        let auth = format!("Bearer {}", self.api_key);
        let response = self.client.post_tls(
            "/v1/completions",
            &body,
            &[("Authorization", &auth)],
            "api.openai.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(OpenAiError::Api(err));
        }

        let parsed: Resp = serde_json::from_str(&response.body)
            .map_err(|e| OpenAiError::Api(format!("JSON parse error: {}", e)))?;

        Ok(parsed.choices.into_iter().next()
            .map(|c| c.text)
            .unwrap_or_default())
    }

    pub fn chat(&self, model: &str, messages: &[crate::ollama::ChatMessage]) -> Result<String, OpenAiError> {
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
            "api.openai.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(OpenAiError::Api(err));
        }

        let parsed: ChatResp = serde_json::from_str(&response.body)
            .map_err(|e| OpenAiError::Api(format!("JSON parse error: {}", e)))?;

        Ok(parsed.choices.into_iter().next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }

    pub fn is_configured() -> bool {
        std::env::var("OPENAI_API_KEY").is_ok()
    }
}
