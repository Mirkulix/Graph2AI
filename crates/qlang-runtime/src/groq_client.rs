use serde::{Deserialize, Serialize};
use crate::cloud_http::{CloudClient, CloudError};

#[derive(Debug, thiserror::Error)]
pub enum GroqError {
    #[error("HTTP error: {0}")]
    Http(#[from] CloudError),
    #[error("API error: {0}")]
    Api(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct GroqClient {
    api_key: String,
    client: CloudClient,
}

impl GroqClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("GROQ_API_KEY").ok()?;
        Some(Self {
            api_key,
            client: CloudClient::new("api.groq.com", 443).with_timeout(60),
        })
    }

    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: CloudClient::new("api.groq.com", 443).with_timeout(60),
        }
    }

    pub fn chat(&self, model: &str, messages: &[crate::ollama::ChatMessage]) -> Result<String, GroqError> {
        #[derive(Serialize)]
        struct Request {
            model: String,
            messages: Vec<Msg>,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(Serialize)]
        struct Msg {
            role: String,
            content: String,
        }

        #[derive(Deserialize)]
        struct Response {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ChatMsg,
        }

        #[derive(Deserialize)]
        struct ChatMsg {
            content: String,
        }

        let msgs: Vec<Msg> = messages.iter().map(|m| Msg {
            role: m.role.clone(),
            content: m.content.clone(),
        }).collect();

        let body = serde_json::to_string(&Request {
            model: model.to_string(),
            messages: msgs,
            max_tokens: 1024,
            temperature: 0.7,
        })?;

        let auth = format!("Bearer {}", self.api_key);
        let response = self.client.post_tls(
            "/openai/v1/chat/completions",
            &body,
            &[("Authorization", &auth)],
            "api.groq.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(GroqError::Api(err));
        }

        let parsed: Response = serde_json::from_str(&response.body)
            .map_err(|e| GroqError::Api(format!("JSON parse error: {}", e)))?;

        Ok(parsed.choices.into_iter().next()
            .map(|c| c.message.content)
            .unwrap_or_default())
    }

    pub fn is_configured() -> bool {
        std::env::var("GROQ_API_KEY").is_ok()
    }
}
