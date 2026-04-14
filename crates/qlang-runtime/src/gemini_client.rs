use serde::{Deserialize, Serialize};
use crate::cloud_http::{CloudClient, CloudError};

#[derive(Debug, thiserror::Error)]
pub enum GeminiError {
    #[error("HTTP error: {0}")]
    Http(#[from] CloudError),
    #[error("API error: {0}")]
    Api(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct GeminiClient {
    api_key: String,
    client: CloudClient,
}

impl GeminiClient {
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("GEMINI_API_KEY").ok()?;
        Some(Self {
            api_key,
            client: CloudClient::new("generativelanguage.googleapis.com", 443).with_timeout(120),
        })
    }

    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: CloudClient::new("generativelanguage.googleapis.com", 443).with_timeout(120),
        }
    }

    pub fn generate(&self, model: &str, prompt: &str) -> Result<String, GeminiError> {
        #[derive(Serialize)]
        struct Request {
            contents: Vec<Content>,
            generation_config: GenerationConfig,
        }

        #[derive(Serialize)]
        struct Content {
            parts: Vec<Part>,
        }

        #[derive(Serialize)]
        struct Part {
            text: String,
        }

        #[derive(Serialize)]
        struct GenerationConfig {
            max_output_tokens: u32,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct Response {
            candidates: Option<Vec<Candidate>>,
        }

        #[derive(Deserialize)]
        struct Candidate {
            content: Option<ContentResp>,
        }

        #[derive(Deserialize)]
        struct ContentResp {
            parts: Option<Vec<PartResp>>,
        }

        #[derive(Deserialize)]
        struct PartResp {
            text: Option<String>,
        }

        let body = serde_json::to_string(&Request {
            contents: vec![Content {
                parts: vec![Part { text: prompt.to_string() }],
            }],
            generation_config: GenerationConfig {
                max_output_tokens: 1024,
                temperature: 0.7,
            },
        })?;

        let path = format!("/v1beta/models/{}:generateContent?key={}", model, self.api_key);
        let response = self.client.post_tls(
            &path,
            &body,
            &[],
            "generativelanguage.googleapis.com",
        )?;

        if response.status != 200 {
            let err = crate::cloud_http::extract_json_error(&response.body)
                .unwrap_or_else(|| format!("HTTP {}", response.status));
            return Err(GeminiError::Api(err));
        }

        let parsed: Response = serde_json::from_str(&response.body)
            .map_err(|e| GeminiError::Api(format!("JSON parse error: {}", e)))?;

        if let Some(candidates) = parsed.candidates {
            for candidate in candidates {
                if let Some(content) = candidate.content {
                    if let Some(parts) = content.parts {
                        for part in parts {
                            if let Some(text) = part.text {
                                return Ok(text);
                            }
                        }
                    }
                }
            }
        }
        Ok(String::new())
    }

    pub fn is_configured() -> bool {
        std::env::var("GEMINI_API_KEY").is_ok()
    }
}
