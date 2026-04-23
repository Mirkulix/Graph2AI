use serde::{Deserialize, Serialize};
use std::error::Error;

/// Single message in a DeepSeek chat exchange. Same shape as the
/// OpenAI / Groq message — DeepSeek's API is OpenAI-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekMessage {
    pub role: String,
    pub content: String,
}

/// Minimal client for the DeepSeek chat-completions endpoint.
///
/// Base URL is hardcoded to `https://api.deepseek.com/v1` because it
/// is the only endpoint DeepSeek currently exposes; if that changes,
/// switch to the generic `CloudClient` instead.
pub struct DeepSeekClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl DeepSeekClient {
    pub fn new(api_key: String) -> Self {
        Self::with_model(api_key, "deepseek-chat".to_string())
    }

    pub fn with_model(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Read back the API key — used by `LlmRouter::chat_with_model`
    /// to spin up a per-call client with a different model name while
    /// reusing the installed credentials.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub async fn chat(
        &self,
        messages: Vec<DeepSeekMessage>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        let response = self
            .client
            .post("https://api.deepseek.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await?;

        let status = response.status();
        let raw_body = response
            .text()
            .await
            .map_err(|e| format!("deepseek body read failed: {e}"))?;
        if !status.is_success() {
            let snippet = raw_body.chars().take(300).collect::<String>();
            return Err(format!("deepseek HTTP {status}: {snippet}").into());
        }

        let json: serde_json::Value =
            serde_json::from_str(&raw_body).map_err(|e| {
                let snippet = raw_body.chars().take(200).collect::<String>();
                format!("deepseek returned non-JSON body ({e}): {snippet}")
            })?;

        if let Some(msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(format!("deepseek returned error: {msg}").into());
        }

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                let snippet = raw_body.chars().take(200).collect::<String>();
                format!("deepseek response has no choices[0].message.content: {snippet}")
            })?
            .to_string();

        Ok(content)
    }
}
