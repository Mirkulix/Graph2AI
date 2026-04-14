use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMessage {
    pub role: String,
    pub content: String,
}

pub struct GroqClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GroqClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model: "llama-3.3-70b-versatile".to_string(),
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<GroqMessage>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        let response = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = response.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("missing content in response")?
            .to_string();

        Ok(content)
    }
}
