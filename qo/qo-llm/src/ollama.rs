use reqwest::Client;

pub struct OllamaClient {
    client: Client,
    pub base_url: String,
    pub model: String,
}

impl OllamaClient {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
            model,
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<(String, String)>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}/v1/chat/completions", self.base_url.trim_end_matches('/'));

        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| {
                serde_json::json!({"role": role, "content": content})
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "max_tokens": 512,  // Tier 1 = short responses
            "temperature": 0.3, // Low temp for deterministic routing/scoring
        });

        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama error {status}: {text}").into());
        }

        let json: serde_json::Value = resp.json().await?;
        json["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| "No content in Ollama response".into())
    }

    /// Check if Ollama is reachable
    pub async fn health(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        self.client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_creation() {
        let client = OllamaClient::new("http://localhost:11434".into(), "qwen2.5:3b".into());
        assert_eq!(client.model, "qwen2.5:3b");
    }

    #[test]
    fn base_url_stored() {
        let client = OllamaClient::new("http://localhost:11434".into(), "orbit-companion-ft-q4".into());
        assert_eq!(client.base_url, "http://localhost:11434");
    }
}
