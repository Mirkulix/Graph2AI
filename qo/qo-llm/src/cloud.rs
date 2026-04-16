use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMessage {
    pub role: String,
    pub content: String,
}

pub struct CloudClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl CloudClient {
    pub fn new(api_key: String, base_url: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            base_url,
            model,
        }
    }

    pub async fn chat(
        &self,
        messages: Vec<CloudMessage>,
    ) -> Result<String, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await?;

        // Check HTTP status before trying to parse — providers love
        // returning 429/503 with HTML Cloudflare pages that blow up
        // response.json() with the misleading "error decoding response
        // body". We surface a useful error instead.
        let status = response.status();
        let raw_body = response
            .text()
            .await
            .map_err(|e| format!("cloud provider body read failed: {e}"))?;
        if !status.is_success() {
            let snippet = raw_body.chars().take(300).collect::<String>();
            return Err(format!("cloud provider HTTP {status}: {snippet}").into());
        }

        // Try JSON — if this fails we emit a clear message that
        // includes the first 200 chars so operators can see whether
        // they got HTML, YAML, or a different JSON schema.
        let json: serde_json::Value =
            serde_json::from_str(&raw_body).map_err(|e| {
                let snippet = raw_body.chars().take(200).collect::<String>();
                format!("cloud provider returned non-JSON body ({e}): {snippet}")
            })?;

        // Provider-side errors often come back as 200 OK with
        // `{"error":{"message":"…"}}` — surface them as a real error.
        if let Some(msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(format!("cloud provider returned error: {msg}").into());
        }

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| {
                let snippet = raw_body.chars().take(200).collect::<String>();
                format!("cloud provider response has no choices[0].message.content: {snippet}")
            })?
            .to_string();

        Ok(content)
    }
}
