use reqwest::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    chat: TelegramChat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}

pub struct TelegramBot {
    pub token: String,
    pub qo_base_url: String,
    client: Client,
    pub allowed_chat_id: Option<i64>,
}

impl TelegramBot {
    pub fn new(token: String, qo_base_url: String, allowed_chat_id: Option<i64>) -> Self {
        Self {
            token,
            qo_base_url,
            client: Client::new(),
            allowed_chat_id,
        }
    }

    /// Run the bot (long polling loop)
    pub async fn run(&self) {
        let mut offset: i64 = 0;
        tracing::info!("Telegram bot started");

        loop {
            match self.get_updates(offset).await {
                Ok(updates) => {
                    for update in updates {
                        offset = update.update_id + 1;
                        if let Some(msg) = update.message {
                            self.handle_message(msg).await;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Telegram poll error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn get_updates(&self, offset: i64) -> Result<Vec<TelegramUpdate>, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("https://api.telegram.org/bot{}/getUpdates", self.token);
        let resp = self.client
            .post(&url)
            .json(&serde_json::json!({
                "offset": offset,
                "timeout": 30,
                "allowed_updates": ["message"]
            }))
            .send()
            .await?;

        let body: serde_json::Value = resp.json().await?;
        let updates: Vec<TelegramUpdate> = serde_json::from_value(
            body.get("result").cloned().unwrap_or(serde_json::Value::Array(vec![]))
        )?;
        Ok(updates)
    }

    async fn handle_message(&self, msg: TelegramMessage) {
        // Check if chat is allowed
        if let Some(allowed) = self.allowed_chat_id {
            if msg.chat.id != allowed {
                let _ = self.send_message(msg.chat.id, "Nicht autorisiert.").await;
                return;
            }
        }

        let text = match msg.text {
            Some(t) => t,
            None => return,
        };

        // Handle commands
        if text.starts_with("/start") || text.starts_with("/help") {
            let help = "QO Telegram Bot\n\n\
                Befehle:\n\
                /chat &lt;nachricht&gt; — Mit QO chatten\n\
                /goal &lt;beschreibung&gt; — Neues Ziel erstellen\n\
                /status — QO Status anzeigen\n\
                /agents — Agenten anzeigen\n\n\
                Oder einfach eine Nachricht schreiben — wird als Chat behandelt.";
            let _ = self.send_message(msg.chat.id, help).await;
            return;
        }

        if text.starts_with("/status") {
            match self.qo_request("GET", "/api/consciousness/state", None).await {
                Ok(body) => {
                    let mood = body["mood"].as_str().unwrap_or("?");
                    let energy = body["energy"].as_f64().unwrap_or(0.0);
                    let hb = body["heartbeat"].as_u64().unwrap_or(0);
                    let status = format!("QO Status\n\nStimmung: {mood}\nEnergie: {energy:.0}%\nHeartbeat: #{hb}");
                    let _ = self.send_message(msg.chat.id, &status).await;
                }
                Err(e) => {
                    let _ = self.send_message(msg.chat.id, &format!("Fehler: {e}")).await;
                }
            }
            return;
        }

        if text.starts_with("/agents") {
            match self.qo_request("GET", "/api/agents", None).await {
                Ok(body) => {
                    let agents = body["agents"].as_array();
                    let mut lines = vec!["QO Agenten\n".to_string()];
                    if let Some(agents) = agents {
                        for a in agents {
                            let role = a["role"].as_str().unwrap_or("?");
                            let status = a["status"].as_str().unwrap_or("?");
                            let completed = a["tasks_completed"].as_u64().unwrap_or(0);
                            lines.push(format!("{role}: {status} ({completed} erledigt)"));
                        }
                    }
                    let _ = self.send_message(msg.chat.id, &lines.join("\n")).await;
                }
                Err(e) => {
                    let _ = self.send_message(msg.chat.id, &format!("Fehler: {e}")).await;
                }
            }
            return;
        }

        if text.starts_with("/goal ") {
            let description = text.trim_start_matches("/goal ").trim();
            if description.is_empty() {
                let _ = self.send_message(msg.chat.id, "Bitte Ziel-Beschreibung angeben: /goal &lt;beschreibung&gt;").await;
                return;
            }
            let _ = self.send_message(msg.chat.id, "Ziel wird erstellt...").await;
            match self.qo_request("POST", "/api/goals", Some(serde_json::json!({"description": description}))).await {
                Ok(body) => {
                    let id = body["id"].as_u64().unwrap_or(0);
                    let _ = self.send_message(msg.chat.id, &format!("Ziel #{id} erstellt! CEO arbeitet daran...")).await;
                }
                Err(e) => {
                    let _ = self.send_message(msg.chat.id, &format!("Fehler: {e}")).await;
                }
            }
            return;
        }

        // Default: treat as chat
        let chat_text = text.trim_start_matches("/chat ").trim_start_matches("/chat");
        let chat_text = if chat_text.is_empty() { &text } else { chat_text };

        match self.qo_request("POST", "/api/chat", Some(serde_json::json!({"message": chat_text}))).await {
            Ok(body) => {
                let response = body["response"].as_str().unwrap_or("Keine Antwort");
                let tier = body["tier"].as_str().unwrap_or("?");
                let msg_text = format!("{response}\n\nvia {tier}");
                let _ = self.send_message(msg.chat.id, &msg_text).await;
            }
            Err(e) => {
                let _ = self.send_message(msg.chat.id, &format!("Fehler: {e}")).await;
            }
        }
    }

    async fn qo_request(
        &self,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("{}{}", self.qo_base_url, path);
        let req = match method {
            "POST" => {
                let mut r = self.client.post(&url);
                if let Some(b) = body {
                    r = r.json(&b);
                }
                r
            }
            _ => self.client.get(&url),
        };
        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Err(format!("QO API error: {}", resp.status()).into());
        }
        Ok(resp.json().await?)
    }

    async fn send_message(&self, chat_id: i64, text: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        self.client
            .post(&url)
            .json(&serde_json::json!({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            }))
            .send()
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bot_creation() {
        let bot = TelegramBot::new("test_token".into(), "http://localhost:4747".into(), Some(12345));
        assert_eq!(bot.token, "test_token");
        assert_eq!(bot.allowed_chat_id, Some(12345));
    }

    #[test]
    fn bot_without_chat_restriction() {
        let bot = TelegramBot::new("token".into(), "http://localhost:4747".into(), None);
        assert!(bot.allowed_chat_id.is_none());
    }
}
