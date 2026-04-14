use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Returns today's date as "YYYY-MM-DD" in UTC (no external deps needed).
pub fn today_date_local() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Days since Unix epoch
    let days = secs / 86400;
    // Compute year/month/day via the proleptic Gregorian algorithm
    let (y, m, d) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Returns current time as "HH:MM:SS" in UTC.
pub fn time_hms_local() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let h = (secs % 86400) / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Using the algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m as u64, d as u64)
}

pub struct ObsidianBridge {
    pub vault_path: PathBuf,
}

impl ObsidianBridge {
    pub fn new(vault_path: impl Into<PathBuf>) -> Self {
        Self {
            vault_path: vault_path.into(),
        }
    }

    pub async fn write_consciousness_log(&self, date: &str, entry: &str) -> std::io::Result<()> {
        let dir = self.vault_path.join("consciousness");
        fs::create_dir_all(&dir).await?;
        let file_path = dir.join(format!("{date}.md"));
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;
        let content = format!(
            "---\ndate: {date}\ntype: consciousness-log\n---\n\n{entry}\n"
        );
        file.write_all(content.as_bytes()).await?;
        Ok(())
    }

    pub async fn write_chat_log(
        &self,
        date: &str,
        role: &str,
        content: &str,
    ) -> std::io::Result<()> {
        let dir = self.vault_path.join("chats");
        fs::create_dir_all(&dir).await?;
        let file_path = dir.join(format!("{date}.md"));
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)
            .await?;
        let entry = format!(
            "---\ndate: {date}\ntype: chat-log\n---\n\n**{role}**: {content}\n"
        );
        file.write_all(entry.as_bytes()).await?;
        Ok(())
    }

    /// Append a formatted consciousness event entry to the vault.
    pub async fn log_consciousness_event(
        &self,
        mood: &str,
        energy: f32,
        heartbeat: u64,
        event_description: &str,
    ) -> std::io::Result<()> {
        let date = today_date_local();
        let time = time_hms_local();
        let entry = format!(
            "### {} — {}\nMood: {} | Energy: {:.0}% | Heartbeat: #{}\n",
            time, event_description, mood, energy, heartbeat
        );
        self.write_consciousness_log(&date, &entry).await
    }

    pub async fn list_sections(&self) -> std::io::Result<Vec<String>> {
        let mut sections = Vec::new();
        let mut read_dir = fs::read_dir(&self.vault_path).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    sections.push(name.to_owned());
                }
            }
        }
        sections.sort();
        Ok(sections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn write_and_read_consciousness() {
        let dir = tempdir().unwrap();
        let bridge = ObsidianBridge::new(dir.path());

        bridge
            .write_consciousness_log("2026-04-07", "Today I thought about embeddings.")
            .await
            .unwrap();

        let file_path = dir.path().join("consciousness").join("2026-04-07.md");
        let content = tokio::fs::read_to_string(&file_path).await.unwrap();
        assert!(content.contains("consciousness-log"));
        assert!(content.contains("Today I thought about embeddings."));
    }

    #[tokio::test]
    async fn list_sections() {
        let dir = tempdir().unwrap();
        let bridge = ObsidianBridge::new(dir.path());

        bridge
            .write_consciousness_log("2026-04-07", "entry one")
            .await
            .unwrap();
        bridge
            .write_chat_log("2026-04-07", "user", "hello")
            .await
            .unwrap();

        let sections = bridge.list_sections().await.unwrap();
        assert!(sections.contains(&"consciousness".to_string()));
        assert!(sections.contains(&"chats".to_string()));
    }
}
