use std::path::Path;
use crate::store::Store;

pub struct ImportResult {
    pub messages: usize,
    pub goals: usize,
    pub patterns: usize,
    pub proposals: usize,
}

/// Import Orbit data from JSONL files into QO's redb store.
pub fn import_orbit_data(store: &Store, import_dir: &Path) -> Result<ImportResult, Box<dyn std::error::Error>> {
    let mut result = ImportResult { messages: 0, goals: 0, patterns: 0, proposals: 0 };

    // Import messages
    let messages_file = import_dir.join("messages.jsonl");
    if messages_file.exists() {
        let content = std::fs::read_to_string(&messages_file)?;
        for (i, line) in content.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
                let id = msg["id"].as_u64().unwrap_or(i as u64);
                let qo_chat = serde_json::json!({
                    "user": msg["content"].as_str().unwrap_or(""),
                    "assistant": "",
                    "role": msg["role"].as_str().unwrap_or("user"),
                    "imported": true,
                    "source": "orbit",
                });
                let _ = store.store_chat(id, &qo_chat.to_string());
                result.messages += 1;
            }
        }
    }

    // Import goals (into action_history since we don't have goals table persistence yet)
    let goals_file = import_dir.join("goals.jsonl");
    if goals_file.exists() {
        let content = std::fs::read_to_string(&goals_file)?;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(_goal) = serde_json::from_str::<serde_json::Value>(line) {
                let _ = store.log_action("goal_imported", "Orbit Goal importiert", line);
                result.goals += 1;
            }
        }
    }

    // Import patterns
    let patterns_file = import_dir.join("patterns.jsonl");
    if patterns_file.exists() {
        let content = std::fs::read_to_string(&patterns_file)?;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(_pattern) = serde_json::from_str::<serde_json::Value>(line) {
                let _ = store.log_action("pattern_imported", "Orbit Pattern importiert", line);
                result.patterns += 1;
            }
        }
    }

    // Import proposals
    let proposals_file = import_dir.join("proposals.jsonl");
    if proposals_file.exists() {
        let content = std::fs::read_to_string(&proposals_file)?;
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            if let Ok(_proposal) = serde_json::from_str::<serde_json::Value>(line) {
                let _ = store.log_action("proposal_imported", "Orbit Vorschlag importiert", line);
                result.proposals += 1;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_store(dir: &std::path::Path) -> Store {
        Store::open(dir.join("test.redb")).unwrap()
    }

    #[test]
    fn test_import_messages_jsonl() {
        let db_dir = tempdir().unwrap();
        let import_dir = tempdir().unwrap();
        let store = make_store(db_dir.path());

        let messages = r#"{"id":1,"role":"user","content":"Hello Orbit"}
{"id":2,"role":"assistant","content":"Hi there"}
"#;
        std::fs::write(import_dir.path().join("messages.jsonl"), messages).unwrap();

        let result = import_orbit_data(&store, import_dir.path()).unwrap();
        assert_eq!(result.messages, 2);
        assert_eq!(result.goals, 0);
        assert_eq!(result.patterns, 0);
        assert_eq!(result.proposals, 0);

        let history = store.chat_history(10).unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_import_empty_dir() {
        let db_dir = tempdir().unwrap();
        let import_dir = tempdir().unwrap();
        let store = make_store(db_dir.path());

        let result = import_orbit_data(&store, import_dir.path()).unwrap();
        assert_eq!(result.messages, 0);
        assert_eq!(result.goals, 0);
        assert_eq!(result.patterns, 0);
        assert_eq!(result.proposals, 0);
    }

    #[test]
    fn test_import_nonexistent_dir() {
        let db_dir = tempdir().unwrap();
        let store = make_store(db_dir.path());

        // A non-existent dir: all files won't exist, so result should be all zeros
        let nonexistent = std::path::Path::new("/tmp/qo_test_nonexistent_dir_12345");
        let result = import_orbit_data(&store, nonexistent).unwrap();
        assert_eq!(result.messages, 0);
        assert_eq!(result.goals, 0);
        assert_eq!(result.patterns, 0);
        assert_eq!(result.proposals, 0);
    }
}
