use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const CHAT_TABLE: TableDefinition<u64, &str> = TableDefinition::new("chat");
const KV_TABLE: TableDefinition<&str, &str> = TableDefinition::new("kv");
const HISTORY_TABLE: TableDefinition<u64, &str> = TableDefinition::new("action_history");
const PROVIDERS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("providers");
const GOALS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("goals");
const AGENTS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("agents");
const PATTERNS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("patterns");
const PROPOSALS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("proposals");
const QUANTUM_TABLE: TableDefinition<&str, &str> = TableDefinition::new("quantum_state");
const EMBEDDINGS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("embeddings");

pub struct Store {
    db: Arc<Database>,
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, redb::Error> {
        let db = Arc::new(Database::create(path)?);
        // Ensure tables exist
        let write_txn = db.begin_write()?;
        {
            write_txn.open_table(CHAT_TABLE)?;
            write_txn.open_table(KV_TABLE)?;
            write_txn.open_table(HISTORY_TABLE)?;
            write_txn.open_table(PROVIDERS_TABLE)?;
            write_txn.open_table(GOALS_TABLE)?;
            write_txn.open_table(AGENTS_TABLE)?;
            write_txn.open_table(PATTERNS_TABLE)?;
            write_txn.open_table(PROPOSALS_TABLE)?;
            write_txn.open_table(QUANTUM_TABLE)?;
            write_txn.open_table(EMBEDDINGS_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    /// Expose the underlying database for sharing with other stores (e.g. GraphStore)
    pub fn db(&self) -> Arc<Database> {
        self.db.clone()
    }

    pub fn log_action(
        &self,
        action_type: &str,
        description: &str,
        details: &str,
    ) -> Result<(), redb::Error> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let entry = format!(
            r#"{{"id":{},"timestamp":{},"action_type":"{}","description":"{}","details":"{}"}}"#,
            id,
            timestamp,
            action_type,
            description.replace('"', "\\\"").replace('\n', "\\n"),
            details.replace('"', "\\\"").replace('\n', "\\n"),
        );
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(HISTORY_TABLE)?;
            table.insert(id, entry.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get_history(&self, limit: usize) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(HISTORY_TABLE)?;
        let mut results = Vec::new();
        // Collect all, then take last N (most recent = highest id)
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
        }
        // Return last `limit` entries (most recent)
        let start = if results.len() > limit { results.len() - limit } else { 0 };
        let mut slice = results[start..].to_vec();
        slice.reverse();
        Ok(slice)
    }

    pub fn store_chat(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CHAT_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn chat_history(&self, limit: usize) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CHAT_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
            if results.len() >= limit {
                break;
            }
        }
        Ok(results)
    }

    pub fn set(&self, key: &str, value: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(KV_TABLE)?;
            table.insert(key, value)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(KV_TABLE)?;
        Ok(table.get(key)?.map(|v| v.value().to_owned()))
    }

    pub fn save_provider(&self, id: &str, config_json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PROVIDERS_TABLE)?;
            table.insert(id, config_json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn get_provider(&self, id: &str) -> Result<Option<String>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(PROVIDERS_TABLE)?;
        Ok(table.get(id)?.map(|v| v.value().to_owned()))
    }

    pub fn list_providers(&self) -> Result<Vec<(String, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(PROVIDERS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value().to_owned(), v.value().to_owned()));
        }
        Ok(results)
    }

    pub fn delete_provider(&self, id: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PROVIDERS_TABLE)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    // ---- Goals ----

    pub fn save_goal(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(GOALS_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn list_goals(&self) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(GOALS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
        }
        Ok(results)
    }

    pub fn delete_goal(&self, id: u64) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(GOALS_TABLE)?;
            table.remove(id)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    // ---- Agent Stats ----

    pub fn save_agent_stats(&self, role: &str, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(AGENTS_TABLE)?;
            table.insert(role, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn load_agent_stats(&self) -> Result<Vec<(String, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(AGENTS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value().to_owned(), v.value().to_owned()));
        }
        Ok(results)
    }

    // ---- Patterns ----

    pub fn save_pattern(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PATTERNS_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn list_patterns(&self) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(PATTERNS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
        }
        Ok(results)
    }

    // ---- Proposals ----

    pub fn save_proposal(&self, id: u64, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PROPOSALS_TABLE)?;
            table.insert(id, json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn list_proposals(&self) -> Result<Vec<(u64, String)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(PROPOSALS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            results.push((k.value(), v.value().to_owned()));
        }
        Ok(results)
    }

    // ---- Quantum State ----

    pub fn save_quantum_state(&self, json: &str) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(QUANTUM_TABLE)?;
            table.insert("current", json)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn load_quantum_state(&self) -> Result<Option<String>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(QUANTUM_TABLE)?;
        Ok(table.get("current")?.map(|v| v.value().to_owned()))
    }

    // ---- Embeddings ----

    /// Serialize an f32 vector to bytes and persist it under `key`.
    pub fn store_embedding(&self, key: &str, bytes: &[u8]) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            table.insert(key, bytes)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Load all persisted embeddings as `(key, Vec<f32>)` pairs.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let bytes = v.value();
            // Deserialize little-endian f32 values
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            results.push((k.value().to_owned(), floats));
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_chat_history_limit() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("chat_limit_test.redb");
        let store = Store::open(&db_path).unwrap();

        for i in 1u64..=10 {
            store.store_chat(i, &format!(r#"{{"id":{}}}"#, i)).unwrap();
        }

        let history = store.chat_history(3).unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_overwrite_kv() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("overwrite_test.redb");
        let store = Store::open(&db_path).unwrap();

        store.set("key1", "first_value").unwrap();
        store.set("key1", "second_value").unwrap();

        let val = store.get("key1").unwrap();
        assert_eq!(val, Some("second_value".to_string()));
    }

    #[test]
    fn store_and_retrieve_chat() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let store = Store::open(&db_path).unwrap();

        store.store_chat(1, r#"{"role":"user","content":"hello"}"#).unwrap();
        store.store_chat(2, r#"{"role":"assistant","content":"hi"}"#).unwrap();

        let history = store.chat_history(10).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0, 1);
        assert!(history[0].1.contains("user"));
        assert_eq!(history[1].0, 2);
        assert!(history[1].1.contains("assistant"));
    }

    #[test]
    fn kv_set_get() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("kv_test.redb");
        let store = Store::open(&db_path).unwrap();

        store.set("model", "qo-1").unwrap();
        store.set("version", "0.1.0").unwrap();

        assert_eq!(store.get("model").unwrap(), Some("qo-1".to_string()));
        assert_eq!(store.get("version").unwrap(), Some("0.1.0".to_string()));
        assert_eq!(store.get("missing").unwrap(), None);
    }

    #[test]
    fn test_save_load_goal() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("goals_test.redb");
        let store = Store::open(&db_path).unwrap();

        let json1 = r#"{"id":1,"description":"Test goal","status":"Pending"}"#;
        let json2 = r#"{"id":2,"description":"Another goal","status":"Completed"}"#;

        store.save_goal(1, json1).unwrap();
        store.save_goal(2, json2).unwrap();

        let goals = store.list_goals().unwrap();
        assert_eq!(goals.len(), 2);
        assert_eq!(goals[0].0, 1);
        assert!(goals[0].1.contains("Test goal"));
        assert_eq!(goals[1].0, 2);
        assert!(goals[1].1.contains("Another goal"));

        store.delete_goal(1).unwrap();
        let goals_after = store.list_goals().unwrap();
        assert_eq!(goals_after.len(), 1);
        assert_eq!(goals_after[0].0, 2);
    }

    #[test]
    fn test_save_load_agent_stats() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("agents_test.redb");
        let store = Store::open(&db_path).unwrap();

        let ceo_json = r#"{"role":"Ceo","tasks_completed":5,"tasks_failed":1}"#;
        let dev_json = r#"{"role":"Developer","tasks_completed":3,"tasks_failed":0}"#;

        store.save_agent_stats("Ceo", ceo_json).unwrap();
        store.save_agent_stats("Developer", dev_json).unwrap();

        let stats = store.load_agent_stats().unwrap();
        assert_eq!(stats.len(), 2);

        let ceo = stats.iter().find(|(k, _)| k == "Ceo").unwrap();
        assert!(ceo.1.contains("tasks_completed"));

        let dev = stats.iter().find(|(k, _)| k == "Developer").unwrap();
        assert!(dev.1.contains("Developer"));
    }

    #[test]
    fn test_save_load_quantum_state() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("quantum_test.redb");
        let store = Store::open(&db_path).unwrap();

        // Initially empty
        assert_eq!(store.load_quantum_state().unwrap(), None);

        let json = r#"{"generation":5,"entropy":1.386,"strategies":["A","B"],"strategy_weights":[0.5,0.5]}"#;
        store.save_quantum_state(json).unwrap();

        let loaded = store.load_quantum_state().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert!(loaded.contains("generation"));
        assert!(loaded.contains("entropy"));

        // Overwrite
        let json2 = r#"{"generation":10,"entropy":0.5,"strategies":["A","B"],"strategy_weights":[0.8,0.2]}"#;
        store.save_quantum_state(json2).unwrap();
        let loaded2 = store.load_quantum_state().unwrap().unwrap();
        assert!(loaded2.contains("\"generation\":10"));
    }
}
