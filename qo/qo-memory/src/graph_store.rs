use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const GRAPH_TABLE: TableDefinition<u64, &str> = TableDefinition::new("qlang_graphs");
/// Stores raw QLBG binary blobs keyed by the same u64 IDs used in GRAPH_TABLE.
const BINARY_GRAPHS_TABLE: TableDefinition<u64, &[u8]> =
    TableDefinition::new("qlang_binary_graphs");

/// A stored QLANG execution graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredGraph {
    pub id: u64,
    pub timestamp: u64,
    pub graph_type: GraphType,
    pub title: String,
    pub nodes: Vec<QlangNode>,
    pub edges: Vec<QlangEdge>,
    pub metadata: GraphMetadata,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GraphType {
    Chat,
    GoalExecution,
    AgentTask,
    Evolution,
    ValueCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QlangNode {
    pub id: String,
    pub op: String,
    pub node_type: NodeType,
    pub label: String,
    pub agent: Option<String>,
    pub status: NodeStatus,
    pub duration_ms: Option<u64>,
    pub input_type: Option<String>,
    pub output_type: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeType {
    Input,
    Output,
    Llm,
    Deterministic,
    Memory,
    Values,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QlangEdge {
    pub from: String,
    pub to: String,
    pub data_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    pub total_duration_ms: Option<u64>,
    pub llm_tier: Option<String>,
    pub tokens_estimated: Option<u64>,
    pub cost_usd: Option<f64>,
}

pub struct GraphStore {
    db: Arc<Database>,
    next_id: std::sync::atomic::AtomicU64,
}

impl GraphStore {
    pub fn new(db: Arc<Database>) -> Result<Self, redb::Error> {
        // Create tables if not exists
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(GRAPH_TABLE)?;
            let _ = write_txn.open_table(BINARY_GRAPHS_TABLE)?;
        }
        write_txn.commit()?;

        // Find max ID
        let max_id = {
            let read_txn = db.begin_read()?;
            let table = read_txn.open_table(GRAPH_TABLE)?;
            table
                .iter()?
                .last()
                .and_then(|e| e.ok().map(|(k, _)| k.value()))
                .unwrap_or(0)
        };

        Ok(Self {
            db,
            next_id: std::sync::atomic::AtomicU64::new(max_id + 1),
        })
    }

    /// Store a graph and return its assigned ID
    pub fn store(&self, graph: &StoredGraph) -> Result<u64, redb::Error> {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut graph_with_id = graph.clone();
        graph_with_id.id = id;
        let json = serde_json::to_string(&graph_with_id).unwrap_or_default();
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(GRAPH_TABLE)?;
            table.insert(id, json.as_str())?;
        }
        write_txn.commit()?;
        Ok(id)
    }

    /// Get a graph by ID
    pub fn get(&self, id: u64) -> Result<Option<StoredGraph>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(GRAPH_TABLE)?;
        Ok(table
            .get(id)?
            .and_then(|v| serde_json::from_str(v.value()).ok()))
    }

    /// List recent graphs (newest first)
    pub fn list_recent(&self, limit: usize) -> Result<Vec<StoredGraph>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(GRAPH_TABLE)?;
        let mut results = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            if let Ok(graph) = serde_json::from_str::<StoredGraph>(v.value()) {
                results.push(graph);
            }
        }
        // Newest first
        results.reverse();
        results.truncate(limit);
        Ok(results)
    }

    /// Count total graphs
    pub fn count(&self) -> Result<u64, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(GRAPH_TABLE)?;
        Ok(table.len()?)
    }

    /// List graphs by type
    pub fn list_by_type(
        &self,
        graph_type: GraphType,
        limit: usize,
    ) -> Result<Vec<StoredGraph>, redb::Error> {
        let all = self.list_recent(500)?;
        let type_str = serde_json::to_string(&graph_type).unwrap_or_default();
        Ok(all
            .into_iter()
            .filter(|g| {
                serde_json::to_string(&g.graph_type).unwrap_or_default() == type_str
            })
            .take(limit)
            .collect())
    }

    // ----------------------------------------------------------------
    // QLANG binary graph storage (.qlbg format)
    // ----------------------------------------------------------------

    /// Store a raw QLBG binary blob alongside its JSON metadata.
    ///
    /// `id` should be the same ID returned by [`Self::store`] so the two
    /// representations can be cross-referenced.
    pub fn store_binary_graph(&self, id: u64, binary: &[u8]) -> Result<(), redb::Error> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(BINARY_GRAPHS_TABLE)?;
            table.insert(id, binary)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Retrieve a raw QLBG binary blob by ID.
    ///
    /// Returns `None` if no binary was stored for this ID.
    pub fn get_binary_graph(&self, id: u64) -> Result<Option<Vec<u8>>, redb::Error> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(BINARY_GRAPHS_TABLE)?;
        Ok(table.get(id)?.map(|v| v.value().to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_builders;
    use tempfile::tempdir;

    fn make_db(path: &std::path::Path) -> Arc<Database> {
        Arc::new(Database::create(path).unwrap())
    }

    #[test]
    fn test_store_and_retrieve() {
        let dir = tempdir().unwrap();
        let db = make_db(&dir.path().join("test.redb"));
        let gs = GraphStore::new(db).unwrap();

        let graph = graph_builders::build_chat_graph("Hello", "World", "groq", 123);
        let id = gs.store(&graph).unwrap();

        let retrieved = gs.get(id).unwrap().expect("graph should exist");
        assert_eq!(retrieved.id, id);
        assert!(matches!(retrieved.graph_type, GraphType::Chat));
        assert!(retrieved.title.contains("Hello"));
    }

    #[test]
    fn test_list_recent_newest_first() {
        let dir = tempdir().unwrap();
        let db = make_db(&dir.path().join("list.redb"));
        let gs = GraphStore::new(db).unwrap();

        let g1 = graph_builders::build_chat_graph("First", "R1", "groq", 10);
        let g2 = graph_builders::build_chat_graph("Second", "R2", "groq", 20);
        let g3 = graph_builders::build_chat_graph("Third", "R3", "groq", 30);

        let id1 = gs.store(&g1).unwrap();
        let id2 = gs.store(&g2).unwrap();
        let id3 = gs.store(&g3).unwrap();

        let recent = gs.list_recent(10).unwrap();
        assert_eq!(recent.len(), 3);
        // Newest first means id3 > id2 > id1
        assert_eq!(recent[0].id, id3);
        assert_eq!(recent[1].id, id2);
        assert_eq!(recent[2].id, id1);
    }

    #[test]
    fn test_count() {
        let dir = tempdir().unwrap();
        let db = make_db(&dir.path().join("count.redb"));
        let gs = GraphStore::new(db).unwrap();

        assert_eq!(gs.count().unwrap(), 0);

        gs.store(&graph_builders::build_chat_graph("a", "b", "groq", 1))
            .unwrap();
        assert_eq!(gs.count().unwrap(), 1);

        gs.store(&graph_builders::build_evolution_graph(2, 1)).unwrap();
        assert_eq!(gs.count().unwrap(), 2);
    }
}
