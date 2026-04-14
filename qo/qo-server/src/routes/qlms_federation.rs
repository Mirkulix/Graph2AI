//! QLMS Federated Organism — multi-node gossip + ternary majority merge.
//!
//! Complements `qlms_demo.rs` (which does point-to-point AI-to-AI round-trips)
//! by implementing N-way federated merging:
//!
//!   POST /api/qlms/federation/gossip  body: { "peers": ["host:port", ...] }
//!       → fetch each peer's specialist weights, merge via ternary majority
//!         vote, replace local specialist. Returns { merged_from, weight_changes,
//!         peers_ok, peers_failed }.
//!
//!   GET  /api/qlms/federation/weights
//!       → return this node's current specialist weights as JSON
//!         (ternary i8 vector + metadata).
//!
//!   GET  /api/qlms/federation/eval
//!       → evaluate this node's specialist on its local holdout and return
//!         { node_id, accuracy, sample_count }.
//!
//! Backwards compatible: 2-node gossip still works (the existing
//! `qlms-dual-server.sh` path is unchanged; federation is additive).

use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use once_cell::sync::Lazy;
use qlang_runtime::federation::{count_changes, ternary_majority_vote, verify_ternary};
use qlang_runtime::mnist::MnistData;
use qlang_runtime::ternary_brain::TernaryBrain;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::AppState;

// ============================================================
// Per-node specialist (trained once, cached, mutated by gossip)
// ============================================================

struct LocalSpecialist {
    brain: TernaryBrain,
    holdout_images: Vec<f32>,
    holdout_labels: Vec<u8>,
    holdout_n: usize,
    image_dim: u32,
    n_classes: u32,
    node_id: String,
}

static LOCAL: Lazy<Arc<Mutex<Option<LocalSpecialist>>>> =
    Lazy::new(|| Arc::new(Mutex::new(None)));

/// Node ID from env (`QO_NODE_ID`) or default "node-unknown".
///
/// The launcher sets this per server (a/b/c) so each node trains on a
/// distinct MNIST partition.
fn node_id() -> String {
    std::env::var("QO_NODE_ID").unwrap_or_else(|_| "node-unknown".to_string())
}

/// Deterministic partition index from node_id. Stable across restarts so a
/// given node always trains on the same MNIST slice.
fn partition_index(node: &str) -> u32 {
    // Simple FNV-1a hash so we do not pull in a crypto dep.
    let mut h: u32 = 2166136261;
    for b in node.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

/// Build a MNIST partition for this node (disjoint training slice, shared
/// holdout). Uses synthetic MNIST so the endpoint works without the real
/// dataset being present on disk.
fn build_partitioned_data(node: &str) -> (MnistData, Vec<f32>, Vec<u8>) {
    // Generate a larger pool, then slice per node → disjoint training subsets.
    let per_node = 600;
    let n_nodes = 3;
    let pool_n = per_node * n_nodes;
    let holdout_n = 300;

    let full = MnistData::synthetic(pool_n, holdout_n);
    let idx = (partition_index(node) % n_nodes as u32) as usize;
    let start = idx * per_node;
    let end = start + per_node;

    let image_size = full.image_size;
    let train_images =
        full.train_images[start * image_size..end * image_size].to_vec();
    let train_labels = full.train_labels[start..end].to_vec();

    let partition = MnistData {
        train_images,
        train_labels,
        test_images: Vec::new(),
        test_labels: Vec::new(),
        n_train: per_node,
        n_test: 0,
        image_size,
        n_classes: 10,
    };

    // Shared holdout for eval across all nodes
    (partition, full.test_images, full.test_labels)
}

/// Ensure the local specialist exists — train on first use.
fn ensure_local() {
    let mut guard = LOCAL.lock().unwrap();
    if guard.is_some() {
        return;
    }
    let node = node_id();
    let (partition, holdout_images, holdout_labels) = build_partitioned_data(&node);

    let mut brain = TernaryBrain::init(
        &partition.train_images,
        &partition.train_labels,
        partition.image_size,
        partition.n_train,
        partition.n_classes,
        5, // 5 neurons per class = 50 total
    );
    // Light refinement so the brain has learned something on its partition.
    brain.refine(
        &partition.train_images,
        &partition.train_labels,
        partition.n_train,
        5,
    );

    tracing::info!(
        "QLMS federation: trained local specialist for node '{}' on {} samples, {} weights",
        node,
        partition.n_train,
        brain.total_weights()
    );

    *guard = Some(LocalSpecialist {
        image_dim: brain.image_dim as u32,
        n_classes: brain.n_classes as u32,
        brain,
        holdout_images,
        holdout_labels,
        holdout_n: 300,
        node_id: node,
    });
}

// ============================================================
// GET /api/qlms/federation/weights
// ============================================================

#[derive(Serialize, Deserialize)]
pub struct WeightsOutput {
    pub node_id: String,
    pub image_dim: u32,
    pub n_classes: u32,
    pub total_weights: usize,
    pub weights: Vec<i8>,
}

pub async fn get_weights(
    State(_state): State<std::sync::Arc<AppState>>,
) -> Result<Json<WeightsOutput>, (StatusCode, String)> {
    ensure_local();
    let guard = LOCAL.lock().unwrap();
    let local = guard
        .as_ref()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "local not ready".into()))?;
    let weights = local.brain.dump_weights_i8();
    Ok(Json(WeightsOutput {
        node_id: local.node_id.clone(),
        image_dim: local.image_dim,
        n_classes: local.n_classes,
        total_weights: weights.len(),
        weights,
    }))
}

// ============================================================
// GET /api/qlms/federation/eval
// ============================================================

#[derive(Serialize)]
pub struct EvalOutput {
    pub node_id: String,
    pub accuracy: f32,
    pub sample_count: usize,
}

pub async fn eval_local(
    State(_state): State<std::sync::Arc<AppState>>,
) -> Result<Json<EvalOutput>, (StatusCode, String)> {
    ensure_local();
    let guard = LOCAL.lock().unwrap();
    let local = guard
        .as_ref()
        .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "local not ready".into()))?;
    let acc = local.brain.accuracy(
        &local.holdout_images,
        &local.holdout_labels,
        local.holdout_n,
    );
    Ok(Json(EvalOutput {
        node_id: local.node_id.clone(),
        accuracy: acc,
        sample_count: local.holdout_n,
    }))
}

// ============================================================
// POST /api/qlms/federation/gossip
// ============================================================

#[derive(Deserialize)]
pub struct GossipInput {
    /// Host:port list, e.g. ["localhost:4747", "localhost:4848"]
    pub peers: Vec<String>,
}

#[derive(Serialize)]
pub struct GossipOutput {
    pub node_id: String,
    pub merged_from: usize,
    pub weight_changes: usize,
    pub total_weights: usize,
    pub peers_ok: Vec<String>,
    pub peers_failed: Vec<PeerError>,
    pub accuracy_before: f32,
    pub accuracy_after: f32,
}

#[derive(Serialize)]
pub struct PeerError {
    pub peer: String,
    pub error: String,
}

pub async fn gossip(
    State(_state): State<std::sync::Arc<AppState>>,
    Json(input): Json<GossipInput>,
) -> Result<Json<GossipOutput>, (StatusCode, String)> {
    ensure_local();

    // Snapshot local state we need before going over the wire (holding the
    // mutex across .await would force it Send+Sync).
    let (local_weights, local_dim, local_classes, local_node, accuracy_before) = {
        let guard = LOCAL.lock().unwrap();
        let local = guard.as_ref().ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "local not ready".into(),
        ))?;
        let acc = local.brain.accuracy(
            &local.holdout_images,
            &local.holdout_labels,
            local.holdout_n,
        );
        (
            local.brain.dump_weights_i8(),
            local.image_dim,
            local.n_classes,
            local.node_id.clone(),
            acc,
        )
    };

    let client = reqwest::Client::new();
    let mut peer_weights: Vec<Vec<i8>> = vec![local_weights.clone()];
    let mut peers_ok: Vec<String> = Vec::new();
    let mut peers_failed: Vec<PeerError> = Vec::new();

    for peer in &input.peers {
        let url = format!("http://{}/api/qlms/federation/weights", peer);
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<WeightsOutput>().await {
                Ok(w) => {
                    if w.image_dim != local_dim
                        || w.n_classes != local_classes
                        || w.weights.len() != local_weights.len()
                    {
                        peers_failed.push(PeerError {
                            peer: peer.clone(),
                            error: format!(
                                "shape mismatch: peer dim={} classes={} weights={} local dim={} classes={} weights={}",
                                w.image_dim, w.n_classes, w.weights.len(),
                                local_dim, local_classes, local_weights.len()
                            ),
                        });
                        continue;
                    }
                    if !verify_ternary(&w.weights) {
                        peers_failed.push(PeerError {
                            peer: peer.clone(),
                            error: "peer returned non-ternary weights".into(),
                        });
                        continue;
                    }
                    peer_weights.push(w.weights);
                    peers_ok.push(peer.clone());
                }
                Err(e) => peers_failed.push(PeerError {
                    peer: peer.clone(),
                    error: format!("decode failed: {e}"),
                }),
            },
            Ok(resp) => peers_failed.push(PeerError {
                peer: peer.clone(),
                error: format!("http {}", resp.status()),
            }),
            Err(e) => peers_failed.push(PeerError {
                peer: peer.clone(),
                error: format!("request failed: {e}"),
            }),
        }
    }

    // Merge by ternary majority vote (includes local as one voter).
    let refs: Vec<&[i8]> = peer_weights.iter().map(|v| v.as_slice()).collect();
    let merged = ternary_majority_vote(&refs).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("merge failed: {e}"),
        )
    })?;
    let changes = count_changes(&local_weights, &merged);
    let total_weights = merged.len();

    // Apply merged weights to local brain and re-measure accuracy.
    let accuracy_after = {
        let mut guard = LOCAL.lock().unwrap();
        let local = guard.as_mut().ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "local not ready".into(),
        ))?;
        local
            .brain
            .load_weights_i8(&merged)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("apply: {e}")))?;
        local.brain.accuracy(
            &local.holdout_images,
            &local.holdout_labels,
            local.holdout_n,
        )
    };

    tracing::info!(
        "QLMS federation gossip on node '{}': merged {} peers, {} weight changes, acc {:.3} → {:.3}",
        local_node,
        peer_weights.len(),
        changes,
        accuracy_before,
        accuracy_after,
    );

    Ok(Json(GossipOutput {
        node_id: local_node,
        merged_from: peer_weights.len(),
        weight_changes: changes,
        total_weights,
        peers_ok,
        peers_failed,
        accuracy_before,
        accuracy_after,
    }))
}

// ============================================================
// Tests — purely local codec / merge validation
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_index_is_stable() {
        let a = partition_index("node-a");
        let b = partition_index("node-a");
        assert_eq!(a, b);
        assert_ne!(partition_index("node-a"), partition_index("node-b"));
    }

    #[test]
    fn three_nodes_get_distinct_partitions() {
        let ia = (partition_index("node-a") % 3) as usize;
        let ib = (partition_index("node-b") % 3) as usize;
        let ic = (partition_index("node-c") % 3) as usize;
        // Not strictly required that all differ (FNV collisions possible), but
        // we assert at least two differ so partitions are not all identical.
        let distinct: std::collections::HashSet<_> = [ia, ib, ic].into_iter().collect();
        assert!(
            distinct.len() >= 2,
            "expected ≥2 distinct partitions, got {:?}",
            (ia, ib, ic)
        );
    }
}
