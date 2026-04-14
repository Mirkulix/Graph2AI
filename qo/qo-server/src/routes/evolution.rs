use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use qo_evolution::{Pattern, Proposal, QuantumSummary, SystemStats};
use qo_memory::graph_builders;
use serde::Serialize;
use std::sync::Arc;

use crate::AppState;

#[derive(Serialize)]
pub struct AnalyzeResponse {
    pub patterns_detected: Vec<Pattern>,
    pub proposals_generated: Vec<Proposal>,
    pub stats: SystemStats,
}

#[derive(Serialize)]
pub struct ApproveRejectResponse {
    pub success: bool,
    pub id: u64,
}

/// GET /api/evolution/state
pub async fn quantum_state(
    State(state): State<Arc<AppState>>,
) -> Result<Json<QuantumSummary>, (StatusCode, String)> {
    let quantum = state.quantum.lock().await;
    Ok(Json(quantum.summary()))
}

/// GET /api/evolution/patterns
pub async fn list_patterns(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Pattern>>, (StatusCode, String)> {
    let detector = state.patterns.lock().await;
    Ok(Json(detector.all_patterns().to_vec()))
}

/// GET /api/evolution/proposals
pub async fn list_proposals(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Proposal>>, (StatusCode, String)> {
    let engine = state.proposals.lock().await;
    Ok(Json(engine.all().to_vec()))
}

/// POST /api/evolution/proposals/:id/approve
pub async fn approve_proposal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ApproveRejectResponse>, (StatusCode, String)> {
    let (success, proposal_title) = {
        let mut engine = state.proposals.lock().await;
        let title = engine.all().iter().find(|p| p.id == id).map(|p| p.title.clone()).unwrap_or_default();
        let success = engine.approve(id);
        (success, title)
    };
    if success {
        if let Err(e) = state.store.log_action("proposal_approved", &proposal_title, &format!("id={}", id)) {
            tracing::warn!("failed to log proposal_approved action: {e}");
        }
        // Persist updated proposal
        {
            let engine = state.proposals.lock().await;
            if let Some(proposal) = engine.all().iter().find(|p| p.id == id) {
                if let Ok(json) = serde_json::to_string(proposal) {
                    if let Err(e) = state.store.save_proposal(id, &json) {
                        tracing::warn!("failed to persist approved proposal {}: {e}", id);
                    }
                }
            }
        }
        Ok(Json(ApproveRejectResponse { success, id }))
    } else {
        Err((StatusCode::NOT_FOUND, format!("Proposal {} not found", id)))
    }
}

/// POST /api/evolution/proposals/:id/reject
pub async fn reject_proposal(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> Result<Json<ApproveRejectResponse>, (StatusCode, String)> {
    let (success, proposal_title) = {
        let mut engine = state.proposals.lock().await;
        let title = engine.all().iter().find(|p| p.id == id).map(|p| p.title.clone()).unwrap_or_default();
        let success = engine.reject(id);
        (success, title)
    };
    if success {
        if let Err(e) = state.store.log_action("proposal_rejected", &proposal_title, &format!("id={}", id)) {
            tracing::warn!("failed to log proposal_rejected action: {e}");
        }
        // Persist updated proposal
        {
            let engine = state.proposals.lock().await;
            if let Some(proposal) = engine.all().iter().find(|p| p.id == id) {
                if let Ok(json) = serde_json::to_string(proposal) {
                    if let Err(e) = state.store.save_proposal(id, &json) {
                        tracing::warn!("failed to persist rejected proposal {}: {e}", id);
                    }
                }
            }
        }
        Ok(Json(ApproveRejectResponse { success, id }))
    } else {
        Err((StatusCode::NOT_FOUND, format!("Proposal {} not found", id)))
    }
}

/// POST /api/evolution/analyze
/// Reads current system state, detects patterns, generates proposals.
pub async fn analyze(
    State(state): State<Arc<AppState>>,
) -> Result<Json<AnalyzeResponse>, (StatusCode, String)> {
    // Gather stats from agents
    let (total_tasks, tasks_completed, tasks_failed, agents_active, agents_idle) = {
        let registry = state.agents.lock().await;
        let summaries = registry.list_agents();
        let total_completed: u32 = summaries.iter().map(|a| a.tasks_completed).sum();
        let total_failed: u32 = summaries.iter().map(|a| a.tasks_failed).sum();
        let active = registry.active_count();
        let idle = registry.idle_count();
        let total = total_completed + total_failed;
        (total, total_completed, total_failed, active, idle)
    };

    // Gather energy from consciousness
    let avg_energy = {
        let consciousness = state.consciousness.lock().await;
        consciousness.energy
    };

    // Estimate completed_streak from tasks_completed (simplified: use completed as streak proxy)
    let completed_streak = tasks_completed.min(10);

    let stats = SystemStats {
        total_tasks,
        tasks_completed,
        tasks_failed,
        agents_active,
        agents_idle,
        avg_energy,
        completed_streak,
    };

    // Run pattern detection
    let detected_pattern_names: Vec<String> = {
        let mut detector = state.patterns.lock().await;
        detector.analyze(&stats).iter().map(|p| p.name.clone()).collect()
    };

    // Collect detected patterns for generating proposals
    let detected_patterns: Vec<Pattern> = {
        let detector = state.patterns.lock().await;
        detector
            .all_patterns()
            .iter()
            .filter(|p| detected_pattern_names.contains(&p.name))
            .cloned()
            .collect()
    };

    // Generate proposals from detected patterns
    let new_proposal_ids: Vec<u64> = {
        let pattern_refs: Vec<&Pattern> = detected_patterns.iter().collect();
        let mut engine = state.proposals.lock().await;
        engine
            .generate_from_patterns(&pattern_refs)
            .iter()
            .map(|p| p.id)
            .collect()
    };

    let new_proposals: Vec<Proposal> = {
        let engine = state.proposals.lock().await;
        engine
            .all()
            .iter()
            .filter(|p| new_proposal_ids.contains(&p.id))
            .cloned()
            .collect()
    };

    let details = format!(
        "{} Muster erkannt, {} Vorschläge generiert",
        detected_pattern_names.len(),
        new_proposals.len()
    );
    if let Err(e) = state.store.log_action("evolution_analyzed", "Systemanalyse durchgeführt", &details) {
        tracing::warn!("failed to log evolution_analyzed action: {e}");
    }

    // Persist patterns
    {
        let detector = state.patterns.lock().await;
        for pattern in detector.all_patterns() {
            if let Ok(json) = serde_json::to_string(pattern) {
                if let Err(e) = state.store.save_pattern(pattern.id, &json) {
                    tracing::warn!("failed to persist pattern {}: {e}", pattern.id);
                }
            }
        }
    }

    // Persist proposals
    {
        let engine = state.proposals.lock().await;
        for proposal in engine.all() {
            if let Ok(json) = serde_json::to_string(proposal) {
                if let Err(e) = state.store.save_proposal(proposal.id, &json) {
                    tracing::warn!("failed to persist proposal {}: {e}", proposal.id);
                }
            }
        }
    }

    // Persist quantum state
    {
        let quantum = state.quantum.lock().await;
        if let Ok(json) = serde_json::to_string(&*quantum) {
            if let Err(e) = state.store.save_quantum_state(&json) {
                tracing::warn!("failed to persist quantum state: {e}");
            }
        }
    }

    // Build and store QLANG graph for this evolution cycle — JSON + QLBG binary
    {
        let graph = graph_builders::build_evolution_graph(
            detected_pattern_names.len(),
            new_proposals.len(),
        );
        let store_id = match state.graph_store.store(&graph) {
            Ok(id) => id,
            Err(e) => { tracing::warn!("failed to store evolution graph: {e}"); 0 }
        };

        // Real QLANG graph for evolution
        let mut qlang_graph = qlang_core::graph::Graph::new("evolution_cycle");
        let str_type = qlang_core::tensor::TensorType::new(
            qlang_core::tensor::Dtype::Utf8,
            qlang_core::tensor::Shape::scalar(),
        );
        let input = qlang_graph.add_node(
            qlang_core::ops::Op::Input { name: "stats".into() },
            vec![], vec![str_type.clone()],
        );
        let analyze = qlang_graph.add_node(
            qlang_core::ops::Op::Entropy, // Pattern analysis as entropy measurement
            vec![str_type.clone()], vec![str_type.clone()],
        );
        let output = qlang_graph.add_node(
            qlang_core::ops::Op::Output { name: "patterns".into() },
            vec![str_type.clone()], vec![],
        );
        qlang_graph.add_edge(input, 0, analyze, 0, str_type.clone());
        qlang_graph.add_edge(analyze, 0, output, 0, str_type.clone());

        let binary = qlang_core::binary::to_binary(&qlang_graph);
        tracing::info!("Evolution QLANG graph: {} nodes, {} bytes .qlbg", qlang_graph.nodes.len(), binary.len());
        if let Err(e) = state.graph_store.store_binary_graph(store_id, &binary) {
            tracing::warn!("failed to store evolution QLBG binary: {e}");
        }
    }

    Ok(Json(AnalyzeResponse {
        patterns_detected: detected_patterns,
        proposals_generated: new_proposals,
        stats,
    }))
}
