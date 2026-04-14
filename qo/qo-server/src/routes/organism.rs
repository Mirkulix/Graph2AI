//! Organism API — interact with the QLANG swarm organism.

use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Deserialize)]
pub struct OrganismInput {
    pub message: String,
}

#[derive(Serialize)]
pub struct OrganismOutput {
    pub text: String,
    pub specialist: String,
    pub confidence: f32,
    pub reasoning: Vec<String>,
    pub memory_stored: bool,
    pub specialists: Vec<SpecialistInfo>,
    pub total_interactions: usize,
    pub generation: u32,
    pub memory_items: usize,
}

#[derive(Serialize)]
pub struct SpecialistInfo {
    pub name: String,
    pub invocations: u64,
    pub success_rate: f32,
}

use once_cell::sync::Lazy;
static ORGANISM: Lazy<Mutex<qlang_runtime::organism::Organism>> = Lazy::new(|| {
    Mutex::new(qlang_runtime::organism::Organism::new(1000))
});

/// POST /api/organism/chat — send message to organism
pub async fn chat(Json(input): Json<OrganismInput>) -> Json<OrganismOutput> {
    let mut org = ORGANISM.lock().await;
    let resp = org.process(&input.message);

    let specialists: Vec<SpecialistInfo> = org.specialists.iter().map(|s| SpecialistInfo {
        name: s.name.clone(),
        invocations: s.invocations,
        success_rate: s.success_rate,
    }).collect();

    Json(OrganismOutput {
        text: resp.text,
        specialist: resp.specialist,
        confidence: resp.confidence,
        reasoning: resp.reasoning,
        memory_stored: resp.memory_stored,
        specialists,
        total_interactions: org.total_interactions(),
        generation: org.generation,
        memory_items: org.shared_memory.items.len(),
    })
}

/// POST /api/organism/evolve — trigger evolution
pub async fn evolve() -> Json<serde_json::Value> {
    let mut org = ORGANISM.lock().await;
    org.evolve();
    Json(serde_json::json!({
        "generation": org.generation,
        "specialists": org.specialist_count(),
        "interactions": org.total_interactions(),
    }))
}

#[derive(Deserialize)]
pub struct LoadModelInput {
    /// Path to the QLMB binary file (e.g. "data/mamba_30m_final.bin")
    pub path: String,
    /// Reference text for rebuilding the tokenizer (can be short)
    #[serde(default)]
    pub tokenizer_text: String,
}

/// POST /api/organism/load-model — load a trained Mamba LM into the organism
pub async fn load_model(Json(input): Json<LoadModelInput>) -> Json<serde_json::Value> {
    let mut org = ORGANISM.lock().await;
    let text = if input.tokenizer_text.is_empty() {
        "the cat sat on the mat the dog ran in the park".to_string()
    } else {
        input.tokenizer_text
    };
    match org.add_language_model(&input.path, &text) {
        Ok(()) => Json(serde_json::json!({
            "ok": true,
            "message": format!("Language model loaded from {}", input.path),
            "specialists": org.specialist_count(),
        })),
        Err(e) => Json(serde_json::json!({
            "ok": false,
            "error": e,
        })),
    }
}

/// Lightweight snapshot used by other routes (e.g. /api/neo/*).
pub struct OrganismSnapshot {
    pub generation: u32,
    pub interactions: usize,
    pub specialists: usize,
    pub items: Vec<String>,
}

pub async fn organism_snapshot() -> OrganismSnapshot {
    let org = ORGANISM.lock().await;
    OrganismSnapshot {
        generation: org.generation,
        interactions: org.total_interactions(),
        specialists: org.specialist_count(),
        items: org
            .shared_memory
            .items
            .iter()
            .map(|(k, _)| k.clone())
            .collect(),
    }
}

/// GET /api/organism/status — get organism status
pub async fn status() -> Json<serde_json::Value> {
    let org = ORGANISM.lock().await;
    let specialists: Vec<serde_json::Value> = org.specialists.iter().map(|s| {
        serde_json::json!({
            "name": s.name,
            "invocations": s.invocations,
            "success_rate": s.success_rate,
        })
    }).collect();

    Json(serde_json::json!({
        "generation": org.generation,
        "specialists": specialists,
        "total_interactions": org.total_interactions(),
        "memory_items": org.shared_memory.items.len(),
    }))
}
