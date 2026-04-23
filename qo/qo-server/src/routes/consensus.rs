//! POST /api/consensus — fan-out one prompt to N agents in parallel and
//! return all replies plus a Jaccard-overlap consensus signal.
//!
//! This is the multi-LLM diversity demo: a single user prompt is asked
//! against several agent personas concurrently, then the replies are
//! lexically compared to estimate how much the agents agree.

use axum::{extract::State, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::AppState;

/// Inbound request — one prompt, a list of agent personas to ask.
#[derive(Debug, Deserialize)]
pub struct ConsensusRequest {
    pub prompt: String,
    pub agents: Vec<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

/// One agent's answer to the shared prompt.
#[derive(Debug, Serialize)]
pub struct AgentReply {
    pub agent: String,
    pub content: String,
    pub latency_ms: u64,
    pub ok: bool,
    pub error: Option<String>,
}

/// Aggregate signal across all replies.
#[derive(Debug, Serialize)]
pub struct Summary {
    pub total_replies: usize,
    pub successful: usize,
    pub failed: usize,
    pub avg_latency_ms: u64,
    /// Lexical Jaccard overlap on word-token sets. Range [0.0, 1.0].
    pub consensus_score: f64,
    pub consensus_label: String,
    /// Pseudo-semantic score: cosine similarity on character-trigram
    /// vectors. Captures partial-word / morphological overlap that
    /// pure word-set Jaccard misses. Range [0.0, 1.0]. Real embedding
    /// models (Ollama / OpenAI / etc.) can be plugged in later via
    /// the LlmRouter without changing this field's contract.
    pub consensus_score_semantic: f64,
    pub consensus_label_semantic: String,
}

/// Outbound response — echo of inputs + every reply + summary.
#[derive(Debug, Serialize)]
pub struct ConsensusResponse {
    pub prompt: String,
    pub agents_asked: Vec<String>,
    pub replies: Vec<AgentReply>,
    pub summary: Summary,
}

/// System-prompt persona table. Mirrors the helper used by the agent
/// mailbox loop in `lib.rs` so consensus replies match the same role
/// voices the rest of the system uses. Kept as a local copy rather than
/// extracted to avoid touching the mailbox loop per the implementation
/// constraints; if a third caller ever needs it, hoist this into a
/// shared module then.
fn system_prompt_for(role: &str) -> &'static str {
    match role {
        "ceo"        => "You are CEO, a coordinator agent. Decompose the user's request into clear steps, suggest which specialist should handle each step (developer, researcher, guardian, strategist, artisan), and give a one-paragraph executive summary.",
        "developer"  => "You are Developer, a senior software engineer. Review code, suggest refactors, write functions, and explain trade-offs. Be precise. Use code blocks for any code you produce.",
        "researcher" => "You are Researcher, a knowledge synthesizer. Find relevant information, cite sources when possible, summarize concisely, and flag uncertainty.",
        "guardian"   => "You are Guardian, a security and safety reviewer. Find vulnerabilities, unsafe patterns, missing validation, and compliance gaps. Suggest concrete mitigations.",
        "strategist" => "You are Strategist, a planning advisor. Lay out multi-step strategies, trade-offs, and second-order effects. Prefer numbered plans.",
        "artisan"    => "You are Artisan, a creative implementer. Generate concrete artifacts (text, prose, examples, snippets) that match the user's intent.",
        _ => "You are an AI assistant. Help the user with their request.",
    }
}

/// POST /api/consensus
pub async fn consensus(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ConsensusRequest>,
) -> Result<Json<ConsensusResponse>, (StatusCode, String)> {
    // 1. Validate at the boundary.
    if req.prompt.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "prompt empty".to_string()));
    }
    if req.agents.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "agents empty".to_string()));
    }

    let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(30_000));
    let llm = state.llm.clone();

    // 2. Fan-out: spawn one task per requested agent.
    let mut handles = Vec::with_capacity(req.agents.len());
    for agent_name in &req.agents {
        let llm = llm.clone();
        let prompt = req.prompt.clone();
        let system = req
            .system_prompt
            .clone()
            .unwrap_or_else(|| system_prompt_for(agent_name).to_string());
        let agent_name = agent_name.clone();
        handles.push(tokio::spawn(async move {
            let start = Instant::now();
            let messages = vec![
                ("system".to_string(), system),
                ("user".to_string(), prompt),
            ];
            // Force DeepSeek tier explicitly — auto-router never selects it.
            // model: None lets DeepSeek use its configured default.
            let result = tokio::time::timeout(
                timeout,
                llm.chat_with_model(Some(qo_llm::Tier::DeepSeek), None, messages),
            )
            .await;
            let latency_ms = start.elapsed().as_millis() as u64;
            match result {
                Ok(Ok((text, _tier))) => AgentReply {
                    agent: agent_name,
                    content: text,
                    latency_ms,
                    ok: true,
                    error: None,
                },
                Ok(Err(e)) => AgentReply {
                    agent: agent_name,
                    content: String::new(),
                    latency_ms,
                    ok: false,
                    error: Some(e.to_string()),
                },
                Err(_) => AgentReply {
                    agent: agent_name,
                    content: String::new(),
                    latency_ms,
                    ok: false,
                    error: Some("timeout".to_string()),
                },
            }
        }));
    }

    // 3. Collect all replies (await every handle, even failed ones).
    let mut replies = Vec::with_capacity(handles.len());
    for h in handles {
        match h.await {
            Ok(reply) => replies.push(reply),
            Err(e) => {
                tracing::warn!("consensus: agent task join error: {e}");
            }
        }
    }

    // 4. Aggregate.
    let summary = compute_summary(&replies);

    Ok(Json(ConsensusResponse {
        prompt: req.prompt,
        agents_asked: req.agents,
        replies,
        summary,
    }))
}

fn compute_summary(replies: &[AgentReply]) -> Summary {
    let total_replies = replies.len();
    let successful = replies.iter().filter(|r| r.ok).count();
    let failed = total_replies - successful;
    let avg_latency_ms = if total_replies > 0 {
        replies.iter().map(|r| r.latency_ms).sum::<u64>() / total_replies as u64
    } else {
        0
    };

    // Both consensus signals are computed across SUCCESSFUL replies only.
    // They are cheap, deterministic, and add no extra LLM call.
    let texts: Vec<&str> = replies
        .iter()
        .filter(|r| r.ok)
        .map(|r| r.content.as_str())
        .collect();

    // 1) Lexical: average pairwise Jaccard on word-token sets. Range [0.0, 1.0].
    let consensus_score = jaccard_overlap(&texts);
    let consensus_label = label_for_score(consensus_score);

    // 2) Pseudo-semantic: average pairwise cosine on character-trigram vectors.
    //    Stronger than Jaccard because it captures partial-word matches and
    //    morphological variants (e.g. "secure"/"security"/"securing" share
    //    most trigrams). Real embeddings would still be better and can be
    //    swapped in later through the LlmRouter.
    let consensus_score_semantic = trigram_cosine_pairwise(&texts);
    let consensus_label_semantic = label_for_score(consensus_score_semantic);

    Summary {
        total_replies,
        successful,
        failed,
        avg_latency_ms,
        consensus_score,
        consensus_label,
        consensus_score_semantic,
        consensus_label_semantic,
    }
}

/// Shared 4-bucket label ladder so both scores use identical thresholds.
fn label_for_score(s: f64) -> String {
    match s {
        s if s >= 0.6 => "strong-agreement",
        s if s >= 0.35 => "majority-agrees",
        s if s >= 0.15 => "mixed-signals",
        _ => "diverse-opinions",
    }
    .to_string()
}

/// Average pairwise Jaccard overlap across all texts.
/// Empty / single-text inputs return 1.0 (trivially "agrees with itself").
fn jaccard_overlap(texts: &[&str]) -> f64 {
    if texts.len() < 2 {
        return 1.0;
    }
    let token_sets: Vec<HashSet<String>> = texts
        .iter()
        .map(|t| {
            t.split_whitespace()
                .map(|w| {
                    w.trim_matches(|c: char| !c.is_alphanumeric())
                        .to_lowercase()
                })
                .filter(|w| w.len() > 3) // drop short/stopword-ish tokens
                .collect()
        })
        .collect();

    let mut sum = 0.0;
    let mut pairs = 0usize;
    for i in 0..token_sets.len() {
        for j in (i + 1)..token_sets.len() {
            let intersection = token_sets[i].intersection(&token_sets[j]).count() as f64;
            let union = token_sets[i].union(&token_sets[j]).count() as f64;
            if union > 0.0 {
                sum += intersection / union;
                pairs += 1;
            }
        }
    }
    if pairs == 0 {
        0.0
    } else {
        sum / pairs as f64
    }
}

/// Average pairwise cosine similarity on character-trigram vectors.
/// Pure-Rust, no extra deps. Range [0.0, 1.0]. Higher = more similar.
///
/// Why trigrams over Jaccard-on-words for an MVP "semantic" signal:
/// - captures partial-word overlap (root sharing, plurals, conjugations)
/// - robust to small spelling/casing differences
/// - still cheap and deterministic
///
/// This is NOT a real embedding model — it does not capture true synonymy
/// (e.g. "car" vs "automobile" still score low). A real embedding backend
/// (Ollama nomic-embed-text, OpenAI /v1/embeddings, etc.) can be plugged
/// in later via qo-llm without changing the public score field.
fn trigram_cosine_pairwise(texts: &[&str]) -> f64 {
    if texts.len() < 2 {
        return 1.0;
    }
    let vecs: Vec<HashMap<String, f64>> =
        texts.iter().map(|t| trigram_vector(t)).collect();
    let mut sum = 0.0;
    let mut pairs = 0usize;
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            sum += cosine(&vecs[i], &vecs[j]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        sum / pairs as f64
    }
}

fn trigram_vector(text: &str) -> HashMap<String, f64> {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let chars: Vec<char> = normalized.chars().collect();
    let mut counts: HashMap<String, f64> = HashMap::new();
    for w in chars.windows(3) {
        let trigram: String = w.iter().collect();
        *counts.entry(trigram).or_insert(0.0) += 1.0;
    }
    counts
}

fn cosine(a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> f64 {
    let mut dot = 0.0;
    for (k, va) in a {
        if let Some(vb) = b.get(k) {
            dot += va * vb;
        }
    }
    let mag_a: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let mag_b: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_texts_is_one() {
        let texts = vec!["alpha beta gamma delta", "alpha beta gamma delta"];
        assert!((jaccard_overlap(&texts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_texts_is_zero() {
        let texts = vec!["alpha beta gamma", "delta epsilon zeta"];
        assert!(jaccard_overlap(&texts) < 1e-9);
    }

    #[test]
    fn jaccard_single_text_is_one() {
        let texts = vec!["alpha beta"];
        assert!((jaccard_overlap(&texts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_partial_overlap() {
        // {alpha, beta, gamma} vs {alpha, beta, delta} → intersection 2, union 4 → 0.5
        let texts = vec!["alpha beta gamma", "alpha beta delta"];
        let score = jaccard_overlap(&texts);
        assert!((score - 0.5).abs() < 1e-9, "expected 0.5, got {score}");
    }

    #[test]
    fn summary_label_thresholds() {
        assert_eq!(label_for_score(0.9), "strong-agreement");
        assert_eq!(label_for_score(0.5), "majority-agrees");
        assert_eq!(label_for_score(0.2), "mixed-signals");
        assert_eq!(label_for_score(0.0), "diverse-opinions");
    }

    #[test]
    fn trigram_identical_texts_is_one() {
        let texts = vec!["alpha beta gamma", "alpha beta gamma"];
        let s = trigram_cosine_pairwise(&texts);
        assert!((s - 1.0).abs() < 1e-9, "expected 1.0, got {s}");
    }

    #[test]
    fn trigram_single_text_is_one() {
        let texts = vec!["alpha"];
        assert!((trigram_cosine_pairwise(&texts) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn trigram_disjoint_texts_is_low() {
        // Wholly different character sets → near-zero cosine.
        let texts = vec!["aaaa", "zzzz"];
        let s = trigram_cosine_pairwise(&texts);
        assert!(s < 0.05, "expected near 0.0, got {s}");
    }

    #[test]
    fn trigram_beats_jaccard_on_morphology() {
        // Jaccard on these word sets is 0 (no exact word matches), but
        // trigram cosine should be clearly > 0 because they share roots.
        let texts = vec!["securing the system", "secure security systems"];
        let j = jaccard_overlap(&texts);
        let t = trigram_cosine_pairwise(&texts);
        assert!(t > j, "trigram ({t}) should exceed jaccard ({j})");
    }
}
