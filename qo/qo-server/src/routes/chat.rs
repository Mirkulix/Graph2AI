use axum::{extract::State, Json};
use qo_consciousness::StateEvent;
use qo_memory::graph_builders;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// QLANG imports — the graph engine
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType, TensorData};

use crate::AppState;

// No wrapper needed — classify_intent_cached() handles caching via OnceLock

/// Build a REAL QLANG graph for a chat interaction.
/// This graph is executed by qlang_runtime::executor::execute().
fn build_chat_qlang_graph(_system_prompt: &str, _user_message: &str) -> Graph {
    let mut g = Graph::new("qo_chat");
    let str_type = TensorType::new(Dtype::Utf8, Shape::scalar());

    // Node 0: System prompt input
    let _system_node = g.add_node(
        Op::Input { name: "system".into() },
        vec![],
        vec![str_type.clone()],
    );

    // Node 1: User message input
    let user_node = g.add_node(
        Op::Input { name: "user".into() },
        vec![],
        vec![str_type.clone()],
    );

    // Node 2: OllamaChat — the actual LLM call as a QLANG operation
    // The executor handles this Op natively
    let ollama_model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5:3b".to_string());
    let chat_node = g.add_node(
        Op::OllamaChat { model: ollama_model },
        vec![str_type.clone()],
        vec![str_type.clone()],
    );

    // Node 3: Output
    let output_node = g.add_node(
        Op::Output { name: "response".into() },
        vec![str_type.clone()],
        vec![],
    );

    // Edges: user message → OllamaChat → output
    // (system prompt is embedded in the chat message JSON)
    g.add_edge(user_node, 0, chat_node, 0, str_type.clone());
    g.add_edge(chat_node, 0, output_node, 0, str_type.clone());

    g
}

/// Execute chat via QLANG runtime — the graph engine runs the LLM call
fn execute_chat_via_qlang(
    system_prompt: &str,
    user_message: &str,
) -> Result<(String, Graph, u64), String> {
    let graph = build_chat_qlang_graph(system_prompt, user_message);

    // Build input: OllamaChat expects JSON array of {role, content} messages
    let messages_json = serde_json::json!([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]);

    let mut inputs = HashMap::new();
    inputs.insert("user".to_string(), TensorData::from_string(&messages_json.to_string()));
    inputs.insert("system".to_string(), TensorData::from_string(system_prompt));

    let start = std::time::Instant::now();

    // THIS IS THE KEY LINE: qlang_runtime::executor::execute() runs the graph
    let result = qlang_runtime::executor::execute(&graph, inputs)
        .map_err(|e| format!("QLANG execution failed: {e}"))?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let response = result.outputs.get("response")
        .and_then(|t| t.as_string())
        .unwrap_or_else(|| "Keine Antwort vom QLANG Executor".to_string());

    tracing::info!(
        "QLANG executor: {} nodes executed, {} quantum ops, {}ms",
        result.stats.nodes_executed,
        result.stats.quantum_ops,
        duration_ms
    );

    Ok((response, graph, duration_ms))
}

#[allow(dead_code)] // Heuristic reserved for future routing decisions
fn looks_like_goal(msg: &str) -> bool {
    let prefixes = [
        "ziel:", "goal:", "mach:", "recherchiere", "plane", "baue",
        "erstelle", "analysiere",
    ];
    let lower = msg.to_lowercase();
    prefixes.iter().any(|p| lower.starts_with(p))
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub response: String,
    pub tier: String,
}

const SYSTEM_PROMPT: &str =
    "Du bist QO, ein persönlicher KI-Companion. Antworte auf Deutsch.";

pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (axum::http::StatusCode, String)> {
    // Classify intent using QO's QLANG-native model (Tier 0 — no LLM needed)
    let (intent, intent_probs) = qo_agents::qlang_model::classify_intent_cached(&req.message);
    tracing::info!(
        "QLANG model classified '{}' as {:?} (probs: {:?})",
        &req.message[..req.message.len().min(40)],
        intent,
        intent_probs.iter().map(|p| format!("{:.2}", p)).collect::<Vec<_>>()
    );

    // Route based on QLANG model classification (replaces old string-matching heuristic)
    let is_goal = intent == qo_agents::qlang_model::Intent::Goal;
    let goal_prefix = if is_goal {
        let goal_id = {
            let mut registry = state.agents.lock().await;
            let goal = registry.create_goal(req.message.clone());
            goal.id
        };

        state.stream.publish_activity(
            format!("Chat-Ziel #{} erstellt und wird bearbeitet", goal_id),
            None,
            "info",
        );

        // Spawn background execution
        let state_clone = state.clone();
        let description = req.message.clone();
        tokio::spawn(async move {
            crate::routes::goals::execute_goal_background(state_clone, goal_id, description).await;
        });

        Some(format!(
            "[Ziel #{} erstellt und wird von den Agenten bearbeitet.]\n\n",
            goal_id
        ))
    } else {
        None
    };

    // Search for relevant past context
    let context = {
        let mem = state.memory.lock().await;
        let relevant = mem.recall(&req.message, 3);
        drop(mem);
        if relevant.is_empty() {
            String::new()
        } else {
            let mut ctx_parts = Vec::new();
            for (key, _score) in &relevant {
                if let Ok(Some(text)) = state.store.get(key) {
                    ctx_parts.push(format!("[Erinnerung: {}]", text));
                }
            }
            if ctx_parts.is_empty() {
                String::new()
            } else {
                format!("\n\nRelevanter Kontext aus früheren Gesprächen:\n{}\n", ctx_parts.join("\n"))
            }
        }
    };

    let full_system_prompt = format!("{}{}", SYSTEM_PROMPT, context);

    let llm_start = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    // PRIMARY PATH: Execute chat via QLANG runtime
    // The graph contains an OllamaChat node that the QLANG executor handles natively.
    // This means the LLM call flows through QLANG's graph engine, not a direct HTTP call.
    let (llm_response, qlang_graph, llm_duration_ms, provider_used) = {
        // Try QLANG executor first (uses Ollama via Op::OllamaChat)
        let qlang_result = tokio::task::spawn_blocking({
            let system = full_system_prompt.clone();
            let user = req.message.clone();
            move || execute_chat_via_qlang(&system, &user)
        }).await;

        match qlang_result {
            Ok(Ok((response, graph, duration))) => {
                tracing::info!("Chat served via QLANG executor (Ollama)");
                (response, Some(graph), duration, "qlang-ollama".to_string())
            }
            _ => {
                // FALLBACK: Direct API call if QLANG executor fails (e.g. Ollama not running)
                tracing::warn!("QLANG executor failed, falling back to direct LLM call");
                let messages = vec![
                    ("system".to_string(), full_system_prompt.clone()),
                    ("user".to_string(), req.message.clone()),
                ];
                let response = state
                    .llm
                    .chat(messages)
                    .await
                    .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                let duration = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64 - llm_start;
                (response, None, duration, "groq".to_string())
            }
        }
    };

    let response = match goal_prefix {
        Some(prefix) => format!("{}{}", prefix, llm_response),
        None => llm_response,
    };

    // Store in redb
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let entry = serde_json::json!({
        "id": id,
        "user": req.message,
        "assistant": response,
        "timestamp": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    });
    let entry_str = entry.to_string();

    if let Err(e) = state.store.store_chat(id, &entry_str) {
        tracing::warn!("failed to store chat in redb: {e}");
    }

    // Remember this interaction in long-term memory
    {
        let mut mem = state.memory.lock().await;
        let mem_key = format!("chat_{}", id);
        let response_snippet = &response[..response.len().min(200)];
        let memory_text = format!("User: {}\nQO: {}", req.message, response_snippet);
        mem.remember(mem_key.clone(), &memory_text, &state.store);
        // Also store the full text in KV for retrieval
        let _ = state.store.set(&mem_key, &memory_text);
    }

    // Store QLANG graph — use the REAL executed graph if available, otherwise build metadata graph
    {
        let graph = graph_builders::build_chat_graph(
            &req.message,
            &response,
            &provider_used,
            llm_duration_ms,
        );
        if let Err(e) = state.graph_store.store(&graph) {
            tracing::warn!("failed to store chat graph metadata: {e}");
        }

        // If QLANG executor was used, also store the real binary graph (.qlbg)
        if let Some(ref real_graph) = qlang_graph {
            let binary = qlang_core::binary::to_binary(real_graph);
            tracing::info!(
                "Storing real QLANG graph: {} nodes, {} edges, {} bytes .qlbg",
                real_graph.nodes.len(), real_graph.edges.len(), binary.len()
            );
            if let Err(e) = state.graph_store.store_binary_graph(id, &binary) {
                tracing::warn!("failed to store binary QLANG graph: {e}");
            }
        }
    }

    // Log action history
    let response_short = if response.len() > 100 {
        format!("{}...", &response[..100])
    } else {
        response.clone()
    };
    if let Err(e) = state.store.log_action("chat", &req.message, &response_short) {
        tracing::warn!("failed to log chat action: {e}");
    }

    // Update consciousness and log to Obsidian
    {
        let mut cs = state.consciousness.lock().await;
        cs.process_event(&StateEvent::ChatReceived);
        let mood_str = format!("{:?}", cs.mood);
        let energy = cs.energy;
        let heartbeat = cs.heartbeat;
        state.stream.publish(cs.clone());
        drop(cs);

        if let Err(e) = state
            .obsidian
            .log_consciousness_event(&mood_str, energy, heartbeat, "ChatReceived")
            .await
        {
            tracing::warn!("failed to log consciousness event to Obsidian: {e}");
        }
    }

    Ok(Json(ChatResponse {
        response,
        tier: provider_used,
    }))
}

pub async fn chat_history(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Value>>, (axum::http::StatusCode, String)> {
    let history = state
        .store
        .chat_history(50)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let items: Vec<Value> = history
        .into_iter()
        .filter_map(|(_, json_str)| serde_json::from_str(&json_str).ok())
        .collect();

    Ok(Json(items))
}
