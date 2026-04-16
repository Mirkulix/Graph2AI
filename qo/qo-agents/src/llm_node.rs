use qo_llm::LlmRouter;
use crate::agent::AgentRole;
use crate::tools;
use std::collections::HashMap;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType, TensorData};

/// Build a QLANG graph for an agent task and execute it via the QLANG runtime.
/// The graph contains an OllamaChat node — the executor handles it natively.
fn execute_via_qlang(
    role: AgentRole,
    context: &str,
    task: &str,
) -> Result<(String, Graph, u64), String> {
    let mut g = Graph::new(format!("agent_{}", role.name().to_lowercase()));
    let str_type = TensorType::new(Dtype::Utf8, Shape::scalar());

    let input = g.add_node(
        Op::Input { name: "task".into() },
        vec![], vec![str_type.clone()],
    );

    let ollama_model = std::env::var("OLLAMA_MODEL")
        .unwrap_or_else(|_| "qwen2.5:3b".to_string());
    let chat = g.add_node(
        Op::OllamaChat { model: ollama_model },
        vec![str_type.clone()], vec![str_type.clone()],
    );

    let output = g.add_node(
        Op::Output { name: "result".into() },
        vec![str_type.clone()], vec![],
    );

    g.add_edge(input, 0, chat, 0, str_type.clone());
    g.add_edge(chat, 0, output, 0, str_type.clone());

    // Build messages JSON for OllamaChat
    let messages = serde_json::json!([
        {"role": "system", "content": role.system_prompt()},
        {"role": "user", "content": format!("Kontext: {context}\n\nAufgabe: {task}")}
    ]);

    let mut inputs = HashMap::new();
    inputs.insert("task".to_string(), TensorData::from_string(&messages.to_string()));

    let start = std::time::Instant::now();
    let result = qlang_runtime::executor::execute(&g, inputs)
        .map_err(|e| format!("QLANG agent execution failed: {e}"))?;
    let duration = start.elapsed().as_millis() as u64;

    let response = result.outputs.get("result")
        .and_then(|t| t.as_string())
        .unwrap_or_else(|| "Keine Antwort vom QLANG Executor".to_string());

    tracing::info!(
        "QLANG agent {}: {} nodes executed, {}ms",
        role.name(), result.stats.nodes_executed, duration
    );

    Ok((response, g, duration))
}

/// Direct LLM call routed through the agent's **preferred provider**
/// plus role-specific tool augmentation.
///
/// Calls web_search for the Researcher role BEFORE the LLM sees the
/// prompt. The fetched snippets are spliced into the context so the
/// model has real facts to work with instead of stale pre-training
/// memory. This path is hit from every orchestration route (parallel
/// subtasks, retries, chat fallbacks) — so the web-search fix lands in
/// one place and benefits all flows.
///
/// The router honours the preferred-tier hint when the tier is
/// configured, otherwise falls back to complexity-based auto routing.
pub async fn llm_reason(
    llm: &LlmRouter,
    role: AgentRole,
    context: &str,
    task: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Role-specific tool augmentation.
    let augmented_context = match role {
        AgentRole::Researcher => {
            let search = tools::tool_web_search(task).await;
            if search.success {
                tracing::info!(
                    "researcher: {} provided {} B of context",
                    search.tool,
                    search.output.len()
                );
                format!(
                    "{context}\n\n## Aktuelle Websuche-Ergebnisse ({})\n\n{}",
                    search.tool, search.output
                )
            } else {
                tracing::warn!(
                    "researcher: web search failed or empty — context unchanged ({})",
                    search.output.chars().take(80).collect::<String>()
                );
                context.to_string()
            }
        }
        _ => context.to_string(),
    };

    let messages = vec![
        ("system".to_string(), role.system_prompt().to_string()),
        (
            "user".to_string(),
            format!("Kontext: {augmented_context}\n\nAufgabe: {task}"),
        ),
    ];
    let preferred = qo_llm::LlmRouter::tier_from_hint(role.preferred_provider());
    let (body, tier) = llm.chat_preferring(preferred, messages).await?;
    tracing::info!(
        "agent {}: response from {:?} (preferred: {})",
        role.name(),
        tier,
        role.preferred_provider()
    );
    Ok(body)
}

/// Execute agent task WITH tools — uses QLANG executor for LLM calls
pub async fn agent_execute_with_tools(
    llm: &LlmRouter,
    role: AgentRole,
    context: &str,
    task: &str,
    values: &qo_values::ValueScores,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    match role {
        AgentRole::Guardian => {
            // Guardian uses DETERMINISTIC value check, NOT LLM
            let check = tools::tool_values_check(task, values);
            Ok(check.output)
        }
        AgentRole::Researcher => {
            // Researcher: web search first, then QLANG executor for LLM
            let search_result = tools::tool_web_search(task).await;
            let enriched_context = if search_result.success {
                format!("{context}\n\nWeb-Recherche Ergebnisse:\n{}", search_result.output)
            } else {
                context.to_string()
            };
            // Try QLANG executor, fallback to direct LLM
            let ctx = enriched_context.clone();
            let task_owned = task.to_string();
            match tokio::task::spawn_blocking(move || execute_via_qlang(role, &ctx, &task_owned)).await {
                Ok(Ok((response, _, _))) => Ok(response),
                _ => llm_reason(llm, role, &enriched_context, task).await,
            }
        }
        AgentRole::Developer => {
            // Developer: file info first, then QLANG executor
            let project_info = tools::tool_shell("ls -la");
            let enriched_context = format!("{context}\n\nProjekt-Verzeichnis:\n{}", project_info.output);
            let ctx = enriched_context.clone();
            let task_owned = task.to_string();
            match tokio::task::spawn_blocking(move || execute_via_qlang(role, &ctx, &task_owned)).await {
                Ok(Ok((response, _, _))) => Ok(response),
                _ => llm_reason(llm, role, &enriched_context, task).await,
            }
        }
        _ => {
            // CEO, Strategist, Artisan: QLANG executor for LLM
            let ctx = context.to_string();
            let task_owned = task.to_string();
            match tokio::task::spawn_blocking(move || execute_via_qlang(role, &ctx, &task_owned)).await {
                Ok(Ok((response, _, _))) => Ok(response),
                _ => llm_reason(llm, role, context, task).await,
            }
        }
    }
}
