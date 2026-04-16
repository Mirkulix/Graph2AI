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
                let base = format!(
                    "{context}\n\n## Aktuelle Websuche-Ergebnisse ({})\n\n{}",
                    search.tool, search.output
                );
                // Follow-up: deep-read the top URL from the search output.
                // Extracted via a simple scan for http/https tokens (no regex
                // dep). We only fetch ONE URL — the point is a thorough read
                // of the best result, not mass-scraping.
                if let Some(top_url) = extract_first_http_url(&search.output) {
                    let fetched = tools::tool_fetch_url(&top_url).await;
                    if fetched.success {
                        tracing::info!(
                            "researcher: fetched {} bytes from {}",
                            fetched.output.len(),
                            top_url
                        );
                        format!(
                            "{base}\n\n## Vollständiger Inhalt der wichtigsten Quelle\n\n{}",
                            fetched.output
                        )
                    } else {
                        tracing::warn!(
                            "researcher: fetch_url failed for {} — continuing without it ({})",
                            top_url,
                            fetched.output.chars().take(80).collect::<String>()
                        );
                        base
                    }
                } else {
                    base
                }
            } else {
                tracing::warn!(
                    "researcher: web search failed or empty — context unchanged ({})",
                    search.output.chars().take(80).collect::<String>()
                );
                context.to_string()
            }
        }
        AgentRole::Developer => {
            // Scan task for `READ:rel/path.ext` tokens and splice existing
            // workspace files into context so the Developer can iterate on
            // real code instead of hallucinating from scratch.
            let mut augmented = context.to_string();
            for token in task.split_whitespace() {
                if let Some(rest) = token.strip_prefix("READ:") {
                    // Trim trailing punctuation that writers naturally append
                    // (e.g. "READ:foo.rs," or "READ:foo.rs.").
                    let path = rest.trim_end_matches(|c: char| {
                        matches!(c, ',' | '.' | ';' | ':' | ')' | ']' | '}' | '!' | '?' | '"' | '\'')
                    });
                    if path.is_empty() {
                        continue;
                    }
                    let result = tools::tool_read_workspace_file(path);
                    if result.success {
                        tracing::info!(
                            "developer: READ:{} -> {} B loaded into context",
                            path,
                            result.output.len()
                        );
                        augmented = format!(
                            "{augmented}\n\n## Existierender Code — {path}\n\n{}",
                            result.output
                        );
                    } else {
                        tracing::warn!(
                            "developer: READ:{} failed ({})",
                            path,
                            result.output.chars().take(120).collect::<String>()
                        );
                    }
                }
            }
            augmented
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

/// Find the first http(s) URL in a blob of text without pulling in a
/// regex crate. We scan for a whitespace-delimited token whose prefix
/// is `http://` or `https://`, then trim trailing punctuation / closing
/// parens that markdown-style links and search snippets typically wrap
/// the URL in (e.g. `(https://example.com/article.html)`, `See:
/// https://example.com/x.` or `… https://example.com/y)`).
///
/// Returns `None` if the text contains no http token at all.
fn extract_first_http_url(text: &str) -> Option<String> {
    for token in text.split_whitespace() {
        // Strip leading brackets/parens that often wrap URLs in markdown
        // (e.g. `(https://…)` or `[https://…]`).
        let start = token.trim_start_matches(|c: char| {
            matches!(c, '(' | '[' | '<' | '"' | '\'')
        });
        if start.starts_with("http://") || start.starts_with("https://") {
            // Cut the URL at any trailing punctuation / closing bracket
            // characters that are not legal in URLs anyway.
            let end = start.trim_end_matches(|c: char| {
                matches!(
                    c,
                    ')' | ']' | '>' | '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\''
                )
            });
            if end.len() > "https://".len() {
                return Some(end.to_string());
            }
        }
    }
    None
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
