//! MCP (Model Context Protocol) server endpoint for QLANG.
//!
//! Exposes QLANG itself as an MCP server so external MCP clients
//! (Claude Desktop, Cursor, ChatGPT Desktop, …) can call QLANG agents
//! as tools. Speaks JSON-RPC 2.0 per the MCP spec (2024-11-05).
//!
//! Three tools are exposed:
//!   * `qlang_research`           — web research via QLANG researcher
//!   * `qlang_run_goal`           — kick off a goal through the orchestrator
//!   * `qlang_read_workspace_file` — read a file from the agent workspace sandbox
//!
//! Single endpoint: `POST /mcp/v1`.

use axum::{extract::{Extension, State}, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::AppState;

/// Advertise MCP server info + capabilities. Returned from `initialize`.
fn server_info() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": { "name": "QLANG", "version": "0.1.0" },
        "capabilities": { "tools": {} },
    })
}

/// Tool definitions exposed via `tools/list`: the 3 agent tools below,
/// plus the 5 `orbit_graph_*` knowledge-graph tools from
/// [`crate::routes::knowledge_tools`].
fn tool_definitions() -> Value {
    let mut tools = vec![
        json!({
            "name": "qlang_research",
            "description": "Multi-source web research (Tavily + Wikipedia + Jina Reader) with the QLANG Researcher agent.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "qlang_run_goal",
            "description": "Run a goal through the QLANG orchestrator. CEO decomposes, specialist agents work in parallel, Guardian value-checks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": { "type": "string" }
                },
                "required": ["goal"]
            }
        }),
        json!({
            "name": "qlang_read_workspace_file",
            "description": "Read a file from the QLANG agent workspace sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }
        }),
    ];
    tools.extend(crate::routes::knowledge_tools::tool_definitions());
    Value::Array(tools)
}

/// How many tools a harness gains by attaching. Derived from the same
/// definitions `tools/list` returns, so the cockpit can never advertise a
/// count that drifts from what is actually exposed.
pub fn tool_count() -> usize {
    match tool_definitions() {
        Value::Array(tools) => tools.len(),
        _ => 0,
    }
}

/// Extract `clientInfo.name` / `.version` from an MCP `initialize` payload.
///
/// Returns `None` when the client sends no name — the spec makes `clientInfo`
/// required, but a lenient server must not fabricate an identity for a client
/// that omits it.
fn client_info(params: &Value) -> Option<(String, Option<String>)> {
    let info = params.get("clientInfo")?;
    let name = info.get("name")?.as_str()?.trim();
    if name.is_empty() {
        return None;
    }
    let version = info
        .get("version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    Some((name.to_string(), version))
}

#[derive(Deserialize)]
pub struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
struct RpcOk {
    jsonrpc: String,
    id: Value,
    result: Value,
}

#[derive(Serialize)]
struct RpcError {
    jsonrpc: String,
    id: Value,
    error: RpcErrInner,
}

#[derive(Serialize)]
struct RpcErrInner {
    code: i32,
    message: String,
}

fn ok(id: Value, result: Value) -> Value {
    serde_json::to_value(RpcOk {
        jsonrpc: "2.0".into(),
        id,
        result,
    })
    .unwrap_or_else(|_| Value::Null)
}

fn err(id: Value, code: i32, message: impl Into<String>) -> Value {
    serde_json::to_value(RpcError {
        jsonrpc: "2.0".into(),
        id,
        error: RpcErrInner {
            code,
            message: message.into(),
        },
    })
    .unwrap_or_else(|_| Value::Null)
}

/// Build an MCP tool-call `content` response payload with a single text
/// block. This is the canonical shape clients expect from `tools/call`.
fn text_content(text: impl Into<String>) -> Value {
    json!({
        "content": [
            { "type": "text", "text": text.into() }
        ]
    })
}

async fn handle_tools_call(
    state: Arc<AppState>,
    principal: &crate::api_keys::Principal,
    params: Value,
) -> Result<Value, (i32, String)> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing tool name".to_string()))?;
    let args = params.get("arguments").cloned().unwrap_or(Value::Null);

    // Knowledge-graph tools live in their own module (which enforces its own
    // per-tool role check).
    if crate::routes::knowledge_tools::handles(name) {
        let out = crate::routes::knowledge_tools::call(state, principal, name, args).await?;
        return Ok(text_content(out));
    }

    match name {
        "qlang_research" => {
            let query = args
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing 'query' argument".to_string()))?;
            let result = qo_agents::tools::tool_web_search(query).await;
            Ok(text_content(result.output))
        }
        "qlang_run_goal" => {
            if !principal.role.can_write() {
                return Err((-32001, "this seat is read-only (viewer); it cannot start goals".to_string()));
            }
            let goal_desc = args
                .get("goal")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing 'goal' argument".to_string()))?
                .to_string();

            // Create the goal via the registry (locked via AppState.agents).
            let goal_id = {
                let mut registry = state.agents.lock().await;
                let goal = registry.create_goal(goal_desc.clone());
                goal.id
            };

            // Persist and log like the normal /api/goals path does, best-effort.
            if let Ok(goal) = {
                let registry = state.agents.lock().await;
                registry
                    .get_goal(goal_id)
                    .cloned()
                    .ok_or(())
            } {
                if let Ok(json_str) = serde_json::to_string(&goal) {
                    let _ = state.store.save_goal(goal_id, &json_str);
                }
            }
            let _ = state.store.log_action("goal_created", &goal_desc, "mcp");

            // Kick off the background executor.
            let state_clone = state.clone();
            let desc_clone = goal_desc.clone();
            tokio::spawn(async move {
                crate::routes::goals::execute_goal_background(
                    state_clone,
                    goal_id,
                    desc_clone,
                )
                .await;
            });

            Ok(text_content(format!(
                "Goal #{} erstellt, läuft im Hintergrund",
                goal_id
            )))
        }
        "qlang_read_workspace_file" => {
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing 'path' argument".to_string()))?;
            let result = qo_agents::tools::tool_read_workspace_file(path);
            Ok(text_content(result.output))
        }
        other => Err((-32601, format!("Unknown tool: {}", other))),
    }
}

/// `POST /mcp/v1` — JSON-RPC 2.0 dispatcher for the MCP protocol.
pub async fn handle_rpc(
    State(state): State<Arc<AppState>>,
    Extension(principal): Extension<crate::api_keys::Principal>,
    body: Option<Json<Value>>,
) -> (StatusCode, Json<Value>) {
    // Parse the envelope. If the body is missing or not a JSON object,
    // return an RPC-style parse error — the spec says id is null in
    // that case.
    let raw = match body {
        Some(Json(v)) => v,
        None => {
            return (
                StatusCode::OK,
                Json(err(Value::Null, -32700, "Parse error")),
            );
        }
    };

    let req: RpcRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::OK,
                Json(err(Value::Null, -32700, format!("Parse error: {e}"))),
            );
        }
    };

    let id = req.id.clone();

    let response = match req.method.as_str() {
        "initialize" => {
            // The handshake is the only place a client names itself. Capturing
            // it here is what makes an attached harness visible in the cockpit.
            if let Some((name, version)) = client_info(&req.params) {
                state.harnesses.record_handshake(&name, version);
            }
            ok(id, server_info())
        }
        "tools/list" => ok(id, json!({ "tools": tool_definitions() })),
        "tools/call" => {
            // Attribute the call before `state` moves into the handler. A
            // client that never sent `initialize` may still identify itself
            // per-call via `_meta.client`; without either it stays anonymous
            // rather than being attributed to the wrong harness.
            let tool = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let caller = client_info(&req.params)
                .map(|(name, _)| name)
                .or_else(|| {
                    req.params
                        .get("_meta")
                        .and_then(|m| m.get("client"))
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                });
            if let Some(name) = caller {
                state.harnesses.record_call(&name, &tool);
            }
            match handle_tools_call(state, &principal, req.params).await {
                Ok(result) => ok(id, result),
                Err((code, msg)) => err(id, code, msg),
            }
        }
        other => err(id, -32601, format!("Method not found: {}", other)),
    };

    (StatusCode::OK, Json(response))
}
