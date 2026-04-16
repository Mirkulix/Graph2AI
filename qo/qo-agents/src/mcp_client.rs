//! Outbound MCP client — calls external MCP servers from inside QLANG
//! agents. The client speaks JSON-RPC 2.0 over HTTP (one of the
//! transports MCP officially supports). Stdio transport is out of
//! scope here — servers that only support stdio can be fronted by a
//! thin HTTP wrapper.
//!
//! The typical call flow from an agent is:
//!
//! ```no_run
//! # use qo_agents::mcp_client::McpClient;
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let client = McpClient::new("http://localhost:8765");
//! let tools = client.list_tools().await?;
//! let result = client
//!     .call_tool("read_file", serde_json::json!({ "path": "README.md" }))
//!     .await?;
//! # let _ = (tools, result); Ok(())
//! # }
//! ```
//!
//! `tool_mcp_call` is the convenience wrapper the server-side router
//! will plug into the agent's tool table — it returns the canonical
//! [`crate::tools::ToolResult`] so the existing executor can render it
//! like any other tool output.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::ToolResult;

/// How long any single HTTP request is allowed to run. MCP servers
/// that do heavy work (headless browser, remote shell) should still
/// return within this window — if they cannot, they must stream
/// progress, which this client does not yet support.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// A single tool exposed by a remote MCP server, as described by the
/// `tools/list` response. The `input_schema` is an arbitrary JSON
/// Schema the caller can forward to an LLM so it knows what
/// arguments are valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema describing the tool's arguments. MCP servers emit
    /// this under the snake_case key `inputSchema` in the wire
    /// format; we rename it on deserialisation so Rust code stays
    /// idiomatic.
    #[serde(rename = "inputSchema", default)]
    pub input_schema: Value,
}

/// Minimal JSON-RPC 2.0 client aimed specifically at MCP servers
/// that expose the `tools/list` and `tools/call` methods. Thread-safe
/// — cloneable because `reqwest::Client` is already cloneable and the
/// atomic counter is `Arc`-free (plain `AtomicU64` inside the struct).
pub struct McpClient {
    base_url: String,
    client: reqwest::Client,
    next_id: AtomicU64,
}

impl McpClient {
    /// Build a new client pointing at the MCP server's JSON-RPC
    /// endpoint (usually something like `http://host:port/` or
    /// `https://…/mcp`). The URL is stored verbatim and used for
    /// every request, so include the full path here.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// List the tools the remote server advertises. Sends
    /// `{"jsonrpc":"2.0","id":N,"method":"tools/list","params":{}}`
    /// and returns the `tools` array from the result, parsed into
    /// [`McpTool`] values.
    pub async fn list_tools(
        &self,
    ) -> Result<Vec<McpTool>, Box<dyn std::error::Error + Send + Sync>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = build_rpc("tools/list", json!({}), id);
        let result = self.send_rpc(body).await?;

        let tools = result
            .get("tools")
            .and_then(|v| v.as_array())
            .ok_or("MCP tools/list result missing 'tools' array")?;

        let parsed: Vec<McpTool> = serde_json::from_value(Value::Array(tools.clone()))?;
        Ok(parsed)
    }

    /// Invoke a remote tool. The `args` value MUST be a JSON object
    /// that matches the tool's `input_schema`. Returns the raw
    /// `result` field of the JSON-RPC response so the caller can
    /// parse nested content blocks (`content[].text`, images, etc.)
    /// on its own terms.
    pub async fn call_tool(
        &self,
        name: &str,
        args: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let params = json!({
            "name": name,
            "arguments": args,
        });
        let body = build_rpc("tools/call", params, id);
        self.send_rpc(body).await
    }

    /// Shared transport + error-handling for both methods. Extracts
    /// the JSON-RPC `result` on success, or turns the JSON-RPC
    /// `error` envelope (or an HTTP failure) into a boxed error.
    async fn send_rpc(
        &self,
        body: Value,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        let resp = self
            .client
            .post(&self.base_url)
            .json(&body)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("MCP HTTP {status}: {text}").into());
        }

        let envelope: Value = resp.json().await?;

        if let Some(err) = envelope.get("error") {
            let code = err.get("code").and_then(|v| v.as_i64()).unwrap_or(0);
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown MCP error");
            return Err(format!("MCP JSON-RPC error {code}: {message}").into());
        }

        envelope
            .get("result")
            .cloned()
            .ok_or_else(|| "MCP response missing 'result' field".into())
    }
}

/// Assemble a JSON-RPC 2.0 request envelope. Extracted as a free
/// function so the unit tests can assert on the wire format without
/// spinning up a mock HTTP server.
fn build_rpc(method: &str, params: Value, id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

/// Best-effort flattener for the MCP `tools/call` response shape. The
/// spec says a successful call returns `{"content": [{"type":"text",
/// "text":"..."}, ...], "isError": false}`. We walk the `content`
/// array, concatenate every `text` block, and hand back the result.
/// If the server returned something non-standard, we fall back to
/// the raw JSON so nothing is silently lost.
fn flatten_mcp_result(result: &Value) -> String {
    if let Some(arr) = result.get("content").and_then(|v| v.as_array()) {
        let mut parts = Vec::with_capacity(arr.len());
        for block in arr {
            if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                parts.push(text.to_string());
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }
    // Non-standard / unknown shape — hand back the JSON so the caller
    // at least sees SOMETHING actionable.
    serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string())
}

/// High-level convenience wrapper: build a one-shot [`McpClient`],
/// invoke `tool_name`, and fold the outcome into the existing
/// [`ToolResult`] shape the agent executor already knows how to
/// render. On success, the tool field is tagged
/// `mcp[<tool_name>]`; on failure, `success` is `false` and the
/// error message lands in `output`.
pub async fn tool_mcp_call(
    base_url: &str,
    tool_name: &str,
    args: Value,
) -> ToolResult {
    let tool_label = format!("mcp[{tool_name}]");
    let client = McpClient::new(base_url);

    match client.call_tool(tool_name, args).await {
        Ok(result) => {
            // The spec lets servers flag tool-level errors via
            // `isError: true` while HTTP + JSON-RPC still succeed.
            let is_error = result
                .get("isError")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            ToolResult {
                tool: tool_label,
                success: !is_error,
                output: flatten_mcp_result(&result),
            }
        }
        Err(e) => ToolResult {
            tool: tool_label,
            success: false,
            output: format!("MCP call failed: {e}"),
        },
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_rpc_shapes_list_tools() {
        let req = build_rpc("tools/list", json!({}), 1);
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], 1);
        assert_eq!(req["method"], "tools/list");
        assert_eq!(req["params"], json!({}));
        // No extra keys beyond the four JSON-RPC fields.
        let obj = req.as_object().expect("rpc payload must be a JSON object");
        assert_eq!(obj.len(), 4);
    }

    #[test]
    fn build_rpc_shapes_call_tool() {
        let params = json!({
            "name": "read_file",
            "arguments": { "path": "/tmp/foo.txt" },
        });
        let req = build_rpc("tools/call", params.clone(), 42);
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], 42);
        assert_eq!(req["method"], "tools/call");
        assert_eq!(req["params"], params);
        assert_eq!(req["params"]["name"], "read_file");
        assert_eq!(req["params"]["arguments"]["path"], "/tmp/foo.txt");
    }

    #[tokio::test]
    async fn tool_mcp_call_fails_gracefully_on_bad_url() {
        // "not-a-url" is not a valid absolute URL → reqwest refuses
        // to build the request → error path is exercised without
        // any network involvement.
        let result = tool_mcp_call("not-a-url", "anything", json!({})).await;
        assert!(!result.success);
        assert_eq!(result.tool, "mcp[anything]");
        assert!(
            result.output.contains("MCP call failed"),
            "expected error prefix, got: {}",
            result.output
        );
    }
}
