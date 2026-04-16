//! QLMS ↔ MCP Bridge (spec §15.3 / PRD Task 2.1).
//!
//! Converts a Model Context Protocol (MCP) JSON-RPC `tools/call` into the
//! three-node QLANG graph defined by `spec/QLMS_PROTOCOL_v1_1.md` §15.3:
//!
//! ```text
//! Input(args) ──▶ Call(tool: "T") ──▶ Output(result)
//! ```
//!
//! The **forward** direction (`mcp_to_qlms`) accepts any MCP message of the
//! shape `{method: "tools/call", params: {name, arguments}}` and preserves
//! the full JSON (including envelope fields like `jsonrpc`, `id`) so the
//! **reverse** direction (`qlms_to_mcp`) is lossless.
//!
//! Per spec §15.3: "The reverse direction (QLMS → MCP) is defined only for
//! graphs that match this shape — arbitrary DAGs do NOT round-trip to MCP."
//! Implementations MAY reject an MCP-directed QLMS graph whose node count
//! is not exactly 3 — this implementation does exactly that.
//!
//! The full original MCP JSON is stored in `Graph::metadata` under the key
//! [`METADATA_KEY_ORIGINAL`]. This is the source of truth for round-trip
//! equality; the three nodes encode the shape for execution planners.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::TensorType;
use serde_json::Value;

/// Key under which [`mcp_to_qlms`] stashes the full original MCP JSON
/// inside [`Graph::metadata`]. [`qlms_to_mcp`] reads this back verbatim.
pub const METADATA_KEY_ORIGINAL: &str = "mcp.original";

/// Name of the `Input` node's `name` field.
pub const NODE_INPUT_NAME: &str = "mcp.args";

/// Name of the `Output` node's `name` field.
pub const NODE_OUTPUT_NAME: &str = "mcp.result";

/// Prefix used to encode the MCP tool name into the middle `SubGraph` op.
/// Example: tool "list_files" → `Op::SubGraph { graph_id: "mcp.tools.call:list_files" }`.
///
/// `SubGraph` already carries the semantics we need ("execute a named external
/// routine"), so MCP tool calls re-use it rather than introducing a new variant.
pub const CUSTOM_OP_PREFIX: &str = "mcp.tools.call:";

/// Errors surfaced by the bridge.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BridgeError {
    /// A required JSON field was missing or had the wrong type.
    #[error("missing or non-string MCP field: {0}")]
    MissingField(&'static str),

    /// The `method` field is present but is not `"tools/call"`.
    #[error("unexpected MCP method: {got:?}, expected \"tools/call\"")]
    UnexpectedMethod { got: String },

    /// The `params.arguments` field exists but is not a JSON object.
    #[error("params.arguments MUST be a JSON object")]
    ArgumentsNotObject,

    /// The graph does not match the 3-node Input → Call → Output shape.
    #[error("graph does not match the MCP 3-node shape")]
    NotMcpShape,

    /// The graph passed the shape check but the stashed original JSON is
    /// missing. Means the graph was not produced by [`mcp_to_qlms`].
    #[error("graph metadata is missing key '{METADATA_KEY_ORIGINAL}'")]
    MissingOriginal,

    /// The stashed original JSON decoded to something that is not valid JSON.
    #[error("stashed MCP JSON failed to decode: {0}")]
    MalformedOriginal(String),
}

// ---------------------------------------------------------------------------
// MCP → QLMS (forward)
// ---------------------------------------------------------------------------

/// Convert an MCP `tools/call` JSON message into a 3-node QLANG graph.
///
/// The input must have the shape
///
/// ```json
/// { "method": "tools/call", "params": { "name": "<tool>", "arguments": { ... } } }
/// ```
///
/// Additional fields (`jsonrpc`, `id`, …) are preserved verbatim via the
/// graph-metadata round-trip channel.
pub fn mcp_to_qlms(call: &Value) -> Result<Graph, BridgeError> {
    let method = call
        .get("method")
        .and_then(Value::as_str)
        .ok_or(BridgeError::MissingField("method"))?;
    if method != "tools/call" {
        return Err(BridgeError::UnexpectedMethod {
            got: method.to_string(),
        });
    }

    let params = call
        .get("params")
        .ok_or(BridgeError::MissingField("params"))?;
    let tool_name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or(BridgeError::MissingField("params.name"))?
        .to_string();
    let arguments = params
        .get("arguments")
        .ok_or(BridgeError::MissingField("params.arguments"))?;
    if !arguments.is_object() {
        return Err(BridgeError::ArgumentsNotObject);
    }

    // The tensor type is a formality — MCP payloads are JSON blobs, not
    // numeric tensors. We pick a cheap placeholder so the edges type-check.
    let opaque = TensorType::f32_vector(1);

    let mut g = Graph::new(format!("mcp:{tool_name}"));

    let input_node = g.add_node(
        Op::Input {
            name: NODE_INPUT_NAME.to_string(),
        },
        Vec::new(),
        vec![opaque.clone()],
    );
    let call_node = g.add_node(
        Op::SubGraph {
            graph_id: format!("{CUSTOM_OP_PREFIX}{tool_name}"),
        },
        vec![opaque.clone()],
        vec![opaque.clone()],
    );
    let output_node = g.add_node(
        Op::Output {
            name: NODE_OUTPUT_NAME.to_string(),
        },
        vec![opaque.clone()],
        Vec::new(),
    );

    g.add_edge(input_node, 0, call_node, 0, opaque.clone());
    g.add_edge(call_node, 0, output_node, 0, opaque);

    // Stash the full, untouched original so qlms_to_mcp is trivially lossless.
    g.metadata.insert(
        METADATA_KEY_ORIGINAL.to_string(),
        serde_json::to_string(call)
            .expect("serde_json::to_string on an already-parsed Value cannot fail"),
    );

    Ok(g)
}

// ---------------------------------------------------------------------------
// QLMS → MCP (reverse)
// ---------------------------------------------------------------------------

/// Extract the tool name from a `SubGraph` op emitted by [`mcp_to_qlms`].
/// Returns `None` if the op is not a `SubGraph` with our prefix.
fn custom_tool_name(op: &Op) -> Option<&str> {
    match op {
        Op::SubGraph { graph_id } => graph_id.strip_prefix(CUSTOM_OP_PREFIX),
        _ => None,
    }
}

/// Verify a graph has the Input → Call → Output shape mandated by spec §15.3.
///
/// Returns `Ok(())` on a structural match. The reverse direction is only
/// defined for graphs with exactly 3 nodes and 2 edges.
fn check_shape(graph: &Graph) -> Result<(), BridgeError> {
    if graph.nodes.len() != 3 || graph.edges.len() != 2 {
        return Err(BridgeError::NotMcpShape);
    }
    let n0 = &graph.nodes[0];
    let n1 = &graph.nodes[1];
    let n2 = &graph.nodes[2];
    let is_input0 = matches!(&n0.op, Op::Input { name } if name == NODE_INPUT_NAME);
    let is_call1 = custom_tool_name(&n1.op).is_some();
    let is_output2 = matches!(&n2.op, Op::Output { name } if name == NODE_OUTPUT_NAME);
    if !(is_input0 && is_call1 && is_output2) {
        return Err(BridgeError::NotMcpShape);
    }
    // Edges must thread 0 → 1 → 2 on port 0 each side.
    let e01 = graph
        .edges
        .iter()
        .find(|e| e.from_node == n0.id && e.to_node == n1.id)
        .ok_or(BridgeError::NotMcpShape)?;
    let e12 = graph
        .edges
        .iter()
        .find(|e| e.from_node == n1.id && e.to_node == n2.id)
        .ok_or(BridgeError::NotMcpShape)?;
    if e01.from_port != 0 || e01.to_port != 0 || e12.from_port != 0 || e12.to_port != 0 {
        return Err(BridgeError::NotMcpShape);
    }
    Ok(())
}

/// Convert a 3-node MCP bridge graph back into its original MCP JSON.
///
/// The graph MUST have been produced by [`mcp_to_qlms`] (or a byte-for-byte
/// equivalent path): the full original JSON is read from graph metadata.
pub fn qlms_to_mcp(graph: &Graph) -> Result<Value, BridgeError> {
    check_shape(graph)?;
    let raw = graph
        .metadata
        .get(METADATA_KEY_ORIGINAL)
        .ok_or(BridgeError::MissingOriginal)?;
    serde_json::from_str(raw).map_err(|e| BridgeError::MalformedOriginal(e.to_string()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_mcp_call() -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "read_file",
                "arguments": {
                    "path": "/tmp/demo.txt",
                    "encoding": "utf-8"
                }
            }
        })
    }

    #[test]
    fn forward_builds_three_node_graph() {
        let call = sample_mcp_call();
        let g = mcp_to_qlms(&call).expect("valid MCP call must convert");

        assert_eq!(g.nodes.len(), 3, "spec §15.3 mandates exactly 3 nodes");
        assert_eq!(g.edges.len(), 2, "Input→Call→Output is 2 edges");
        assert!(matches!(&g.nodes[0].op, Op::Input { name } if name == NODE_INPUT_NAME));
        assert!(matches!(&g.nodes[2].op, Op::Output { name } if name == NODE_OUTPUT_NAME));
        assert_eq!(custom_tool_name(&g.nodes[1].op), Some("read_file"));
        assert!(g.metadata.contains_key(METADATA_KEY_ORIGINAL));
    }

    #[test]
    fn round_trip_preserves_bit_equality() {
        let original = sample_mcp_call();
        let g = mcp_to_qlms(&original).expect("forward");
        let back = qlms_to_mcp(&g).expect("reverse");
        assert_eq!(
            back, original,
            "spec §15.3 contract: 3-node MCP bridge graphs MUST round-trip"
        );
    }

    #[test]
    fn round_trip_preserves_arbitrary_jsonrpc_envelope_fields() {
        // Fields outside of method/params are part of the JSON-RPC envelope
        // and must survive the round-trip.
        let original = json!({
            "jsonrpc": "2.0",
            "id": "client-xyz-42",
            "trace_id": "abc123",
            "_priv": [1, 2, 3],
            "method": "tools/call",
            "params": { "name": "noop", "arguments": {} }
        });
        let back = qlms_to_mcp(&mcp_to_qlms(&original).unwrap()).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn round_trip_preserves_nested_argument_objects() {
        let original = json!({
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": "rust mcp",
                    "filters": {
                        "since": "2026-01-01",
                        "tags": ["ai", "protocol"]
                    },
                    "limit": 25,
                    "verbose": false
                }
            }
        });
        let back = qlms_to_mcp(&mcp_to_qlms(&original).unwrap()).unwrap();
        assert_eq!(back, original);
    }

    #[test]
    fn rejects_missing_method() {
        let bad = json!({ "params": { "name": "x", "arguments": {} } });
        assert_eq!(mcp_to_qlms(&bad), Err(BridgeError::MissingField("method")));
    }

    #[test]
    fn rejects_unexpected_method() {
        let bad = json!({
            "method": "tools/list",
            "params": { "name": "x", "arguments": {} }
        });
        assert_eq!(
            mcp_to_qlms(&bad),
            Err(BridgeError::UnexpectedMethod {
                got: "tools/list".to_string(),
            })
        );
    }

    #[test]
    fn rejects_missing_params() {
        let bad = json!({ "method": "tools/call" });
        assert_eq!(mcp_to_qlms(&bad), Err(BridgeError::MissingField("params")));
    }

    #[test]
    fn rejects_missing_tool_name() {
        let bad = json!({
            "method": "tools/call",
            "params": { "arguments": {} }
        });
        assert_eq!(
            mcp_to_qlms(&bad),
            Err(BridgeError::MissingField("params.name"))
        );
    }

    #[test]
    fn rejects_non_object_arguments() {
        let bad = json!({
            "method": "tools/call",
            "params": { "name": "x", "arguments": "not an object" }
        });
        assert_eq!(mcp_to_qlms(&bad), Err(BridgeError::ArgumentsNotObject));
    }

    #[test]
    fn reverse_rejects_non_mcp_shape_graph() {
        // Arbitrary graph with unrelated ops → reverse MUST refuse per spec §15.3.
        let mut g = Graph::new("not-mcp");
        g.add_node(Op::Relu, vec![], vec![]);
        assert_eq!(qlms_to_mcp(&g), Err(BridgeError::NotMcpShape));
    }

    #[test]
    fn reverse_rejects_three_node_graph_with_wrong_ops() {
        // 3 nodes but no Custom/mcp prefix in the middle.
        let mut g = Graph::new("pretender");
        let opaque = TensorType::f32_vector(1);
        let a = g.add_node(
            Op::Input {
                name: NODE_INPUT_NAME.to_string(),
            },
            vec![],
            vec![opaque.clone()],
        );
        let b = g.add_node(Op::Relu, vec![opaque.clone()], vec![opaque.clone()]);
        let c = g.add_node(
            Op::Output {
                name: NODE_OUTPUT_NAME.to_string(),
            },
            vec![opaque.clone()],
            vec![],
        );
        g.add_edge(a, 0, b, 0, opaque.clone());
        g.add_edge(b, 0, c, 0, opaque);
        assert_eq!(qlms_to_mcp(&g), Err(BridgeError::NotMcpShape));
    }

    #[test]
    fn reverse_rejects_graph_without_stashed_original() {
        // Build the structural shape by hand but omit the metadata stash.
        let mut g = Graph::new("no-metadata");
        let opaque = TensorType::f32_vector(1);
        let a = g.add_node(
            Op::Input {
                name: NODE_INPUT_NAME.to_string(),
            },
            vec![],
            vec![opaque.clone()],
        );
        let b = g.add_node(
            Op::SubGraph {
                graph_id: format!("{CUSTOM_OP_PREFIX}foo"),
            },
            vec![opaque.clone()],
            vec![opaque.clone()],
        );
        let c = g.add_node(
            Op::Output {
                name: NODE_OUTPUT_NAME.to_string(),
            },
            vec![opaque.clone()],
            vec![],
        );
        g.add_edge(a, 0, b, 0, opaque.clone());
        g.add_edge(b, 0, c, 0, opaque);
        assert_eq!(qlms_to_mcp(&g), Err(BridgeError::MissingOriginal));
    }

    #[test]
    fn reverse_rejects_malformed_stashed_original() {
        let mut g = mcp_to_qlms(&sample_mcp_call()).unwrap();
        g.metadata
            .insert(METADATA_KEY_ORIGINAL.to_string(), "{not json".to_string());
        match qlms_to_mcp(&g) {
            Err(BridgeError::MalformedOriginal(_)) => {}
            other => panic!("expected MalformedOriginal, got {other:?}"),
        }
    }
}
