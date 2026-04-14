//! Tests for the QLANG REST API module.

use super::*;

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

#[test]
fn parse_get_request() {
    let raw = b"GET /info HTTP/1.1\r\nHost: localhost\r\n\r\n";
    let req = parse_request_from_bytes(raw).unwrap();
    assert_eq!(req.method, Method::Get);
    assert_eq!(req.path, "/info");
    assert!(req.body.is_empty());
    assert_eq!(req.headers.get("host").unwrap(), "localhost");
}

#[test]
fn parse_post_request_with_body() {
    let body = r#"{"key":"value"}"#;
    let raw = format!(
        "POST /graph HTTP/1.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        body.len(),
        body,
    );
    let req = parse_request_from_bytes(raw.as_bytes()).unwrap();
    assert_eq!(req.method, Method::Post);
    assert_eq!(req.path, "/graph");
    assert_eq!(req.body, body.as_bytes());
    assert_eq!(
        req.headers.get("content-type").unwrap(),
        "application/json"
    );
}

#[test]
fn parse_options_request() {
    let raw = b"OPTIONS /graph HTTP/1.1\r\nHost: localhost\r\n\r\n";
    let req = parse_request_from_bytes(raw).unwrap();
    assert_eq!(req.method, Method::Options);
    assert_eq!(req.path, "/graph");
}

#[test]
fn parse_request_missing_terminator() {
    let raw = b"GET /info HTTP/1.1\r\nHost: localhost";
    let result = parse_request_from_bytes(raw);
    assert!(result.is_err());
}

#[test]
fn parse_request_no_body_when_content_length_zero() {
    let raw = b"POST /parse HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
    let req = parse_request_from_bytes(raw).unwrap();
    assert!(req.body.is_empty());
}

// ---------------------------------------------------------------------------
// JSON response helpers
// ---------------------------------------------------------------------------

#[test]
fn json_ok_response_format() {
    let val = serde_json::json!({"status": "ok"});
    let resp = json_ok(&val);
    assert_eq!(resp.status_code, 200);
    assert_eq!(resp.status_text, "OK");
    assert_eq!(
        resp.headers.get("Content-Type").unwrap(),
        "application/json"
    );
    assert_eq!(
        resp.headers.get("Access-Control-Allow-Origin").unwrap(),
        "*"
    );
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["status"], "ok");
}

#[test]
fn json_error_response_format() {
    let resp = json_error(404, "Not Found", "no such endpoint");
    assert_eq!(resp.status_code, 404);
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["error"], "no such endpoint");
}

#[test]
fn response_serialization_includes_content_length() {
    let resp = json_ok(&serde_json::json!({"a": 1}));
    let bytes = resp.to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("Content-Length:"));
    assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

#[test]
fn route_unknown_endpoint_returns_404() {
    let req = HttpRequest {
        method: Method::Get,
        path: "/nonexistent".into(),
        headers: HashMap::new(),
        body: Vec::new(),
    };
    let resp = route(&req);
    assert_eq!(resp.status_code, 404);
}

#[test]
fn route_options_returns_204() {
    let req = HttpRequest {
        method: Method::Options,
        path: "/graph".into(),
        headers: HashMap::new(),
        body: Vec::new(),
    };
    let resp = route(&req);
    assert_eq!(resp.status_code, 204);
    assert_eq!(
        resp.headers.get("Access-Control-Allow-Origin").unwrap(),
        "*"
    );
}

// ---------------------------------------------------------------------------
// GET /info
// ---------------------------------------------------------------------------

#[test]
fn info_endpoint_returns_metadata() {
    let resp = handle_info();
    assert_eq!(resp.status_code, 200);
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["name"], "qlang-api");
    assert!(body["capabilities"].is_array());
    let caps = body["capabilities"].as_array().unwrap();
    assert!(caps.iter().any(|c| c == "graph"));
    assert!(caps.iter().any(|c| c == "execute"));
    assert!(caps.iter().any(|c| c == "compress"));
    assert!(caps.iter().any(|c| c == "parse"));
    assert!(caps.iter().any(|c| c == "optimize"));
}

// ---------------------------------------------------------------------------
// POST /graph
// ---------------------------------------------------------------------------

#[test]
fn graph_endpoint_valid_qlang() {
    let source = r#"
graph hello {
  input x: f32[4]
  node r = relu(x)
  output y = r
}
"#;
    let resp = handle_graph(source.as_bytes());
    assert_eq!(resp.status_code, 200);
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["ok"], true);
    assert!(body["graph"].is_object());
    assert_eq!(body["graph"]["id"], "hello");
}

#[test]
fn graph_endpoint_invalid_qlang() {
    let source = "this is not valid qlang";
    let resp = handle_graph(source.as_bytes());
    assert_eq!(resp.status_code, 400);
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert!(body["error"].as_str().unwrap().contains("parse error"));
}

#[test]
fn graph_endpoint_invalid_utf8() {
    let resp = handle_graph(&[0xFF, 0xFE, 0xFD]);
    assert_eq!(resp.status_code, 400);
}

// ---------------------------------------------------------------------------
// POST /compress
// ---------------------------------------------------------------------------

#[test]
fn compress_endpoint_basic() {
    let payload = serde_json::json!({
        "weights": [0.9, -0.8, 0.1, -0.05, 0.0, 1.2, -1.5],
        "threshold": 0.5,
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = handle_compress(&body);
    assert_eq!(resp.status_code, 200);
    let result: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(result["ok"], true);

    let compressed = result["compressed"].as_array().unwrap();
    assert_eq!(compressed.len(), 7);
    // 0.9 -> 1, -0.8 -> -1, 0.1 -> 0, -0.05 -> 0, 0.0 -> 0, 1.2 -> 1, -1.5 -> -1
    assert_eq!(compressed[0], 1);
    assert_eq!(compressed[1], -1);
    assert_eq!(compressed[2], 0);
    assert_eq!(compressed[3], 0);
    assert_eq!(compressed[4], 0);
    assert_eq!(compressed[5], 1);
    assert_eq!(compressed[6], -1);

    let stats = &result["stats"];
    assert_eq!(stats["original_count"], 7);
    assert_eq!(stats["positives"], 2);
    assert_eq!(stats["negatives"], 2);
    assert_eq!(stats["zeros"], 3);
}

#[test]
fn compress_endpoint_missing_weights() {
    let payload = serde_json::json!({});
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = handle_compress(&body);
    assert_eq!(resp.status_code, 400);
}

#[test]
fn compress_endpoint_default_threshold() {
    let payload = serde_json::json!({
        "weights": [0.3, -0.3, 0.7],
    });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = handle_compress(&body);
    assert_eq!(resp.status_code, 200);
    let result: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    let compressed = result["compressed"].as_array().unwrap();
    // Default threshold = 0.5, so 0.3 -> 0, -0.3 -> 0, 0.7 -> 1
    assert_eq!(compressed[0], 0);
    assert_eq!(compressed[1], 0);
    assert_eq!(compressed[2], 1);
}

#[test]
fn compress_endpoint_empty_weights() {
    let payload = serde_json::json!({ "weights": [] });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = handle_compress(&body);
    assert_eq!(resp.status_code, 200);
    let result: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(result["compressed"].as_array().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// POST /parse
// ---------------------------------------------------------------------------

#[test]
fn parse_endpoint_valid_source() {
    let source = r#"
graph test {
  input x: f32[8]
  node r = relu(x)
  output y = r
}
"#;
    let resp = handle_parse(source.as_bytes());
    assert_eq!(resp.status_code, 200);
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["ok"], true);
    assert_eq!(body["valid"], true);
    assert_eq!(body["summary"]["graph_id"], "test");
    assert_eq!(body["summary"]["nodes"], 3);
    assert_eq!(body["summary"]["inputs"], 1);
    assert_eq!(body["summary"]["outputs"], 1);
    assert!(body["diagnostics"].as_array().unwrap().is_empty());
}

#[test]
fn parse_endpoint_invalid_source() {
    let source = "not valid qlang syntax";
    let resp = handle_parse(source.as_bytes());
    assert_eq!(resp.status_code, 200); // 200 with ok=false (diagnostics response)
    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["ok"], false);
    assert_eq!(body["valid"], false);
    assert!(!body["diagnostics"].as_array().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// POST /optimize
// ---------------------------------------------------------------------------

#[test]
fn optimize_endpoint_removes_dead_nodes() {
    // Build a graph with a dead node, serialize it, and send it.
    let mut g = qlang_core::graph::Graph::new("opt_test");
    let tt = qlang_core::tensor::TensorType::f32_vector(4);

    let inp = g.add_node(
        qlang_core::ops::Op::Input { name: "x".into() },
        vec![],
        vec![tt.clone()],
    );
    let relu = g.add_node(
        qlang_core::ops::Op::Relu,
        vec![tt.clone()],
        vec![tt.clone()],
    );
    // Dead node: not connected to output.
    let _dead = g.add_node(
        qlang_core::ops::Op::Neg,
        vec![tt.clone()],
        vec![tt.clone()],
    );
    let out = g.add_node(
        qlang_core::ops::Op::Output { name: "y".into() },
        vec![tt.clone()],
        vec![],
    );
    g.add_edge(inp, 0, relu, 0, tt.clone());
    g.add_edge(relu, 0, out, 0, tt);

    let json_body = serde_json::to_vec(&g).unwrap();
    let resp = handle_optimize(&json_body);
    assert_eq!(resp.status_code, 200);

    let body: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert_eq!(body["ok"], true);
    assert!(body["report"]["dead_nodes_removed"].as_u64().unwrap() >= 1);
    assert!(body["report"]["nodes_after"].as_u64().unwrap() < body["report"]["nodes_before"].as_u64().unwrap());
}

#[test]
fn optimize_endpoint_invalid_json() {
    let resp = handle_optimize(b"not json");
    assert_eq!(resp.status_code, 400);
}

// ---------------------------------------------------------------------------
// POST /execute
// ---------------------------------------------------------------------------

#[test]
fn execute_endpoint_missing_graph() {
    let payload = serde_json::json!({ "inputs": {} });
    let body = serde_json::to_vec(&payload).unwrap();
    let resp = handle_execute(&body);
    assert_eq!(resp.status_code, 400);
    let result: serde_json::Value = serde_json::from_slice(&resp.body).unwrap();
    assert!(result["error"].as_str().unwrap().contains("graph"));
}

#[test]
fn execute_endpoint_invalid_json() {
    let resp = handle_execute(b"not json");
    assert_eq!(resp.status_code, 400);
}

// ---------------------------------------------------------------------------
// Method parsing
// ---------------------------------------------------------------------------

#[test]
fn method_from_str_known() {
    assert_eq!(Method::from_str("GET"), Method::Get);
    assert_eq!(Method::from_str("POST"), Method::Post);
    assert_eq!(Method::from_str("OPTIONS"), Method::Options);
}

#[test]
fn method_from_str_unknown() {
    assert_eq!(Method::from_str("DELETE"), Method::Unknown);
    assert_eq!(Method::from_str("PATCH"), Method::Unknown);
}

// ---------------------------------------------------------------------------
// CORS headers
// ---------------------------------------------------------------------------

#[test]
fn all_responses_have_cors_headers() {
    // Test a few different endpoints to make sure CORS is always present.
    let endpoints: Vec<HttpResponse> = vec![
        handle_info(),
        handle_graph(b"bad"),
        handle_compress(b"{}"),
        json_error(500, "Internal Server Error", "test"),
    ];

    for resp in &endpoints {
        assert_eq!(
            resp.headers.get("Access-Control-Allow-Origin").unwrap(),
            "*",
            "CORS header missing on status {}",
            resp.status_code,
        );
    }
}
