//! QLANG REST API — HTTP server using only std::net
//!
//! Provides a simple HTTP interface for interacting with QLANG over the network.
//! No external HTTP frameworks are used; requests are parsed from raw TCP streams.
//!
//! Endpoints:
//!   POST /graph    — submit .qlang text, returns JSON graph
//!   POST /execute  — submit JSON graph + inputs, returns outputs
//!   POST /compress — submit weights as JSON array, returns ternary compressed
//!   GET  /info     — returns server info (version, capabilities)
//!   POST /parse    — parse .qlang text, return diagnostics
//!   POST /optimize — optimize a graph, return optimized graph + report

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

use qlang_core::graph::Graph;
use qlang_core::serial;
use qlang_core::tensor::{Dim, Shape, TensorData};

use crate::optimize;
use crate::parser;

// ---------------------------------------------------------------------------
// HTTP primitives
// ---------------------------------------------------------------------------

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
    Options,
    Unknown,
}

impl Method {
    fn from_str(s: &str) -> Self {
        match s {
            "GET" => Method::Get,
            "POST" => Method::Post,
            "OPTIONS" => Method::Options,
            _ => Method::Unknown,
        }
    }
}

/// A parsed HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: Method,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

/// An HTTP response to be sent back.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status_code: u16,
    pub status_text: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Construct a response with the given status.
    pub fn new(status_code: u16, status_text: &str) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Access-Control-Allow-Origin".into(), "*".into());
        headers.insert(
            "Access-Control-Allow-Methods".into(),
            "GET, POST, OPTIONS".into(),
        );
        headers.insert(
            "Access-Control-Allow-Headers".into(),
            "Content-Type".into(),
        );
        headers.insert("Connection".into(), "close".into());
        Self {
            status_code,
            status_text: status_text.to_string(),
            headers,
            body: Vec::new(),
        }
    }

    /// Serialize the response to bytes suitable for writing to a TCP stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let status_line = format!(
            "HTTP/1.1 {} {}\r\n",
            self.status_code, self.status_text
        );
        buf.extend_from_slice(status_line.as_bytes());

        // Always include Content-Length.
        let cl = format!("Content-Length: {}\r\n", self.body.len());
        buf.extend_from_slice(cl.as_bytes());

        for (k, v) in &self.headers {
            let header = format!("{}: {}\r\n", k, v);
            buf.extend_from_slice(header.as_bytes());
        }

        buf.extend_from_slice(b"\r\n");
        buf.extend_from_slice(&self.body);
        buf
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Create a JSON success response (200 OK).
pub fn json_ok(value: &serde_json::Value) -> HttpResponse {
    let mut resp = HttpResponse::new(200, "OK");
    resp.headers
        .insert("Content-Type".into(), "application/json".into());
    resp.body = serde_json::to_vec_pretty(value).unwrap_or_default();
    resp
}

/// Create a JSON error response with the given HTTP status code.
pub fn json_error(status: u16, status_text: &str, message: &str) -> HttpResponse {
    let mut resp = HttpResponse::new(status, status_text);
    resp.headers
        .insert("Content-Type".into(), "application/json".into());
    let body = serde_json::json!({ "error": message });
    resp.body = serde_json::to_vec_pretty(&body).unwrap_or_default();
    resp
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

/// Parse an HTTP request from a raw TCP stream.
///
/// Reads the request line, headers, and body (based on Content-Length).
pub fn parse_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    // --- Request line ---
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(|e| format!("failed to read request line: {e}"))?;
    let request_line = request_line.trim_end().to_string();

    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(format!("malformed request line: {request_line}"));
    }

    let method = Method::from_str(parts[0]);
    let path = parts[1].to_string();

    // --- Headers ---
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| format!("failed to read header: {e}"))?;
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(
                key.trim().to_lowercase(),
                value.trim().to_string(),
            );
        }
    }

    // --- Body ---
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("failed to read body: {e}"))?;
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

/// Parse an HTTP request from raw bytes (useful for testing).
pub fn parse_request_from_bytes(data: &[u8]) -> Result<HttpRequest, String> {
    // Find the end of headers (\r\n\r\n).
    let header_end = find_subsequence(data, b"\r\n\r\n")
        .ok_or_else(|| "no header terminator found".to_string())?;
    let header_bytes = &data[..header_end];
    let body_start = header_end + 4;

    let header_str =
        std::str::from_utf8(header_bytes).map_err(|e| format!("invalid UTF-8 in headers: {e}"))?;
    let mut lines = header_str.lines();

    // Request line.
    let request_line = lines.next().ok_or("empty request")?;
    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(format!("malformed request line: {request_line}"));
    }
    let method = Method::from_str(parts[0]);
    let path = parts[1].to_string();

    // Headers.
    let mut headers = HashMap::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(key.trim().to_lowercase(), value.trim().to_string());
        }
    }

    // Body.
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let body = if content_length > 0 && body_start < data.len() {
        let end = std::cmp::min(body_start + content_length, data.len());
        data[body_start..end].to_vec()
    } else {
        Vec::new()
    };

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------

/// GET /info — server metadata.
pub fn handle_info() -> HttpResponse {
    let info = serde_json::json!({
        "name": "qlang-api",
        "version": env!("CARGO_PKG_VERSION"),
        "capabilities": [
            "graph",
            "execute",
            "compress",
            "parse",
            "optimize"
        ],
        "language": "QLANG",
        "description": "Graph-based AI-to-AI programming language REST API"
    });
    json_ok(&info)
}

/// POST /graph — compile .qlang text into a JSON graph.
pub fn handle_graph(body: &[u8]) -> HttpResponse {
    let source = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(e) => return json_error(400, "Bad Request", &format!("invalid UTF-8: {e}")),
    };

    match parser::parse(source) {
        Ok(graph) => match serial::to_json(&graph) {
            Ok(json_str) => {
                // Parse back to Value so we can wrap it properly.
                match serde_json::from_str::<serde_json::Value>(&json_str) {
                    Ok(val) => {
                        let resp = serde_json::json!({
                            "ok": true,
                            "graph": val,
                        });
                        json_ok(&resp)
                    }
                    Err(e) => json_error(
                        500,
                        "Internal Server Error",
                        &format!("serialization error: {e}"),
                    ),
                }
            }
            Err(e) => json_error(
                500,
                "Internal Server Error",
                &format!("serialization error: {e}"),
            ),
        },
        Err(e) => json_error(400, "Bad Request", &format!("parse error: {e}")),
    }
}

/// POST /execute — execute a graph with provided inputs.
///
/// Expects JSON body: `{ "graph": <Graph JSON>, "inputs": { "<name>": [f32...], ... } }`
pub fn handle_execute(body: &[u8]) -> HttpResponse {
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return json_error(400, "Bad Request", &format!("invalid JSON: {e}")),
    };

    // Deserialize graph.
    let graph_val = match payload.get("graph") {
        Some(v) => v,
        None => return json_error(400, "Bad Request", "missing 'graph' field"),
    };

    let graph: Graph = match serde_json::from_value(graph_val.clone()) {
        Ok(g) => g,
        Err(e) => {
            return json_error(400, "Bad Request", &format!("invalid graph: {e}"))
        }
    };

    // Build inputs map.
    let inputs_val = payload.get("inputs").cloned().unwrap_or(serde_json::json!({}));
    let inputs_map = match inputs_val.as_object() {
        Some(m) => m,
        None => return json_error(400, "Bad Request", "'inputs' must be an object"),
    };

    let mut inputs: HashMap<String, TensorData> = HashMap::new();
    for (name, arr) in inputs_map {
        if let Some(values) = arr.as_array() {
            let floats: Vec<f32> = values
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            let len = floats.len();
            let shape = Shape(vec![Dim::Fixed(len)]);
            inputs.insert(name.clone(), TensorData::from_f32(shape, &floats));
        }
    }

    // Execute.
    match qlang_runtime::executor::execute(&graph, inputs) {
        Ok(result) => {
            let mut output_json = serde_json::Map::new();
            for (name, tensor) in &result.outputs {
                let floats = tensor.as_f32_slice().unwrap_or_default();
                let arr: Vec<serde_json::Value> = floats
                    .iter()
                    .map(|&f| serde_json::Value::from(f))
                    .collect();
                output_json.insert(name.clone(), serde_json::Value::Array(arr));
            }
            let resp = serde_json::json!({
                "ok": true,
                "outputs": output_json,
                "stats": {
                    "nodes_executed": result.stats.nodes_executed,
                    "quantum_ops": result.stats.quantum_ops,
                    "total_flops": result.stats.total_flops,
                }
            });
            json_ok(&resp)
        }
        Err(e) => json_error(
            500,
            "Internal Server Error",
            &format!("execution error: {e}"),
        ),
    }
}

/// POST /compress — ternary weight compression.
///
/// Expects JSON body: `{ "weights": [f32, ...], "threshold": <optional f32> }`
/// Returns `{ "compressed": [i8, ...], "stats": { ... } }`.
pub fn handle_compress(body: &[u8]) -> HttpResponse {
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return json_error(400, "Bad Request", &format!("invalid JSON: {e}")),
    };

    let weights = match payload.get("weights").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>(),
        None => return json_error(400, "Bad Request", "missing or invalid 'weights' array"),
    };

    let threshold = payload
        .get("threshold")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32)
        .unwrap_or(0.5);

    // Ternary compression: map to {-1, 0, +1}.
    let mut zeros = 0usize;
    let mut positives = 0usize;
    let mut negatives = 0usize;

    let compressed: Vec<i8> = weights
        .iter()
        .map(|&w| {
            if w.abs() < threshold {
                zeros += 1;
                0i8
            } else if w > 0.0 {
                positives += 1;
                1i8
            } else {
                negatives += 1;
                -1i8
            }
        })
        .collect();

    let original_bits = weights.len() * 32;
    let compressed_bits = weights.len() * 2; // 2 bits per ternary value
    let ratio = if compressed_bits > 0 {
        original_bits as f64 / compressed_bits as f64
    } else {
        0.0
    };

    let resp = serde_json::json!({
        "ok": true,
        "compressed": compressed,
        "stats": {
            "original_count": weights.len(),
            "positives": positives,
            "negatives": negatives,
            "zeros": zeros,
            "sparsity": if weights.is_empty() { 0.0 } else { zeros as f64 / weights.len() as f64 },
            "compression_ratio": ratio,
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
        }
    });
    json_ok(&resp)
}

/// POST /parse — parse .qlang text, return diagnostics.
pub fn handle_parse(body: &[u8]) -> HttpResponse {
    let source = match std::str::from_utf8(body) {
        Ok(s) => s,
        Err(e) => return json_error(400, "Bad Request", &format!("invalid UTF-8: {e}")),
    };

    match parser::parse(source) {
        Ok(graph) => {
            let validation = graph.validate();
            let diagnostics: Vec<String> = match &validation {
                Ok(()) => Vec::new(),
                Err(errors) => errors.iter().map(|e| e.to_string()).collect(),
            };
            let resp = serde_json::json!({
                "ok": true,
                "valid": validation.is_ok(),
                "diagnostics": diagnostics,
                "summary": {
                    "graph_id": graph.id,
                    "nodes": graph.nodes.len(),
                    "edges": graph.edges.len(),
                    "inputs": graph.input_nodes().len(),
                    "outputs": graph.output_nodes().len(),
                }
            });
            json_ok(&resp)
        }
        Err(e) => {
            let resp = serde_json::json!({
                "ok": false,
                "valid": false,
                "diagnostics": [e.to_string()],
                "summary": null,
            });
            json_ok(&resp)
        }
    }
}

/// POST /optimize — optimize a JSON graph, return optimized graph + report.
///
/// Expects JSON body: the serialized `Graph`.
pub fn handle_optimize(body: &[u8]) -> HttpResponse {
    let mut graph: Graph = match serde_json::from_slice(body) {
        Ok(g) => g,
        Err(e) => return json_error(400, "Bad Request", &format!("invalid graph JSON: {e}")),
    };

    let nodes_before = graph.nodes.len();
    let edges_before = graph.edges.len();

    let report = optimize::optimize(&mut graph);

    let graph_json = match serial::to_json(&graph) {
        Ok(s) => match serde_json::from_str::<serde_json::Value>(&s) {
            Ok(v) => v,
            Err(e) => {
                return json_error(
                    500,
                    "Internal Server Error",
                    &format!("serialization error: {e}"),
                )
            }
        },
        Err(e) => {
            return json_error(
                500,
                "Internal Server Error",
                &format!("serialization error: {e}"),
            )
        }
    };

    let resp = serde_json::json!({
        "ok": true,
        "graph": graph_json,
        "report": {
            "nodes_before": nodes_before,
            "nodes_after": graph.nodes.len(),
            "edges_before": edges_before,
            "edges_after": graph.edges.len(),
            "dead_nodes_removed": report.dead_nodes_removed,
            "constants_folded": report.constants_folded,
            "ops_fused": report.ops_fused,
            "fused_descriptions": report.fused_descriptions,
            "identity_ops_removed": report.identity_ops_removed,
            "common_subexpressions_eliminated": report.common_subexpressions_eliminated,
        }
    });
    json_ok(&resp)
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Route an HTTP request to the appropriate handler.
pub fn route(request: &HttpRequest) -> HttpResponse {
    // Handle CORS preflight.
    if request.method == Method::Options {
        return HttpResponse::new(204, "No Content");
    }

    match (request.method, request.path.as_str()) {
        (Method::Get, "/info") => handle_info(),
        (Method::Post, "/graph") => handle_graph(&request.body),
        (Method::Post, "/execute") => handle_execute(&request.body),
        (Method::Post, "/compress") => handle_compress(&request.body),
        (Method::Post, "/parse") => handle_parse(&request.body),
        (Method::Post, "/optimize") => handle_optimize(&request.body),
        (_, path) => json_error(
            404,
            "Not Found",
            &format!("unknown endpoint: {path}"),
        ),
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// A minimal HTTP server for the QLANG REST API.
pub struct ApiServer {
    listener: TcpListener,
}

impl ApiServer {
    /// Bind the server to the given address (e.g. "127.0.0.1:8080").
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self { listener })
    }

    /// Return the local address the server is bound to.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Run the server, handling connections in a loop. This blocks forever.
    pub fn run(&self) -> std::io::Result<()> {
        eprintln!(
            "qlang-api listening on {}",
            self.listener.local_addr()?
        );

        for stream in self.listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    if let Err(e) = Self::handle_connection(&mut stream) {
                        eprintln!("connection error: {e}");
                    }
                }
                Err(e) => {
                    eprintln!("accept error: {e}");
                }
            }
        }
        Ok(())
    }

    fn handle_connection(stream: &mut TcpStream) -> Result<(), String> {
        let request = parse_request(stream)?;
        let response = route(&request);
        let bytes = response.to_bytes();
        stream
            .write_all(&bytes)
            .map_err(|e| format!("write error: {e}"))?;
        stream.flush().map_err(|e| format!("flush error: {e}"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests (in separate file)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "api_test.rs"]
mod api_test;
