//! Model Hub — HTTP API for the QLANG model registry.
//!
//! Provides a REST API server using `std::net::TcpListener` (no external deps):
//!
//! Endpoints:
//!   GET  /api/models              — List all registered models
//!   GET  /api/models/:name        — List versions of a model
//!   GET  /api/models/:name/:ver   — Get model metadata
//!   POST /api/models              — Register a new model
//!   DELETE /api/models/:name/:ver — Delete a model
//!   GET  /api/health              — Health check
//!
//! Run with: `qlang-cli hub --port 8080`

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use crate::registry::{ModelEntry, Registry, RegistryError};

/// Configuration for the model hub server.
#[derive(Debug, Clone)]
pub struct HubConfig {
    pub host: String,
    pub port: u16,
    pub registry_root: Option<String>,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            registry_root: None,
        }
    }
}

/// Start the model hub HTTP server.
pub fn start_hub(config: &HubConfig) -> std::io::Result<()> {
    let registry = match &config.registry_root {
        Some(root) => Registry::with_root(root),
        None => Registry::new(),
    };

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr)?;
    eprintln!("QLANG Model Hub listening on http://{addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = handle_request(&registry, stream) {
                    eprintln!("Request error: {e}");
                }
            }
            Err(e) => eprintln!("Connection error: {e}"),
        }
    }

    Ok(())
}

/// Parse and handle a single HTTP request.
fn handle_request(
    registry: &Registry,
    mut stream: std::net::TcpStream,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
    if parts.len() < 2 {
        return send_response(&mut stream, 400, "Bad Request");
    }

    let method = parts[0];
    let path = parts[1];

    // Read headers (skip body for now)
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            content_length = line.split(':')
                .nth(1)
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(0);
        }
    }

    // Read body if present
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        std::io::Read::read_exact(&mut reader, &mut body)?;
    }

    // Route request
    match (method, path) {
        ("GET", "/api/health") => {
            send_json(&mut stream, 200, r#"{"status":"ok","service":"qlang-hub"}"#)
        }
        ("GET", "/api/models") => {
            match registry.list() {
                Ok(models) => {
                    let json = format_model_list(&models);
                    send_json(&mut stream, 200, &json)
                }
                Err(e) => send_json(&mut stream, 500, &format!(r#"{{"error":"{}"}}"#, e)),
            }
        }
        ("GET", p) if p.starts_with("/api/models/") => {
            let path_parts: Vec<&str> = p.trim_start_matches("/api/models/").split('/').collect();
            match path_parts.len() {
                1 => {
                    // List versions of a model
                    let name = path_parts[0];
                    match registry.list() {
                        Ok(models) => {
                            let filtered: Vec<&ModelEntry> = models.iter()
                                .filter(|m| m.name == name)
                                .collect();
                            let json = format_model_list_ref(&filtered);
                            send_json(&mut stream, 200, &json)
                        }
                        Err(e) => send_json(&mut stream, 500, &format!(r#"{{"error":"{}"}}"#, e)),
                    }
                }
                2 => {
                    // Get specific version metadata
                    let (name, version) = (path_parts[0], path_parts[1]);
                    match registry.load(name, version) {
                        Ok((entry, _)) => {
                            let json = format_model_entry(&entry);
                            send_json(&mut stream, 200, &json)
                        }
                        Err(RegistryError::NotFound(_, _)) => {
                            send_json(&mut stream, 404, r#"{"error":"model not found"}"#)
                        }
                        Err(e) => send_json(&mut stream, 500, &format!(r#"{{"error":"{}"}}"#, e)),
                    }
                }
                _ => send_response(&mut stream, 404, "Not Found"),
            }
        }
        ("DELETE", p) if p.starts_with("/api/models/") => {
            let path_parts: Vec<&str> = p.trim_start_matches("/api/models/").split('/').collect();
            if path_parts.len() == 2 {
                let (name, version) = (path_parts[0], path_parts[1]);
                match registry.delete(name, version) {
                    Ok(()) => send_json(&mut stream, 200, r#"{"deleted":true}"#),
                    Err(RegistryError::NotFound(_, _)) => {
                        send_json(&mut stream, 404, r#"{"error":"model not found"}"#)
                    }
                    Err(e) => send_json(&mut stream, 500, &format!(r#"{{"error":"{}"}}"#, e)),
                }
            } else {
                send_response(&mut stream, 400, "Bad Request: need /api/models/:name/:version")
            }
        }
        ("POST", "/api/models") => {
            let body_str = String::from_utf8_lossy(&body);
            match serde_json::from_str::<ModelEntry>(&body_str) {
                Ok(entry) => {
                    // Create a minimal checkpoint for metadata-only registration
                    let checkpoint = crate::checkpoint::Checkpoint::new(
                        qlang_core::graph::Graph::new(&entry.name),
                    );
                    match registry.save(&entry, &checkpoint) {
                        Ok(path) => {
                            let json = format!(
                                r#"{{"saved":true,"path":"{}"}}"#,
                                path.display()
                            );
                            send_json(&mut stream, 201, &json)
                        }
                        Err(e) => send_json(&mut stream, 409, &format!(r#"{{"error":"{}"}}"#, e)),
                    }
                }
                Err(e) => send_json(&mut stream, 400, &format!(r#"{{"error":"invalid JSON: {}"}}"#, e)),
            }
        }
        _ => send_response(&mut stream, 404, "Not Found"),
    }
}

fn send_response(stream: &mut std::net::TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, reason, body.len(), body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn send_json(stream: &mut std::net::TcpStream, status: u16, json: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
        status, reason, json.len(), json
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}

fn format_model_entry(entry: &ModelEntry) -> String {
    serde_json::to_string(entry).unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.into())
}

fn format_model_list(models: &[ModelEntry]) -> String {
    let entries: Vec<String> = models.iter().map(|m| format_model_entry(m)).collect();
    format!("[{}]", entries.join(","))
}

fn format_model_list_ref(models: &[&ModelEntry]) -> String {
    let entries: Vec<String> = models.iter().map(|m| format_model_entry(m)).collect();
    format!("[{}]", entries.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_config_default() {
        let config = HubConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_format_model_entry() {
        let entry = ModelEntry {
            name: "test-model".into(),
            version: "1.0.0".into(),
            created_at: "2026-04-03".into(),
            param_count: 1000,
            accuracy: Some(0.95), // Test fixture, not from real training
            loss: Some(0.05),
            compressed: false,
            compression_ratio: None,
            tags: vec!["mnist".into()],
            description: "Test model".into(),
        };
        let json = format_model_entry(&entry);
        assert!(json.contains("test-model"));
        assert!(json.contains("1.0.0"));
    }

    #[test]
    fn test_format_model_list() {
        let entries = vec![
            ModelEntry {
                name: "a".into(), version: "1.0".into(), created_at: "".into(),
                param_count: 100, accuracy: None, loss: None,
                compressed: false, compression_ratio: None,
                tags: vec![], description: "".into(),
            },
            ModelEntry {
                name: "b".into(), version: "2.0".into(), created_at: "".into(),
                param_count: 200, accuracy: None, loss: None,
                compressed: false, compression_ratio: None,
                tags: vec![], description: "".into(),
            },
        ];
        let json = format_model_list(&entries);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"a\""));
        assert!(json.contains("\"b\""));
    }
}
