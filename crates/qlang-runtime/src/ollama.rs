//! Ollama API client — raw HTTP over `std::net::TcpStream`.
//!
//! Communicates with a local Ollama instance for LLM inference.
//! No external HTTP crates; requests are constructed manually.
//!
//! # Environment Variables
//!
//! - `QLANG_OLLAMA_HOST` — Ollama server host (default `127.0.0.1`)
//! - `QLANG_OLLAMA_PORT` — Ollama server port (default `11434`)
//!
//! # Example
//!
//! ```no_run
//! use qlang_runtime::ollama::{OllamaClient, ChatMessage};
//!
//! let client = OllamaClient::from_env();
//! let models = client.list_models().unwrap();
//! let reply = client.generate("llama3", "Hello!", None).unwrap();
//! ```

use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::net::TcpStream;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// Trait object that supports both Read and Write
trait ReadWrite: std::io::Read + std::io::Write {}
impl<T: std::io::Read + std::io::Write> ReadWrite for T {}
// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur when communicating with Ollama.
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("connection failed: {0}")]
    Connection(#[from] std::io::Error),

    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("timeout waiting for response")]
    Timeout,
}

/// Convenience alias used throughout this module.
pub type Result<T> = std::result::Result<T, OllamaError>;

// ---------------------------------------------------------------------------
// ChatMessage
// ---------------------------------------------------------------------------

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

// ---------------------------------------------------------------------------
// OllamaClient
// ---------------------------------------------------------------------------

/// HTTP client for the Ollama REST API.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub host: String,
    pub port: u16,
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11434,
        }
    }
}

impl OllamaClient {
    /// Create a client with explicit host and port.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self { host: host.into(), port }
    }

    /// Create a client from `QLANG_OLLAMA_HOST` / `QLANG_OLLAMA_PORT` env
    /// vars, falling back to the defaults (`127.0.0.1:11434`).
    pub fn from_env() -> Self {
        let host = std::env::var("QLANG_OLLAMA_HOST")
            .unwrap_or_else(|_| "127.0.0.1".into());
        let port = std::env::var("QLANG_OLLAMA_PORT")
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(11434);
        Self { host, port }
    }

    /// Address string used for `TcpStream::connect`.
    fn addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    // -- public API --------------------------------------------------------

    /// Returns `true` if the Ollama server is reachable and responds to
    /// `GET /api/tags` with HTTP 200.
    pub fn health(&self) -> Result<bool> {
        match self.http_get("/api/tags") {
            Ok(_) => Ok(true),
            Err(OllamaError::Connection(_)) => Ok(false),
            Err(OllamaError::Timeout) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// List the names of all locally-available models.
    pub fn list_models(&self) -> Result<Vec<String>> {
        let body = self.http_get("/api/tags")?;
        let parsed: TagsResponse = serde_json::from_str(&body)?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    /// Run a one-shot generation (non-streaming).
    ///
    /// Returns the full generated text.
    pub fn generate(
        &self,
        model: &str,
        prompt: &str,
        system: Option<&str>,
    ) -> Result<String> {
        let mut payload = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        });
        if let Some(sys) = system {
            payload["system"] = serde_json::Value::String(sys.into());
        }
        let body = serde_json::to_string(&payload)?;
        let resp = self.http_post("/api/generate", &body)?;
        let parsed: GenerateResponse = serde_json::from_str(&resp)?;
        Ok(parsed.response)
    }

    /// Run a multi-turn chat completion (non-streaming).
    ///
    /// Returns the assistant's reply text.
    pub fn chat(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<String> {
        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });
        let body = serde_json::to_string(&payload)?;
        let resp = self.http_post("/api/chat", &body)?;
        let parsed: ChatResponse = serde_json::from_str(&resp)?;
        Ok(parsed.message.content)
    }

    // -- low-level HTTP ----------------------------------------------------

    #[cfg(not(feature = "tls"))]
    fn connect(&self) -> Result<Box<dyn ReadWrite>> {
        let stream = TcpStream::connect(self.addr())?;
        stream.set_read_timeout(Some(Duration::from_secs(120)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        Ok(Box::new(stream))
    }

    #[cfg(feature = "tls")]
    fn connect(&self) -> Result<Box<dyn ReadWrite>> {
        let tcp = TcpStream::connect(self.addr())?;
        tcp.set_read_timeout(Some(Duration::from_secs(120)))?;
        tcp.set_write_timeout(Some(Duration::from_secs(10)))?;
        let tls_enabled = std::env::var("QLANG_OLLAMA_TLS")
            .ok()
            .map(|v| {
                let l = v.to_ascii_lowercase();
                l == "1" || l == "true" || l == "yes" || l == "on" || l == "https"
            })
            .unwrap_or(false);
        if !tls_enabled {
            return Ok(Box::new(tcp));
        }
        use std::sync::Arc;
        let roots = rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let server_name = rustls::pki_types::ServerName::try_from(self.host.as_str())
            .map_err(|_| OllamaError::InvalidResponse("invalid server name".into()))?;
        let conn = rustls::ClientConnection::new(Arc::new(config), server_name)
            .map_err(|e| OllamaError::InvalidResponse(format!("tls: {e}")))?;
        let stream = rustls::StreamOwned::new(conn, tcp);
        Ok(Box::new(stream))
    }

    fn http_get(&self, path: &str) -> Result<String> {
        let mut stream = self.connect()?;
        let request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Accept: application/json\r\n\
             Connection: close\r\n\
             \r\n",
            host = self.host,
            port = self.port,
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;
        self.read_response(stream)
    }

    fn http_post(&self, path: &str, body: &str) -> Result<String> {
        let mut stream = self.connect()?;
        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Accept: application/json\r\n\
             Connection: close\r\n\
             \r\n{body}",
            host = self.host,
            port = self.port,
            len = body.len(),
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;
        self.read_response(stream)
    }

    /// Read a full HTTP response: status line, headers, body.
    ///
    /// Supports both `Content-Length` and `Transfer-Encoding: chunked`.
    fn read_response(&self, stream: Box<dyn ReadWrite>) -> Result<String> {
        let mut reader = BufReader::new(stream);

        // -- status line ---------------------------------------------------
        let mut status_line = String::new();
        if let Err(e) = reader.read_line(&mut status_line) {
            return Err(if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock
            {
                OllamaError::Timeout
            } else {
                OllamaError::Connection(e)
            });
        }

        let status = parse_status_code(&status_line)?;

        // -- headers -------------------------------------------------------
        let mut content_length: Option<usize> = None;
        let mut chunked = false;

        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            let lower = trimmed.to_ascii_lowercase();
            if lower.starts_with("content-length:") {
                content_length = lower
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().parse::<usize>().ok());
            } else if lower.starts_with("transfer-encoding:") {
                if lower.contains("chunked") {
                    chunked = true;
                }
            }
        }

        // -- body ----------------------------------------------------------
        let body = if chunked {
            read_chunked_body(&mut reader)?
        } else if let Some(len) = content_length {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            // No content-length and not chunked — read until EOF.
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf)?;
            String::from_utf8_lossy(&buf).into_owned()
        };

        if status >= 400 {
            return Err(OllamaError::Http { status, body });
        }

        Ok(body)
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Parse the numeric status code from an HTTP status line like
/// `HTTP/1.1 200 OK\r\n`.
fn parse_status_code(line: &str) -> Result<u16> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(OllamaError::InvalidResponse(format!(
            "malformed status line: {line:?}"
        )));
    }
    parts[1]
        .parse::<u16>()
        .map_err(|_| OllamaError::InvalidResponse(format!("bad status code: {:?}", parts[1])))
}

/// Read an HTTP chunked transfer-encoded body.
///
/// Each chunk is: `<hex-size>\r\n<data>\r\n`, terminated by a `0\r\n\r\n`
/// chunk.
fn read_chunked_body<R: std::io::Read + std::io::BufRead>(reader: &mut R) -> Result<String> {
    let mut body = Vec::new();

    loop {
        // Read chunk size line.
        let mut size_line = String::new();
        reader.read_line(&mut size_line)?;
        let size_str = size_line.trim();
        // Chunk extensions (`;...`) are allowed by HTTP but Ollama doesn't
        // use them. Strip them just in case.
        let size_hex = size_str.split(';').next().unwrap_or("0").trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|_| {
            OllamaError::InvalidResponse(format!("bad chunk size: {size_str:?}"))
        })?;

        if size == 0 {
            // Terminal chunk — consume optional trailers + final CRLF.
            loop {
                let mut trailer = String::new();
                reader.read_line(&mut trailer)?;
                if trailer.trim().is_empty() {
                    break;
                }
            }
            break;
        }

        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk)?;
        body.extend_from_slice(&chunk);

        // Consume trailing CRLF after chunk data.
        let mut crlf = String::new();
        reader.read_line(&mut crlf)?;
    }

    Ok(String::from_utf8_lossy(&body).into_owned())
}

// ---------------------------------------------------------------------------
// Ollama JSON response types (private)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Deserialize)]
struct TagModel {
    name: String,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ChatMessage -------------------------------------------------------

    #[test]
    fn test_chat_message_constructors() {
        let u = ChatMessage::user("hi");
        assert_eq!(u.role, "user");
        assert_eq!(u.content, "hi");

        let s = ChatMessage::system("you are helpful");
        assert_eq!(s.role, "system");

        let a = ChatMessage::assistant("hello!");
        assert_eq!(a.role, "assistant");
    }

    #[test]
    fn test_chat_message_serialize() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"hello""#));
    }

    #[test]
    fn test_chat_message_deserialize() {
        let json = r#"{"role":"assistant","content":"world"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "world");
    }

    #[test]
    fn test_chat_message_clone() {
        let msg = ChatMessage::user("test");
        let cloned = msg.clone();
        assert_eq!(cloned.role, msg.role);
        assert_eq!(cloned.content, msg.content);
    }

    // -- OllamaClient construction ----------------------------------------

    #[test]
    fn test_default_client() {
        let client = OllamaClient::default();
        assert_eq!(client.host, "127.0.0.1");
        assert_eq!(client.port, 11434);
        assert_eq!(client.addr(), "127.0.0.1:11434");
    }

    #[test]
    fn test_new_client() {
        let client = OllamaClient::new("localhost", 9999);
        assert_eq!(client.host, "localhost");
        assert_eq!(client.port, 9999);
        assert_eq!(client.addr(), "localhost:9999");
    }

    // -- JSON parsing (TagsResponse) ---------------------------------------

    #[test]
    fn test_parse_tags_response_multiple() {
        let json = r#"{
            "models": [
                {"name": "llama3:latest", "size": 12345},
                {"name": "mistral:7b",    "size": 67890}
            ]
        }"#;
        let resp: TagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 2);
        assert_eq!(resp.models[0].name, "llama3:latest");
        assert_eq!(resp.models[1].name, "mistral:7b");
    }

    #[test]
    fn test_parse_tags_response_empty() {
        let json = r#"{"models":[]}"#;
        let resp: TagsResponse = serde_json::from_str(json).unwrap();
        assert!(resp.models.is_empty());
    }

    #[test]
    fn test_parse_tags_response_missing_models_field() {
        // `models` has `#[serde(default)]` so a missing field yields an
        // empty vec rather than an error.
        let json = r#"{}"#;
        let resp: TagsResponse = serde_json::from_str(json).unwrap();
        assert!(resp.models.is_empty());
    }

    // -- JSON parsing (GenerateResponse) -----------------------------------

    #[test]
    fn test_parse_generate_response() {
        let json = r#"{
            "model": "llama3",
            "response": "Hello there!",
            "done": true,
            "total_duration": 123456
        }"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response, "Hello there!");
    }

    #[test]
    fn test_parse_generate_response_empty() {
        let json = r#"{"response": ""}"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.response, "");
    }

    // -- JSON parsing (ChatResponse) ---------------------------------------

    #[test]
    fn test_parse_chat_response() {
        let json = r#"{
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "I can help with that."
            },
            "done": true
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.role, "assistant");
        assert_eq!(resp.message.content, "I can help with that.");
    }

    // -- HTTP helpers ------------------------------------------------------

    #[test]
    fn test_parse_status_code_200() {
        assert_eq!(parse_status_code("HTTP/1.1 200 OK\r\n").unwrap(), 200);
    }

    #[test]
    fn test_parse_status_code_404() {
        assert_eq!(
            parse_status_code("HTTP/1.1 404 Not Found\r\n").unwrap(),
            404
        );
    }

    #[test]
    fn test_parse_status_code_malformed() {
        assert!(parse_status_code("garbage").is_err());
    }

    #[test]
    fn test_parse_status_code_bad_number() {
        assert!(parse_status_code("HTTP/1.1 XYZ OK\r\n").is_err());
    }

    // -- Serialization round-trip for generate request payload -------------

    #[test]
    fn test_generate_request_payload() {
        let payload = serde_json::json!({
            "model": "llama3",
            "prompt": "What is 2+2?",
            "stream": false,
        });
        let s = serde_json::to_string(&payload).unwrap();
        assert!(s.contains("\"stream\":false"));
        assert!(s.contains("\"model\":\"llama3\""));
    }

    #[test]
    fn test_generate_request_payload_with_system() {
        let mut payload = serde_json::json!({
            "model": "llama3",
            "prompt": "hi",
            "stream": false,
        });
        payload["system"] = serde_json::Value::String("Be concise".into());
        let s = serde_json::to_string(&payload).unwrap();
        assert!(s.contains("\"system\":\"Be concise\""));
    }

    #[test]
    fn test_chat_request_payload() {
        let msgs = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let payload = serde_json::json!({
            "model": "mistral",
            "messages": msgs,
            "stream": false,
        });
        let s = serde_json::to_string(&payload).unwrap();
        assert!(s.contains("\"role\":\"system\""));
        assert!(s.contains("\"role\":\"user\""));
        assert!(s.contains("\"stream\":false"));
    }

    // -- Error display -----------------------------------------------------

    #[test]
    fn test_error_display_http() {
        let err = OllamaError::Http {
            status: 404,
            body: "not found".into(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("404"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_error_display_invalid_response() {
        let err = OllamaError::InvalidResponse("bad".into());
        assert!(format!("{err}").contains("bad"));
    }

    #[test]
    fn test_error_display_timeout() {
        let err = OllamaError::Timeout;
        assert!(format!("{err}").contains("timeout"));
    }
}
