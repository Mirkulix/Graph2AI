//! Generic Cloud HTTP Client for LLM providers.
//!
//! Uses raw HTTP over std::net::TcpStream (no reqwest dependency).
//! Each provider implements its own authentication and request/response format.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[cfg(feature = "tls")]
use std::io::Read as TLSRead;

#[cfg(feature = "tls")]
use std::sync::Arc;

#[cfg(feature = "tls")]
use rustls::ClientConnection;

#[derive(Debug, thiserror::Error)]
pub enum CloudError {
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

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("API error: {0}")]
    Api(String),
}

pub type Result<T> = std::result::Result<T, CloudError>;

trait ReadWrite: Read + Write {}
impl<T: Read + Write> ReadWrite for T {}

#[cfg(feature = "tls")]
trait TlsReadWrite: TLSRead + Write {}
#[cfg(feature = "tls")]
impl<T: TLSRead + Write> TlsReadWrite for T {}

pub struct Response {
    pub status: u16,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct CloudClient {
    host: String,
    port: u16,
    use_tls: bool,
    timeout_ms: u64,
}

impl CloudClient {
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            use_tls: port == 443,
            timeout_ms: 120_000,
        }
    }

    pub fn with_tls(mut self, tls: bool) -> Self {
        self.use_tls = tls;
        self
    }

    /// Set timeout in seconds
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_ms = secs * 1000;
        self
    }

    fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    fn connect(&self) -> Result<Box<dyn ReadWrite>> {
        let addr = format!("{}:{}", self.host, self.port);
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e: std::net::AddrParseError| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
            })?,
            self.timeout(),
        )?;
        stream.set_read_timeout(Some(self.timeout()))?;
        stream.set_write_timeout(Some(self.timeout()))?;
        Ok(Box::new(stream))
    }

    #[cfg(feature = "tls")]
    fn connect_tls(&self, server_name: &str) -> Result<Box<dyn TlsReadWrite>> {
        use std::sync::Arc as StdArc;
        let addr = format!("{}:{}", self.host, self.port);
        let tcp = TcpStream::connect_timeout(
            &addr.parse().map_err(|e: std::net::AddrParseError| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
            })?,
            self.timeout(),
        )?;
        tcp.set_read_timeout(Some(self.timeout()))?;
        tcp.set_write_timeout(Some(self.timeout()))?;

        let roots = rustls::RootCertStore::from_iter(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let server_name = rustls::pki_types::ServerName::try_from(server_name)
            .map_err(|_| CloudError::InvalidResponse("invalid server name".into()))?;
        let conn = ClientConnection::new(StdArc::new(config), server_name)
            .map_err(|e| CloudError::Tls(e.to_string()))?;
        let stream = rustls::StreamOwned::new(conn, tcp);
        Ok(Box::new(stream))
    }

    #[cfg(not(feature = "tls"))]
    fn connect_tls(&self, _server_name: &str) -> Result<Box<dyn ReadWrite>> {
        Err(CloudError::Tls("TLS support not enabled. Compile with --features tls".into()))
    }

    pub fn get(&self, path: &str, headers: &[(&str, &str)]) -> Result<Response> {
        let mut stream = self.connect()?;
        let request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Accept: application/json\r\n\
             {headers}\
             Connection: close\r\n\
             \r\n",
            host = self.host,
            port = self.port,
            headers = headers.iter().map(|(k, v)| format!("{}: {}\r\n", k, v)).collect::<String>(),
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;
        Self::read_response(stream)
    }

    pub fn post(&self, path: &str, body: &str, headers: &[(&str, &str)]) -> Result<Response> {
        let mut stream = self.connect()?;
        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Accept: application/json\r\n\
             {headers}\
             Connection: close\r\n\
             \r\n\
             {body}",
            host = self.host,
            port = self.port,
            len = body.len(),
            headers = headers.iter().map(|(k, v)| format!("{}: {}\r\n", k, v)).collect::<String>(),
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;
        Self::read_response(stream)
    }

    pub fn post_tls(&self, path: &str, body: &str, headers: &[(&str, &str)], server_name: &str) -> Result<Response> {
        let mut stream = self.connect_tls(server_name)?;
        let request = format!(
            "POST {path} HTTP/1.1\r\n\
             Host: {host}:{port}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Accept: application/json\r\n\
             {headers}\
             Connection: close\r\n\
             \r\n\
             {body}",
            host = self.host,
            port = self.port,
            len = body.len(),
            headers = headers.iter().map(|(k, v)| format!("{}: {}\r\n", k, v)).collect::<String>(),
        );
        stream.write_all(request.as_bytes())?;
        stream.flush()?;
        Self::read_response_tls(stream)
    }

    fn read_response<R: Read>(reader: R) -> Result<Response> {
        let mut reader = BufReader::new(reader);
        let mut status_line = String::new();
        if let Err(e) = reader.read_line(&mut status_line) {
            return Err(if e.kind() == std::io::ErrorKind::TimedOut
                || e.kind() == std::io::ErrorKind::WouldBlock
            {
                CloudError::Timeout
            } else {
                CloudError::Connection(e)
            });
        }
        let status = status_line.split_whitespace().nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);

        let mut content_length: Option<usize> = None;
        let mut chunked = false;
        let mut body_bytes = Vec::new();

        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            if line == "\r\n" {
                break;
            }
            let lower = line.to_lowercase();
            if let Some(val) = lower.strip_prefix("content-length: ") {
                content_length = val.trim().parse().ok();
            }
            if lower.contains("chunked") {
                chunked = true;
            }
        }

        if chunked {
            let mut chunk_reader = ChunkedReader { reader };
            chunk_reader.read_to_end(&mut body_bytes)?;
        } else if let Some(cl) = content_length {
            body_bytes.resize(cl, 0);
            reader.read_exact(&mut body_bytes)?;
        } else {
            reader.read_to_end(&mut body_bytes)?;
        }

        let body = String::from_utf8_lossy(&body_bytes).into_owned();
        Ok(Response { status, body })
    }

    #[cfg(feature = "tls")]
    fn read_response_tls<R: TLSRead>(reader: R) -> Result<Response> {
        Self::read_response(reader)
    }

    #[cfg(not(feature = "tls"))]
    fn read_response_tls<R: Read>(_reader: R) -> Result<Response> {
        Err(CloudError::Tls("TLS support not enabled. Compile with --features tls".into()))
    }
}

struct ChunkedReader<R> {
    reader: R,
}

impl<R: BufRead> ChunkedReader<R> {
    fn read_to_end(&mut self, dest: &mut Vec<u8>) -> std::io::Result<()> {
        loop {
            let mut size_line = String::new();
            if self.reader.read_line(&mut size_line)? == 0 {
                return Ok(());
            }
            let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
            if size == 0 {
                let mut trailing = String::new();
                self.reader.read_line(&mut trailing)?;
                break;
            }
            let mut chunk = vec![0u8; size];
            self.reader.read_exact(&mut chunk)?;
            dest.extend(chunk);
            let mut trailing = String::new();
            self.reader.read_line(&mut trailing)?;
        }
        Ok(())
    }
}

pub fn extract_json_error(body: &str) -> Option<String> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(err) = v.get("error") {
            if let Some(msg) = err.get("message") {
                return Some(msg.to_string());
            }
            if let Some(msg) = err.as_str() {
                return Some(msg.to_string());
            }
        }
        if let Some(msg) = v.get("message") {
            return Some(msg.to_string());
        }
    }
    None
}
