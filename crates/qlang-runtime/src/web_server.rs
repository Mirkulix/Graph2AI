//! WebSocket server for streaming QLANG events to the web dashboard.
//!
//! Implements HTTP file serving and WebSocket protocol (RFC 6455) using only `std::net`.
//! No external crates are used for SHA-1, Base64, or WebSocket framing.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;



// ---------------------------------------------------------------------------
// WebEvent
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum WebEvent {
    GraphNodeExecuted { node_id: u32, op: String, shape: String, time_us: u64, values: Option<Vec<f32>> },
    AgentMessage { from: String, to: String, message: String },
    SystemLog { level: String, message: String },
    GraphLoaded { name: String, num_nodes: usize, num_edges: usize },
    ModelSaved { name: String, version: String },
}

impl WebEvent {
    pub fn to_json(&self) -> String {
        match self {
            WebEvent::GraphNodeExecuted { node_id, op, shape, time_us, values } => {
                let time_ms = *time_us as f64 / 1000.0;
                let vals = values.as_ref().map(|v| format!("[{}]", v.iter().map(|f| format!("{f}")).collect::<Vec<_>>().join(","))).unwrap_or_else(|| "null".to_string());
                format!(r#"{{"type":"node_exec","node_id":{node_id},"op":"{op}","name":"{op}","shape":"{shape}","time_ms":{time_ms},"values":{vals}}}"#)
            }
            WebEvent::AgentMessage { from, to, message } => {
                format!(r#"{{"type":"agent","from":"{from}","to":"{to}","content":"{}"}}"#, json_escape(message))
            }
            WebEvent::SystemLog { level: _, message } => {
                format!(r#"{{"type":"system","text":"{}"}}"#, json_escape(message))
            }
            WebEvent::GraphLoaded { name, num_nodes, num_edges } => {
                let nodes: Vec<_> = (0..(*num_nodes).min(8)).map(|i| format!(r#"{{"id":{i},"label":"node","type":"op"}}"#)).collect();
                let edges: Vec<_> = (0..(*num_edges).min(7)).map(|i| format!(r#"{{"from":{i},"to":{}}}"#, i + 1)).collect();
                format!(r#"{{"type":"graph","nodes":[{}],"edges":[{}],"name":"{}"}}"#, nodes.join(","), edges.join(","), json_escape(name))
            }
            WebEvent::ModelSaved { name, version } => {
                format!(r#"{{"type":"model_saved","name":"{name}","size":"{version}"}}"#)
            }
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Server core
// ---------------------------------------------------------------------------

type Clients = Arc<Mutex<Vec<Arc<Mutex<TcpStream>>>>>;

pub fn spawn_web_server(port: u16) -> Clients {
    let clients: Clients = Arc::new(Mutex::new(Vec::new()));
    let clients_inner = Arc::clone(&clients);

    thread::spawn(move || {
        let listener = TcpListener::bind(format!("0.0.0.0:{port}")).expect("Failed to bind web server");
        eprintln!("[web_server] Listening on http://localhost:{port}");

        for stream in listener.incoming() {
            if let Ok(mut stream) = stream {
                let clients_for_stream = Arc::clone(&clients_inner);
                thread::spawn(move || {
                    handle_stream(&mut stream, &clients_for_stream);
                });
            }
        }
    });

    clients
}

fn handle_stream(stream: &mut TcpStream, clients: &Clients) {
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let request = String::from_utf8_lossy(&buf[..n]);
    
    if request.contains("Upgrade: websocket") {
        let lines: Vec<_> = request.lines().collect();
        let key = lines.iter().find(|l| l.starts_with("Sec-WebSocket-Key: "))
            .map(|l| &l[19..]).unwrap_or("");
        
        let accept = compute_accept_key(key);
        let response = format!(
            "HTTP/1.1 101 Switching Protocols\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Accept: {accept}\r\n\r\n"
        );
        let _ = stream.write_all(response.as_bytes());
        handle_websocket(stream, clients);
    } else if request.starts_with("GET /") {
        let path = request.split_whitespace().nth(1).unwrap_or("/");
        serve_static_file(stream, path);
    }
}

fn serve_static_file(stream: &mut TcpStream, path: &str) {
    let mut full_path = if path == "/" { "frontend/dist/index.html".to_string() } else { format!("frontend/dist{}", path) };
    if !std::path::Path::new(&full_path).exists() {
        full_path = "frontend/dist/index.html".to_string();
    }
    
    let content = std::fs::read(&full_path).unwrap_or_else(|_| b"404 Not Found".to_vec());
    let mime = if full_path.ends_with(".html") { "text/html" }
        else if full_path.ends_with(".js") { "application/javascript" }
        else if full_path.ends_with(".css") { "text/css" }
        else { "text/plain" };
    
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.write_all(&content);
}

fn handle_websocket(stream: &mut TcpStream, clients: &Clients) {
    let stream_arc = Arc::new(Mutex::new(stream.try_clone().unwrap()));
    clients.lock().unwrap().push(Arc::clone(&stream_arc));

    loop {
        let frame = match WsFrame::decode(stream) {
            Ok(f) => f,
            Err(_) => break,
        };

        if frame.opcode == 0x1 { // Text frame
            if let Ok(text) = std::str::from_utf8(&frame.payload) {
                if let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) {
                    match msg["type"].as_str() {
                        Some("exec") => {
                            let code = msg["code"].as_str().unwrap_or("").to_string();
                            let clients_clone = Arc::clone(clients);
                            thread::spawn(move || {
                                let result = match crate::bytecode::run_bytecode(&code) {
                                    Ok((_, out)) => format!(r#"{{"type":"exec_result","output":{},"error":null}}"#, serde_json::to_string(&out).unwrap()),
                                    Err(_) => match crate::unified::execute_unified(&code) {
                                        Ok(r) => format!(r#"{{"type":"exec_result","output":{},"error":null}}"#, serde_json::to_string(&r.output).unwrap()),
                                        Err(e) => format!(r#"{{"type":"exec_result","output":[],"error":"{}"}}"#, json_escape(&format!("{e}"))),
                                    }
                                };
                                broadcast_to_clients(&clients_clone, &result);
                            });
                        }
                        _ => {}
                    }
                }
            }
        } else if frame.opcode == 0x8 { // Close
            break;
        }
    }
}

fn broadcast_to_clients(clients: &Clients, json: &str) {
    let frame = WsFrame::text(json).encode();
    let mut list = clients.lock().unwrap();
    list.retain(|c| {
        let mut s = c.lock().unwrap();
        s.write_all(&frame).is_ok()
    });
}

// ---------------------------------------------------------------------------
// RFC 6455 / SHA-1 / Base64 Boilerplate
// ---------------------------------------------------------------------------

struct WsFrame {
    opcode: u8,
    payload: Vec<u8>,
}

impl WsFrame {
    fn text(s: &str) -> Self { Self { opcode: 0x1, payload: s.as_bytes().to_vec() } }
    
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(0x80 | self.opcode);
        if self.payload.len() <= 125 {
            out.push(self.payload.len() as u8);
        } else if self.payload.len() <= 65535 {
            out.push(126);
            out.extend_from_slice(&(self.payload.len() as u16).to_be_bytes());
        } else {
            out.push(127);
            out.extend_from_slice(&(self.payload.len() as u64).to_be_bytes());
        }
        out.extend_from_slice(&self.payload);
        out
    }

    fn decode<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut head = [0u8; 2];
        reader.read_exact(&mut head)?;
        let opcode = head[0] & 0x0F;
        let masked = (head[1] & 0x80) != 0;
        let mut len = (head[1] & 0x7F) as u64;
        
        if len == 126 {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            len = u16::from_be_bytes(buf) as u64;
        } else if len == 127 {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            len = u64::from_be_bytes(buf);
        }

        let mask = if masked {
            let mut m = [0u8; 4];
            reader.read_exact(&mut m)?;
            Some(m)
        } else { None };

        let mut payload = vec![0u8; len as usize];
        reader.read_exact(&mut payload)?;
        if let Some(m) = mask {
            for i in 0..payload.len() { payload[i] ^= m[i % 4]; }
        }
        Ok(Self { opcode, payload })
    }
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h = [0x67452301u32, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 { w[i] = u32::from_be_bytes(block[i*4..i*4+4].try_into().unwrap()); }
        for i in 16..80 { w[i] = (w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]).rotate_left(1); }
        let mut a = h;
        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((a[1] & a[2]) | ((!a[1]) & a[3]), 0x5A827999),
                20..=39 => (a[1] ^ a[2] ^ a[3], 0x6ED9EBA1),
                40..=59 => ((a[1] & a[2]) | (a[1] & a[3]) | (a[2] & a[3]), 0x8F1BBCDC),
                _ => (a[1] ^ a[2] ^ a[3], 0xCA62C1D6),
            };
            let tmp = a[0].rotate_left(5).wrapping_add(f).wrapping_add(a[4]).wrapping_add(k).wrapping_add(w[i]);
            a[4] = a[3]; a[3] = a[2]; a[2] = a[1].rotate_left(30); a[1] = a[0]; a[0] = tmp;
        }
        for i in 0..5 { h[i] = h[i].wrapping_add(a[i]); }
    }
    let mut out = [0u8; 20];
    for i in 0..5 { out[i*4..i*4+4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for chunk in data.chunks(3) {
        let b = (chunk[0] as u32) << 16 | (chunk.get(1).cloned().unwrap_or(0) as u32) << 8 | chunk.get(2).cloned().unwrap_or(0) as u32;
        s.push(T[(b >> 18 & 0x3F) as usize] as char);
        s.push(T[(b >> 12 & 0x3F) as usize] as char);
        s.push(if chunk.len() > 1 { T[(b >> 6 & 0x3F) as usize] as char } else { '=' });
        s.push(if chunk.len() > 2 { T[(b & 0x3F) as usize] as char } else { '=' });
    }
    s
}

fn compute_accept_key(key: &str) -> String {
    base64_encode(&sha1(format!("{}258EAFA5-E914-47DA-95CA-C5AB0DC85B11", key.trim()).as_bytes()))
}
