//! QLMS Binary Protocol vs MCP JSON — Real measurement benchmark.
//!
//! Compares the binary QLMS wire format (see `qo-server/routes/qlms_demo.rs`)
//! against a simulated MCP-style JSON-RPC weight-transfer payload.
//!
//! Run with:
//!   LIBTORCH_USE_PYTORCH=1 cargo run --release --no-default-features \
//!     --example qlms_benchmark -p qlang-runtime
//!
//! Measures (1000-iteration average):
//!   - serialized size (bytes)
//!   - serialize time (ns)
//!   - deserialize time (ns)
//!   - localhost HTTP round-trip (µs)
//!
//! No external bench crate — uses `std::time::Instant`.
//! HTTP RTT uses a tiny TCP echo loopback (std::net) so we isolate
//! serialization cost from any heavy server logic.

use qlang_core::crypto::{ct_eq, hmac_sha256, sha256};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

// ---- QLMS wire format constants (mirroring qlms_demo.rs) ----
const QLMS_MAGIC: &[u8; 4] = b"QLMS";
const QLMS_VERSION: u16 = 1;
const QLMS_KIND_MODEL: u16 = 0x0001;

fn shared_key() -> [u8; 32] {
    sha256(b"qlang-demo-2026")
}

// ---- QLMS payload (binary) ----
struct ModelPayload<'a> {
    specialist_id: &'a str,
    image_dim: u32,
    n_classes: u32,
    class_names: &'a [&'a str],
    timestamp_ms: u64,
    weights: &'a [i8],
}

fn qlms_encode_payload(p: &ModelPayload) -> Vec<u8> {
    let total_weights = p.weights.len() as u32;
    let mut buf = Vec::with_capacity(64 + p.specialist_id.len() + p.weights.len());
    let id_bytes = p.specialist_id.as_bytes();
    buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf.extend_from_slice(&p.image_dim.to_le_bytes());
    buf.extend_from_slice(&p.n_classes.to_le_bytes());
    buf.extend_from_slice(&total_weights.to_le_bytes());
    buf.extend_from_slice(&(p.class_names.len() as u16).to_le_bytes());
    for name in p.class_names {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        buf.extend_from_slice(nb);
    }
    buf.extend_from_slice(&p.timestamp_ms.to_le_bytes());
    // weights as raw i8 bytes
    let wbytes: &[u8] =
        unsafe { std::slice::from_raw_parts(p.weights.as_ptr() as *const u8, p.weights.len()) };
    buf.extend_from_slice(wbytes);
    buf
}

fn qlms_encode_frame(payload: &[u8]) -> Vec<u8> {
    let sig = hmac_sha256(&shared_key(), payload);
    let mut buf = Vec::with_capacity(4 + 2 + 2 + 32 + 4 + payload.len());
    buf.extend_from_slice(QLMS_MAGIC);
    buf.extend_from_slice(&QLMS_VERSION.to_le_bytes());
    buf.extend_from_slice(&QLMS_KIND_MODEL.to_le_bytes());
    buf.extend_from_slice(&sig);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

struct DecodedPayload {
    #[allow(dead_code)]
    specialist_id: String,
    #[allow(dead_code)]
    image_dim: u32,
    #[allow(dead_code)]
    n_classes: u32,
    #[allow(dead_code)]
    class_names: Vec<String>,
    #[allow(dead_code)]
    timestamp_ms: u64,
    weights: Vec<i8>,
}

fn qlms_decode_frame(data: &[u8]) -> Result<DecodedPayload, String> {
    if data.len() < 4 + 2 + 2 + 32 + 4 {
        return Err("frame too short".into());
    }
    if &data[0..4] != QLMS_MAGIC {
        return Err("bad magic".into());
    }
    let mut sig = [0u8; 32];
    sig.copy_from_slice(&data[8..40]);
    let payload_len = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
    let payload = &data[44..44 + payload_len];
    let expected = hmac_sha256(&shared_key(), payload);
    if !ct_eq(&expected, &sig) {
        return Err("hmac fail".into());
    }
    // payload decode
    let mut o = 0usize;
    let id_len = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
    o += 2;
    let id = String::from_utf8(payload[o..o + id_len].to_vec()).map_err(|e| e.to_string())?;
    o += id_len;
    let image_dim = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
    o += 4;
    let n_classes = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
    o += 4;
    let total_weights = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
    o += 4;
    let n_names = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
    o += 2;
    let mut class_names = Vec::with_capacity(n_names);
    for _ in 0..n_names {
        let nlen = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
        o += 2;
        class_names.push(
            String::from_utf8(payload[o..o + nlen].to_vec()).map_err(|e| e.to_string())?,
        );
        o += nlen;
    }
    let timestamp_ms = u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
    o += 8;
    let weights: Vec<i8> = payload[o..o + total_weights].iter().map(|&b| b as i8).collect();
    Ok(DecodedPayload {
        specialist_id: id,
        image_dim,
        n_classes,
        class_names,
        timestamp_ms,
        weights,
    })
}

// ---- MCP JSON-RPC equivalent ----
#[derive(Serialize, Deserialize)]
struct McpParams {
    specialist_id: String,
    image_dim: u32,
    n_classes: u32,
    class_names: Vec<String>,
    timestamp_ms: u64,
    hmac_hex: String,
    weights: Vec<i8>,
}

#[derive(Serialize, Deserialize)]
struct McpRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: McpParams,
}

fn mcp_build(weights: &[i8]) -> McpRequest {
    let hmac = hmac_sha256(&shared_key(), b"payload-placeholder");
    McpRequest {
        jsonrpc: "2.0".into(),
        id: 42,
        method: "weights.transfer".into(),
        params: McpParams {
            specialist_id: "demo-classifier-1700000000".into(),
            image_dim: 64,
            n_classes: 24,
            class_names: (0..24).map(|i| format!("class_{}", i)).collect(),
            timestamp_ms: 1_700_000_000_000,
            hmac_hex: hmac.iter().map(|b| format!("{:02x}", b)).collect(),
            weights: weights.to_vec(),
        },
    }
}

// ---- Localhost HTTP-like loopback ----
// Trivial framed echo over TCP — the "server" reads length-prefixed bytes,
// echoes a small "ACK" reply. Measures raw TCP+serialize RTT.
fn spawn_echo_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            if let Ok(mut s) = stream {
                let mut len_buf = [0u8; 4];
                if s.read_exact(&mut len_buf).is_err() {
                    continue;
                }
                let n = u32::from_le_bytes(len_buf) as usize;
                let mut buf = vec![0u8; n];
                if s.read_exact(&mut buf).is_err() {
                    continue;
                }
                // Respond with a short ACK
                let ack = b"OK";
                let _ = s.write_all(&(ack.len() as u32).to_le_bytes());
                let _ = s.write_all(ack);
            }
        }
    });
    port
}

fn rtt_send(port: u16, frame: &[u8]) -> u128 {
    let t0 = Instant::now();
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    s.set_nodelay(true).ok();
    s.write_all(&(frame.len() as u32).to_le_bytes()).unwrap();
    s.write_all(frame).unwrap();
    let mut len_buf = [0u8; 4];
    s.read_exact(&mut len_buf).unwrap();
    let n = u32::from_le_bytes(len_buf) as usize;
    let mut ack = vec![0u8; n];
    s.read_exact(&mut ack).unwrap();
    t0.elapsed().as_nanos() / 1000 // microseconds
}

fn main() {
    // TernaryBrain: 64-dim input × 24 neurons = 1536 ternary weights
    let weights: Vec<i8> = (0..1536).map(|i| ((i % 3) as i8) - 1).collect();
    let class_names: Vec<&str> = (0..24)
        .map(|_| "class")
        .collect();
    let payload = ModelPayload {
        specialist_id: "demo-classifier-1700000000",
        image_dim: 64,
        n_classes: 24,
        class_names: &class_names,
        timestamp_ms: 1_700_000_000_000,
        weights: &weights,
    };

    const N: usize = 1000;

    // ============ QLMS binary ============
    // Size (single-shot, representative)
    let pb = qlms_encode_payload(&payload);
    let qlms_frame = qlms_encode_frame(&pb);
    let qlms_size = qlms_frame.len();

    // Serialize timing
    let t = Instant::now();
    for _ in 0..N {
        let pb = qlms_encode_payload(&payload);
        let f = qlms_encode_frame(&pb);
        std::hint::black_box(f);
    }
    let qlms_ser_ns = t.elapsed().as_nanos() / (N as u128);

    // Deserialize timing
    let t = Instant::now();
    for _ in 0..N {
        let d = qlms_decode_frame(&qlms_frame).unwrap();
        std::hint::black_box(d.weights.len());
    }
    let qlms_deser_ns = t.elapsed().as_nanos() / (N as u128);

    // ============ MCP JSON ============
    let mcp = mcp_build(&weights);
    let mcp_bytes = serde_json::to_vec(&mcp).unwrap();
    let mcp_size = mcp_bytes.len();

    let t = Instant::now();
    for _ in 0..N {
        let bytes = serde_json::to_vec(&mcp).unwrap();
        std::hint::black_box(bytes);
    }
    let mcp_ser_ns = t.elapsed().as_nanos() / (N as u128);

    let t = Instant::now();
    for _ in 0..N {
        let v: McpRequest = serde_json::from_slice(&mcp_bytes).unwrap();
        std::hint::black_box(v.params.weights.len());
    }
    let mcp_deser_ns = t.elapsed().as_nanos() / (N as u128);

    // ============ Round-trip over localhost ============
    let port = spawn_echo_server();
    // Warmup
    for _ in 0..20 {
        rtt_send(port, &qlms_frame);
        rtt_send(port, &mcp_bytes);
    }
    let mut qlms_rtts = Vec::with_capacity(N);
    let mut mcp_rtts = Vec::with_capacity(N);
    for _ in 0..N {
        qlms_rtts.push(rtt_send(port, &qlms_frame));
        mcp_rtts.push(rtt_send(port, &mcp_bytes));
    }
    let qlms_rtt = qlms_rtts.iter().sum::<u128>() / (N as u128);
    let mcp_rtt = mcp_rtts.iter().sum::<u128>() / (N as u128);

    // ============ Report ============
    println!();
    println!("QLMS vs MCP JSON — Real Measurements (1000 iterations, TernaryBrain 64x24=1536 weights)");
    println!("================================================================================");
    println!("| Method    | Size     | Ser (ns)  | Deser (ns)  | RTT localhost (us) |");
    println!("|-----------|----------|-----------|-------------|--------------------|");
    println!(
        "| QLMS bin  | {:>5} B  | {:>7}   | {:>7}     | {:>10}         |",
        qlms_size, qlms_ser_ns, qlms_deser_ns, qlms_rtt
    );
    println!(
        "| MCP JSON  | {:>5} B  | {:>7}   | {:>7}     | {:>10}         |",
        mcp_size, mcp_ser_ns, mcp_deser_ns, mcp_rtt
    );
    println!();

    let size_ratio = mcp_size as f64 / qlms_size as f64;
    let ser_ratio = mcp_ser_ns as f64 / qlms_ser_ns.max(1) as f64;
    let deser_ratio = mcp_deser_ns as f64 / qlms_deser_ns.max(1) as f64;
    let rtt_ratio = mcp_rtt as f64 / qlms_rtt.max(1) as f64;

    println!("Ratios (MCP / QLMS):");
    println!("  Size       : {:.2}x  (QLMS is {:.2}x smaller)", size_ratio, size_ratio);
    println!("  Serialize  : {:.2}x  (QLMS is {:.2}x faster)", ser_ratio, ser_ratio);
    println!("  Deserialize: {:.2}x  (QLMS is {:.2}x faster)", deser_ratio, deser_ratio);
    println!("  RTT        : {:.2}x  (QLMS is {:.2}x faster)", rtt_ratio, rtt_ratio);
    println!();
    println!("Caveats: localhost loopback (no real network); single-shot frames (no batching);");
    println!("         echo server is length-prefixed TCP (not full HTTP stack).");
}
