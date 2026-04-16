//! MCP ↔ QLMS Bridge endpoints (spec §15.2 / PRD Task 2.2).
//!
//! Exposes two HTTP routes that wrap binary QLMS envelopes in the
//! JSON-RPC-friendly base64 pattern described in
//! `spec/QLMS_PROTOCOL_v1_1.md` §15.2:
//!
//!   POST /qlms/v1.1/deliver
//!     body:  { "encoding": "base64", "frame": "<b64 envelope>" }
//!     reply: { version, flags, signed, signature_verified, msg_count, messages[] }
//!
//!   POST /qlms/v1.1/reply
//!     body:  { "messages": [...], "seed_hex"?: "<64 hex chars>" }
//!     reply: { "encoding": "base64", "frame": "<b64 envelope>", "size_bytes": N }
//!
//! `/deliver` accepts an inbound QLMS frame smuggled through an
//! MCP-style JSON-RPC channel, verifies its signature (if signed), and
//! hands back the decoded [`GraphMessage`] list.
//!
//! `/reply` wraps a locally-constructed message list back into a QLMS
//! envelope (signed iff `seed_hex` is supplied) and base64-encodes it so
//! the caller can slot it into the next MCP hop.
//!
//! No side effects on global state — these handlers are pure
//! parse + verify + re-encode. Dispatch to the wider QO runtime can be
//! layered on top once Task 5.1 wires up the VSCode extension.

use axum::{http::StatusCode, Json};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use qlang_agent::protocol::GraphMessage;
use qlang_core::crypto::{sha256, Keypair};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire format constants (mirror crates/qlang-agent/src/protocol.rs)
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = [0x51, 0x4C, 0x4D, 0x53];
const VERSION_V2: u16 = 2;
const FLAG_SIGNED: u16 = 0x0001;
const SIG_LEN: usize = 64;
const PUB_LEN: usize = 32;
const HASH_LEN: usize = 32;
/// Minimum v2 envelope size before any JSON payload.
const V2_HEADER_LEN: usize = 4 + 2 + 2 + SIG_LEN + PUB_LEN + HASH_LEN + 4;
/// Minimum v1 envelope size before the JSON payload.
const V1_HEADER_LEN: usize = 4 + 4;

// ---------------------------------------------------------------------------
// Request / response bodies
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct DeliverRequest {
    /// Optional; only `base64` is supported, missing is treated as base64.
    pub encoding: Option<String>,
    pub frame: String,
}

#[derive(Debug, Serialize)]
pub struct DeliverResponse {
    pub version: u16,
    pub flags: u16,
    pub signed: bool,
    pub signature_verified: bool,
    pub msg_count: u32,
    pub messages: Vec<GraphMessage>,
}

#[derive(Deserialize)]
pub struct ReplyRequest {
    pub messages: Vec<GraphMessage>,
    /// Hex-encoded 32-byte seed. When present the envelope is signed via
    /// `Keypair::from_seed(hex_decode(seed_hex))`; when absent a v1
    /// unsigned envelope is emitted.
    pub seed_hex: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ReplyResponse {
    pub encoding: String,
    pub frame: String,
    pub size_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}

type ApiError = (StatusCode, Json<ErrorBody>);

fn bad_request(msg: impl Into<String>) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(ErrorBody { error: msg.into() }))
}

fn unauthorized(msg: impl Into<String>) -> ApiError {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorBody { error: msg.into() }),
    )
}

// ---------------------------------------------------------------------------
// Envelope parsing
// ---------------------------------------------------------------------------

struct Parsed<'a> {
    version: u16,
    flags: u16,
    signed: bool,
    signature: Option<[u8; SIG_LEN]>,
    pubkey: Option<[u8; PUB_LEN]>,
    payload_hash: Option<[u8; HASH_LEN]>,
    msg_count: u32,
    payload: &'a [u8],
}

fn parse_envelope(buf: &[u8]) -> Result<Parsed<'_>, String> {
    if buf.len() < 4 || buf[..4] != MAGIC {
        return Err("bad QLMS magic".into());
    }
    // v2 disambiguation: bytes 4-5 LE == 2 AND bytes 6-7 LE has SIGNED bit.
    if buf.len() >= 8 {
        let maybe_ver = u16::from_le_bytes([buf[4], buf[5]]);
        let maybe_flags = u16::from_le_bytes([buf[6], buf[7]]);
        if maybe_ver == VERSION_V2 && (maybe_flags & FLAG_SIGNED) != 0 {
            return parse_v2(buf);
        }
    }
    parse_v1(buf)
}

fn parse_v1(buf: &[u8]) -> Result<Parsed<'_>, String> {
    if buf.len() < V1_HEADER_LEN {
        return Err(format!(
            "v1 envelope too short: got {} B, need >= {}",
            buf.len(),
            V1_HEADER_LEN
        ));
    }
    let msg_count = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Ok(Parsed {
        version: 1,
        flags: 0,
        signed: false,
        signature: None,
        pubkey: None,
        payload_hash: None,
        msg_count,
        payload: &buf[V1_HEADER_LEN..],
    })
}

fn parse_v2(buf: &[u8]) -> Result<Parsed<'_>, String> {
    if buf.len() < V2_HEADER_LEN {
        return Err(format!(
            "v2 envelope too short: got {} B, need >= {}",
            buf.len(),
            V2_HEADER_LEN
        ));
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    let flags = u16::from_le_bytes([buf[6], buf[7]]);
    let mut off = 8;

    let mut sig = [0u8; SIG_LEN];
    sig.copy_from_slice(&buf[off..off + SIG_LEN]);
    off += SIG_LEN;

    let mut pk = [0u8; PUB_LEN];
    pk.copy_from_slice(&buf[off..off + PUB_LEN]);
    off += PUB_LEN;

    let mut hash = [0u8; HASH_LEN];
    hash.copy_from_slice(&buf[off..off + HASH_LEN]);
    off += HASH_LEN;

    let msg_count = u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    off += 4;

    Ok(Parsed {
        version,
        flags,
        signed: (flags & FLAG_SIGNED) != 0,
        signature: Some(sig),
        pubkey: Some(pk),
        payload_hash: Some(hash),
        msg_count,
        payload: &buf[off..],
    })
}

/// Verify the full v2 envelope: payload-hash integrity AND keypair signature.
/// v1 envelopes always return `false` (they carry no signature at all).
fn verify_parsed(p: &Parsed<'_>) -> bool {
    let (Some(sig), Some(pk), Some(hash)) = (&p.signature, &p.pubkey, &p.payload_hash) else {
        return false;
    };
    if sha256(p.payload) != *hash {
        return false;
    }
    Keypair::verify(pk, hash, sig)
}

// ---------------------------------------------------------------------------
// Envelope building
// ---------------------------------------------------------------------------

fn build_v1_unsigned(payload: &[u8], count: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(V1_HEADER_LEN + payload.len());
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

fn build_v2_signed(kp: &Keypair, payload: &[u8], count: u32) -> Vec<u8> {
    let payload_hash = sha256(payload);
    let signature = kp.sign(&payload_hash);
    let mut buf = Vec::with_capacity(V2_HEADER_LEN + payload.len());
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION_V2.to_le_bytes());
    buf.extend_from_slice(&FLAG_SIGNED.to_le_bytes());
    buf.extend_from_slice(&signature);
    buf.extend_from_slice(kp.public_key());
    buf.extend_from_slice(&payload_hash);
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

fn hex_to_seed(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!(
            "seed_hex must be 64 hex chars (32 bytes), got {}",
            hex.len()
        ));
    }
    let mut out = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        other => Err(format!("non-hex char in seed_hex: 0x{other:02x}")),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /qlms/v1.1/deliver — decode + verify a base64-wrapped QLMS envelope.
pub async fn deliver(Json(req): Json<DeliverRequest>) -> Result<Json<DeliverResponse>, ApiError> {
    match req.encoding.as_deref() {
        None | Some("base64") => {}
        Some(other) => return Err(bad_request(format!("unsupported encoding: {other}"))),
    }

    let bytes = B64
        .decode(req.frame.as_bytes())
        .map_err(|e| bad_request(format!("base64 decode failed: {e}")))?;

    let parsed = parse_envelope(&bytes).map_err(bad_request)?;

    let sig_ok = parsed.signed && verify_parsed(&parsed);
    if parsed.signed && !sig_ok {
        return Err(unauthorized("QLMS signature verification failed"));
    }

    let messages: Vec<GraphMessage> = serde_json::from_slice(parsed.payload)
        .map_err(|e| bad_request(format!("payload JSON decode failed: {e}")))?;

    Ok(Json(DeliverResponse {
        version: parsed.version,
        flags: parsed.flags,
        signed: parsed.signed,
        // Unsigned frames are structurally valid but carry no cryptographic
        // proof — we report `false` rather than `true` to avoid implying a
        // trust decision the caller did not ask for.
        signature_verified: sig_ok,
        msg_count: parsed.msg_count,
        messages,
    }))
}

/// POST /qlms/v1.1/reply — wrap a message list into a (signed) QLMS frame.
pub async fn reply(Json(req): Json<ReplyRequest>) -> Result<Json<ReplyResponse>, ApiError> {
    let payload = serde_json::to_vec(&req.messages)
        .map_err(|e| bad_request(format!("serialize messages: {e}")))?;
    let count = req.messages.len() as u32;

    let frame_bytes = match req.seed_hex.as_deref() {
        Some(hex) => {
            let seed = hex_to_seed(hex).map_err(bad_request)?;
            let kp = Keypair::from_seed(&seed);
            build_v2_signed(&kp, &payload, count)
        }
        None => build_v1_unsigned(&payload, count),
    };

    let frame = B64.encode(&frame_bytes);
    Ok(Json(ReplyResponse {
        size_bytes: frame_bytes.len(),
        frame,
        encoding: "base64".to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Tests (deliver/reply round-trip against the real Keypair + hex helpers)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_agent::protocol::{
        AgentConversation, AgentId, Capability, MessageIntent,
    };
    use qlang_core::graph::Graph;
    use std::collections::HashMap;

    fn trainer() -> AgentId {
        AgentId {
            name: "trainer".into(),
            capabilities: vec![Capability::Execute, Capability::Train],
        }
    }

    fn compressor() -> AgentId {
        AgentId {
            name: "compressor".into(),
            capabilities: vec![Capability::Compress, Capability::Verify],
        }
    }

    fn build_signed_frame_b64(seed: [u8; 32]) -> String {
        let kp = Keypair::from_seed(&seed);
        let mut conv = AgentConversation::new();
        conv.send(
            trainer(),
            compressor(),
            Graph::new("mcp-bridge-test"),
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );
        B64.encode(conv.to_signed_binary(&kp).unwrap())
    }

    #[tokio::test]
    async fn deliver_verifies_signed_frame() {
        let seed = [0xA3u8; 32];
        let frame = build_signed_frame_b64(seed);

        let Json(resp) = deliver(Json(DeliverRequest {
            encoding: Some("base64".to_string()),
            frame,
        }))
        .await
        .expect("deliver must accept a valid signed frame");

        assert_eq!(resp.version, 2);
        assert!(resp.signed);
        assert!(resp.signature_verified, "HMAC signature must verify");
        assert_eq!(resp.msg_count, 1);
        assert_eq!(resp.messages.len(), 1);
    }

    #[tokio::test]
    async fn deliver_rejects_tampered_frame() {
        let seed = [0xB7u8; 32];
        let frame_b64 = build_signed_frame_b64(seed);
        let mut bytes = B64.decode(frame_b64.as_bytes()).unwrap();
        // Flip a bit inside the JSON payload (which the hash covers).
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let tampered = B64.encode(&bytes);

        let err = deliver(Json(DeliverRequest {
            encoding: Some("base64".into()),
            frame: tampered,
        }))
        .await
        .expect_err("tampered frame must be rejected");
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn deliver_accepts_unsigned_v1_frame() {
        let mut conv = AgentConversation::new();
        conv.send(
            trainer(),
            compressor(),
            Graph::new("v1-test"),
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );
        let bytes = conv.to_binary().unwrap();
        let frame = B64.encode(bytes);
        let Json(resp) = deliver(Json(DeliverRequest {
            encoding: None,
            frame,
        }))
        .await
        .expect("v1 frame must deliver");
        assert_eq!(resp.version, 1);
        assert!(!resp.signed);
        assert!(!resp.signature_verified);
    }

    #[tokio::test]
    async fn deliver_rejects_unknown_encoding() {
        let err = deliver(Json(DeliverRequest {
            encoding: Some("hex".into()),
            frame: "deadbeef".into(),
        }))
        .await
        .expect_err("unknown encoding must fail");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn deliver_rejects_bad_base64() {
        let err = deliver(Json(DeliverRequest {
            encoding: None,
            frame: "!!!not valid base64!!!".into(),
        }))
        .await
        .expect_err("malformed base64 must fail");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn deliver_rejects_bad_magic() {
        // Valid base64 but bytes do not start with QLMS.
        let frame = B64.encode(b"XXXXfake envelope");
        let err = deliver(Json(DeliverRequest {
            encoding: Some("base64".into()),
            frame,
        }))
        .await
        .expect_err("bad magic must fail");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn reply_then_deliver_round_trip_signed() {
        // Build a signed reply frame, then feed it back into deliver — the
        // same messages must come out.
        let seed_hex = "7f".repeat(32);
        let mut conv = AgentConversation::new();
        conv.send(
            trainer(),
            compressor(),
            Graph::new("round-trip"),
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let messages = conv.messages().to_vec();

        let Json(reply_body) = reply(Json(ReplyRequest {
            messages: messages.clone(),
            seed_hex: Some(seed_hex),
        }))
        .await
        .expect("signed reply must build");

        assert_eq!(reply_body.encoding, "base64");
        assert!(reply_body.size_bytes >= V2_HEADER_LEN);

        let Json(deliver_body) = deliver(Json(DeliverRequest {
            encoding: None,
            frame: reply_body.frame,
        }))
        .await
        .expect("signed reply must verify on deliver");

        assert!(deliver_body.signed);
        assert!(deliver_body.signature_verified);
        assert_eq!(deliver_body.messages.len(), messages.len());
        assert_eq!(deliver_body.messages[0].id, messages[0].id);
    }

    #[tokio::test]
    async fn reply_unsigned_emits_v1_frame() {
        let mut conv = AgentConversation::new();
        conv.send(
            trainer(),
            compressor(),
            Graph::new("v1-reply"),
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let Json(r) = reply(Json(ReplyRequest {
            messages: conv.messages().to_vec(),
            seed_hex: None,
        }))
        .await
        .expect("unsigned reply");
        let bytes = B64.decode(r.frame.as_bytes()).unwrap();
        assert_eq!(&bytes[..4], &MAGIC);
        // v1 has no version bytes; 8-byte header then payload. Recognise
        // this by delegating to parse_envelope — it must pick v1.
        let parsed = parse_envelope(&bytes).unwrap();
        assert_eq!(parsed.version, 1);
        assert!(!parsed.signed);
    }

    #[tokio::test]
    async fn reply_rejects_bad_seed_hex() {
        let err = reply(Json(ReplyRequest {
            messages: Vec::new(),
            seed_hex: Some("zz".repeat(32)),
        }))
        .await
        .expect_err("invalid hex must fail");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }
}
