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

use axum::{extract::State, http::StatusCode, Json};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use qlang_agent::protocol::{decode_qlms_frame, GraphMessage};
use qlang_core::crypto::{Keypair};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::AppState;

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
    /// Hex-encoded 32-byte seed.
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
    (StatusCode::UNAUTHORIZED, Json(ErrorBody { error: msg.into() }))
}

fn hex_to_seed(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("seed_hex must be 64 hex chars, got {}", hex.len()));
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
        other => Err(format!("non-hex char: 0x{other:02x}")),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /qlms/v1.1/deliver — decode + verify a base64-wrapped QLMS envelope.
pub async fn deliver(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeliverRequest>,
) -> Result<Json<DeliverResponse>, ApiError> {
    match req.encoding.as_deref() {
        None | Some("base64") => {}
        Some(other) => return Err(bad_request(format!("unsupported encoding: {other}"))),
    }

    let bytes = B64
        .decode(req.frame.as_bytes())
        .map_err(|e| bad_request(format!("base64 decode failed: {e}")))?;

    let decoded = decode_qlms_frame(&bytes).map_err(|e| bad_request(format!("invalid QLMS frame: {e}")))?;

    if decoded.signed && !decoded.signature_verified {
        return Err(unauthorized("QLMS signature verification failed"));
    }

    let messages = decoded.messages;

    // Dispatch messages to the internal bus
    for msg in &messages {
        eprintln!(
            "[QLMS Bridge] Received graph from {} for {} (intent: {:?})",
            msg.from.name, msg.to.name, msg.intent
        );
        state.message_bus.send(msg.clone()).await;
    }

    Ok(Json(DeliverResponse {
        version: decoded.version,
        flags: decoded.flags,
        signed: decoded.signed,
        signature_verified: decoded.signature_verified,
        msg_count: decoded.msg_count,
        messages,
    }))
}

/// POST /qlms/v1.1/reply — wrap a message list into a (signed) QLMS frame.
pub async fn reply(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ReplyRequest>,
) -> Result<Json<ReplyResponse>, ApiError> {
    let mut conv = qlang_agent::protocol::AgentConversation::new();
    for m in req.messages {
        conv.send(m.from, m.to, m.graph, m.inputs, m.intent, m.in_reply_to);
    }

    let frame_bytes = match req.seed_hex.as_deref() {
        Some(hex) => {
            let seed = hex_to_seed(hex).map_err(bad_request)?;
            let kp = Keypair::from_seed(&seed);
            conv.to_signed_binary(&kp).map_err(|e| bad_request(format!("serialize signed: {e}")))?
        }
        None => conv.to_binary().map_err(|e| bad_request(format!("serialize unsigned: {e}")))?
    };

    let frame = B64.encode(&frame_bytes);
    Ok(Json(ReplyResponse {
        size_bytes: frame_bytes.len(),
        frame,
        encoding: "base64".to_string(),
    }))
}
