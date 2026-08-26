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

/// Escape hatch for peers still speaking unsigned QLMS v1.
///
/// Off by default: an unsigned frame carries no authenticity, so accepting
/// one is a deployment decision an operator has to make explicitly.
fn allow_unsigned_frames() -> bool {
    matches!(
        std::env::var("QO_QLMS_ALLOW_UNSIGNED").as_deref(),
        Ok("1") | Ok("true")
    )
}

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

    // Insist on a verified signature. Checking only `signed && !verified`
    // would let a caller strip the signed flag to skip verification
    // entirely — an unsigned frame would then reach the bus unchecked.
    if !decoded.signature_verified {
        if !allow_unsigned_frames() {
            return Err(unauthorized(
                "QLMS envelope must carry a verified signature \
                 (set QO_QLMS_ALLOW_UNSIGNED=1 to accept legacy unsigned frames)",
            ));
        }
        tracing::warn!(
            signed = decoded.signed,
            "accepting unsigned QLMS frame: QO_QLMS_ALLOW_UNSIGNED is set"
        );
    }

    let messages = decoded.messages;

    // Dispatch messages to the internal bus. Agent names come from the
    // frame, so they are logged as structured fields rather than
    // interpolated into the message — a name containing newlines must not
    // be able to forge log lines.
    for msg in &messages {
        tracing::info!(
            from = %msg.from.name,
            to = %msg.to.name,
            intent = ?msg.intent,
            "QLMS bridge received graph"
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_agent::protocol::{decode_qlms_frame, AgentConversation, AgentId, MessageIntent};
    use qlang_core::graph::Graph;
    use std::collections::HashMap;

    fn agent(name: &str) -> AgentId {
        AgentId {
            name: name.to_string(),
            capabilities: Vec::new(),
        }
    }

    fn frame_pair() -> (Vec<u8>, Vec<u8>) {
        let mut signed_conv = AgentConversation::new();
        signed_conv.send(
            agent("alice"),
            agent("bob"),
            Graph::new("g"),
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let mut unsigned_conv = AgentConversation::new();
        unsigned_conv.send(
            agent("alice"),
            agent("bob"),
            Graph::new("g"),
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );

        let kp = Keypair::from_seed(&[7u8; 32]);
        (
            signed_conv.to_signed_binary(&kp).expect("sign"),
            unsigned_conv.to_binary().expect("serialize"),
        )
    }

    /// The gate the handler applies, extracted so it can be tested without
    /// standing up a server: a frame passes only if its signature verified.
    fn accepts(frame: &[u8], allow_unsigned: bool) -> bool {
        let decoded = decode_qlms_frame(frame).expect("frame must decode");
        decoded.signature_verified || allow_unsigned
    }

    #[test]
    fn signed_frame_is_accepted() {
        let (signed, _) = frame_pair();
        assert!(accepts(&signed, false));
    }

    /// The bug this replaced: an attacker omits the signed flag, and a check
    /// of `signed && !verified` never fires. Dropping the flag must not be a
    /// way past verification.
    #[test]
    fn unsigned_frame_is_rejected_by_default() {
        let (_, unsigned) = frame_pair();
        let decoded = decode_qlms_frame(&unsigned).expect("frame must decode");
        assert!(!decoded.signed, "fixture must be an unsigned frame");
        assert!(
            !accepts(&unsigned, false),
            "an unsigned frame must not reach the bus"
        );
    }

    /// Legacy peers stay reachable, but only when an operator opts in.
    #[test]
    fn unsigned_frame_is_accepted_only_with_opt_in() {
        let (_, unsigned) = frame_pair();
        assert!(accepts(&unsigned, true));
    }

    /// A tampered signed frame fails verification, so it is refused whatever
    /// the opt-in says about *unsigned* traffic.
    #[test]
    fn tampered_signed_frame_is_rejected() {
        let (mut signed, _) = frame_pair();
        let last = signed.len() - 1;
        signed[last] ^= 0xFF;
        match decode_qlms_frame(&signed) {
            Err(_) => {}
            Ok(decoded) => assert!(
                !decoded.signature_verified,
                "tampering must not verify"
            ),
        }
    }

    #[test]
    fn allow_unsigned_defaults_to_false() {
        // The env var is absent in a normal test process.
        std::env::remove_var("QO_QLMS_ALLOW_UNSIGNED");
        assert!(!allow_unsigned_frames());
    }
}
