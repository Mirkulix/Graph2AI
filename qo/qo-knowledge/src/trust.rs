//! Who may write to the graph, and proof that they did.
//!
//! A signature proves *integrity* — the delta was not altered in transit. It
//! does not prove *authorisation*: anyone can generate a keypair and sign
//! their own forgery. The trust store is what turns one into the other, by
//! naming which keys an operator has decided to believe.
//!
//! ## The rule
//!
//! A delta is trusted when all of these hold:
//!
//! - its `producer.id` has an entry in the store;
//! - the `key_id` it names belongs to *that* producer (never to another);
//! - the key is active at the receiver's clock, not at the delta's own
//!   `emitted_at` — which the submitter controls and can therefore backdate;
//! - the Ed25519 signature verifies over [`GraphDelta::signing_payload`].
//!
//! ## Key rotation and revocation
//!
//! An entry may carry `accept_until` so a replaced key keeps working through
//! a rollout window. `revoked_at` overrides that immediately: a leaked key
//! must stop being accepted the moment it is known to be leaked, whatever the
//! overlap window says.

use crate::delta::GraphDelta;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The only algorithm accepted. Named explicitly so an attacker cannot
/// downgrade to something weaker by claiming a different one.
pub const SIGNING_ALGORITHM: &str = "ed25519";

/// One key an operator has decided to trust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustedKey {
    /// Names this key within its producer. Must be unique per producer.
    pub key_id: String,
    /// 64 hex characters — a 32-byte Ed25519 public key.
    pub public_key_hex: String,
    /// Unix seconds. The key is not accepted before this.
    pub active_from: u64,
    /// Unix seconds. When set, the key is accepted only up to this point —
    /// use it to phase out a rotated key without a flag day.
    pub accept_until: Option<u64>,
    /// Unix seconds. When set, the key is refused from this point on,
    /// regardless of `accept_until`. A revocation is not a schedule.
    pub revoked_at: Option<u64>,
    /// Free-form note for whoever reads the file later.
    pub comment: Option<String>,
}

impl TrustedKey {
    /// Is this key usable at `now`? `now` is the receiver's clock.
    fn is_active_at(&self, now: u64) -> Result<(), TrustError> {
        if let Some(revoked) = self.revoked_at {
            if now >= revoked {
                return Err(TrustError::KeyRevoked {
                    key_id: self.key_id.clone(),
                });
            }
        }
        if now < self.active_from {
            return Err(TrustError::KeyNotYetActive {
                key_id: self.key_id.clone(),
            });
        }
        if let Some(until) = self.accept_until {
            if now > until {
                return Err(TrustError::KeyExpired {
                    key_id: self.key_id.clone(),
                });
            }
        }
        Ok(())
    }

    fn decode_public_key(&self) -> Result<[u8; 32], TrustError> {
        decode_hex_32(&self.public_key_hex).ok_or(TrustError::MalformedTrustedKey {
            key_id: self.key_id.clone(),
        })
    }
}

/// Producer id -> the keys that producer may sign with.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrustStore {
    #[serde(default)]
    pub producers: HashMap<String, Vec<TrustedKey>>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a key for a producer, replacing any entry with the same `key_id`.
    pub fn trust(&mut self, producer: impl Into<String>, key: TrustedKey) {
        let keys = self.producers.entry(producer.into()).or_default();
        keys.retain(|existing| existing.key_id != key.key_id);
        keys.push(key);
    }

    pub fn is_empty(&self) -> bool {
        self.producers.values().all(|keys| keys.is_empty())
    }

    pub fn from_json(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Why a delta was not trusted. Every variant is safe to show a submitter —
/// none of them leak key material or say which keys exist.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TrustError {
    #[error("delta carries no signature")]
    Unsigned,

    #[error("unsupported signing algorithm {algorithm:?}; only {SIGNING_ALGORITHM} is accepted")]
    UnsupportedAlgorithm { algorithm: String },

    #[error("no trusted key for producer {producer:?}")]
    UnknownProducer { producer: String },

    #[error("producer {producer:?} has no key {key_id:?}")]
    UnknownKey { producer: String, key_id: String },

    #[error("key {key_id:?} is revoked")]
    KeyRevoked { key_id: String },

    #[error("key {key_id:?} is not active yet")]
    KeyNotYetActive { key_id: String },

    #[error("key {key_id:?} is past its acceptance window")]
    KeyExpired { key_id: String },

    #[error("trusted key {key_id:?} is malformed in the trust store")]
    MalformedTrustedKey { key_id: String },

    #[error("signature is not valid 128-character hex")]
    MalformedSignature,

    #[error("cannot derive the signing payload: {0}")]
    PayloadFailed(String),

    #[error("signature does not verify for producer {producer:?} key {key_id:?}")]
    BadSignature { producer: String, key_id: String },
}

/// Verify that `delta` was signed by a key this store trusts at `now`.
///
/// `now` must come from the receiver's clock. Using the delta's `emitted_at`
/// would let a submitter choose a moment when a revoked key was still valid.
pub fn verify_delta(store: &TrustStore, delta: &GraphDelta, now: u64) -> Result<(), TrustError> {
    let Some(signature) = &delta.producer.signature else {
        return Err(TrustError::Unsigned);
    };

    if signature.algorithm != SIGNING_ALGORITHM {
        return Err(TrustError::UnsupportedAlgorithm {
            algorithm: signature.algorithm.clone(),
        });
    }

    let producer = &delta.producer.id;
    // The key is looked up *within* this producer. A key that is trusted for
    // someone else must not authorise a delta claiming this producer's name,
    // or one legitimate signer could impersonate another in provenance.
    let keys = store
        .producers
        .get(producer)
        .filter(|keys| !keys.is_empty())
        .ok_or_else(|| TrustError::UnknownProducer {
            producer: producer.clone(),
        })?;

    let key = keys
        .iter()
        .find(|k| k.key_id == signature.key_id)
        .ok_or_else(|| TrustError::UnknownKey {
            producer: producer.clone(),
            key_id: signature.key_id.clone(),
        })?;

    // Cheap checks before the expensive one: an unknown or revoked key should
    // never cost a curve operation.
    key.is_active_at(now)?;
    let public_key = key.decode_public_key()?;
    let signature_bytes = decode_hex_64(&signature.value).ok_or(TrustError::MalformedSignature)?;

    let payload = delta
        .signing_payload()
        .map_err(|e| TrustError::PayloadFailed(e.to_string()))?;

    if !qlang_core::crypto::Keypair::verify(&public_key, &payload, &signature_bytes) {
        return Err(TrustError::BadSignature {
            producer: producer.clone(),
            key_id: signature.key_id.clone(),
        });
    }

    Ok(())
}

/// Sign a delta in place, so a worker can produce something this store accepts.
///
/// Returns the signature that was attached. The seed is the caller's private
/// key material and is never stored or logged here.
pub fn sign_delta(
    delta: &mut GraphDelta,
    key_id: impl Into<String>,
    seed: &[u8; 32],
) -> Result<crate::delta::DeltaSignature, TrustError> {
    // Clear any existing signature first: the payload is defined over the
    // unsigned delta, and leaving a stale value here would sign the wrong
    // bytes on a re-sign.
    delta.producer.signature = None;

    // The text layer derives every claim's provenance from the `BY` line, so
    // a claim whose provenance disagrees with its producer would come back
    // different after a round-trip — and the signature would then fail on the
    // receiving side for no visible reason. Normalise here instead, so what
    // gets signed is what will be parsed.
    normalise_claim_provenance(delta);

    let payload = delta
        .signing_payload()
        .map_err(|e| TrustError::PayloadFailed(e.to_string()))?;

    let keypair = qlang_core::crypto::Keypair::from_seed(seed);
    let signature = crate::delta::DeltaSignature {
        algorithm: SIGNING_ALGORITHM.to_string(),
        key_id: key_id.into(),
        value: encode_hex(&keypair.sign(&payload)),
    };
    delta.producer.signature = Some(signature.clone());
    Ok(signature)
}

/// Make every claim's provenance match what the parser will derive from the
/// producer line.
///
/// This is not a convenience: without it, `sign -> to_orbitql -> from_orbitql
/// -> verify` can fail on a delta nobody tampered with, because the text layer
/// reconstructs provenance rather than carrying it per claim. Signing the
/// normalised form makes the round trip stable by construction.
fn normalise_claim_provenance(delta: &mut GraphDelta) {
    let provenance = delta.provenance();
    for op in &mut delta.operations {
        if let crate::delta::GraphDeltaOp::AddClaim { claim } = op {
            claim.provenance = provenance.clone();
        }
    }
}

/// The public key matching a seed, as the hex a trust store entry expects.
pub fn public_key_hex(seed: &[u8; 32]) -> String {
    encode_hex(&qlang_core::crypto::Keypair::from_seed(seed).public_key())
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex_32(hex: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    decode_hex_into(hex, &mut out).then_some(out)
}

fn decode_hex_64(hex: &str) -> Option<[u8; 64]> {
    let mut out = [0u8; 64];
    decode_hex_into(hex, &mut out).then_some(out)
}

fn decode_hex_into(hex: &str, out: &mut [u8]) -> bool {
    if hex.len() != out.len() * 2 {
        return false;
    }
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        match (nibble(pair[0]), nibble(pair[1])) {
            (Some(hi), Some(lo)) => out[i] = (hi << 4) | lo,
            _ => return false,
        }
    }
    true
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}
