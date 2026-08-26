//! Versioned, append-only graph deltas exchanged by agent workers.

use crate::{Claim, ClaimId, ClaimStatus, Entity, EntityId, Evidence, Provenance, Relation};
use serde::{Deserialize, Serialize};

/// The supported serialised graph-delta format.
pub const GRAPH_DELTA_VERSION: u16 = 1;

/// Domain separator for [`GraphDelta::signing_payload`].
///
/// Ed25519 signs arbitrary bytes, so a signature is only meaningful together
/// with a statement of *what kind of thing* was signed. Without this prefix, a
/// signature over a delta could be presented as a signature over any other
/// structure whose serialization happened to match.
const SIGNING_DOMAIN: &[u8] = b"orbitqlang.graph-delta.v";

/// A compact, auditable change set. Applying it remains a policy decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphDelta {
    pub version: u16,
    pub id: String,
    pub producer: DeltaProducer,
    pub operations: Vec<GraphDeltaOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeltaProducer {
    pub id: String,
    pub source_revision: Option<String>,
    pub run_id: Option<String>,
    pub emitted_at: u64,
    /// Signature metadata; cryptographic verification belongs to the transport
    /// boundary, not this data model.
    pub signature: Option<DeltaSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeltaSignature {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

/// No operation can silently replace or delete a previous graph fact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphDeltaOp {
    AddEntity { entity: Entity },
    /// Worker-written claims are always proposals until independently checked.
    AddClaim { claim: Claim },
    AddRelation { claim_id: ClaimId, relation: Relation, object: EntityId },
    VerifyClaim { claim_id: ClaimId, evidence: Evidence },
    RefuteClaim { claim_id: ClaimId, evidence: Evidence },
}

/// Deterministic error that a CLI or MCP adapter can surface to a worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaValidationError {
    pub operation: Option<usize>,
    pub message: String,
}

impl std::fmt::Display for DeltaValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.operation {
            Some(index) => write!(f, "operation {index}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}
impl std::error::Error for DeltaValidationError {}

impl GraphDelta {
    /// Validate a worker submission before it reaches the merger.
    pub fn validate(&self) -> Result<(), DeltaValidationError> {
        if self.version != GRAPH_DELTA_VERSION {
            return Err(DeltaValidationError { operation: None, message: format!("unsupported graph-delta version {}; expected {GRAPH_DELTA_VERSION}", self.version) });
        }
        if self.id.trim().is_empty() || self.producer.id.trim().is_empty() {
            return Err(DeltaValidationError { operation: None, message: "delta id and producer id must not be empty".into() });
        }
        if self.operations.is_empty() {
            return Err(DeltaValidationError { operation: None, message: "delta has no operations".into() });
        }
        for (index, operation) in self.operations.iter().enumerate() {
            let error = |message| DeltaValidationError { operation: Some(index), message };
            match operation {
                GraphDeltaOp::AddEntity { entity }
                    if entity.id.0.trim().is_empty() || entity.name.trim().is_empty() =>
                {
                    return Err(error("entity id and name must not be empty".into()));
                }
                GraphDeltaOp::AddClaim { claim } => {
                    if claim.id.0.trim().is_empty() || claim.statement.trim().is_empty() {
                        return Err(error("claim id and statement must not be empty".into()));
                    }
                    if claim.status != ClaimStatus::Proposed {
                        return Err(error("worker claim additions must have status proposed".into()));
                    }
                    if claim.revision != 1 || claim.superseded_by.is_some() {
                        return Err(error("new claim must start at revision 1 and not be superseded".into()));
                    }
                }
                GraphDeltaOp::AddRelation { claim_id, object, .. }
                    if claim_id.0.trim().is_empty() || object.0.trim().is_empty() =>
                {
                    return Err(error("relation claim id and object must not be empty".into()));
                }
                GraphDeltaOp::VerifyClaim { claim_id, evidence }
                    if claim_id.0.trim().is_empty() || !evidence.supports || evidence.locator.trim().is_empty() =>
                {
                    return Err(error("verification needs a claim id and supporting evidence locator".into()));
                }
                GraphDeltaOp::RefuteClaim { claim_id, evidence }
                    if claim_id.0.trim().is_empty() || evidence.supports || evidence.locator.trim().is_empty() =>
                {
                    return Err(error("refutation needs a claim id and counter-evidence locator".into()));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Stable JSON representation suitable for signed QLMS envelopes.
    pub fn to_canonical_json(&self) -> Result<String, DeltaValidationError> {
        self.validate()?;
        serde_json::to_string(self).map_err(|error| DeltaValidationError { operation: None, message: format!("cannot serialize graph delta: {error}") })
    }

    /// The exact bytes a producer signs, and a verifier re-derives.
    ///
    /// Three properties this has that [`Self::to_canonical_json`] does not:
    ///
    /// 1. **Signature-free.** `producer.signature` is cleared before
    ///    serializing. Signing over a structure that contains the signature
    ///    is circular — the value cannot be known before it is computed.
    /// 2. **Domain-separated.** The bytes are prefixed with a fixed tag and
    ///    the delta version, so a signature over a delta can never be
    ///    replayed as a signature over some other structure that happened to
    ///    serialize identically.
    /// 3. **Independent of the submitted text.** The OrbitQLang document may
    ///    carry comments, blank lines and incidental whitespace; two
    ///    semantically identical documents produce different bytes. The typed
    ///    delta does not.
    ///
    /// Note this deliberately does *not* call `validate()`: a verifier must
    /// be able to re-derive the signing payload of a delta it is about to
    /// reject, in order to report *why* it was rejected.
    pub fn signing_payload(&self) -> Result<Vec<u8>, DeltaValidationError> {
        let mut unsigned = self.clone();
        unsigned.producer.signature = None;

        let body = serde_json::to_string(&unsigned).map_err(|error| DeltaValidationError {
            operation: None,
            message: format!("cannot serialize graph delta for signing: {error}"),
        })?;

        let mut out = Vec::with_capacity(body.len() + 32);
        out.extend_from_slice(SIGNING_DOMAIN);
        out.extend_from_slice(self.version.to_string().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(body.as_bytes());
        Ok(out)
    }

    /// Parse and validate an adapter submission.
    pub fn from_json(input: &str) -> Result<Self, DeltaValidationError> {
        let delta: Self = serde_json::from_str(input).map_err(|error| DeltaValidationError { operation: None, message: format!("invalid graph-delta JSON: {error}") })?;
        delta.validate()?;
        Ok(delta)
    }

    /// Provenance used by the merger for operations in this delta.
    pub fn provenance(&self) -> Provenance {
        Provenance { producer: self.producer.id.clone(), observed_at: self.producer.emitted_at, git_revision: self.producer.source_revision.clone(), run_id: self.producer.run_id.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EntityKind;

    fn delta() -> GraphDelta {
        GraphDelta {
            version: GRAPH_DELTA_VERSION,
            id: "worker-7:1".into(),
            producer: DeltaProducer { id: "worker-7".into(), source_revision: Some("abc".into()), run_id: Some("run-1".into()), emitted_at: 7, signature: None },
            operations: vec![GraphDeltaOp::AddClaim { claim: Claim::proposed("claim-1", "a proposed change", EntityId::derive(EntityKind::File, "src/lib.rs"), Provenance { producer: "worker-7".into(), observed_at: 7, git_revision: Some("abc".into()), run_id: Some("run-1".into()) }) }],
        }
    }

    #[test]
    fn canonical_json_round_trips() {
        let original = delta();
        assert_eq!(GraphDelta::from_json(&original.to_canonical_json().unwrap()).unwrap(), original);
    }

    #[test]
    fn non_proposed_worker_claim_is_rejected() {
        let mut invalid = delta();
        let GraphDeltaOp::AddClaim { claim } = &mut invalid.operations[0] else { unreachable!() };
        claim.status = ClaimStatus::Observed;
        assert_eq!(invalid.validate().unwrap_err().operation, Some(0));
    }
}
