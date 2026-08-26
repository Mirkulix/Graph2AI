//! # qo-knowledge
//!
//! A durable, checkable knowledge layer for projects, agents and their runs.
//! It complements the execution and communication graph; it does not replace
//! it.
//!
//! ## The rule this crate exists to enforce
//!
//! An LLM may *propose* knowledge. It may not store an unbacked proposal as
//! truth. Every [`Claim`] carries [`Provenance`], and the only way to reach
//! [`ClaimStatus::Verified`] is [`KnowledgeStore::verify_claim`] with
//! supporting [`Evidence`].
//!
//! Contradicting claims are not overwritten. Each status change appends a
//! revision and marks the previous one superseded, so a disagreement stays
//! visible with both sides' evidence.
//!
//! ## Example
//!
//! ```
//! use qo_knowledge::*;
//!
//! let dir = tempfile::tempdir().unwrap();
//! let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
//!
//! let file = EntityId::derive(EntityKind::File, "src/auth.rs");
//! let prov = Provenance {
//!     producer: "researcher".into(),
//!     observed_at: 1_700_000_000,
//!     git_revision: None,
//!     run_id: None,
//! };
//!
//! // An agent guesses. It is not treated as truth.
//! let guess = Claim::proposed("c1", "auth uses bcrypt", file.clone(), prov.clone());
//! store.add_claim(&guess).unwrap();
//! assert!(store.load_bearing_context(&file, 10).unwrap().is_empty());
//!
//! // Someone checks, and points at the line that proves it.
//! store.verify_claim(
//!     &ClaimId("c1".into()),
//!     Evidence {
//!         kind: EvidenceKind::Source,
//!         locator: "src/auth.rs".into(),
//!         lines: Some((42, 42)),
//!         excerpt: Some("use bcrypt::hash;".into()),
//!         supports: true,
//!     },
//!     prov,
//! ).unwrap();
//!
//! // Now it counts.
//! assert_eq!(store.load_bearing_context(&file, 10).unwrap().len(), 1);
//! ```

pub mod archive;
pub mod context;
pub mod delta;
pub mod divergence;
pub mod extract;
pub mod health;
pub mod orbitql;
pub mod merge;
pub mod model;
pub mod receipt;
pub mod sourcecheck;
pub mod store;
pub mod trust;

pub use delta::{
    DeltaProducer, DeltaSignature, DeltaValidationError, GraphDelta, GraphDeltaOp,
    GRAPH_DELTA_VERSION,
};
pub use model::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance,
    Relation,
};
pub use archive::{export, import, list_backups, write_backup, Archive, ImportReport, ARCHIVE_VERSION};
pub use context::{compile_context, CompiledContext, ContextRequest};
pub use divergence::{divergences, Divergence, DivergenceReport};
pub use extract::{
    propose_from_text, proposal_system_prompt, ProposalOutcome, ProposalPolicy, ProposalViolation,
};
pub use health::{health, GraphHealth};
pub use merge::{
    merge_delta, merge_signed_delta, Conflict, ConflictKind, MergeReport, OpOutcome, SubmitError,
};
pub use orbitql::{from_orbitql, parse_recovering, to_orbitql, OrbitQlError, ParseOutcome};
pub use receipt::{build_receipt, Receipt};
pub use sourcecheck::{
    assess, distinctive_terms, heal_stale, refresh_sources, verify_all_proposals,
    verify_claim_against_source, Assessment, HealOutcome, HealOutcomeKind, HealReport, SourceCheck,
    StaleOutcome, StaleOutcomeKind, StaleReport, SweepOutcome, SweepReport, Verdict,
};
pub use store::KnowledgeStore;
pub use trust::{sign_delta, verify_delta, TrustError, TrustStore, TrustedKey, SIGNING_ALGORITHM};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("storage: {0}")]
    Storage(String),

    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("no such claim: {0}")]
    NoSuchClaim(String),

    #[error("claim already exists: {0}")]
    ClaimExists(String),

    /// An `Observed` claim with nothing to point at is a proposal, not an
    /// observation.
    #[error("an observed claim requires at least one piece of evidence")]
    ObservationWithoutEvidence,

    /// Verifying with evidence marked `supports: false` would let a caller
    /// launder a refutation into a confirmation.
    #[error("cannot verify a claim with counter-evidence")]
    CounterEvidenceForVerify,

    #[error("cannot refute a claim with supporting evidence")]
    SupportingEvidenceForRefute,
}

// redb's error types are many and all mean "storage failed" here.
macro_rules! from_redb {
    ($($t:ty),*) => {$(
        impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Error::Storage(e.to_string())
            }
        }
    )*};
}
from_redb!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError
);
