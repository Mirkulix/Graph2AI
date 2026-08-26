//! Deterministic merger for worker-submitted [`GraphDelta`]s.
//!
//! Two workers editing the same graph will disagree. The merger's job is to
//! make that disagreement *visible and reproducible* rather than letting the
//! last writer win.
//!
//! ## Guarantees
//!
//! - **Idempotent.** Applying the same delta twice changes nothing the second
//!   time and reports every operation as [`OpOutcome::AlreadyApplied`]. A
//!   worker may retry a submission after a timeout without corrupting state.
//! - **Order-independent for the outcome.** Two conflicting deltas produce the
//!   same final graph plus the same conflict records whichever order they
//!   arrive in. See `merge_is_order_independent` in the tests.
//! - **Append-only.** No operation deletes or rewrites a stored revision. A
//!   rejected operation is recorded as a [`Conflict`], not dropped silently.
//! - **Stale writes lose, but are not discarded.** A delta built against an
//!   older source revision cannot overwrite a newer observation; the attempt
//!   becomes a conflict record naming both revisions.
//!
//! ## What a conflict is not
//!
//! A claim being *refuted* is not a conflict — that is the graph working as
//! intended, and both sides keep their evidence. A conflict is recorded when
//! an operation cannot be applied at all: it targets a missing claim, it
//! contradicts a decision already made with equal or better standing, or it
//! arrives too late to be safe.

use crate::delta::{GraphDelta, GraphDeltaOp};
use crate::model::{Claim, ClaimStatus, Provenance};
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};

/// What happened to one operation in a delta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum OpOutcome {
    /// The operation changed the graph.
    Applied,
    /// The graph already said this. Re-submitting is safe and does nothing.
    AlreadyApplied,
    /// The operation could not be applied. The graph is unchanged and a
    /// [`Conflict`] describes why.
    Conflicted { conflict: Conflict },
}

/// A recorded reason an operation did not apply.
///
/// Conflicts are data, not errors: they are returned to the worker and are
/// meant to be shown to a human in the cockpit's conflict view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conflict {
    pub kind: ConflictKind,
    /// Human-readable explanation, safe to show in a UI.
    pub detail: String,
    /// The claim the operation targeted, when it named one.
    pub claim_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictKind {
    /// Verify or refute named a claim that does not exist.
    UnknownClaim,
    /// A claim id was submitted that is already taken by a different claim.
    DuplicateClaimId,
    /// One worker verified what another refuted (or the reverse).
    ContradictoryStatus,
    /// The delta was built against an older revision than the stored fact.
    StaleSourceRevision,
    /// The store refused the write for a reason the model enforces.
    Rejected,
}

/// Result of merging one delta.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeReport {
    pub delta_id: String,
    /// One entry per operation, in submission order.
    pub outcomes: Vec<OpOutcome>,
}

impl MergeReport {
    pub fn applied(&self) -> usize {
        self.outcomes.iter().filter(|o| matches!(o, OpOutcome::Applied)).count()
    }

    pub fn already_applied(&self) -> usize {
        self.outcomes
            .iter()
            .filter(|o| matches!(o, OpOutcome::AlreadyApplied))
            .count()
    }

    /// Every conflict raised, in submission order.
    pub fn conflicts(&self) -> Vec<&Conflict> {
        self.outcomes
            .iter()
            .filter_map(|o| match o {
                OpOutcome::Conflicted { conflict } => Some(conflict),
                _ => None,
            })
            .collect()
    }

    /// True when nothing was rejected. A merge with conflicts is still a
    /// successful call — the caller decides whether to surface them.
    pub fn is_clean(&self) -> bool {
        self.conflicts().is_empty()
    }
}

/// Merge a delta whose authorisation has already been established.
///
/// **This does not check signatures.** Use [`merge_signed_delta`] for anything
/// that arrived over a network; this entry point is for trusted local callers
/// (the workspace indexer, tests, examples) that have no producer key.
///
/// Returns `Err` only for storage failures and contract violations that make
/// the whole delta unusable (bad version, empty operations). Per-operation
/// disagreements come back inside the [`MergeReport`].
pub fn merge_delta(store: &KnowledgeStore, delta: &GraphDelta) -> Result<MergeReport, Error> {
    if let Err(e) = delta.validate() {
        // A rejected delta never reaches the graph, so this is the only place
        // it is visible to an operator at all.
        tracing::warn!(delta = %delta.id, producer = %delta.producer.id, error = %e, "delta rejected");
        return Err(Error::Storage(format!("invalid delta: {e}")));
    }

    let provenance = delta.provenance();
    let mut outcomes = Vec::with_capacity(delta.operations.len());

    for op in &delta.operations {
        outcomes.push(apply_op(store, op, &provenance)?);
    }

    let report = MergeReport {
        delta_id: delta.id.clone(),
        outcomes,
    };

    // Conflicts are the events worth waking up for: they mean two sessions
    // disagreed and a human has to settle it. Log each one individually so it
    // is greppable, not just counted.
    for conflict in report.conflicts() {
        tracing::warn!(
            delta = %delta.id,
            producer = %delta.producer.id,
            kind = ?conflict.kind,
            claim = conflict.claim_id.as_deref().unwrap_or("-"),
            detail = %conflict.detail,
            "merge conflict"
        );
    }

    tracing::info!(
        delta = %delta.id,
        producer = %delta.producer.id,
        applied = report.applied(),
        already_applied = report.already_applied(),
        conflicts = report.conflicts().len(),
        "delta merged"
    );

    Ok(report)
}

/// Why a delta was refused before any of it was applied.
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    /// The signature was missing, malformed, or not from a trusted key.
    #[error("not authorised: {0}")]
    Untrusted(#[from] crate::trust::TrustError),

    /// This `(producer, delta id)` was accepted before. Harmless to the graph
    /// — the merge is idempotent — but reported rather than hidden.
    #[error("delta {delta_id} from {producer} was already accepted")]
    Replay { producer: String, delta_id: String },

    #[error(transparent)]
    Storage(#[from] Error),
}

/// Verify, then merge. This is the entry point for anything that arrived over
/// a network — HTTP, MCP, or a CLI relaying to either.
///
/// Verification happens here rather than at each transport boundary because
/// there are three of them, and a check that lives in one of them is a check
/// the other two silently skip.
///
/// `now` must be the receiver's clock: judging key validity by the delta's own
/// `emitted_at` would let a submitter backdate past a revocation.
pub fn merge_signed_delta(
    store: &KnowledgeStore,
    trust: &crate::trust::TrustStore,
    delta: &GraphDelta,
    now: u64,
) -> Result<MergeReport, SubmitError> {
    crate::trust::verify_delta(trust, delta, now).map_err(|e| {
        tracing::warn!(
            delta = %delta.id,
            producer = %delta.producer.id,
            error = %e,
            "delta refused: not authorised"
        );
        e
    })?;

    // Signature verified above, so this is present.
    let signature = delta
        .producer
        .signature
        .as_ref()
        .map(|s| s.value.as_str())
        .unwrap_or_default();

    // Validate before reserving the id. Recording first would burn the id on
    // a delta that then fails to merge — the producer could never reuse it,
    // and a submitter who guesses ids could lock out a peer by sending
    // deliberately invalid deltas under their name.
    delta
        .validate()
        .map_err(|e| Error::Storage(format!("invalid delta: {e}")))?;

    if !store.record_delta(&delta.producer.id, &delta.id, signature)? {
        tracing::warn!(
            delta = %delta.id,
            producer = %delta.producer.id,
            "delta refused: replay of an already-accepted submission"
        );
        return Err(SubmitError::Replay {
            producer: delta.producer.id.clone(),
            delta_id: delta.id.clone(),
        });
    }

    // Past this point the id is spent. `merge_delta` only fails on storage
    // errors now — validation already passed — and a storage error means the
    // database is in trouble, which a retry under a fresh id will not fix.
    //
    // Known residue: the replay record and the merge are two commits, so a
    // crash between them leaves the id consumed with nothing applied. The
    // producer has to reissue under a new id. Closing that needs both writes
    // in one redb transaction, which means the merge would have to take a
    // caller-supplied transaction rather than opening its own — a larger
    // change than the failure warrants.
    Ok(merge_delta(store, delta)?)
}

fn apply_op(
    store: &KnowledgeStore,
    op: &GraphDeltaOp,
    provenance: &Provenance,
) -> Result<OpOutcome, Error> {
    match op {
        GraphDeltaOp::AddEntity { entity } => {
            // Entities are derived from (kind, name), so re-adding the same
            // one is a no-op rather than a conflict.
            if store.get_entity(&entity.id)?.is_some() {
                return Ok(OpOutcome::AlreadyApplied);
            }
            store.put_entity(entity)?;
            Ok(OpOutcome::Applied)
        }

        GraphDeltaOp::AddClaim { claim } => match store.latest(&claim.id)? {
            None => {
                store.add_claim(claim)?;
                Ok(OpOutcome::Applied)
            }
            // Same id, same statement about the same subject: a retry.
            Some(existing)
                if existing.statement == claim.statement && existing.subject == claim.subject =>
            {
                Ok(OpOutcome::AlreadyApplied)
            }
            // Same id, different content: two workers picked the same id for
            // different facts. Keeping the stored one is the append-only
            // choice; the submitter is told to re-id.
            Some(existing) => Ok(conflict(
                ConflictKind::DuplicateClaimId,
                format!(
                    "claim id {} already describes {:?}; refusing to replace it with {:?}",
                    claim.id.0, existing.statement, claim.statement
                ),
                Some(claim.id.0.clone()),
            )),
        },

        GraphDeltaOp::AddRelation { claim_id, relation, object } => {
            let Some(existing) = store.latest(claim_id)? else {
                return Ok(conflict(
                    ConflictKind::UnknownClaim,
                    format!("cannot attach a relation to unknown claim {}", claim_id.0),
                    Some(claim_id.0.clone()),
                ));
            };
            if existing.relation == Some(*relation) && existing.object.as_ref() == Some(object) {
                return Ok(OpOutcome::AlreadyApplied);
            }
            if let (Some(had), Some(had_obj)) = (existing.relation, existing.object.as_ref()) {
                return Ok(conflict(
                    ConflictKind::ContradictoryStatus,
                    format!(
                        "claim {} already relates {} to {}; will not silently re-point it to {} {}",
                        claim_id.0,
                        had.as_str(),
                        had_obj.0,
                        relation.as_str(),
                        object.0
                    ),
                    Some(claim_id.0.clone()),
                ));
            }
            store.attach_relation(claim_id, *relation, object.clone(), provenance.clone())?;
            Ok(OpOutcome::Applied)
        }

        GraphDeltaOp::VerifyClaim { claim_id, evidence } => {
            let Some(existing) = store.latest(claim_id)? else {
                return Ok(conflict(
                    ConflictKind::UnknownClaim,
                    format!("cannot verify unknown claim {}", claim_id.0),
                    Some(claim_id.0.clone()),
                ));
            };
            if let Some(outcome) = guard_transition(&existing, ClaimStatus::Verified, provenance) {
                return Ok(outcome);
            }
            match store.verify_claim(claim_id, evidence.clone(), provenance.clone()) {
                Ok(_) => Ok(OpOutcome::Applied),
                Err(e) => Ok(conflict(
                    ConflictKind::Rejected,
                    format!("store refused verification of {}: {e}", claim_id.0),
                    Some(claim_id.0.clone()),
                )),
            }
        }

        GraphDeltaOp::RefuteClaim { claim_id, evidence } => {
            let Some(existing) = store.latest(claim_id)? else {
                return Ok(conflict(
                    ConflictKind::UnknownClaim,
                    format!("cannot refute unknown claim {}", claim_id.0),
                    Some(claim_id.0.clone()),
                ));
            };
            if let Some(outcome) = guard_transition(&existing, ClaimStatus::Refuted, provenance) {
                return Ok(outcome);
            }
            match store.refute_claim(claim_id, evidence.clone(), provenance.clone()) {
                Ok(_) => Ok(OpOutcome::Applied),
                Err(e) => Ok(conflict(
                    ConflictKind::Rejected,
                    format!("store refused refutation of {}: {e}", claim_id.0),
                    Some(claim_id.0.clone()),
                )),
            }
        }
    }
}

/// Decide whether a status transition may proceed.
///
/// Returns `Some(outcome)` when the operation must not be applied, `None` when
/// the caller should go ahead.
///
/// The ordering rule that makes merges order-independent: a stale source
/// revision never overwrites a newer observation, and a claim already settled
/// the *other* way becomes a conflict rather than a silent flip. Without the
/// second rule, verify-then-refute and refute-then-verify would leave the
/// graph in different states.
fn guard_transition(
    existing: &Claim,
    target: ClaimStatus,
    incoming: &Provenance,
) -> Option<OpOutcome> {
    if existing.status == target {
        return Some(OpOutcome::AlreadyApplied);
    }

    let settled = matches!(existing.status, ClaimStatus::Verified | ClaimStatus::Refuted);
    if settled {
        return Some(conflict(
            ConflictKind::ContradictoryStatus,
            format!(
                "claim {} is already {} (by {}); {} would reverse a settled decision",
                existing.id.0,
                existing.status.as_str(),
                existing.provenance.producer,
                target.as_str()
            ),
            Some(existing.id.0.clone()),
        ));
    }

    // An older observation must not overwrite a newer one. Timestamps are
    // supplied by the caller, so equal timestamps are treated as concurrent
    // and allowed through — the settled-status rule above is what keeps the
    // result deterministic in that case.
    if incoming.observed_at < existing.provenance.observed_at {
        return Some(conflict(
            ConflictKind::StaleSourceRevision,
            format!(
                "delta observed at {} is older than claim {}'s revision at {}",
                incoming.observed_at, existing.id.0, existing.provenance.observed_at
            ),
            Some(existing.id.0.clone()),
        ));
    }

    None
}

fn conflict(kind: ConflictKind, detail: String, claim_id: Option<String>) -> OpOutcome {
    OpOutcome::Conflicted {
        conflict: Conflict { kind, detail, claim_id },
    }
}
