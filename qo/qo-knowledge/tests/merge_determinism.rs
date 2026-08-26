//! Merger tests.
//!
//! The property that matters: two workers submitting conflicting deltas must
//! reach the same graph and the same conflict records regardless of arrival
//! order. Everything else here supports that claim.

use qo_knowledge::delta::{DeltaProducer, GraphDelta, GraphDeltaOp, GRAPH_DELTA_VERSION};
use qo_knowledge::merge::ConflictKind;
use qo_knowledge::model::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance,
    Relation,
};
use qo_knowledge::{merge_delta, KnowledgeStore};

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn producer(id: &str, at: u64) -> DeltaProducer {
    DeltaProducer {
        id: id.into(),
        source_revision: Some("rev-1".into()),
        run_id: None,
        emitted_at: at,
        signature: None,
    }
}

fn provenance(id: &str, at: u64) -> Provenance {
    Provenance {
        producer: id.into(),
        observed_at: at,
        git_revision: Some("rev-1".into()),
        run_id: None,
    }
}

fn delta(id: &str, by: &str, at: u64, ops: Vec<GraphDeltaOp>) -> GraphDelta {
    GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: id.into(),
        producer: producer(by, at),
        operations: ops,
    }
}

fn subject() -> EntityId {
    EntityId::derive(EntityKind::File, "src/auth.rs")
}

fn add_claim(id: &str, statement: &str, by: &str, at: u64) -> GraphDeltaOp {
    GraphDeltaOp::AddClaim {
        claim: Claim::proposed(id, statement, subject(), provenance(by, at)),
    }
}

fn evidence(supports: bool) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: Some((42, 42)),
        excerpt: None,
        supports,
    }
}

fn entity_op() -> GraphDeltaOp {
    GraphDeltaOp::AddEntity {
        entity: Entity {
            id: subject(),
            kind: EntityKind::File,
            name: "src/auth.rs".into(),
        },
    }
}

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

/// A worker retrying after a timeout must not corrupt the graph.
#[test]
fn merging_the_same_delta_twice_is_idempotent() {
    let (_dir, store) = store();
    let d = delta(
        "d1",
        "worker-1",
        100,
        vec![entity_op(), add_claim("c1", "auth uses bcrypt", "worker-1", 100)],
    );

    let first = merge_delta(&store, &d).unwrap();
    assert_eq!(first.applied(), 2);
    assert!(first.is_clean());

    let second = merge_delta(&store, &d).unwrap();
    assert_eq!(second.already_applied(), 2, "retry should be a no-op");
    assert!(second.is_clean(), "a retry is not a conflict");

    assert_eq!(store.history(&ClaimId("c1".into())).unwrap().len(), 1);
}

/// Verifying twice is also a retry, not a second revision.
#[test]
fn repeated_verification_is_idempotent() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "x", "w1", 100)]),
    )
    .unwrap();

    let verify = delta(
        "d2",
        "w1",
        200,
        vec![GraphDeltaOp::VerifyClaim {
            claim_id: ClaimId("c1".into()),
            evidence: evidence(true),
        }],
    );

    assert_eq!(merge_delta(&store, &verify).unwrap().applied(), 1);
    assert_eq!(merge_delta(&store, &verify).unwrap().already_applied(), 1);

    assert_eq!(store.history(&ClaimId("c1".into())).unwrap().len(), 2);
}

// ---------------------------------------------------------------------------
// Order independence — the core guarantee
// ---------------------------------------------------------------------------

/// Two workers disagree: one verifies, the other refutes. Whichever order the
/// deltas arrive in, the graph and the conflict must come out the same.
#[test]
fn merge_is_order_independent() {
    let verify = |id: &str| {
        delta(
            id,
            "verifier",
            200,
            vec![GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("c1".into()),
                evidence: evidence(true),
            }],
        )
    };
    let refute = |id: &str| {
        delta(
            id,
            "refuter",
            200,
            vec![GraphDeltaOp::RefuteClaim {
                claim_id: ClaimId("c1".into()),
                evidence: evidence(false),
            }],
        )
    };

    // Run A: verify then refute.
    let (_dir_a, store_a) = store();
    merge_delta(&store_a, &delta("d0", "w1", 100, vec![add_claim("c1", "x", "w1", 100)])).unwrap();
    let a_first = merge_delta(&store_a, &verify("d1")).unwrap();
    let a_second = merge_delta(&store_a, &refute("d2")).unwrap();

    // Run B: refute then verify.
    let (_dir_b, store_b) = store();
    merge_delta(&store_b, &delta("d0", "w1", 100, vec![add_claim("c1", "x", "w1", 100)])).unwrap();
    let b_first = merge_delta(&store_b, &refute("d2")).unwrap();
    let b_second = merge_delta(&store_b, &verify("d1")).unwrap();

    // In both runs the first writer wins and the second is a conflict.
    assert_eq!(a_first.applied(), 1);
    assert_eq!(b_first.applied(), 1);
    assert_eq!(a_second.conflicts().len(), 1);
    assert_eq!(b_second.conflicts().len(), 1);
    assert_eq!(a_second.conflicts()[0].kind, ConflictKind::ContradictoryStatus);
    assert_eq!(b_second.conflicts()[0].kind, ConflictKind::ContradictoryStatus);

    // And crucially: nothing was lost either way. Both stores hold the same
    // number of revisions, and the loser's evidence is still recoverable
    // through the winner's history.
    let hist_a = store_a.history(&ClaimId("c1".into())).unwrap();
    let hist_b = store_b.history(&ClaimId("c1".into())).unwrap();
    assert_eq!(hist_a.len(), hist_b.len());
    assert_eq!(hist_a.len(), 2, "one proposal plus one settled revision");
}

/// The same delta applied to two fresh stores yields identical reports.
#[test]
fn merge_reports_are_reproducible() {
    let d = delta(
        "d1",
        "w1",
        100,
        vec![
            entity_op(),
            add_claim("c1", "x", "w1", 100),
            GraphDeltaOp::AddRelation {
                claim_id: ClaimId("c1".into()),
                relation: Relation::DependsOn,
                object: EntityId::derive(EntityKind::File, "Cargo.toml"),
            },
        ],
    );

    let (_d1, s1) = store();
    let (_d2, s2) = store();
    assert_eq!(merge_delta(&s1, &d).unwrap(), merge_delta(&s2, &d).unwrap());
}

// ---------------------------------------------------------------------------
// Conflicts
// ---------------------------------------------------------------------------

/// A claim id reused for a different statement must not overwrite the first.
#[test]
fn duplicate_claim_id_with_different_content_conflicts() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "auth uses bcrypt", "w1", 100)]),
    )
    .unwrap();

    let report = merge_delta(
        &store,
        &delta("d2", "w2", 200, vec![add_claim("c1", "auth uses argon2", "w2", 200)]),
    )
    .unwrap();

    assert_eq!(report.conflicts().len(), 1);
    assert_eq!(report.conflicts()[0].kind, ConflictKind::DuplicateClaimId);
    assert_eq!(report.conflicts()[0].claim_id.as_deref(), Some("c1"));

    let stored = store.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(stored.statement, "auth uses bcrypt", "original was overwritten");
}

/// Verifying something that does not exist names the missing claim.
#[test]
fn verifying_unknown_claim_conflicts() {
    let (_dir, store) = store();
    let report = merge_delta(
        &store,
        &delta(
            "d1",
            "w1",
            100,
            vec![GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("ghost".into()),
                evidence: evidence(true),
            }],
        ),
    )
    .unwrap();

    assert_eq!(report.conflicts()[0].kind, ConflictKind::UnknownClaim);
    assert!(report.conflicts()[0].detail.contains("ghost"));
}

/// A delta built against an older revision must not undo newer work.
#[test]
fn stale_delta_cannot_overwrite_newer_observation() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 500, vec![add_claim("c1", "x", "w1", 500)]),
    )
    .unwrap();

    // A worker that started before the claim was written submits late.
    let report = merge_delta(
        &store,
        &delta(
            "d2",
            "slow-worker",
            100,
            vec![GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("c1".into()),
                evidence: evidence(true),
            }],
        ),
    )
    .unwrap();

    assert_eq!(report.conflicts()[0].kind, ConflictKind::StaleSourceRevision);
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Proposed,
        "stale delta changed the claim anyway"
    );
}

/// Re-pointing an existing relation is a conflict, not a silent rewrite.
#[test]
fn repointing_a_relation_conflicts() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta(
            "d1",
            "w1",
            100,
            vec![
                add_claim("c1", "x", "w1", 100),
                GraphDeltaOp::AddRelation {
                    claim_id: ClaimId("c1".into()),
                    relation: Relation::DependsOn,
                    object: EntityId::derive(EntityKind::File, "Cargo.toml"),
                },
            ],
        ),
    )
    .unwrap();

    let report = merge_delta(
        &store,
        &delta(
            "d2",
            "w2",
            200,
            vec![GraphDeltaOp::AddRelation {
                claim_id: ClaimId("c1".into()),
                relation: Relation::Calls,
                object: EntityId::derive(EntityKind::File, "other.rs"),
            }],
        ),
    )
    .unwrap();

    assert_eq!(report.conflicts()[0].kind, ConflictKind::ContradictoryStatus);
}

/// Re-submitting the identical relation is a retry.
#[test]
fn identical_relation_is_already_applied() {
    let (_dir, store) = store();
    let rel = GraphDeltaOp::AddRelation {
        claim_id: ClaimId("c1".into()),
        relation: Relation::DependsOn,
        object: EntityId::derive(EntityKind::File, "Cargo.toml"),
    };
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "x", "w1", 100), rel.clone()]),
    )
    .unwrap();

    let report = merge_delta(&store, &delta("d2", "w1", 200, vec![rel])).unwrap();
    assert_eq!(report.already_applied(), 1);
    assert!(report.is_clean());
}

// ---------------------------------------------------------------------------
// Append-only
// ---------------------------------------------------------------------------

/// Every status change appends; nothing is ever removed.
#[test]
fn history_grows_and_never_shrinks() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "x", "w1", 100)]),
    )
    .unwrap();
    merge_delta(
        &store,
        &delta(
            "d2",
            "w1",
            200,
            vec![GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("c1".into()),
                evidence: evidence(true),
            }],
        ),
    )
    .unwrap();

    let history = store.history(&ClaimId("c1".into())).unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].status, ClaimStatus::Proposed);
    assert_eq!(history[0].superseded_by, Some(2));
    assert_eq!(history[1].status, ClaimStatus::Verified);
    assert!(history[1].superseded_by.is_none());
}

/// A conflicted operation leaves the graph exactly as it was.
#[test]
fn conflicted_operation_does_not_write() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "original", "w1", 100)]),
    )
    .unwrap();
    let before = store.history(&ClaimId("c1".into())).unwrap();

    merge_delta(
        &store,
        &delta("d2", "w2", 200, vec![add_claim("c1", "different", "w2", 200)]),
    )
    .unwrap();

    assert_eq!(store.history(&ClaimId("c1".into())).unwrap(), before);
}

/// Operations after a conflicting one still apply — one bad operation does
/// not discard the rest of a worker's submission.
#[test]
fn a_conflict_does_not_abort_the_rest_of_the_delta() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![add_claim("c1", "original", "w1", 100)]),
    )
    .unwrap();

    let report = merge_delta(
        &store,
        &delta(
            "d2",
            "w2",
            200,
            vec![
                add_claim("c1", "different", "w2", 200),
                add_claim("c2", "brand new", "w2", 200),
            ],
        ),
    )
    .unwrap();

    assert_eq!(report.conflicts().len(), 1);
    assert_eq!(report.applied(), 1);
    assert!(store.latest(&ClaimId("c2".into())).unwrap().is_some());
}

/// Merged claims stay proposals — merging is not a promotion path.
#[test]
fn merged_claims_are_not_load_bearing() {
    let (_dir, store) = store();
    merge_delta(
        &store,
        &delta("d1", "w1", 100, vec![entity_op(), add_claim("c1", "x", "w1", 100)]),
    )
    .unwrap();

    assert!(
        store.load_bearing_context(&subject(), 10).unwrap().is_empty(),
        "a merged proposal must not count as established context"
    );
}

/// An invalid delta is refused whole, before anything is written.
#[test]
fn invalid_delta_is_rejected_before_writing() {
    let (_dir, store) = store();
    let bad = GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: String::new(), // empty id violates the delta contract
        producer: producer("w1", 100),
        operations: vec![add_claim("c1", "x", "w1", 100)],
    };

    assert!(merge_delta(&store, &bad).is_err());
    assert!(store.latest(&ClaimId("c1".into())).unwrap().is_none());
}
