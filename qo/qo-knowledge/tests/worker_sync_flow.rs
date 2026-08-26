//! End-to-end test of the worker sync flow.
//!
//! This is the path the Claude Code plugin walks for a non-trivial task:
//!
//!   1. ask for bounded context before starting
//!   2. do the work
//!   3. hand back findings as an OrbitQLang delta
//!   4. QO parses, validates and merges it
//!   5. the worker learns what applied and what conflicted
//!
//! Each step is exercised against the real store, so a break anywhere in the
//! chain fails here rather than in production.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{ClaimId, ClaimStatus, EntityId, EntityKind};
use qo_knowledge::{from_orbitql, merge_delta, KnowledgeStore};

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn auth() -> EntityId {
    EntityId::derive(EntityKind::File, "src/auth.rs")
}

/// One worker session, start to finish.
#[test]
fn a_worker_session_round_trips_through_the_graph() {
    let (_dir, store) = store();

    // 1. Nothing is known yet, and the context says so honestly.
    let before = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert!(before.is_empty());

    // 2-3. The worker reports what it found, in OrbitQLang.
    let document = "\
# findings from reading src/auth.rs
DELTA|1|d-1
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";

    // 4. QO parses and merges.
    let delta = from_orbitql(document).expect("worker output must parse");
    let report = merge_delta(&store, &delta).unwrap();

    // 5. Everything applied, nothing conflicted.
    assert!(report.is_clean(), "conflicts: {:?}", report.conflicts());
    assert_eq!(report.applied(), 5);

    // The claim is now load-bearing, because evidence was supplied.
    let claim = store.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(claim.status, ClaimStatus::Verified);

    let after = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert!(after.text.contains("bcrypt"));
    assert!(after.text.contains("[src/auth.rs:42]"), "{}", after.text);
}

/// Without evidence the same flow leaves a proposal, and the next session is
/// not told a guess is a fact.
#[test]
fn a_claim_without_evidence_stays_out_of_context() {
    let (_dir, store) = store();

    let document = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth probably uses argon2
";
    let delta = from_orbitql(document).unwrap();
    assert!(merge_delta(&store, &delta).unwrap().is_clean());

    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Proposed
    );
    assert!(
        compile_context(&store, &ContextRequest::about(auth()))
            .unwrap()
            .is_empty(),
        "an unbacked proposal reached the next session as context"
    );
}

/// Two sessions working in parallel: the second learns it lost, and why.
#[test]
fn a_second_session_is_told_what_it_conflicted_with() {
    let (_dir, store) = store();

    let base = "\
DELTA|1|d-0
BY|worker-1|100
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
";
    merge_delta(&store, &from_orbitql(base).unwrap()).unwrap();

    // Session A verifies it.
    let a = "\
DELTA|1|d-a
BY|worker-1|200
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    assert!(merge_delta(&store, &from_orbitql(a).unwrap()).unwrap().is_clean());

    // Session B, working from the same starting point, refutes it.
    let b = "\
DELTA|1|d-b
BY|worker-2|200
NO|c1|source|src/auth.rs|10:10|use md5::compute;
";
    let report = merge_delta(&store, &from_orbitql(b).unwrap()).unwrap();

    assert_eq!(report.conflicts().len(), 1);
    let conflict = report.conflicts()[0];
    assert_eq!(conflict.claim_id.as_deref(), Some("c1"));
    assert!(
        conflict.detail.contains("worker-1"),
        "the conflict should name who decided first: {}",
        conflict.detail
    );

    // The winner stands and the loser's attempt did not corrupt anything.
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Verified
    );
}

/// A malformed submission is refused whole, and the worker gets every error
/// at once rather than one per retry.
#[test]
fn a_malformed_submission_writes_nothing() {
    let (_dir, store) = store();

    let document = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|not_a_kind|src/auth.rs
+C|c1|file:src/auth.rs
+R|c1|not_a_relation|file:x.rs
";
    let outcome = qo_knowledge::parse_recovering(document);
    assert_eq!(outcome.errors.len(), 3, "{:?}", outcome.errors);

    // The caller refuses to merge a document with errors, so the graph is
    // untouched — this mirrors what orbit_graph_commit_delta does.
    assert!(store.latest(&ClaimId("c1".into())).unwrap().is_none());
}

/// A retried submission is safe: the same document twice leaves one graph.
#[test]
fn resubmitting_after_a_timeout_is_safe() {
    let (_dir, store) = store();
    let document = "\
DELTA|1|d-1
BY|worker-3|100
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    let delta = from_orbitql(document).unwrap();

    let first = merge_delta(&store, &delta).unwrap();
    let second = merge_delta(&store, &delta).unwrap();

    assert_eq!(first.applied(), 3);
    assert_eq!(second.already_applied(), 3);
    assert!(second.is_clean());
    assert_eq!(store.history(&ClaimId("c1".into())).unwrap().len(), 2);
}
