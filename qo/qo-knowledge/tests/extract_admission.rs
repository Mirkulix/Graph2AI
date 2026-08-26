//! Integration tests for the text-to-graph proposal pipeline.
//!
//! The unit tests in `extract.rs` cover the admission rules themselves; here
//! the pipeline is exercised against the real store, end to end: model text
//! in, proposed delta out, merged into the graph, and *not* load-bearing until
//! an authorised verifier promotes it.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{ClaimId, ClaimStatus, EntityId, EntityKind};
use qo_knowledge::{
    from_orbitql, merge_delta, propose_from_text, KnowledgeStore, ProposalPolicy,
};

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn auth() -> EntityId {
    EntityId::derive(EntityKind::File, "src/auth.rs")
}

/// A proposal merged into the graph stays a proposal: it never reaches the
/// next session as an established fact, and an authorised verifier is what
/// promotes it.
#[test]
fn a_proposal_stays_out_of_context_until_verified() {
    let (_dir, store) = store();

    // Model output — note it is a plain proposal, no OK/NO lines.
    let text = "\
DELTA|1|d-1
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
";
    let outcome = propose_from_text(text, &ProposalPolicy::default());
    assert!(outcome.is_ok(), "{:?}", outcome.violations);
    let delta = outcome.delta.unwrap();

    // Merged (locally trusted path; the signed path is covered elsewhere).
    let report = merge_delta(&store, &delta).unwrap();
    assert!(report.is_clean(), "{:?}", report.conflicts());
    assert_eq!(report.applied(), 4);

    // The claim exists and is proposed — and the next session is not told a
    // guess is a fact.
    let claim = store.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(claim.status, ClaimStatus::Proposed);
    assert!(
        compile_context(&store, &ContextRequest::about(auth()))
            .unwrap()
            .is_empty(),
        "an unverified proposal reached the next session as context"
    );

    // An authorised verifier points at the line that proves it.
    let verify = "\
DELTA|1|d-2
BY|reviewer-1|1700000100
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    assert!(merge_delta(&store, &from_orbitql(verify).unwrap())
        .unwrap()
        .is_clean());

    let claim = store.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(claim.status, ClaimStatus::Verified);
    let context = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert!(context.text.contains("bcrypt"));
}

/// Refusing a document is all-or-nothing: one bad line means nothing is
/// admitted, and the worker sees every problem at once.
#[test]
fn a_refused_proposal_writes_nothing() {
    let (_dir, store) = store();

    // Two violations: the OK line (LLMs may not promote) and the dangling
    // relation to an undeclared claim. The parser accepts both lines, so the
    // policy must catch them.
    let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth uses bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
+R|c9|depends_on|file:Cargo.toml
";
    let outcome = propose_from_text(text, &ProposalPolicy::default());
    assert!(!outcome.is_ok());
    assert!(outcome.delta.is_none());
    assert_eq!(outcome.violations.len(), 2, "{:?}", outcome.violations);
    assert!(outcome.violations.iter().all(|v| v.line.is_some()));

    // Nothing reached the graph.
    assert!(store.latest(&ClaimId("c1".into())).unwrap().is_none());
}

/// Proposals referencing entities the graph already knows are admitted
/// without re-declaring those entities — the server passes context entities
/// into the policy.
#[test]
fn known_entities_come_from_context() {
    let (_dir, store) = store();

    // Establish src/auth.rs and Cargo.toml as known entities first.
    let seed = "\
DELTA|1|d-0
BY|indexer|100
+E|file|src/auth.rs
+E|file|Cargo.toml
";
    merge_delta(&store, &from_orbitql(seed).unwrap()).unwrap();

    let known: Vec<EntityId> = store.list_entities().unwrap().into_iter().map(|e| e.id).collect();
    let policy = ProposalPolicy::default().with_known_entities(known);

    // The proposal never declares the entities, only references them.
    let text = "\
DELTA|1|d-1
BY|worker-3|200
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
";
    let outcome = propose_from_text(text, &policy);
    assert!(outcome.is_ok(), "{:?}", outcome.violations);
    assert!(merge_delta(&store, &outcome.delta.unwrap()).unwrap().is_clean());
}

/// The proposal path and the signed commit path stay distinct: a worker with
/// a key may still verify via the commit path, while model text may not.
#[test]
fn the_commit_path_still_allows_verification() {
    // `from_orbitql` + `merge_delta` accept OK lines (trusted local caller) —
    // only the proposal admission refuses them. This documents the split.
    let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    assert!(from_orbitql(text).is_ok());
    let outcome = propose_from_text(text, &ProposalPolicy::default());
    assert!(!outcome.is_ok(), "the proposal path must refuse OK lines");
}
