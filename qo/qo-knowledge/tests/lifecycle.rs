//! The complete knowledge lifecycle in one run.
//!
//! This is the capstone regression test: proposal → verification → disagreement
//! → rot → healing, exercised end to end against the real store and real
//! fixture files. If any single stage of the pipeline breaks, the whole story
//! breaks here — so `cargo test -p qo-knowledge --test lifecycle` is the one
//! command that reproduces the project's central claim.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{ClaimId, ClaimStatus, EntityId, EntityKind, Evidence, EvidenceKind, Provenance};
use qo_knowledge::{
    build_receipt, divergences, heal_stale, merge_delta, propose_from_text, refresh_sources,
    verify_all_proposals, KnowledgeStore, ProposalPolicy,
};

fn prov(producer: &str, at: u64) -> Provenance {
    Provenance {
        producer: producer.into(),
        observed_at: at,
        git_revision: None,
        run_id: None,
    }
}

#[test]
fn the_whole_lifecycle_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/auth.rs"),
        "// auth hashes passwords with bcrypt\npub fn hash_password(pw: &str) -> String { bcrypt::hash(pw) }\n",
    )
    .unwrap();

    let auth = EntityId::derive(EntityKind::File, "src/auth.rs");

    // 1. PROPOSE from LLM text — one substantiable claim, one not.
    let document = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|src/lib.rs
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+C|c2|file:src/auth.rs|auth validates tokens
";
    let outcome = propose_from_text(document, &ProposalPolicy::default());
    assert!(outcome.is_ok(), "{:?}", outcome.violations);
    assert!(merge_delta(&store, &outcome.delta.unwrap()).unwrap().is_clean());
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Proposed
    );

    // 2. A proposal is not load-bearing: the next session sees nothing.
    assert!(
        compile_context(&store, &ContextRequest::about(auth.clone()))
            .unwrap()
            .is_empty(),
        "a proposal leaked into context before verification"
    );

    // 3. SWEEP: the graph checks every proposal against source.
    let sweep = verify_all_proposals(&store, root, prov("sweeper", 1_700_000_001)).unwrap();
    assert_eq!(sweep.verified, 1, "{sweep:?}");
    assert_eq!(sweep.inconclusive, 1, "{sweep:?}");
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Verified
    );
    assert_eq!(
        store.latest(&ClaimId("c2".into())).unwrap().unwrap().status,
        ClaimStatus::Proposed
    );

    // 4. Now c1 is load-bearing context for the next session.
    let context = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    assert!(context.text.contains("bcrypt"), "{context:?}");

    // 5. DISAGREEMENT: a second session claims the opposite; it is refuted and
    //    the divergence report surfaces both sides.
    store
        .add_claim(&qo_knowledge::Claim::proposed(
            "c3",
            "auth uses md5",
            auth.clone(),
            prov("worker-9", 1_700_000_100),
        ))
        .unwrap();
    store
        .refute_claim(
            &ClaimId("c3".into()),
            Evidence {
                kind: EvidenceKind::Source,
                locator: "src/auth.rs".into(),
                lines: None,
                excerpt: None,
                supports: false,
            },
            prov("reviewer", 1_700_000_200),
        )
        .unwrap();
    let div = divergences(&store).unwrap();
    assert_eq!(div.divergences.len(), 1, "{div:?}");
    assert_eq!(div.divergences[0].subject, auth);

    // 6. ROT: the source moves on; the recorded excerpt is gone → stale.
    std::fs::write(
        root.join("src/auth.rs"),
        "fn probe() { /* auth hashes passwords with bcrypt (relocated) */ }\n",
    )
    .unwrap();
    let refresh = refresh_sources(&store, root, prov("refresher", 1_700_000_300)).unwrap();
    assert_eq!(refresh.stale, 1, "{refresh:?}");
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Stale
    );

    // 7. HEAL: the fact still holds, just relocated → re-verified with fresh
    //    evidence.
    let heal = heal_stale(&store, root, prov("healer", 1_700_000_400)).unwrap();
    assert_eq!(heal.healed, 1, "{heal:?}");
    assert_eq!(
        store.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Verified
    );

    // 8. The receipt proves the whole trail: proposed → verified → stale →
    //    verified.
    let receipt = build_receipt(&store, &ClaimId("c1".into())).unwrap();
    assert_eq!(receipt.history.len(), 4, "{receipt:?}");
    let rendered = receipt.render();
    assert!(rendered.contains("VERIFIED"), "{rendered}");
    assert!(rendered.contains("rev 4"), "{rendered}");
}
