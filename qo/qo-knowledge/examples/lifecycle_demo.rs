//! The whole knowledge lifecycle in one narrated run.
//!
//! `cargo run -p qo-knowledge --example lifecycle_demo`
//!
//! This is the showpiece: it walks proposal → verification → disagreement →
//! rot → healing → proof → health against a real store and real fixture files,
//! printing every step. It is the narrative twin of `tests/lifecycle.rs` (which
//! asserts the same story); here the story is told.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{ClaimId, EntityId, EntityKind, Evidence, EvidenceKind, Provenance};
use qo_knowledge::{
    build_receipt, divergences, heal_stale, health, merge_delta, propose_from_text, refresh_sources,
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

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = KnowledgeStore::open(dir.path().join("k.redb")).expect("open store");
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).expect("mkdir");
    std::fs::write(
        root.join("src/auth.rs"),
        "// auth hashes passwords with bcrypt\npub fn hash_password(pw: &str) -> String { bcrypt::hash(pw) }\n",
    )
    .expect("write auth.rs");

    let auth = EntityId::derive(EntityKind::File, "src/auth.rs");

    println!("=== 1. An LLM proposes findings (extraction) ===");
    let document = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|src/lib.rs
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+C|c2|file:src/auth.rs|auth validates tokens
";
    print!("{document}");
    let outcome = propose_from_text(document, &ProposalPolicy::default());
    assert!(outcome.is_ok(), "{:?}", outcome.violations);
    merge_delta(&store, &outcome.delta.expect("delta")).expect("merge");
    println!("-> both land as `proposed` (unverified).\n");

    println!("=== 2. Unverified proposals never reach the next session ===");
    let before = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    println!("context: {}", if before.is_empty() { "(empty)" } else { &before.text });
    println!();

    println!("=== 3. Sweep: the graph checks each proposal against source ===");
    let sweep = verify_all_proposals(&store, root, prov("sweeper", 1_700_000_001)).unwrap();
    println!("{}", sweep.render());
    let after = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    println!("context now carries c1:\n{}", after.text);

    println!("=== 4. Disagreement: another session is refuted, and it stays visible ===");
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
    println!("{}", divergences(&store).unwrap().render());

    println!("=== 5. Rot: the source moves on, the excerpt is gone ===");
    std::fs::write(
        root.join("src/auth.rs"),
        "fn probe() { /* auth hashes passwords with bcrypt (relocated) */ }\n",
    )
    .unwrap();
    println!("{}", refresh_sources(&store, root, prov("refresher", 1_700_000_300)).unwrap().render());

    println!("=== 6. Heal: the fact still holds, just relocated ===");
    println!("{}", heal_stale(&store, root, prov("healer", 1_700_000_400)).unwrap().render());

    println!("=== 7. The proof: a receipt with the whole trail ===");
    println!("{}", build_receipt(&store, &ClaimId("c1".into())).unwrap().render());

    println!("=== 8. Operator summary ===");
    println!("{}", health(&store).unwrap().render());

    println!("That is the closed loop: propose freely, verify deterministically, keep");
    println!("disagreements visible, notice rot, heal what still holds, and prove it all.");
}
