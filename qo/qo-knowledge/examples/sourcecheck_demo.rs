//! The graph checks its own claims against real source, printed step by step.
//!
//! `cargo run -p qo-knowledge --example sourcecheck_demo`
//!
//! Shows the deterministic bridge between "an LLM proposed it" and "the graph
//! checked it": a proposed claim is promoted to a verified fact only when every
//! distinctive term is literally present in the source, with the exact line
//! captured as evidence. Partial matches and path escapes are refused.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{ClaimStatus, EntityId, EntityKind, Evidence, EvidenceKind, Provenance};
use qo_knowledge::{verify_claim_against_source, KnowledgeStore, Verdict};

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

    // A workspace the graph can read: one file that substantiates the claim,
    // one that does not.
    std::fs::create_dir_all(root.join("src")).expect("mkdir");
    std::fs::write(
        root.join("src/auth.rs"),
        "// auth hashes passwords with bcrypt\npub fn hash_password(pw: &str) -> String {\n    bcrypt::hash(pw)\n}\n",
    )
    .expect("write auth.rs");
    std::fs::write(
        root.join("src/payments.rs"),
        "pub fn charge(amount: u32) { /* stripe */ }\n",
    )
    .expect("write payments.rs");

    println!("=== 1. An LLM proposes, pointing at its source ===");
    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    let mut claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        subject.clone(),
        prov("worker-3", 1_700_000_000),
    );
    claim.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&claim).expect("add claim");
    println!("   proposed: {}", claim.statement);

    println!("\n=== 2. The graph checks it against src/auth.rs ===");
    let check = verify_claim_against_source(&store, &qo_knowledge::ClaimId("c1".into()), root, prov("source-checker", 1_700_000_001)).expect("check");
    match &check.verdict {
        Verdict::Verified => {
            let e = check.evidence.as_ref().unwrap();
            println!("   -> VERIFIED. all {} terms matched:", check.terms.len());
            println!("      terms: {}", check.terms.join(", "));
            println!("      evidence: {}", e.excerpt.as_deref().unwrap_or("(none)"));
        }
        other => panic!("expected verification, got {other:?}"),
    }
    assert_eq!(
        store.latest(&qo_knowledge::ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Verified
    );

    println!("\n=== 3. Context is now load-bearing for the next session ===");
    print!("{}", compile_context(&store, &ContextRequest::about(subject.clone())).unwrap().text);

    println!("=== 4. A claim the source does not substantiate stays put ===");
    let mut guess = qo_knowledge::Claim::proposed(
        "c2",
        "payments validates tokens",
        EntityId::derive(EntityKind::File, "src/payments.rs"),
        prov("worker-3", 1_700_000_000),
    );
    guess.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "src/payments.rs".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&guess).expect("add guess");
    let partial = verify_claim_against_source(&store, &qo_knowledge::ClaimId("c2".into()), root, prov("source-checker", 1_700_000_001)).unwrap();
    match &partial.verdict {
        Verdict::Inconclusive { reason } => println!("   -> INCONCLUSIVE: {reason}"),
        other => panic!("expected inconclusive, got {other:?}"),
    }
    println!("   claim c2 remains {}", store.latest(&qo_knowledge::ClaimId("c2".into())).unwrap().unwrap().status.as_str());

    println!("\n=== 5. A path that escapes the root is refused ===");
    let mut escape = qo_knowledge::Claim::proposed(
        "c3",
        "auth hashes passwords with bcrypt",
        EntityId::derive(EntityKind::File, "src/auth.rs"),
        prov("worker-3", 1_700_000_000),
    );
    escape.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "../../etc/passwd".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&escape).expect("add escape");
    let refused = verify_claim_against_source(&store, &qo_knowledge::ClaimId("c3".into()), root, prov("source-checker", 1)).unwrap();
    match &refused.verdict {
        Verdict::Unavailable { reason } => println!("   -> REFUSED: {reason}"),
        other => panic!("expected unavailable, got {other:?}"),
    }

    println!("\nThat is the loop: propose freely, let the graph check against");
    println!("source, and promote only what the code literally substantiates.");
}
