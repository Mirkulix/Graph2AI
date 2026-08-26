//! The signed worker sync flow, and the attacks it turns away.
//!
//! `cargo run -p qo-knowledge --example signed_sync`
//!
//! Everything here runs against a real store — the rejections are actual
//! rejections, not narration.

use qo_knowledge::delta::{DeltaProducer, GraphDelta, GraphDeltaOp, GRAPH_DELTA_VERSION};
use qo_knowledge::merge::{merge_signed_delta, SubmitError};
use qo_knowledge::model::{Claim, EntityId, EntityKind, Provenance};
use qo_knowledge::trust::{public_key_hex, sign_delta, TrustStore, TrustedKey};
use qo_knowledge::{from_orbitql, to_orbitql, KnowledgeStore};

const NOW: u64 = 1_700_000_000;

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = KnowledgeStore::open(dir.path().join("k.redb")).expect("open store");

    // The operator has decided to trust worker-3's key. An attacker has a
    // perfectly valid keypair of their own — it is simply not in the store.
    let worker_seed = [7u8; 32];
    let attacker_seed = [9u8; 32];

    let mut trust = TrustStore::new();
    trust.trust(
        "worker-3",
        TrustedKey {
            key_id: "k1".into(),
            public_key_hex: public_key_hex(&worker_seed),
            active_from: 0,
            accept_until: None,
            revoked_at: None,
            comment: Some("worker-3 laptop".into()),
        },
    );

    println!("=== 1. worker-3 signs and submits ===");
    let mut delta = findings("d-1", "worker-3");
    sign_delta(&mut delta, "k1", &worker_seed).unwrap();
    let document = to_orbitql(&delta);
    print!("{document}");

    let report = merge_signed_delta(&store, &trust, &delta, NOW).expect("should be accepted");
    println!("\n-> accepted: {} operations applied\n", report.applied());

    println!("=== 2. an attacker signs the same claim with their own key ===");
    let mut forged = findings("d-evil", "worker-3");
    sign_delta(&mut forged, "k1", &attacker_seed).unwrap();
    show(merge_signed_delta(&store, &trust, &forged, NOW));

    println!("=== 3. the document is edited after signing ===");
    let tampered = document.replace("bcrypt", "md5");
    let parsed = from_orbitql(&tampered).unwrap();
    show(merge_signed_delta(&store, &trust, &parsed, NOW));

    println!("=== 4. the original submission is replayed ===");
    show(merge_signed_delta(&store, &trust, &delta, NOW));

    println!("=== 5. worker-3's key is revoked, then they try again ===");
    let mut revoked = TrustStore::new();
    revoked.trust(
        "worker-3",
        TrustedKey {
            key_id: "k1".into(),
            public_key_hex: public_key_hex(&worker_seed),
            active_from: 0,
            accept_until: None,
            revoked_at: Some(NOW),
            comment: None,
        },
    );
    let mut later = findings("d-2", "worker-3");
    sign_delta(&mut later, "k1", &worker_seed).unwrap();
    show(merge_signed_delta(&store, &revoked, &later, NOW + 1));

    println!("=== What the graph actually holds ===");
    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    for claim in store.claims_about(&subject).unwrap() {
        println!(
            "  [{}] {}  (by {})",
            claim.status.as_str(),
            claim.statement,
            claim.provenance.producer
        );
    }

    // A signature proves who wrote this. It does not make the statement true —
    // that still needs evidence, and this claim has none.
    let load_bearing = store.load_bearing_context(&subject, 10).unwrap();
    println!(
        "\nload-bearing: {} — a signed proposal is still only a proposal.",
        load_bearing.len()
    );
    println!("Only the one legitimate write survived. Nothing else touched the graph.");
}

fn show(result: Result<qo_knowledge::MergeReport, SubmitError>) {
    match result {
        Ok(report) => println!("-> ACCEPTED ({} applied) — this should not happen\n", report.applied()),
        Err(e) => println!("-> refused: {e}\n"),
    }
}

fn findings(id: &str, producer: &str) -> GraphDelta {
    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: id.into(),
        producer: DeltaProducer {
            id: producer.into(),
            source_revision: Some("abc123".into()),
            run_id: None,
            emitted_at: NOW,
            signature: None,
        },
        operations: vec![GraphDeltaOp::AddClaim {
            claim: Claim::proposed(
                "c1",
                "auth hashes passwords with bcrypt",
                subject,
                Provenance {
                    producer: producer.into(),
                    observed_at: NOW,
                    git_revision: Some("abc123".into()),
                    run_id: None,
                },
            ),
        }],
    }
}
