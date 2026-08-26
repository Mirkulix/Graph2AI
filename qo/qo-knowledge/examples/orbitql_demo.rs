//! Prints a delta in both encodings, so the size claim can be re-checked by
//! hand rather than taken on trust.
//!
//! `cargo run -p qo-knowledge --example orbitql_demo`

use qo_knowledge::delta::{DeltaProducer, GraphDelta, GraphDeltaOp, GRAPH_DELTA_VERSION};
use qo_knowledge::model::{
    Claim, ClaimId, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance, Relation,
};
use qo_knowledge::{from_orbitql, to_orbitql};

fn main() {
    let producer = DeltaProducer {
        id: "worker-3".into(),
        source_revision: Some("abc123".into()),
        run_id: Some("run-7".into()),
        emitted_at: 1_700_000_000,
        signature: None,
    };
    let provenance = Provenance {
        producer: producer.id.clone(),
        observed_at: producer.emitted_at,
        git_revision: producer.source_revision.clone(),
        run_id: producer.run_id.clone(),
    };
    let auth = EntityId::derive(EntityKind::File, "src/auth.rs");

    let delta = GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: "d-42".into(),
        producer,
        operations: vec![
            GraphDeltaOp::AddEntity {
                entity: Entity {
                    id: auth.clone(),
                    kind: EntityKind::File,
                    name: "src/auth.rs".into(),
                },
            },
            GraphDeltaOp::AddClaim {
                claim: Claim::proposed("c1", "auth uses bcrypt", auth.clone(), provenance),
            },
            GraphDeltaOp::AddRelation {
                claim_id: ClaimId("c1".into()),
                relation: Relation::DependsOn,
                object: EntityId::derive(EntityKind::File, "Cargo.toml"),
            },
            GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("c1".into()),
                evidence: Evidence {
                    kind: EvidenceKind::Source,
                    locator: "src/auth.rs".into(),
                    lines: Some((42, 42)),
                    excerpt: Some("use bcrypt::hash;".into()),
                    supports: true,
                },
            },
        ],
    };

    let text = to_orbitql(&delta);
    let json = delta.to_canonical_json().unwrap();

    println!("--- OrbitQLang ({} bytes) ---\n{text}", text.len());
    println!("--- canonical JSON ({} bytes) ---\n{json}\n", json.len());
    println!("ratio: {:.1}x smaller", json.len() as f64 / text.len() as f64);

    assert_eq!(from_orbitql(&text).unwrap(), delta, "round-trip must hold");
    println!("round-trip: ok");
}
