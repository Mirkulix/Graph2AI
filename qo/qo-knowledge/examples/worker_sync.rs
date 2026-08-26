//! The full worker sync flow, printed step by step.
//!
//! `cargo run -p qo-knowledge --example worker_sync`
//!
//! Shows what a coding-agent session actually exchanges with QO: bounded
//! context in, an OrbitQLang delta out, and a merge report back — including
//! what happens when a second session disagrees.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{EntityId, EntityKind};
use qo_knowledge::{from_orbitql, merge_delta, KnowledgeStore};

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = KnowledgeStore::open(dir.path().join("k.redb")).expect("open store");
    let auth = EntityId::derive(EntityKind::File, "src/auth.rs");

    println!("=== 1. Context before the task ===");
    let before = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    println!(
        "{}",
        if before.is_empty() {
            "(the graph knows nothing about this file yet)".to_string()
        } else {
            before.text.clone()
        }
    );

    println!("\n=== 2. Worker submits its findings ===");
    let document = "\
DELTA|1|d-1
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    print!("{document}");

    let delta = from_orbitql(document).expect("worker output must parse");
    let report = merge_delta(&store, &delta).expect("merge");
    println!(
        "\n-> {} applied, {} already present, {} conflict(s)",
        report.applied(),
        report.already_applied(),
        report.conflicts().len()
    );

    println!("\n=== 3. Context after the merge ===");
    let after = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    print!("{}", after.text);

    println!("\n=== 4. A second session disagrees ===");
    let dissent = "\
DELTA|1|d-2
BY|worker-9|1700000500
NO|c1|source|src/auth.rs|10:10|use md5::compute;
";
    print!("{dissent}");
    let report = merge_delta(&store, &from_orbitql(dissent).unwrap()).unwrap();
    for conflict in report.conflicts() {
        println!("\n-> conflict [{:?}]: {}", conflict.kind, conflict.detail);
    }

    println!("\n=== 5. Nothing was lost ===");
    for revision in store
        .history(&qo_knowledge::ClaimId("c1".into()))
        .unwrap()
    {
        println!(
            "  rev {} — {} by {} ({} evidence)",
            revision.revision,
            revision.status.as_str(),
            revision.provenance.producer,
            revision.evidence.len()
        );
    }
    println!("\nThe refutation was refused, not discarded: the graph keeps every");
    println!("revision, and the conflict names both sides for a human to settle.");
}
