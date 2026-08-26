//! The constrained text-to-graph proposal pipeline, printed step by step.
//!
//! `cargo run -p qo-knowledge --example extract_demo`
//!
//! Shows what an LLM actually exchanges with QO when it proposes knowledge:
//! a bounded system prompt in, model text out, the admission gate in between
//! — and why a stray "this is verified" line gets the whole document refused.
//!
//! The model call itself is the integration layer's job (this crate has no
//! LLM dependency); here the model's output is written literally.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{EntityId, EntityKind};
use qo_knowledge::{merge_delta, propose_from_text, proposal_system_prompt, KnowledgeStore};

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = KnowledgeStore::open(dir.path().join("k.redb")).expect("open store");
    let auth = EntityId::derive(EntityKind::File, "src/auth.rs");

    println!("=== 1. The system prompt a worker LLM is given ===");
    println!("{}", proposal_system_prompt());

    println!("=== 2. Context before the task ===");
    let before = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    println!(
        "{}",
        if before.is_empty() {
            "(the graph knows nothing about this file yet)".to_string()
        } else {
            before.text.clone()
        }
    );

    println!("\n=== 3. The model answers — and overreaches ===");
    let overreach = "\
DELTA|1|d-1
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
+R|c1|depends_on|file:Cargo.toml
";
    print!("{overreach}");

    let refused = propose_from_text(overreach, &qo_knowledge::ProposalPolicy::default());
    println!("\n-> refused. The admission gate reports:");
    for violation in &refused.violations {
        println!("   - {violation}");
    }
    println!("   The model may propose; it may not verify. Nothing was stored.");

    println!("\n=== 4. The worker resubmits without the OK line ===");
    let clean = "\
DELTA|1|d-1
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
";
    print!("{clean}");
    let outcome = propose_from_text(clean, &qo_knowledge::ProposalPolicy::default());
    assert!(outcome.is_ok(), "{:?}", outcome.violations);
    let report = merge_delta(&store, &outcome.delta.expect("admitted delta")).unwrap();
    println!(
        "\n-> admitted and merged: {} applied, {} conflict(s)",
        report.applied(),
        report.conflicts().len()
    );

    println!("\n=== 5. Context after the merge ===");
    let after = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    println!(
        "{}",
        if after.is_empty() {
            "(still empty — the proposal is not load-bearing until verified)".to_string()
        } else {
            after.text.clone()
        }
    );

    println!("\n=== 6. An authorised verifier promotes it ===");
    let verify = "\
DELTA|1|d-2
BY|reviewer-1|1700000100
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    print!("{verify}");
    assert!(merge_delta(&store, &qo_knowledge::from_orbitql(verify).unwrap())
        .unwrap()
        .is_clean());

    let verified = compile_context(&store, &ContextRequest::about(auth.clone())).unwrap();
    print!("\n{}", verified.text);
    println!("That is the loop: propose freely, verify separately, and only");
    println!("reproducible evidence turns a claim into context a peer can trust.");
}
