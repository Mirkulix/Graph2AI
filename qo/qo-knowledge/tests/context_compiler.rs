//! Context compiler tests.
//!
//! The rule under test: an unverified proposal must never reach a worker
//! looking like an established fact, and the output must stay inside its
//! budget while saying what it dropped.

use qo_knowledge::context::{compile_context, ContextRequest};
use qo_knowledge::model::{
    Claim, ClaimId, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance, Relation,
};
use qo_knowledge::KnowledgeStore;

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn prov(at: u64) -> Provenance {
    Provenance {
        producer: "indexer".into(),
        observed_at: at,
        git_revision: None,
        run_id: None,
    }
}

fn auth() -> EntityId {
    EntityId::derive(EntityKind::File, "src/auth.rs")
}

fn source_evidence(supports: bool) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: Some((42, 42)),
        excerpt: Some("use bcrypt::hash;".into()),
        supports,
    }
}

fn put_entity(store: &KnowledgeStore, kind: EntityKind, name: &str) -> EntityId {
    let id = EntityId::derive(kind, name);
    store
        .put_entity(&Entity {
            id: id.clone(),
            kind,
            name: name.into(),
        })
        .unwrap();
    id
}

/// The headline rule: a proposal is absent by default, and labelled when asked
/// for — never rendered as established.
#[test]
fn proposals_never_appear_as_established_facts() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed(
            "c1",
            "auth probably uses argon2",
            auth(),
            prov(100),
        ))
        .unwrap();

    let default = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert!(
        default.is_empty(),
        "a proposal leaked into default context: {}",
        default.text
    );
    assert!(!default.text.contains("argon2"));

    let asked = compile_context(
        &store,
        &ContextRequest::about(auth()).including_proposals(),
    )
    .unwrap();
    assert!(asked.text.contains("argon2"));
    assert!(
        asked.text.contains("Unverified proposals"),
        "proposal was not labelled: {}",
        asked.text
    );
    assert!(
        !asked.text.contains("## Established"),
        "there is nothing established to show"
    );
}

/// A verified claim is established, and carries its locator so a worker can
/// check it rather than trust it.
#[test]
fn verified_claims_are_established_and_carry_their_locator() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed("c1", "auth uses bcrypt", auth(), prov(100)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), source_evidence(true), prov(200))
        .unwrap();

    let ctx = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert_eq!(ctx.included, 1);
    assert!(ctx.text.contains("## Established"));
    assert!(ctx.text.contains("[verified]"));
    assert!(
        ctx.text.contains("[src/auth.rs:42]"),
        "locator missing: {}",
        ctx.text
    );
}

/// Established facts get the budget before proposals do.
#[test]
fn established_facts_outrank_proposals_under_pressure() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed("c1", "verified fact here", auth(), prov(100)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), source_evidence(true), prov(200))
        .unwrap();
    store
        .add_claim(&Claim::proposed("c2", "a mere guess", auth(), prov(300)))
        .unwrap();

    // Room for the established section and its one claim, but not for the
    // proposal heading and line that would follow.
    let ctx = compile_context(
        &store,
        &ContextRequest::about(auth())
            .including_proposals()
            .with_budget(110),
    )
    .unwrap();

    assert!(ctx.text.contains("verified fact here"));
    assert!(
        !ctx.text.contains("a mere guess"),
        "a proposal crowded out under budget pressure: {}",
        ctx.text
    );
}

/// The budget is a hard cap, and truncation is stated rather than silent.
#[test]
fn budget_is_respected_and_truncation_is_reported() {
    let (_dir, store) = store();
    for i in 0..20 {
        let id = format!("c{i}");
        store
            .add_claim(&Claim::proposed(
                id.clone(),
                format!("claim number {i} with some length to it"),
                auth(),
                prov(100 + i as u64),
            ))
            .unwrap();
        store
            .verify_claim(&ClaimId(id), source_evidence(true), prov(200))
            .unwrap();
    }

    let budget = 300;
    let ctx = compile_context(
        &store,
        &ContextRequest::about(auth()).with_budget(budget),
    )
    .unwrap();

    assert!(
        ctx.text.len() <= budget,
        "budget exceeded: {} > {budget}",
        ctx.text.len()
    );
    assert!(ctx.omitted > 0, "expected some claims to be dropped");
    assert!(
        ctx.text.contains("omitted for space"),
        "truncation was silent: {}",
        ctx.text
    );
    assert_eq!(ctx.included + ctx.omitted, 20);
}

/// Same graph, same request, same bytes — the context can be cached or diffed.
#[test]
fn compilation_is_deterministic() {
    let (_dir, store) = store();
    for i in 0..5 {
        let id = format!("c{i}");
        store
            .add_claim(&Claim::proposed(id.clone(), format!("fact {i}"), auth(), prov(100)))
            .unwrap();
        store
            .verify_claim(&ClaimId(id), source_evidence(true), prov(200))
            .unwrap();
    }

    let req = ContextRequest::about(auth());
    assert_eq!(
        compile_context(&store, &req).unwrap(),
        compile_context(&store, &req).unwrap()
    );
}

/// Depth 0 stays on the focus entity; depth 1 pulls in its neighbours.
#[test]
fn depth_bounds_the_walk() {
    let (_dir, store) = store();
    let cargo = put_entity(&store, EntityKind::File, "Cargo.toml");
    put_entity(&store, EntityKind::File, "src/auth.rs");

    // A relation from auth.rs to Cargo.toml.
    store
        .add_claim(&Claim::proposed("c1", "auth depends on the manifest", auth(), prov(100)))
        .unwrap();
    store
        .attach_relation(
            &ClaimId("c1".into()),
            Relation::DependsOn,
            cargo.clone(),
            prov(110),
        )
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), source_evidence(true), prov(120))
        .unwrap();

    // A fact that lives on the neighbour only.
    store
        .add_claim(&Claim::proposed("c2", "manifest pins redb 2", cargo, prov(130)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c2".into()), source_evidence(true), prov(140))
        .unwrap();

    let shallow = compile_context(&store, &ContextRequest::about(auth()).with_depth(0)).unwrap();
    assert!(!shallow.text.contains("redb 2"), "depth 0 walked too far");

    let deep = compile_context(&store, &ContextRequest::about(auth()).with_depth(1)).unwrap();
    assert!(
        deep.text.contains("redb 2"),
        "depth 1 did not reach the neighbour: {}",
        deep.text
    );
}

/// A refuted claim is settled, not pending — it belongs in neither section.
#[test]
fn refuted_claims_are_excluded_from_both_sections() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed("c1", "auth uses md5", auth(), prov(100)))
        .unwrap();
    store
        .refute_claim(&ClaimId("c1".into()), source_evidence(false), prov(200))
        .unwrap();

    let ctx = compile_context(
        &store,
        &ContextRequest::about(auth()).including_proposals(),
    )
    .unwrap();
    assert!(!ctx.text.contains("md5"), "refuted claim resurfaced: {}", ctx.text);
}

/// A superseded revision must not be rendered alongside its successor.
#[test]
fn superseded_revisions_are_not_rendered() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed("c1", "auth uses bcrypt", auth(), prov(100)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), source_evidence(true), prov(200))
        .unwrap();

    let ctx = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert_eq!(ctx.included, 1, "both revisions were rendered: {}", ctx.text);
    assert!(!ctx.text.contains("[proposed]"));
}

/// An empty graph produces empty context, not a misleading header.
#[test]
fn empty_graph_yields_empty_context() {
    let (_dir, store) = store();
    let ctx = compile_context(&store, &ContextRequest::about(auth())).unwrap();
    assert!(ctx.is_empty());
    assert!(ctx.text.is_empty());
}

/// A budget too small even for a header drops everything and says so, rather
/// than emitting a header with nothing under it.
#[test]
fn tiny_budget_does_not_emit_a_dangling_header() {
    let (_dir, store) = store();
    store
        .add_claim(&Claim::proposed("c1", "some fact", auth(), prov(100)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), source_evidence(true), prov(200))
        .unwrap();

    let ctx = compile_context(&store, &ContextRequest::about(auth()).with_budget(5)).unwrap();
    assert_eq!(ctx.included, 0);
    assert_eq!(ctx.omitted, 1);
    assert!(ctx.text.len() <= 5);
}
