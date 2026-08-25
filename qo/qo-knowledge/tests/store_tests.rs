//! Behaviour tests for the knowledge store: provenance, revisions, and the
//! rule that an unbacked proposal never counts as truth.

use qo_knowledge::*;
use tempfile::TempDir;

fn store() -> (TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let s = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, s)
}

fn prov(producer: &str, at: u64) -> Provenance {
    Provenance {
        producer: producer.into(),
        observed_at: at,
        git_revision: Some("abc123".into()),
        run_id: Some("run-1".into()),
    }
}

fn support(loc: &str) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        locator: loc.into(),
        lines: Some((10, 20)),
        excerpt: None,
        supports: true,
    }
}

fn counter(loc: &str) -> Evidence {
    Evidence {
        kind: EvidenceKind::TestRun,
        locator: loc.into(),
        lines: None,
        excerpt: Some("assertion failed".into()),
        supports: false,
    }
}

fn file(name: &str) -> EntityId {
    EntityId::derive(EntityKind::File, name)
}

#[test]
fn proposal_is_stored_but_not_load_bearing() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(&Claim::proposed("c1", "a calls b", f.clone(), prov("llm", 1)))
        .unwrap();

    assert_eq!(s.claims_about(&f).unwrap().len(), 1, "claim is stored");
    assert!(
        s.load_bearing_context(&f, 10).unwrap().is_empty(),
        "but an unverified proposal must not be handed out as reliable"
    );
}

#[test]
fn verification_requires_evidence_and_promotes() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(&Claim::proposed("c1", "a calls b", f.clone(), prov("llm", 1)))
        .unwrap();

    let after = s
        .verify_claim(&ClaimId("c1".into()), support("a.rs"), prov("human", 2))
        .unwrap();

    assert_eq!(after.status, ClaimStatus::Verified);
    assert_eq!(after.revision, 2);
    assert_eq!(after.evidence.len(), 1);
    assert_eq!(s.load_bearing_context(&f, 10).unwrap().len(), 1);
}

#[test]
fn cannot_verify_with_counter_evidence() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm", 1)))
        .unwrap();

    let r = s.verify_claim(&ClaimId("c1".into()), counter("test"), prov("human", 2));
    assert!(matches!(r, Err(Error::CounterEvidenceForVerify)));
}

#[test]
fn cannot_refute_with_supporting_evidence() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm", 1)))
        .unwrap();

    let r = s.refute_claim(&ClaimId("c1".into()), support("a.rs"), prov("human", 2));
    assert!(matches!(r, Err(Error::SupportingEvidenceForRefute)));
}

#[test]
fn refutation_removes_claim_from_load_bearing_context() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(
        &Claim::observed(
            "c1",
            "a calls b",
            f.clone(),
            prov("indexer", 1),
            vec![support("a.rs")],
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(s.load_bearing_context(&f, 10).unwrap().len(), 1);

    s.refute_claim(&ClaimId("c1".into()), counter("cargo test"), prov("ci", 2))
        .unwrap();

    assert!(
        s.load_bearing_context(&f, 10).unwrap().is_empty(),
        "a refuted claim must stop being reliable context"
    );
}

#[test]
fn history_is_append_only_and_keeps_superseded_revisions() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm", 1)))
        .unwrap();
    s.verify_claim(&ClaimId("c1".into()), support("a.rs"), prov("human", 2))
        .unwrap();
    s.refute_claim(&ClaimId("c1".into()), counter("test"), prov("ci", 3))
        .unwrap();

    let h = s.history(&ClaimId("c1".into())).unwrap();
    assert_eq!(h.len(), 3, "every revision is kept");
    assert_eq!(h[0].status, ClaimStatus::Proposed);
    assert_eq!(h[1].status, ClaimStatus::Verified);
    assert_eq!(h[2].status, ClaimStatus::Refuted);

    assert_eq!(h[0].superseded_by, Some(2));
    assert_eq!(h[1].superseded_by, Some(3));
    assert_eq!(h[2].superseded_by, None, "the newest revision stands");
}

#[test]
fn contradicting_claims_coexist_rather_than_overwrite() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(
        &Claim::observed(
            "yes",
            "auth uses bcrypt",
            f.clone(),
            prov("indexer", 1),
            vec![support("a.rs")],
        )
        .unwrap(),
    )
    .unwrap();
    s.add_claim(
        &Claim::observed(
            "no",
            "auth uses argon2",
            f.clone(),
            prov("other", 2),
            vec![support("a.rs")],
        )
        .unwrap(),
    )
    .unwrap();

    let all = s.claims_about(&f).unwrap();
    assert_eq!(all.len(), 2, "both sides stay visible with their evidence");
}

#[test]
fn duplicate_claim_id_is_rejected() {
    let (_d, s) = store();
    let c = Claim::proposed("c1", "x", file("a.rs"), prov("llm", 1));
    s.add_claim(&c).unwrap();
    assert!(matches!(s.add_claim(&c), Err(Error::ClaimExists(_))));
}

#[test]
fn advancing_an_unknown_claim_fails() {
    let (_d, s) = store();
    let r = s.verify_claim(&ClaimId("ghost".into()), support("a"), prov("h", 1));
    assert!(matches!(r, Err(Error::NoSuchClaim(_))));
}

#[test]
fn reverse_index_answers_impact_questions() {
    let (_d, s) = store();
    let a = file("a.rs");
    let b = file("b.rs");
    s.add_claim(
        &Claim::observed(
            "c1",
            "a depends on b",
            a.clone(),
            prov("indexer", 1),
            vec![support("a.rs")],
        )
        .unwrap()
        .relating(Relation::DependsOn, b.clone()),
    )
    .unwrap();

    let impact = s.claims_referencing(&b).unwrap();
    assert_eq!(impact.len(), 1, "b must know that a depends on it");
    assert_eq!(impact[0].subject, a);
}

#[test]
fn neighbors_traverses_both_directions() {
    let (_d, s) = store();
    let a = file("a.rs");
    let b = file("b.rs");
    s.add_claim(
        &Claim::observed(
            "c1",
            "a calls b",
            a.clone(),
            prov("indexer", 1),
            vec![support("a.rs")],
        )
        .unwrap()
        .relating(Relation::Calls, b.clone()),
    )
    .unwrap();

    assert_eq!(s.neighbors(&a).unwrap().len(), 1);
    let from_b = s.neighbors(&b).unwrap();
    assert_eq!(from_b.len(), 1);
    assert_eq!(from_b[0].1, a, "traversing from b leads back to a");
}

#[test]
fn status_index_reflects_only_the_newest_revision() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm", 1)))
        .unwrap();
    assert_eq!(s.claims_with_status(ClaimStatus::Proposed).unwrap().len(), 1);

    s.verify_claim(&ClaimId("c1".into()), support("a.rs"), prov("h", 2))
        .unwrap();

    assert!(
        s.claims_with_status(ClaimStatus::Proposed).unwrap().is_empty(),
        "the claim left Proposed"
    );
    assert_eq!(s.claims_with_status(ClaimStatus::Verified).unwrap().len(), 1);
}

#[test]
fn stale_claims_are_not_load_bearing() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(
        &Claim::observed("c1", "x", f.clone(), prov("indexer", 1), vec![support("a.rs")]).unwrap(),
    )
    .unwrap();
    s.mark_stale(&ClaimId("c1".into()), prov("watcher", 2))
        .unwrap();

    assert!(s.load_bearing_context(&f, 10).unwrap().is_empty());
}

#[test]
fn verified_outranks_observed_in_context() {
    let (_d, s) = store();
    let f = file("a.rs");
    s.add_claim(
        &Claim::observed(
            "obs",
            "observed one",
            f.clone(),
            prov("indexer", 5),
            vec![support("a")],
        )
        .unwrap(),
    )
    .unwrap();
    s.add_claim(&Claim::proposed(
        "ver",
        "verified one",
        f.clone(),
        prov("llm", 1),
    ))
    .unwrap();
    s.verify_claim(&ClaimId("ver".into()), support("a"), prov("h", 2))
        .unwrap();

    let ctx = s.load_bearing_context(&f, 10).unwrap();
    assert_eq!(ctx.len(), 2);
    assert_eq!(ctx[0].status, ClaimStatus::Verified, "verified ranks first");
}

#[test]
fn provenance_survives_a_round_trip() {
    let (_d, s) = store();
    let c = Claim::proposed("c1", "x", file("a.rs"), prov("researcher", 42));
    s.add_claim(&c).unwrap();

    let back = s.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(back.provenance.producer, "researcher");
    assert_eq!(back.provenance.observed_at, 42);
    assert_eq!(back.provenance.git_revision.as_deref(), Some("abc123"));
    assert_eq!(back.provenance.run_id.as_deref(), Some("run-1"));
}

#[test]
fn search_finds_by_statement_and_skips_superseded() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed(
        "c1",
        "auth uses bcrypt",
        file("a.rs"),
        prov("llm", 1),
    ))
    .unwrap();
    s.add_claim(&Claim::proposed(
        "c2",
        "routing is keyword based",
        file("b.rs"),
        prov("llm", 1),
    ))
    .unwrap();

    let hits = s.search("BCRYPT", 10).unwrap();
    assert_eq!(hits.len(), 1, "search is case-insensitive");
    assert_eq!(hits[0].id, ClaimId("c1".into()));

    s.verify_claim(&ClaimId("c1".into()), support("a.rs"), prov("h", 2))
        .unwrap();
    let hits = s.search("bcrypt", 10).unwrap();
    assert_eq!(hits.len(), 1, "only the newest revision is returned");
    assert_eq!(hits[0].status, ClaimStatus::Verified);
}

#[test]
fn entities_persist_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("k.redb");
    let e = Entity {
        id: file("a.rs"),
        kind: EntityKind::File,
        name: "a.rs".into(),
    };
    {
        let s = KnowledgeStore::open(&path).unwrap();
        s.put_entity(&e).unwrap();
        s.add_claim(
            &Claim::observed("c1", "x", file("a.rs"), prov("i", 1), vec![support("a")]).unwrap(),
        )
        .unwrap();
    }
    let s = KnowledgeStore::open(&path).unwrap();
    assert_eq!(s.get_entity(&file("a.rs")).unwrap(), Some(e));
    assert_eq!(
        s.latest(&ClaimId("c1".into())).unwrap().unwrap().status,
        ClaimStatus::Observed
    );
}
