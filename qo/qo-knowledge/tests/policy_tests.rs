//! Tests for the promise the crate makes to its callers: an LLM may propose,
//! but it cannot make something true by asserting it.
//!
//! These are written as adversarial cases — "what would a model do if it
//! wanted its guess treated as fact?" — because that is the failure mode the
//! design exists to prevent.

use qo_knowledge::*;

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let s = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, s)
}

fn prov(who: &str) -> Provenance {
    Provenance {
        producer: who.into(),
        observed_at: 100,
        git_revision: None,
        run_id: None,
    }
}

fn file(n: &str) -> EntityId {
    EntityId::derive(EntityKind::File, n)
}

fn support() -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        locator: "src/a.rs".into(),
        lines: Some((1, 2)),
        excerpt: None,
        supports: true,
    }
}

/// The headline rule. A proposal is stored, but it is not context.
#[test]
fn an_llm_guess_never_reaches_load_bearing_context_on_its_own() {
    let (_d, s) = store();
    let f = file("a.rs");

    for i in 0..10 {
        s.add_claim(&Claim::proposed(
            format!("guess-{i}"),
            "the auth module is thread-safe",
            f.clone(),
            prov("llm"),
        ))
        .unwrap();
    }

    assert_eq!(s.claims_about(&f).unwrap().len(), 10);
    assert!(
        s.load_bearing_context(&f, 100).unwrap().is_empty(),
        "ten confident guesses are still zero evidence"
    );
}

/// Repeating a claim does not make it truer. Same id is rejected outright;
/// different ids simply produce more proposals.
#[test]
fn repetition_does_not_promote_a_claim() {
    let (_d, s) = store();
    let f = file("a.rs");
    let c = Claim::proposed("c1", "x is true", f.clone(), prov("llm"));

    s.add_claim(&c).unwrap();
    assert!(matches!(s.add_claim(&c), Err(Error::ClaimExists(_))));

    s.add_claim(&Claim::proposed("c2", "x is true", f.clone(), prov("llm")))
        .unwrap();
    s.add_claim(&Claim::proposed("c3", "x is true", f.clone(), prov("llm")))
        .unwrap();

    assert!(s.load_bearing_context(&f, 100).unwrap().is_empty());
}

/// A caller cannot launder a refutation into a confirmation by mislabelling
/// the evidence direction.
#[test]
fn evidence_direction_cannot_be_laundered() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm")))
        .unwrap();

    let counter = Evidence {
        kind: EvidenceKind::TestRun,
        locator: "cargo test".into(),
        lines: None,
        excerpt: Some("FAILED".into()),
        supports: false,
    };

    assert!(matches!(
        s.verify_claim(&ClaimId("c1".into()), counter, prov("llm")),
        Err(Error::CounterEvidenceForVerify)
    ));

    // And the claim is untouched.
    let c = s.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(c.status, ClaimStatus::Proposed);
    assert_eq!(c.revision, 1);
}

/// Verification is recorded with *who* did it, so a later reader can weigh it.
#[test]
fn verification_records_the_verifier_not_the_proposer() {
    let (_d, s) = store();
    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm-guess")))
        .unwrap();
    s.verify_claim(&ClaimId("c1".into()), support(), prov("ci-runner"))
        .unwrap();

    let h = s.history(&ClaimId("c1".into())).unwrap();
    assert_eq!(h[0].provenance.producer, "llm-guess", "who proposed it");
    assert_eq!(h[1].provenance.producer, "ci-runner", "who confirmed it");
}

/// The audit trail survives a full lifecycle and stays readable.
#[test]
fn full_lifecycle_leaves_a_complete_audit_trail() {
    let (_d, s) = store();
    let id = ClaimId("c1".into());

    s.add_claim(&Claim::proposed("c1", "x", file("a.rs"), prov("llm")))
        .unwrap();
    s.verify_claim(&id, support(), prov("human")).unwrap();
    s.mark_stale(&id, prov("watcher")).unwrap();
    s.refute_claim(
        &id,
        Evidence {
            kind: EvidenceKind::Commit,
            locator: "deadbeef".into(),
            lines: None,
            excerpt: None,
            supports: false,
        },
        prov("reviewer"),
    )
    .unwrap();

    let h = s.history(&id).unwrap();
    assert_eq!(h.len(), 4);
    let statuses: Vec<_> = h.iter().map(|c| c.status).collect();
    assert_eq!(
        statuses,
        vec![
            ClaimStatus::Proposed,
            ClaimStatus::Verified,
            ClaimStatus::Stale,
            ClaimStatus::Refuted
        ]
    );
    // Every revision but the last is superseded.
    for (i, c) in h.iter().enumerate() {
        if i + 1 < h.len() {
            assert_eq!(c.superseded_by, Some(c.revision + 1));
        } else {
            assert_eq!(c.superseded_by, None);
        }
    }
    // Evidence accumulates rather than being replaced.
    assert_eq!(h[3].evidence.len(), 2);
}

/// Two agents disagreeing produce two visible claims, not one winner.
#[test]
fn disagreement_between_agents_stays_visible() {
    let (_d, s) = store();
    let f = file("router.rs");

    s.add_claim(&Claim::proposed(
        "a",
        "routing is role-based via llm_routing.toml",
        f.clone(),
        prov("agent-a"),
    ))
    .unwrap();
    s.add_claim(&Claim::proposed(
        "b",
        "routing is a hardcoded table in agent_models.rs",
        f.clone(),
        prov("agent-b"),
    ))
    .unwrap();

    // Only the one someone actually checked becomes load-bearing.
    s.verify_claim(
        &ClaimId("b".into()),
        Evidence {
            kind: EvidenceKind::Source,
            locator: "qo/qo-server/src/agent_models.rs".into(),
            lines: Some((31, 41)),
            excerpt: Some("match agent { \"developer\" => ...".into()),
            supports: true,
        },
        prov("human"),
    )
    .unwrap();

    let ctx = s.load_bearing_context(&f, 10).unwrap();
    assert_eq!(ctx.len(), 1);
    assert_eq!(ctx[0].id, ClaimId("b".into()));

    // But the losing claim is still on record, not deleted.
    assert_eq!(s.claims_about(&f).unwrap().len(), 2);
    assert_eq!(
        s.latest(&ClaimId("a".into())).unwrap().unwrap().status,
        ClaimStatus::Proposed
    );
}

/// A claim's subject entity does not need to exist first — the graph records
/// what it is told about, and entities are registered lazily.
#[test]
fn claims_about_unknown_entities_are_allowed() {
    let (_d, s) = store();
    let ghost = EntityId::derive(EntityKind::Symbol, "never_registered");
    assert!(s.get_entity(&ghost).unwrap().is_none());

    s.add_claim(&Claim::proposed("c1", "x", ghost.clone(), prov("llm")))
        .unwrap();
    assert_eq!(s.claims_about(&ghost).unwrap().len(), 1);
}

/// Limit is honoured, so a caller cannot be flooded by one entity's history.
#[test]
fn context_respects_its_limit() {
    let (_d, s) = store();
    let f = file("busy.rs");
    for i in 0..50 {
        s.add_claim(
            &Claim::observed(
                format!("c{i}"),
                format!("fact {i}"),
                f.clone(),
                prov("indexer"),
                vec![support()],
            )
            .unwrap(),
        )
        .unwrap();
    }
    assert_eq!(s.load_bearing_context(&f, 5).unwrap().len(), 5);
    assert_eq!(s.load_bearing_context(&f, 1000).unwrap().len(), 50);
}
