//! Integration tests for deterministic source-evidence verification.
//!
//! These exercise [`qo_knowledge::verify_claim_against_source`] against real
//! fixture files on disk: the promote path, the conservative "inconclusive"
//! path, the path-safety boundary, and the re-run safety.

use qo_knowledge::model::{ClaimId, ClaimStatus, EntityId, EntityKind, Evidence, EvidenceKind, Provenance};
use qo_knowledge::{verify_claim_against_source, KnowledgeStore, Verdict};

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn prov(producer: &str, at: u64) -> Provenance {
    Provenance {
        producer: producer.into(),
        observed_at: at,
        git_revision: None,
        run_id: None,
    }
}

fn fixture(root: &std::path::Path) {
    // Note the claim's four distinctive terms appear verbatim in the comment.
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/auth.rs"),
        "// auth hashes passwords with bcrypt\npub fn hash_password(pw: &str) -> String {\n    bcrypt::hash(pw)\n}\n",
    )
    .unwrap();
}

/// A proposal whose every distinctive term appears in the source is promoted
/// to `Verified`, with the exact matching line captured as evidence.
#[test]
fn a_fully_substantiated_claim_is_verified() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    let mut claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        subject,
        prov("worker-3", 1_700_000_000),
    );
    claim.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&claim).unwrap();

    let check = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1_700_000_001)).unwrap();

    assert_eq!(check.verdict, Verdict::Verified, "{:?}", check);
    let evidence = check.evidence.as_ref().unwrap();
    assert!(evidence.supports);
    assert!(evidence.excerpt.as_ref().unwrap().contains("bcrypt"));

    let promoted = store.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(promoted.status, ClaimStatus::Verified);
    // The promotion is a new revision carrying the captured evidence.
    assert!(promoted.evidence.iter().any(|e| e.kind == EvidenceKind::Source));
}

/// A partial match never promotes: the graph stays silent rather than
/// laundering a guess into fact.
#[test]
fn a_partial_match_stays_inconclusive() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    let mut claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth validates tokens",
        subject,
        prov("worker-3", 1_700_000_000),
    );
    claim.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&claim).unwrap();

    let check = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1_700_000_001)).unwrap();

    match &check.verdict {
        Verdict::Inconclusive { reason } => assert!(reason.contains("of 3"), "{reason}"),
        other => panic!("expected inconclusive, got {other:?}"),
    }
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Proposed);
}

/// A claim whose subject is a file is checked against that file even without
/// an explicit evidence locator.
#[test]
fn the_file_subject_is_used_when_there_is_no_locator() {
    let (dir, store) = store();
    fixture(dir.path());

    let claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        EntityId::derive(EntityKind::File, "src/auth.rs"),
        prov("worker-3", 1_700_000_000),
    );
    store.add_claim(&claim).unwrap();

    let check = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1)).unwrap();
    assert_eq!(check.verdict, Verdict::Verified, "{:?}", check);
}

/// A path that escapes the workspace root is refused and nothing changes.
#[test]
fn a_traversal_escape_is_refused() {
    let (dir, store) = store();
    fixture(dir.path());

    // A secret planted *outside* the root.
    let secret = dir.path().parent().unwrap().join("secret.txt");
    std::fs::write(&secret, "auth hashes passwords with bcrypt").unwrap();

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    let mut claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        subject,
        prov("worker-3", 1_700_000_000),
    );
    claim.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "../secret.txt".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&claim).unwrap();

    let check = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1)).unwrap();
    match &check.verdict {
        Verdict::Unavailable { reason } => assert!(reason.contains("escapes"), "{reason}"),
        other => panic!("expected unavailable, got {other:?}"),
    }
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Proposed);
}

/// A missing source is reported, and the claim is left alone.
#[test]
fn a_missing_source_is_unavailable() {
    let (dir, store) = store();

    let mut claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        EntityId::derive(EntityKind::File, "src/auth.rs"),
        prov("worker-3", 1_700_000_000),
    );
    claim.evidence.push(Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: None,
        excerpt: None,
        supports: true,
    });
    store.add_claim(&claim).unwrap();

    let check = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1)).unwrap();
    assert!(matches!(check.verdict, Verdict::Unavailable { .. }), "{:?}", check);
}

/// A settled claim is not re-promoted; the checker refuses to touch it.
#[test]
fn a_settled_claim_is_not_repromoted() {
    let (dir, store) = store();
    fixture(dir.path());

    let claim = qo_knowledge::Claim::proposed(
        "c1",
        "auth hashes passwords with bcrypt",
        EntityId::derive(EntityKind::File, "src/auth.rs"),
        prov("worker-3", 1_700_000_000),
    );
    store.add_claim(&claim).unwrap();

    let first = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1)).unwrap();
    assert_eq!(first.verdict, Verdict::Verified);

    let second = verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 2)).unwrap();
    match &second.verdict {
        Verdict::NotProposed { status } => assert_eq!(*status, ClaimStatus::Verified),
        other => panic!("expected not-proposed, got {other:?}"),
    }
    // No duplicate revision was appended.
    assert_eq!(store.history(&ClaimId("c1".into())).unwrap().len(), 2);
}

/// The sweep harvests every open proposal in one pass: the substantiated one
/// is promoted, the partial one stays, and the one with no source is reported.
#[test]
fn a_sweep_promotes_what_the_source_substantiates() {
    let (dir, store) = store();
    fixture(dir.path());

    for (id, statement, file) in [
        ("c1", "auth hashes passwords with bcrypt", "src/auth.rs"),
        ("c2", "auth validates tokens", "src/auth.rs"),
        ("c3", "auth hashes passwords with bcrypt", "src/missing.rs"),
    ] {
        store
            .add_claim(&qo_knowledge::Claim::proposed(
                id,
                statement,
                EntityId::derive(EntityKind::File, file),
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();
    }

    let report = qo_knowledge::verify_all_proposals(&store, dir.path(), prov("sweeper", 1_700_000_001)).unwrap();

    assert_eq!(report.checked, 3);
    assert_eq!(report.verified, 1, "{report:?}");
    assert_eq!(report.inconclusive, 1, "{report:?}");
    assert_eq!(report.unavailable, 1, "{report:?}");
    assert!(!report.fully_verified());

    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Verified);
    assert_eq!(store.latest(&ClaimId("c2".into())).unwrap().unwrap().status, ClaimStatus::Proposed);
    assert_eq!(store.latest(&ClaimId("c3".into())).unwrap().unwrap().status, ClaimStatus::Proposed);
}

/// The graph notices when the code moves on: a claim whose recorded excerpt
/// is no longer in its source is marked stale, deterministically.
#[test]
fn a_source_refresh_marks_stale_when_the_code_moves_on() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    store
        .add_claim(&qo_knowledge::Claim::proposed(
            "c1",
            "auth hashes passwords with bcrypt",
            subject,
            prov("worker-3", 1_700_000_000),
        ))
        .unwrap();
    assert_eq!(
        verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1))
            .unwrap()
            .verdict,
        Verdict::Verified
    );

    // Unchanged → still current.
    let report = qo_knowledge::refresh_sources(&store, dir.path(), prov("refresher", 2)).unwrap();
    assert_eq!(report.still_current, 1, "{report:?}");
    assert_eq!(report.stale, 0, "{report:?}");
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Verified);

    // The code moves on: the excerpt line is gone.
    std::fs::write(dir.path().join("src/auth.rs"), "fn probe() { /* no bcrypt here */ }\n").unwrap();
    let report = qo_knowledge::refresh_sources(&store, dir.path(), prov("refresher", 3)).unwrap();
    assert_eq!(report.stale, 1, "{report:?}");
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Stale);
}

/// A settled claim without a verbatim excerpt is skipped, not guessed at.
#[test]
fn a_settled_claim_without_an_excerpt_is_skipped() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    store
        .add_claim(&qo_knowledge::Claim::proposed(
            "c1",
            "auth uses bcrypt",
            subject,
            prov("worker-3", 1),
        ))
        .unwrap();
    store
        .verify_claim(
            &ClaimId("c1".into()),
            Evidence {
                kind: EvidenceKind::Source,
                locator: "src/auth.rs".into(),
                lines: None,
                excerpt: None,
                supports: true,
            },
            prov("reviewer", 2),
        )
        .unwrap();

    let report = qo_knowledge::refresh_sources(&store, dir.path(), prov("refresher", 3)).unwrap();
    assert_eq!(report.skipped, 1, "{report:?}");
    assert_eq!(report.stale, 0, "{report:?}");
    // Left untouched.
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Verified);
}

/// The full rot-then-heal cycle: the source moved, so the fact is marked
/// stale; but the fact still holds elsewhere, so the graph heals it back to
/// verified with fresh evidence — keeping the whole trail.
#[test]
fn a_stale_claim_is_healed_when_its_fact_still_holds() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    store
        .add_claim(&qo_knowledge::Claim::proposed(
            "c1",
            "auth hashes passwords with bcrypt",
            subject,
            prov("worker-3", 1_700_000_000),
        ))
        .unwrap();
    assert_eq!(
        verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1))
            .unwrap()
            .verdict,
        Verdict::Verified
    );

    // The line moved (comment style changed), but the terms are all still there.
    std::fs::write(
        dir.path().join("src/auth.rs"),
        "fn probe() { /* auth hashes passwords with bcrypt (relocated) */ }\n",
    )
    .unwrap();

    // Refresh: the old excerpt is gone -> stale.
    let refresh = qo_knowledge::refresh_sources(&store, dir.path(), prov("refresher", 2)).unwrap();
    assert_eq!(refresh.stale, 1, "{refresh:?}");
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Stale);

    // Heal: the fact still holds -> re-verified with fresh evidence.
    let heal = qo_knowledge::heal_stale(&store, dir.path(), prov("healer", 3)).unwrap();
    assert_eq!(heal.healed, 1, "{heal:?}");
    assert_eq!(heal.remained_stale, 0, "{heal:?}");
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Verified);

    // The full trail is kept: proposed -> verified -> stale -> verified.
    assert_eq!(store.history(&ClaimId("c1".into())).unwrap().len(), 4);
}

/// A genuinely rotted fact stays stale — the graph does not heal what the
/// source no longer substantiates.
#[test]
fn a_rotted_fact_stays_stale() {
    let (dir, store) = store();
    fixture(dir.path());

    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    store
        .add_claim(&qo_knowledge::Claim::proposed(
            "c1",
            "auth hashes passwords with bcrypt",
            subject,
            prov("worker-3", 1_700_000_000),
        ))
        .unwrap();
    verify_claim_against_source(&store, &ClaimId("c1".into()), dir.path(), prov("checker", 1)).unwrap();

    std::fs::write(dir.path().join("src/auth.rs"), "fn probe() { /* nothing relevant */ }\n").unwrap();
    assert_eq!(
        qo_knowledge::refresh_sources(&store, dir.path(), prov("refresher", 2))
            .unwrap()
            .stale,
        1
    );

    let heal = qo_knowledge::heal_stale(&store, dir.path(), prov("healer", 3)).unwrap();
    assert_eq!(heal.healed, 0, "{heal:?}");
    assert_eq!(heal.remained_stale, 1, "{heal:?}");
    assert_eq!(store.latest(&ClaimId("c1".into())).unwrap().unwrap().status, ClaimStatus::Stale);
}
