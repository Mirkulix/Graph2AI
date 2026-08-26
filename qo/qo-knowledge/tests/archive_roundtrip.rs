//! Archive tests.
//!
//! The property that matters: an export/import cycle must preserve the *audit
//! trail*, not just the current state. Superseded revisions, refutations and
//! their counter-evidence are the reason this crate exists — an archive that
//! quietly flattened them would be worse than no archive at all.

use qo_knowledge::archive::{export, import, Archive, ARCHIVE_VERSION};
use qo_knowledge::model::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance,
};
use qo_knowledge::KnowledgeStore;

fn store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

fn prov(at: u64) -> Provenance {
    Provenance {
        producer: "worker-1".into(),
        observed_at: at,
        git_revision: Some("abc123".into()),
        run_id: Some("run-1".into()),
    }
}

fn auth() -> EntityId {
    EntityId::derive(EntityKind::File, "src/auth.rs")
}

fn evidence(supports: bool) -> Evidence {
    Evidence {
        kind: EvidenceKind::Source,
        locator: "src/auth.rs".into(),
        lines: Some((42, 42)),
        excerpt: Some("use bcrypt::hash;".into()),
        supports,
    }
}

/// Build a store with a non-trivial history: a verified claim (3 revisions)
/// and a refuted one (2 revisions).
fn populated() -> (tempfile::TempDir, KnowledgeStore) {
    let (dir, store) = store();
    store
        .put_entity(&Entity {
            id: auth(),
            kind: EntityKind::File,
            name: "src/auth.rs".into(),
        })
        .unwrap();

    store
        .add_claim(&Claim::proposed("c1", "auth uses bcrypt", auth(), prov(100)))
        .unwrap();
    store
        .verify_claim(&ClaimId("c1".into()), evidence(true), prov(200))
        .unwrap();

    store
        .add_claim(&Claim::proposed("c2", "auth uses md5", auth(), prov(300)))
        .unwrap();
    store
        .refute_claim(&ClaimId("c2".into()), evidence(false), prov(400))
        .unwrap();

    (dir, store)
}

/// The headline guarantee: every revision survives, not just the newest.
#[test]
fn export_import_preserves_the_full_history() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    let (_dst_dir, target) = store();
    let report = import(&target, &archive).unwrap();
    assert!(report.is_complete(), "skipped: {:?}", report.claims_skipped);

    for id in ["c1", "c2"] {
        let before = source.history(&ClaimId(id.into())).unwrap();
        let after = target.history(&ClaimId(id.into())).unwrap();
        assert_eq!(after, before, "history for {id} changed across the archive");
    }
}

/// A refuted claim keeps its counter-evidence — the losing side of a
/// disagreement must remain readable after a restore.
#[test]
fn counter_evidence_survives_the_round_trip() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    let (_dst_dir, target) = store();
    import(&target, &archive).unwrap();

    let refuted = target.latest(&ClaimId("c2".into())).unwrap().unwrap();
    assert_eq!(refuted.status, ClaimStatus::Refuted);
    let counter = refuted
        .evidence
        .iter()
        .find(|e| !e.supports)
        .expect("counter-evidence was dropped");
    assert_eq!(counter.locator, "src/auth.rs");
}

/// Provenance must not be rewritten by the restore. If import re-derived a
/// verification, it would stamp the importer as the verifier — a forgery of
/// who decided what.
#[test]
fn restore_does_not_rewrite_provenance() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    let (_dst_dir, target) = store();
    import(&target, &archive).unwrap();

    let restored = target.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(restored.provenance.producer, "worker-1");
    assert_eq!(restored.provenance.observed_at, 200);
    assert_eq!(restored.provenance.git_revision.as_deref(), Some("abc123"));
    // proposed (rev 1) -> verified (rev 2). A restore that re-derived the
    // verification would land on a different number.
    assert_eq!(restored.revision, 2, "revision numbers were renumbered");
}

/// Restored verified claims are load-bearing again — the graph is usable
/// immediately after a restore, not degraded to proposals.
#[test]
fn restored_graph_is_immediately_usable() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    let (_dst_dir, target) = store();
    import(&target, &archive).unwrap();

    let context = target.load_bearing_context(&auth(), 10).unwrap();
    assert_eq!(context.len(), 1, "expected exactly the verified claim");
    assert_eq!(context[0].id, ClaimId("c1".into()));
}

/// Import never overwrites. An id already present is reported, not merged —
/// reconciling divergent histories is the merger's job, not the archive's.
#[test]
fn import_into_a_populated_store_skips_rather_than_overwrites() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    // Target already holds a *different* claim under the same id.
    let (_dst_dir, target) = store();
    target
        .add_claim(&Claim::proposed("c1", "something else entirely", auth(), prov(50)))
        .unwrap();

    let report = import(&target, &archive).unwrap();
    assert!(!report.is_complete());
    assert!(report.claims_skipped.contains(&"c1".to_string()));

    let kept = target.latest(&ClaimId("c1".into())).unwrap().unwrap();
    assert_eq!(kept.statement, "something else entirely");
    assert_eq!(kept.revision, 1, "the local claim was modified");

    // The claim that did not collide still landed.
    assert!(target.latest(&ClaimId("c2".into())).unwrap().is_some());
}

/// Entities are derived from (kind, name), so re-importing them is a no-op.
#[test]
fn importing_twice_is_idempotent_for_entities() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();

    let (_dst_dir, target) = store();
    let first = import(&target, &archive).unwrap();
    let second = import(&target, &archive).unwrap();

    assert_eq!(first.entities_added, 1);
    assert_eq!(second.entities_added, 0);
    assert_eq!(target.list_entities().unwrap().len(), 1);
}

/// JSON is the transport, so it has to survive it.
#[test]
fn archive_survives_json() {
    let (_src_dir, source) = populated();
    let archive = export(&source, 999).unwrap();
    let parsed = Archive::from_json(&archive.to_json().unwrap()).unwrap();
    assert_eq!(parsed, archive);
}

/// An archive from a future version is refused rather than half-read.
#[test]
fn unsupported_archive_version_is_rejected() {
    let json = format!(
        r#"{{"version":{},"exported_at":1,"entities":[],"claims":[]}}"#,
        ARCHIVE_VERSION + 1
    );
    let error = Archive::from_json(&json).unwrap_err();
    assert!(
        error.to_string().contains("unsupported archive version"),
        "unhelpful error: {error}"
    );
}

/// An empty graph exports and restores without special-casing.
#[test]
fn empty_graph_round_trips() {
    let (_src_dir, source) = store();
    let archive = export(&source, 999).unwrap();
    assert!(archive.entities.is_empty());
    assert!(archive.claims.is_empty());

    let (_dst_dir, target) = store();
    let report = import(&target, &archive).unwrap();
    assert!(report.is_complete());
    assert_eq!(report.claims_added, 0);
}
