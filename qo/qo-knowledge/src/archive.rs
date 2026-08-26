//! Export and restore the knowledge graph as a portable JSON archive.
//!
//! The graph lives in one redb file. That is fine until it is not: a corrupted
//! database, a machine change, or a need to inspect the graph somewhere that
//! cannot open redb. This module is the escape hatch.
//!
//! ## What an archive contains
//!
//! **Every revision, not just the current one.** An export that kept only the
//! latest state would silently discard the audit trail — the superseded
//! revisions, the refuted claims and their counter-evidence. That trail is the
//! reason this crate exists, so dropping it on export would be a quiet
//! betrayal of the whole design.
//!
//! ## Import is additive
//!
//! [`import`] never deletes and never overwrites. A claim id already present
//! in the target store is reported as skipped, not merged — reconciling two
//! divergent histories is a merge decision, and [`crate::merge`] is where
//! merge decisions live. Import exists to rebuild an empty store, or to pull
//! in a graph the target has not seen.

use crate::model::{Claim, Entity};
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Bumped when the archive layout changes incompatibly.
pub const ARCHIVE_VERSION: u16 = 1;

/// A portable snapshot of the whole graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Archive {
    pub version: u16,
    /// Unix seconds, supplied by the caller so exports stay deterministic and
    /// testable — the store never reads the clock.
    pub exported_at: u64,
    pub entities: Vec<Entity>,
    /// Every revision of every claim, oldest first within each claim.
    pub claims: Vec<Claim>,
}

impl Archive {
    pub fn to_json(&self) -> Result<String, Error> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn from_json(input: &str) -> Result<Self, Error> {
        let archive: Archive = serde_json::from_str(input)?;
        if archive.version != ARCHIVE_VERSION {
            return Err(Error::Storage(format!(
                "unsupported archive version {} (expected {ARCHIVE_VERSION})",
                archive.version
            )));
        }
        Ok(archive)
    }
}

/// What an import actually did. Nothing was overwritten either way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportReport {
    pub entities_added: usize,
    pub claims_added: usize,
    /// Claim ids already present in the target, left untouched.
    pub claims_skipped: Vec<String>,
}

impl ImportReport {
    /// True when every claim in the archive landed. A false here means the
    /// target already held some of these ids — look at `claims_skipped`.
    pub fn is_complete(&self) -> bool {
        self.claims_skipped.is_empty()
    }
}

/// Read the entire graph out, every revision included.
pub fn export(store: &KnowledgeStore, exported_at: u64) -> Result<Archive, Error> {
    let entities = store.list_entities()?;

    // Walk every claim id we can see, then take its full history. Collecting
    // ids first keeps this independent of how the status index is laid out.
    let mut ids: Vec<_> = crate::model::ClaimStatus::ALL
        .iter()
        .flat_map(|status| store.claims_with_status(*status).unwrap_or_default())
        .map(|claim| claim.id)
        .collect();
    ids.sort();
    ids.dedup();

    let mut claims = Vec::new();
    for id in &ids {
        claims.extend(store.history(id)?);
    }

    tracing::info!(
        entities = entities.len(),
        claims = ids.len(),
        revisions = claims.len(),
        "knowledge graph exported"
    );

    Ok(Archive {
        version: ARCHIVE_VERSION,
        exported_at,
        entities,
        claims,
    })
}

/// Write an archive into a store, additively.
///
/// Entities are idempotent (their id is derived from kind and name, so
/// re-adding one is a no-op). Claims are only written when their id is absent;
/// an id that already exists is skipped and reported.
pub fn import(store: &KnowledgeStore, archive: &Archive) -> Result<ImportReport, Error> {
    let mut entities_added = 0;
    for entity in &archive.entities {
        if store.get_entity(&entity.id)?.is_none() {
            store.put_entity(entity)?;
            entities_added += 1;
        }
    }

    let mut claims_added = 0;
    let mut claims_skipped = Vec::new();
    // Revisions arrive oldest-first; replaying them in order rebuilds the
    // history exactly, including `superseded_by` links.
    for claim in &archive.claims {
        if store.latest(&claim.id)?.is_some() && claim.revision == 1 {
            claims_skipped.push(claim.id.0.clone());
            continue;
        }
        if claims_skipped.contains(&claim.id.0) {
            continue;
        }
        store.restore_revision(claim)?;
        claims_added += 1;
    }

    tracing::info!(
        entities_added,
        claims_added,
        skipped = claims_skipped.len(),
        "knowledge graph imported"
    );

    Ok(ImportReport {
        entities_added,
        claims_added,
        claims_skipped,
    })
}

/// Export the graph and write it to `<dir>/knowledge-<exported_at>.json`,
/// returning the path written. This is the manual backup primitive: the
/// *schedule* that calls it is an operator decision (cron), not something this
/// crate does — see the honest split in the module docs.
pub fn write_backup(
    store: &KnowledgeStore,
    dir: &std::path::Path,
    exported_at: u64,
) -> Result<std::path::PathBuf, Error> {
    std::fs::create_dir_all(dir).map_err(|e| {
        Error::Storage(format!("cannot create backup directory {}: {e}", dir.display()))
    })?;
    let archive = export(store, exported_at)?;
    let path = dir.join(format!("knowledge-{exported_at}.json"));
    std::fs::write(&path, archive.to_json()?).map_err(|e| {
        Error::Storage(format!("cannot write backup {}: {e}", path.display()))
    })?;
    tracing::info!(path = %path.display(), "knowledge backup written");
    Ok(path)
}

/// List the backups in `dir`, newest first, as `(path, exported_at)` pairs.
/// Missing directories and unparseable names are skipped rather than fatal —
/// listing should never fail because one stale file is in the way.
pub fn list_backups(dir: &std::path::Path) -> Vec<(std::path::PathBuf, u64)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut backups: Vec<(std::path::PathBuf, u64)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let ts = name
                .strip_prefix("knowledge-")?
                .strip_suffix(".json")?
                .parse::<u64>()
                .ok()?;
            Some((entry.path(), ts))
        })
        .collect();
    // Newest first.
    backups.sort_by(|a, b| b.1.cmp(&a.1));
    backups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EntityId, EntityKind, Provenance};

    fn store() -> (tempfile::TempDir, KnowledgeStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.redb");
        (dir, KnowledgeStore::open(path).unwrap())
    }

    #[test]
    fn write_backup_writes_a_listable_file() {
        let (dir, store) = store();
        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses bcrypt",
                EntityId::derive(EntityKind::File, "src/auth.rs"),
                Provenance {
                    producer: "w".into(),
                    observed_at: 1,
                    git_revision: None,
                    run_id: None,
                },
            ))
            .unwrap();

        let path = write_backup(&store, dir.path(), 1000).unwrap();
        assert!(path.exists());
        assert!(
            path.file_name().unwrap().to_string_lossy().starts_with("knowledge-1000")
        );

        let listed = list_backups(dir.path());
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1, 1000);
    }

    #[test]
    fn list_backups_is_newest_first_and_skips_junk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("knowledge-100.json"), "{}").unwrap();
        std::fs::write(dir.path().join("knowledge-300.json"), "{}").unwrap();
        std::fs::write(dir.path().join("not-a-backup.txt"), "x").unwrap();
        std::fs::write(dir.path().join("knowledge-abc.json"), "x").unwrap();

        let listed = list_backups(dir.path());
        assert_eq!(listed.len(), 2, "junk files must be skipped");
        assert_eq!(listed[0].1, 300, "newest first");
        assert_eq!(listed[1].1, 100);
    }
}
