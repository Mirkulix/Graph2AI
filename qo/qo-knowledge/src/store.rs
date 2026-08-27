//! Persistence, revisions and indices for the knowledge graph.
//!
//! Storage is redb, following the pattern already used by `qo-memory`.
//! Claims are append-only: a status change or an added piece of evidence
//! writes a new revision and marks the old one `superseded_by`. Nothing is
//! overwritten, so a contradiction stays visible instead of silently winning.
//!
//! Four indices exist so queries do not have to scan every claim:
//!   * `claims`      — (claim_id, revision) -> Claim JSON
//!   * `idx_subject` — entity_id -> claim_ids
//!   * `idx_object`  — entity_id -> claim_ids  (reverse traversal)
//!   * `idx_status`  — status    -> claim_ids
//!   * `entities`    — entity_id -> Entity JSON

use crate::model::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, Evidence, Provenance, Relation,
};
use crate::Error;
use redb::{Database, ReadableTable, TableDefinition};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// Key is `"<claim_id>\u{1}<revision:010}"` so revisions of one claim sort
/// together and the latest is the last in range.
const CLAIMS: TableDefinition<&str, &str> = TableDefinition::new("k_claims");
const ENTITIES: TableDefinition<&str, &str> = TableDefinition::new("k_entities");
const IDX_SUBJECT: TableDefinition<&str, &str> = TableDefinition::new("k_idx_subject");
const IDX_OBJECT: TableDefinition<&str, &str> = TableDefinition::new("k_idx_object");
const IDX_STATUS: TableDefinition<&str, &str> = TableDefinition::new("k_idx_status");
/// Signatures already accepted, so a captured delta cannot be replayed.
/// Key is `producer/delta_id`, value is the signature that was accepted.
const SEEN_DELTAS: TableDefinition<&str, &str> = TableDefinition::new("k_seen_deltas");
/// Schema bookkeeping (`schema_version`). Lets the server fail fast instead of
/// silently misreading a database written by an incompatible binary.
const META: TableDefinition<&str, &str> = TableDefinition::new("k_meta");
/// Bump when a knowledge-table layout changes incompatibly. The migration path
/// is always the portable archive: export with the compatible binary, re-import
/// with the new one.
const SCHEMA_VERSION: &str = "1";

const SEP: char = '\u{1}';

fn claim_key(id: &ClaimId, revision: u32) -> String {
    format!("{}{}{:010}", id.0, SEP, revision)
}

fn claim_prefix(id: &ClaimId) -> String {
    format!("{}{}", id.0, SEP)
}

/// Index values are a JSON array of ids. Small enough for the volumes this
/// serves, and keeps the schema readable in a redb dump.
fn decode_ids(raw: Option<&str>) -> BTreeSet<String> {
    raw.and_then(|s| serde_json::from_str::<BTreeSet<String>>(s).ok())
        .unwrap_or_default()
}

pub struct KnowledgeStore {
    db: Arc<Database>,
}

impl KnowledgeStore {
    /// Open (or create) a store at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let db = Arc::new(Database::create(path)?);
        Self::init(&db)?;
        Ok(Self { db })
    }

    /// Share an existing database, the way `GraphStore` shares `Store`'s.
    pub fn from_db(db: Arc<Database>) -> Result<Self, Error> {
        Self::init(&db)?;
        Ok(Self { db })
    }

    fn init(db: &Database) -> Result<(), Error> {
        let w = db.begin_write()?;
        {
            w.open_table(CLAIMS)?;
            w.open_table(ENTITIES)?;
            w.open_table(IDX_SUBJECT)?;
            w.open_table(IDX_OBJECT)?;
            w.open_table(IDX_STATUS)?;
            w.open_table(SEEN_DELTAS)?;
            w.open_table(META)?;
        }
        // Fail fast on an incompatible on-disk schema instead of silently
        // misreading it. A missing key means "first open with this binary":
        // write the current version so future drift is detectable.
        {
            let mut meta = w.open_table(META)?;
            // Copy the version out first so the immutable read borrow does not
            // block the later insert.
            let version = meta.get("schema_version")?.map(|v| v.value().to_string());
            match version.as_deref() {
                Some(v) if v != SCHEMA_VERSION => {
                    return Err(Error::Storage(format!(
                        "knowledge-store schema version {v:?} is not supported \
                         (this binary expects {SCHEMA_VERSION}); export with a \
                         compatible binary and re-import"
                    )));
                }
                None => {
                    meta.insert("schema_version", SCHEMA_VERSION)?;
                }
                _ => {}
            }
        }
        w.commit()?;
        Ok(())
    }

    // ---- replay guard ----

    /// Record that a signed delta was accepted, and report whether it is new.
    ///
    /// Returns `Ok(true)` the first time a `(producer, delta_id)` pair is
    /// seen, `Ok(false)` on any repeat — including a repeat that carries a
    /// *different* signature, which is a submitter reusing an id rather than
    /// a network retry.
    ///
    /// The merge itself is idempotent, so a replay cannot corrupt the graph.
    /// This exists so a replay is visible rather than silent: an attacker
    /// re-submitting a captured delta is not the same event as a worker
    /// retrying after a timeout, even though both are harmless to the data.
    pub fn record_delta(
        &self,
        producer: &str,
        delta_id: &str,
        signature: &str,
    ) -> Result<bool, Error> {
        let key = format!("{producer}/{delta_id}");
        let w = self.db.begin_write()?;
        let is_new = {
            let mut table = w.open_table(SEEN_DELTAS)?;
            let existing = table.get(key.as_str())?.map(|v| v.value().to_string());
            match existing {
                Some(_) => false,
                None => {
                    table.insert(key.as_str(), signature)?;
                    true
                }
            }
        };
        w.commit()?;
        Ok(is_new)
    }

    /// The signature recorded for a delta, if it has been accepted before.
    pub fn seen_delta(&self, producer: &str, delta_id: &str) -> Result<Option<String>, Error> {
        let r = self.db.begin_read()?;
        let table = r.open_table(SEEN_DELTAS)?;
        Ok(table
            .get(format!("{producer}/{delta_id}").as_str())?
            .map(|v| v.value().to_string()))
    }

    // ---- entities ----

    pub fn put_entity(&self, entity: &Entity) -> Result<(), Error> {
        let json = serde_json::to_string(entity)?;
        let w = self.db.begin_write()?;
        {
            let mut t = w.open_table(ENTITIES)?;
            t.insert(entity.id.0.as_str(), json.as_str())?;
        }
        w.commit()?;
        Ok(())
    }

    pub fn get_entity(&self, id: &EntityId) -> Result<Option<Entity>, Error> {
        let r = self.db.begin_read()?;
        let t = r.open_table(ENTITIES)?;
        match t.get(id.0.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    pub fn list_entities(&self) -> Result<Vec<Entity>, Error> {
        let r = self.db.begin_read()?;
        let t = r.open_table(ENTITIES)?;
        let mut out = Vec::new();
        for row in t.iter()? {
            let (_, v) = row?;
            out.push(serde_json::from_str(v.value())?);
        }
        Ok(out)
    }

    // ---- claims ----

    /// Write a brand-new claim at revision 1.
    ///
    /// Rejects a duplicate id — use [`Self::verify_claim`] or
    /// [`Self::refute_claim`] to advance an existing claim instead.
    pub fn add_claim(&self, claim: &Claim) -> Result<(), Error> {
        if self.latest(&claim.id)?.is_some() {
            return Err(Error::ClaimExists(claim.id.0.clone()));
        }
        let mut c = claim.clone();
        c.revision = 1;
        c.superseded_by = None;
        self.write_revision(&c, None)
    }

    /// Return the newest revision of a claim.
    pub fn latest(&self, id: &ClaimId) -> Result<Option<Claim>, Error> {
        let r = self.db.begin_read()?;
        let t = r.open_table(CLAIMS)?;
        let prefix = claim_prefix(id);
        let mut found: Option<Claim> = None;
        for row in t.range(prefix.as_str()..)? {
            let (k, v) = row?;
            if !k.value().starts_with(&prefix) {
                break;
            }
            found = Some(serde_json::from_str(v.value())?);
        }
        Ok(found)
    }

    /// Return every revision of a claim, oldest first. This is the audit
    /// trail — superseded revisions are kept, never deleted.
    pub fn history(&self, id: &ClaimId) -> Result<Vec<Claim>, Error> {
        let r = self.db.begin_read()?;
        let t = r.open_table(CLAIMS)?;
        let prefix = claim_prefix(id);
        let mut out = Vec::new();
        for row in t.range(prefix.as_str()..)? {
            let (k, v) = row?;
            if !k.value().starts_with(&prefix) {
                break;
            }
            out.push(serde_json::from_str(v.value())?);
        }
        Ok(out)
    }

    /// Confirm a claim with supporting evidence, moving it to `Verified`.
    ///
    /// This is the only path to `Verified`. The evidence must actually
    /// support the claim; passing counter-evidence here is rejected so a
    /// caller cannot "verify" something into truth with a refutation.
    pub fn verify_claim(
        &self,
        id: &ClaimId,
        evidence: Evidence,
        by: Provenance,
    ) -> Result<Claim, Error> {
        if !evidence.supports {
            return Err(Error::CounterEvidenceForVerify);
        }
        self.advance(id, ClaimStatus::Verified, Some(evidence), by)
    }

    /// Disprove a claim with counter-evidence, moving it to `Refuted`.
    pub fn refute_claim(
        &self,
        id: &ClaimId,
        evidence: Evidence,
        by: Provenance,
    ) -> Result<Claim, Error> {
        if evidence.supports {
            return Err(Error::SupportingEvidenceForRefute);
        }
        self.advance(id, ClaimStatus::Refuted, Some(evidence), by)
    }

    /// Mark a claim as possibly out of date.
    pub fn mark_stale(&self, id: &ClaimId, by: Provenance) -> Result<Claim, Error> {
        self.advance(id, ClaimStatus::Stale, None, by)
    }

    /// Record a fresh deterministic observation after a source changed. This
    /// is intentionally separate from `verify_claim`: an indexer may only
    /// refresh direct source evidence, never promote an LLM proposal.
    pub fn refresh_observed(
        &self,
        id: &ClaimId,
        evidence: Evidence,
        by: Provenance,
    ) -> Result<Claim, Error> {
        if !evidence.supports {
            return Err(Error::CounterEvidenceForVerify);
        }
        self.advance(id, ClaimStatus::Observed, Some(evidence), by)
    }

    /// Append a new revision with a new status, superseding the previous one.
    /// Attach a relation and object to a claim that does not yet have one.
    ///
    /// Like every other mutation this appends a revision rather than editing
    /// in place, so the pre-relation form stays in `history`. Refuses to
    /// re-point an existing relation — that is a merge conflict, not a write.
    pub fn attach_relation(
        &self,
        id: &ClaimId,
        relation: Relation,
        object: EntityId,
        by: Provenance,
    ) -> Result<Claim, Error> {
        let prev = self.latest(id)?.ok_or_else(|| Error::NoSuchClaim(id.0.clone()))?;
        if prev.relation.is_some() {
            return Err(Error::Storage(format!(
                "claim {} already carries a relation",
                id.0
            )));
        }

        let mut next = prev.clone();
        next.revision = prev.revision + 1;
        next.provenance = by;
        next.superseded_by = None;
        next.relation = Some(relation);
        next.object = Some(object);

        self.write_revision(&next, Some(prev))?;
        Ok(next)
    }

    /// Write a revision exactly as it was recorded elsewhere.
    ///
    /// This is for [`crate::archive::import`] only, and it deliberately
    /// bypasses the status rules: a restored `Verified` claim was verified
    /// once, with its evidence, and re-deriving that through `verify_claim`
    /// would rewrite provenance and renumber revisions — turning a faithful
    /// restore into a forgery of who decided what, when.
    ///
    /// It is not a general write path. Everything that *creates* knowledge
    /// still goes through `add_claim` / `verify_claim` / `refute_claim`.
    pub fn restore_revision(&self, claim: &Claim) -> Result<(), Error> {
        let previous = if claim.revision > 1 {
            self.latest(&claim.id)?
        } else {
            None
        };
        self.write_revision(claim, previous)
    }

    fn advance(
        &self,
        id: &ClaimId,
        status: ClaimStatus,
        evidence: Option<Evidence>,
        by: Provenance,
    ) -> Result<Claim, Error> {
        let prev = self.latest(id)?.ok_or_else(|| Error::NoSuchClaim(id.0.clone()))?;

        let mut next = prev.clone();
        next.revision = prev.revision + 1;
        next.status = status;
        next.provenance = by;
        next.superseded_by = None;
        if let Some(e) = evidence {
            next.evidence.push(e);
        }

        // A status change is what makes a claim load-bearing (or stops it
        // being so). That transition is the auditable event, not the write.
        tracing::info!(
            claim = %id,
            from = prev.status.as_str(),
            to = status.as_str(),
            revision = next.revision,
            by = %next.provenance.producer,
            "claim status changed"
        );

        self.write_revision(&next, Some(prev))?;
        Ok(next)
    }

    /// Write one revision and update every index. `previous`, when given, is
    /// rewritten with `superseded_by` set.
    fn write_revision(&self, claim: &Claim, previous: Option<Claim>) -> Result<(), Error> {
        let json = serde_json::to_string(claim)?;
        let key = claim_key(&claim.id, claim.revision);

        let w = self.db.begin_write()?;
        {
            let mut claims = w.open_table(CLAIMS)?;
            claims.insert(key.as_str(), json.as_str())?;

            if let Some(mut prev) = previous {
                prev.superseded_by = Some(claim.revision);
                let pk = claim_key(&prev.id, prev.revision);
                let pj = serde_json::to_string(&prev)?;
                claims.insert(pk.as_str(), pj.as_str())?;

                // The old status no longer describes this claim.
                let mut status_idx = w.open_table(IDX_STATUS)?;
                let old = prev.status.as_str();
                let mut ids = decode_ids(status_idx.get(old)?.as_ref().map(|v| v.value()));
                ids.remove(&claim.id.0);
                let encoded = serde_json::to_string(&ids)?;
                status_idx.insert(old, encoded.as_str())?;
            }

            let mut subj = w.open_table(IDX_SUBJECT)?;
            let mut ids = decode_ids(subj.get(claim.subject.0.as_str())?.as_ref().map(|v| v.value()));
            ids.insert(claim.id.0.clone());
            let encoded = serde_json::to_string(&ids)?;
            subj.insert(claim.subject.0.as_str(), encoded.as_str())?;

            if let Some(obj) = &claim.object {
                let mut o = w.open_table(IDX_OBJECT)?;
                let mut ids = decode_ids(o.get(obj.0.as_str())?.as_ref().map(|v| v.value()));
                ids.insert(claim.id.0.clone());
                let encoded = serde_json::to_string(&ids)?;
                o.insert(obj.0.as_str(), encoded.as_str())?;
            }

            let mut status_idx = w.open_table(IDX_STATUS)?;
            let s = claim.status.as_str();
            let mut ids = decode_ids(status_idx.get(s)?.as_ref().map(|v| v.value()));
            ids.insert(claim.id.0.clone());
            let encoded = serde_json::to_string(&ids)?;
            status_idx.insert(s, encoded.as_str())?;
        }
        w.commit()?;
        Ok(())
    }

    // ---- queries ----

    /// Latest revision of every claim whose subject is `entity`.
    pub fn claims_about(&self, entity: &EntityId) -> Result<Vec<Claim>, Error> {
        let ids = {
            let r = self.db.begin_read()?;
            let t = r.open_table(IDX_SUBJECT)?;
            decode_ids(t.get(entity.0.as_str())?.as_ref().map(|v| v.value()))
        };
        self.collect_latest(ids)
    }

    /// Latest revision of every claim pointing *at* `entity` — the reverse
    /// direction, used for impact questions ("what depends on this?").
    pub fn claims_referencing(&self, entity: &EntityId) -> Result<Vec<Claim>, Error> {
        let ids = {
            let r = self.db.begin_read()?;
            let t = r.open_table(IDX_OBJECT)?;
            decode_ids(t.get(entity.0.as_str())?.as_ref().map(|v| v.value()))
        };
        self.collect_latest(ids)
    }

    /// Latest revision of every claim currently in `status`.
    pub fn claims_with_status(&self, status: ClaimStatus) -> Result<Vec<Claim>, Error> {
        let ids = {
            let r = self.db.begin_read()?;
            let t = r.open_table(IDX_STATUS)?;
            decode_ids(t.get(status.as_str())?.as_ref().map(|v| v.value()))
        };
        let mut out = self.collect_latest(ids)?;
        // The index is updated on write, but a claim only belongs to a status
        // if its newest revision says so.
        out.retain(|c| c.status == status);
        Ok(out)
    }

    /// Neighbours of `entity`: claims that connect it to something else,
    /// in either direction.
    pub fn neighbors(&self, entity: &EntityId) -> Result<Vec<(Relation, EntityId, Claim)>, Error> {
        let mut out = Vec::new();
        for c in self.claims_about(entity)? {
            if let (Some(r), Some(o)) = (c.relation, c.object.clone()) {
                out.push((r, o, c));
            }
        }
        for c in self.claims_referencing(entity)? {
            if let Some(r) = c.relation {
                out.push((r, c.subject.clone(), c));
            }
        }
        Ok(out)
    }

    /// Free-text search over statements. Case-insensitive substring match —
    /// deliberately not semantic, because this crate makes no claim it
    /// cannot back up.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Claim>, Error> {
        let needle = query.to_lowercase();
        let mut out = Vec::new();
        let r = self.db.begin_read()?;
        let t = r.open_table(CLAIMS)?;
        for row in t.iter()? {
            let (_, v) = row?;
            let c: Claim = serde_json::from_str(v.value())?;
            if c.superseded_by.is_some() {
                continue;
            }
            if c.statement.to_lowercase().contains(&needle) {
                out.push(c);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Claims safe to hand a planning agent without a caveat: `Observed` or
    /// `Verified`, newest revision only.
    ///
    /// This is the function that enforces the core rule of the design. If a
    /// caller wants proposals too, it has to ask for them explicitly and
    /// label them itself.
    pub fn load_bearing_context(
        &self,
        entity: &EntityId,
        limit: usize,
    ) -> Result<Vec<Claim>, Error> {
        let mut out: Vec<Claim> = self
            .claims_about(entity)?
            .into_iter()
            .filter(|c| c.is_load_bearing())
            .collect();
        // Verified outranks observed; newer outranks older.
        out.sort_by(|a, b| {
            let rank = |c: &Claim| match c.status {
                ClaimStatus::Verified => 0,
                ClaimStatus::Observed => 1,
                _ => 2,
            };
            rank(a)
                .cmp(&rank(b))
                .then(b.provenance.observed_at.cmp(&a.provenance.observed_at))
        });
        out.truncate(limit);
        Ok(out)
    }

    fn collect_latest(&self, ids: BTreeSet<String>) -> Result<Vec<Claim>, Error> {
        let mut out = Vec::new();
        for id in ids {
            if let Some(c) = self.latest(&ClaimId(id))? {
                out.push(c);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First open stamps the current schema version; a database written by an
    /// incompatible binary is refused with a migration hint instead of being
    /// silently misread.
    #[test]
    fn an_incompatible_schema_version_fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.redb");

        // A normal first open succeeds and stamps the current version.
        {
            let store = KnowledgeStore::open(&path).unwrap();
            assert!(store.claims_with_status(ClaimStatus::Proposed).unwrap().is_empty());
        }

        // Simulate a future binary's database: bump the version to 999.
        {
            let db = redb::Database::create(&path).unwrap();
            let w = db.begin_write().unwrap();
            {
                let mut t = w.open_table(META).unwrap();
                t.insert("schema_version", "999").unwrap();
            }
            w.commit().unwrap();
        }

        // Re-opening must fail fast with the migration path, not misread data.
        let err = match KnowledgeStore::open(&path) {
            Ok(_) => panic!("expected open to refuse an incompatible schema"),
            Err(e) => e,
        };
        let text = err.to_string();
        assert!(text.contains("not supported"), "{text}");
        assert!(text.contains("re-import"), "{text}");
    }
}
