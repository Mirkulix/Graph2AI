//! Graph health — the operator's one-line answer to "how is my graph doing?".
//!
//! The individual signals (load-bearing count, stale count, divergence count)
//! already exist across the crate; this module gathers them into one
//! deterministic snapshot. It is a *summary*, not a new source of truth — every
//! number here is read back from the store the same way the other tools do.

use crate::model::ClaimStatus;
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};

/// A point-in-time health snapshot of the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphHealth {
    /// Claims a planning agent may rely on without a caveat.
    pub load_bearing: usize,
    pub verified: usize,
    pub observed: usize,
    pub proposed: usize,
    pub stale: usize,
    pub refuted: usize,
    /// Subjects where a load-bearing and a refuted claim coexist.
    pub divergences: usize,
    pub entities: usize,
}

impl GraphHealth {
    /// Render the snapshot as a deterministic, human-readable block.
    pub fn render(&self) -> String {
        let mut out = String::from("Knowledge graph health:\n");
        out.push_str(&format!(
            "  load-bearing: {} ({} verified, {} observed)\n",
            self.load_bearing, self.verified, self.observed
        ));
        out.push_str(&format!("  open proposals: {}\n", self.proposed));
        out.push_str(&format!("  stale: {}\n", self.stale));
        out.push_str(&format!("  refuted: {}\n", self.refuted));
        out.push_str(&format!(
            "  divergences: {} subject(s) where agents disagree\n",
            self.divergences
        ));
        out.push_str(&format!("  entities: {}\n", self.entities));
        out
    }
}

/// Read the whole graph's health in one pass.
pub fn health(store: &KnowledgeStore) -> Result<GraphHealth, Error> {
    let count = |status: ClaimStatus| store.claims_with_status(status).map(|c| c.len());
    let verified = count(ClaimStatus::Verified)?;
    let observed = count(ClaimStatus::Observed)?;
    let divergences = crate::divergence::divergences(store)?.divergences.len();

    Ok(GraphHealth {
        load_bearing: verified + observed,
        verified,
        observed,
        proposed: count(ClaimStatus::Proposed)?,
        stale: count(ClaimStatus::Stale)?,
        refuted: count(ClaimStatus::Refuted)?,
        divergences,
        entities: store.list_entities()?.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance};

    fn store() -> (tempfile::TempDir, KnowledgeStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.redb");
        (dir, KnowledgeStore::open(path).unwrap())
    }

    fn prov(producer: &str, at: u64) -> Provenance {
        Provenance {
            producer: producer.into(),
            observed_at: at,
            git_revision: None,
            run_id: None,
        }
    }

    #[test]
    fn an_empty_graph_reports_zero_everywhere() {
        let (_dir, store) = store();
        let h = health(&store).unwrap();
        assert_eq!(h.load_bearing, 0);
        assert_eq!(h.proposed, 0);
        assert_eq!(h.stale, 0);
        assert_eq!(h.divergences, 0);
        assert_eq!(h.entities, 0);
    }

    #[test]
    fn health_counts_each_status_and_divergences() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
        // `add_claim` records the subject index but does not create the entity;
        // an entity exists once it is declared (a `+E` line or `put_entity`).
        store
            .put_entity(&Entity {
                id: subject.clone(),
                kind: EntityKind::File,
                name: "src/auth.rs".into(),
            })
            .unwrap();

        // one verified + one refuted on the same subject -> 1 divergence
        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject.clone(),
                prov("w", 1),
            ))
            .unwrap();
        store
            .verify_claim(
                &crate::ClaimId("c1".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "auth.rs".into(),
                    lines: None,
                    excerpt: None,
                    supports: true,
                },
                prov("r", 2),
            )
            .unwrap();
        store
            .add_claim(&crate::Claim::proposed(
                "c2",
                "auth uses md5",
                subject.clone(),
                prov("w", 3),
            ))
            .unwrap();
        store
            .refute_claim(
                &crate::ClaimId("c2".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "auth.rs".into(),
                    lines: None,
                    excerpt: None,
                    supports: false,
                },
                prov("r", 4),
            )
            .unwrap();

        let h = health(&store).unwrap();
        assert_eq!(h.verified, 1);
        assert_eq!(h.refuted, 1);
        assert_eq!(h.load_bearing, 1);
        assert_eq!(h.divergences, 1);
        assert_eq!(h.entities, 1);

        let text = h.render();
        assert!(text.contains("load-bearing: 1"), "{text}");
        assert!(text.contains("divergences: 1"), "{text}");
    }
}
