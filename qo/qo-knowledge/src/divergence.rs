//! Divergence report — where the graph's agents disagree.
//!
//! The merger keeps disagreements visible claim by claim (a refuted claim is
//! never deleted, and a conflict names both sides). What it does *not* do is
//! answer the aggregate question: **where, across the whole graph, do we hold
//! a settled fact and a settled counter-fact about the same thing?** That is
//! what this module answers.
//!
//! ## The rule, stated honestly
//!
//! A subject is *divergent* when it carries at least one load-bearing claim
//! (`verified` or `observed`) **and** at least one `refuted` claim. That is
//! the graph saying "we believe X, and we have also formally refuted a rival
//! claim about X". It is a *status* observation, not a semantic claim that
//! two statements contradict — the report shows both sides with their
//! evidence and leaves the final judgement to a human. It never invents a
//! contradiction from prose.
//!
//! Deterministic: the same graph state produces the same report, ordered by
//! subject id.

use crate::model::{Claim, ClaimStatus, EntityId};
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};

/// One divergent subject: the claims settled *for* it, and the claims settled
/// *against* it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Divergence {
    pub subject: EntityId,
    /// `verified` or `observed` claims about this subject.
    pub load_bearing: Vec<Claim>,
    /// `refuted` claims about this subject.
    pub refuted: Vec<Claim>,
}

/// Every subject where the graph holds both a load-bearing and a refuted claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceReport {
    pub divergences: Vec<Divergence>,
}

impl DivergenceReport {
    /// True when no subject is divergent.
    pub fn is_clean(&self) -> bool {
        self.divergences.is_empty()
    }

    /// Render the report as a deterministic text block.
    pub fn render(&self) -> String {
        if self.divergences.is_empty() {
            return "No divergences: no subject holds both a load-bearing claim and a refuted claim.\n".into();
        }
        let mut out = format!(
            "{} subject(s) hold both a load-bearing claim and a refuted claim:\n\n",
            self.divergences.len()
        );
        for divergence in &self.divergences {
            out.push_str(&format!("== {}\n", divergence.subject));
            out.push_str("  load-bearing:\n");
            for claim in &divergence.load_bearing {
                out.push_str(&format!(
                    "    - [{}] {} ({})\n",
                    claim.status.as_str(),
                    claim.statement,
                    claim.id.0
                ));
            }
            out.push_str("  refuted:\n");
            for claim in &divergence.refuted {
                out.push_str(&format!(
                    "    - [{}] {} ({})\n",
                    claim.status.as_str(),
                    claim.statement,
                    claim.id.0
                ));
            }
            out.push('\n');
        }
        out
    }
}

/// Collect every divergent subject in the graph.
pub fn divergences(store: &KnowledgeStore) -> Result<DivergenceReport, Error> {
    let mut by_subject: std::collections::BTreeMap<EntityId, (Vec<Claim>, Vec<Claim>)> =
        std::collections::BTreeMap::new();

    for claim in store.claims_with_status(ClaimStatus::Verified)? {
        by_subject.entry(claim.subject.clone()).or_default().0.push(claim);
    }
    for claim in store.claims_with_status(ClaimStatus::Observed)? {
        by_subject.entry(claim.subject.clone()).or_default().0.push(claim);
    }
    for claim in store.claims_with_status(ClaimStatus::Refuted)? {
        by_subject.entry(claim.subject.clone()).or_default().1.push(claim);
    }

    let mut divergences = Vec::new();
    for (subject, (load_bearing, refuted)) in by_subject {
        if !load_bearing.is_empty() && !refuted.is_empty() {
            divergences.push(Divergence {
                subject,
                load_bearing,
                refuted,
            });
        }
    }

    Ok(DivergenceReport { divergences })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EntityId, EntityKind, Evidence, EvidenceKind, Provenance};

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
    fn an_empty_graph_has_no_divergences() {
        let (_dir, store) = store();
        let report = divergences(&store).unwrap();
        assert!(report.is_clean());
        assert!(report.render().contains("No divergences"));
    }

    #[test]
    fn a_subject_with_a_verified_and_a_refuted_claim_is_divergent() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");

        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject.clone(),
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();
        store
            .verify_claim(
                &crate::ClaimId("c1".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "src/auth.rs".into(),
                    lines: Some((42, 42)),
                    excerpt: Some("use bcrypt::hash;".into()),
                    supports: true,
                },
                prov("reviewer", 1_700_000_001),
            )
            .unwrap();

        store
            .add_claim(&crate::Claim::proposed(
                "c2",
                "auth uses md5",
                subject.clone(),
                prov("worker-9", 1_700_000_100),
            ))
            .unwrap();
        store
            .refute_claim(
                &crate::ClaimId("c2".into()),
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "src/auth.rs".into(),
                    lines: Some((10, 10)),
                    excerpt: Some("use md5::compute;".into()),
                    supports: false,
                },
                prov("reviewer", 1_700_000_200),
            )
            .unwrap();

        let report = divergences(&store).unwrap();
        assert_eq!(report.divergences.len(), 1);
        assert_eq!(report.divergences[0].subject, subject);
        assert_eq!(report.divergences[0].load_bearing.len(), 1);
        assert_eq!(report.divergences[0].refuted.len(), 1);

        let text = report.render();
        assert!(text.contains("auth uses bcrypt"), "{text}");
        assert!(text.contains("auth uses md5"), "{text}");
    }

    #[test]
    fn different_subjects_do_not_diverge() {
        let (_dir, store) = store();
        let a = EntityId::derive(EntityKind::File, "src/a.rs");
        let b = EntityId::derive(EntityKind::File, "src/b.rs");

        store.add_claim(&crate::Claim::proposed("c1", "a does x", a, prov("w", 1))).unwrap();
        store.verify_claim(
            &crate::ClaimId("c1".into()),
            Evidence { kind: EvidenceKind::Source, locator: "a.rs".into(), lines: None, excerpt: None, supports: true },
            prov("r", 2),
        ).unwrap();

        store.add_claim(&crate::Claim::proposed("c2", "b does y", b, prov("w", 3))).unwrap();
        store.refute_claim(
            &crate::ClaimId("c2".into()),
            Evidence { kind: EvidenceKind::Source, locator: "b.rs".into(), lines: None, excerpt: None, supports: false },
            prov("r", 4),
        ).unwrap();

        assert!(divergences(&store).unwrap().is_clean());
    }

    #[test]
    fn rendering_is_deterministic() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
        store.add_claim(&crate::Claim::proposed("c1", "auth uses bcrypt", subject.clone(), prov("w", 1))).unwrap();
        store.verify_claim(
            &crate::ClaimId("c1".into()),
            Evidence { kind: EvidenceKind::Source, locator: "auth.rs".into(), lines: None, excerpt: None, supports: true },
            prov("r", 2),
        ).unwrap();
        store.add_claim(&crate::Claim::proposed("c2", "auth uses md5", subject, prov("w", 3))).unwrap();
        store.refute_claim(
            &crate::ClaimId("c2".into()),
            Evidence { kind: EvidenceKind::Source, locator: "auth.rs".into(), lines: None, excerpt: None, supports: false },
            prov("r", 4),
        ).unwrap();

        let a = divergences(&store).unwrap().render();
        let b = divergences(&store).unwrap().render();
        assert_eq!(a, b);
    }
}
