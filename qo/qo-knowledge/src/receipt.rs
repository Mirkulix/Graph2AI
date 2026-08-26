//! Proof receipts — the graph answering "why should I believe this?"
//!
//! A claim alone is an assertion. A receipt is the assertion plus everything
//! the graph recorded around it: every revision it went through, who made
//! each one, the evidence that promoted it, and what *other* claims say about
//! the same subject — including the disagreements, which are kept, never
//! overwritten. This is the "check a claim rather than trust it" promise of
//! the whole crate, rendered as one deterministic block.
//!
//! ## Bounded and honest
//!
//! [`Receipt::render`] is deterministic (same graph state → byte-identical
//! text) and bounded: revision history, evidence and related claims are each
//! capped, and every cap is stated, so a long audit trail can never silently
//! look complete. A receipt for a `Proposed` claim says so explicitly — it
//! must not read like a fact.

use crate::model::{Claim, ClaimId, ClaimStatus};
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};

/// Caps so a receipt stays a proof, not a dump. Each is surfaced as a
/// truncation note rather than hidden.
const MAX_REVISIONS: usize = 20;
const MAX_RELATED: usize = 20;

/// Everything the graph recorded about one claim, gathered for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Receipt {
    /// The newest revision — what the claim says right now.
    pub claim: Claim,
    /// Every revision, oldest first. This is the append-only audit trail.
    pub history: Vec<Claim>,
    /// Latest claims about the same subject, excluding this one. Disagreements
    /// appear here with their own status — the graph never erases dissent.
    pub related: Vec<Claim>,
}

/// Gather a receipt for `id`. Fails with [`Error::NoSuchClaim`] when the id
/// does not exist.
pub fn build_receipt(store: &KnowledgeStore, id: &ClaimId) -> Result<Receipt, Error> {
    let claim = store.latest(id)?.ok_or_else(|| Error::NoSuchClaim(id.0.clone()))?;
    let history = store.history(id)?;

    // Related = latest claims about the same subject, minus this claim, minus
    // anything superseded. Ordered deterministically: strongest status first,
    // then newest, then id, mirroring the context compiler.
    let mut related: Vec<Claim> = store
        .claims_about(&claim.subject)?
        .into_iter()
        .filter(|c| c.id != claim.id && c.superseded_by.is_none())
        .collect();
    related.sort_by(|a, b| {
        let rank = |c: &Claim| match c.status {
            ClaimStatus::Verified => 0,
            ClaimStatus::Observed => 1,
            ClaimStatus::Proposed => 2,
            ClaimStatus::Stale => 3,
            ClaimStatus::Refuted => 4,
        };
        rank(a)
            .cmp(&rank(b))
            .then(b.provenance.observed_at.cmp(&a.provenance.observed_at))
            .then(a.id.0.cmp(&b.id.0))
    });

    Ok(Receipt {
        claim,
        history,
        related,
    })
}

impl Receipt {
    /// Render the receipt as a bounded, deterministic text block.
    pub fn render(&self) -> String {
        let mut out = String::new();

        let c = &self.claim;
        out.push_str("== PROOF RECEIPT ==\n");
        out.push_str(&format!("claim:     {}\n", c.id.0));
        out.push_str(&format!(
            "status:    {}{}\n",
            c.status.as_str().to_uppercase(),
            if c.is_load_bearing() {
                " (load-bearing)"
            } else {
                " (NOT load-bearing — do not treat as fact)"
            }
        ));
        out.push_str(&format!("subject:   {}\n", c.subject));
        out.push_str(&format!("statement: {}\n", c.statement));
        if let (Some(r), Some(o)) = (c.relation, c.object.as_ref()) {
            out.push_str(&format!("relation:  {} -> {}\n", r.as_str(), o));
        }

        out.push_str("\nrevisions (oldest -> newest):\n");
        let omitted = self.history.len().saturating_sub(MAX_REVISIONS);
        for rev in self.history.iter().take(MAX_REVISIONS) {
            out.push_str(&format!(
                "  rev {}  {:<9} by {} @{}\n",
                rev.revision,
                rev.status.as_str().to_uppercase(),
                rev.provenance.producer,
                rev.provenance.observed_at
            ));
        }
        if omitted > 0 {
            out.push_str(&format!("  ({omitted} earlier revision(s) omitted)\n"));
        }

        out.push_str("\nevidence:\n");
        if c.evidence.is_empty() {
            out.push_str("  (none — that is why this claim is not load-bearing)\n");
        } else {
            for e in &c.evidence {
                let lines = match e.lines {
                    Some((a, b)) if a == b => format!(":{a}"),
                    Some((a, b)) => format!(":{a}-{b}"),
                    None => String::new(),
                };
                let excerpt = e
                    .excerpt
                    .as_ref()
                    .map(|x| format!(" — {:?}", x))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "  [{}] {}{}  {}{}\n",
                    kind_str(e.kind),
                    e.locator,
                    lines,
                    if e.supports { "supports" } else { "contradicts" },
                    excerpt
                ));
            }
        }

        if !self.related.is_empty() {
            out.push_str(&format!(
                "\nrelated claims on the same subject ({}):\n",
                self.related.len()
            ));
            let omitted = self.related.len().saturating_sub(MAX_RELATED);
            for r in self.related.iter().take(MAX_RELATED) {
                let caveat = if r.is_load_bearing() { "" } else { " — kept, not overwritten" };
                out.push_str(&format!(
                    "  {}  {:<9} {:?}{}\n",
                    r.id.0,
                    r.status.as_str().to_uppercase(),
                    r.statement,
                    caveat
                ));
            }
            if omitted > 0 {
                out.push_str(&format!("  ({omitted} more related claim(s) omitted)\n"));
            }
        }

        out
    }
}

fn kind_str(kind: crate::model::EvidenceKind) -> &'static str {
    use crate::model::EvidenceKind;
    match kind {
        EvidenceKind::Source => "source",
        EvidenceKind::Commit => "commit",
        EvidenceKind::TestRun => "test_run",
        EvidenceKind::ToolRun => "tool_run",
        EvidenceKind::External => "external",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EntityId, EntityKind, Evidence, EvidenceKind, Provenance};

    fn store() -> (tempfile::TempDir, KnowledgeStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.redb");
        let store = KnowledgeStore::open(path).unwrap();
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

    #[test]
    fn a_verified_claim_receipt_tells_the_whole_story() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
        let id = ClaimId("c1".into());

        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth hashes passwords with bcrypt",
                subject.clone(),
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();
        store
            .verify_claim(
                &id,
                Evidence {
                    kind: EvidenceKind::Source,
                    locator: "src/auth.rs".into(),
                    lines: Some((42, 42)),
                    excerpt: Some("use bcrypt::hash;".into()),
                    supports: true,
                },
                prov("source-checker", 1_700_000_001),
            )
            .unwrap();

        let receipt = build_receipt(&store, &id).unwrap();
        let text = receipt.render();

        assert_eq!(receipt.history.len(), 2, "two revisions expected");
        assert!(text.contains("VERIFIED"), "{text}");
        assert!(text.contains("load-bearing"), "{text}");
        assert!(text.contains("rev 1"), "{text}");
        assert!(text.contains("rev 2"), "{text}");
        assert!(text.contains("worker-3"), "{text}");
        assert!(text.contains("source-checker"), "{text}");
        assert!(text.contains("use bcrypt::hash;"), "{text}");
    }

    #[test]
    fn a_proposed_receipt_says_it_is_not_a_fact() {
        let (_dir, store) = store();
        let id = ClaimId("c1".into());
        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses argon2",
                EntityId::derive(EntityKind::File, "src/auth.rs"),
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();

        let text = build_receipt(&store, &id).unwrap().render();
        assert!(text.contains("PROPOSED"), "{text}");
        assert!(text.contains("NOT load-bearing"), "{text}");
        assert!(text.contains("none"), "{text}");
    }

    #[test]
    fn disagreement_appears_in_related_claims() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
        let id = ClaimId("c1".into());

        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject.clone(),
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();
        // A second session disagrees with a different claim id.
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
                &ClaimId("c2".into()),
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

        let receipt = build_receipt(&store, &id).unwrap();
        assert_eq!(receipt.related.len(), 1);
        assert_eq!(receipt.related[0].id.0, "c2");
        assert_eq!(receipt.related[0].status, ClaimStatus::Refuted);

        let text = receipt.render();
        assert!(text.contains("c2"), "{text}");
        assert!(text.contains("REFUTED"), "{text}");
        assert!(text.contains("kept, not overwritten"), "{text}");
    }

    #[test]
    fn a_missing_claim_is_an_error() {
        let (_dir, store) = store();
        assert!(matches!(
            build_receipt(&store, &ClaimId("nope".into())),
            Err(Error::NoSuchClaim(_))
        ));
    }

    #[test]
    fn rendering_is_deterministic() {
        let (_dir, store) = store();
        let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
        store
            .add_claim(&crate::Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject,
                prov("worker-3", 1_700_000_000),
            ))
            .unwrap();
        let a = build_receipt(&store, &ClaimId("c1".into())).unwrap().render();
        let b = build_receipt(&store, &ClaimId("c1".into())).unwrap().render();
        assert_eq!(a, b);
    }
}
