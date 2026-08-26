//! Typed concepts of the knowledge graph.
//!
//! The rule this module exists to enforce: a claim is never separable from
//! its provenance. `Claim` has no constructor that omits `Provenance`, and
//! `ClaimStatus` cannot be set to `Verified` by a writer — only
//! [`crate::store::KnowledgeStore::verify_claim`] can move a claim there,
//! and only with an `Evidence` attached.

use serde::{Deserialize, Serialize};

/// A thing the graph can make statements about.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    /// Human-readable name — a file path, a symbol name, an agent name.
    pub name: String,
}

/// Stable identifier for an entity. Derived from `kind` + `name` so the same
/// file referred to twice resolves to the same entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(pub String);

impl EntityId {
    /// Derive a deterministic id. Same (kind, name) always yields the same id.
    pub fn derive(kind: EntityKind, name: &str) -> Self {
        EntityId(format!("{}:{}", kind.as_str(), name))
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    Repository,
    File,
    Symbol,
    Service,
    Endpoint,
    Concept,
    Agent,
    Run,
}

impl EntityKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityKind::Repository => "repository",
            EntityKind::File => "file",
            EntityKind::Symbol => "symbol",
            EntityKind::Service => "service",
            EntityKind::Endpoint => "endpoint",
            EntityKind::Concept => "concept",
            EntityKind::Agent => "agent",
            EntityKind::Run => "run",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "repository" => EntityKind::Repository,
            "file" => EntityKind::File,
            "symbol" => EntityKind::Symbol,
            "service" => EntityKind::Service,
            "endpoint" => EntityKind::Endpoint,
            "concept" => EntityKind::Concept,
            "agent" => EntityKind::Agent,
            "run" => EntityKind::Run,
            _ => return None,
        })
    }
}

/// A directed, typed connection between two entities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    Defines,
    Calls,
    DependsOn,
    Implements,
    Contradicts,
    Documents,
    Tests,
    Produces,
}

impl Relation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Relation::Defines => "defines",
            Relation::Calls => "calls",
            Relation::DependsOn => "depends_on",
            Relation::Implements => "implements",
            Relation::Contradicts => "contradicts",
            Relation::Documents => "documents",
            Relation::Tests => "tests",
            Relation::Produces => "produces",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "defines" => Relation::Defines,
            "calls" => Relation::Calls,
            "depends_on" => Relation::DependsOn,
            "implements" => Relation::Implements,
            "contradicts" => Relation::Contradicts,
            "documents" => Relation::Documents,
            "tests" => Relation::Tests,
            "produces" => Relation::Produces,
            _ => return None,
        })
    }
}

/// How much weight a claim carries.
///
/// Only `Observed` and `Verified` may be handed to planning agents without a
/// caveat — see [`ClaimStatus::is_load_bearing`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    /// Captured directly from code, config, test or tool output.
    Observed,
    /// Suggested by an LLM or agent. Not yet backed by reproducible evidence.
    Proposed,
    /// Confirmed by reproducible evidence or an authorised check.
    Verified,
    /// A newer revision may have invalidated this.
    Stale,
    /// Disproved by counter-evidence.
    Refuted,
}

impl ClaimStatus {
    /// Every status, for callers that must enumerate the whole space (export,
    /// snapshot views). Kept next to the enum so adding a variant without
    /// updating this is an obvious omission.
    pub const ALL: [ClaimStatus; 5] = [
        ClaimStatus::Observed,
        ClaimStatus::Proposed,
        ClaimStatus::Verified,
        ClaimStatus::Stale,
        ClaimStatus::Refuted,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            ClaimStatus::Observed => "observed",
            ClaimStatus::Proposed => "proposed",
            ClaimStatus::Verified => "verified",
            ClaimStatus::Stale => "stale",
            ClaimStatus::Refuted => "refuted",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "observed" => ClaimStatus::Observed,
            "proposed" => ClaimStatus::Proposed,
            "verified" => ClaimStatus::Verified,
            "stale" => ClaimStatus::Stale,
            "refuted" => ClaimStatus::Refuted,
            _ => return None,
        })
    }

    /// True only for statuses that may be presented as reliable context
    /// without an explicit caveat.
    pub fn is_load_bearing(&self) -> bool {
        matches!(self, ClaimStatus::Observed | ClaimStatus::Verified)
    }
}

/// Where a claim came from. Every claim carries this — there is no
/// constructor that omits it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Provenance {
    /// Who produced this — an agent name, a tool name, an indexer.
    pub producer: String,
    /// Unix seconds. Supplied by the caller so the store stays deterministic
    /// and testable.
    pub observed_at: u64,
    /// Git revision the observation was made against, when known.
    pub git_revision: Option<String>,
    /// Run or goal id that produced this, when known.
    pub run_id: Option<String>,
}

/// Support for (or against) a claim.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Evidence {
    pub kind: EvidenceKind,
    /// What to look at: a file path, a URL, a command, a run id.
    pub locator: String,
    /// Optional line range within `locator`, as (start, end), 1-indexed.
    pub lines: Option<(u32, u32)>,
    /// Verbatim excerpt, when short enough to be worth storing.
    pub excerpt: Option<String>,
    /// Does this support the claim, or contradict it?
    pub supports: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// A file and (optionally) a line range.
    Source,
    /// A git commit.
    Commit,
    /// A test run and its outcome.
    TestRun,
    /// Output of a tool invocation.
    ToolRun,
    /// An external document or URL.
    External,
}

/// A checkable statement about entities and relations.
///
/// Construct via [`Claim::observed`] or [`Claim::proposed`]. There is
/// deliberately no `Claim::verified` — reaching `Verified` requires going
/// through the store with evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Claim {
    pub id: ClaimId,
    /// The statement in plain language.
    pub statement: String,
    pub subject: EntityId,
    /// Set when the claim is specifically about a relation between two
    /// entities rather than a property of one.
    pub relation: Option<Relation>,
    pub object: Option<EntityId>,
    pub status: ClaimStatus,
    pub provenance: Provenance,
    pub evidence: Vec<Evidence>,
    /// Monotonic revision counter. Starts at 1; every status change or
    /// evidence addition writes a new revision rather than mutating.
    pub revision: u32,
    /// Set when a later revision supersedes this one.
    pub superseded_by: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClaimId(pub String);

impl std::fmt::Display for ClaimId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Claim {
    /// A claim captured directly from a deterministic source.
    ///
    /// `Observed` is load-bearing, so this requires at least one piece of
    /// evidence — an observation with nothing to point at is a proposal.
    pub fn observed(
        id: impl Into<String>,
        statement: impl Into<String>,
        subject: EntityId,
        provenance: Provenance,
        evidence: Vec<Evidence>,
    ) -> Result<Self, crate::Error> {
        if evidence.is_empty() {
            return Err(crate::Error::ObservationWithoutEvidence);
        }
        Ok(Self {
            id: ClaimId(id.into()),
            statement: statement.into(),
            subject,
            relation: None,
            object: None,
            status: ClaimStatus::Observed,
            provenance,
            evidence,
            revision: 1,
            superseded_by: None,
        })
    }

    /// A claim suggested by an LLM or agent. Evidence is optional here —
    /// that is exactly what makes it a proposal rather than an observation.
    pub fn proposed(
        id: impl Into<String>,
        statement: impl Into<String>,
        subject: EntityId,
        provenance: Provenance,
    ) -> Self {
        Self {
            id: ClaimId(id.into()),
            statement: statement.into(),
            subject,
            relation: None,
            object: None,
            status: ClaimStatus::Proposed,
            provenance,
            evidence: Vec::new(),
            revision: 1,
            superseded_by: None,
        }
    }

    /// Attach a relation and object, turning a claim about one entity into a
    /// claim about a connection between two.
    pub fn relating(mut self, relation: Relation, object: EntityId) -> Self {
        self.relation = Some(relation);
        self.object = Some(object);
        self
    }

    /// True if this claim may be presented to a planning agent as reliable
    /// context without a caveat.
    pub fn is_load_bearing(&self) -> bool {
        self.status.is_load_bearing() && self.superseded_by.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prov() -> Provenance {
        Provenance {
            producer: "test".into(),
            observed_at: 1_700_000_000,
            git_revision: None,
            run_id: None,
        }
    }

    fn ev() -> Evidence {
        Evidence {
            kind: EvidenceKind::Source,
            locator: "src/lib.rs".into(),
            lines: Some((1, 10)),
            excerpt: None,
            supports: true,
        }
    }

    #[test]
    fn entity_id_is_deterministic() {
        let a = EntityId::derive(EntityKind::File, "src/lib.rs");
        let b = EntityId::derive(EntityKind::File, "src/lib.rs");
        assert_eq!(a, b);
        assert_eq!(a.0, "file:src/lib.rs");
    }

    #[test]
    fn entity_id_separates_kinds() {
        let f = EntityId::derive(EntityKind::File, "foo");
        let s = EntityId::derive(EntityKind::Symbol, "foo");
        assert_ne!(f, s);
    }

    #[test]
    fn observation_requires_evidence() {
        let r = Claim::observed(
            "c1",
            "stmt",
            EntityId::derive(EntityKind::File, "a"),
            prov(),
            vec![],
        );
        assert!(matches!(r, Err(crate::Error::ObservationWithoutEvidence)));
    }

    #[test]
    fn observation_with_evidence_is_load_bearing() {
        let c = Claim::observed(
            "c1",
            "stmt",
            EntityId::derive(EntityKind::File, "a"),
            prov(),
            vec![ev()],
        )
        .unwrap();
        assert_eq!(c.status, ClaimStatus::Observed);
        assert!(c.is_load_bearing());
        assert_eq!(c.revision, 1);
    }

    #[test]
    fn proposal_is_not_load_bearing() {
        let c = Claim::proposed("c2", "guess", EntityId::derive(EntityKind::File, "a"), prov());
        assert_eq!(c.status, ClaimStatus::Proposed);
        assert!(!c.is_load_bearing());
    }

    #[test]
    fn only_observed_and_verified_are_load_bearing() {
        assert!(ClaimStatus::Observed.is_load_bearing());
        assert!(ClaimStatus::Verified.is_load_bearing());
        assert!(!ClaimStatus::Proposed.is_load_bearing());
        assert!(!ClaimStatus::Stale.is_load_bearing());
        assert!(!ClaimStatus::Refuted.is_load_bearing());
    }

    #[test]
    fn superseded_claim_is_not_load_bearing() {
        let mut c = Claim::observed(
            "c1",
            "stmt",
            EntityId::derive(EntityKind::File, "a"),
            prov(),
            vec![ev()],
        )
        .unwrap();
        assert!(c.is_load_bearing());
        c.superseded_by = Some(2);
        assert!(!c.is_load_bearing());
    }

    #[test]
    fn relating_attaches_relation_and_object() {
        let c = Claim::proposed("c3", "s", EntityId::derive(EntityKind::File, "a"), prov())
            .relating(Relation::Calls, EntityId::derive(EntityKind::Symbol, "f"));
        assert_eq!(c.relation, Some(Relation::Calls));
        assert_eq!(c.object, Some(EntityId::derive(EntityKind::Symbol, "f")));
    }

    #[test]
    fn status_round_trips_through_str() {
        for s in [
            ClaimStatus::Observed,
            ClaimStatus::Proposed,
            ClaimStatus::Verified,
            ClaimStatus::Stale,
            ClaimStatus::Refuted,
        ] {
            assert_eq!(ClaimStatus::parse(s.as_str()), Some(s));
        }
    }

    #[test]
    fn relation_round_trips_through_str() {
        for r in [
            Relation::Defines,
            Relation::Calls,
            Relation::DependsOn,
            Relation::Implements,
            Relation::Contradicts,
            Relation::Documents,
            Relation::Tests,
            Relation::Produces,
        ] {
            assert_eq!(Relation::parse(r.as_str()), Some(r));
        }
    }

    #[test]
    fn entity_kind_round_trips_through_str() {
        for k in [
            EntityKind::Repository,
            EntityKind::File,
            EntityKind::Symbol,
            EntityKind::Service,
            EntityKind::Endpoint,
            EntityKind::Concept,
            EntityKind::Agent,
            EntityKind::Run,
        ] {
            assert_eq!(EntityKind::parse(k.as_str()), Some(k));
        }
    }
}
