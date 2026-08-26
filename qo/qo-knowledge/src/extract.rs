//! Constrained text-to-graph proposal pipeline.
//!
//! The admission gate between "an LLM wrote something" and "the graph may
//! consider it". An LLM proposes knowledge; it never decides what is true.
//! This module turns raw model output into a typed [`GraphDelta`] whose claims
//! are *all* proposals, or it reports every reason the output was refused —
//! with a source line wherever one exists.
//!
//! ## The path this enforces
//!
//! ```text
//! prose / findings
//!      │  (integration layer calls an LLM with [`proposal_system_prompt`])
//!      ▼
//! OrbitQLang text ──► propose_from_text ──► GraphDelta (all claims proposed)
//!                         │  strict parse + admission policy               │
//!                         ▼                                               ▼
//!              violations (line-accurate)                    sign + merge (commit path)
//! ```
//!
//! The parser already guarantees a lot: claims are `proposed` by construction
//! (the grammar has no status field), and malformed lines are reported with
//! line numbers. What this module adds is the *policy* layer, because the
//! grammar alone cannot express these rules:
//!
//! 1. **No promotion, no refutation from model text.** `OK` and `NO`
//!    operations are refused. An LLM cannot verify or refute a claim — that
//!    requires reproducible evidence or an authorised check, both outside
//!    what a text document can assert. (The signed commit path keeps `OK`/`NO`
//!    for workers with keys and real locators; this path is for unverified
//!    model output.)
//! 2. **A proposal is self-contained.** Every claim subject and relation
//!    object must resolve to an entity declared in the same document or listed
//!    in [`ProposalPolicy::known_entities`] (the caller, e.g. the server,
//!    supplies what the graph already knows). Every relation must target a
//!    claim declared in the same document. A dangling reference is refused at
//!    admission rather than becoming a merge conflict later.
//! 3. **Bound and checkable.** Statements have a length cap and must be
//!    non-empty, so a proposal cannot smuggle an essay in as a "statement".
//! 4. **All-or-nothing.** One bad line refuses the whole document, and the
//!    worker receives *every* violation at once — the same contract as
//!    `parse_recovering` and `orbit_graph_commit_delta`. Nothing is partially
//!    admitted, so no unvalidated fragment can reach the graph.
//!
//! ## What this module deliberately does not do
//!
//! It does not call an LLM, does not read the store, and does not sign. It is
//! the deterministic validation half; the integration layer owns the model
//! call, the context, and the signing key.

use crate::delta::{GraphDelta, GraphDeltaOp};
use crate::model::{ClaimId, EntityId};
use crate::orbitql::parse_recovering;
use std::collections::HashSet;

/// The admission policy a caller applies to model output.
///
/// Callers that have graph context (the server, an agent session) pass the
/// entities they already know about so a proposal may reference them without
/// re-declaring them. Everything else must be declared in the document.
#[derive(Debug, Clone)]
pub struct ProposalPolicy {
    /// Entities the caller already knows about, from graph context.
    pub known_entities: Vec<EntityId>,
    /// Hard cap on claim statement length, in characters.
    pub max_statement_chars: usize,
}

impl Default for ProposalPolicy {
    fn default() -> Self {
        Self {
            known_entities: Vec::new(),
            // A checkable statement, not an essay. Callers with a different
            // tokenizer may raise this.
            max_statement_chars: 500,
        }
    }
}

impl ProposalPolicy {
    pub fn with_known_entities(mut self, entities: impl IntoIterator<Item = EntityId>) -> Self {
        self.known_entities = entities.into_iter().collect();
        self
    }

    pub fn with_max_statement_chars(mut self, cap: usize) -> Self {
        self.max_statement_chars = cap;
        self
    }
}

/// One reason the proposal was not admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalViolation {
    /// 1-indexed source line, when the violation can be attributed to one.
    /// Parse errors always have it; document-level checks may not.
    pub line: Option<usize>,
    pub message: String,
}

impl ProposalViolation {
    fn at(line: usize, message: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            message: message.into(),
        }
    }

    fn whole(message: impl Into<String>) -> Self {
        Self {
            line: None,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProposalViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.line {
            Some(line) => write!(f, "line {line}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}
impl std::error::Error for ProposalViolation {}

/// Result of admitting a proposal: a validated delta, or every reason it was
/// refused. Never both.
#[derive(Debug, Clone)]
pub struct ProposalOutcome {
    /// The admitted delta, present only when there are no violations. All its
    /// claims are `proposed`; it is ready for `sign_delta` + `merge_signed_delta`.
    pub delta: Option<GraphDelta>,
    /// Every violation, in document order. When non-empty, `delta` is `None`.
    pub violations: Vec<ProposalViolation>,
}

impl ProposalOutcome {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty() && self.delta.is_some()
    }
}

/// Admit (or refuse) an LLM-written OrbitQLang document as a proposal.
///
/// Deterministic: the same text and policy always produce the same outcome.
/// The document is parsed recoveringly, so a worker sees every problem at
/// once; then the admission policy runs; then the delta's own validation runs
/// as a final gate (it catches e.g. an unsupported `DELTA` version).
pub fn propose_from_text(text: &str, policy: &ProposalPolicy) -> ProposalOutcome {
    let outcome = parse_recovering(text);

    // Parse errors refuse the document whole. A worker must fix them and
    // resubmit; admitting the valid half of a broken document would let a
    // partial truth in.
    if !outcome.errors.is_empty() {
        let violations = outcome
            .errors
            .iter()
            .map(|e| ProposalViolation {
                line: Some(e.line),
                message: e.message.clone(),
            })
            .collect();
        return ProposalOutcome {
            delta: None,
            violations,
        };
    }

    let Some(delta) = outcome.delta else {
        return ProposalOutcome {
            delta: None,
            violations: vec![ProposalViolation::whole(
                "empty document: expected a DELTA header and BY producer line",
            )],
        };
    };

    let mut violations = admit(&delta, &outcome.op_lines, policy);

    // Final gate: the delta's own contract (version, non-empty ids, proposed
    // status, evidence direction). The parser normally produces valid deltas,
    // but this is the one place every path through the crate is held to it.
    if violations.is_empty() {
        if let Err(e) = delta.validate() {
            violations.push(ProposalViolation::whole(format!(
                "document fails delta validation: {e}"
            )));
        }
    }

    ProposalOutcome {
        delta: if violations.is_empty() {
            Some(delta)
        } else {
            None
        },
        violations,
    }
}

/// The policy checks, attributed to source lines via `op_lines`.
fn admit(delta: &GraphDelta, op_lines: &[usize], policy: &ProposalPolicy) -> Vec<ProposalViolation> {
    let declared_entities: HashSet<&EntityId> = delta
        .operations
        .iter()
        .filter_map(|op| match op {
            GraphDeltaOp::AddEntity { entity } => Some(&entity.id),
            _ => None,
        })
        .collect();
    let known: HashSet<&EntityId> = policy.known_entities.iter().collect();

    // Collected in a first pass so a relation may reference a claim declared
    // anywhere in the document, not just earlier. The parser is already
    // order-tolerant (BY may follow claims); admission is too.
    let declared_claims: HashSet<&ClaimId> = delta
        .operations
        .iter()
        .filter_map(|op| match op {
            GraphDeltaOp::AddClaim { claim } => Some(&claim.id),
            _ => None,
        })
        .collect();

    let mut violations = Vec::new();

    // A claim id may be declared only once per document; two claims sharing an
    // id would race in the merger (DuplicateClaimId) and must not be admitted
    // as if they were one.
    let mut seen_claim_ids = HashSet::new();

    for (index, op) in delta.operations.iter().enumerate() {
        let line = op_lines.get(index).copied();
        match op {
            GraphDeltaOp::AddEntity { .. } => {}

            GraphDeltaOp::AddClaim { claim } => {
                if claim.statement.chars().count() > policy.max_statement_chars {
                    violations.push(violation(
                        line,
                        format!(
                            "claim statement is {} characters, over the {} cap",
                            claim.statement.chars().count(),
                            policy.max_statement_chars
                        ),
                    ));
                }
                if !declared_entities.contains(&claim.subject) && !known.contains(&claim.subject) {
                    violations.push(violation(
                        line,
                        format!(
                            "claim subject {} is not declared in this document (add a +E line) nor known to the graph",
                            claim.subject
                        ),
                    ));
                }
                if !seen_claim_ids.insert(&claim.id) {
                    violations.push(violation(
                        line,
                        format!("claim id {} is declared more than once", claim.id.0),
                    ));
                }
            }

            GraphDeltaOp::AddRelation { claim_id, object, .. } => {
                if !declared_claims.contains(&claim_id) {
                    violations.push(violation(
                        line,
                        format!(
                            "relation targets claim {}, which is not declared in this document",
                            claim_id.0
                        ),
                    ));
                }
                if !declared_entities.contains(object) && !known.contains(object) {
                    violations.push(violation(
                        line,
                        format!(
                            "relation object {} is not declared in this document (add a +E line) nor known to the graph",
                            object
                        ),
                    ));
                }
            }

            // The heart of the proposal policy: model text never promotes or
            // refutes. Verification is a separate, authorised step with
            // reproducible evidence; a text document cannot provide it.
            GraphDeltaOp::VerifyClaim { .. } => violations.push(violation(
                line,
                "a proposal may not verify a claim (OK): verification is an authorised step with reproducible evidence",
            )),
            GraphDeltaOp::RefuteClaim { .. } => violations.push(violation(
                line,
                "a proposal may not refute a claim (NO): refutation is an authorised step with reproducible counter-evidence",
            )),
        }
    }

    violations
}

fn violation(line: Option<usize>, message: impl Into<String>) -> ProposalViolation {
    match line {
        Some(line) => ProposalViolation::at(line, message),
        None => ProposalViolation::whole(message),
    }
}

/// The system-prompt block an integration hands an LLM so its output can be
/// admitted by [`propose_from_text`].
///
/// Deterministic and self-contained: grammar, constraints and an example. The
/// integration appends its own task prose and the caller's identity/context.
pub fn proposal_system_prompt() -> String {
    format!(
        "\
You propose updates to a shared knowledge graph. You never decide what is true \
— a separate, authorised verification step does.

Emit exactly one OrbitQLang document, and nothing else: no prose, no \
explanation, no markdown fences. One operation per line, fields separated by |.

Operations you may emit:
  +E|<entity_kind>|<name>          declare an entity (repository, file, symbol,
                                   service, endpoint, concept, agent, run)
  +C|<claim_id>|<subject>|<statement>   propose a checkable statement about an
                                   entity (subject is <kind>:<name>)
  +R|<claim_id>|<relation>|<object>     relate a claim to another entity
                                   (defines, calls, depends_on, implements,
                                   contradicts, documents, tests, produces)

Rules:
  - Declare every entity with +E before referencing it.
  - Every claim must be a single, checkable statement in plain language.
  - Statements stay under {cap} characters.
  - Never emit OK or NO: you may not verify or refute claims.
  - Free text may contain | and newlines; escape them as \\p, \\n, \\r, \\\\.
  - The document starts with a DELTA header and a BY producer line.

Example:
DELTA|1|d-42
BY|worker-3|1700000000
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
",
        cap = ProposalPolicy::default().max_statement_chars
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ClaimStatus, EntityKind};

    fn policy() -> ProposalPolicy {
        ProposalPolicy::default()
    }

    #[test]
    fn a_clean_proposal_is_admitted_whole() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
+R|c1|depends_on|file:Cargo.toml
";
        let outcome = propose_from_text(text, &policy());
        assert!(outcome.is_ok(), "{:?}", outcome.violations);
        let delta = outcome.delta.unwrap();
        assert_eq!(delta.operations.len(), 4);
        // Claims came through as proposals — the only status the grammar and
        // the admission policy allow.
        let GraphDeltaOp::AddClaim { claim } = &delta.operations[2] else {
            panic!("expected a claim operation");
        };
        assert_eq!(claim.status, ClaimStatus::Proposed);
    }

    #[test]
    fn an_llm_cannot_verify_or_refute() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
        let outcome = propose_from_text(text, &policy());
        assert!(!outcome.is_ok());
        assert!(outcome.delta.is_none(), "a refused proposal never yields a delta");
        assert_eq!(outcome.violations.len(), 1);
        assert_eq!(outcome.violations[0].line, Some(5), "{:?}", outcome.violations);
        assert!(
            outcome.violations[0].message.contains("may not verify"),
            "{}",
            outcome.violations[0]
        );

        let refute = text.replace("OK|", "NO|");
        let outcome = propose_from_text(&refute, &policy());
        assert!(
            outcome.violations[0].message.contains("may not refute"),
            "{}",
            outcome.violations[0]
        );
    }

    #[test]
    fn a_relation_to_an_undeclared_claim_is_refused() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|Cargo.toml
+R|c99|depends_on|file:Cargo.toml
";
        let outcome = propose_from_text(text, &policy());
        assert!(!outcome.is_ok());
        assert_eq!(outcome.violations.len(), 1, "{:?}", outcome.violations);
        assert_eq!(outcome.violations[0].line, Some(5));
        assert!(outcome.violations[0].message.contains("c99"));
    }

    #[test]
    fn an_undeclared_subject_is_refused() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+C|c1|file:src/auth.rs|auth uses bcrypt
";
        let outcome = propose_from_text(text, &policy());
        assert!(!outcome.is_ok());
        assert_eq!(outcome.violations[0].line, Some(3));
        assert!(outcome.violations[0].message.contains("subject"));
    }

    #[test]
    fn known_entities_from_context_satisfy_resolution() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+C|c1|file:src/auth.rs|auth uses bcrypt
";
        let policy = policy().with_known_entities([
            EntityId::derive(EntityKind::File, "src/auth.rs"),
            EntityId::derive(EntityKind::File, "Cargo.toml"),
        ]);
        let outcome = propose_from_text(text, &policy);
        assert!(outcome.is_ok(), "{:?}", outcome.violations);
    }

    #[test]
    fn an_overlong_statement_is_refused() {
        let text = format!(
            "DELTA|1|d-1\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|{}\n",
            "x".repeat(600)
        );
        let outcome = propose_from_text(&text, &policy());
        assert!(!outcome.is_ok());
        assert_eq!(outcome.violations.len(), 1);
        assert_eq!(outcome.violations[0].line, Some(4));
        assert!(outcome.violations[0].message.contains("600"));
    }

    #[test]
    fn every_malformed_line_is_reported_with_its_line() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|not_a_kind|src/auth.rs
+C|c1|file:src/auth.rs
+R|c1|not_a_relation|file:x.rs
";
        let outcome = propose_from_text(text, &policy());
        assert!(!outcome.is_ok());
        assert!(outcome.delta.is_none());
        assert_eq!(outcome.violations.len(), 3, "{:?}", outcome.violations);
        assert!(outcome.violations.iter().all(|v| v.line.is_some()));
    }

    #[test]
    fn an_unsupported_version_is_refused() {
        let text = "\
DELTA|2|d-1
BY|worker-3|1700000000
+C|c1|file:src/auth.rs|auth uses bcrypt
";
        let policy = policy().with_known_entities([EntityId::derive(
            EntityKind::File,
            "src/auth.rs",
        )]);
        let outcome = propose_from_text(text, &policy);
        assert!(!outcome.is_ok());
        assert!(
            outcome.violations[0].message.contains("version"),
            "{}",
            outcome.violations[0]
        );
    }

    #[test]
    fn a_duplicate_claim_id_is_refused() {
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
+C|c1|file:src/auth.rs|auth uses argon2
";
        let outcome = propose_from_text(text, &policy());
        assert!(!outcome.is_ok());
        assert_eq!(outcome.violations.len(), 1, "{:?}", outcome.violations);
        assert_eq!(outcome.violations[0].line, Some(5));
        assert!(outcome.violations[0].message.contains("more than once"));
    }

    #[test]
    fn statement_escaping_survives_admission() {
        // The document carries `\p` (a pipe) and `\\` (a backslash) escaped;
        // the admitted delta must carry the unescaped statement.
        let text = "DELTA|1|d-1\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|splits on \\p and \\\\ line breaks\n";
        let outcome = propose_from_text(text, &policy());
        assert!(outcome.is_ok(), "{:?}", outcome.violations);
        let delta = outcome.delta.unwrap();
        let GraphDeltaOp::AddClaim { claim } = &delta.operations[1] else {
            panic!("expected a claim operation");
        };
        assert_eq!(claim.statement, "splits on | and \\ line breaks");
    }

    #[test]
    fn a_relation_may_precede_its_claim() {
        // The parser tolerates out-of-order lines (BY may follow claims);
        // admission must too, so a worker never has to sort its document.
        let text = "\
DELTA|1|d-1
BY|worker-3|1700000000
+E|file|src/auth.rs
+E|file|Cargo.toml
+R|c1|depends_on|file:Cargo.toml
+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt
";
        let outcome = propose_from_text(text, &policy());
        assert!(outcome.is_ok(), "{:?}", outcome.violations);
    }

    #[test]
    fn the_system_prompt_mentions_the_forbidden_verbs() {
        let prompt = proposal_system_prompt();
        assert!(prompt.contains("Never emit OK or NO"));
        assert!(prompt.contains("+E|"));
        assert!(prompt.contains("DELTA|1|"));
    }
}
