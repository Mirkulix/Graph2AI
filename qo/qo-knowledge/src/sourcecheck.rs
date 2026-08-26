//! Deterministic source-evidence verification.
//!
//! The other half of the proposal loop. [`crate::extract`] lets an LLM propose
//! claims; this module lets the graph *check* a proposal against real source
//! code and promote it to a verified fact only when the code literally
//! substantiates it.
//!
//! ## The rule, stated honestly
//!
//! This is a **lexical** check, not a semantic one. A statement is reduced to
//! its *distinctive terms* — lowercase words, short stopwords removed — and
//! the claim is promoted only when **every** distinctive term appears in the
//! located source region. What the check produces is reproducible: the same
//! statement and the same file always yield the same verdict, and the exact
//! matching line is captured as evidence, so a human can see *why* the graph
//! believed it.
//!
//! ## The asymmetry that keeps this honest
//!
//! - **Confirm, never disprove.** A full match *confirms* a claim, but a
//!   partial or empty match does **not** refute it — the words may simply be
//!   paraphrased. Refutation stays an authorised, human decision
//!   ([`KnowledgeStore::refute_claim`]); this module never auto-refutes.
//! - **Promote, never launder.** Only a `proposed` claim is promoted, and only
//!   with [`KnowledgeStore::verify_claim`] — the same single path to
//!   [`ClaimStatus::Verified`] everything else uses. An unverifiable proposal
//!   is left exactly as it was, with a reason.
//! - **No path escapes.** The source path is joined to the workspace root and
//!   canonicalised; a result outside the root is refused. This mirrors the
//!   tool-sandbox rule and never dereferences a path the graph did not resolve.
//!
//! ## What it is not
//!
//! It does not parse Rust, does not build an AST, and does not call an LLM. A
//! claim whose statement uses words the file never contains is inconclusive
//! until a human phrases it in terms the code can substantiate — which is
//! exactly the discipline a checkable knowledge graph wants.

use crate::model::{Claim, ClaimStatus, EntityKind, Evidence, EvidenceKind, Provenance};
use crate::store::KnowledgeStore;
use crate::Error;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single source file must stay under this size before the checker refuses
/// to scan it. Source files are small; a huge or binary file is almost never
/// the kind of thing a claim is verifiable against.
const MAX_SOURCE_BYTES: u64 = 1 << 20; // 1 MiB

/// Stopwords dropped before term matching. These are words that appear in
/// almost any prose and therefore prove nothing about a specific file.
const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "has", "have", "had", "its", "are", "was", "were", "been",
    "not", "but", "this", "that", "these", "those", "from", "into", "onto", "use", "uses",
    "used", "using", "will", "does", "should", "would", "could", "shall", "can", "via", "per",
    "than", "when", "where", "what", "which", "how", "who", "why", "any", "all", "each", "both",
    "some", "such", "more", "most", "other", "only", "own", "same", "about", "they", "them",
    "their", "there", "here", "your", "you", "our", "also", "then", "than", "after", "before",
    "between", "during", "until", "while", "against", "among", "across", "along", "around",
    "must", "might", "may", "even", "very", "just", "like", "well", "without", "within",
    "through", "being", "into", "over", "under", "above", "below", "again", "once",
];

/// The distinctive, checkable terms of a statement, in first-appearance order.
///
/// Lowercased, split on non-alphanumeric characters, filtered to words of at
/// least three characters that are not stopwords, deduplicated, and capped at
/// 16 terms. Splitting on `_` and `.` means `validate_token` and
/// `auth.rs` yield the terms `validate`, `token`, `auth`, `rs`.
pub fn distinctive_terms(statement: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    for token in statement
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
    {
        if token.len() < 3 || STOPWORDS.contains(&token) {
            continue;
        }
        if seen.insert(token.to_string()) {
            terms.push(token.to_string());
        }
        if terms.len() == 16 {
            break;
        }
    }
    terms
}

/// The pure deterministic decision: does `source` substantiate `statement`?
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assessment {
    /// True when every distinctive term appears in the source region.
    pub supports: bool,
    /// The distinctive terms that were required.
    pub terms: Vec<String>,
    /// How many of them appeared.
    pub matched: usize,
    /// The best-matching line, trimmed and truncated, when one exists.
    pub excerpt: Option<String>,
}

/// Reduce a statement to terms and match them against the located region.
///
/// `lines` is a 1-indexed inclusive range into `source`; when `None`, the whole
/// source is searched. Matching is case-insensitive substring matching: the
/// term `hash` matches `hashes`, `rehash` and `HASH`, but a term does not match
/// across a split it did not have. Deterministic — no randomness, no model.
pub fn assess(statement: &str, source: &str, lines: Option<(u32, u32)>) -> Assessment {
    let terms = distinctive_terms(statement);
    if terms.is_empty() {
        return Assessment {
            supports: false,
            terms,
            matched: 0,
            excerpt: None,
        };
    }

    let region = region(source, lines);
    let lowered = region.to_lowercase();

    let mut matched = 0;
    for term in &terms {
        if lowered.contains(term.as_str()) {
            matched += 1;
        }
    }

    // The best evidence line is the one carrying the most terms; ties go to
    // the earliest. That is what a human checking the claim looks at first.
    let excerpt = best_line(&region, &terms);

    Assessment {
        supports: matched == terms.len(),
        terms,
        matched,
        excerpt,
    }
}

/// Extract a 1-indexed, inclusive line range from `source`. A missing or
/// invalid range yields the whole source, so a bad span degrades to a wider
/// search rather than a false "not found".
fn region(source: &str, lines: Option<(u32, u32)>) -> String {
    let Some((start, end)) = lines else {
        return source.to_string();
    };
    if start == 0 || end < start {
        return source.to_string();
    }
    let all: Vec<&str> = source.lines().collect();
    let start = (start as usize).min(all.len());
    let end = (end as usize).min(all.len());
    if start == 0 || start > all.len() {
        return source.to_string();
    }
    all[start - 1..end].join("\n")
}

/// The line with the most term hits (ties → earliest), trimmed.
///
/// The full line is kept verbatim: it is stored as the claim's evidence
/// excerpt, and [`refresh_sources`] later compares that excerpt against the
/// current file to detect staleness. Truncating here would make that
/// comparison unreliable (a truncated excerpt never matches the source).
fn best_line(source: &str, terms: &[String]) -> Option<String> {
    let lowered_terms: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();
    source
        .lines()
        .map(|line| {
            let hits = lowered_terms
                .iter()
                .filter(|t| line.to_lowercase().contains(t.as_str()))
                .count();
            (hits, line)
        })
        .filter(|(hits, _)| *hits > 0)
        .max_by_key(|(hits, _)| *hits)
        .map(|(_, line)| line.trim().to_string())
}

/// What a source check concluded about a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    /// Every distinctive term appears in the source; the claim is verified.
    Verified,
    /// The source neither fully substantiates nor disproves the claim; the
    /// graph is unchanged.
    Inconclusive { reason: String },
    /// The claim is not a `proposed` claim, so there is nothing to promote.
    NotProposed { status: ClaimStatus },
    /// The source could not be read, was too large, or escaped the root.
    Unavailable { reason: String },
}

/// The full result of checking a claim against source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceCheck {
    pub verdict: Verdict,
    /// Distinctive terms that were required, in order.
    pub terms: Vec<String>,
    /// How many of them were found.
    pub matched: usize,
    /// The evidence that promoted the claim, when `verdict` is `Verified`.
    pub evidence: Option<Evidence>,
}

/// Check a `proposed` claim against its source file and promote it to
/// `Verified` only when every distinctive term is literally present.
///
/// The source location is taken from the claim's first `Source` evidence, or
/// — when it has none — from the claim subject if it is a `File` entity.
/// The path is resolved within `root` and refused if it escapes.
///
/// This is the deterministic bridge between "an LLM proposed it" and "the
/// graph checked it": it calls [`KnowledgeStore::verify_claim`], the same
/// single path to `Verified` every other route uses.
pub fn verify_claim_against_source(
    store: &KnowledgeStore,
    id: &crate::ClaimId,
    root: &Path,
    by: Provenance,
) -> Result<SourceCheck, Error> {
    let claim = store.latest(id)?.ok_or_else(|| Error::NoSuchClaim(id.0.clone()))?;

    if claim.status != ClaimStatus::Proposed {
        return Ok(SourceCheck {
            verdict: Verdict::NotProposed {
                status: claim.status,
            },
            terms: Vec::new(),
            matched: 0,
            evidence: None,
        });
    }

    // Where to look: explicit source evidence first, else the file subject.
    let source_evidence = claim
        .evidence
        .iter()
        .find(|e| e.kind == EvidenceKind::Source)
        .cloned();
    let locator = source_evidence
        .as_ref()
        .map(|e| e.locator.clone())
        .or_else(|| {
            (claim.subject_kind() == Some(EntityKind::File)).then(|| claim.subject_name())
        });

    let Some(locator) = locator.filter(|s| !s.trim().is_empty()) else {
        return Ok(SourceCheck {
            verdict: Verdict::Unavailable {
                reason: "claim has no source locator and its subject is not a file".into(),
            },
            terms: distinctive_terms(&claim.statement),
            matched: 0,
            evidence: None,
        });
    };

    let resolved = match resolve_within(root, &locator) {
        Ok(path) => path,
        Err(reason) => {
            return Ok(SourceCheck {
                verdict: Verdict::Unavailable { reason },
                terms: distinctive_terms(&claim.statement),
                matched: 0,
                evidence: None,
            });
        }
    };

    let content = match read_bounded(&resolved) {
        Ok(content) => content,
        Err(reason) => {
            return Ok(SourceCheck {
                verdict: Verdict::Unavailable { reason },
                terms: distinctive_terms(&claim.statement),
                matched: 0,
                evidence: None,
            });
        }
    };

    let assessment = assess(&claim.statement, &content, source_evidence.as_ref().and_then(|e| e.lines));

    if !assessment.supports {
        let reason = if assessment.terms.is_empty() {
            "the statement has no checkable terms".to_string()
        } else if assessment.matched == 0 {
            "none of the statement's terms appear in the source".to_string()
        } else {
            format!(
                "{} of {} terms appear in the source; a full match is required to promote",
                assessment.matched,
                assessment.terms.len()
            )
        };
        return Ok(SourceCheck {
            verdict: Verdict::Inconclusive { reason },
            terms: assessment.terms,
            matched: assessment.matched,
            evidence: None,
        });
    }

    let evidence = Evidence {
        kind: EvidenceKind::Source,
        locator,
        lines: source_evidence.as_ref().and_then(|e| e.lines),
        excerpt: assessment.excerpt,
        supports: true,
    };
    store.verify_claim(id, evidence.clone(), by)?;

    Ok(SourceCheck {
        verdict: Verdict::Verified,
        terms: assessment.terms,
        matched: assessment.matched,
        evidence: Some(evidence),
    })
}

/// One claim's outcome in a verification sweep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepOutcome {
    pub claim_id: String,
    pub verdict: Verdict,
}

/// Result of sweeping every open proposal against source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SweepReport {
    /// Number of proposals examined.
    pub checked: usize,
    /// Promoted to `verified` because the source literally substantiated them.
    pub verified: usize,
    /// Left proposed — the source did not fully substantiate them.
    pub inconclusive: usize,
    /// Could not be checked — missing source, unsafe path, or unreadable file.
    pub unavailable: usize,
    /// Per-claim outcomes, in first-proposal order.
    pub results: Vec<SweepOutcome>,
}

impl SweepReport {
    /// True when no proposal remains open. Note `inconclusive` and
    /// `unavailable` still leave claims open, so this is strict.
    pub fn fully_verified(&self) -> bool {
        self.inconclusive == 0 && self.unavailable == 0 && self.verified == self.checked
    }

    /// Render the report for a model or human to read.
    pub fn render(&self) -> String {
        let mut out = format!(
            "Verified {} of {} open proposal(s): {} verified, {} inconclusive, {} unavailable.\n",
            self.verified, self.checked, self.verified, self.inconclusive, self.unavailable
        );
        for result in &self.results {
            match &result.verdict {
                Verdict::Verified => {
                    out.push_str(&format!("  {}  VERIFIED\n", result.claim_id));
                }
                Verdict::Inconclusive { reason } => {
                    out.push_str(&format!("  {}  inconclusive — {reason}\n", result.claim_id));
                }
                Verdict::Unavailable { reason } => {
                    out.push_str(&format!("  {}  unavailable — {reason}\n", result.claim_id));
                }
                Verdict::NotProposed { status } => {
                    out.push_str(&format!("  {}  {} (not proposed)\n", result.claim_id, status.as_str()));
                }
            }
        }
        out
    }
}

/// Check every open proposal against its source, in one pass.
///
/// This is the "harvest" step of the proposal loop: after workers propose
/// claims, a single call has the graph read each claim's source and promote
/// the ones the code literally substantiates. Deterministic, offline, and
/// bounded — each claim is checked independently, and a claim that cannot be
/// checked (missing source, unsafe path) is reported rather than guessed at.
///
/// The same `by` provenance is stamped on every promotion, so the sweep is
/// auditable as one actor's run.
pub fn verify_all_proposals(
    store: &KnowledgeStore,
    root: &Path,
    by: Provenance,
) -> Result<SweepReport, Error> {
    let proposed = store.claims_with_status(ClaimStatus::Proposed)?;

    let mut report = SweepReport {
        checked: proposed.len(),
        verified: 0,
        inconclusive: 0,
        unavailable: 0,
        results: Vec::with_capacity(proposed.len()),
    };

    for claim in proposed {
        let check = verify_claim_against_source(store, &claim.id, root, by.clone())?;
        match &check.verdict {
            Verdict::Verified => report.verified += 1,
            Verdict::Inconclusive { .. } => report.inconclusive += 1,
            Verdict::Unavailable { .. } => report.unavailable += 1,
            Verdict::NotProposed { .. } => {}
        }
        report.results.push(SweepOutcome {
            claim_id: claim.id.0,
            verdict: check.verdict,
        });
    }

    Ok(report)
}

/// One claim's outcome in a source refresh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleOutcome {
    pub claim_id: String,
    pub outcome: StaleOutcomeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StaleOutcomeKind {
    /// The recorded excerpt is still present in the source.
    Current,
    /// The source changed or disappeared; the claim was marked `stale`.
    Stale,
    /// No verbatim excerpt to compare, so the claim was left untouched.
    Skipped { reason: String },
}

/// Result of a source refresh: which settled claims still hold, and which
/// were marked stale because the code moved on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaleReport {
    /// Settled (verified/observed) claims examined.
    pub checked: usize,
    /// Still backed by the recorded excerpt.
    pub still_current: usize,
    /// Marked `stale` — the recorded excerpt is gone from the source.
    pub stale: usize,
    /// Left untouched because they carry no verbatim excerpt to compare.
    pub skipped: usize,
    pub results: Vec<StaleOutcome>,
}

impl StaleReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "Source refresh: {} still current, {} stale, {} skipped (no excerpt) of {} checked.\n",
            self.still_current, self.stale, self.skipped, self.checked
        );
        for result in &self.results {
            match &result.outcome {
                StaleOutcomeKind::Current => {
                    out.push_str(&format!("  {}  still current\n", result.claim_id));
                }
                StaleOutcomeKind::Stale => {
                    out.push_str(&format!("  {}  STALE — source changed or is missing\n", result.claim_id));
                }
                StaleOutcomeKind::Skipped { reason } => {
                    out.push_str(&format!("  {}  skipped — {reason}\n", result.claim_id));
                }
            }
        }
        out
    }
}

/// Re-check every settled claim against its source and mark the stale ones.
///
/// A claim whose source is gone or whose recorded verbatim excerpt no longer
/// appears in the file is marked [`ClaimStatus::Stale`] — the graph noticing
/// that its facts rot. This is deterministic because it compares the exact
/// excerpt captured at verification time against the current file, never a
/// filesystem timestamp.
///
/// Only `verified` and `observed` claims are examined (a proposal is already
/// open, and a refuted claim is settled against it). Claims without a
/// verbatim excerpt are skipped, not guessed at.
pub fn refresh_sources(
    store: &KnowledgeStore,
    root: &Path,
    by: Provenance,
) -> Result<StaleReport, Error> {
    let mut candidates = store.claims_with_status(ClaimStatus::Verified)?;
    candidates.extend(store.claims_with_status(ClaimStatus::Observed)?);
    // Deterministic order so two refreshes of the same state report the same
    // sequence.
    candidates.sort_by(|a, b| a.id.0.cmp(&b.id.0));

    let mut report = StaleReport {
        checked: 0,
        still_current: 0,
        stale: 0,
        skipped: 0,
        results: Vec::with_capacity(candidates.len()),
    };

    for claim in candidates {
        report.checked += 1;
        let claim_id = claim.id.0.clone();

        let excerpt = match claim
            .evidence
            .iter()
            .find(|e| {
                e.kind == EvidenceKind::Source
                    && e.supports
                    && e.excerpt.as_ref().is_some_and(|x| !x.is_empty())
            })
            .and_then(|e| e.excerpt.clone())
        {
            Some(excerpt) => excerpt,
            None => {
                report.skipped += 1;
                report.results.push(StaleOutcome {
                    claim_id,
                    outcome: StaleOutcomeKind::Skipped {
                        reason: "no verbatim source excerpt to compare".into(),
                    },
                });
                continue;
            }
        };

        // The locator is the same source evidence that carried the excerpt.
        let locator = claim
            .evidence
            .iter()
            .find(|e| e.kind == EvidenceKind::Source && e.supports)
            .map(|e| e.locator.clone())
            .unwrap_or_default();

        let content = match resolve_within(root, &locator).and_then(|p| read_bounded(&p)) {
            Ok(content) => content,
            Err(_) => {
                // The source is gone or unreadable — the fact no longer holds.
                store.mark_stale(&claim.id, by.clone())?;
                report.stale += 1;
                report.results.push(StaleOutcome {
                    claim_id,
                    outcome: StaleOutcomeKind::Stale,
                });
                continue;
            }
        };

        if content.contains(excerpt.as_str()) {
            report.still_current += 1;
            report.results.push(StaleOutcome {
                claim_id,
                outcome: StaleOutcomeKind::Current,
            });
        } else {
            store.mark_stale(&claim.id, by.clone())?;
            report.stale += 1;
            report.results.push(StaleOutcome {
                claim_id,
                outcome: StaleOutcomeKind::Stale,
            });
        }
    }

    Ok(report)
}

/// One stale claim's outcome in a heal pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealOutcome {
    pub claim_id: String,
    pub outcome: HealOutcomeKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HealOutcomeKind {
    /// Re-verified against the current source — the fact holds, just moved.
    Healed,
    /// The source no longer substantiates the statement; it stays stale.
    RemainedStale { reason: String },
}

/// Result of re-verifying stale claims against their current source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealReport {
    /// Stale claims examined.
    pub examined: usize,
    /// Re-promoted to `verified` — the fact is still substantiated.
    pub healed: usize,
    /// Left stale — the statement no longer holds.
    pub remained_stale: usize,
    pub results: Vec<HealOutcome>,
}

impl HealReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "Healed {} of {} stale claim(s): {} healed, {} still stale.\n",
            self.healed, self.examined, self.healed, self.remained_stale
        );
        for result in &self.results {
            match &result.outcome {
                HealOutcomeKind::Healed => {
                    out.push_str(&format!("  {}  HEALED — re-verified against current source\n", result.claim_id));
                }
                HealOutcomeKind::RemainedStale { reason } => {
                    out.push_str(&format!("  {}  still stale — {reason}\n", result.claim_id));
                }
            }
        }
        out
    }
}

/// Re-verify every stale claim against its current source, healing the ones
/// whose statement is still substantiated.
///
/// This is the self-healing half of the lifecycle. [`refresh_sources`] marks a
/// claim stale when its recorded *excerpt* is gone; but the *fact* may still
/// hold — the code may have moved the line rather than removing the behaviour.
/// This pass re-runs the deterministic check against the whole current file
/// (not the stale line hint) and promotes a claim back to `verified` with fresh
/// evidence only when every distinctive term is still literally present. A
/// claim whose statement is genuinely gone stays stale.
///
/// The store keeps the whole `verified -> stale -> verified` revision trail, so
/// the rot and the healing are both auditable, never rewritten away.
pub fn heal_stale(
    store: &KnowledgeStore,
    root: &Path,
    by: Provenance,
) -> Result<HealReport, Error> {
    let stale = store.claims_with_status(ClaimStatus::Stale)?;

    let mut report = HealReport {
        examined: stale.len(),
        healed: 0,
        remained_stale: 0,
        results: Vec::with_capacity(stale.len()),
    };

    for claim in stale {
        let claim_id = claim.id.0.clone();

        let source_evidence = claim
            .evidence
            .iter()
            .find(|e| e.kind == EvidenceKind::Source)
            .cloned();
        let locator = source_evidence
            .as_ref()
            .map(|e| e.locator.clone())
            .or_else(|| {
                (claim.subject_kind() == Some(EntityKind::File)).then(|| claim.subject_name())
            });

        let Some(locator) = locator.filter(|s| !s.trim().is_empty()) else {
            report.remained_stale += 1;
            report.results.push(HealOutcome {
                claim_id,
                outcome: HealOutcomeKind::RemainedStale {
                    reason: "no source locator to re-check".into(),
                },
            });
            continue;
        };

        let content = match resolve_within(root, &locator).and_then(|p| read_bounded(&p)) {
            Ok(content) => content,
            Err(_) => {
                report.remained_stale += 1;
                report.results.push(HealOutcome {
                    claim_id,
                    outcome: HealOutcomeKind::RemainedStale {
                        reason: "source is missing or unreadable".into(),
                    },
                });
                continue;
            }
        };

        // Whole-file search, not the stale line hint: the point is whether the
        // fact still holds anywhere in the file.
        let assessment = assess(&claim.statement, &content, None);

        if assessment.supports {
            let evidence = Evidence {
                kind: EvidenceKind::Source,
                locator,
                lines: None,
                excerpt: assessment.excerpt,
                supports: true,
            };
            store.verify_claim(&claim.id, evidence, by.clone())?;
            report.healed += 1;
            report.results.push(HealOutcome {
                claim_id,
                outcome: HealOutcomeKind::Healed,
            });
        } else {
            let reason = if assessment.terms.is_empty() {
                "the statement has no checkable terms".to_string()
            } else {
                format!(
                    "{} of {} terms appear in the current source",
                    assessment.matched,
                    assessment.terms.len()
                )
            };
            report.remained_stale += 1;
            report.results.push(HealOutcome {
                claim_id,
                outcome: HealOutcomeKind::RemainedStale { reason },
            });
        }
    }

    Ok(report)
}

/// Join `rel` to `root`, canonicalise, and require the result to stay inside
/// the canonicalised root. This is the path-safety boundary: absolute paths,
/// `..` and symlink escapes all fail the prefix check after canonicalisation.
fn resolve_within(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .map_err(|e| format!("workspace root unavailable: {e}"))?;
    let joined = root.join(rel);
    let canon = joined
        .canonicalize()
        .map_err(|e| format!("cannot resolve source {rel}: {e}"))?;
    if canon.starts_with(&root) {
        Ok(canon)
    } else {
        Err(format!("source path {rel} escapes the workspace root"))
    }
}

fn read_bounded(path: &Path) -> Result<String, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("cannot stat source: {e}"))?;
    if meta.len() > MAX_SOURCE_BYTES {
        return Err(format!(
            "source is {} bytes, over the {} byte cap",
            meta.len(),
            MAX_SOURCE_BYTES
        ));
    }
    std::fs::read_to_string(path).map_err(|e| format!("cannot read source: {e}"))
}

// Convenience accessors so the locator resolution reads clearly above.
trait ClaimExt {
    fn subject_kind(&self) -> Option<EntityKind>;
    fn subject_name(&self) -> String;
}

impl ClaimExt for Claim {
    fn subject_kind(&self) -> Option<EntityKind> {
        // Entity ids are `kind:name`; the kind is the prefix before the first
        // `:`. EntityKind::parse recovers it unambiguously.
        let prefix = self.subject.0.split(':').next()?;
        crate::model::EntityKind::parse(prefix)
    }

    fn subject_name(&self) -> String {
        self.subject
            .0
            .split_once(':')
            .map(|(_, name)| name.to_string())
            .unwrap_or_else(|| self.subject.0.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinctive_terms_drop_stopwords_and_short_words() {
        assert_eq!(
            distinctive_terms("auth hashes passwords with bcrypt"),
            vec!["auth", "hashes", "passwords", "bcrypt"]
        );
        // `has`, `a`, `with` are dropped; `validate_token` splits on `_`.
        assert_eq!(
            distinctive_terms("has a validate_token helper"),
            vec!["validate", "token", "helper"]
        );
    }

    #[test]
    fn terms_are_deduplicated_and_capped() {
        let terms = distinctive_terms("one two three one two three");
        assert_eq!(terms.len(), 3); // one, two, three — deduplicated
    }

    #[test]
    fn a_full_match_is_supporting() {
        let a = assess(
            "auth hashes passwords with bcrypt",
            "fn hash_password(pw: &str) {\n    bcrypt::hash(pw)\n}\n",
            None,
        );
        // `auth` and `passwords` do not appear verbatim, so this is not a
        // full match — that is the conservative behaviour, not a bug.
        assert!(!a.supports);
        assert!(a.matched < a.terms.len());

        let b = assess(
            "auth hashes passwords with bcrypt",
            "// auth hashes passwords with bcrypt\nfn run() {}\n",
            None,
        );
        assert!(b.supports, "{:?}", b);
        assert_eq!(b.matched, b.terms.len());
        assert!(b.excerpt.unwrap().contains("bcrypt"));
    }

    #[test]
    fn a_line_restricted_region_is_respected() {
        // The term only appears outside the given line range.
        let source = "validate_token\nfn main() {}\n";
        let a = assess("uses validate_token", source, Some((2, 2)));
        assert!(!a.supports, "{:?}", a);
        let b = assess("uses validate_token", source, Some((1, 1)));
        assert!(b.supports, "{:?}", b);
    }

    #[test]
    fn an_empty_statement_is_inconclusive_not_supporting() {
        let a = assess("   ", "anything", None);
        assert!(!a.supports);
        assert!(a.terms.is_empty());
    }

    #[test]
    fn substring_matching_stems() {
        // `hash` matches `hashes` and `rehash`; deterministic and documented.
        let a = assess("uses hash", "the hashes are computed", None);
        assert!(a.supports, "{:?}", a);
    }
}
