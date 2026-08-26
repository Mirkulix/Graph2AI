//! Bounded graph-to-prompt context compiler.
//!
//! Turns a region of the knowledge graph into a compact block a worker can be
//! given before a task. Two rules shape every decision here:
//!
//! 1. **An unverified proposal never appears as an established fact.** Only
//!    `Observed` and `Verified` claims reach the `## Established` section.
//!    Proposals may be included, but under a heading that says what they are,
//!    and only when the caller asks for them.
//! 2. **The output is bounded and deterministic.** The same graph state and
//!    the same request produce byte-identical text, and the budget is a hard
//!    cap — what does not fit is dropped in a stated order, with a line saying
//!    how much was left out.
//!
//! ## Why a character budget
//!
//! Tokens are what actually cost money, but tokenization is model-specific and
//! not available here. Characters are a stable, conservative proxy: for the
//! locator-and-prose text this emits, a token is rarely fewer than three
//! characters, so a character budget never *under*-counts the token cost.
//! Callers who know their tokenizer can convert before calling.

use crate::model::{Claim, ClaimStatus, EntityId};
use crate::store::KnowledgeStore;
use crate::Error;

/// What to compile and how much of it.
#[derive(Debug, Clone)]
pub struct ContextRequest {
    /// The entity the task is about. Its claims come first.
    pub focus: EntityId,
    /// How far to walk the relation graph. 0 = the focus entity only.
    pub depth: u8,
    /// Hard cap on the rendered size, in characters.
    pub budget: usize,
    /// Include proposals under an explicit "unverified" heading.
    pub include_proposals: bool,
}

impl ContextRequest {
    /// A request with the defaults a planning agent should get: one hop out,
    /// roughly 2k characters, established facts only.
    pub fn about(focus: EntityId) -> Self {
        Self {
            focus,
            depth: 1,
            budget: 2000,
            include_proposals: false,
        }
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    pub fn with_depth(mut self, depth: u8) -> Self {
        self.depth = depth;
        self
    }

    /// Include proposals, clearly labelled as unverified.
    pub fn including_proposals(mut self) -> Self {
        self.include_proposals = true;
        self
    }
}

/// A compiled context block plus what it cost and what it left out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledContext {
    /// The text to put in the prompt. Empty when the graph knows nothing.
    pub text: String,
    /// Claims actually rendered.
    pub included: usize,
    /// Claims that matched but did not fit the budget.
    pub omitted: usize,
}

impl CompiledContext {
    pub fn is_empty(&self) -> bool {
        self.included == 0
    }
}

/// Compile a bounded context block for a task about `request.focus`.
pub fn compile_context(
    store: &KnowledgeStore,
    request: &ContextRequest,
) -> Result<CompiledContext, Error> {
    let established = collect(store, request, true)?;
    let proposals = if request.include_proposals {
        collect(store, request, false)?
    } else {
        Vec::new()
    };

    let mut out = String::new();
    let mut included = 0usize;
    let mut omitted = 0usize;

    // Reserve room for the truncation note up front. Without this the note is
    // dropped exactly when claims fill the budget — that is, precisely when
    // something *was* omitted and the reader most needs to be told.
    //
    // The reservation is capped at a quarter of the budget: on a very small
    // budget, spending it all on "things were omitted" would tell the reader
    // nothing but that. There, content wins and the note may be dropped.
    let total = established.len() + proposals.len();
    let reserve = truncation_note(total).len().min(request.budget / 4);
    let claim_budget = request.budget - reserve;

    // Established facts get the budget first: a proposal must never crowd out
    // something the graph actually knows.
    if !established.is_empty() {
        let header = "## Established\n";
        if header.len() <= claim_budget {
            out.push_str(header);
            let (n, skipped) = render_claims(&mut out, &established, claim_budget);
            included += n;
            omitted += skipped;
        } else {
            omitted += established.len();
        }
    }

    if !proposals.is_empty() {
        let header = "\n## Unverified proposals — do not treat as fact\n";
        if out.len() + header.len() <= claim_budget {
            out.push_str(header);
            let (n, skipped) = render_claims(&mut out, &proposals, claim_budget);
            included += n;
            omitted += skipped;
        } else {
            omitted += proposals.len();
        }
    }

    // Silence about a truncation would read as "this is everything".
    if omitted > 0 {
        let note = truncation_note(omitted);
        if out.len() + note.len() <= request.budget {
            out.push_str(&note);
        }
    }

    // Truncation is the failure mode that does not look like one: the worker
    // gets a plausible prompt that is quietly missing facts. Surface it at
    // warn, not debug.
    if omitted > 0 {
        tracing::warn!(
            focus = %request.focus,
            budget = request.budget,
            included,
            omitted,
            "context truncated — the worker did not see every relevant claim"
        );
    }

    tracing::debug!(
        focus = %request.focus,
        depth = request.depth,
        included,
        omitted,
        bytes = out.len(),
        proposals = request.include_proposals,
        "context compiled"
    );

    Ok(CompiledContext {
        text: out,
        included,
        omitted,
    })
}

/// The line that tells the reader something was left out.
///
/// Also used to size the reservation before rendering, so the note is
/// guaranteed to fit — hence taking the count rather than formatting inline.
fn truncation_note(count: usize) -> String {
    format!("\n({count} more claim(s) omitted for space)\n")
}

/// Gather claims in the neighbourhood of the focus entity.
///
/// `load_bearing` selects which half of the graph to return: established
/// facts, or proposals. They are collected separately so a proposal can never
/// be rendered under the established heading by accident.
fn collect(
    store: &KnowledgeStore,
    request: &ContextRequest,
    load_bearing: bool,
) -> Result<Vec<Claim>, Error> {
    let mut seen_entities = vec![request.focus.clone()];
    let mut frontier = vec![request.focus.clone()];

    for _ in 0..request.depth {
        let mut next = Vec::new();
        for entity in &frontier {
            for (_, neighbour, _) in store.neighbors(entity)? {
                if !seen_entities.contains(&neighbour) {
                    seen_entities.push(neighbour.clone());
                    next.push(neighbour);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    let mut claims: Vec<Claim> = Vec::new();
    for entity in &seen_entities {
        for claim in store.claims_about(entity)? {
            if claim.superseded_by.is_some() {
                continue;
            }
            if claim.is_load_bearing() != load_bearing {
                continue;
            }
            // A refuted claim is settled, not pending — it is neither an
            // established fact nor an open proposal.
            if claim.status == ClaimStatus::Refuted {
                continue;
            }
            if !claims.iter().any(|c| c.id == claim.id) {
                claims.push(claim);
            }
        }
    }

    // Deterministic order: strongest first, then newest, then by id so equal
    // claims never swap places between runs.
    claims.sort_by(|a, b| {
        let rank = |c: &Claim| match c.status {
            ClaimStatus::Verified => 0,
            ClaimStatus::Observed => 1,
            ClaimStatus::Proposed => 2,
            _ => 3,
        };
        rank(a)
            .cmp(&rank(b))
            .then(b.provenance.observed_at.cmp(&a.provenance.observed_at))
            .then(a.id.0.cmp(&b.id.0))
    });

    Ok(claims)
}

/// Append as many claims as fit. Returns (rendered, skipped).
fn render_claims(out: &mut String, claims: &[Claim], budget: usize) -> (usize, usize) {
    let mut rendered = 0;
    for (i, claim) in claims.iter().enumerate() {
        let line = render_claim(claim);
        if out.len() + line.len() > budget {
            return (rendered, claims.len() - i);
        }
        out.push_str(&line);
        rendered += 1;
    }
    (rendered, 0)
}

/// One claim as a single line: status, statement, relation and where to look.
///
/// The evidence locator is the load-bearing part — it is what lets a worker
/// check the claim instead of believing it.
fn render_claim(claim: &Claim) -> String {
    let mut line = format!("- [{}] {}", claim.status.as_str(), claim.statement);

    if let (Some(relation), Some(object)) = (claim.relation, claim.object.as_ref()) {
        line.push_str(&format!(" ({} {})", relation.as_str(), object.0));
    }

    if let Some(evidence) = claim.evidence.iter().find(|e| e.supports) {
        line.push_str(&format!(" [{}", evidence.locator));
        if let Some((start, end)) = evidence.lines {
            if start == end {
                line.push_str(&format!(":{start}"));
            } else {
                line.push_str(&format!(":{start}-{end}"));
            }
        }
        line.push(']');
    }

    line.push('\n');
    line
}
