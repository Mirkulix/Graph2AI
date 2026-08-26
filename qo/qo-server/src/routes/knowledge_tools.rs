//! MCP tool surface for the knowledge graph.
//!
//! Tools mirroring `qo-knowledge.md`:
//!   * `orbit_graph_search`       — find entities and backed claims
//!   * `orbit_graph_neighbors`    — traverse relations and impact
//!   * `orbit_graph_impact`       — what a change would reach
//!   * `orbit_graph_add_claim`    — record a claim as a *proposal*
//!   * `orbit_graph_verify_claim` — confirm or refute with evidence
//!   * `orbit_graph_context`      — compact, source-bound context for a task
//!   * `orbit_graph_commit_delta` — submit a batch of changes as OrbitQLang
//!   * `orbit_graph_swarm_state`  — what other sessions are doing
//!
//! Two rules are enforced here rather than left to the caller:
//!
//! 1. `add_claim` always writes `Proposed`. There is no argument that lets a
//!    caller declare its own claim observed or verified — promotion requires
//!    going through `verify_claim` with evidence.
//! 2. Every rendered claim is labelled with its status, and
//!    `orbit_graph_context` returns only load-bearing claims. A model reading
//!    the output cannot mistake a guess for a fact.
//!
//! Writes require an agent identity (`by`), which is stored as provenance and
//! written to the action log for audit.

use axum::extract::Query;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::{HashSet, VecDeque}, sync::Arc};

use crate::AppState;
use qo_knowledge::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance,
    Relation, Verdict,
};

/// Tool definitions appended to the MCP `tools/list` response.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "orbit_graph_search",
            "description": "Search the knowledge graph for claims. Every result carries its status (observed/proposed/verified/stale/refuted), provenance and evidence. Only 'observed' and 'verified' claims are reliable context; 'proposed' claims are unverified suggestions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Substring to match against claim statements." },
                    "limit": { "type": "integer", "description": "Maximum results (default 20)." }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "orbit_graph_neighbors",
            "description": "Traverse relations from an entity in both directions: what it points at, and what points at it. Use to answer dependency and impact questions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "description": "Entity kind: repository|file|symbol|service|endpoint|concept|agent|run" },
                    "name": { "type": "string", "description": "Entity name, e.g. a repo-relative file path." }
                },
                "required": ["kind", "name"]
            }
        }),
        json!({
            "name": "orbit_graph_impact",
            "description": "Traverse load-bearing dependency relations across multiple bounded hops. Use for impact analysis; proposed, stale and refuted claims are excluded.",
            "inputSchema": { "type": "object", "properties": {
                "kind": { "type": "string" }, "name": { "type": "string" },
                "depth": { "type": "integer", "description": "Traversal depth, 1-4; default 2." }
            }, "required": ["kind", "name"] }
        }),
        json!({
            "name": "orbit_graph_add_claim",
            "description": "Record a claim as a PROPOSAL. Proposals are never treated as truth and never appear in orbit_graph_context until confirmed via orbit_graph_verify_claim with evidence. Do not use this to assert something you already verified — record it, then verify it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Stable, unique id for this claim." },
                    "statement": { "type": "string", "description": "The claim in plain language." },
                    "subject_kind": { "type": "string", "description": "Entity kind of the subject." },
                    "subject_name": { "type": "string", "description": "Entity name of the subject." },
                    "relation": { "type": "string", "description": "Optional: defines|calls|depends_on|implements|contradicts|documents|tests|produces" },
                    "object_kind": { "type": "string", "description": "Optional: entity kind of the object." },
                    "object_name": { "type": "string", "description": "Optional: entity name of the object." },
                    "by": { "type": "string", "description": "Agent identity recording this claim." },
                    "git_revision": { "type": "string", "description": "Optional git revision this was observed against." },
                    "run_id": { "type": "string", "description": "Optional run or goal id." }
                },
                "required": ["id", "statement", "subject_kind", "subject_name", "by"]
            }
        }),
        json!({
            "name": "orbit_graph_verify_claim",
            "description": "Confirm or refute a claim with reproducible evidence. Set supports=true to verify, supports=false to refute. Verifying with counter-evidence (or refuting with supporting evidence) is rejected.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The claim id to advance." },
                    "supports": { "type": "boolean", "description": "true = verify, false = refute." },
                    "evidence_kind": { "type": "string", "description": "source|commit|test_run|tool_run|external" },
                    "locator": { "type": "string", "description": "What to look at: file path, URL, command or run id." },
                    "line_start": { "type": "integer" },
                    "line_end": { "type": "integer" },
                    "excerpt": { "type": "string", "description": "Short verbatim excerpt, when useful." },
                    "by": { "type": "string", "description": "Agent identity performing the check." }
                },
                "required": ["id", "supports", "evidence_kind", "locator", "by"]
            }
        }),
        json!({
            "name": "orbit_graph_context",
            "description": "Compact, source-bound context for a task. Returns ONLY load-bearing claims (observed or verified), newest and most-confirmed first, each with its evidence. Proposals are deliberately excluded.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "description": "Entity kind." },
                    "name": { "type": "string", "description": "Entity name." },
                    "limit": { "type": "integer", "description": "Maximum claims (default 10)." }
                },
                "required": ["kind", "name"]
            }
        }),
        json!({
            "name": "orbit_graph_commit_delta",
            "description": "Submit a batch of graph changes written in OrbitQLang. This is the way a worker session hands its findings back: entities, proposed claims, relations and evidence in one validated submission. The document MUST carry a SIG line from a producer key this server trusts, or it is refused — produce one with `qlang graph sign`. Each delta id may be submitted once per producer; a replay is refused rather than silently re-applied. Claims arrive as PROPOSALS regardless of what the document says. Returns per-operation outcomes, including conflicts when another session already decided otherwise.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "OrbitQLang document. One operation per line, '|' separated, no nesting. The SIG line is required — produce it with `qlang graph sign`. Example:\nDELTA|1|d-42\nBY|worker-3|1700000000|abc123\nSIG|ed25519|k1|<128 hex chars>\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|auth uses bcrypt\n+R|c1|depends_on|file:Cargo.toml\nOK|c1|source|src/auth.rs|42:42|use bcrypt::hash;"
                    }
                },
                "required": ["document"]
            }
        }),
        json!({
            "name": "orbit_graph_swarm_state",
            "description": "What the other agent sessions are doing right now: registered identities, their last activity, and how much of the knowledge graph is currently load-bearing. Use before starting work to avoid duplicating another session's task.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "orbit_graph_verify_source",
            "description": "Check a proposed claim against its actual source file and promote it to 'verified' ONLY when every distinctive term of the statement appears verbatim in that file. This is the deterministic, lexical counterpart to orbit_graph_verify_claim: it reads the file itself instead of trusting a caller-supplied excerpt. It confirms, never disproves — a partial match leaves the claim proposed — and it never re-promotes a settled claim. The source path is resolved inside the workspace root; escapes are refused.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The proposed claim id to check against source." },
                    "by": { "type": "string", "description": "Agent identity performing the check." }
                },
                "required": ["id", "by"]
            }
        }),
        json!({
            "name": "orbit_graph_receipt",
            "description": "Render a proof receipt for one claim: its current status, every revision it went through (who decided what, when), the evidence that promoted it, and other claims about the same subject — including disagreements, which are kept and never overwritten. Use this to answer 'why should I believe this?' with a deterministic, evidence-backed answer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The claim id to prove." }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "orbit_graph_verify_all",
            "description": "Sweep every open (proposed) claim against its source file and promote each one the code literally substantiates, in one deterministic pass. Returns how many were verified, inconclusive (left proposed) and unavailable (missing source or unsafe path). This is the harvest step after workers have proposed claims.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "by": { "type": "string", "description": "Agent identity performing the sweep." }
                },
                "required": ["by"]
            }
        }),
        json!({
            "name": "orbit_graph_refresh_sources",
            "description": "Re-check every settled (verified/observed) claim against its source and mark each one stale whose recorded excerpt no longer appears in the file. Deterministic (compares the exact captured excerpt, never a timestamp). Claims without a verbatim excerpt are skipped. This is how the graph notices that the code moved on and a fact has rotted.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "by": { "type": "string", "description": "Agent identity performing the refresh." }
                },
                "required": ["by"]
            }
        }),
        json!({
            "name": "orbit_graph_divergences",
            "description": "List every subject where the graph holds BOTH a load-bearing (verified/observed) claim and a refuted claim — i.e. where agents settled in opposite directions. Both sides are shown with their statements; the report never asserts a contradiction, it surfaces the disagreement for a human to judge.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "orbit_graph_heal_stale",
            "description": "Re-verify every stale claim against its current source and heal the ones whose statement is still literally substantiated (the code moved, the fact did not). A healed claim is promoted back to verified with fresh evidence; a genuinely rotted claim stays stale. The full verified -> stale -> verified trail is kept.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "by": { "type": "string", "description": "Agent identity performing the heal." }
                },
                "required": ["by"]
            }
        }),
        json!({
            "name": "orbit_graph_health",
            "description": "A one-block summary of the knowledge graph's health: how many claims are load-bearing (verified/observed), how many are open proposals, stale, refuted, how many subjects hold a disagreement, and how many entities exist.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "orbit_graph_proposal_prompt",
            "description": "Return the exact grammar and constraints a model must follow so its output is admissible by orbit_graph_propose. Use this before producing a proposal document, so the text passes the deterministic admission gate on the first try.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "orbit_graph_propose",
            "description": "Submit an OrbitQLang document as PROPOSALS through the deterministic admission gate (the text-to-graph pipeline's server half): it parses recoveringly, refuses OK/NO (a model may not verify or refute), requires self-contained entity/claim references, caps statement length, and merges every admitted claim as 'proposed'. No signature is required — this is the unsigned proposal path; use orbit_graph_commit_delta for signed submissions. Produce the document with orbit_graph_proposal_prompt.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "document": { "type": "string", "description": "OrbitQLang document. Example:\nDELTA|1|d-42\nBY|worker-3|1700000000\n+E|file|src/auth.rs\n+C|c1|file:src/auth.rs|auth hashes passwords with bcrypt\n+R|c1|depends_on|file:Cargo.toml" }
                },
                "required": ["document"]
            }
        }),
    ]
}

/// True if `name` is one of this module's tools.
pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "orbit_graph_search"
            | "orbit_graph_neighbors"
            | "orbit_graph_impact"
            | "orbit_graph_add_claim"
            | "orbit_graph_verify_claim"
            | "orbit_graph_context"
            | "orbit_graph_commit_delta"
            | "orbit_graph_swarm_state"
            | "orbit_graph_verify_source"
            | "orbit_graph_receipt"
            | "orbit_graph_verify_all"
            | "orbit_graph_refresh_sources"
            | "orbit_graph_divergences"
            | "orbit_graph_heal_stale"
            | "orbit_graph_health"
            | "orbit_graph_proposal_prompt"
            | "orbit_graph_propose"
    )
}

type ToolResult = Result<String, (i32, String)>;

const INVALID_PARAMS: i32 = -32602;
const INTERNAL: i32 = -32603;
/// A role forbade the tool call (e.g. a viewer seat trying to write). Surfaced
/// as an RPC error, not a silent no-op.
const FORBIDDEN: i32 = -32001;

fn missing(field: &str) -> (i32, String) {
    (INVALID_PARAMS, format!("Missing '{}' argument", field))
}

fn str_arg(args: &Value, field: &str) -> Result<String, (i32, String)> {
    args.get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| missing(field))
}

fn opt_str(args: &Value, field: &str) -> Option<String> {
    args.get(field)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

fn entity_arg(args: &Value, kind_field: &str, name_field: &str) -> Result<EntityId, (i32, String)> {
    let kind_raw = str_arg(args, kind_field)?;
    let kind = EntityKind::parse(&kind_raw).ok_or((
        INVALID_PARAMS,
        format!(
            "Unknown entity kind '{}'. Expected one of: repository, file, symbol, service, endpoint, concept, agent, run",
            kind_raw
        ),
    ))?;
    let name = str_arg(args, name_field)?;
    Ok(EntityId::derive(kind, &name))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Render a claim with its status spelled out, so a model reading the text
/// cannot mistake a proposal for a fact.
fn render(c: &Claim) -> String {
    let mut s = String::new();
    let caveat = if c.status.is_load_bearing() {
        ""
    } else {
        " (NOT reliable context)"
    };
    s.push_str(&format!(
        "[{}{}] {} — {}\n",
        c.status.as_str().to_uppercase(),
        caveat,
        c.id,
        c.statement
    ));
    s.push_str(&format!("    subject: {}\n", c.subject));
    if let (Some(r), Some(o)) = (c.relation, &c.object) {
        s.push_str(&format!("    relation: {} -> {}\n", r.as_str(), o));
    }
    s.push_str(&format!(
        "    by: {} at {} (rev {}){}\n",
        c.provenance.producer,
        c.provenance.observed_at,
        c.revision,
        c.provenance
            .git_revision
            .as_ref()
            .map(|g| format!(" @{}", g))
            .unwrap_or_default()
    ));
    for e in &c.evidence {
        let lines = e
            .lines
            .map(|(a, b)| format!(":{}-{}", a, b))
            .unwrap_or_default();
        s.push_str(&format!(
            "    evidence[{}]: {}{}{}\n",
            if e.supports { "supports" } else { "contradicts" },
            e.locator,
            lines,
            e.excerpt
                .as_ref()
                .map(|x| format!(" — {:?}", x))
                .unwrap_or_default()
        ));
    }
    s
}

fn render_all(claims: &[Claim], empty: &str) -> String {
    if claims.is_empty() {
        return empty.to_string();
    }
    claims.iter().map(render).collect::<Vec<_>>().join("\n")
}

/// Render the result of a deterministic source check for a model to read.
fn render_source_check(id: &str, check: &qo_knowledge::SourceCheck) -> String {
    match &check.verdict {
        Verdict::Verified => {
            let terms = check.terms.join(", ");
            let excerpt = check
                .evidence
                .as_ref()
                .and_then(|e| e.excerpt.as_deref())
                .unwrap_or("(no excerpt)");
            format!(
                "Claim {id} VERIFIED against source. All {} distinctive term(s) present: {terms}.\nEvidence: {excerpt}",
                check.matched
            )
        }
        Verdict::Inconclusive { reason } => {
            format!("Claim {id} stays PROPOSED. Inconclusive: {reason}")
        }
        Verdict::NotProposed { status } => {
            format!(
                "Claim {id} is {}; only a proposed claim is checked against source.",
                status.as_str()
            )
        }
        Verdict::Unavailable { reason } => {
            format!("Claim {id} could not be checked: {reason}")
        }
    }
}

/// The knowledge tools that mutate graph state. Every other tool is read-only
/// and reachable by a viewer seat; these require a write-capable role.
fn requires_write(name: &str) -> bool {
    matches!(
        name,
        "orbit_graph_add_claim"
            | "orbit_graph_verify_claim"
            | "orbit_graph_commit_delta"
            | "orbit_graph_verify_source"
            | "orbit_graph_verify_all"
            | "orbit_graph_refresh_sources"
            | "orbit_graph_heal_stale"
            | "orbit_graph_propose"
    )
}

/// Dispatch one of the `orbit_graph_*` tools.
///
/// `principal` is the authenticated seat; write tools are refused for a
/// `viewer` before any work happens.
pub async fn call(
    state: Arc<AppState>,
    principal: &crate::api_keys::Principal,
    name: &str,
    args: Value,
) -> ToolResult {
    if requires_write(name) && !principal.role.can_write() {
        return Err((
            FORBIDDEN,
            "this seat is read-only (viewer); it cannot change graph state".into(),
        ));
    }
    match name {
        "orbit_graph_search" => {
            let query = str_arg(&args, "query")?;
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(20)
                .clamp(1, 200) as usize;

            let hits = state
                .knowledge
                .search(&query, limit)
                .map_err(|e| (INTERNAL, e.to_string()))?;

            Ok(render_all(
                &hits,
                &format!("No claims matching {:?}.", query),
            ))
        }

        "orbit_graph_neighbors" => {
            let entity = entity_arg(&args, "kind", "name")?;
            let neighbors = state
                .knowledge
                .neighbors(&entity)
                .map_err(|e| (INTERNAL, e.to_string()))?;

            if neighbors.is_empty() {
                return Ok(format!("No relations recorded for {}.", entity));
            }
            let mut out = format!("Relations for {}:\n\n", entity);
            for (rel, other, claim) in &neighbors {
                let caveat = if claim.status.is_load_bearing() {
                    ""
                } else {
                    " (NOT reliable context)"
                };
                out.push_str(&format!(
                    "  {} -> {}  [{}{}] via {}\n",
                    rel.as_str(),
                    other,
                    claim.status.as_str(),
                    caveat,
                    claim.id
                ));
            }
            Ok(out)
        }

        "orbit_graph_impact" => {
            let start = entity_arg(&args, "kind", "name")?;
            let max_depth = args.get("depth").and_then(|v| v.as_u64()).unwrap_or(2).clamp(1, 4) as usize;
            let mut queue = VecDeque::from([(start.clone(), 0usize)]);
            let mut seen = HashSet::from([start.clone()]);
            let mut lines = Vec::new();
            while let Some((entity, depth)) = queue.pop_front() {
                if depth >= max_depth || seen.len() > 200 { continue; }
                for (relation, other, claim) in state.knowledge.neighbors(&entity).map_err(|e| (INTERNAL, e.to_string()))? {
                    if !claim.status.is_load_bearing() || !seen.insert(other.clone()) { continue; }
                    lines.push(format!("{} --{}--> {} [via {}]", entity, relation.as_str(), other, claim.id));
                    queue.push_back((other, depth + 1));
                }
            }
            Ok(if lines.is_empty() { format!("No load-bearing impact relations recorded for {start}.") } else { format!("Impact from {start} (depth ≤ {max_depth}):\n{}", lines.join("\n")) })
        }

        "orbit_graph_add_claim" => {
            let id = str_arg(&args, "id")?;
            let statement = str_arg(&args, "statement")?;
            let subject = entity_arg(&args, "subject_kind", "subject_name")?;
            let by = str_arg(&args, "by")?;

            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };

            // Always a proposal. A caller cannot declare its own claim true.
            let mut claim = Claim::proposed(id.clone(), statement.clone(), subject.clone(), provenance);

            if let Some(rel_raw) = opt_str(&args, "relation") {
                let rel = Relation::parse(&rel_raw).ok_or((
                    INVALID_PARAMS,
                    format!(
                        "Unknown relation '{}'. Expected one of: defines, calls, depends_on, implements, contradicts, documents, tests, produces",
                        rel_raw
                    ),
                ))?;
                let object = entity_arg(&args, "object_kind", "object_name")?;
                register_entity(&state, &object);
                claim = claim.relating(rel, object);
            }

            state
                .knowledge
                .add_claim(&claim)
                .map_err(|e| (INVALID_PARAMS, e.to_string()))?;
            register_entity(&state, &subject);

            let _ = state
                .store
                .log_action("knowledge_claim_proposed", &statement, &by);

            Ok(format!(
                "Claim {} recorded as PROPOSED. It will not appear in orbit_graph_context \
                 until confirmed via orbit_graph_verify_claim with evidence.",
                id
            ))
        }

        "orbit_graph_verify_claim" => {
            let id = str_arg(&args, "id")?;
            let by = str_arg(&args, "by")?;
            let supports = args
                .get("supports")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| missing("supports"))?;

            let kind_raw = str_arg(&args, "evidence_kind")?;
            let kind = match kind_raw.as_str() {
                "source" => EvidenceKind::Source,
                "commit" => EvidenceKind::Commit,
                "test_run" => EvidenceKind::TestRun,
                "tool_run" => EvidenceKind::ToolRun,
                "external" => EvidenceKind::External,
                other => {
                    return Err((
                        INVALID_PARAMS,
                        format!(
                            "Unknown evidence kind '{}'. Expected one of: source, commit, test_run, tool_run, external",
                            other
                        ),
                    ))
                }
            };

            let lines = match (
                args.get("line_start").and_then(|v| v.as_u64()),
                args.get("line_end").and_then(|v| v.as_u64()),
            ) {
                (Some(a), Some(b)) => Some((a as u32, b as u32)),
                _ => None,
            };

            let evidence = Evidence {
                kind,
                locator: str_arg(&args, "locator")?,
                lines,
                excerpt: opt_str(&args, "excerpt"),
                supports,
            };

            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };

            let claim_id = ClaimId(id.clone());
            let updated = if supports {
                state.knowledge.verify_claim(&claim_id, evidence, provenance)
            } else {
                state.knowledge.refute_claim(&claim_id, evidence, provenance)
            }
            .map_err(|e| (INVALID_PARAMS, e.to_string()))?;

            let _ = state.store.log_action(
                if supports {
                    "knowledge_claim_verified"
                } else {
                    "knowledge_claim_refuted"
                },
                &updated.statement,
                &by,
            );

            Ok(format!(
                "Claim {} is now {} (revision {}). Previous revisions are kept.\n\n{}",
                id,
                updated.status.as_str().to_uppercase(),
                updated.revision,
                render(&updated)
            ))
        }

        "orbit_graph_verify_source" => {
            let id = str_arg(&args, "id")?;
            let by = str_arg(&args, "by")?;

            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };

            let check = qo_knowledge::verify_claim_against_source(
                &state.knowledge,
                &ClaimId(id.clone()),
                &state.workspace_root,
                provenance,
            )
            .map_err(|e| (INVALID_PARAMS, e.to_string()))?;

            let _ = state.store.log_action(
                "knowledge_claim_source_verified",
                &format!("claim {id}: {:?}", check.verdict),
                &by,
            );

            Ok(render_source_check(&id, &check))
        }

        "orbit_graph_receipt" => {
            let id = str_arg(&args, "id")?;
            let receipt = qo_knowledge::build_receipt(&state.knowledge, &ClaimId(id.clone()))
                .map_err(|e| (INVALID_PARAMS, e.to_string()))?;
            Ok(receipt.render())
        }

        "orbit_graph_verify_all" => {
            let by = str_arg(&args, "by")?;
            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };
            let report = qo_knowledge::verify_all_proposals(
                &state.knowledge,
                &state.workspace_root,
                provenance,
            )
            .map_err(|e| (INTERNAL, e.to_string()))?;
            let _ = state.store.log_action(
                "knowledge_sweep",
                &format!("verified {} of {}", report.verified, report.checked),
                &by,
            );
            Ok(report.render())
        }

        "orbit_graph_refresh_sources" => {
            let by = str_arg(&args, "by")?;
            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };
            let report = qo_knowledge::refresh_sources(
                &state.knowledge,
                &state.workspace_root,
                provenance,
            )
            .map_err(|e| (INTERNAL, e.to_string()))?;
            let _ = state.store.log_action(
                "knowledge_refresh_sources",
                &format!("{} stale of {} checked", report.stale, report.checked),
                &by,
            );
            Ok(report.render())
        }

        "orbit_graph_divergences" => {
            let report = qo_knowledge::divergences(&state.knowledge)
                .map_err(|e| (INTERNAL, e.to_string()))?;
            Ok(report.render())
        }

        "orbit_graph_heal_stale" => {
            let by = str_arg(&args, "by")?;
            let provenance = Provenance {
                producer: by.clone(),
                observed_at: now_secs(),
                git_revision: opt_str(&args, "git_revision"),
                run_id: opt_str(&args, "run_id"),
            };
            let report = qo_knowledge::heal_stale(
                &state.knowledge,
                &state.workspace_root,
                provenance,
            )
            .map_err(|e| (INTERNAL, e.to_string()))?;
            let _ = state.store.log_action(
                "knowledge_heal_stale",
                &format!("healed {} of {}", report.healed, report.examined),
                &by,
            );
            Ok(report.render())
        }

        "orbit_graph_health" => {
            let health = qo_knowledge::health(&state.knowledge)
                .map_err(|e| (INTERNAL, e.to_string()))?;
            Ok(health.render())
        }

        "orbit_graph_proposal_prompt" => Ok(qo_knowledge::proposal_system_prompt()),

        "orbit_graph_propose" => {
            let document = args
                .get("document")
                .and_then(|v| v.as_str())
                .ok_or((INVALID_PARAMS, "missing 'document'".to_string()))?;

            // Known entities from the graph satisfy the self-containment rule,
            // so a proposal may reference existing entities without re-declaring
            // them.
            let known: Vec<EntityId> = state
                .knowledge
                .list_entities()
                .unwrap_or_default()
                .into_iter()
                .map(|e| e.id)
                .collect();
            let policy = qo_knowledge::ProposalPolicy::default().with_known_entities(known);

            let outcome = qo_knowledge::propose_from_text(document, &policy);
            if !outcome.is_ok() {
                let detail = outcome
                    .violations
                    .iter()
                    .map(|v| format!("  {v}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(format!(
                    "Rejected: {} violation(s). Nothing was written.\n{detail}",
                    outcome.violations.len()
                ));
            }
            let delta = outcome.delta.expect("is_ok implies a delta");
            let report = qo_knowledge::merge_delta(&state.knowledge, &delta)
                .map_err(|e| (INTERNAL, e.to_string()))?;

            for op in &delta.operations {
                if let qo_knowledge::GraphDeltaOp::AddEntity { entity } = op {
                    register_entity(&state, &entity.id);
                }
            }
            let _ = state.store.log_action(
                "orbit_graph_propose",
                &format!("delta {} proposed: {} applied", delta.id, report.applied()),
                &delta.producer.id,
            );
            Ok(format!(
                "Proposed {} operation(s) from delta {} (all claims are `proposed`), {} conflict(s).\nPromote them with orbit_graph_verify_source or orbit_graph_verify_claim.",
                report.applied(),
                delta.id,
                report.conflicts().len()
            ))
        }

        "orbit_graph_context" => {
            let entity = entity_arg(&args, "kind", "name")?;
            let limit = args
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(10)
                .clamp(1, 100) as usize;

            let claims = state
                .knowledge
                .load_bearing_context(&entity, limit)
                .map_err(|e| (INTERNAL, e.to_string()))?;

            if claims.is_empty() {
                // Say why it is empty — there may be proposals that simply do
                // not qualify, and hiding that would be misleading.
                let total = state
                    .knowledge
                    .claims_about(&entity)
                    .map(|c| c.len())
                    .unwrap_or(0);
                return Ok(if total == 0 {
                    format!("No claims recorded for {}.", entity)
                } else {
                    format!(
                        "No load-bearing claims for {}. {} claim(s) exist but are \
                         proposed, stale or refuted — use orbit_graph_search to inspect them.",
                        entity, total
                    )
                });
            }

            Ok(format!(
                "Load-bearing claims for {} (observed or verified only):\n\n{}",
                entity,
                render_all(&claims, "")
            ))
        }

        "orbit_graph_commit_delta" => {
            let document = args
                .get("document")
                .and_then(|v| v.as_str())
                .ok_or((INVALID_PARAMS, "missing 'document'".to_string()))?;

            // Parse in recovering mode so a worker gets every syntax problem
            // at once instead of one per round-trip.
            let outcome = qo_knowledge::parse_recovering(document);
            if !outcome.errors.is_empty() {
                let detail = outcome
                    .errors
                    .iter()
                    .map(|e| format!("  {e}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                return Ok(format!(
                    "Rejected: the document has {} syntax error(s). Nothing was written.\n{}",
                    outcome.errors.len(),
                    detail
                ));
            }
            let Some(delta) = outcome.delta else {
                return Ok("Rejected: no DELTA header found. Nothing was written.".to_string());
            };

            let Some(now) = now_unix() else {
                return Err((
                    INTERNAL,
                    "system clock is before the Unix epoch; refusing to judge key validity"
                        .to_string(),
                ));
            };
            let report = qo_knowledge::merge_signed_delta(
                &state.knowledge,
                &state.delta_trust,
                &delta,
                now,
            )
            .map_err(|e| (INVALID_PARAMS, e.to_string()))?;

            // Entities named by the delta should show up in listings.
            for op in &delta.operations {
                if let qo_knowledge::GraphDeltaOp::AddEntity { entity } = op {
                    register_entity(&state, &entity.id);
                }
            }

            let _ = state.store.log_action(
                "orbit_graph_commit_delta",
                &format!(
                    "delta {} merged: {} applied, {} conflict(s)",
                    delta.id,
                    report.applied(),
                    report.conflicts().len()
                ),
                &delta.producer.id,
            );

            let mut out = format!(
                "Delta {} merged: {} applied, {} already present, {} conflict(s).\n",
                report.delta_id,
                report.applied(),
                report.already_applied(),
                report.conflicts().len()
            );
            if report.is_clean() {
                out.push_str(
                    "\nAll claims were recorded as PROPOSALS. \
                     Use orbit_graph_verify_claim with evidence to make one load-bearing.",
                );
            } else {
                out.push_str("\nConflicts (nothing was overwritten):\n");
                for conflict in report.conflicts() {
                    out.push_str(&format!("  [{:?}] {}\n", conflict.kind, conflict.detail));
                }
            }
            Ok(out)
        }

        "orbit_graph_swarm_state" => {
            let sessions: Vec<String> = {
                let presence = state.presence.lock().await;
                let mut rows: Vec<_> = presence.values().collect();
                rows.sort_by(|a, b| a.identity.cmp(&b.identity));
                rows.iter()
                    .map(|entry| {
                        format!(
                            "  {} ({}) — last seen {}",
                            entry.identity,
                            entry.ide_name.as_deref().unwrap_or("unknown ide"),
                            entry.last_seen_at
                        )
                    })
                    .collect()
            };

            let count = |s: ClaimStatus| {
                state
                    .knowledge
                    .claims_with_status(s)
                    .map(|c| c.len())
                    .unwrap_or(0)
            };
            let verified = count(ClaimStatus::Verified);
            let observed = count(ClaimStatus::Observed);
            let proposed = count(ClaimStatus::Proposed);

            let mut out = String::new();
            if sessions.is_empty() {
                out.push_str("No other agent sessions are registered.\n");
            } else {
                out.push_str(&format!("Active sessions ({}):\n", sessions.len()));
                out.push_str(&sessions.join("\n"));
                out.push('\n');
            }
            out.push_str(&format!(
                "\nKnowledge graph: {} load-bearing ({} verified, {} observed), \
                 {} unverified proposal(s).",
                verified + observed,
                verified,
                observed,
                proposed
            ));
            Ok(out)
        }

        other => Err((-32601, format!("Unknown tool: {}", other))),
    }
}

/// The receiver's clock, for key-validity decisions.
///
/// Never the delta's own `emitted_at`: a submitter controls that field and
/// could backdate it to a moment before their key was revoked.
///
/// Returns `None` when the system clock is before the Unix epoch. Falling back
/// to `0` there would be fail-open: `0` precedes every `revoked_at`, so a
/// revoked key would start working again on a machine with a broken clock.
/// A caller that cannot establish the time must refuse, not guess.
fn now_unix() -> Option<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Best-effort entity registration so `list_entities` stays useful. A failure
/// here must not fail the claim write that triggered it.
fn register_entity(state: &Arc<AppState>, id: &EntityId) {
    // EntityId is "kind:name"; recover both halves.
    let Some((kind_raw, name)) = id.0.split_once(':') else {
        return;
    };
    let Some(kind) = EntityKind::parse(kind_raw) else {
        return;
    };
    let _ = state.knowledge.put_entity(&Entity {
        id: id.clone(),
        kind,
        name: name.to_string(),
    });
}

/// `GET /api/knowledge/stats` — how much of the graph is actually backed.
///
/// `load_bearing` is the number the cockpit should show prominently: it is
/// what an agent may rely on without a caveat.
pub async fn knowledge_stats(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<Value> {
    let count = |s: ClaimStatus| {
        state
            .knowledge
            .claims_with_status(s)
            .map(|c| c.len())
            .unwrap_or(0)
    };
    let verified = count(ClaimStatus::Verified);
    let observed = count(ClaimStatus::Observed);
    let proposed = count(ClaimStatus::Proposed);
    let stale = count(ClaimStatus::Stale);
    let refuted = count(ClaimStatus::Refuted);

    axum::Json(json!({
        "verified": verified,
        "observed": observed,
        "proposed": proposed,
        "stale": stale,
        "refuted": refuted,
        "load_bearing": verified + observed,
        "total": verified + observed + proposed + stale + refuted,
        "entities": state.knowledge.list_entities().map(|e| e.len()).unwrap_or(0),
    }))
}

/// Query bounds for the cockpit snapshot. The endpoint is read-only and only
/// returns latest claim revisions, never the append-only audit history.
#[derive(Deserialize)]
pub struct KnowledgeSnapshotQuery {
    pub limit: Option<usize>,
}

/// `GET /api/knowledge/snapshot` — bounded, render-ready graph data for the
/// QO cockpit. This keeps the frontend on the HTTP API rather than making it
/// speak MCP JSON-RPC directly.
pub async fn knowledge_snapshot(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Query(query): Query<KnowledgeSnapshotQuery>,
) -> axum::Json<Value> {
    let limit = query.limit.unwrap_or(100).clamp(1, 500);
    let statuses = [
        ClaimStatus::Verified,
        ClaimStatus::Observed,
        ClaimStatus::Proposed,
        ClaimStatus::Stale,
        ClaimStatus::Refuted,
    ];
    let mut claims = statuses
        .into_iter()
        .flat_map(|status| state.knowledge.claims_with_status(status).unwrap_or_default())
        .collect::<Vec<_>>();
    claims.sort_by(|a, b| b.provenance.observed_at.cmp(&a.provenance.observed_at));
    claims.truncate(limit);

    axum::Json(json!({
        "entities": state.knowledge.list_entities().unwrap_or_default(),
        "claims": claims,
    }))
}

// ---------------------------------------------------------------------------
// Delta log — the cockpit's live feed and conflict view
// ---------------------------------------------------------------------------

/// Cap on the in-memory delta log. Old entries fall off the front; the merges
/// themselves stay durable in redb, only the reports are bounded.
const DELTA_LOG_CAP: usize = 200;

/// One merged delta, kept for display.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DeltaLogEntry {
    pub delta_id: String,
    pub producer: String,
    /// Unix seconds, taken from the delta's own producer stamp so the feed
    /// reflects when the work happened, not when it arrived.
    pub emitted_at: u64,
    pub applied: usize,
    pub already_applied: usize,
    pub conflicts: Vec<qo_knowledge::Conflict>,
    /// The submitted document, so a reviewer can see exactly what was sent.
    pub document: String,
}

async fn push_delta_log(state: &Arc<AppState>, entry: DeltaLogEntry) {
    let mut log = state.delta_log.write().await;
    if log.len() >= DELTA_LOG_CAP {
        log.pop_front();
    }
    log.push_back(entry);
}

#[derive(Deserialize)]
pub struct CommitDeltaRequest {
    /// An OrbitQLang document.
    pub document: String,
}

/// `POST /api/knowledge/delta` — parse, validate and merge an OrbitQLang
/// document, then record the report for the cockpit feed.
///
/// A syntactically bad document is rejected whole with every error listed;
/// nothing is written. Per-operation disagreements come back as conflicts
/// with a 200, because the merge itself succeeded — the graph simply refused
/// to overwrite what another session decided.
pub async fn commit_delta(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<CommitDeltaRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let outcome = qo_knowledge::parse_recovering(&req.document);
    if !outcome.errors.is_empty() {
        let errors: Vec<Value> = outcome
            .errors
            .iter()
            .map(|e| json!({ "line": e.line, "message": e.message }))
            .collect();
        return (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({ "error": "parse failed", "errors": errors })),
        );
    }
    let Some(delta) = outcome.delta else {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({ "error": "no DELTA header" })),
        );
    };

    let Some(now) = now_unix() else {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({
                "error": "system clock is before the Unix epoch; refusing to judge key validity"
            })),
        );
    };

    let report = match qo_knowledge::merge_signed_delta(
        &state.knowledge,
        &state.delta_trust,
        &delta,
        now,
    ) {
        Ok(r) => r,
        Err(qo_knowledge::SubmitError::Untrusted(e)) => {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(json!({ "error": e.to_string() })),
            )
        }
        Err(e @ qo_knowledge::SubmitError::Replay { .. }) => {
            return (
                axum::http::StatusCode::CONFLICT,
                axum::Json(json!({ "error": e.to_string() })),
            )
        }
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({ "error": e.to_string() })),
            )
        }
    };

    for op in &delta.operations {
        if let qo_knowledge::GraphDeltaOp::AddEntity { entity } = op {
            register_entity(&state, &entity.id);
        }
    }

    let conflicts: Vec<qo_knowledge::Conflict> =
        report.conflicts().into_iter().cloned().collect();

    push_delta_log(
        &state,
        DeltaLogEntry {
            delta_id: report.delta_id.clone(),
            producer: delta.producer.id.clone(),
            emitted_at: delta.producer.emitted_at,
            applied: report.applied(),
            already_applied: report.already_applied(),
            conflicts: conflicts.clone(),
            document: req.document.clone(),
        },
    )
    .await;

    let _ = state.store.log_action(
        "knowledge_delta_merged",
        &format!(
            "delta {} merged: {} applied, {} conflict(s)",
            report.delta_id,
            report.applied(),
            conflicts.len()
        ),
        &delta.producer.id,
    );

    (
        axum::http::StatusCode::OK,
        axum::Json(json!({
            "delta_id": report.delta_id,
            "applied": report.applied(),
            "already_applied": report.already_applied(),
            "conflicts": conflicts,
        })),
    )
}

#[derive(Deserialize)]
pub struct DeltaLogQuery {
    pub limit: Option<usize>,
    /// When true, return only entries that carry at least one conflict —
    /// this is what the conflict view asks for.
    pub conflicts_only: Option<bool>,
}

/// `GET /api/knowledge/deltas` — the cockpit's live delta feed, newest first.
pub async fn delta_log(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Query(query): Query<DeltaLogQuery>,
) -> axum::Json<Value> {
    let limit = query.limit.unwrap_or(50).clamp(1, DELTA_LOG_CAP);
    let conflicts_only = query.conflicts_only.unwrap_or(false);

    let log = state.delta_log.read().await;
    let entries: Vec<&DeltaLogEntry> = log
        .iter()
        .rev()
        .filter(|e| !conflicts_only || !e.conflicts.is_empty())
        .take(limit)
        .collect();

    axum::Json(json!({
        "deltas": entries,
        "total": log.len(),
        "unresolved_conflicts": log.iter().map(|e| e.conflicts.len()).sum::<usize>(),
    }))
}

/// Index only QO's configured workspace; callers cannot choose a path.
pub async fn index_workspace(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<crate::repository_indexer::IndexReport> {
    axum::Json(crate::repository_indexer::index_repository(&state.workspace_root, &state.knowledge))
}

#[derive(Deserialize)]
pub struct VerifySourceRequest {
    /// The proposed claim id to check against its source file.
    pub id: String,
    /// Agent identity performing the check.
    pub by: String,
}

/// `POST /api/knowledge/verify-source` — check a proposed claim against its
/// actual source file and promote it only when the code literally
/// substantiates every distinctive term.
///
/// This is the deterministic counterpart to a signed `OK` delta: it reads the
/// file itself instead of trusting a caller-supplied excerpt, so the graph —
/// not the caller — decides what the source actually says.
pub async fn verify_source(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<VerifySourceRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let provenance = Provenance {
        producer: req.by.clone(),
        observed_at: now_secs(),
        git_revision: None,
        run_id: None,
    };

    match qo_knowledge::verify_claim_against_source(
        &state.knowledge,
        &ClaimId(req.id.clone()),
        &state.workspace_root,
        provenance,
    ) {
        Ok(check) => {
            let _ = state.store.log_action(
                "knowledge_claim_source_verified",
                &format!("claim {}: {:?}", req.id, check.verdict),
                &req.by,
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "claim_id": req.id,
                    "verdict": check.verdict,
                    "terms": check.terms,
                    "matched": check.matched,
                    "evidence": check.evidence,
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

#[derive(Deserialize)]
pub struct ReceiptQuery {
    pub claim_id: String,
}

/// `GET /api/knowledge/receipt?claim_id=...` — a proof receipt for one claim:
/// its status, every revision, its evidence and the related claims (including
/// disagreements). Rendered deterministically and bounded.
pub async fn knowledge_receipt(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Query(query): Query<ReceiptQuery>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    match qo_knowledge::build_receipt(&state.knowledge, &ClaimId(query.claim_id.clone())) {
        Ok(receipt) => {
            let rendered = receipt.render();
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "claim_id": query.claim_id,
                    "rendered": rendered,
                    "claim": receipt.claim,
                    "history": receipt.history,
                    "related": receipt.related,
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

#[derive(Deserialize)]
pub struct VerifyAllRequest {
    /// Agent identity performing the sweep.
    pub by: String,
}

/// `POST /api/knowledge/verify-all` — sweep every open proposal against its
/// source in one pass, promoting what the code literally substantiates.
pub async fn verify_all(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<VerifyAllRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let provenance = Provenance {
        producer: req.by.clone(),
        observed_at: now_secs(),
        git_revision: None,
        run_id: None,
    };

    match qo_knowledge::verify_all_proposals(&state.knowledge, &state.workspace_root, provenance) {
        Ok(report) => {
            let _ = state.store.log_action(
                "knowledge_sweep",
                &format!("verified {} of {}", report.verified, report.checked),
                &req.by,
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "checked": report.checked,
                    "verified": report.verified,
                    "inconclusive": report.inconclusive,
                    "unavailable": report.unavailable,
                    "results": report.results,
                    "rendered": report.render(),
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

#[derive(Deserialize)]
pub struct RefreshSourcesRequest {
    /// Agent identity performing the refresh.
    pub by: String,
}

/// `POST /api/knowledge/refresh-sources` — mark settled claims stale when the
/// code they were verified against has moved on.
pub async fn refresh_sources(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<RefreshSourcesRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let provenance = Provenance {
        producer: req.by.clone(),
        observed_at: now_secs(),
        git_revision: None,
        run_id: None,
    };

    match qo_knowledge::refresh_sources(&state.knowledge, &state.workspace_root, provenance) {
        Ok(report) => {
            let _ = state.store.log_action(
                "knowledge_refresh_sources",
                &format!("{} stale of {} checked", report.stale, report.checked),
                &req.by,
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "checked": report.checked,
                    "still_current": report.still_current,
                    "stale": report.stale,
                    "skipped": report.skipped,
                    "results": report.results,
                    "rendered": report.render(),
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// `GET /api/knowledge/divergences` — every subject where the graph holds both
/// a load-bearing and a refuted claim (agents settled in opposite directions).
pub async fn divergences(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<Value> {
    match qo_knowledge::divergences(&state.knowledge) {
        Ok(report) => axum::Json(json!({
            "divergences": report.divergences,
            "rendered": report.render(),
        })),
        Err(e) => axum::Json(json!({ "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
pub struct HealStaleRequest {
    /// Agent identity performing the heal.
    pub by: String,
}

/// `POST /api/knowledge/heal-stale` — re-verify stale claims and heal the ones
/// whose statement still holds in the current source.
pub async fn heal_stale(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<HealStaleRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let provenance = Provenance {
        producer: req.by.clone(),
        observed_at: now_secs(),
        git_revision: None,
        run_id: None,
    };

    match qo_knowledge::heal_stale(&state.knowledge, &state.workspace_root, provenance) {
        Ok(report) => {
            let _ = state.store.log_action(
                "knowledge_heal_stale",
                &format!("healed {} of {}", report.healed, report.examined),
                &req.by,
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "examined": report.examined,
                    "healed": report.healed,
                    "remained_stale": report.remained_stale,
                    "results": report.results,
                    "rendered": report.render(),
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// `GET /api/knowledge/export` — snapshot the whole graph as portable JSON:
/// every entity and every revision of every claim, counter-evidence included.
/// Read-only; this is the backup half of the archive pair.
pub async fn knowledge_export(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::Json<qo_knowledge::Archive>, axum::http::StatusCode> {
    qo_knowledge::export(&state.knowledge, now_secs())
        .map(axum::Json)
        .map_err(|e| {
            tracing::error!(error = %e, "knowledge export failed");
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// `POST /api/knowledge/import` — restore a previously exported archive.
///
/// Additive only: it never deletes or overwrites; a claim id already present
/// is skipped and reported. Provenance is restored verbatim (never re-derived),
/// so this is a privileged operator action — it sits behind the same auth
/// layer as every other route (loopback-only when untokened).
pub async fn knowledge_import(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(archive): axum::Json<qo_knowledge::Archive>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    if archive.version != qo_knowledge::ARCHIVE_VERSION {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({
                "error": format!(
                    "unsupported archive version {} (expected {})",
                    archive.version,
                    qo_knowledge::ARCHIVE_VERSION
                )
            })),
        );
    }
    match qo_knowledge::import(&state.knowledge, &archive) {
        Ok(report) => {
            let _ = state.store.log_action(
                "knowledge_import",
                &format!(
                    "{} entities, {} claims, {} skipped",
                    report.entities_added,
                    report.claims_added,
                    report.claims_skipped.len()
                ),
                "operator",
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "entities_added": report.entities_added,
                    "claims_added": report.claims_added,
                    "claims_skipped": report.claims_skipped,
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// `GET /api/knowledge/health` — a one-block operator summary of the graph's
/// health: load-bearing count, open proposals, stale, refuted, divergences and
/// entities.
pub async fn knowledge_health(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<axum::Json<qo_knowledge::GraphHealth>, axum::http::StatusCode> {
    qo_knowledge::health(&state.knowledge)
        .map(axum::Json)
        .map_err(|e| {
            tracing::error!(error = %e, "knowledge health failed");
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        })
}

/// `POST /api/knowledge/backup` — write a timestamped snapshot of the whole
/// graph to the backup directory and return its path. The schedule is an
/// operator decision; this is the primitive a cron job calls.
pub async fn knowledge_backup(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    match qo_knowledge::write_backup(&state.knowledge, &state.backup_dir, now_secs()) {
        Ok(path) => (
            axum::http::StatusCode::OK,
            axum::Json(json!({ "path": path.display().to_string() })),
        ),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// `GET /api/knowledge/backups` — list the existing backups, newest first.
pub async fn knowledge_backups(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<Value> {
    let backups: Vec<Value> = qo_knowledge::list_backups(&state.backup_dir)
        .into_iter()
        .map(|(path, exported_at)| json!({ "path": path.display().to_string(), "exported_at": exported_at }))
        .collect();
    axum::Json(json!({ "backups": backups }))
}

#[derive(Deserialize)]
pub struct RestoreRequest {
    /// Backup timestamp to restore; omitted means the newest backup.
    pub exported_at: Option<u64>,
}

/// `POST /api/knowledge/restore` — recover the graph from a backup file.
/// This is the operator's recovery path after a redb loss: it reads a backup
/// archive and imports it additively (never overwrites). With no body, it
/// restores the newest backup.
pub async fn knowledge_restore(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<RestoreRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let backups = qo_knowledge::list_backups(&state.backup_dir);
    let target = match req.exported_at {
        Some(ts) => backups.iter().find(|(_, t)| *t == ts).map(|(p, _)| p.clone()),
        None => backups.first().map(|(p, _)| p.clone()),
    };
    let Some(path) = target else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(json!({ "error": "no backup to restore — run `qlang graph backup` first" })),
        );
    };

    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({ "error": format!("cannot read backup {}: {e}", path.display()) })),
            )
        }
    };
    let archive = match qo_knowledge::Archive::from_json(&contents) {
        Ok(a) => a,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({ "error": format!("backup {} is malformed: {e}", path.display()) })),
            )
        }
    };

    match qo_knowledge::import(&state.knowledge, &archive) {
        Ok(report) => {
            let _ = state.store.log_action(
                "knowledge_restore",
                &format!("restored {} from {}", report.claims_added, path.display()),
                "operator",
            );
            (
                axum::http::StatusCode::OK,
                axum::Json(json!({
                    "restored_from": path.display().to_string(),
                    "entities_added": report.entities_added,
                    "claims_added": report.claims_added,
                    "claims_skipped": report.claims_skipped,
                })),
            )
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "error": e.to_string() })),
        ),
    }
}

#[derive(Deserialize)]
pub struct ProposeRequest {
    /// An OrbitQLang document (unsigned — this is the proposal path).
    pub document: String,
}

/// `POST /api/knowledge/propose` — submit an OrbitQLang document through the
/// deterministic admission gate and merge every admitted claim as `proposed`.
/// Refused whole (with every violation) when the document fails admission;
/// never promotes, never refutes.
pub async fn propose(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<ProposeRequest>,
) -> (axum::http::StatusCode, axum::Json<Value>) {
    let known: Vec<EntityId> = state
        .knowledge
        .list_entities()
        .unwrap_or_default()
        .into_iter()
        .map(|e| e.id)
        .collect();
    let policy = qo_knowledge::ProposalPolicy::default().with_known_entities(known);

    let outcome = qo_knowledge::propose_from_text(&req.document, &policy);
    if !outcome.is_ok() {
        let violations: Vec<Value> = outcome
            .violations
            .iter()
            .map(|v| json!({ "line": v.line, "message": v.message }))
            .collect();
        return (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(json!({ "error": "proposal rejected", "violations": violations })),
        );
    }
    let delta = outcome.delta.expect("is_ok implies a delta");

    let report = match qo_knowledge::merge_delta(&state.knowledge, &delta) {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({ "error": e.to_string() })),
            )
        }
    };
    for op in &delta.operations {
        if let qo_knowledge::GraphDeltaOp::AddEntity { entity } = op {
            register_entity(&state, &entity.id);
        }
    }
    let _ = state.store.log_action(
        "orbit_graph_propose",
        &format!("delta {} proposed: {} applied", delta.id, report.applied()),
        &delta.producer.id,
    );

    (
        axum::http::StatusCode::OK,
        axum::Json(json!({
            "delta_id": delta.id,
            "applied": report.applied(),
            "already_applied": report.already_applied(),
            "conflicts": report.conflicts(),
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tool that is advertised in `tools/list` but missing from `handles`
    /// would be invisible to `tools/call`. Guard against that drift: every
    /// definition must be dispatchable, and vice versa.
    #[test]
    fn every_advertised_tool_is_handled() {
        for tool in tool_definitions() {
            let name = tool["name"].as_str().expect("tool name");
            assert!(
                handles(name),
                "tool {name} is advertised but not handled"
            );
        }
        assert!(handles("orbit_graph_verify_source"));
    }

    /// Write tools are exactly the graph-mutating ones; read tools are
    /// reachable by a viewer. A drift here either lets a viewer write or
    /// locks a member out of a read.
    #[test]
    fn write_tools_are_classified_exactly() {
        for write in [
            "orbit_graph_add_claim",
            "orbit_graph_verify_claim",
            "orbit_graph_commit_delta",
            "orbit_graph_verify_source",
            "orbit_graph_verify_all",
            "orbit_graph_refresh_sources",
            "orbit_graph_heal_stale",
            "orbit_graph_propose",
        ] {
            assert!(requires_write(write), "{write} must require write");
        }
        for read in [
            "orbit_graph_search",
            "orbit_graph_neighbors",
            "orbit_graph_impact",
            "orbit_graph_context",
            "orbit_graph_receipt",
            "orbit_graph_divergences",
            "orbit_graph_health",
            "orbit_graph_swarm_state",
            "orbit_graph_proposal_prompt",
        ] {
            assert!(!requires_write(read), "{read} must be viewer-reachable");
        }
    }
}
