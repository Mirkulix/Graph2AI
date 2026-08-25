//! MCP tool surface for the knowledge graph.
//!
//! Five tools, mirroring `qo-knowledge.md`:
//!   * `orbit_graph_search`       — find entities and backed claims
//!   * `orbit_graph_neighbors`    — traverse relations and impact
//!   * `orbit_graph_add_claim`    — record a claim as a *proposal*
//!   * `orbit_graph_verify_claim` — confirm or refute with evidence
//!   * `orbit_graph_context`      — compact, source-bound context for a task
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

use serde_json::{json, Value};
use std::sync::Arc;

use crate::AppState;
use qo_knowledge::{
    Claim, ClaimId, ClaimStatus, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance,
    Relation,
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
    ]
}

/// True if `name` is one of this module's tools.
pub fn handles(name: &str) -> bool {
    matches!(
        name,
        "orbit_graph_search"
            | "orbit_graph_neighbors"
            | "orbit_graph_add_claim"
            | "orbit_graph_verify_claim"
            | "orbit_graph_context"
    )
}

type ToolResult = Result<String, (i32, String)>;

const INVALID_PARAMS: i32 = -32602;
const INTERNAL: i32 = -32603;

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

/// Dispatch one of the `orbit_graph_*` tools.
pub async fn call(state: Arc<AppState>, name: &str, args: Value) -> ToolResult {
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

        other => Err((-32601, format!("Unknown tool: {}", other))),
    }
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
