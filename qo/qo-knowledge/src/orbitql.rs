//! OrbitQLang — the surface syntax for [`GraphDelta`].
//!
//! This is the text layer an LLM writes and reads. The delta is the source of
//! truth; this module only renders it and parses it back.
//!
//! ## Why this shape
//!
//! The grammar is line-oriented and bracket-free: one operation per line, a
//! leading verb, then `|`-separated fields in fixed positions. Nothing nests.
//! That buys three things a bracketed syntax does not:
//!
//! - **Token economy.** No structural punctuation to spend tokens on, and no
//!   repeated JSON keys. A field is identified by its position.
//! - **Recoverable parsing.** A malformed line is reported with its line
//!   number and skipped; the surrounding lines still parse. A missing brace
//!   in a nested format destroys the rest of the document.
//! - **Constrained decoding.** At any point the set of legal next tokens is a
//!   function of the current column, so a grammar-constrained decoder can mask
//!   the vocabulary without tracking a stack.
//!
//! ## Grammar (EBNF)
//!
//! ```ebnf
//! document    = { line } ;
//! line        = delta_hdr | producer | signature | entity | claim | relation
//!             | verify | refute | comment | blank ;
//!
//! delta_hdr   = "DELTA" , "|" , version , "|" , id ;
//! producer    = "BY" , "|" , id , "|" , emitted_at , [ "|" , source_rev ,
//!               [ "|" , run_id ] ] ;
//! signature   = "SIG" , "|" , algorithm , "|" , key_id , "|" , value ;
//! entity      = "+E" , "|" , entity_kind , "|" , name ;
//! claim       = "+C" , "|" , claim_id , "|" , subject , "|" , statement ;
//! relation    = "+R" , "|" , claim_id , "|" , relation_kind , "|" , object ;
//! verify      = "OK" , "|" , claim_id , "|" , evidence ;
//! refute      = "NO" , "|" , claim_id , "|" , evidence ;
//! evidence    = evidence_kind , "|" , locator , [ "|" , span ,
//!               [ "|" , excerpt ] ] ;
//! span        = "-" | line_no , ":" , line_no ;
//! comment     = "#" , { any } ;
//! ```
//!
//! Entities are written as `kind:name`, matching [`EntityId::derive`], so a
//! subject reference and an entity declaration agree by construction.
//!
//! `SIG` carries the producer's signature over
//! [`GraphDelta::signing_payload`] — which is derived from the *typed* delta,
//! not from this text. Comments, blank lines and incidental whitespace
//! therefore do not affect it, and stripping the `SIG` line reproduces
//! exactly what was signed.
//!
//! ## Example
//!
//! ```text
//! DELTA|1|d-42
//! BY|worker-3|1700000000|abc123
//! +E|file|src/auth.rs
//! +C|c1|file:src/auth.rs|auth uses bcrypt
//! +R|c1|depends_on|file:Cargo.toml
//! OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
//! ```
//!
//! ## Escaping
//!
//! Free-text fields (`name`, `statement`, `excerpt`) may contain `|` and
//! newlines. They are escaped as `\p`, `\n`, `\r` and `\\`. Everything else
//! is passed through, so ordinary prose costs nothing.

use crate::delta::{DeltaProducer, DeltaSignature, GraphDelta, GraphDeltaOp};
use crate::model::{
    Claim, ClaimId, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance, Relation,
};

/// A parse problem, located at a line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrbitQlError {
    /// 1-indexed line in the source document.
    pub line: usize,
    pub message: String,
}

impl std::fmt::Display for OrbitQlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}
impl std::error::Error for OrbitQlError {}

/// Outcome of a recovering parse: whatever parsed, plus whatever did not.
///
/// A worker's output is not trusted to be well-formed, so the parser reports
/// every bad line instead of stopping at the first.
#[derive(Debug, Clone)]
pub struct ParseOutcome {
    pub delta: Option<GraphDelta>,
    pub errors: Vec<OrbitQlError>,
    /// 1-indexed source line of each operation in `delta.operations`, in
    /// submission order. Lets a policy layer attribute an operation-level
    /// rejection to the exact line a worker must fix.
    pub op_lines: Vec<usize>,
}

impl ParseOutcome {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty() && self.delta.is_some()
    }
}

// ---------------------------------------------------------------------------
// Escaping
// ---------------------------------------------------------------------------

/// Escape a free-text field so it cannot introduce a separator or a line break.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '|' => out.push_str("\\p"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

/// Inverse of [`escape`]. An unknown escape is left verbatim rather than
/// rejected — a worker writing a Windows path should not fail the parse.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('p') => out.push('|'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Serialize: GraphDelta -> OrbitQLang
// ---------------------------------------------------------------------------

/// Render a delta as an OrbitQLang document.
///
/// Deterministic: the same delta always produces byte-identical output, so a
/// document can be hashed or diffed.
pub fn to_orbitql(delta: &GraphDelta) -> String {
    let mut out = String::new();

    out.push_str(&format!("DELTA|{}|{}\n", delta.version, escape(&delta.id)));

    let p = &delta.producer;
    out.push_str(&format!("BY|{}|{}", escape(&p.id), p.emitted_at));
    // Trailing optionals are only written when present, and run_id implies a
    // placeholder for source_revision so the positions stay unambiguous.
    match (&p.source_revision, &p.run_id) {
        (None, None) => {}
        (rev, run) => {
            out.push('|');
            out.push_str(&rev.as_deref().map(escape).unwrap_or_else(|| "-".into()));
            if let Some(r) = run {
                out.push('|');
                out.push_str(&escape(r));
            }
        }
    }
    out.push('\n');

    // The signature is a separate line rather than more `BY` fields: it is
    // optional, variable in length, and a verifier must be able to strip it
    // to re-derive what was signed. Its own verb makes both trivial.
    if let Some(sig) = &p.signature {
        out.push_str(&format!(
            "SIG|{}|{}|{}\n",
            escape(&sig.algorithm),
            escape(&sig.key_id),
            escape(&sig.value)
        ));
    }

    for op in &delta.operations {
        match op {
            GraphDeltaOp::AddEntity { entity } => {
                out.push_str(&format!(
                    "+E|{}|{}\n",
                    entity.kind.as_str(),
                    escape(&entity.name)
                ));
            }
            GraphDeltaOp::AddClaim { claim } => {
                out.push_str(&format!(
                    "+C|{}|{}|{}\n",
                    escape(&claim.id.0),
                    escape(&claim.subject.0),
                    escape(&claim.statement)
                ));
            }
            GraphDeltaOp::AddRelation { claim_id, relation, object } => {
                out.push_str(&format!(
                    "+R|{}|{}|{}\n",
                    escape(&claim_id.0),
                    relation.as_str(),
                    escape(&object.0)
                ));
            }
            GraphDeltaOp::VerifyClaim { claim_id, evidence } => {
                out.push_str(&format!(
                    "OK|{}|{}\n",
                    escape(&claim_id.0),
                    render_evidence(evidence)
                ));
            }
            GraphDeltaOp::RefuteClaim { claim_id, evidence } => {
                out.push_str(&format!(
                    "NO|{}|{}\n",
                    escape(&claim_id.0),
                    render_evidence(evidence)
                ));
            }
        }
    }

    out
}

/// `kind|locator[|span[|excerpt]]`.
///
/// `supports` is not written: it is implied by the verb (`OK` vs `NO`), which
/// is also what the store enforces.
fn render_evidence(e: &Evidence) -> String {
    let mut s = format!("{}|{}", evidence_kind_str(e.kind), escape(&e.locator));
    match (&e.lines, &e.excerpt) {
        (None, None) => {}
        (lines, excerpt) => {
            s.push('|');
            match lines {
                Some((a, b)) => s.push_str(&format!("{a}:{b}")),
                None => s.push('-'),
            }
            if let Some(x) = excerpt {
                s.push('|');
                s.push_str(&escape(x));
            }
        }
    }
    s
}

fn evidence_kind_str(k: EvidenceKind) -> &'static str {
    match k {
        EvidenceKind::Source => "source",
        EvidenceKind::Commit => "commit",
        EvidenceKind::TestRun => "test_run",
        EvidenceKind::ToolRun => "tool_run",
        EvidenceKind::External => "external",
    }
}

fn parse_evidence_kind(s: &str) -> Option<EvidenceKind> {
    Some(match s {
        "source" => EvidenceKind::Source,
        "commit" => EvidenceKind::Commit,
        "test_run" => EvidenceKind::TestRun,
        "tool_run" => EvidenceKind::ToolRun,
        "external" => EvidenceKind::External,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Parse: OrbitQLang -> GraphDelta
// ---------------------------------------------------------------------------

/// Parse a document, stopping at the first error.
///
/// Use [`parse_recovering`] when the input came from an LLM and you want to
/// report every problem at once.
pub fn from_orbitql(source: &str) -> Result<GraphDelta, OrbitQlError> {
    let outcome = parse_recovering(source);
    match (outcome.delta, outcome.errors.into_iter().next()) {
        (_, Some(first)) => Err(first),
        (Some(delta), None) => Ok(delta),
        (None, None) => Err(OrbitQlError {
            line: 0,
            message: "empty document: expected a DELTA header".into(),
        }),
    }
}

/// Parse a document, collecting every malformed line instead of stopping.
pub fn parse_recovering(source: &str) -> ParseOutcome {
    let mut errors = Vec::new();
    let mut header: Option<(u16, String)> = None;
    let mut producer: Option<DeltaProducer> = None;
    let mut signature: Option<DeltaSignature> = None;
    let mut operations: Vec<GraphDeltaOp> = Vec::new();
    let mut op_lines: Vec<usize> = Vec::new();
    // Claim positions are staged so the producer, which may be declared after
    // them, can be stamped onto their provenance once the document is read.
    let mut claim_positions: Vec<usize> = Vec::new();

    for (i, raw) in source.lines().enumerate() {
        let line_no = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('|').collect();
        let err = |message: String| OrbitQlError { line: line_no, message };

        match fields[0] {
            "DELTA" => {
                if fields.len() != 3 {
                    errors.push(err(format!(
                        "DELTA expects 2 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                match fields[1].parse::<u16>() {
                    Ok(v) => header = Some((v, unescape(fields[2]))),
                    Err(_) => errors.push(err(format!("bad version {:?}", fields[1]))),
                }
            }
            "BY" => {
                if fields.len() < 3 || fields.len() > 5 {
                    errors.push(err(format!(
                        "BY expects 2-4 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                let emitted_at = match fields[2].parse::<u64>() {
                    Ok(t) => t,
                    Err(_) => {
                        errors.push(err(format!("bad timestamp {:?}", fields[2])));
                        continue;
                    }
                };
                producer = Some(DeltaProducer {
                    id: unescape(fields[1]),
                    emitted_at,
                    source_revision: fields.get(3).and_then(|s| opt_field(s)),
                    run_id: fields.get(4).and_then(|s| opt_field(s)),
                    signature: None,
                });
            }
            "SIG" => {
                if fields.len() != 4 {
                    errors.push(err(format!(
                        "SIG expects 3 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                signature = Some(DeltaSignature {
                    algorithm: unescape(fields[1]),
                    key_id: unescape(fields[2]),
                    value: unescape(fields[3]),
                });
            }
            "+E" => {
                if fields.len() != 3 {
                    errors.push(err(format!(
                        "+E expects 2 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                let Some(kind) = EntityKind::parse(fields[1]) else {
                    errors.push(err(format!("unknown entity kind {:?}", fields[1])));
                    continue;
                };
                let name = unescape(fields[2]);
                operations.push(GraphDeltaOp::AddEntity {
                    entity: Entity {
                        id: EntityId::derive(kind, &name),
                        kind,
                        name,
                    },
                });
                op_lines.push(line_no);
            }
            "+C" => {
                // A raw separator never survives escaping, so the field count
                // is exact even though the statement is free text.
                if fields.len() != 4 {
                    errors.push(err(format!(
                        "+C expects 3 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                claim_positions.push(operations.len());
                operations.push(GraphDeltaOp::AddClaim {
                    claim: Claim::proposed(
                        unescape(fields[1]),
                        unescape(fields[3]),
                        EntityId(unescape(fields[2])),
                        provenance_from(producer.as_ref()),
                    ),
                });
                op_lines.push(line_no);
            }
            "+R" => {
                if fields.len() != 4 {
                    errors.push(err(format!(
                        "+R expects 3 fields, found {}",
                        fields.len() - 1
                    )));
                    continue;
                }
                let Some(relation) = Relation::parse(fields[2]) else {
                    errors.push(err(format!("unknown relation {:?}", fields[2])));
                    continue;
                };
                operations.push(GraphDeltaOp::AddRelation {
                    claim_id: ClaimId(unescape(fields[1])),
                    relation,
                    object: EntityId(unescape(fields[3])),
                });
                op_lines.push(line_no);
            }
            verb @ ("OK" | "NO") => {
                if fields.len() < 2 {
                    errors.push(err(format!("{verb} expects a claim id")));
                    continue;
                }
                let supports = verb == "OK";
                match parse_evidence(&fields[2..], supports) {
                    Ok(evidence) => {
                        let claim_id = ClaimId(unescape(fields[1]));
                        operations.push(if supports {
                            GraphDeltaOp::VerifyClaim { claim_id, evidence }
                        } else {
                            GraphDeltaOp::RefuteClaim { claim_id, evidence }
                        });
                        op_lines.push(line_no);
                    }
                    Err(message) => errors.push(err(message)),
                }
            }
            other => {
                errors.push(err(format!("unknown verb {other:?}")));
            }
        }
    }

    // SIG may appear before BY, so it is attached once the whole document is
    // read rather than at the line that carried it.
    if let (Some(p), Some(sig)) = (producer.as_mut(), signature) {
        p.signature = Some(sig);
    }

    // A claim written before the BY line still belongs to the same producer.
    if let Some(p) = producer.as_ref() {
        let provenance = provenance_from(Some(p));
        for index in &claim_positions {
            if let Some(GraphDeltaOp::AddClaim { claim }) = operations.get_mut(*index) {
                claim.provenance = provenance.clone();
            }
        }
    }

    let delta = match (header, producer) {
        (Some((version, id)), Some(producer)) => Some(GraphDelta {
            version,
            id,
            producer,
            operations,
        }),
        (None, _) => {
            errors.push(OrbitQlError {
                line: 0,
                message: "missing DELTA header".into(),
            });
            None
        }
        (_, None) => {
            errors.push(OrbitQlError {
                line: 0,
                message: "missing BY producer line".into(),
            });
            None
        }
    };

    ParseOutcome {
        delta,
        errors,
        op_lines,
    }
}

/// `-` is the explicit absent marker for a positional optional.
fn opt_field(s: &str) -> Option<String> {
    match s {
        "" | "-" => None,
        other => Some(unescape(other)),
    }
}

/// A claim inherits the document producer: whoever wrote the delta made the
/// claim. Provenance is therefore never absent, matching the model rule.
fn provenance_from(p: Option<&DeltaProducer>) -> Provenance {
    match p {
        Some(p) => Provenance {
            producer: p.id.clone(),
            observed_at: p.emitted_at,
            git_revision: p.source_revision.clone(),
            run_id: p.run_id.clone(),
        },
        None => Provenance {
            producer: String::new(),
            observed_at: 0,
            git_revision: None,
            run_id: None,
        },
    }
}

fn parse_evidence(fields: &[&str], supports: bool) -> Result<Evidence, String> {
    if fields.len() < 2 || fields.len() > 4 {
        return Err(format!(
            "evidence expects 2-4 fields, found {}",
            fields.len()
        ));
    }
    let Some(kind) = parse_evidence_kind(fields[0]) else {
        return Err(format!("unknown evidence kind {:?}", fields[0]));
    };
    let lines = match fields.get(2) {
        None | Some(&"-") | Some(&"") => None,
        Some(span) => {
            let Some((a, b)) = span.split_once(':') else {
                return Err(format!("bad line span {span:?}, expected start:end"));
            };
            match (a.parse::<u32>(), b.parse::<u32>()) {
                (Ok(a), Ok(b)) => Some((a, b)),
                _ => return Err(format!("bad line span {span:?}, expected start:end")),
            }
        }
    };
    Ok(Evidence {
        kind,
        locator: unescape(fields[1]),
        lines,
        excerpt: fields.get(3).and_then(|s| opt_field(s)),
        supports,
    })
}
