//! Round-trip and grammar tests for the OrbitQLang surface syntax.
//!
//! The contract under test: `delta -> text -> delta` returns the original
//! value. Anything the text cannot carry is a silent data loss bug, so the
//! cases below deliberately push at the fields most likely to be dropped —
//! optionals, separators inside free text, and empty collections.

use qo_knowledge::delta::{DeltaProducer, GraphDelta, GraphDeltaOp, GRAPH_DELTA_VERSION};
use qo_knowledge::model::{
    Claim, ClaimId, Entity, EntityId, EntityKind, Evidence, EvidenceKind, Provenance, Relation,
};
use qo_knowledge::{from_orbitql, parse_recovering, to_orbitql};

fn producer() -> DeltaProducer {
    DeltaProducer {
        id: "worker-3".into(),
        source_revision: Some("abc123".into()),
        run_id: Some("run-7".into()),
        emitted_at: 1_700_000_000,
        signature: None,
    }
}

fn provenance_of(p: &DeltaProducer) -> Provenance {
    Provenance {
        producer: p.id.clone(),
        observed_at: p.emitted_at,
        git_revision: p.source_revision.clone(),
        run_id: p.run_id.clone(),
    }
}

fn delta_with(ops: Vec<GraphDeltaOp>) -> GraphDelta {
    GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: "d-42".into(),
        producer: producer(),
        operations: ops,
    }
}

fn file_entity(path: &str) -> Entity {
    Entity {
        id: EntityId::derive(EntityKind::File, path),
        kind: EntityKind::File,
        name: path.into(),
    }
}

/// The core guarantee: a delta survives the text layer unchanged.
#[test]
fn roundtrip_preserves_every_operation_kind() {
    let p = producer();
    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");

    let original = delta_with(vec![
        GraphDeltaOp::AddEntity {
            entity: file_entity("src/auth.rs"),
        },
        GraphDeltaOp::AddClaim {
            claim: Claim::proposed(
                "c1",
                "auth uses bcrypt",
                subject.clone(),
                provenance_of(&p),
            ),
        },
        GraphDeltaOp::AddRelation {
            claim_id: ClaimId("c1".into()),
            relation: Relation::DependsOn,
            object: EntityId::derive(EntityKind::File, "Cargo.toml"),
        },
        GraphDeltaOp::VerifyClaim {
            claim_id: ClaimId("c1".into()),
            evidence: Evidence {
                kind: EvidenceKind::Source,
                locator: "src/auth.rs".into(),
                lines: Some((42, 43)),
                excerpt: Some("use bcrypt::hash;".into()),
                supports: true,
            },
        },
        GraphDeltaOp::RefuteClaim {
            claim_id: ClaimId("c2".into()),
            evidence: Evidence {
                kind: EvidenceKind::TestRun,
                locator: "cargo test auth".into(),
                lines: None,
                excerpt: None,
                supports: false,
            },
        },
    ]);

    let text = to_orbitql(&original);
    let parsed = from_orbitql(&text).expect("valid document should parse");

    assert_eq!(parsed, original, "round-trip changed the delta\n{text}");
}

/// Every entity kind and relation must survive; a missing match arm in either
/// direction would otherwise only show up for the one kind nobody tested.
#[test]
fn roundtrip_covers_all_entity_kinds_and_relations() {
    let kinds = [
        EntityKind::Repository,
        EntityKind::File,
        EntityKind::Symbol,
        EntityKind::Service,
        EntityKind::Endpoint,
        EntityKind::Concept,
        EntityKind::Agent,
        EntityKind::Run,
    ];
    let relations = [
        Relation::Defines,
        Relation::Calls,
        Relation::DependsOn,
        Relation::Implements,
        Relation::Contradicts,
        Relation::Documents,
        Relation::Tests,
        Relation::Produces,
    ];

    let mut ops = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        let name = format!("thing-{i}");
        ops.push(GraphDeltaOp::AddEntity {
            entity: Entity {
                id: EntityId::derive(*kind, &name),
                kind: *kind,
                name,
            },
        });
    }
    for (i, relation) in relations.iter().enumerate() {
        ops.push(GraphDeltaOp::AddRelation {
            claim_id: ClaimId(format!("c{i}")),
            relation: *relation,
            object: EntityId::derive(EntityKind::Concept, "target"),
        });
    }

    let original = delta_with(ops);
    let parsed = from_orbitql(&to_orbitql(&original)).unwrap();
    assert_eq!(parsed, original);
}

/// All five evidence kinds, and the optional span/excerpt in every combination
/// — these are the positional optionals most likely to shift a field.
#[test]
fn roundtrip_covers_evidence_variants() {
    let kinds = [
        EvidenceKind::Source,
        EvidenceKind::Commit,
        EvidenceKind::TestRun,
        EvidenceKind::ToolRun,
        EvidenceKind::External,
    ];
    let shapes: [(Option<(u32, u32)>, Option<String>); 4] = [
        (None, None),
        (Some((1, 9)), None),
        (None, Some("excerpt only".into())),
        (Some((5, 5)), Some("both".into())),
    ];

    let mut ops = Vec::new();
    for kind in kinds {
        for (lines, excerpt) in &shapes {
            ops.push(GraphDeltaOp::VerifyClaim {
                claim_id: ClaimId("c1".into()),
                evidence: Evidence {
                    kind,
                    locator: "loc".into(),
                    lines: *lines,
                    excerpt: excerpt.clone(),
                    supports: true,
                },
            });
        }
    }

    let original = delta_with(ops);
    let text = to_orbitql(&original);
    let parsed = from_orbitql(&text).unwrap();
    assert_eq!(parsed, original, "evidence optionals were not preserved\n{text}");
}

/// Free text containing the separator, backslashes and newlines must not be
/// able to inject a field or a line — this is the escaping contract, and also
/// the injection boundary for LLM-written statements.
#[test]
fn roundtrip_survives_separators_and_newlines_in_free_text() {
    let nasty = "a|b\\c\nd\re|+C|forged|injected";
    let p = producer();

    let original = delta_with(vec![
        GraphDeltaOp::AddEntity {
            entity: {
                let name = nasty.to_string();
                Entity {
                    id: EntityId::derive(EntityKind::Concept, &name),
                    kind: EntityKind::Concept,
                    name,
                }
            },
        },
        GraphDeltaOp::AddClaim {
            claim: Claim::proposed(
                "c1",
                nasty,
                EntityId::derive(EntityKind::File, "x.rs"),
                provenance_of(&p),
            ),
        },
        GraphDeltaOp::VerifyClaim {
            claim_id: ClaimId("c1".into()),
            evidence: Evidence {
                kind: EvidenceKind::External,
                locator: nasty.into(),
                lines: None,
                excerpt: Some(nasty.into()),
                supports: true,
            },
        },
    ]);

    let text = to_orbitql(&original);
    // The escaped document is still exactly one line per operation, plus the
    // two header lines — proof that no free text broke out of its line.
    assert_eq!(text.lines().count(), 5, "free text escaped its line\n{text}");

    let parsed = from_orbitql(&text).unwrap();
    assert_eq!(parsed, original);
}

/// A producer with no revision and no run id: the trailing optionals are
/// simply absent rather than written as empty fields.
#[test]
fn roundtrip_handles_absent_producer_optionals() {
    let original = GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: "d-1".into(),
        producer: DeltaProducer {
            id: "solo".into(),
            source_revision: None,
            run_id: None,
            emitted_at: 42,
            signature: None,
        },
        operations: vec![GraphDeltaOp::AddEntity {
            entity: file_entity("a.rs"),
        }],
    };

    let text = to_orbitql(&original);
    assert!(text.contains("BY|solo|42\n"), "unexpected producer line\n{text}");
    assert_eq!(from_orbitql(&text).unwrap(), original);
}

/// A run id without a source revision must keep its position, or the two
/// optionals would swap on the way back.
#[test]
fn roundtrip_handles_run_id_without_source_revision() {
    let original = GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: "d-1".into(),
        producer: DeltaProducer {
            id: "solo".into(),
            source_revision: None,
            run_id: Some("run-9".into()),
            emitted_at: 42,
            signature: None,
        },
        operations: vec![GraphDeltaOp::AddEntity {
            entity: file_entity("a.rs"),
        }],
    };

    let parsed = from_orbitql(&to_orbitql(&original)).unwrap();
    assert_eq!(parsed.producer.source_revision, None);
    assert_eq!(parsed.producer.run_id.as_deref(), Some("run-9"));
    assert_eq!(parsed, original);
}

/// Serialization is deterministic, so a document can be hashed or diffed.
#[test]
fn serialization_is_stable() {
    let original = delta_with(vec![GraphDeltaOp::AddEntity {
        entity: file_entity("src/auth.rs"),
    }]);
    assert_eq!(to_orbitql(&original), to_orbitql(&original));
}

/// Text -> delta -> text also round-trips, which is what a diff view needs.
#[test]
fn text_roundtrip_is_canonical() {
    let source = "\
DELTA|1|d-42
BY|worker-3|1700000000|abc123|run-7
+E|file|src/auth.rs
+C|c1|file:src/auth.rs|auth uses bcrypt
+R|c1|depends_on|file:Cargo.toml
OK|c1|source|src/auth.rs|42:42|use bcrypt::hash;
";
    let delta = from_orbitql(source).expect("documented example must parse");
    assert_eq!(to_orbitql(&delta), source, "example is not canonical");
}

/// Comments and blank lines are ignored, so a worker may annotate its output.
#[test]
fn comments_and_blank_lines_are_ignored() {
    let source = "\
# this delta was produced by hand
DELTA|1|d-1

BY|solo|42
# an entity follows
+E|file|a.rs
";
    let delta = from_orbitql(source).unwrap();
    assert_eq!(delta.operations.len(), 1);
}

/// A claim declared before the producer line still inherits its provenance —
/// the model has no claim without provenance, so neither may the parser.
#[test]
fn claim_before_producer_line_still_gets_provenance() {
    let source = "\
DELTA|1|d-1
+C|c1|file:a.rs|something
BY|worker-1|99|rev-1|run-1
";
    let delta = from_orbitql(source).unwrap();
    let GraphDeltaOp::AddClaim { claim } = &delta.operations[0] else {
        panic!("expected a claim");
    };
    assert_eq!(claim.provenance.producer, "worker-1");
    assert_eq!(claim.provenance.observed_at, 99);
    assert_eq!(claim.provenance.git_revision.as_deref(), Some("rev-1"));
    assert_eq!(claim.provenance.run_id.as_deref(), Some("run-1"));
}

/// Parsed claims are proposals. A worker cannot write itself a verified fact,
/// which is the rule `GraphDelta::validate` enforces downstream.
#[test]
fn parsed_claims_are_always_proposals() {
    let source = "\
DELTA|1|d-1
BY|worker-1|99
+C|c1|file:a.rs|something
";
    let delta = from_orbitql(source).unwrap();
    delta.validate().expect("a parsed delta must satisfy the delta contract");
}

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

/// A bad line is reported with its number and does not stop the parse — an
/// LLM should get every problem back at once, not one per attempt.
#[test]
fn parse_recovers_and_reports_every_bad_line() {
    let source = "\
DELTA|1|d-1
BY|solo|42
+E|file|good.rs
+E|nonsense|bad.rs
+R|c1|not_a_relation|file:x.rs
WAT|whatever
+E|file|also-good.rs
";
    let outcome = parse_recovering(source);

    let lines: Vec<usize> = outcome.errors.iter().map(|e| e.line).collect();
    assert_eq!(lines, vec![4, 5, 6], "wrong lines flagged: {:?}", outcome.errors);

    let delta = outcome.delta.expect("good lines should still produce a delta");
    assert_eq!(delta.operations.len(), 2, "good lines were dropped");
}

/// A document without a header is an error rather than a silently empty delta.
#[test]
fn missing_header_is_an_error() {
    let outcome = parse_recovering("+E|file|a.rs\n");
    assert!(outcome.delta.is_none());
    assert!(!outcome.errors.is_empty());
    assert!(from_orbitql("+E|file|a.rs\n").is_err());
}

/// Field-count mistakes name the verb and the count, so the fix is mechanical.
#[test]
fn wrong_field_count_is_reported_precisely() {
    let outcome = parse_recovering("DELTA|1|d-1\nBY|solo|42\n+C|c1|file:a.rs\n");
    let error = outcome.errors.first().expect("expected an error");
    assert_eq!(error.line, 3);
    assert!(
        error.message.contains("+C expects 3 fields"),
        "unhelpful message: {}",
        error.message
    );
}

/// A malformed line span is rejected rather than silently dropped, since a
/// wrong span points a reader at the wrong evidence.
#[test]
fn bad_line_span_is_rejected() {
    let outcome = parse_recovering("DELTA|1|d-1\nBY|solo|42\nOK|c1|source|a.rs|42\n");
    let error = outcome.errors.first().expect("expected an error");
    assert!(
        error.message.contains("bad line span"),
        "unhelpful message: {}",
        error.message
    );
}

// ---------------------------------------------------------------------------
// Token economy
// ---------------------------------------------------------------------------

/// The point of the format is that it is smaller than the JSON it replaces.
/// Character count is a proxy for tokens, but a sound one here: the JSON
/// spends its extra characters on punctuation and repeated keys, which are
/// exactly what a BPE tokenizer does not merge away.
#[test]
fn orbitql_is_smaller_than_the_json_it_replaces() {
    let p = producer();
    let subject = EntityId::derive(EntityKind::File, "src/auth.rs");
    let delta = delta_with(vec![
        GraphDeltaOp::AddEntity {
            entity: file_entity("src/auth.rs"),
        },
        GraphDeltaOp::AddClaim {
            claim: Claim::proposed("c1", "auth uses bcrypt", subject, provenance_of(&p)),
        },
        GraphDeltaOp::AddRelation {
            claim_id: ClaimId("c1".into()),
            relation: Relation::DependsOn,
            object: EntityId::derive(EntityKind::File, "Cargo.toml"),
        },
    ]);

    let text = to_orbitql(&delta);
    let json = delta.to_canonical_json().unwrap();

    assert!(
        text.len() * 3 < json.len(),
        "expected a >3x saving, got {} vs {} bytes",
        text.len(),
        json.len()
    );
}
