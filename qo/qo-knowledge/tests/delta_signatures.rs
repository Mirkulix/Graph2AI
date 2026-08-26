//! Signature and trust tests.
//!
//! These are written as attacks. A signature scheme that only gets tested with
//! well-formed input tells you nothing — the previous hand-rolled scheme in
//! this repository passed its happy-path tests right up until someone forged
//! against it. Each test below is a way in that must stay shut.

use qo_knowledge::delta::{DeltaProducer, GraphDelta, GraphDeltaOp, GRAPH_DELTA_VERSION};
use qo_knowledge::model::{Claim, EntityId, EntityKind, Provenance};
use qo_knowledge::trust::{public_key_hex, sign_delta, verify_delta, TrustError, TrustStore, TrustedKey};
use qo_knowledge::{from_orbitql, to_orbitql};

const NOW: u64 = 1_700_000_000;
const WORKER_SEED: [u8; 32] = [7u8; 32];
const ATTACKER_SEED: [u8; 32] = [9u8; 32];

fn key(seed: &[u8; 32], key_id: &str) -> TrustedKey {
    TrustedKey {
        key_id: key_id.into(),
        public_key_hex: public_key_hex(seed),
        active_from: 0,
        accept_until: None,
        revoked_at: None,
        comment: None,
    }
}

fn store_trusting(producer: &str, seed: &[u8; 32], key_id: &str) -> TrustStore {
    let mut store = TrustStore::new();
    store.trust(producer, key(seed, key_id));
    store
}

fn delta(id: &str, producer: &str) -> GraphDelta {
    GraphDelta {
        version: GRAPH_DELTA_VERSION,
        id: id.into(),
        producer: DeltaProducer {
            id: producer.into(),
            source_revision: Some("abc123".into()),
            run_id: None,
            emitted_at: NOW,
            signature: None,
        },
        operations: vec![GraphDeltaOp::AddClaim {
            claim: Claim::proposed(
                "c1",
                "auth uses bcrypt",
                EntityId::derive(EntityKind::File, "src/auth.rs"),
                Provenance {
                    producer: producer.into(),
                    observed_at: NOW,
                    git_revision: None,
                    run_id: None,
                },
            ),
        }],
    }
}

fn signed(id: &str, producer: &str, seed: &[u8; 32], key_id: &str) -> GraphDelta {
    let mut d = delta(id, producer);
    sign_delta(&mut d, key_id, seed).unwrap();
    d
}

// ---------------------------------------------------------------------------
// The happy path, so the attacks below mean something
// ---------------------------------------------------------------------------

#[test]
fn a_properly_signed_delta_verifies() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert_eq!(verify_delta(&store, &d, NOW), Ok(()));
}

/// Signing is deterministic, so a re-sign of an unchanged delta is stable and
/// a signature can be compared or cached.
#[test]
fn signing_is_deterministic() {
    let a = signed("d1", "worker-3", &WORKER_SEED, "k1");
    let b = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert_eq!(a.producer.signature, b.producer.signature);
}

/// Re-signing an already-signed delta must sign the *unsigned* payload, not
/// one that contains the previous signature. Otherwise the second signature
/// covers different bytes than the first and verification breaks.
#[test]
fn re_signing_produces_the_same_signature() {
    let mut d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    let first = d.producer.signature.clone();
    sign_delta(&mut d, "k1", &WORKER_SEED).unwrap();
    assert_eq!(d.producer.signature, first);

    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    assert_eq!(verify_delta(&store, &d, NOW), Ok(()));
}

// ---------------------------------------------------------------------------
// Forgery
// ---------------------------------------------------------------------------

/// The whole point: someone with their own keypair cannot write to the graph.
#[test]
fn an_attackers_own_key_does_not_authorise_anything() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");

    // Attacker signs a delta claiming to be worker-3, with a valid signature
    // over their own key. The signature is real; the authority is not.
    let mut forged = delta("d-evil", "worker-3");
    sign_delta(&mut forged, "k1", &ATTACKER_SEED).unwrap();

    assert!(matches!(
        verify_delta(&store, &forged, NOW),
        Err(TrustError::BadSignature { .. })
    ));
}

/// A key trusted for one producer must not authorise a delta claiming to be
/// another. Otherwise any legitimate signer could forge everyone else's
/// provenance — and provenance is what every conflict record names.
#[test]
fn a_key_trusted_for_one_producer_cannot_sign_as_another() {
    let mut store = TrustStore::new();
    store.trust("worker-3", key(&WORKER_SEED, "k1"));
    store.trust("worker-9", key(&ATTACKER_SEED, "k1"));

    // worker-9's key, signing a delta that claims to be worker-3.
    let mut forged = delta("d-evil", "worker-3");
    sign_delta(&mut forged, "k1", &ATTACKER_SEED).unwrap();

    assert!(matches!(
        verify_delta(&store, &forged, NOW),
        Err(TrustError::BadSignature { .. })
    ));
}

/// Every field is covered by the signature — tampering anywhere invalidates it.
#[test]
fn tampering_with_any_field_breaks_the_signature() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");

    let mutations: Vec<(&str, fn(&mut GraphDelta))> = vec![
        ("delta id", |d| d.id = "d-other".into()),
        ("producer id", |d| d.producer.id = "someone-else".into()),
        ("emitted_at", |d| d.producer.emitted_at += 1),
        ("source revision", |d| d.producer.source_revision = Some("deadbeef".into())),
        ("claim statement", |d| {
            if let Some(GraphDeltaOp::AddClaim { claim }) = d.operations.first_mut() {
                claim.statement = "auth uses md5".into();
            }
        }),
        ("operation removed", |d| {
            d.operations.clear();
        }),
    ];

    for (what, mutate) in mutations {
        let mut d = signed("d1", "worker-3", &WORKER_SEED, "k1");
        mutate(&mut d);
        assert!(
            verify_delta(&store, &d, NOW).is_err(),
            "tampering with {what} was not detected"
        );
    }
}

/// An unsigned delta is refused outright — no silent acceptance.
#[test]
fn an_unsigned_delta_is_refused() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    assert_eq!(
        verify_delta(&store, &delta("d1", "worker-3"), NOW),
        Err(TrustError::Unsigned)
    );
}

/// A signature claiming a different algorithm must not be waved through —
/// this is the downgrade path.
#[test]
fn an_unsupported_algorithm_is_refused() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let mut d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    d.producer.signature.as_mut().unwrap().algorithm = "none".into();

    assert!(matches!(
        verify_delta(&store, &d, NOW),
        Err(TrustError::UnsupportedAlgorithm { .. })
    ));
}

/// Garbage in the signature field is a clean rejection, not a panic.
#[test]
fn a_malformed_signature_is_rejected_without_panicking() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    for bad in ["", "zz", "not-hex", &"ab".repeat(63), &"ab".repeat(65)] {
        let mut d = signed("d1", "worker-3", &WORKER_SEED, "k1");
        d.producer.signature.as_mut().unwrap().value = bad.to_string();
        assert_eq!(
            verify_delta(&store, &d, NOW),
            Err(TrustError::MalformedSignature),
            "input {bad:?} was not rejected cleanly"
        );
    }
}

/// An empty store trusts nobody. A fresh install must not be open.
#[test]
fn an_empty_store_trusts_nobody() {
    let store = TrustStore::new();
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert!(matches!(
        verify_delta(&store, &d, NOW),
        Err(TrustError::UnknownProducer { .. })
    ));
}

/// A producer that exists but does not have the named key is refused.
#[test]
fn an_unknown_key_id_is_refused() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let d = signed("d1", "worker-3", &WORKER_SEED, "k-other");
    assert!(matches!(
        verify_delta(&store, &d, NOW),
        Err(TrustError::UnknownKey { .. })
    ));
}

// ---------------------------------------------------------------------------
// Key lifecycle
// ---------------------------------------------------------------------------

/// A revoked key stops working immediately, whatever its acceptance window
/// says. A leaked key must not keep working through a rollout period.
#[test]
fn revocation_overrides_the_acceptance_window() {
    let mut store = TrustStore::new();
    store.trust(
        "worker-3",
        TrustedKey {
            accept_until: Some(NOW + 10_000),
            revoked_at: Some(NOW - 1),
            ..key(&WORKER_SEED, "k1")
        },
    );

    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert!(matches!(
        verify_delta(&store, &d, NOW),
        Err(TrustError::KeyRevoked { .. })
    ));
}

/// A rotated key keeps working until its window closes, then stops.
#[test]
fn a_rotated_key_works_through_its_overlap_then_expires() {
    let mut store = TrustStore::new();
    store.trust(
        "worker-3",
        TrustedKey {
            accept_until: Some(NOW + 100),
            ..key(&WORKER_SEED, "k-old")
        },
    );

    let d = signed("d1", "worker-3", &WORKER_SEED, "k-old");
    assert_eq!(verify_delta(&store, &d, NOW), Ok(()), "inside the window");
    assert_eq!(verify_delta(&store, &d, NOW + 100), Ok(()), "boundary is inclusive");
    assert!(matches!(
        verify_delta(&store, &d, NOW + 101),
        Err(TrustError::KeyExpired { .. })
    ));
}

/// A key is not accepted before it is meant to exist.
#[test]
fn a_key_is_not_accepted_before_it_is_active() {
    let mut store = TrustStore::new();
    store.trust(
        "worker-3",
        TrustedKey {
            active_from: NOW + 50,
            ..key(&WORKER_SEED, "k1")
        },
    );

    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert!(matches!(
        verify_delta(&store, &d, NOW),
        Err(TrustError::KeyNotYetActive { .. })
    ));
}

/// Validity is judged by the receiver's clock. A submitter that backdates
/// `emitted_at` to before a revocation must not get in.
#[test]
fn backdating_emitted_at_does_not_revive_a_revoked_key() {
    let mut store = TrustStore::new();
    store.trust(
        "worker-3",
        TrustedKey {
            revoked_at: Some(NOW),
            ..key(&WORKER_SEED, "k1")
        },
    );

    let mut d = delta("d1", "worker-3");
    d.producer.emitted_at = NOW - 10_000; // "I sent this before you revoked me"
    sign_delta(&mut d, "k1", &WORKER_SEED).unwrap();

    // The receiver's clock must decide, not the delta's.
    assert!(matches!(
        verify_delta(&store, &d, NOW + 1),
        Err(TrustError::KeyRevoked { .. })
    ));
}

/// Two keys for one producer: rotation without downtime.
#[test]
fn a_producer_may_hold_several_keys() {
    let mut store = TrustStore::new();
    store.trust("worker-3", key(&WORKER_SEED, "k-old"));
    store.trust("worker-3", key(&ATTACKER_SEED, "k-new"));

    assert_eq!(
        verify_delta(&store, &signed("d1", "worker-3", &WORKER_SEED, "k-old"), NOW),
        Ok(())
    );
    assert_eq!(
        verify_delta(&store, &signed("d2", "worker-3", &ATTACKER_SEED, "k-new"), NOW),
        Ok(())
    );
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/// The signature has to survive the text format, or verification can never
/// happen on the receiving side. This was the gap that made the whole feature
/// inert before the `SIG` line existed.
#[test]
fn a_signature_survives_the_orbitql_round_trip() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let original = signed("d1", "worker-3", &WORKER_SEED, "k1");

    let text = to_orbitql(&original);
    assert!(text.contains("SIG|ed25519|k1|"), "SIG line missing:\n{text}");

    let parsed = from_orbitql(&text).expect("signed document must parse");
    assert_eq!(parsed, original);
    assert_eq!(
        verify_delta(&store, &parsed, NOW),
        Ok(()),
        "signature did not survive the text layer"
    );
}

/// Comments and blank lines do not change what was signed — the payload comes
/// from the typed delta, not from the document bytes.
#[test]
fn incidental_whitespace_does_not_break_verification() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let original = signed("d1", "worker-3", &WORKER_SEED, "k1");

    let mut text = String::from("# a note the worker added\n\n");
    text.push_str(&to_orbitql(&original));
    text.push_str("\n# trailing comment\n");

    let parsed = from_orbitql(&text).unwrap();
    assert_eq!(verify_delta(&store, &parsed, NOW), Ok(()));
}

/// Tampering with the document text after signing is caught.
#[test]
fn editing_the_document_after_signing_is_caught() {
    let store = store_trusting("worker-3", &WORKER_SEED, "k1");
    let original = signed("d1", "worker-3", &WORKER_SEED, "k1");

    let tampered = to_orbitql(&original).replace("auth uses bcrypt", "auth uses md5");
    let parsed = from_orbitql(&tampered).unwrap();

    assert!(matches!(
        verify_delta(&store, &parsed, NOW),
        Err(TrustError::BadSignature { .. })
    ));
}

/// The trust store is operator-edited, so it has to survive its own file format.
#[test]
fn trust_store_round_trips_through_json() {
    let mut store = TrustStore::new();
    store.trust(
        "worker-3",
        TrustedKey {
            accept_until: Some(NOW + 100),
            comment: Some("laptop key, rotate in Q4".into()),
            ..key(&WORKER_SEED, "k1")
        },
    );

    let parsed = TrustStore::from_json(&store.to_json().unwrap()).unwrap();
    assert_eq!(parsed, store);
}

/// Trusting the same key id twice replaces rather than duplicates, so an
/// operator correcting a typo does not end up with two live entries.
#[test]
fn trusting_the_same_key_id_twice_replaces_it() {
    let mut store = TrustStore::new();
    store.trust("worker-3", key(&ATTACKER_SEED, "k1"));
    store.trust("worker-3", key(&WORKER_SEED, "k1"));

    assert_eq!(store.producers["worker-3"].len(), 1);
    assert_eq!(
        verify_delta(&store, &signed("d1", "worker-3", &WORKER_SEED, "k1"), NOW),
        Ok(())
    );
}

/// A signature never leaks the private seed into the document.
#[test]
fn the_document_never_contains_key_material() {
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    let text = to_orbitql(&d);
    let seed_hex: String = WORKER_SEED.iter().map(|b| format!("{b:02x}")).collect();
    assert!(!text.contains(&seed_hex), "the seed leaked into the document");
}

// ---------------------------------------------------------------------------
// The merge gate
// ---------------------------------------------------------------------------

use qo_knowledge::merge::{merge_signed_delta, SubmitError};
use qo_knowledge::model::ClaimId;
use qo_knowledge::KnowledgeStore;

fn knowledge_store() -> (tempfile::TempDir, KnowledgeStore) {
    let dir = tempfile::tempdir().unwrap();
    let store = KnowledgeStore::open(dir.path().join("k.redb")).unwrap();
    (dir, store)
}

/// A trusted, signed delta merges normally.
#[test]
fn a_trusted_delta_merges() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");

    let report = merge_signed_delta(&store, &trust, &d, NOW).unwrap();
    assert_eq!(report.applied(), 1);
    assert!(store.latest(&ClaimId("c1".into())).unwrap().is_some());
}

/// An untrusted delta writes nothing at all — the gate runs before any
/// operation is applied, not per-operation.
#[test]
fn an_untrusted_delta_writes_nothing() {
    let (_dir, store) = knowledge_store();
    let trust = TrustStore::new(); // trusts nobody
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");

    let error = merge_signed_delta(&store, &trust, &d, NOW).unwrap_err();
    assert!(matches!(error, SubmitError::Untrusted(_)));
    assert!(
        store.latest(&ClaimId("c1".into())).unwrap().is_none(),
        "an unauthorised delta reached the graph"
    );
}

/// A captured delta cannot be re-submitted. The merge would be idempotent
/// anyway, but a replay is reported rather than passed off as a retry.
#[test]
fn replaying_a_delta_is_refused() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");

    assert!(merge_signed_delta(&store, &trust, &d, NOW).is_ok());

    let error = merge_signed_delta(&store, &trust, &d, NOW).unwrap_err();
    assert!(matches!(error, SubmitError::Replay { .. }), "{error}");
}

/// Reusing a delta id with different content is caught by the same guard —
/// the replay record is keyed on the id, so a second submission under that id
/// cannot slip through by changing its payload.
#[test]
fn reusing_a_delta_id_with_new_content_is_refused() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");

    assert!(merge_signed_delta(&store, &trust, &signed("d1", "worker-3", &WORKER_SEED, "k1"), NOW).is_ok());

    let mut second = delta("d1", "worker-3");
    if let Some(GraphDeltaOp::AddClaim { claim }) = second.operations.first_mut() {
        claim.statement = "something entirely different".into();
    }
    sign_delta(&mut second, "k1", &WORKER_SEED).unwrap();

    assert!(matches!(
        merge_signed_delta(&store, &trust, &second, NOW).unwrap_err(),
        SubmitError::Replay { .. }
    ));
}

/// Different ids from the same producer are independent submissions.
#[test]
fn distinct_deltas_from_one_producer_all_apply() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");

    assert!(merge_signed_delta(&store, &trust, &signed("d1", "worker-3", &WORKER_SEED, "k1"), NOW).is_ok());
    assert!(merge_signed_delta(&store, &trust, &signed("d2", "worker-3", &WORKER_SEED, "k1"), NOW).is_ok());
}

/// The replay guard is scoped per producer: two workers may each use "d1".
#[test]
fn the_replay_guard_is_scoped_per_producer() {
    let (_dir, store) = knowledge_store();
    let mut trust = TrustStore::new();
    trust.trust("worker-3", key(&WORKER_SEED, "k1"));
    trust.trust("worker-9", key(&ATTACKER_SEED, "k1"));

    assert!(merge_signed_delta(&store, &trust, &signed("d1", "worker-3", &WORKER_SEED, "k1"), NOW).is_ok());
    // Same delta id, different producer — a separate submission.
    let other = signed("d1", "worker-9", &ATTACKER_SEED, "k1");
    let result = merge_signed_delta(&store, &trust, &other, NOW);
    assert!(!matches!(result, Err(SubmitError::Replay { .. })), "producers collided");
}

/// An invalid delta must not burn its id. Recording the id before validating
/// would let anyone who can guess a peer's next delta id lock them out of it
/// by submitting garbage under that name.
#[test]
fn an_invalid_delta_does_not_consume_its_id() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");

    // Empty operations violate the delta contract.
    let mut invalid = delta("d1", "worker-3");
    invalid.operations.clear();
    sign_delta(&mut invalid, "k1", &WORKER_SEED).unwrap();
    assert!(merge_signed_delta(&store, &trust, &invalid, NOW).is_err());

    // The same id must still be usable for a well-formed submission.
    let valid = signed("d1", "worker-3", &WORKER_SEED, "k1");
    let report = merge_signed_delta(&store, &trust, &valid, NOW)
        .expect("a rejected delta must not consume its id");
    assert_eq!(report.applied(), 1);
}

/// An unauthorised delta likewise leaves no trace — the id stays free.
#[test]
fn an_untrusted_delta_does_not_consume_its_id() {
    let (_dir, store) = knowledge_store();
    let trust = store_trusting("worker-3", &WORKER_SEED, "k1");

    let mut forged = delta("d1", "worker-3");
    sign_delta(&mut forged, "k1", &ATTACKER_SEED).unwrap();
    assert!(merge_signed_delta(&store, &trust, &forged, NOW).is_err());

    let genuine = signed("d1", "worker-3", &WORKER_SEED, "k1");
    assert!(
        merge_signed_delta(&store, &trust, &genuine, NOW).is_ok(),
        "a forgery locked the legitimate producer out of its own delta id"
    );
}

/// Two deltas that differ in any way must produce different signing bytes.
/// A collision here would let a signature be lifted from one delta onto
/// another — the signature would verify over content nobody signed.
#[test]
fn signing_payloads_do_not_collide() {
    let mut variants: Vec<(String, Vec<u8>)> = Vec::new();

    let mut push = |label: &str, d: &GraphDelta| {
        variants.push((label.to_string(), d.signing_payload().unwrap()));
    };

    push("baseline", &delta("d1", "worker-3"));

    let mut other_id = delta("d2", "worker-3");
    other_id.id = "d2".into();
    push("different id", &other_id);

    push("different producer", &delta("d1", "worker-9"));

    let mut later = delta("d1", "worker-3");
    later.producer.emitted_at += 1;
    push("different timestamp", &later);

    let mut no_rev = delta("d1", "worker-3");
    no_rev.producer.source_revision = None;
    push("absent source revision", &no_rev);

    // The classic concatenation ambiguity: moving a character across a field
    // boundary must not produce the same bytes.
    let mut split_a = delta("ab", "worker-3");
    split_a.producer.id = "c".into();
    push("id=ab producer=c", &split_a);
    let mut split_b = delta("a", "worker-3");
    split_b.producer.id = "bc".into();
    push("id=a producer=bc", &split_b);

    let mut empty_ops = delta("d1", "worker-3");
    empty_ops.operations.clear();
    push("no operations", &empty_ops);

    for (i, (label_a, a)) in variants.iter().enumerate() {
        for (label_b, b) in variants.iter().skip(i + 1) {
            assert_ne!(a, b, "{label_a} and {label_b} share signing bytes");
        }
    }
}

/// The signature covers a delta of this version only. A payload must not be
/// reinterpretable as one from a different protocol version.
#[test]
fn the_signing_payload_is_domain_separated() {
    let payload = delta("d1", "worker-3").signing_payload().unwrap();
    let text = String::from_utf8_lossy(&payload);
    assert!(
        text.starts_with("orbitqlang.graph-delta.v1\n"),
        "missing or wrong domain tag: {}",
        &text[..text.len().min(40)]
    );
}

/// The payload must not contain the signature field, or signing would be
/// circular and a verifier could never reproduce the bytes.
#[test]
fn the_signing_payload_excludes_the_signature() {
    let d = signed("d1", "worker-3", &WORKER_SEED, "k1");
    let payload = String::from_utf8(d.signing_payload().unwrap()).unwrap();
    let value = &d.producer.signature.as_ref().unwrap().value;
    assert!(!payload.contains(value.as_str()), "payload contains its own signature");

    assert!(
        payload.contains("\"signature\":null"),
        "the signature field should be present but cleared"
    );

    // Signing twice must cover identical bytes — otherwise the second
    // signature would not verify against what the first one covered. (Compare
    // against a re-signed copy rather than a raw one: `sign_delta` also
    // normalises claim provenance, so an unsigned fixture legitimately
    // differs.)
    let mut again = d.clone();
    sign_delta(&mut again, "k1", &WORKER_SEED).unwrap();
    assert_eq!(d.signing_payload().unwrap(), again.signing_payload().unwrap());
}
