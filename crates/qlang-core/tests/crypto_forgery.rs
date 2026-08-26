//! Regression tests for the signature scheme.
//!
//! These exist because the previous scheme was forgeable from public data
//! alone. `attacker_cannot_forge_from_public_key_alone` is the exact attack
//! that used to succeed — if it ever passes again, signing has been
//! replaced by something that does not authenticate.

use qlang_core::crypto::{sha256, Keypair};

/// The original break: `verify` recomputed the second half of the signature
/// from `r`, the public key and the message — all public — and never
/// consulted the secret. An attacker could pick any `r` and derive a
/// matching `s`.
///
/// Under Ed25519 this must fail.
#[test]
fn attacker_cannot_forge_from_public_key_alone() {
    let victim = Keypair::from_seed(&[7u8; 32]);
    let pubkey = victim.public_key();
    let message = b"TRANSFER ALL FUNDS TO ATTACKER";

    // Attacker's side: only `pubkey` and `message` are used. No secret.
    let r = [0x41u8; 32];
    let mut tag_input = Vec::new();
    tag_input.extend_from_slice(&r);
    tag_input.extend_from_slice(&pubkey);
    tag_input.extend_from_slice(message);
    let s = sha256(&sha256(&tag_input));

    let mut forged = [0u8; 64];
    forged[..32].copy_from_slice(&r);
    forged[32..].copy_from_slice(&s);

    assert!(
        !Keypair::verify(&pubkey, message, &forged),
        "signature forged from public data alone was accepted"
    );
}

/// A signature must not verify for a message other than the one signed.
#[test]
fn signature_does_not_transfer_to_another_message() {
    let kp = Keypair::from_seed(&[3u8; 32]);
    let sig = kp.sign(b"approve payment of 10");

    assert!(Keypair::verify(&kp.public_key(), b"approve payment of 10", &sig));
    assert!(!Keypair::verify(
        &kp.public_key(),
        b"approve payment of 1000000",
        &sig
    ));
}

/// A signature must not verify under a different key.
#[test]
fn signature_does_not_verify_under_another_key() {
    let alice = Keypair::from_seed(&[1u8; 32]);
    let mallory = Keypair::from_seed(&[2u8; 32]);
    let msg = b"from alice";
    let sig = alice.sign(msg);

    assert!(Keypair::verify(&alice.public_key(), msg, &sig));
    assert!(!Keypair::verify(&mallory.public_key(), msg, &sig));
}

/// Flipping any single bit of the signature must invalidate it.
#[test]
fn every_signature_bit_matters() {
    let kp = Keypair::from_seed(&[5u8; 32]);
    let msg = b"integrity";
    let sig = kp.sign(msg);

    for byte in 0..64 {
        for bit in 0..8 {
            let mut tampered = sig;
            tampered[byte] ^= 1 << bit;
            assert!(
                !Keypair::verify(&kp.public_key(), msg, &tampered),
                "signature still verified after flipping bit {bit} of byte {byte}"
            );
        }
    }
}

/// A malformed public key must return false, not panic.
#[test]
fn invalid_public_key_is_rejected_without_panicking() {
    let kp = Keypair::from_seed(&[9u8; 32]);
    let msg = b"hello";
    let sig = kp.sign(msg);

    // All-ones is not a valid compressed Edwards point.
    assert!(!Keypair::verify(&[0xFFu8; 32], msg, &sig));
    // Nor is an arbitrary non-canonical value.
    assert!(!Keypair::verify(&[0xEEu8; 32], msg, &sig));
}

/// Key derivation is deterministic, so a persisted seed round-trips.
#[test]
fn same_seed_yields_same_key_and_signature() {
    let a = Keypair::from_seed(&[42u8; 32]);
    let b = Keypair::from_seed(&[42u8; 32]);
    let msg = b"deterministic";

    assert_eq!(a.public_key(), b.public_key());
    // Ed25519 is deterministic (RFC 8032), so signatures match too.
    assert_eq!(a.sign(msg), b.sign(msg));
    assert!(Keypair::verify(&b.public_key(), msg, &a.sign(msg)));
}

/// Different seeds must give different identities.
#[test]
fn different_seeds_yield_different_keys() {
    let a = Keypair::from_seed(&[1u8; 32]);
    let b = Keypair::from_seed(&[2u8; 32]);
    assert_ne!(a.public_key(), b.public_key());
}

/// `generate()` must use real entropy — two calls must not collide.
///
/// The previous implementation seeded from `SystemTime` nanos through
/// xorshift, which made keys guessable from the approximate creation time.
#[test]
fn generate_produces_unpredictable_distinct_keys() {
    let keys: Vec<[u8; 32]> = (0..16).map(|_| Keypair::generate().public_key()).collect();
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            assert_ne!(keys[i], keys[j], "generate() produced a duplicate key");
        }
    }
}

/// The seed round-trips, so an agent identity can be persisted and restored.
#[test]
fn secret_seed_round_trips() {
    let original = Keypair::generate();
    let restored = Keypair::from_seed(&original.secret_seed());

    assert_eq!(original.public_key(), restored.public_key());
    let msg = b"restored identity";
    assert!(Keypair::verify(&original.public_key(), msg, &restored.sign(msg)));
}

/// Signing an empty message is well defined.
#[test]
fn empty_message_is_signable() {
    let kp = Keypair::from_seed(&[11u8; 32]);
    let sig = kp.sign(b"");
    assert!(Keypair::verify(&kp.public_key(), b"", &sig));
    assert!(!Keypair::verify(&kp.public_key(), b"x", &sig));
}

/// The Debug impl must never leak secret material.
#[test]
fn debug_does_not_leak_the_secret() {
    let kp = Keypair::from_seed(&[0xABu8; 32]);
    let rendered = format!("{:?}", kp);

    assert!(rendered.contains("REDACTED"));
    // The seed as hex must not appear anywhere in the output.
    let seed_hex: String = kp.secret_seed().iter().map(|b| format!("{:02x}", b)).collect();
    assert!(!rendered.contains(&seed_hex));
}
