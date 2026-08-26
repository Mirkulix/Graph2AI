//! Cryptographic primitives for QLANG protocol security.
//!
//! Hashing ([`sha256`], [`hmac_sha256`], [`hash_graph`], [`ct_eq`]) is
//! implemented in-tree, per the project rule in `CLAUDE.md`.
//!
//! Signing is not. [`Keypair`] wraps `ed25519-dalek`, because the previous
//! hand-rolled scheme turned out to be forgeable from public data alone —
//! see the note above [`Keypair`] for what went wrong and why a dependency
//! is the right call for signatures specifically.

use crate::graph::Graph;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) -- minimal implementation
// ---------------------------------------------------------------------------

/// Round constants for SHA-256 (first 32 bits of the fractional parts
/// of the cube roots of the first 64 primes).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Compute SHA-256 hash of arbitrary data.
///
/// Returns a 32-byte digest.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    // Initial hash values (first 32 bits of the fractional parts
    // of the square roots of the first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;

    // Pre-processing: pad message
    // Append bit '1' (0x80 byte), then zeros, then 64-bit big-endian length
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in msg.chunks_exact(64) {
        // Prepare message schedule
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // Initialize working variables
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        // Compression function
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    // Produce final hash
    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    result
}

/// Format a hash as lowercase hex string.
pub fn hex(hash: &[u8]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Graph hashing
// ---------------------------------------------------------------------------

/// Hash a QLANG graph deterministically.
///
/// Serializes the graph to JSON, then computes SHA-256.
/// The JSON serialization is deterministic because serde_json
/// preserves field order (struct fields are always in declaration order).
pub fn hash_graph(graph: &Graph) -> [u8; 32] {
    let json = serde_json::to_vec(graph).expect("graph serialization cannot fail");
    sha256(&json)
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

/// HMAC-SHA256: keyed hash for signing.
pub fn hmac_sha256(key: &[u8; 32], message: &[u8]) -> [u8; 32] {
    // HMAC(K, m) = SHA-256((K ^ opad) || SHA-256((K ^ ipad) || m))
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];

    for i in 0..32 {
        ipad[i] ^= key[i];
        opad[i] ^= key[i];
    }

    // Inner hash: SHA-256(ipad || message)
    let mut inner_input = Vec::with_capacity(64 + message.len());
    inner_input.extend_from_slice(&ipad);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // Outer hash: SHA-256(opad || inner_hash)
    let mut outer_input = Vec::with_capacity(64 + 32);
    outer_input.extend_from_slice(&opad);
    outer_input.extend_from_slice(&inner_hash);
    sha256(&outer_input)
}

/// Constant-time byte comparison. Returns true if a == b.
/// Takes same time regardless of where bytes differ (prevents timing attacks).
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Deprecated alias — kept for backward compatibility. Use `ct_eq`.
#[doc(hidden)]
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    ct_eq(a, b)
}

// ---------------------------------------------------------------------------
// Signing: Ed25519 (RFC 8032) via ed25519-dalek
// ---------------------------------------------------------------------------
//
// Wire format, unchanged from the previous scheme:
//   - 32-byte seed (secret) -> 32-byte public key
//   - sign(message) -> 64-byte signature
//   - verify(pubkey, message, signature) -> bool
//
// ## Why this is not hand-rolled
//
// This module previously implemented signing from SHA-256 and HMAC:
//
//     r   = HMAC-SHA256(secret, message)
//     s   = SHA-256(SHA-256(r || pubkey || message))
//     sig = r || s
//
// That scheme was forgeable. `verify` recomputed `s` from `r`, `pubkey` and
// `message` — all public — and never touched the secret, and nothing
// constrained `r`. An attacker holding only the public key could pick any
// `r`, compute the matching `s`, and produce a signature that verified for
// an arbitrary message. A proof-of-concept confirmed it end to end.
//
// The lesson is the reason for the dependency: the flaw was not a typo but
// a design error that read as plausible for as long as nobody attacked it.
// `CLAUDE.md` prefers in-tree crypto and that rule still holds for hashes —
// `sha256` and `hmac_sha256` below are ours and are used for content
// addressing. Signature schemes are the exception: ed25519-dalek is audited
// and constant-time, and reimplementing curve arithmetic here would
// reintroduce exactly the class of bug this replaces.
//
// Note: signatures produced by the old scheme do not verify under Ed25519.
// That is intended — they never carried authenticity to begin with.

/// A signing keypair.
///
/// Holds a 32-byte Ed25519 seed and its derived 32-byte public key.
/// Derivation follows RFC 8032, so a given seed always yields the same key.
#[derive(Clone)]
pub struct Keypair {
    signing: ed25519_dalek::SigningKey,
}

impl Keypair {
    /// Create a keypair from a 32-byte seed.
    ///
    /// Deterministic: the same seed always produces the same keypair, per
    /// RFC 8032 key derivation. Every 32-byte value is a valid seed, so this
    /// cannot fail.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        Self {
            signing: ed25519_dalek::SigningKey::from_bytes(seed),
        }
    }

    /// Generate a keypair from operating-system entropy.
    ///
    /// Uses `getrandom`, which reads the OS CSPRNG (`BCryptGenRandom` on
    /// Windows, `getrandom(2)` on Linux). Panics if the OS cannot supply
    /// entropy — continuing with a predictable key would be worse than
    /// failing loudly.
    ///
    /// An earlier version seeded from `SystemTime` nanos through xorshift,
    /// which made generated keys guessable to anyone who knew roughly when
    /// they were created.
    pub fn generate() -> Self {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("OS entropy source unavailable");
        Self::from_seed(&seed)
    }

    /// Get the public key.
    pub fn public_key(&self) -> [u8; 32] {
        self.signing.verifying_key().to_bytes()
    }

    /// Return the 32-byte seed this keypair was built from.
    ///
    /// Handle as secret material: anyone holding it can sign as this agent.
    pub fn secret_seed(&self) -> [u8; 32] {
        self.signing.to_bytes()
    }

    /// Sign a message, producing a 64-byte Ed25519 signature.
    pub fn sign(&self, data: &[u8]) -> [u8; 64] {
        use ed25519_dalek::Signer;
        self.signing.sign(data).to_bytes()
    }

    /// Verify a signature against a message and public key.
    ///
    /// Returns `false` — never panics — if the signature does not verify or
    /// the public key is not a valid Ed25519 point.
    pub fn verify(pubkey: &[u8; 32], data: &[u8], sig: &[u8; 64]) -> bool {
        use ed25519_dalek::Verifier;
        let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(pubkey) else {
            return false;
        };
        vk.verify(data, &ed25519_dalek::Signature::from_bytes(sig))
            .is_ok()
    }
}

impl std::fmt::Debug for Keypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keypair")
            .field("public", &hex(&self.public_key()))
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SignedGraph -- a Graph bundled with its cryptographic signature
// ---------------------------------------------------------------------------

/// A QLANG graph with a cryptographic signature.
///
/// Proves that a specific agent (identified by public key) produced this
/// exact graph. Any modification to the graph invalidates the signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedGraph {
    /// The computation graph.
    pub graph: Graph,
    /// Signature (64 bytes), serialized as a Vec for serde compatibility.
    #[serde(with = "fixed_64")]
    pub signature: [u8; 64],
    /// Signer's public key (32 bytes).
    #[serde(with = "fixed_32")]
    pub pubkey: [u8; 32],
    /// SHA-256 hash of the graph at signing time.
    #[serde(with = "fixed_32")]
    pub hash: [u8; 32],
}

impl SignedGraph {
    /// Sign a graph with a keypair.
    ///
    /// Computes the graph hash, signs it, and bundles everything together.
    pub fn sign(graph: Graph, keypair: &Keypair) -> Self {
        let hash = hash_graph(&graph);
        let signature = keypair.sign(&hash);

        Self {
            graph,
            signature,
            pubkey: keypair.public_key(),
            hash,
        }
    }

    /// Verify the signature on this graph.
    ///
    /// Checks:
    /// 1. The stored hash matches the current graph content.
    /// 2. The signature is valid for the stored hash and public key.
    pub fn verify(&self) -> bool {
        // Check hash matches current graph
        let current_hash = hash_graph(&self.graph);
        if !constant_time_eq(&current_hash, &self.hash) {
            return false;
        }

        // Check signature
        Keypair::verify(&self.pubkey, &current_hash, &self.signature)
    }
}

// ---------------------------------------------------------------------------
// Serde helpers for fixed-size byte arrays
// ---------------------------------------------------------------------------

mod fixed_32 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8; 32], ser: S) -> Result<S::Ok, S::Error> {
        data.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[u8; 32], D::Error> {
        let v: Vec<u8> = Vec::deserialize(de)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

mod fixed_64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8; 64], ser: S) -> Result<S::Ok, S::Error> {
        data.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[u8; 64], D::Error> {
        let v: Vec<u8> = Vec::deserialize(de)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 bytes"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SHA-256 known test vectors ----

    #[test]
    fn sha256_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256(b"");
        assert_eq!(
            hex(&hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hash = sha256(b"abc");
        assert_eq!(
            hex(&hash),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_longer_message() {
        // SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
        let hash = sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&hash),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_deterministic() {
        let h1 = sha256(b"QLANG graph data");
        let h2 = sha256(b"QLANG graph data");
        assert_eq!(h1, h2);

        let h3 = sha256(b"QLANG graph data modified");
        assert_ne!(h1, h3);
    }

    #[test]
    fn sha256_exactly_56_bytes() {
        // Edge case: message length == 56 bytes (padding boundary).
        let data = vec![0x41u8; 56];
        let hash = sha256(&data);
        assert_eq!(hash.len(), 32);
        assert_eq!(sha256(&data), hash);
    }

    #[test]
    fn sha256_exactly_64_bytes() {
        // Edge case: message length == 64 bytes (one full block before padding).
        let data = vec![0x42u8; 64];
        let hash = sha256(&data);
        assert_eq!(hash.len(), 32);
        assert_eq!(sha256(&data), hash);
    }

    // ---- Graph hashing ----

    #[test]
    fn hash_graph_deterministic() {
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let mut g = Graph::new("hash_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        g.add_node(
            Op::Relu,
            vec![TensorType::f32_vector(4)],
            vec![TensorType::f32_vector(4)],
        );
        g.add_edge(0, 0, 1, 0, TensorType::f32_vector(4));

        let h1 = hash_graph(&g);
        let h2 = hash_graph(&g);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_graph_changes_with_graph() {
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let mut g1 = Graph::new("graph_a");
        g1.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );

        let mut g2 = Graph::new("graph_b");
        g2.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );

        assert_ne!(hash_graph(&g1), hash_graph(&g2));
    }

    // ---- Keypair generation ----

    #[test]
    fn keypair_from_seed_deterministic() {
        let kp1 = Keypair::from_seed(&[42u8; 32]);
        let kp2 = Keypair::from_seed(&[42u8; 32]);
        assert_eq!(kp1.public_key(), kp2.public_key());

        let kp3 = Keypair::from_seed(&[99u8; 32]);
        assert_ne!(kp1.public_key(), kp3.public_key());
    }

    #[test]
    fn keypair_generate_produces_different_keys() {
        let kp1 = Keypair::generate();
        let kp2 = Keypair::generate();
        assert_ne!(kp1.public_key(), kp2.public_key());
    }

    #[test]
    fn public_key_is_32_bytes() {
        let kp = Keypair::from_seed(&[6u8; 32]);
        assert_eq!(kp.public_key().len(), 32);
    }

    // ---- Signing ----

    #[test]
    fn sign_and_verify_roundtrip() {
        let kp = Keypair::from_seed(&[1u8; 32]);
        let message = b"hello QLANG";
        let sig = kp.sign(message);

        assert!(Keypair::verify(&kp.public_key(), message, &sig));
    }

    #[test]
    fn signature_is_64_bytes() {
        let kp = Keypair::from_seed(&[5u8; 32]);
        let sig = kp.sign(b"test");
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn verify_rejects_tampered_data() {
        let kp = Keypair::from_seed(&[2u8; 32]);
        let sig = kp.sign(b"original");

        assert!(!Keypair::verify(&kp.public_key(), b"tampered", &sig));
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let kp1 = Keypair::from_seed(&[3u8; 32]);
        let kp2 = Keypair::from_seed(&[4u8; 32]);
        let sig = kp1.sign(b"data");

        assert!(!Keypair::verify(&kp2.public_key(), b"data", &sig));
    }

    // ---- SignedGraph ----

    #[test]
    fn signed_graph_roundtrip() {
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let kp = Keypair::from_seed(&[7u8; 32]);

        let mut g = Graph::new("signed_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );

        let signed = SignedGraph::sign(g, &kp);
        assert!(signed.verify());
    }

    // ---- Constant-time equality ----

    #[test]
    fn test_ct_eq_finds_differences() {
        // 1000 equal pairs
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };

        for _ in 0..1000 {
            let mut a = [0u8; 32];
            for chunk in a.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rnd().to_le_bytes());
            }
            let b = a;
            assert!(ct_eq(&a, &b), "equal pairs must compare equal");
        }

        // 1000 random unequal pairs
        for _ in 0..1000 {
            let mut a = [0u8; 32];
            let mut b = [0u8; 32];
            for chunk in a.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rnd().to_le_bytes());
            }
            for chunk in b.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rnd().to_le_bytes());
            }
            // Overwhelmingly unequal; if equal by chance, flip a bit.
            if a == b {
                b[0] ^= 1;
            }
            assert!(!ct_eq(&a, &b), "random pairs must compare unequal");
        }

        // Edge: different lengths
        assert!(!ct_eq(&[1, 2, 3], &[1, 2, 3, 4]));
        // Edge: empty
        assert!(ct_eq(&[], &[]));
        // Edge: single-byte difference at end
        let mut a = [0u8; 64];
        let mut b = [0u8; 64];
        b[63] = 1;
        assert!(!ct_eq(&a, &b));
        a[63] = 1;
        assert!(ct_eq(&a, &b));
    }

    #[test]
    fn test_ct_eq_timing_statistical() {
        // Informational only — measures mean/stddev of timing for three cases:
        //   (1) equal buffers
        //   (2) buffers differing at the first byte
        //   (3) buffers differing at the last byte
        // A timing-safe impl should have similar means. We do NOT assert a
        // strict bound (flaky on CI), but we print the numbers for audit.
        use std::time::Instant;

        let a = [0x42u8; 4096];
        let b = a;
        let mut c = a;
        c[0] ^= 1;
        let mut d = a;
        d[4095] ^= 1;

        let iters = 20_000;

        let mut warm = 0u64;
        for _ in 0..1000 {
            warm = warm.wrapping_add(ct_eq(&a, &b) as u64);
        }
        std::hint::black_box(warm);

        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc = acc.wrapping_add(ct_eq(&a, &b) as u64);
        }
        let eq_ns = t0.elapsed().as_nanos() / iters as u128;
        std::hint::black_box(acc);

        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc = acc.wrapping_add(ct_eq(&a, &c) as u64);
        }
        let diff_first_ns = t0.elapsed().as_nanos() / iters as u128;
        std::hint::black_box(acc);

        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..iters {
            acc = acc.wrapping_add(ct_eq(&a, &d) as u64);
        }
        let diff_last_ns = t0.elapsed().as_nanos() / iters as u128;
        std::hint::black_box(acc);

        eprintln!(
            "ct_eq timing (4096B, {} iters): equal={}ns, diff@first={}ns, diff@last={}ns",
            iters, eq_ns, diff_first_ns, diff_last_ns
        );
        // Sanity: all three must be non-zero (compiler must not elide).
        assert!(eq_ns > 0);
        assert!(diff_first_ns > 0);
        assert!(diff_last_ns > 0);
    }

    #[test]
    fn signed_graph_detects_tampering() {
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let kp = Keypair::from_seed(&[8u8; 32]);

        let mut g = Graph::new("tamper_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );

        let mut signed = SignedGraph::sign(g, &kp);
        // Tamper with the graph
        signed.graph.metadata.insert("evil".into(), "data".into());
        assert!(!signed.verify());
    }
}
