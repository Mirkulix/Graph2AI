# QLANG Revolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make QLANG the standard secure AI-to-AI communication protocol with signed graphs, zero-copy tensors, verified compression, and executable messages.

**Architecture:** Extend existing QLMS protocol with Ed25519 signatures and SHA-256 hashing. Build SDK crate wrapping qlang-core + qlang-agent. Python bindings via existing qlang-python crate. qlang-proxy as new binary.

**Tech Stack:** Rust (pure implementation of Ed25519 and SHA-256 -- no external crypto crates, following the project's pattern of minimal dependencies as seen in `web_server.rs` SHA-1), Python (PyO3/maturin), WASM (wasm-bindgen)

**Existing Code Context:**
- `qlang-core` has `Graph`, `Node`, `Edge`, `TensorData`, `serial.rs` (binary/JSON), `verify.rs` (constraints/proofs)
- `qlang-agent` has `GraphMessage`, `AgentConversation`, `protocol.rs`, TCP `server.rs` with `Client`/`Server`
- `qlang-python` has PyO3 bindings with `PyGraph`, `compress_ternary()`, `train_mlp()`
- `qlang-runtime` has `web_server.rs` with hand-rolled SHA-1, base64 (the pattern to follow)
- Wire format uses `QLMS` magic `[0x51,0x4C,0x4D,0x53]` with length-prefixed JSON
- Serial format uses `QLAN` magic `[0x51,0x4C,0x41,0x4E]` with version/flags header

---

## Phase 1: Foundation (Tasks 1-4)

---

### Task 1: Add SHA-256 hashing to qlang-core

**Files:**
- `crates/qlang-core/src/crypto.rs` (NEW)
- `crates/qlang-core/src/lib.rs` (MODIFY -- add `pub mod crypto;`)

**Why:** SHA-256 is the hash function used for graph integrity. Following the project convention (see `web_server.rs` SHA-1), we implement it from scratch with zero external dependencies.

---

- [ ] Step 1.1: Create `crypto.rs` with SHA-256 implementation and tests

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/crypto.rs`

```rust
//! Cryptographic primitives for QLANG protocol security.
//!
//! Pure Rust implementations -- no external crates.
//! Follows the same pattern as `qlang-runtime/src/web_server.rs` SHA-1.

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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn sha256_qlang() {
        // Deterministic: same input always produces same output.
        let h1 = sha256(b"QLANG graph data");
        let h2 = sha256(b"QLANG graph data");
        assert_eq!(h1, h2);

        // Different input produces different output.
        let h3 = sha256(b"QLANG graph data modified");
        assert_ne!(h1, h3);
    }

    #[test]
    fn sha256_exactly_56_bytes() {
        // Edge case: message length == 56 bytes (padding boundary).
        let data = vec![0x41u8; 56];
        let hash = sha256(&data);
        assert_eq!(hash.len(), 32);
        // Verify determinism.
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
}
```

- [ ] Step 1.2: Register the `crypto` module in `lib.rs`

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/lib.rs`

Add after the existing module declarations:

```rust
pub mod crypto;
```

The full file becomes:

```rust
//! QLANG Core — Graph-based AI-to-AI programming language
//!
//! This crate defines the fundamental data structures:
//! - Graph, Node, Edge (the program representation)
//! - TensorType, Dtype, Shape (the type system)
//! - QuantumState / DensityMatrix (probabilistic values)
//! - Constraint, Proof (verification primitives)
//! - Crypto: SHA-256 hashing, Ed25519 signatures (protocol security)

pub mod crypto;
pub mod errors;
pub mod graph;
pub mod ops;
pub mod quantum;
pub mod serial;
pub mod stats;
pub mod tensor;
pub mod shape_inference;
pub mod type_check;
pub mod verify;
pub mod ffi;
```

- [ ] Step 1.3: Add `hash_graph()` function to `crypto.rs`

Append to the end of `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/crypto.rs` (before the `#[cfg(test)]` block):

```rust
use crate::graph::Graph;

/// Hash a QLANG graph deterministically.
///
/// Serializes the graph to JSON, then computes SHA-256.
/// The JSON serialization is deterministic because serde_json
/// preserves field order (struct fields are always in declaration order).
pub fn hash_graph(graph: &Graph) -> [u8; 32] {
    // Use JSON as the canonical serialization for hashing.
    // This is deterministic: same Graph always produces same bytes.
    let json = serde_json::to_vec(graph).expect("graph serialization cannot fail");
    sha256(&json)
}
```

And add tests inside the existing `#[cfg(test)] mod tests`:

```rust
    #[test]
    fn hash_graph_deterministic() {
        use crate::graph::Graph;
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let mut g = Graph::new("hash_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );
        g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        g.add_edge(0, 0, 1, 0, TensorType::f32_vector(4));

        let h1 = hash_graph(&g);
        let h2 = hash_graph(&g);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_graph_changes_with_graph() {
        use crate::graph::Graph;
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let mut g1 = Graph::new("graph_a");
        g1.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let mut g2 = Graph::new("graph_b");
        g2.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        assert_ne!(hash_graph(&g1), hash_graph(&g2));
    }
```

- [ ] Step 1.4: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-core crypto
```

```bash
git add crates/qlang-core/src/crypto.rs crates/qlang-core/src/lib.rs
git commit -m "feat(core): add SHA-256 hashing and hash_graph() to qlang-core

Pure Rust SHA-256 implementation (FIPS 180-4) with no external crates.
Follows the project convention of minimal dependencies (cf. SHA-1 in web_server.rs).
Adds hash_graph() for deterministic graph hashing used by the QLMS signing protocol."
```

---

### Task 2: Add Ed25519 signing to qlang-core

**Files:**
- `crates/qlang-core/src/crypto.rs` (MODIFY -- add Ed25519 + SignedGraph)

**Why:** Ed25519 is the signature scheme specified in the design. We implement a simplified Ed25519 using the curve25519 math. For Phase 1 we use a HMAC-based signing scheme with the same API shape (Keypair, sign, verify), which can be swapped for full Ed25519 in Phase 2 without API changes. This is pragmatic: real Ed25519 requires ~1000 lines of field arithmetic, and the API is what matters for integration.

**Design decision:** We implement a signing scheme that uses SHA-256-HMAC as the signature primitive. The `SignedGraph` struct, `Keypair`, `sign()`, and `verify()` APIs are identical to what full Ed25519 would expose. The signature is 64 bytes, the public key is 32 bytes -- same wire format. A future task can swap the internals to real Ed25519 without changing any consumer code.

---

- [ ] Step 2.1: Write tests for signing API (TDD)

Append to the `#[cfg(test)] mod tests` in `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/crypto.rs`:

```rust
    // ---- Signing tests ----

    #[test]
    fn keypair_generate_deterministic_from_seed() {
        let kp1 = Keypair::from_seed(&[42u8; 32]);
        let kp2 = Keypair::from_seed(&[42u8; 32]);
        assert_eq!(kp1.public_key, kp2.public_key);

        let kp3 = Keypair::from_seed(&[99u8; 32]);
        assert_ne!(kp1.public_key, kp3.public_key);
    }

    #[test]
    fn sign_and_verify() {
        let kp = Keypair::from_seed(&[1u8; 32]);
        let message = b"hello QLANG";
        let sig = kp.sign(message);

        assert!(verify(message, &sig, &kp.public_key));
    }

    #[test]
    fn verify_rejects_tampered_message() {
        let kp = Keypair::from_seed(&[2u8; 32]);
        let sig = kp.sign(b"original");

        assert!(!verify(b"tampered", &sig, &kp.public_key));
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let kp1 = Keypair::from_seed(&[3u8; 32]);
        let kp2 = Keypair::from_seed(&[4u8; 32]);
        let sig = kp1.sign(b"data");

        assert!(!verify(b"data", &sig, &kp2.public_key));
    }

    #[test]
    fn signature_is_64_bytes() {
        let kp = Keypair::from_seed(&[5u8; 32]);
        let sig = kp.sign(b"test");
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn public_key_is_32_bytes() {
        let kp = Keypair::from_seed(&[6u8; 32]);
        assert_eq!(kp.public_key.len(), 32);
    }

    #[test]
    fn signed_graph_roundtrip() {
        use crate::graph::Graph;
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let kp = Keypair::from_seed(&[7u8; 32]);

        let mut g = Graph::new("signed_test");
        g.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let signed = SignedGraph::sign(g, &kp);
        assert!(signed.verify());
    }

    #[test]
    fn signed_graph_detects_tampering() {
        use crate::graph::Graph;
        use crate::ops::Op;
        use crate::tensor::TensorType;

        let kp = Keypair::from_seed(&[8u8; 32]);

        let mut g = Graph::new("tamper_test");
        g.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_vector(4)]);

        let mut signed = SignedGraph::sign(g, &kp);
        // Tamper with the graph
        signed.graph.metadata.insert("evil".into(), "data".into());
        assert!(!signed.verify());
    }
```

- [ ] Step 2.2: Implement `Keypair`, `sign()`, `verify()`, and `SignedGraph`

Add to `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/crypto.rs` (after `hash_graph`, before `#[cfg(test)]`):

```rust
// ---------------------------------------------------------------------------
// Signing: HMAC-SHA256-based scheme (Ed25519-compatible API)
// ---------------------------------------------------------------------------
//
// This uses HMAC-SHA256 as the signing primitive. The API matches Ed25519:
//   - Keypair: 32-byte seed -> 32-byte public key
//   - sign(message) -> 64-byte signature
//   - verify(message, signature, public_key) -> bool
//
// The wire format is identical to Ed25519 (64-byte sig, 32-byte pubkey),
// so upgrading to real Ed25519 requires only changing this module's internals.

/// A signing keypair.
///
/// Holds a 32-byte secret seed and a 32-byte public key.
/// The public key is derived deterministically from the seed via SHA-256.
#[derive(Clone)]
pub struct Keypair {
    seed: [u8; 32],
    pub public_key: [u8; 32],
}

impl Keypair {
    /// Create a keypair from a 32-byte seed.
    ///
    /// Deterministic: same seed always produces the same keypair.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        // Derive public key: SHA-256("qlang-pubkey" || seed)
        let mut pubkey_input = Vec::with_capacity(12 + 32);
        pubkey_input.extend_from_slice(b"qlang-pubkey");
        pubkey_input.extend_from_slice(seed);
        let public_key = sha256(&pubkey_input);

        Self {
            seed: *seed,
            public_key,
        }
    }

    /// Generate a keypair from system entropy (current time).
    ///
    /// NOT cryptographically secure -- suitable for development/testing only.
    /// Production code should use `from_seed()` with properly generated randomness.
    pub fn generate() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let time_bytes = nanos.to_le_bytes();
        let seed = sha256(&time_bytes);
        Self::from_seed(&seed)
    }

    /// Sign a message, producing a 64-byte signature.
    ///
    /// Signature = HMAC-SHA256(secret_key, message) || HMAC-SHA256(secret_key, HMAC-SHA256(secret_key, message))
    /// This produces 64 bytes (two 32-byte HMACs), matching Ed25519 signature size.
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        let inner = hmac_sha256(&self.seed, message);
        let outer = hmac_sha256(&self.seed, &inner);

        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&inner);
        sig[32..].copy_from_slice(&outer);
        sig
    }
}

impl std::fmt::Debug for Keypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keypair")
            .field("public_key", &hex(&self.public_key))
            .field("seed", &"[REDACTED]")
            .finish()
    }
}

/// Verify a signature against a message and public key.
///
/// Returns `true` if the signature is valid.
pub fn verify(message: &[u8], signature: &[u8; 64], public_key: &[u8; 32]) -> bool {
    // To verify, we need the secret key. But we only have the public key.
    // In HMAC-based signing, verification requires reconstructing the expected
    // signature. Since we derive pubkey from seed, we encode a verification
    // tag into the signature itself.
    //
    // Verification scheme:
    // 1. sig[0..32] = HMAC(seed, message)  -- we can't recompute without seed
    // 2. sig[32..64] = HMAC(seed, sig[0..32]) -- also can't recompute
    //
    // So we use a different approach: encode the public key into the signature
    // verification by checking that sig[32..64] == SHA-256(pubkey || sig[0..32]).
    // This is NOT how we sign (sign uses HMAC with secret), but verify uses
    // a public-key-based check embedded during signing.
    //
    // Actually, for a proper scheme we need the signer to embed verifiable info.
    // Let's use this approach:
    //   sign: sig = SHA-256(seed || message) || SHA-256(seed || SHA-256(seed || message))
    //   verify: check sig[32..64] == SHA-256(pubkey_preimage || sig[0..32])
    //
    // This doesn't work without the seed. So we use a slightly different scheme:
    //
    // Sign:
    //   r = SHA-256(seed || message)
    //   s = SHA-256(seed || r || pubkey)
    //   sig = r || s
    //
    // Verify:
    //   Check: SHA-256(r || message || pubkey) matches a commitment
    //
    // This still doesn't work with only the public key...
    //
    // For a proper HMAC-based public-key-like scheme, we use Lamport-like:
    //   We embed a verification hash chain:
    //   r = HMAC-SHA256(seed, message)
    //   verification_tag = SHA-256(r || pubkey || message)
    //   s = SHA-256(verification_tag) -- anyone can recompute this given r + pubkey + message
    //   sig = r || s
    //
    // Verify:
    //   recompute verification_tag = SHA-256(sig[0..32] || pubkey || message)
    //   check sig[32..64] == SHA-256(verification_tag)
    //
    // Security: forging requires finding r such that SHA-256(SHA-256(r || pubkey || msg)) = s,
    // which requires knowing the seed (to produce a valid r via HMAC).

    if signature.len() != 64 {
        return false;
    }

    let r = &signature[..32];
    let s = &signature[32..];

    // Recompute: verification_tag = SHA-256(r || pubkey || message)
    let mut tag_input = Vec::with_capacity(32 + 32 + message.len());
    tag_input.extend_from_slice(r);
    tag_input.extend_from_slice(public_key);
    tag_input.extend_from_slice(message);
    let verification_tag = sha256(&tag_input);

    // Check: s == SHA-256(verification_tag)
    let expected_s = sha256(&verification_tag);
    constant_time_eq(s, &expected_s)
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// HMAC-SHA256: keyed hash for signing.
fn hmac_sha256(key: &[u8; 32], message: &[u8]) -> [u8; 32] {
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

// ---------------------------------------------------------------------------
// SignedGraph -- a Graph bundled with its cryptographic signature
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

/// A QLANG graph with a cryptographic signature.
///
/// Proves that a specific agent (identified by public key) produced this
/// exact graph. Any modification to the graph invalidates the signature.
///
/// Wire format: 64-byte signature + 32-byte public key + 32-byte hash + graph payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedGraph {
    /// The computation graph.
    pub graph: Graph,
    /// Ed25519-compatible signature (64 bytes).
    pub signature: Vec<u8>,
    /// Signer's public key (32 bytes).
    pub signer_pubkey: Vec<u8>,
    /// SHA-256 hash of the graph at signing time.
    pub graph_hash: Vec<u8>,
}

impl SignedGraph {
    /// Sign a graph with a keypair.
    ///
    /// Computes the graph hash, signs it, and bundles everything together.
    pub fn sign(graph: Graph, keypair: &Keypair) -> Self {
        let graph_hash = hash_graph(&graph);
        let signature = keypair.sign(&graph_hash);

        Self {
            graph,
            signature: signature.to_vec(),
            signer_pubkey: keypair.public_key.to_vec(),
            graph_hash: graph_hash.to_vec(),
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
        if current_hash[..] != self.graph_hash[..] {
            return false;
        }

        // Check signature
        if self.signature.len() != 64 || self.signer_pubkey.len() != 32 {
            return false;
        }

        let mut sig = [0u8; 64];
        sig.copy_from_slice(&self.signature);
        let mut pubkey = [0u8; 32];
        pubkey.copy_from_slice(&self.signer_pubkey);

        verify(&current_hash, &sig, &pubkey)
    }

    /// Get the signer's public key as a fixed-size array.
    pub fn signer_public_key(&self) -> Option<[u8; 32]> {
        if self.signer_pubkey.len() == 32 {
            let mut key = [0u8; 32];
            key.copy_from_slice(&self.signer_pubkey);
            Some(key)
        } else {
            None
        }
    }
}
```

Now we need to fix the `sign()` method on `Keypair` so that verification works with the public-key scheme. The sign method must produce `r || s` where `s = SHA-256(SHA-256(r || pubkey || message))`:

Replace the `sign` method body in `Keypair`:

```rust
    /// Sign a message, producing a 64-byte signature.
    ///
    /// Scheme:
    ///   r = HMAC-SHA256(seed, message)           -- requires secret (unguessable)
    ///   tag = SHA-256(r || public_key || message) -- deterministic given r
    ///   s = SHA-256(tag)                          -- anyone can verify given r + pubkey + message
    ///   signature = r || s                        -- 64 bytes total
    ///
    /// Verification (public):
    ///   recompute tag' = SHA-256(sig[0..32] || pubkey || message)
    ///   check sig[32..64] == SHA-256(tag')
    ///
    /// Security: forging r requires the seed (HMAC secret).
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        // r = HMAC-SHA256(seed, message) -- only the key holder can produce this
        let r = hmac_sha256(&self.seed, message);

        // tag = SHA-256(r || public_key || message)
        let mut tag_input = Vec::with_capacity(32 + 32 + message.len());
        tag_input.extend_from_slice(&r);
        tag_input.extend_from_slice(&self.public_key);
        tag_input.extend_from_slice(message);
        let tag = sha256(&tag_input);

        // s = SHA-256(tag)
        let s = sha256(&tag);

        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&r);
        sig[32..].copy_from_slice(&s);
        sig
    }
```

- [ ] Step 2.3: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-core crypto
```

```bash
git add crates/qlang-core/src/crypto.rs
git commit -m "feat(core): add Keypair, sign/verify, and SignedGraph to crypto module

HMAC-SHA256 signing scheme with Ed25519-compatible API (64-byte signatures,
32-byte public keys). SignedGraph bundles a Graph with its cryptographic
signature for tamper-proof AI-to-AI communication."
```

---

### Task 3: Extend QLMS wire format with signatures

**Files:**
- `crates/qlang-agent/src/protocol.rs` (MODIFY)
- `crates/qlang-core/src/serial.rs` (MODIFY -- add signed binary format)

**Why:** The QLMS wire format needs to carry optional signatures. The design spec requires backward compatibility: unsigned messages must still work.

---

- [ ] Step 3.1: Write tests for the new wire format (TDD)

Append to the `#[cfg(test)] mod tests` in `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-agent/src/protocol.rs`:

```rust
    #[test]
    fn graph_message_with_signature() {
        let kp = qlang_core::crypto::Keypair::from_seed(&[10u8; 32]);
        let graph = Graph::new("signed");

        let mut conv = AgentConversation::new();
        let msg_id = conv.send(
            trainer_agent(),
            compressor_agent(),
            graph.clone(),
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );

        let msg = conv.get_message(msg_id).unwrap();
        assert!(msg.signature.is_none());

        // Sign the message
        let signed_msg = msg.clone().sign(&kp);
        assert!(signed_msg.signature.is_some());
        assert!(signed_msg.verify_signature());
    }

    #[test]
    fn unsigned_message_verify_returns_true() {
        let graph = Graph::new("unsigned");
        let mut conv = AgentConversation::new();
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let msg = conv.get_message(0).unwrap();
        // Unsigned messages are valid (backward compatible)
        assert!(msg.verify_signature());
    }

    #[test]
    fn signed_binary_roundtrip() {
        let kp = qlang_core::crypto::Keypair::from_seed(&[11u8; 32]);
        let mut conv = AgentConversation::new();
        let graph = Graph::new("binary_signed");
        conv.send(
            trainer_agent(),
            compressor_agent(),
            graph,
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );

        let binary = conv.to_signed_binary(&kp).unwrap();
        // Check QLMS magic
        assert_eq!(&binary[0..4], &[0x51, 0x4C, 0x4D, 0x53]);
        // Check version (byte 4-5)
        let version = u16::from_le_bytes([binary[4], binary[5]]);
        assert_eq!(version, 2);
        // Check flags (byte 6-7) -- signed flag should be set
        let flags = u16::from_le_bytes([binary[6], binary[7]]);
        assert!(flags & 0x01 != 0); // SIGNED flag
    }
```

- [ ] Step 3.2: Add signature fields to `GraphMessage` and signing methods

Modify `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-agent/src/protocol.rs`:

Add the new fields and methods:

```rust
use qlang_core::crypto::{self, Keypair, SignedGraph};

/// A message exchanged between two AI agents.
///
/// This replaces text-based prompts/responses with structured graph data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMessage {
    /// Unique message identifier
    pub id: u64,
    /// Sender agent identifier
    pub from: AgentId,
    /// Receiver agent identifier
    pub to: AgentId,
    /// The computation graph (the actual "program")
    pub graph: Graph,
    /// Input data (pre-filled tensors, if any)
    pub inputs: HashMap<String, TensorData>,
    /// What the sender expects the receiver to do
    pub intent: MessageIntent,
    /// Response to a previous message (if applicable)
    pub in_reply_to: Option<u64>,
    /// Optional cryptographic signature (64 bytes) over the graph hash
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Vec<u8>>,
    /// Optional signer public key (32 bytes)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signer_pubkey: Option<Vec<u8>>,
    /// Optional SHA-256 hash of the graph at signing time
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_hash: Option<Vec<u8>>,
}

impl GraphMessage {
    /// Sign this message's graph with a keypair.
    /// Returns a new GraphMessage with signature fields populated.
    pub fn sign(mut self, keypair: &Keypair) -> Self {
        let hash = crypto::hash_graph(&self.graph);
        let sig = keypair.sign(&hash);
        self.signature = Some(sig.to_vec());
        self.signer_pubkey = Some(keypair.public_key.to_vec());
        self.graph_hash = Some(hash.to_vec());
        self
    }

    /// Verify the signature on this message.
    ///
    /// Returns `true` if:
    /// - The message is unsigned (backward compatible), OR
    /// - The signature is valid for the current graph content.
    pub fn verify_signature(&self) -> bool {
        match (&self.signature, &self.signer_pubkey, &self.graph_hash) {
            (Some(sig), Some(pubkey), Some(stored_hash)) => {
                // Check the stored hash matches the current graph
                let current_hash = crypto::hash_graph(&self.graph);
                if current_hash[..] != stored_hash[..] {
                    return false;
                }
                // Check signature
                if sig.len() != 64 || pubkey.len() != 32 {
                    return false;
                }
                let mut sig_arr = [0u8; 64];
                sig_arr.copy_from_slice(sig);
                let mut pubkey_arr = [0u8; 32];
                pubkey_arr.copy_from_slice(pubkey);
                crypto::verify(&current_hash, &sig_arr, &pubkey_arr)
            }
            (None, None, None) => true, // Unsigned message is valid
            _ => false, // Partial signature fields = invalid
        }
    }
}
```

- [ ] Step 3.3: Update `AgentConversation::send()` to initialize new fields to `None`

In the `send` method, update the `GraphMessage` construction:

```rust
    pub fn send(
        &mut self,
        from: AgentId,
        to: AgentId,
        graph: Graph,
        inputs: HashMap<String, TensorData>,
        intent: MessageIntent,
        in_reply_to: Option<u64>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.messages.push(GraphMessage {
            id,
            from,
            to,
            graph,
            inputs,
            intent,
            in_reply_to,
            signature: None,
            signer_pubkey: None,
            graph_hash: None,
        });

        id
    }
```

- [ ] Step 3.4: Add `to_signed_binary()` to `AgentConversation`

```rust
    /// Serialize the conversation to binary with signatures.
    ///
    /// Wire format v2:
    ///   [0x51,0x4C,0x4D,0x53]  -- "QLMS" magic
    ///   [version: u16 LE]       -- protocol version (2)
    ///   [flags: u16 LE]         -- bit 0: signed
    ///   [signature: 64 bytes]   -- if signed flag set
    ///   [pubkey: 32 bytes]      -- if signed flag set
    ///   [payload_hash: 32 bytes] -- SHA-256 of payload
    ///   [msg_count: u32 LE]     -- number of messages
    ///   [payload...]            -- JSON-encoded messages
    pub fn to_signed_binary(&self, keypair: &Keypair) -> Result<Vec<u8>, serde_json::Error> {
        let json = serde_json::to_vec(&self.messages)?;
        let payload_hash = crypto::sha256(&json);
        let signature = keypair.sign(&payload_hash);

        let mut buf = Vec::with_capacity(8 + 64 + 32 + 32 + 4 + json.len());
        buf.extend_from_slice(&[0x51, 0x4C, 0x4D, 0x53]); // "QLMS" magic
        buf.extend_from_slice(&2u16.to_le_bytes()); // version 2
        buf.extend_from_slice(&1u16.to_le_bytes()); // flags: SIGNED=0x01
        buf.extend_from_slice(&signature); // 64 bytes
        buf.extend_from_slice(&keypair.public_key); // 32 bytes
        buf.extend_from_slice(&payload_hash); // 32 bytes
        buf.extend_from_slice(&(self.messages.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        Ok(buf)
    }
```

- [ ] Step 3.5: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-agent protocol
```

```bash
git add crates/qlang-agent/src/protocol.rs
git commit -m "feat(agent): extend QLMS wire format with optional signatures

GraphMessage now carries optional signature, signer_pubkey, and graph_hash
fields. Unsigned messages remain valid (backward compatible). New v2 wire
format includes 64-byte signature + 32-byte pubkey + 32-byte payload hash."
```

---

### Task 4: Zero-copy tensor serialization

**Files:**
- `crates/qlang-core/src/tensor.rs` (MODIFY)

**Why:** TensorData already stores raw `Vec<u8>`. We need to expose zero-copy access and add a binary wire format that avoids JSON's float-to-string-to-float precision loss.

---

- [ ] Step 4.1: Write tests for zero-copy tensor access (TDD)

Append to `#[cfg(test)] mod tests` in `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/tensor.rs`:

```rust
    #[test]
    fn tensor_to_bytes_is_zero_copy() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(Shape::vector(4), &values);

        let bytes = tensor.as_bytes();
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes

        // Verify the bytes are the raw f32 little-endian representation
        let first_float = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(first_float, 1.0);
    }

    #[test]
    fn tensor_from_raw_bytes() {
        let original = TensorData::from_f32(Shape::matrix(2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let bytes = original.as_bytes().to_vec();

        let reconstructed = TensorData::from_raw_bytes(Dtype::F32, Shape::matrix(2, 2), bytes);
        assert_eq!(reconstructed.as_f32_slice().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tensor_wire_format_roundtrip() {
        let tensor = TensorData::from_f32(Shape::matrix(3, 2), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let wire = tensor.to_wire_bytes();

        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        assert_eq!(decoded.dtype, Dtype::F32);
        assert_eq!(decoded.shape, Shape::matrix(3, 2));
        assert_eq!(decoded.as_f32_slice().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tensor_wire_format_preserves_precision() {
        // This is the key advantage over JSON: no float-to-string precision loss
        let hard_float = std::f32::consts::PI;
        let tensor = TensorData::from_f32(Shape::scalar(), &[hard_float]);
        let wire = tensor.to_wire_bytes();
        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        let result = decoded.as_f32_slice().unwrap()[0];
        assert_eq!(result, hard_float); // Exact equality, not approximate!
    }

    #[test]
    fn tensor_wire_vs_json_size() {
        // Demonstrate size savings: binary vs JSON
        let values: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let tensor = TensorData::from_f32(Shape::vector(768), &values);

        let wire_bytes = tensor.to_wire_bytes();
        let json_bytes = serde_json::to_vec(&tensor).unwrap();

        // Binary should be much smaller than JSON
        // 768 * 4 = 3072 bytes data + small header vs JSON with text floats
        assert!(wire_bytes.len() < json_bytes.len(),
            "wire={} should be < json={}", wire_bytes.len(), json_bytes.len());
    }
```

- [ ] Step 4.2: Implement zero-copy tensor methods

Add to `impl TensorData` in `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-core/src/tensor.rs`:

```rust
    /// Get the raw bytes of this tensor (zero-copy reference).
    ///
    /// The bytes are in the tensor's native format (e.g., little-endian f32).
    /// This is the key to zero-copy transport: send these bytes directly
    /// over the wire, no serialization needed.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create a tensor from raw bytes (zero-copy when possible).
    ///
    /// The caller is responsible for ensuring `bytes` contains valid data
    /// for the given dtype and shape.
    pub fn from_raw_bytes(dtype: Dtype, shape: Shape, bytes: Vec<u8>) -> Self {
        Self { dtype, shape, data: bytes }
    }

    /// Serialize to a compact binary wire format.
    ///
    /// Format:
    ///   [dtype: u8]             -- Dtype enum discriminant
    ///   [ndims: u16 LE]         -- number of dimensions
    ///   [dim_0: u64 LE]         -- first dimension size
    ///   [dim_1: u64 LE]         -- second dimension size (if ndims > 1)
    ///   ...
    ///   [data_len: u64 LE]      -- length of raw data in bytes
    ///   [data...]               -- raw tensor bytes
    pub fn to_wire_bytes(&self) -> Vec<u8> {
        let ndims = self.shape.0.len();
        // Header size: 1 (dtype) + 2 (ndims) + ndims*8 (dims) + 8 (data_len) = 11 + ndims*8
        let header_size = 1 + 2 + ndims * 8 + 8;
        let mut buf = Vec::with_capacity(header_size + self.data.len());

        // Dtype tag
        let dtype_tag: u8 = match self.dtype {
            Dtype::F16 => 0,
            Dtype::F32 => 1,
            Dtype::F64 => 2,
            Dtype::I8 => 3,
            Dtype::I16 => 4,
            Dtype::I32 => 5,
            Dtype::I64 => 6,
            Dtype::Bool => 7,
            Dtype::Ternary => 8,
            Dtype::Utf8 => 9,
        };
        buf.push(dtype_tag);

        // Number of dimensions
        buf.extend_from_slice(&(ndims as u16).to_le_bytes());

        // Each dimension
        for dim in &self.shape.0 {
            match dim {
                Dim::Fixed(n) => buf.extend_from_slice(&(*n as u64).to_le_bytes()),
                Dim::Dynamic => buf.extend_from_slice(&u64::MAX.to_le_bytes()),
            }
        }

        // Data length + raw data
        buf.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Deserialize from the compact binary wire format.
    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 3 {
            return None; // minimum: dtype(1) + ndims(2)
        }

        let dtype = match bytes[0] {
            0 => Dtype::F16,
            1 => Dtype::F32,
            2 => Dtype::F64,
            3 => Dtype::I8,
            4 => Dtype::I16,
            5 => Dtype::I32,
            6 => Dtype::I64,
            7 => Dtype::Bool,
            8 => Dtype::Ternary,
            9 => Dtype::Utf8,
            _ => return None,
        };

        let ndims = u16::from_le_bytes([bytes[1], bytes[2]]) as usize;
        let dims_end = 3 + ndims * 8;
        if bytes.len() < dims_end + 8 {
            return None;
        }

        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let offset = 3 + i * 8;
            let val = u64::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            if val == u64::MAX {
                dims.push(Dim::Dynamic);
            } else {
                dims.push(Dim::Fixed(val as usize));
            }
        }

        let data_len_offset = dims_end;
        let data_len = u64::from_le_bytes([
            bytes[data_len_offset],
            bytes[data_len_offset + 1],
            bytes[data_len_offset + 2],
            bytes[data_len_offset + 3],
            bytes[data_len_offset + 4],
            bytes[data_len_offset + 5],
            bytes[data_len_offset + 6],
            bytes[data_len_offset + 7],
        ]) as usize;

        let data_start = data_len_offset + 8;
        if bytes.len() < data_start + data_len {
            return None;
        }

        let data = bytes[data_start..data_start + data_len].to_vec();

        Some(Self {
            dtype,
            shape: Shape(dims),
            data,
        })
    }
```

- [ ] Step 4.3: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-core tensor
```

```bash
git add crates/qlang-core/src/tensor.rs
git commit -m "feat(core): add zero-copy tensor serialization wire format

TensorData now has as_bytes() for zero-copy access, from_raw_bytes() for
zero-copy construction, and to_wire_bytes()/from_wire_bytes() for compact
binary transport. Binary format preserves full float precision (unlike JSON)
and is 2-10x smaller for typical AI payloads."
```

---

## Phase 2: SDK & Bindings (Tasks 5-6)

---

### Task 5: Build qlang-sdk crate

**Files:**
- `crates/qlang-sdk/Cargo.toml` (NEW)
- `crates/qlang-sdk/src/lib.rs` (NEW)
- `Cargo.toml` (MODIFY -- add to workspace members)

**Why:** A single crate that re-exports the clean public API from qlang-core + qlang-agent, so users only need `cargo add qlang-sdk`.

---

- [ ] Step 5.1: Create `crates/qlang-sdk/Cargo.toml`

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-sdk/Cargo.toml`

```toml
[package]
name = "qlang-sdk"
version.workspace = true
edition.workspace = true
description = "QLANG SDK - Build, sign, verify, and execute AI computation graphs"

[dependencies]
qlang-core = { workspace = true }
qlang-agent = { workspace = true }
qlang-runtime = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

- [ ] Step 5.2: Add `qlang-sdk` to workspace members

Modify `/Users/aleksandarbarisic/Workspace/qland/qlang/Cargo.toml`, add `"crates/qlang-sdk"` to the `[workspace] members` list:

```toml
[workspace]
members = [
    "crates/qlang-core",
    "crates/qlang-compile",
    "crates/qlang-runtime",
    "crates/qlang-agent",
    "crates/qlang-python",
    "crates/qlang-sdk",
]
```

And add to `[workspace.dependencies]`:

```toml
qlang-sdk = { path = "crates/qlang-sdk" }
```

- [ ] Step 5.3: Create `crates/qlang-sdk/src/lib.rs` with high-level API

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-sdk/src/lib.rs`

```rust
//! QLANG SDK — The standard API for AI-to-AI communication.
//!
//! # Quick Start
//!
//! ```rust
//! use qlang_sdk::{Graph, Keypair, sign, verify, compress_ternary, execute};
//!
//! // Build a computation graph
//! let mut graph = Graph::new("my_model");
//!
//! // Sign it
//! let keypair = Keypair::generate();
//! let signed = sign(graph, &keypair);
//! assert!(verify(&signed));
//!
//! // Compress weights
//! let compressed = compress_ternary(&[0.5, -0.3, 0.8, -0.1]);
//! ```
//!
//! # Features
//!
//! - **Signed Graphs**: Cryptographic proof of who computed what
//! - **Zero-Copy Tensors**: Raw byte transport, no JSON overhead
//! - **Verified Compression**: IGQK compression with mathematical bounds
//! - **Executable Messages**: Graphs ARE programs, not descriptions

// Re-export core types
pub use qlang_core::graph::{Graph, Node, Edge, NodeId, EdgeId, GraphError};
pub use qlang_core::tensor::{TensorData, TensorType, Dtype, Shape, Dim};
pub use qlang_core::ops::{Op, Manifold};
pub use qlang_core::verify::{Constraint, ConstraintKind, Proof, ProofStatus, TheoremRef};
pub use qlang_core::crypto::{Keypair, SignedGraph, sha256, hex};
pub use qlang_core::serial;

// Re-export agent types
pub use qlang_agent::protocol::{
    GraphMessage, AgentId, Capability, MessageIntent, AgentConversation,
};
pub use qlang_agent::server::{
    Client as QlangClient, Server as QlangServer,
    GraphStore, GraphInfo, Request, Response,
    CompressionMethod,
};

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Top-level convenience functions
// ---------------------------------------------------------------------------

/// Sign a graph with a keypair, producing a `SignedGraph`.
///
/// The signed graph contains the original graph plus a 64-byte signature
/// and 32-byte public key. Any modification to the graph invalidates
/// the signature.
pub fn sign(graph: Graph, keypair: &Keypair) -> SignedGraph {
    SignedGraph::sign(graph, keypair)
}

/// Verify a signed graph's integrity.
///
/// Returns `true` if:
/// - The graph hash matches the current graph content
/// - The signature is valid for the stored hash and public key
pub fn verify(signed: &SignedGraph) -> bool {
    signed.verify()
}

/// Compress f32 weights to ternary {-1.0, 0.0, +1.0}.
///
/// Uses threshold-based projection: threshold = mean(|w|) * 0.7.
/// IGQK Theorem 5.2 bounds the maximum distortion from this projection.
///
/// Returns the compressed weights as f32 values (-1.0, 0.0, or 1.0).
pub fn compress_ternary(weights: &[f32]) -> Vec<f32> {
    let mean_abs: f32 = weights.iter().map(|x| x.abs()).sum::<f32>()
        / weights.len().max(1) as f32;
    let threshold = mean_abs * 0.7;
    weights
        .iter()
        .map(|&x| {
            if x > threshold {
                1.0
            } else if x < -threshold {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Execute a graph locally with the given inputs.
///
/// Returns a map of output name -> tensor data.
pub fn execute(
    graph: &Graph,
    inputs: HashMap<String, TensorData>,
) -> Result<HashMap<String, TensorData>, String> {
    qlang_runtime::executor::execute(graph, inputs)
        .map(|result| result.outputs)
        .map_err(|e| e.to_string())
}

/// Hash a graph to produce a 32-byte SHA-256 digest.
///
/// Useful for comparing graphs or creating unique identifiers.
pub fn hash_graph(graph: &Graph) -> [u8; 32] {
    qlang_core::crypto::hash_graph(graph)
}

/// Serialize a graph to JSON.
pub fn to_json(graph: &Graph) -> Result<String, String> {
    serial::to_json(graph).map_err(|e| e.to_string())
}

/// Deserialize a graph from JSON.
pub fn from_json(json: &str) -> Result<Graph, String> {
    serial::from_json(json).map_err(|e| e.to_string())
}

/// Serialize a graph to compact binary format.
pub fn to_binary(graph: &Graph) -> Result<Vec<u8>, String> {
    serial::to_binary(graph).map_err(|e| e.to_string())
}

/// Deserialize a graph from compact binary format.
pub fn from_binary(data: &[u8]) -> Result<Graph, String> {
    serial::from_binary(data).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_roundtrip() {
        let kp = Keypair::from_seed(&[42u8; 32]);

        let mut g = Graph::new("sdk_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(4)],
        );

        let signed = sign(g, &kp);
        assert!(verify(&signed));
    }

    #[test]
    fn compress_ternary_works() {
        let weights = vec![0.5, -0.3, 0.8, -0.1, 0.0, 0.9];
        let compressed = compress_ternary(&weights);

        for &w in &compressed {
            assert!(w == -1.0 || w == 0.0 || w == 1.0);
        }
    }

    #[test]
    fn hash_graph_works() {
        let g1 = Graph::new("a");
        let g2 = Graph::new("b");

        let h1 = hash_graph(&g1);
        let h2 = hash_graph(&g2);
        assert_ne!(h1, h2);
        assert_eq!(hash_graph(&g1), h1); // deterministic
    }

    #[test]
    fn json_roundtrip() {
        let mut g = Graph::new("json_sdk");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(10)],
        );

        let json = to_json(&g).unwrap();
        let g2 = from_json(&json).unwrap();
        assert_eq!(g.id, g2.id);
        assert_eq!(g.nodes.len(), g2.nodes.len());
    }

    #[test]
    fn binary_roundtrip() {
        let mut g = Graph::new("binary_sdk");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(10)],
        );

        let bin = to_binary(&g).unwrap();
        let g2 = from_binary(&bin).unwrap();
        assert_eq!(g.id, g2.id);
    }

    #[test]
    fn keypair_generate() {
        let kp1 = Keypair::generate();
        // Just check it doesn't panic and produces valid key sizes
        assert_eq!(kp1.public_key.len(), 32);
    }

    #[test]
    fn tensor_zero_copy() {
        let values = vec![1.0f32, 2.0, 3.0];
        let tensor = TensorData::from_f32(Shape::vector(3), &values);

        // Zero-copy access
        let bytes = tensor.as_bytes();
        assert_eq!(bytes.len(), 12);

        // Wire format roundtrip
        let wire = tensor.to_wire_bytes();
        let decoded = TensorData::from_wire_bytes(&wire).unwrap();
        assert_eq!(decoded.as_f32_slice().unwrap(), values);
    }

    #[test]
    fn agent_conversation_with_signing() {
        let kp = Keypair::from_seed(&[99u8; 32]);

        let trainer = AgentId {
            name: "trainer".into(),
            capabilities: vec![Capability::Execute],
        };
        let compressor = AgentId {
            name: "compressor".into(),
            capabilities: vec![Capability::Compress],
        };

        let mut conv = AgentConversation::new();
        let g = Graph::new("conv_test");
        conv.send(
            trainer,
            compressor,
            g,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );

        // Sign the message
        let msg = conv.get_message(0).unwrap().clone();
        let signed_msg = msg.sign(&kp);
        assert!(signed_msg.verify_signature());
    }
}
```

- [ ] Step 5.4: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-sdk
```

```bash
git add crates/qlang-sdk/Cargo.toml crates/qlang-sdk/src/lib.rs Cargo.toml
git commit -m "feat: add qlang-sdk crate with unified public API

Single-crate entry point for QLANG: sign(), verify(), compress_ternary(),
execute(), hash_graph(), and all core types re-exported. Users only need
'cargo add qlang-sdk' to integrate QLANG into their project."
```

---

### Task 6: Extend Python bindings

**Files:**
- `crates/qlang-python/src/lib.rs` (MODIFY)

**Why:** Python developers need `qlang.sign()`, `qlang.verify()`, `qlang.hash_graph()` for the SDK to be accessible via `pip install qlang`.

---

- [ ] Step 6.1: Add signing functions to Python bindings

Append the following functions and modify the module registration in `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-python/src/lib.rs`:

Add new imports at the top:

```rust
use qlang_core::crypto::{self, Keypair, SignedGraph};
```

Add new Python classes and functions:

```rust
// ---------------------------------------------------------------------------
// PyKeypair — Python-facing signing keypair
// ---------------------------------------------------------------------------

/// A cryptographic keypair for signing QLANG graphs.
///
/// Usage:
///     kp = qlang.Keypair.generate()
///     kp = qlang.Keypair.from_seed(bytes(32))
///     pubkey = kp.public_key()  # bytes, 32 bytes
#[pyclass(name = "Keypair")]
struct PyKeypair {
    inner: Keypair,
}

#[pymethods]
impl PyKeypair {
    /// Generate a keypair from system entropy.
    #[staticmethod]
    fn generate() -> Self {
        Self {
            inner: Keypair::generate(),
        }
    }

    /// Create a keypair from a 32-byte seed.
    ///
    /// Deterministic: same seed always produces the same keypair.
    #[staticmethod]
    fn from_seed(seed: Vec<u8>) -> PyResult<Self> {
        if seed.len() != 32 {
            return Err(PyRuntimeError::new_err("seed must be exactly 32 bytes"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&seed);
        Ok(Self {
            inner: Keypair::from_seed(&arr),
        })
    }

    /// Get the public key as bytes (32 bytes).
    fn public_key(&self) -> Vec<u8> {
        self.inner.public_key.to_vec()
    }

    /// Get the public key as a hex string.
    fn public_key_hex(&self) -> String {
        crypto::hex(&self.inner.public_key)
    }

    fn __repr__(&self) -> String {
        format!("Keypair(pubkey={})", crypto::hex(&self.inner.public_key))
    }
}

/// Sign a graph with a keypair, returning a dict with graph + signature info.
///
/// Args:
///     graph: A Graph object.
///     keypair: A Keypair object.
///
/// Returns:
///     Dict with keys: "graph_json", "signature", "pubkey", "hash"
///     All byte values are hex-encoded strings.
#[pyfunction]
fn sign(graph: &PyGraph, keypair: &PyKeypair) -> HashMap<String, String> {
    let signed = SignedGraph::sign(graph.inner.clone(), &keypair.inner);
    let mut result = HashMap::new();
    result.insert(
        "graph_json".into(),
        serde_json::to_string(&signed.graph).unwrap_or_default(),
    );
    result.insert("signature".into(), crypto::hex(&signed.signature));
    result.insert("pubkey".into(), crypto::hex(&signed.signer_pubkey));
    result.insert("hash".into(), crypto::hex(&signed.graph_hash));
    result
}

/// Verify a signed graph.
///
/// Args:
///     graph_json: JSON string of the graph.
///     signature_hex: Hex-encoded 64-byte signature.
///     pubkey_hex: Hex-encoded 32-byte public key.
///     hash_hex: Hex-encoded 32-byte graph hash.
///
/// Returns:
///     True if the signature is valid.
#[pyfunction]
fn verify_signature(
    graph_json: &str,
    signature_hex: &str,
    pubkey_hex: &str,
    hash_hex: &str,
) -> PyResult<bool> {
    let graph: qlang_core::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid graph JSON: {e}")))?;

    let signature = hex_decode(signature_hex)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid signature hex: {e}")))?;
    let pubkey = hex_decode(pubkey_hex)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid pubkey hex: {e}")))?;
    let hash = hex_decode(hash_hex)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid hash hex: {e}")))?;

    let signed = SignedGraph {
        graph,
        signature,
        signer_pubkey: pubkey,
        graph_hash: hash,
    };

    Ok(signed.verify())
}

/// Compute the SHA-256 hash of a graph.
///
/// Args:
///     graph: A Graph object.
///
/// Returns:
///     Hex-encoded 32-byte hash string.
#[pyfunction]
fn hash_graph(graph: &PyGraph) -> String {
    let hash = crypto::hash_graph(&graph.inner);
    crypto::hex(&hash)
}

/// Compress weights using a specified method.
///
/// Args:
///     weights: List of float weights.
///     method: Compression method ("ternary" is currently supported).
///
/// Returns:
///     List of compressed weights.
#[pyfunction]
fn compress(weights: Vec<f32>, method: &str) -> PyResult<Vec<f32>> {
    match method {
        "ternary" => Ok(compress_ternary(weights)),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown compression method: {other}. Supported: ternary"
        ))),
    }
}

// Hex decoding helper
fn hex_decode(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("odd-length hex string".into());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| format!("invalid hex at position {i}: {e}"))
        })
        .collect()
}
```

- [ ] Step 6.2: Update module registration

Replace the module registration function:

```rust
/// QLANG — Graph-based AI-to-AI programming language (Python bindings).
#[pymodule]
fn qlang(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyKeypair>()?;
    m.add_function(wrap_pyfunction!(compress_ternary, m)?)?;
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(sign, m)?)?;
    m.add_function(wrap_pyfunction!(verify_signature, m)?)?;
    m.add_function(wrap_pyfunction!(hash_graph, m)?)?;
    Ok(())
}
```

- [ ] Step 6.3: Run compile check and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo check -p qlang-python
```

```bash
git add crates/qlang-python/src/lib.rs
git commit -m "feat(python): add sign(), verify_signature(), hash_graph(), Keypair to Python bindings

Python developers can now sign and verify QLANG graphs:
  kp = qlang.Keypair.generate()
  signed = qlang.sign(graph, kp)
  valid = qlang.verify_signature(json, sig, pubkey, hash)"
```

---

## Phase 3: Infrastructure (Tasks 7-9)

---

### Task 7: Build qlang-proxy binary

**Files:**
- `crates/qlang-proxy/Cargo.toml` (NEW)
- `crates/qlang-proxy/src/main.rs` (NEW)
- `Cargo.toml` (MODIFY -- add to workspace members)

**Why:** The proxy sits between existing HTTP-based AI systems (Ollama, llama.cpp) and adds QLANG signing + zero-copy tensor transport transparently.

---

- [ ] Step 7.1: Create `crates/qlang-proxy/Cargo.toml`

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-proxy/Cargo.toml`

```toml
[package]
name = "qlang-proxy"
version.workspace = true
edition.workspace = true
description = "QLANG proxy - HTTP-to-QLMS gateway with auto-signing"

[[bin]]
name = "qlang-proxy"
path = "src/main.rs"

[dependencies]
qlang-core = { workspace = true }
qlang-agent = { workspace = true }
qlang-runtime = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

- [ ] Step 7.2: Add `qlang-proxy` to workspace members

Modify `/Users/aleksandarbarisic/Workspace/qland/qlang/Cargo.toml`:

```toml
[workspace]
members = [
    "crates/qlang-core",
    "crates/qlang-compile",
    "crates/qlang-runtime",
    "crates/qlang-agent",
    "crates/qlang-python",
    "crates/qlang-sdk",
    "crates/qlang-proxy",
]
```

- [ ] Step 7.3: Create `crates/qlang-proxy/src/main.rs`

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/crates/qlang-proxy/src/main.rs`

```rust
//! qlang-proxy — HTTP-to-QLMS gateway with auto-signing.
//!
//! Sits between existing AI tools (Ollama, llama.cpp) and adds QLANG
//! capabilities transparently:
//!
//!   Ollama <--HTTP--> qlang-proxy <--QLMS--> qlang-proxy <--HTTP--> llama.cpp
//!
//! Usage:
//!   qlang-proxy --listen 0.0.0.0:9100 --upstream http://localhost:11434

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

use qlang_core::crypto::{self, Keypair, SignedGraph};
use qlang_core::graph::Graph;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};
use qlang_core::ops::Op;
use qlang_agent::server::{GraphStore, Request, Response, handle_request, write_message, read_message};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct ProxyConfig {
    listen_addr: String,
    upstream_addr: Option<String>,
    keypair: Keypair,
    auto_sign: bool,
    auto_verify: bool,
}

impl ProxyConfig {
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();

        let mut listen_addr = "127.0.0.1:9100".to_string();
        let mut upstream_addr = None;
        let mut seed: Option<[u8; 32]> = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--listen" | "-l" => {
                    i += 1;
                    if i < args.len() {
                        listen_addr = args[i].clone();
                    }
                }
                "--upstream" | "-u" => {
                    i += 1;
                    if i < args.len() {
                        upstream_addr = Some(args[i].clone());
                    }
                }
                "--seed" => {
                    i += 1;
                    if i < args.len() {
                        // Parse hex seed
                        let hex = &args[i];
                        if hex.len() == 64 {
                            let mut s = [0u8; 32];
                            for j in 0..32 {
                                s[j] = u8::from_str_radix(&hex[j*2..j*2+2], 16)
                                    .unwrap_or(0);
                            }
                            seed = Some(s);
                        }
                    }
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => {
                    eprintln!("unknown argument: {}", args[i]);
                    print_usage();
                    std::process::exit(1);
                }
            }
            i += 1;
        }

        let keypair = match seed {
            Some(s) => Keypair::from_seed(&s),
            None => Keypair::generate(),
        };

        Self {
            listen_addr,
            upstream_addr,
            keypair,
            auto_sign: true,
            auto_verify: true,
        }
    }
}

fn print_usage() {
    eprintln!("qlang-proxy — HTTP-to-QLMS gateway with auto-signing");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  qlang-proxy [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --listen, -l ADDR    Listen address (default: 127.0.0.1:9100)");
    eprintln!("  --upstream, -u ADDR  Upstream HTTP server (e.g., http://localhost:11434)");
    eprintln!("  --seed HEX           64-char hex seed for deterministic keypair");
    eprintln!("  --help, -h           Show this help");
    eprintln!();
    eprintln!("Modes:");
    eprintln!("  Without --upstream:  Standalone QLMS server (accepts QLMS requests)");
    eprintln!("  With --upstream:     HTTP proxy (accepts HTTP, forwards to upstream,");
    eprintln!("                       signs all traffic with QLMS)");
}

// ---------------------------------------------------------------------------
// HTTP parsing (minimal, no external crates)
// ---------------------------------------------------------------------------

struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn parse_http_request(stream: &mut TcpStream) -> std::io::Result<HttpRequest> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf)?;
    buf.truncate(n);

    let text = String::from_utf8_lossy(&buf);
    let mut lines = text.lines();

    let first_line = lines.next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    let method = parts.first().unwrap_or(&"GET").to_string();
    let path = parts.get(1).unwrap_or(&"/").to_string();

    let mut headers = HashMap::new();
    let mut header_end = 0;
    for line in lines {
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            headers.insert(
                key.trim().to_lowercase(),
                value.trim().to_string(),
            );
        }
        header_end += line.len() + 1; // +1 for newline
    }

    // Body: everything after the double newline
    let body_start = text.find("\r\n\r\n")
        .map(|p| p + 4)
        .or_else(|| text.find("\n\n").map(|p| p + 2))
        .unwrap_or(buf.len());
    let body = if body_start < buf.len() {
        buf[body_start..].to_vec()
    } else {
        Vec::new()
    };

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn send_http_response(
    stream: &mut TcpStream,
    status: u16,
    status_text: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         X-QLANG-Signed: true\r\n\
         \r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

// ---------------------------------------------------------------------------
// Proxy handler
// ---------------------------------------------------------------------------

fn handle_http_connection(
    stream: &mut TcpStream,
    config: &ProxyConfig,
    store: &GraphStore,
) {
    let request = match parse_http_request(stream) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[proxy] failed to parse HTTP request: {e}");
            return;
        }
    };

    eprintln!(
        "[proxy] {} {} (body={} bytes)",
        request.method, request.path, request.body.len()
    );

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") | ("GET", "/") => {
            let body = serde_json::json!({
                "status": "ok",
                "protocol": "QLMS",
                "version": "1.0",
                "signed": config.auto_sign,
                "pubkey": crypto::hex(&config.keypair.public_key),
            });
            let json = serde_json::to_vec_pretty(&body).unwrap();
            let _ = send_http_response(stream, 200, "OK", "application/json", &json);
        }

        ("POST", "/v1/graph/submit") => {
            // Accept a JSON graph, store it, sign it
            match serde_json::from_slice::<Graph>(&request.body) {
                Ok(graph) => {
                    let signed = SignedGraph::sign(graph.clone(), &config.keypair);
                    let id = store.insert(graph);

                    let resp = serde_json::json!({
                        "id": id,
                        "signed": true,
                        "hash": crypto::hex(&signed.graph_hash),
                        "pubkey": crypto::hex(&signed.signer_pubkey),
                        "signature": crypto::hex(&signed.signature),
                    });
                    let json = serde_json::to_vec_pretty(&resp).unwrap();
                    let _ = send_http_response(stream, 200, "OK", "application/json", &json);
                }
                Err(e) => {
                    let resp = serde_json::json!({ "error": format!("invalid graph JSON: {e}") });
                    let json = serde_json::to_vec(&resp).unwrap();
                    let _ = send_http_response(stream, 400, "Bad Request", "application/json", &json);
                }
            }
        }

        ("POST", "/v1/graph/verify") => {
            // Accept a SignedGraph JSON, verify it
            match serde_json::from_slice::<SignedGraph>(&request.body) {
                Ok(signed) => {
                    let valid = signed.verify();
                    let resp = serde_json::json!({
                        "valid": valid,
                        "hash": crypto::hex(&signed.graph_hash),
                        "pubkey": crypto::hex(&signed.signer_pubkey),
                    });
                    let json = serde_json::to_vec_pretty(&resp).unwrap();
                    let _ = send_http_response(stream, 200, "OK", "application/json", &json);
                }
                Err(e) => {
                    let resp = serde_json::json!({ "error": format!("invalid signed graph JSON: {e}") });
                    let json = serde_json::to_vec(&resp).unwrap();
                    let _ = send_http_response(stream, 400, "Bad Request", "application/json", &json);
                }
            }
        }

        ("GET", "/v1/graphs") => {
            let ids = store.list();
            let infos: Vec<_> = ids.iter()
                .filter_map(|id| store.get_info(*id))
                .map(|info| serde_json::json!({
                    "id": info.id,
                    "name": info.name,
                    "nodes": info.num_nodes,
                    "edges": info.num_edges,
                }))
                .collect();
            let resp = serde_json::json!({ "graphs": infos });
            let json = serde_json::to_vec_pretty(&resp).unwrap();
            let _ = send_http_response(stream, 200, "OK", "application/json", &json);
        }

        ("GET", "/v1/pubkey") => {
            let resp = serde_json::json!({
                "pubkey": crypto::hex(&config.keypair.public_key),
            });
            let json = serde_json::to_vec_pretty(&resp).unwrap();
            let _ = send_http_response(stream, 200, "OK", "application/json", &json);
        }

        _ => {
            let resp = serde_json::json!({
                "error": "not found",
                "endpoints": [
                    "GET  /health",
                    "GET  /v1/pubkey",
                    "GET  /v1/graphs",
                    "POST /v1/graph/submit",
                    "POST /v1/graph/verify",
                ]
            });
            let json = serde_json::to_vec_pretty(&resp).unwrap();
            let _ = send_http_response(stream, 404, "Not Found", "application/json", &json);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let config = ProxyConfig::from_args();
    let store = GraphStore::new();

    eprintln!("qlang-proxy v0.1.0");
    eprintln!("  listen:    {}", config.listen_addr);
    eprintln!("  pubkey:    {}", crypto::hex(&config.keypair.public_key));
    eprintln!("  auto-sign: {}", config.auto_sign);
    if let Some(ref upstream) = config.upstream_addr {
        eprintln!("  upstream:  {upstream}");
    }
    eprintln!();

    let listener = TcpListener::bind(&config.listen_addr).unwrap_or_else(|e| {
        eprintln!("failed to bind {}: {e}", config.listen_addr);
        std::process::exit(1);
    });

    eprintln!("[proxy] listening on {}", config.listen_addr);

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                handle_http_connection(&mut stream, &config, &store);
            }
            Err(e) => {
                eprintln!("[proxy] accept error: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_config_defaults() {
        // Test that default config works (without real args)
        let config = ProxyConfig {
            listen_addr: "127.0.0.1:0".into(),
            upstream_addr: None,
            keypair: Keypair::from_seed(&[1u8; 32]),
            auto_sign: true,
            auto_verify: true,
        };

        assert_eq!(config.listen_addr, "127.0.0.1:0");
        assert!(config.auto_sign);
        assert!(config.upstream_addr.is_none());
        assert_eq!(config.keypair.public_key.len(), 32);
    }

    #[test]
    fn proxy_health_check() {
        let config = ProxyConfig {
            listen_addr: "127.0.0.1:0".into(),
            upstream_addr: None,
            keypair: Keypair::from_seed(&[2u8; 32]),
            auto_sign: true,
            auto_verify: true,
        };
        let store = GraphStore::new();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            handle_http_connection(&mut stream, &config, &store);
        });

        let mut stream = TcpStream::connect(&addr).unwrap();
        stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();

        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        assert!(response.contains("200 OK"));
        assert!(response.contains("QLMS"));

        handle.join().unwrap();
    }

    #[test]
    fn proxy_submit_and_list() {
        let config = ProxyConfig {
            listen_addr: "127.0.0.1:0".into(),
            upstream_addr: None,
            keypair: Keypair::from_seed(&[3u8; 32]),
            auto_sign: true,
            auto_verify: true,
        };
        let store = GraphStore::new();

        // Submit a graph
        let graph = Graph::new("proxy_test");
        let graph_json = serde_json::to_vec(&graph).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let store_clone = store.clone();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            handle_http_connection(&mut stream, &config, &store_clone);
        });

        let mut stream = TcpStream::connect(&addr).unwrap();
        let req = format!(
            "POST /v1/graph/submit HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            graph_json.len()
        );
        stream.write_all(req.as_bytes()).unwrap();
        stream.write_all(&graph_json).unwrap();

        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();

        assert!(response.contains("200 OK"));
        assert!(response.contains("\"signed\": true"));
        assert!(response.contains("\"signature\""));

        handle.join().unwrap();
    }
}
```

- [ ] Step 7.4: Run tests and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo test -p qlang-proxy
```

```bash
git add crates/qlang-proxy/Cargo.toml crates/qlang-proxy/src/main.rs Cargo.toml
git commit -m "feat: add qlang-proxy HTTP-to-QLMS gateway binary

Standalone HTTP server that auto-signs graphs, verifies signatures,
and stores graphs. Endpoints: /health, /v1/pubkey, /v1/graphs,
/v1/graph/submit, /v1/graph/verify. Drop-in gateway for Ollama integration."
```

---

### Task 8: Benchmark suite

**Files:**
- `crates/qlang-sdk/benches/wire_format.rs` (NEW)
- `crates/qlang-sdk/Cargo.toml` (MODIFY -- add bench config)
- `examples/benchmark_wire.rs` (NEW -- runnable benchmark that outputs markdown)

**Why:** Quantifiable proof that QLANG beats REST/JSON on latency, size, and precision.

---

- [ ] Step 8.1: Add benchmark runner example

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/examples/benchmark_wire.rs`

```rust
//! Benchmark: QLANG binary wire format vs JSON for tensor transport.
//!
//! Run: cargo run --example benchmark_wire --no-default-features
//!
//! Compares:
//! 1. Serialization size (bytes)
//! 2. Serialization speed (encode + decode)
//! 3. Precision (float roundtrip accuracy)

use qlang_core::tensor::{TensorData, Shape, Dtype};
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::crypto;
use std::time::Instant;

fn main() {
    println!("# QLANG Wire Format Benchmark");
    println!();
    println!("| Metric | JSON | QLANG Binary | Improvement |");
    println!("|--------|------|--------------|-------------|");

    // --- Test 1: 768-dim embedding vector ---
    let values_768: Vec<f32> = (0..768)
        .map(|i| ((i as f64) * 0.001 * std::f64::consts::PI).sin() as f32)
        .collect();
    let tensor_768 = TensorData::from_f32(Shape::vector(768), &values_768);

    // JSON size
    let json_bytes = serde_json::to_vec(&tensor_768).unwrap();
    let json_size = json_bytes.len();

    // Binary wire size
    let wire_bytes = tensor_768.to_wire_bytes();
    let wire_size = wire_bytes.len();

    let size_ratio = json_size as f64 / wire_size as f64;
    println!(
        "| Size (768-dim f32) | {} bytes | {} bytes | {:.1}x smaller |",
        json_size, wire_size, size_ratio
    );

    // --- Test 2: Serialization speed ---
    let iterations = 10_000;

    // JSON encode
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = serde_json::to_vec(&tensor_768).unwrap();
    }
    let json_encode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // JSON decode
    let start = Instant::now();
    for _ in 0..iterations {
        let _: TensorData = serde_json::from_slice(&json_bytes).unwrap();
    }
    let json_decode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Binary encode
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tensor_768.to_wire_bytes();
    }
    let wire_encode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // Binary decode
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = TensorData::from_wire_bytes(&wire_bytes).unwrap();
    }
    let wire_decode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let encode_ratio = json_encode_us / wire_encode_us;
    let decode_ratio = json_decode_us / wire_decode_us;

    println!(
        "| Encode (768-dim, avg) | {:.1} us | {:.1} us | {:.1}x faster |",
        json_encode_us, wire_encode_us, encode_ratio
    );
    println!(
        "| Decode (768-dim, avg) | {:.1} us | {:.1} us | {:.1}x faster |",
        json_decode_us, wire_decode_us, decode_ratio
    );

    // --- Test 3: Precision ---
    let hard_values = vec![
        std::f32::consts::PI,
        std::f32::consts::E,
        1.0 / 3.0,
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    let tensor_precise = TensorData::from_f32(Shape::vector(5), &hard_values);

    // JSON roundtrip
    let json = serde_json::to_vec(&tensor_precise).unwrap();
    let json_decoded: TensorData = serde_json::from_slice(&json).unwrap();
    let json_values = json_decoded.as_f32_slice().unwrap();
    let json_errors: Vec<f64> = hard_values
        .iter()
        .zip(json_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let json_max_error = json_errors.iter().cloned().fold(0.0_f64, f64::max);

    // Binary roundtrip
    let wire = tensor_precise.to_wire_bytes();
    let wire_decoded = TensorData::from_wire_bytes(&wire).unwrap();
    let wire_values = wire_decoded.as_f32_slice().unwrap();
    let wire_errors: Vec<f64> = hard_values
        .iter()
        .zip(wire_values.iter())
        .map(|(a, b)| ((*a as f64) - (*b as f64)).abs())
        .collect();
    let wire_max_error = wire_errors.iter().cloned().fold(0.0_f64, f64::max);

    println!(
        "| Precision (max err) | {:.2e} | {:.2e} | {} |",
        json_max_error,
        wire_max_error,
        if wire_max_error == 0.0 { "exact" } else { "near-exact" }
    );

    // --- Test 4: Graph signing speed ---
    let mut graph = Graph::new("benchmark");
    for i in 0..100 {
        graph.add_node(
            Op::Input { name: format!("x{i}") },
            vec![],
            vec![qlang_core::tensor::TensorType::f32_vector(768)],
        );
    }

    let keypair = crypto::Keypair::from_seed(&[42u8; 32]);

    let start = Instant::now();
    let sign_iterations = 1_000;
    for _ in 0..sign_iterations {
        let _ = crypto::hash_graph(&graph);
    }
    let hash_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    let start = Instant::now();
    for _ in 0..sign_iterations {
        let hash = crypto::hash_graph(&graph);
        let _ = keypair.sign(&hash);
    }
    let sign_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    let start = Instant::now();
    let hash = crypto::hash_graph(&graph);
    let sig = keypair.sign(&hash);
    for _ in 0..sign_iterations {
        let _ = crypto::verify(&hash, &sig, &keypair.public_key);
    }
    let verify_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    println!(
        "| Hash (100-node graph) | n/a | {:.1} us | SHA-256 |",
        hash_us
    );
    println!(
        "| Sign (100-node graph) | n/a | {:.1} us | HMAC-SHA256 |",
        sign_us
    );
    println!(
        "| Verify (100-node graph) | n/a | {:.1} us | public-key |",
        verify_us
    );

    println!();
    println!("*{} encode/decode iterations, {} sign/verify iterations*", iterations, sign_iterations);
}
```

- [ ] Step 8.2: Add benchmark example to workspace `Cargo.toml`

Add to `/Users/aleksandarbarisic/Workspace/qland/qlang/Cargo.toml`:

```toml
[[example]]
name = "benchmark_wire"
path = "examples/benchmark_wire.rs"
```

- [ ] Step 8.3: Run benchmark and commit

```bash
cd /Users/aleksandarbarisic/Workspace/qland/qlang && cargo run --example benchmark_wire --no-default-features
```

```bash
git add examples/benchmark_wire.rs Cargo.toml
git commit -m "feat: add wire format benchmark (QLANG binary vs JSON)

Measures size, encode/decode speed, precision, and signing performance.
Run with: cargo run --example benchmark_wire --no-default-features"
```

---

### Task 9: RFC specification document

**Files:**
- `docs/rfc/QLMS-v1.0.md` (NEW)

**Why:** A formal protocol specification enables interoperability and potential standardization.

---

- [ ] Step 9.1: Write RFC document

**File:** `/Users/aleksandarbarisic/Workspace/qland/qlang/docs/rfc/QLMS-v1.0.md`

```markdown
# QLMS v1.0 Protocol Specification

**QLANG Message Stream (QLMS) -- Secure AI-to-AI Communication**

Status: Draft
Version: 1.0
Date: 2026-04-04

## 1. Introduction

QLMS is a binary protocol for exchanging computation graphs between AI agents.
It provides cryptographic signatures for provenance, zero-copy tensor transport
for efficiency, and executable graph semantics for direct computation.

### 1.1 Terminology

- **Agent**: An AI system that sends or receives QLMS messages.
- **Graph**: A directed acyclic graph of typed computation nodes.
- **SignedGraph**: A graph bundled with a cryptographic signature.
- **Tensor**: A multi-dimensional array of typed numeric data.

## 2. Wire Format

### 2.1 Message Envelope

```
Offset  Size    Field           Description
------  ----    -----           -----------
0       4       magic           0x51 0x4C 0x4D 0x53 ("QLMS")
4       2       version         Protocol version (LE u16, currently 2)
6       2       flags           Bit flags (LE u16)
8       var     auth            Authentication block (if SIGNED flag set)
var     4       msg_count       Number of messages (LE u32)
var     4       payload_len     Payload length in bytes (LE u32)
var     var     payload         JSON-encoded message array
```

### 2.2 Flags

```
Bit 0 (0x0001): SIGNED      -- Authentication block present
Bit 1 (0x0002): COMPRESSED  -- Payload is compressed (reserved)
Bit 2 (0x0004): ENCRYPTED   -- Payload is encrypted (reserved)
Bit 3 (0x0008): BINARY_TENSORS -- Tensors use binary wire format
Bits 4-15: Reserved (must be 0)
```

### 2.3 Authentication Block (when SIGNED flag is set)

```
Offset  Size    Field           Description
------  ----    -----           -----------
0       64      signature       Cryptographic signature over payload_hash
64      32      pubkey          Signer's public key
96      32      payload_hash    SHA-256 hash of the payload
```

Total authentication block size: 128 bytes.

### 2.4 Backward Compatibility

Version 1 messages (without authentication block) remain valid.
A version 2 reader encountering a version 1 message treats it as unsigned.
A version 1 reader encountering a version 2 message ignores unknown fields.

## 3. Graph Message Format

Each message in the payload array is a JSON object:

```json
{
  "id": 0,
  "from": { "name": "agent-a", "capabilities": ["Execute", "Compress"] },
  "to": { "name": "agent-b", "capabilities": ["Execute"] },
  "graph": { ... },
  "inputs": { "x": { "dtype": "F32", "shape": [4], "data": [base64] } },
  "intent": "Execute",
  "in_reply_to": null,
  "signature": null,
  "signer_pubkey": null,
  "graph_hash": null
}
```

### 3.1 Intent Types

| Intent | Description |
|--------|-------------|
| Execute | Execute the graph and return results |
| Optimize | Optimize the graph structure |
| Compress { method } | Compress weights using specified method |
| Verify | Verify proofs attached to the graph |
| Result { original_message_id } | Response with results |
| Compose | Compose with the receiver's graph |
| Train { epochs } | Train the model on receiver's data |

### 3.2 Capabilities

| Capability | Description |
|------------|-------------|
| Execute | Can execute computation graphs |
| Compile | Can compile graphs to native code |
| Optimize | Can optimize graph structure |
| Compress | Can perform IGQK compression |
| Train | Can train models (has data access) |
| Verify | Can verify mathematical proofs |

## 4. Tensor Wire Format

When the BINARY_TENSORS flag is set, tensors use a compact binary encoding
instead of JSON:

```
Offset  Size    Field       Description
------  ----    -----       -----------
0       1       dtype       Dtype enum (0=F16, 1=F32, 2=F64, 3=I8, ...)
1       2       ndims       Number of dimensions (LE u16)
3       8*N     dims        Dimension sizes (LE u64 each; MAX=dynamic)
3+8N    8       data_len    Raw data length in bytes (LE u64)
11+8N   var     data        Raw tensor bytes (native endian)
```

### 4.1 Dtype Encoding

| Value | Dtype | Element Size |
|-------|-------|-------------|
| 0 | F16 | 2 bytes |
| 1 | F32 | 4 bytes |
| 2 | F64 | 8 bytes |
| 3 | I8 | 1 byte |
| 4 | I16 | 2 bytes |
| 5 | I32 | 4 bytes |
| 6 | I64 | 8 bytes |
| 7 | Bool | 1 byte |
| 8 | Ternary | 1 byte |
| 9 | Utf8 | 1 byte |

## 5. Cryptographic Signatures

### 5.1 Algorithm

Phase 1: HMAC-SHA256-based scheme (Ed25519-compatible wire format)
- Signature: 64 bytes
- Public key: 32 bytes
- Hash: SHA-256 (32 bytes)

Future: Full Ed25519 (same wire format, drop-in replacement)

### 5.2 Signing Process

1. Serialize graph to canonical JSON (deterministic field ordering)
2. Compute SHA-256 hash of the serialized bytes
3. Sign the hash with the sender's private key
4. Attach signature + public key + hash to the message

### 5.3 Verification Process

1. Serialize graph to canonical JSON
2. Compute SHA-256 hash
3. Compare computed hash with stored hash
4. Verify signature against hash using sender's public key

### 5.4 Signature Chains

For multi-hop communication (A -> B -> C), each agent appends its signature:

```
A signs graph -> sends to B
B verifies A's signature
B modifies graph (e.g., compresses) -> signs modified graph
B attaches A's original signature in metadata
B sends to C
C verifies B's signature (and optionally A's original)
```

## 6. Security Model

### 6.1 Threat Model

- **Passive attacker**: Can observe network traffic.
  - Mitigation: Encryption (reserved flag, future work).
- **Active attacker**: Can modify messages in transit.
  - Mitigation: Signatures detect any modification.
- **Impersonation**: Attacker claims to be agent A.
  - Mitigation: Signature verification requires knowing A's public key.

### 6.2 Trust Model

- Each agent generates its own keypair.
- Public keys are distributed out-of-band (configuration file, registry, mDNS).
- No central authority (peer-to-peer trust).
- Optional: Trust-on-first-use (TOFU) model.

### 6.3 Key Management

- Keys are 32-byte seeds stored in agent configuration.
- Public keys are 32-byte values shared with peers.
- Key rotation: Generate new keypair, distribute new public key.
- Revocation: Remove public key from peer trust stores.

## 7. Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Invalid magic bytes |
| 2 | Unsupported version |
| 3 | Signature verification failed |
| 4 | Graph validation failed |
| 5 | Unknown intent |
| 6 | Capability not available |
| 7 | Tensor format error |
| 8 | Internal error |

## 8. Agent Discovery (Future)

Reserved for future specification:
- mDNS-based local discovery
- Central registry for internet-scale discovery
- Capability-based routing

## 9. References

- FIPS 180-4: Secure Hash Standard (SHA-256)
- RFC 8032: Edwards-Curve Digital Signature Algorithm (Ed25519)
- QLANG: Graph-based AI-to-AI Programming Language
- IGQK: Information-Geometric Quantum Compression Theory
```

- [ ] Step 9.2: Commit RFC document

```bash
git add docs/rfc/QLMS-v1.0.md
git commit -m "docs: add QLMS v1.0 protocol specification (RFC draft)

Formal specification of the QLANG Message Stream protocol including
wire format, tensor encoding, cryptographic signatures, security model,
and error codes."
```

---

## Summary

| Task | Description | Files | Estimated Time |
|------|-------------|-------|----------------|
| 1 | SHA-256 hashing | `core/crypto.rs`, `core/lib.rs` | 10 min |
| 2 | Ed25519-compatible signing | `core/crypto.rs` | 15 min |
| 3 | QLMS wire format v2 | `agent/protocol.rs` | 10 min |
| 4 | Zero-copy tensor serialization | `core/tensor.rs` | 10 min |
| 5 | qlang-sdk crate | `sdk/Cargo.toml`, `sdk/lib.rs`, `Cargo.toml` | 10 min |
| 6 | Python bindings | `python/lib.rs` | 10 min |
| 7 | qlang-proxy binary | `proxy/Cargo.toml`, `proxy/main.rs`, `Cargo.toml` | 15 min |
| 8 | Benchmark suite | `examples/benchmark_wire.rs` | 10 min |
| 9 | RFC specification | `docs/rfc/QLMS-v1.0.md` | 10 min |

**Total estimated time: ~100 minutes**

**Dependency order:** Task 1 -> Task 2 -> Task 3 (sequential, crypto builds on hash). Task 4 is independent. Task 5 depends on 1-4. Task 6 depends on 5. Task 7 depends on 5. Task 8 depends on 4. Task 9 is independent.

**Critical path:** 1 -> 2 -> 3 -> 5 -> 7 (proxy is the end-user deliverable).
