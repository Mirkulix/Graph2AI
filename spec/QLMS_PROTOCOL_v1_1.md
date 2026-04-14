# QLMS Protocol Specification v1.1

# QLANG Message Stream — Secure AI-to-AI Communication

## Status: Draft — targeted at Linux Foundation AAIF submission

## Author: Aleksandar Barisic

## Date: 2026-04-12

---

## 0. Relationship to v1.0

This document is **additive** to `QLMS_PROTOCOL_v1.md`. All sections from v1.0
remain normative. v1.1 introduces:

| Area | Section in v1.1 | Status |
|------|-----------------|--------|
| Federation (N-way ternary merge) | §12 | New — OPTIONAL capability |
| Measured performance (real numbers, hardware-dated) | §13 | Replaces §13.2 of v1.0 |
| Security improvements (replay, constant-time, rotation) | §14 | Extends §7 of v1.0 |
| Interoperability with MCP | §15 | New — OPTIONAL bridge |
| IANA / IPR / contact for standardization | §16 | New |

Implementations conforming to v1.0 remain valid v1.1 implementations for the
core protocol. Federation (§12) and MCP bridge (§15) are OPTIONAL capabilities;
a v1.1 implementation MAY omit them and still call itself "v1.1 core".

Sections 1–11 from `QLMS_PROTOCOL_v1.md` are incorporated by reference. A
summary is provided in Appendix C for readers who have not read v1.0.

---

## 12. Federation (OPTIONAL)

### 12.1 Motivation

QLMS v1.0 defines point-to-point AI-to-AI communication. In federated learning
and swarm settings, N ≥ 3 agents must merge their locally-trained ternary
specialists into a shared consensus without a central coordinator.

Because QLANG weights are already ternary (values in `{-1, 0, +1}`), a
**pure-count majority vote** is both cheap (no multiplication, no gradient
synchronisation) and preserves the ternary invariant.

### 12.2 Federation Capability

A new capability string is defined:

| Capability | Description |
|------------|-------------|
| `Federate` | Agent participates in N-way ternary merges: can serve its current specialist weights, can accept remote weights, and can compute the majority-vote merge. |

An agent that advertises `Federate` MUST implement §12.3 through §12.6.

### 12.3 Gossip Endpoints

Federation uses three HTTP endpoints (transport-bound, see §12.7 for the
equivalent binary framing). These match the reference implementation in
`qo/qo-server/src/routes/qlms_federation.rs`.

```
GET  /api/qlms/federation/weights
     → WeightsResponse  (this node's current ternary weights + metadata)

GET  /api/qlms/federation/eval
     → EvalResponse     (local holdout accuracy, sample count)

POST /api/qlms/federation/gossip
     body: GossipRequest { "peers": ["host:port", ...] }
     → GossipResponse   (merged_from, weight_changes, peers_ok, peers_failed)
```

### 12.4 WeightsResponse (Wire Format)

```json
{
  "node_id":    "node-a",
  "image_dim":  784,
  "n_classes":  10,
  "n_weights":  1536,
  "weights":    [-1, 0, 1, 1, -1, ...],
  "signature":  "<64 hex bytes, OPTIONAL>",
  "pubkey":     "<32 hex bytes, OPTIONAL>"
}
```

Constraints:

1. Every element of `weights` MUST be in `{-1, 0, +1}`.
2. `weights.len() == n_weights`.
3. `n_weights == image_dim * n_classes` for the reference TernaryBrain
   specialist; other topologies MAY set a different product, but both peers in
   a merge MUST agree on `n_weights`.
4. If `signature` is present, it MUST be computed over the canonical JSON of
   the object **with the `signature` and `pubkey` fields removed**, using the
   scheme defined in v1.0 §7.2. Receivers SHOULD verify.

### 12.5 Majority Vote Merge

Given `N` peer weight vectors `W_1 .. W_N`, all of equal length `K`, the merged
vector `M` is defined element-wise:

```
for k in 0..K:
    neg  = count(peers where W_i[k] <  0)
    pos  = count(peers where W_i[k] >  0)
    zero = count(peers where W_i[k] == 0)

    M[k] = +1  if pos > neg and pos > zero
    M[k] = -1  if neg > pos and neg > zero
    M[k] =  0  otherwise     (tie-break → 0, the "uncertain" state)
```

Reference: `crates/qlang-runtime/src/federation.rs::ternary_majority_vote`.

Properties proven by unit tests in that module:

- Output is always ternary (Proposition 12.5.1).
- Empty peer list → empty output.
- Single peer → identity (merged == input).
- 2-peer case is backwards compatible with v1.0 point-to-point merges.
- Length mismatch is a hard error; partial merges are NOT defined.

**Tie-break rationale.** When `pos == neg` or either equals `zero`, the merged
value is `0`. This prevents a 50/50 split from being forced into an arbitrary
sign and corresponds to "peers do not agree → prune this weight". This is
conservative and preserves the sparsity bias of ternary training.

### 12.6 GossipRequest / GossipResponse

**Request.** The initiator sends:

```json
{ "peers": ["host-a:4646", "host-b:4646", "host-c:4646"] }
```

The initiator's own weights are *always* included as an implicit peer. A
single-entry `peers` list therefore produces a 2-way merge; an empty list is a
no-op.

**Response.**

```json
{
  "node_id":         "node-a",
  "merged_from":     3,          // peers_ok + 1 (self)
  "weight_changes":  217,        // count_changes(self_before, merged)
  "peers_ok":        ["host-b:4646", "host-c:4646"],
  "peers_failed":    []
}
```

`weight_changes` is computed by
`crates/qlang-runtime/src/federation.rs::count_changes` and is the Hamming
distance between the pre-merge and post-merge weight vectors. Receivers MAY
treat a large `weight_changes` value as a signal that consensus is unstable
and back off the gossip rate.

### 12.7 Federation over Binary QLMS (OPTIONAL)

The HTTP endpoints in §12.3 are the normative reference. An implementation MAY
additionally carry the same payloads inside a QLMS v2 envelope (v1.0 §3.1.2)
using the following new intents:

| Intent | Payload | Reply Intent |
|--------|---------|--------------|
| `FederateFetchWeights` | empty | `Result` containing WeightsResponse |
| `FederateEval` | empty | `Result` containing EvalResponse |
| `FederateGossip { peers: [String] }` | GossipRequest | `Result` containing GossipResponse |

The three intents MUST be SIGNED (v1.0 §7.4) when exchanged across a trust
boundary. They MAY be unsigned on loopback or trusted cluster networks.

### 12.8 Conflict Resolution

Federation is intentionally **leaderless**: there is no authoritative node.
Conflict resolution is handled by:

1. **Majority vote** (§12.5) — the primary mechanism.
2. **Tie → 0** — deterministic, order-independent.
3. **Signed weights** (§12.4) — peers MAY reject unsigned or
   untrusted-pubkey weights before including them in the vote.
4. **`n_weights` mismatch** — hard error, peer excluded from this round.

No sequence numbers, no quorum protocol, no view changes. The gossip step is
eventually consistent under the assumption that peers repeatedly re-gossip.

---

## 13. Measured Performance (replaces v1.0 §13.2)

### 13.1 Benchmark Environment

All numbers in this section come from the benchmark at
`crates/qlang-runtime/examples/qlms_benchmark.rs`, executed on 2026-04-12
with full results recorded in `docs/vault/QLMS_BENCHMARK.md`.

| Item | Value |
|------|-------|
| Date | 2026-04-12 |
| CPU | AMD Ryzen 9 3900X (12 cores / 24 threads) |
| Arch | x86_64 |
| OS | Linux 6.19.11-200.fc43.x86_64 (Fedora 43) |
| Build | `cargo --release --no-default-features` |
| Env | `LIBTORCH_USE_PYTORCH=1` |
| Workload | TernaryBrain 64 × 24 = **1536 ternary weights**, i8 storage |
| Iterations | 1000 (serialize / deserialize / RTT), 20-iter warmup |
| Transport | Length-prefixed TCP loopback echo server |

The v1.0 spec's §13.2 contained projected numbers ("< 5 µs") with no
reproducible benchmark. Those estimates are **superseded** by §13.2 below.

### 13.2 Measured Results

| Method    | Serialized size | Serialize (ns) | Deserialize (ns) | Loopback RTT (µs) |
|-----------|-----------------|----------------|------------------|-------------------|
| QLMS bin  | **1 798 B**     | **7 658**      | **8 249**        | **77**            |
| MCP JSON  | 4 112 B         | 9 393          | 18 900           | 78                |

**Ratios (MCP / QLMS):**

| Dimension    | Ratio | Winner |
|--------------|-------|--------|
| Payload size | 2.29× | QLMS 2.29× smaller |
| Serialize    | 1.23× | QLMS 1.23× faster |
| Deserialize  | 2.29× | QLMS 2.29× faster |
| Loopback RTT | 1.01× | tie (syscall bound) |

### 13.3 Live Dual-Server Measurements

The reference federation demo
(`qo/qo-server/src/routes/qlms_demo.rs`, launched by
`scripts/qlms-dual-server.sh` on ports 4646/4747) was measured on the same
hardware on 2026-04-12:

| Exchange | Measured |
|----------|----------|
| Signed binary frame, in-process round-trip | ≈ 195 µs |
| Signed binary frame, HTTP localhost | ≈ 4 ms |
| `send-model` → `receive` → verify → ack | < 10 ms end-to-end |

The ~200× gap between §13.2 (loopback echo, 77 µs) and the in-process frame
(195 µs) is dominated by graph-hash + HMAC-SHA256 signing (~100 µs for the
1536-weight specialist, consistent with v1.0 §13.2's predictions for
SHA-256 on a 1 798-byte payload).

### 13.4 Honest Caveats

Copied verbatim from `docs/vault/QLMS_BENCHMARK.md`:

- **Localhost only.** No real network, no TLS, no proxies — a strict lower
  bound. The size advantage will widen once transport framing differences are
  included.
- **Single-shot frames.** No batching or pipelining. A realistic MCP server
  amortising multiple tool calls per TCP connection would reduce per-RTT
  overhead equally for both protocols.
- **Simulated MCP envelope.** The JSON shape matches a typical
  `jsonrpc`/`method`/`params` weight transfer; it is not negotiated against a
  live MCP peer. The size comparison is apples-to-apples for the payload
  itself.
- **HMAC parity.** QLMS carries a 32-byte HMAC-SHA256 tag; the MCP payload
  includes an equivalent `hmac_hex` field (64 hex chars). Security surface is
  comparable.
- **Small payload regime.** At 1.5 KB the absolute numbers are
  nanosecond-scale. For 1 M+ weight tensors the 2.29× size ratio compounds
  into multi-MB transfer differences.

### 13.5 What the Numbers Do NOT Claim

- No wide-area measurements. Any "×N over MCP on the open internet" claim is
  out of scope for v1.1 and would require a separate benchmark document.
- No multi-node scaling measurements. Federation (§12) is functionally tested
  (unit tests + 3-node demo) but not benchmarked for latency under load.
- No GPU path timings. The signing / merge path is pure CPU by design.

---

## 14. Security Improvements (extends v1.0 §7)

### 14.1 Replay Attack Mitigation

v1.0 §7.7 acknowledged replay as a threat and suggested message-ID
deduplication. v1.1 strengthens this:

**REQUIRED for signed messages:**

1. Every signed `GraphMessage` MUST carry a `nonce` field (u64 LE or 8-byte
   hex string in JSON) chosen uniformly at random or from a monotonic
   counter.
2. Every signed envelope MUST carry a `timestamp` field: UNIX seconds (u64
   LE) at the time of signing.
3. The `nonce` and `timestamp` MUST be included in the bytes signed (§7.2
   step 2 becomes `hash = SHA-256(graph_json || nonce || timestamp)`).

**Receiver policy (RECOMMENDED):**

- Reject any message whose `timestamp` is outside a skew window `[now - W,
  now + W]`, with `W` default 60 seconds.
- Maintain a sliding-window cache of `(pubkey, nonce)` pairs seen in the last
  `2 * W` seconds. Reject any repeat.

Error code 14 (`REPLAY_DETECTED`) is assigned for this case (previously
reserved).

### 14.2 Constant-Time HMAC Comparison

v1.0 §7.3 step 4 says "Compare `s` with `expected_s` using constant-time
comparison". The current reference implementation in
`qlang-runtime/src/crypto.rs` uses a naive byte-equality check in one code
path. v1.1 records this as an **outstanding issue** (see
`docs/vault/QLMS_SUBMISSION_CHECKLIST.md` §2.3) and mandates:

- All 32-byte and 64-byte signature/tag comparisons MUST use a
  constant-time routine of the form:

```rust
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() { return false; }
    let mut diff: u8 = 0;
    for i in 0..a.len() { diff |= a[i] ^ b[i]; }
    diff == 0
}
```

- The timing side-channel is considered **in scope** for the threat model
  (v1.0 §7.7 table row "Impersonation" is updated accordingly).

A v1.1 conformance test suite (§16.4) MUST include a statistical timing test
with ≥ 10⁴ verification calls on mismatched / matched signatures; the ratio of
mean latencies MUST lie in `[0.95, 1.05]`.

### 14.3 Key Rotation Protocol

v1.0 §7.6 said "rotate the keypair" without specifying how. v1.1 defines:

**New intent: `RotateKey`.**

```json
{
  "intent": {
    "RotateKey": {
      "old_pubkey":  "<32 hex bytes>",
      "new_pubkey":  "<32 hex bytes>",
      "effective_from": 1_744_500_000,
      "overlap_until":  1_744_586_400
    }
  }
}
```

**Semantics:**

1. The message MUST be double-signed: the envelope signature uses the **old**
   keypair, and the payload contains an inner `SignedGraph` signed with the
   **new** keypair binding `new_pubkey` to `old_pubkey`.
2. Receivers that trust `old_pubkey` SHOULD add `new_pubkey` to the trust
   store immediately.
3. During the `[effective_from, overlap_until]` window, messages signed
   with either key MUST be accepted.
4. After `overlap_until`, `old_pubkey` MUST be removed from the trust store.

`overlap_until - effective_from` SHOULD be at least 24 hours to survive
partitioned peers.

A dedicated revocation intent (`RevokeKey`) is reserved but not normative in
v1.1; it will be defined in v1.2 together with a revocation registry.

### 14.4 Updated Threat Model Table

Additions to v1.0 §7.7:

| Threat | Attack Vector | Mitigation (v1.1) |
|--------|---------------|-------------------|
| Replay attack | Attacker re-sends a previously captured signed message | §14.1 nonce + timestamp, error 14 |
| Timing side-channel | Attacker measures HMAC verification latency to learn partial tag | §14.2 constant-time compare, conformance test |
| Stale key | Agent's secret key is retired but peers still accept messages signed with it | §14.3 `RotateKey` with overlap window |

---

## 15. Interoperability with MCP (OPTIONAL)

### 15.1 Motivation

The Model Context Protocol (MCP) is the de-facto JSON-RPC schema for
LLM ↔ tool communication. v1.1 defines a **bridge** so QLMS nodes can
participate in MCP ecosystems without requiring every MCP peer to speak
binary QLMS.

Goals:

- A QLMS frame MUST be embeddable inside an MCP message round-trip.
- An MCP tool-call MUST be convertible to a trivial QLMS graph.
- No loss of QLMS signing guarantees when crossing the bridge.

Non-goals:

- Performance parity. The bridge is correctness-first; the 2.29× advantage
  of §13.2 is lost whenever a frame is base64-wrapped.

### 15.2 Embedding a QLMS Frame in MCP JSON

An entire QLMS envelope (§3.1 of v1.0) is base64-encoded and carried inside
an MCP notification:

```json
{
  "jsonrpc": "2.0",
  "method":  "qlms/v1.1/deliver",
  "params": {
    "encoding": "base64",
    "frame":    "UUxNUwIA...<base64 of the full QLMS envelope>..."
  }
}
```

The receiver:

1. Decodes `frame` (base64 → bytes).
2. Parses the bytes as a QLMS envelope.
3. Verifies magic, version, flags, and (if SIGNED) the signature.
4. Dispatches the inner `GraphMessage` array to its normal QLMS handler.

Responses follow the same pattern with method `qlms/v1.1/reply` or a standard
JSON-RPC response containing a `{"frame": "<base64>"}` result.

Because the QLMS signature covers the raw envelope bytes, base64-wrapping is
transparent to the security model: tampering inside the MCP layer fails
verification on decode.

### 15.3 Converting MCP Tool-Calls to QLMS Graphs

Every MCP tool-call of the form

```json
{
  "method": "tools/call",
  "params": { "name": "T", "arguments": { ... } }
}
```

maps to a QLMS graph with three nodes:

```
Input(args: Struct) ─▶ Call(tool: "T") ─▶ Output(result: Struct)
```

The reverse direction (QLMS → MCP) is defined only for graphs that match this
shape — arbitrary DAGs do NOT round-trip to MCP. Implementations MAY reject
an MCP-directed QLMS graph whose node count is not exactly 3.

A reference converter SHOULD live in a new module (suggested location:
`crates/qlang-runtime/src/mcp_bridge.rs`) with the following public API:

```rust
pub fn mcp_to_qlms(call: &serde_json::Value) -> Result<Graph, BridgeError>;
pub fn qlms_to_mcp(graph: &Graph) -> Result<serde_json::Value, BridgeError>;
```

**Status:** the module does not yet exist in the reference implementation;
v1.1 defines the contract so third parties can implement it before the
reference does.

### 15.4 Signing Across the Bridge

When a QLMS frame is carried inside MCP (§15.2), the v1.0 signature is
preserved as-is. No additional MCP-level signature is required by v1.1. If
the MCP channel itself is authenticated (OAuth, mTLS), the two layers are
independent — compromising one does not compromise the other.

---

## 16. Standardization Considerations

### 16.1 IANA Considerations

This section is a **placeholder** for IANA registration actions. Upon
acceptance by a standards body the following registrations are proposed:

| Registry | Value | Reference |
|----------|-------|-----------|
| TCP port registry | Port `9900/tcp` name `qlms` | v1.0 §12.1 |
| MIME types | `application/vnd.qlms+binary` | v1.0 §3.1 |
| URI schemes | `qlms://` | new |
| Media type suffix | `+qlms` | new |

No IANA action is required for v1.1 as a draft. The editor commits to filing
the requests at the `Internet-Draft → Proposed Standard` transition.

### 16.2 IPR Statement

The author (Aleksandar Barisic) hereby places the QLMS protocol
specification under the **Apache License 2.0**, matching the license of the
reference implementation in `github.com/abarisic/qlang` (to be confirmed in
the submission checklist).

No known patents read on QLMS v1.1 as of 2026-04-12. If a patent is
subsequently identified, the author commits to a **royalty-free,
irrevocable license** under W3C-style "RF on RAND terms" for any
implementation of this specification.

This statement supersedes any contrary terms in contributing-agreement
boilerplate in the source tree.

### 16.3 Contact

| Role | Contact |
|------|---------|
| Author / Editor | Aleksandar Barisic |
| Reference implementation | `github.com/abarisic/qlang` *(verify before submission)* |
| Specification source | `spec/QLMS_PROTOCOL_v1_1.md` in the above repo |
| Bug tracker | GitHub Issues in the above repo |
| Interop plugfest coordinator | TBD — see submission checklist §4.2 |

### 16.4 Conformance Test Suite

A v1.1 implementation claiming conformance MUST pass:

1. All v1.0 round-trip tests (envelope parse, tensor encode/decode,
   sign/verify, canonical JSON hash).
2. §14.1 replay rejection: re-submit a previously accepted signed message →
   error 14.
3. §14.2 constant-time timing test (ratio in `[0.95, 1.05]`).
4. §14.3 key-rotation overlap: messages signed by old key accepted up to
   `overlap_until`, rejected after.
5. §12.5 majority-vote reference vectors (see Appendix D).
6. §15.2 MCP embedding round-trip for a three-node graph.

A compliant test vector bundle lives at
`crates/qlang-runtime/tests/qlms_v1_1_conformance.rs` *(to be added; tracked
in submission checklist §2.1).*

---

## Appendix C: v1.0 Summary (non-normative)

For reviewers who have not read `QLMS_PROTOCOL_v1.md`:

- **Envelope** (§3): 4-byte magic `QLMS`, optional 2-byte version, 2-byte
  flags, optional 128-byte auth block (sig + pubkey + hash), 4-byte
  msg_count, payload.
- **Tensor wire** (§4): 1-byte dtype, 2-byte ndims, 8N-byte shape, 8-byte
  length, raw bytes. 19-byte overhead for a 1-D vector.
- **Graph** (§5): JSON-encoded DAG of typed tensor ops. Canonical field order
  for hashing.
- **Signing** (§7): SHA-256 over canonical JSON, HMAC-SHA256-based 64-byte
  signature, 32-byte public key derived as `SHA-256("qlang-pubkey" || seed)`.
  Wire-compatible with Ed25519.
- **Intents** (§6.2): Execute, Optimize, Compress, Verify, Result, Compose,
  Train.
- **Compression** (§10): Ternary, lowrank, sparse, project — native IGQK
  annotations.

## Appendix D: Federation Reference Vectors

Input peers (each a 4-element ternary vector):

```
a = [ 1, -1,  0,  1 ]
b = [ 1, -1,  0,  0 ]
c = [ 1,  1,  0, -1 ]
```

Expected merged output (from `ternary_majority_vote`):

```
m = [ 1, -1,  0,  0 ]
```

Breakdown per index:

- `k=0`: pos=3 → `+1`
- `k=1`: neg=2, pos=1, zero=0 → `-1`
- `k=2`: zero=3 → `0`
- `k=3`: pos=1, neg=1, zero=1 → tie → `0`

Additional vectors for tie-break and backwards-compat cases are provided by
the unit tests in `crates/qlang-runtime/src/federation.rs`.

---

*End of QLMS v1.1 draft. See `docs/vault/QLMS_SUBMISSION_CHECKLIST.md` for
the LF AAIF submission checklist.*
