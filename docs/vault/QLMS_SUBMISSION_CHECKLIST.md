# QLMS v1.1 — Linux Foundation AAIF Submission Checklist

**Spec under review:** `spec/QLMS_PROTOCOL_v1_1.md`
**Benchmark evidence:** `docs/vault/QLMS_BENCHMARK.md` (dated 2026-04-12,
Ryzen 9 3900X)
**Reference implementation:** `crates/qlang-runtime/` + `qo/qo-server/`
**Date of this checklist:** 2026-04-12
**Maintainer:** Aleksandar Barisic

This checklist tracks submission readiness to the Linux Foundation
**AI Alliance / Agentic Interop Forum (AAIF)**. Items are marked:

- `[x]` — complete, evidence attached
- `[~]` — partial, known gap documented
- `[ ]` — not started

---

## 1. Technical Review

### 1.1 Specification completeness

- [x] Wire format fully specified (v1.0 §3, §4 + v1.1 §12.4, §15.2)
- [x] Canonical JSON ordering documented (v1.0 §5.4)
- [x] Intents enumerated (v1.0 §6.2 + v1.1 §12.7, §14.3, §15.3)
- [x] Error codes table (v1.0 §9 + v1.1 error 14 `REPLAY_DETECTED`)
- [x] Version negotiation + backward compat (v1.0 §11)
- [x] Transport sections (v1.0 §12 + v1.1 §12.7 federation-over-binary)
- [~] Formal grammar (ABNF/ASN.1) of the envelope — **GAP**: envelope
  described in tables only, no formal grammar. Acceptable for AAIF draft but
  reviewers may request ABNF before Proposed Standard.

### 1.2 Reference implementation parity

- [x] `qlang-runtime::federation::ternary_majority_vote` matches §12.5
- [x] `qlang-runtime::federation::count_changes` matches §12.6
- [x] `qo-server::routes::qlms_federation` matches §12.3 endpoints
- [x] `qo-server::routes::qlms_demo` demonstrates §13.3 dual-server path
- [x] `qlang-runtime::examples::qlms_benchmark` is the source of §13.2 numbers
- [ ] `qlang-runtime::mcp_bridge` module (§15.3) — **not yet written**;
  contract is specified, reference implementation is a follow-up.
- [ ] `qlang-runtime::tests::qlms_v1_1_conformance` (§16.4) — **not yet
  written**; test vectors for Appendix D exist in `federation.rs` unit tests.

### 1.3 Performance claims

- [x] All numbers in §13 cite `docs/vault/QLMS_BENCHMARK.md`
- [x] Hardware + date + build flags recorded
- [x] Run command reproducible (single `cargo run` line)
- [x] Caveats section (§13.4, §13.5) explicitly bounds the claims
- [ ] Independent reproduction by a second reviewer on different hardware —
  **REQUIRED before AAIF presentation**.

### 1.4 Security

- [x] Threat model (v1.0 §7.7 + v1.1 §14.4)
- [x] Replay mitigation (v1.1 §14.1) with error code
- [x] Key rotation (v1.1 §14.3)
- [~] Constant-time HMAC compare (v1.1 §14.2) — **spec says MUST, reference
  impl is not yet audited**. See §2.3 below.
- [ ] Third-party crypto review of the HMAC-based signing scheme. Strongly
  recommended before Standards-Track advancement; the wire format is
  Ed25519-compatible so the upgrade path is clean.
- [ ] Fuzz-test corpus for the envelope parser. Low-hanging work;
  `cargo-fuzz` target should exist before submission.

---

## 2. Outstanding Gaps (must-fix before submission)

### 2.1 Conformance test suite

**File:** `crates/qlang-runtime/tests/qlms_v1_1_conformance.rs`
**Status:** not written.
**Scope:** six tests listed in v1.1 §16.4.
**Effort:** ~1 day. The reference vectors for test 5 (majority vote) already
exist in `crates/qlang-runtime/src/federation.rs::tests`.

### 2.2 MCP bridge module

**File:** `crates/qlang-runtime/src/mcp_bridge.rs`
**Status:** spec contract only (v1.1 §15.3).
**Effort:** 2–3 days including round-trip tests. Blocking for any claim
"interop with MCP ecosystems".

### 2.3 Constant-time comparison audit

**Files:** `crates/qlang-runtime/src/crypto.rs` (check), any `==` on tag or
signature byte slices across the workspace.
**Status:** spec mandates constant-time; impl likely uses naive `==`.
**Action:** grep for `signature ==`, `tag ==`, `s ==` in crypto paths;
replace with the `ct_eq` helper from v1.1 §14.2; add the statistical
timing-regression test required by §16.4.
**Effort:** 2 hours.

### 2.4 Nonce + timestamp fields on signed messages

**Status:** spec'd in v1.1 §14.1; `GraphMessage` struct does not yet carry
them. Adding them is backwards-incompatible with v1.0 signatures and is the
single biggest structural change in v1.1.
**Decision needed:** either
  (a) bump signing-scheme version to v2 and keep v1.0 as-is, or
  (b) carry nonce/timestamp in envelope flags + separate field.
Pick one before submission.

### 2.5 IPR confirmation

**Status:** v1.1 §16.2 declares Apache-2.0. Verify the repository `LICENSE`
file matches and that no dependency license conflicts. Run `cargo deny
check licenses` before submission.

---

## 3. Interop Test Plan

### 3.1 Self-interop (reference vs reference)

- [x] Two `qo-server` instances on ports 4646 / 4747 exchange signed binary
  frames (`scripts/qlms-dual-server.sh`). Round-trip measured at ~195 µs
  in-process, ~4 ms HTTP.
- [x] Three-node federation demo: gossip reduces `weight_changes` over
  successive rounds (evidence: `qo-server::routes::qlms_federation` tests).
- [ ] Long-duration soak (24 h, ≥ 10⁶ messages) without memory growth.
  **Not yet run.**

### 3.2 Cross-implementation (Rust ↔ other)

- [ ] Python implementation of the envelope parser + signer. Smallest path:
  a ~200-line script using `hashlib` + `hmac`. Confirms the canonical JSON
  hash is reproducible in a second language. **Required for AAIF interop
  plugfest.**
- [ ] TypeScript / JS implementation for browser interop (§v1.0 12.2
  WebSocket transport). Nice-to-have for v1.1; not blocking.
- [ ] Conformance vectors: a golden-file corpus of 20+ envelopes with known
  hashes/signatures that any implementation must reproduce byte-for-byte.

### 3.3 MCP interop

- [ ] Round-trip a real MCP tool-call through §15.3 converter. Requires
  §2.2 to be done first.
- [ ] Verify that at least one off-the-shelf MCP client (Claude Desktop,
  Continue.dev, or similar) can deliver a `qlms/v1.1/deliver` notification
  without schema complaints.

### 3.4 Adversarial

- [ ] Run the fuzz corpus (§1.4) through the envelope parser for ≥ 1 CPU-hour
  without panics or unwinds.
- [ ] Attempt replay: capture a signed message, re-submit after `W+1`
  seconds, confirm error 14.
- [ ] Attempt tag-byte flip: single-bit flip in the signature → verification
  fails (should already be covered by v1.0 sign/verify tests; confirm).

---

## 4. Community Adoption Evidence

### 4.1 Current state

- **Deployments:** 1 (reference: `qo-server` running on the maintainer's
  dev machine).
- **Third-party implementations:** 0.
- **Citations / references:** internal docs only
  (`docs/vault/Protocol.md`, `docs/vault/Comparison.md`).
- **Public demo:** live dual-server federation demo reachable via the QO-UI
  frontend (commit `e353e57` Neural Noir redesign).

### 4.2 Adoption asks for AAIF sponsorship

- [ ] At least one external implementation (Python or TS) before submission.
  The spec is small enough that a motivated contributor can do this in a
  weekend.
- [ ] Coordinator for an interop plugfest (currently TBD, see spec §16.3).
- [ ] Mailing list / Matrix channel for the specification. Create at:
  `qlms-dev@lists.linuxfoundation.org` (requested at submission time).
- [ ] Two independent reviewers who are not the author. At least one should
  be a protocol-design expert (e.g., from the CNCF or IETF community); one
  should be an ML-systems expert (e.g., vLLM, Llama.cpp, Candle).

### 4.3 Positioning

QLMS is positioned as a **narrow complement** to MCP, not a replacement:

- MCP owns LLM ↔ tool semantics.
- QLMS owns AI ↔ AI tensor/graph transfer with integrity guarantees.

The §15 bridge makes the two non-competing. This positioning should be
stated explicitly in the AAIF cover letter to avoid the "yet another
protocol" objection.

---

## 5. Submission Package Contents

When filing the AAIF request, bundle the following files:

| File | Purpose |
|------|---------|
| `spec/QLMS_PROTOCOL_v1.md` | v1.0 normative base |
| `spec/QLMS_PROTOCOL_v1_1.md` | v1.1 additive draft |
| `docs/vault/QLMS_BENCHMARK.md` | Performance evidence (§13) |
| `docs/vault/QLMS_SUBMISSION_CHECKLIST.md` | This file |
| `crates/qlang-runtime/src/federation.rs` | Reference merge algorithm |
| `crates/qlang-runtime/examples/qlms_benchmark.rs` | Benchmark source |
| `qo/qo-server/src/routes/qlms_federation.rs` | HTTP endpoint reference |
| `qo/qo-server/src/routes/qlms_demo.rs` | Dual-server demo |
| Cover letter (~1 page) | Scope, positioning vs MCP, maintainer commitment |

---

## 6. Go / No-Go Summary

**Ready now:**

- Core protocol (v1.0) — stable, implemented, tested.
- Federation (v1.1 §12) — implemented, tested, documented.
- Measured performance (v1.1 §13) — real numbers, reproducible.
- Spec prose for security upgrades (§14) and MCP bridge (§15).

**Must close before filing:**

- §2.1 Conformance test suite
- §2.3 Constant-time comparison audit
- §2.4 Nonce/timestamp structural decision
- §3.2 Second-language implementation + golden-file vectors
- §4.2 Two external reviewers identified

**Recommendation:** File after §2.1, §2.3, §2.4, and §3.2 are complete. §4.2
can be acquired in parallel with review. §2.2 (MCP bridge code) can ship in
v1.2; the contract is spec'd, which is sufficient for AAIF draft status.
