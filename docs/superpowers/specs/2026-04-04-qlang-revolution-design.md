# QLANG Revolution: Secure AI-to-AI Communication Standard

## Goal

Make QLANG the standard protocol for AI-to-AI communication by providing four capabilities that no existing protocol (MCP, REST, gRPC) can offer: cryptographically signed computation graphs, zero-copy tensor transport, mathematically verified compression, and executable messages.

## Target Audience

Open-source AI developers (Ollama, llama.cpp, vLLM community). These developers currently use REST/JSON APIs and need a compelling reason to adopt a new protocol.

## Success Criteria

1. A developer can `pip install qlang` or `cargo add qlang` and integrate QLANG into an existing project in under 5 minutes.
2. A public benchmark demonstrates measurable advantages over REST/JSON (speed, size, correctness).
3. A formal RFC specification exists that could be submitted to a standardization body.
4. The Ollama project could accept a PR adding QLANG as an alternative transport.

## Non-Goals

- Replacing text-based human-to-AI communication (that stays as REST/MCP).
- Training large language models (QLANG trains small specialized models only).
- Building a commercial product (open-source protocol first).

---

## Architecture

### The Four Killer Features

#### 1. Signed Graphs (Proof of Computation)

Every QLANG graph carries a cryptographic signature. When KI-A sends a computation to KI-B, the result is signed by KI-B. This creates an unforgeable chain of who computed what.

```
KI-A creates graph → signs with KI-A private key → sends
KI-B receives → verifies KI-A signature → executes → signs result with KI-B key → sends back
KI-A receives → verifies KI-B signature → trusts the result

Forgery is impossible. Every computation is attributable.
```

**Implementation:** Ed25519 signatures on graph hashes. Each agent has a keypair. The QLMS header includes sender signature and optional chain of previous signatures.

**Wire format addition:**
```
[QLMS Header]
[Signature: 64 bytes Ed25519]
[Signer PublicKey: 32 bytes]
[Graph Hash: 32 bytes SHA-256]
[Payload...]
```

#### 2. Zero-Copy Tensor Transport

Tensors are transmitted as raw bytes in their native format. No JSON serialization, no base64 encoding, no floating-point-to-string conversion. The receiver reads the bytes directly into GPU memory.

```
REST/JSON:   [0.342, -0.891, 0.012]  →  28 bytes text  →  parse  →  3 floats
QLANG:       [bytes: 0x3EAF...      ]  →  12 bytes raw   →  zero copy  →  3 floats
```

**Savings:** 2-10x smaller, zero parsing overhead, zero precision loss.

**Implementation:** TensorData already uses raw `Vec<u8>` internally. The wire format transmits dtype + shape + raw bytes. On Apple Silicon with MLX, the receiver can map the bytes directly into unified GPU memory without copying.

#### 3. Verified Compression (IGQK)

When QLANG compresses a model, it attaches a mathematical proof annotation that bounds the maximum accuracy loss. No other compression tool provides this guarantee.

```
GPTQ/AWQ:    "We compressed your model. Good luck."
QLANG IGQK:  "Compressed 16x. Theorem 5.2 guarantees max 2.3% accuracy loss."
```

**Implementation:** Already implemented. The `@proof theorem_5_2` annotation in QLANG graphs references the distortion bound from the IGQK theory. The bound is computed from the model's Fisher information metric and included in the graph metadata.

#### 4. Executable Messages

A QLANG message is not data - it IS a program. When KI-A sends a graph to KI-B, KI-B does not parse a description and generate code. It directly executes the graph. The message IS the computation.

```
REST:   {"operation": "matmul", "a": [[1,2],[3,4]], "b": [5,6]}
        → Parse JSON → Interpret "matmul" → Build computation → Execute

QLANG:  [Graph: matmul(input[2,2], input[2]) + tensor data]
        → Execute immediately. No interpretation step.
```

**Implementation:** The executor already does this. A graph is a DAG of typed operations with embedded tensor data. The executor walks the DAG and computes. No parsing, no interpretation, no code generation.

---

## Components to Build

### Component 1: QLANG Protocol SDK (qlang-sdk)

A library available in three languages that any developer can integrate.

**Rust (primary):**
```toml
[dependencies]
qlang-sdk = "0.1"
```

**Python (binding via PyO3/maturin):**
```bash
pip install qlang
```

**JavaScript/TypeScript (WASM):**
```bash
npm install @qlang/sdk
```

The SDK provides:
- `QlangClient` - connect to a QLANG-speaking server
- `QlangServer` - expose capabilities over QLANG protocol
- `Graph` - build computation graphs programmatically
- `Keypair` - generate and manage signing keys
- `verify()` - verify a signed graph
- `compress()` - IGQK compression with proof
- `execute()` - run a graph locally

### Component 2: qlang-proxy (Drop-in Gateway)

A standalone binary that sits between existing AI systems and adds QLANG capabilities.

```
Ollama ←HTTP→ qlang-proxy ←QLMS→ qlang-proxy ←HTTP→ llama.cpp
                  |                     |
                  + Signs all traffic   + Verifies signatures
                  + Compresses tensors  + Zero-copy transport
                  + Logs everything     + Executable graphs
```

**Usage:**
```bash
qlang-proxy --upstream http://localhost:11434 --listen 0.0.0.0:9100
```

Any existing HTTP client talks to the proxy. The proxy translates to QLMS for inter-proxy communication, adding signatures and zero-copy tensor transport transparently.

### Component 3: Benchmark Suite

A reproducible benchmark comparing QLANG vs REST/JSON for common AI operations:

1. **Latency:** Send a 768-dim embedding vector, measure round-trip time
2. **Throughput:** Stream 1000 inference requests, measure total time
3. **Size:** Compare wire format sizes for typical AI payloads
4. **Correctness:** Show that REST/JSON loses precision (float→string→float) while QLANG is exact
5. **Security:** Show signed vs unsigned message verification

**Output:** A results page at qlang.dev/benchmark with charts and raw data.

### Component 4: RFC Specification

A formal protocol specification document following RFC style:

- **QLMS Wire Format:** Header, graph encoding, tensor encoding, signature format
- **Agent Discovery:** How agents find each other (mDNS or registry)
- **Capability Negotiation:** How agents declare what they can do
- **Error Handling:** Error codes and recovery
- **Security Model:** Key management, signature chains, trust model

---

## Phased Rollout

### Phase 1: Foundation (Week 1-4)

- Implement Ed25519 signing in QLMS protocol
- Formalize wire format specification
- Build `qlang-sdk` Rust crate with signing, verification, zero-copy tensors
- Write RFC draft v0.1

### Phase 2: SDK & Proxy (Week 5-8)

- Python bindings via PyO3/maturin (`pip install qlang`)
- JavaScript/WASM bindings (`npm install @qlang/sdk`)
- Build `qlang-proxy` binary
- Integration test: Ollama ←→ proxy ←→ proxy ←→ llama.cpp

### Phase 3: Benchmark & Launch (Week 9-12)

- Build benchmark suite
- Create qlang.dev website with live demo
- Write blog post / demo video
- Submit PR to Ollama repository
- Publish RFC to relevant mailing lists

---

## Technical Decisions

### Cryptography: Ed25519

- Fast (sign: 15,000/sec, verify: 8,000/sec on modern CPU)
- Small signatures (64 bytes) and keys (32 bytes)
- No external dependencies needed (pure Rust implementation: `ed25519-dalek` crate or hand-rolled)
- Already standard in blockchain, SSH, TLS

### Wire Format: Binary with magic header

Keep existing QLMS format, extend with signature fields:
```
[0x51,0x4C,0x4D,0x53]   // "QLMS" magic
[version: u16]            // Protocol version
[flags: u16]              // Signed? Compressed? Encrypted?
[signature: 64 bytes]     // Ed25519 signature (if signed flag set)
[pubkey: 32 bytes]        // Signer public key (if signed flag set)
[payload_hash: 32 bytes]  // SHA-256 of payload
[payload_len: u64]        // Payload length
[payload...]              // Graph + Tensors
```

### Python Bindings: PyO3 + maturin

The `qlang-python` crate already exists in the workspace. Extend it with SDK functions.

### JavaScript Bindings: wasm-pack

Compile qlang-core to WASM, wrap with TypeScript types.

---

## Risks

1. **Adoption inertia:** Developers are comfortable with REST/JSON. Mitigation: qlang-proxy makes adoption zero-effort.
2. **"Not invented here":** Big players may ignore QLANG. Mitigation: Target open-source community first.
3. **Complexity:** Adding cryptography adds complexity. Mitigation: Signing is optional (flag in header).
4. **Performance claims:** Benchmarks must be honest and reproducible. Mitigation: Open-source benchmark suite, anyone can verify.
