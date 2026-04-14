# Decisions

Key architectural and design decisions made in the QLANG project, with rationale.

## Why Rust?

**Decision:** Pure Rust for all 6 crates.

**Rationale:**
- Single binary deployment (no Python runtime, no node_modules)
- Memory safety without garbage collection (important for real-time ML inference)
- Zero-cost abstractions for performance-critical paths
- LLVM bindings via inkwell crate fit naturally
- Cross-compilation to WASM, ARM, x86 from one codebase
- Type system catches graph construction errors at compile time

## Why Zero External Dependencies (Core)?

**Decision:** `qlang-core` has zero external dependencies beyond serde. Crypto (SHA-256, HMAC-SHA256), WebSocket protocol, and HTTP are all implemented from scratch.

**Rationale:**
- Supply chain security -- no transitive dependency can inject code
- Reproducible builds across all platforms
- Understanding every byte on the wire (critical for a security protocol)
- The implementations are small (SHA-256 is ~60 lines, WebSocket framing ~100 lines)
- Follows the principle: if you can implement it in an afternoon, don't add a crate

**Exceptions:** `serde` and `serde_json` for serialization (too complex to reimplement well), `inkwell` for LLVM bindings (wrapping a C++ library), `pyo3` for Python bindings, `mlx-rs` for Apple GPU.

## Why Binary Protocol (Not JSON)?

**Decision:** QLMS uses a binary wire format with `0x51 0x4C 0x4D 0x53` ("QLMS") magic bytes.

**Rationale:**
- JSON loses IEEE 754 float precision (decimal string roundtrip)
- A 768-dim f32 vector: 3 KB raw vs 11 KB JSON (3.5x overhead)
- Binary enables zero-copy tensor transport
- No parsing ambiguity (no escaping, no Unicode edge cases)
- Cryptographic signing works naturally on raw bytes

See [[Protocol]] for the full wire format specification.

## Why Graph-First (Not Text)?

**Decision:** The graph (DAG) is the source of truth. The `.qlang` text syntax is a view for humans.

**Rationale:**
- AI agents don't need human-readable syntax
- Graphs are valid by construction (no syntax errors possible)
- Composition is edge-wiring (not string concatenation)
- 4 structured decisions vs 47 text tokens to describe a computation
- Type checking, optimization, and compilation operate on the graph directly
- The text format exists only because humans sometimes need to read/write programs

## Why LLVM is Optional?

**Decision:** LLVM compilation is behind `--features llvm`. The project builds and runs without it.

**Rationale:**
- LLVM 18 is a 500MB+ dependency that's hard to install on some platforms
- The interpreter works everywhere (including Windows without LLVM)
- Users who only need graph construction, protocol, or agent features shouldn't need LLVM
- CI tests both modes: with and without LLVM

## Why Density Matrices (Not Just Weights)?

**Decision:** Neural network weights are modeled as quantum density matrices, not plain float vectors.

**Rationale:**
- Density matrices naturally represent uncertainty about weights
- The quantum gradient flow simultaneously explores (unitary evolution) and optimizes (dissipative evolution)
- Compression to ternary {-1, 0, +1} is formalized as quantum measurement
- Mathematical theorems give provable bounds on compression distortion
- See [[IGQK]] for the full theory

## Why No Python for Core?

**Decision:** The entire runtime, compiler, and agent system are in Rust. Python bindings exist but are optional.

**Rationale:**
- QLANG targets production deployment, not Jupyter notebooks
- Python adds ~100MB runtime and GIL limitations
- AI agents sending graphs to each other shouldn't need Python installed
- The Rust implementation matches C/C++ performance for inference
- PyO3 bindings let Python users access QLANG when needed

## Why Implement Own ONNX Parser?

**Decision:** Minimal protobuf wire-format parser instead of using the `onnx` crate.

**Rationale:**
- The `onnx` crate pulls in protobuf, which pulls in 20+ dependencies
- QLANG only needs to read graph structure and weight tensors
- The minimal parser is ~200 lines and handles the subset QLANG needs
- Consistent with the zero-deps philosophy

## Why SHA-256 for Graph Hashing?

**Decision:** Every binary-encoded graph ends with a SHA-256 content hash.

**Rationale:**
- Detects corruption in transit or storage
- Enables Merkle tree verification of individual nodes
- Consistent with the signing scheme (HMAC-SHA256)
- SHA-256 is universally supported and well-understood
- The pure-Rust implementation is compact and auditable

## Why Merkle Trees?

**Decision:** Graph nodes form a Merkle tree for partial verification.

**Rationale:**
- An [[Agents]] receiving a graph can verify a single node without re-hashing the entire graph
- Incremental updates only need to re-hash the changed path
- Enables distributed trust: share proofs without sharing the full graph
- Natural fit with the DAG structure of computation graphs

#decisions #architecture #design
