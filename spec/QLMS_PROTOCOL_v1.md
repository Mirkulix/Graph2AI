# QLMS Protocol Specification v1.0

# QLANG Message Stream -- Secure AI-to-AI Communication

## Status: Draft

## Author: Aleksandar Barisic

## Date: 2026-04-04

---

## 1. Introduction

### 1.1 Purpose

The QLANG Message Stream (QLMS) protocol defines a binary wire format for
exchanging typed computation graphs between autonomous AI agents. It addresses
three fundamental limitations of text-based AI communication:

1. **Precision loss.** JSON encodes IEEE 754 floats as decimal strings. A
   768-dimensional embedding vector roundtripped through JSON loses up to
   1 ULP of precision per element. QLMS transmits raw bytes -- bit-exact
   fidelity, zero conversion cost.

2. **Size overhead.** A 768-dim f32 vector requires ~3 072 bytes of raw data.
   JSON inflates this to ~11 000 bytes (field names, decimal digits, brackets).
   QLMS adds only an 19-byte header (dtype + shape + length) to the raw payload,
   yielding a typical 3.5x size reduction.

3. **Provenance and integrity.** Text-based protocols provide no built-in
   mechanism for one agent to verify that a computation graph was produced by a
   specific other agent and has not been modified in transit. QLMS includes
   cryptographic signatures (HMAC-SHA256 with Ed25519-compatible wire format)
   so that every graph carries a verifiable chain of authorship.

### 1.2 Scope

This specification covers:

- The binary envelope format for QLMS messages (Section 3).
- The tensor wire format for zero-copy transport (Section 4).
- The graph encoding within messages (Section 5).
- Message types and intent semantics (Section 6).
- The cryptographic signing and verification model (Section 7).
- Agent capability declaration and discovery (Section 8).
- Error codes (Section 9).
- Compression annotations including IGQK (Section 10).
- Version negotiation and backward compatibility (Section 11).

### 1.3 Design Principles

- **Binary-first.** The canonical format is binary. JSON is used only as the
  internal payload encoding for graphs (to be replaced by a fully binary graph
  encoding in a future version).
- **Zero external dependencies.** All cryptographic primitives (SHA-256,
  HMAC-SHA256) are implemented in pure Rust with no external crates, following
  the project convention established in `qlang-runtime/src/web_server.rs`.
- **Wire-compatible upgrade path.** The signing scheme uses HMAC-SHA256 today
  but the wire format allocates 64 bytes for signatures and 32 bytes for public
  keys -- identical to Ed25519. Upgrading to Ed25519 requires changing only the
  internal signing logic; no wire format changes are needed.

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **Agent** | An AI system that speaks the QLMS protocol. Agents send and receive `GraphMessage` values over TCP or other byte-stream transports. |
| **Graph** | A directed acyclic graph (DAG) of typed tensor operations. The graph *is* the program -- there is no textual source code. |
| **Node** | A single computation in a graph, identified by a `NodeId` (u32). Each node has an operation type (`Op`), typed input ports, and typed output ports. |
| **Edge** | A directed data-flow connection from one node's output port to another node's input port, carrying a typed tensor. |
| **Tensor** | A multi-dimensional array of elements with a declared `Dtype` and `Shape`. The fundamental data type in QLANG. |
| **TensorData** | A concrete tensor: dtype + shape + raw bytes. |
| **Capability** | A declared function that an agent can perform (Execute, Compile, Optimize, Compress, Train, Verify). |
| **Intent** | What the sender expects the receiver to do with a graph (Execute, Optimize, Compress, Verify, Result, Compose, Train). |
| **SignedGraph** | A `Graph` bundled with a 64-byte cryptographic signature, a 32-byte signer public key, and a 32-byte SHA-256 hash. |
| **Envelope** | The outermost binary framing of a QLMS message, including magic bytes, version, flags, optional authentication block, and payload. |
| **Payload** | The body of a QLMS message: a JSON-encoded array of `GraphMessage` objects (v1/v2) or a binary-encoded graph (future). |

---

## 3. Wire Format

### 3.1 Message Envelope

All QLMS messages begin with a fixed header. The format depends on the protocol
version and flags.

#### 3.1.1 Version 1 (Unsigned)

```
Offset  Size    Field         Description
------  ------  -----------   ------------------------------------------
0       4       magic         0x51 0x4C 0x4D 0x53 ("QLMS")
4       4       msg_count     Number of messages in payload (u32 LE)
8       var     payload       JSON-encoded array of GraphMessage objects
```

Total fixed header size: 8 bytes.

Version 1 messages carry no version field, no flags, and no authentication.
They are identified by the absence of a version field: the 4 bytes at offset 4
are interpreted directly as `msg_count`.

#### 3.1.2 Version 2 (Signed)

```
Offset  Size    Field           Description
------  ------  --------------- ------------------------------------------
0       4       magic           0x51 0x4C 0x4D 0x53 ("QLMS")
4       2       version         Protocol version (u16 LE, value: 2)
6       2       flags           Bit flags (u16 LE, see Section 3.2)
8       64      signature       Cryptographic signature (present if SIGNED)
72      32      pubkey          Signer's public key (present if SIGNED)
104     32      payload_hash    SHA-256 hash of the payload bytes
136     4       msg_count       Number of messages in payload (u32 LE)
140     var     payload         JSON-encoded array of GraphMessage objects
```

If the SIGNED flag (bit 0) is NOT set, the signature, pubkey, and
payload_hash fields are omitted and the layout is:

```
Offset  Size    Field           Description
------  ------  --------------- ------------------------------------------
0       4       magic           0x51 0x4C 0x4D 0x53 ("QLMS")
4       2       version         Protocol version (u16 LE, value: 2)
6       2       flags           Bit flags (u16 LE, SIGNED=0)
8       4       msg_count       Number of messages in payload (u32 LE)
12      var     payload         JSON-encoded array of GraphMessage objects
```

### 3.2 Flags

```
Bit   Mask     Name            Description
---   ------   --------------- -----------------------------------------
0     0x0001   SIGNED          Authentication block is present after the
                               flags field. Includes signature (64 bytes),
                               public key (32 bytes), and payload hash
                               (32 bytes).
1     0x0002   COMPRESSED      Payload is zstd-compressed. (Reserved for
                               future use; MUST NOT be set in v1.0.)
2     0x0004   ENCRYPTED       Payload is AES-256-GCM encrypted. (Reserved
                               for future use; MUST NOT be set in v1.0.)
3     0x0008   BINARY_TENSORS  Tensor values within the payload use the
                               compact binary tensor wire format (Section 4)
                               instead of JSON arrays.
4-15  --       (reserved)      MUST be zero. Readers MUST reject messages
                               with unknown flags set.
```

### 3.3 Magic Bytes

The magic sequence `0x51 0x4C 0x4D 0x53` encodes the ASCII string "QLMS"
(QLANG Message Stream). Readers MUST reject any message whose first four bytes
do not match this sequence.

Note: The `QLAN` magic (`0x51 0x4C 0x41 0x4E`) is used for the graph
serialization format (`.qlg` files) defined in `qlang-core/serial.rs`. The two
formats are distinct: QLMS frames messages; QLAN frames individual graphs.

### 3.4 Byte Order

All multi-byte integer fields in the envelope and tensor wire format use
**little-endian** byte order. Tensor data bytes use the platform's native
representation (which is little-endian on all supported targets: x86_64,
aarch64, wasm32).

---

## 4. Tensor Wire Format

### 4.1 Binary Tensor Encoding

When the `BINARY_TENSORS` flag is set, or when using the `TensorData::to_wire_bytes()`
API directly, tensors are encoded as:

```
Offset     Size       Field       Description
------     ------     -------     ------------------------------------------
0          1          dtype       Dtype tag (see Section 4.2)
1          2          ndims       Number of dimensions (u16 LE)
3          8*N        dims        Dimension sizes (u64 LE each)
                                  Value u64::MAX (0xFFFFFFFFFFFFFFFF)
                                  denotes a Dynamic dimension.
3+8*N      8          data_len    Raw data length in bytes (u64 LE)
11+8*N     data_len   data        Raw tensor bytes (little-endian)
```

The total header overhead for an N-dimensional tensor is `1 + 2 + 8*N + 8`
bytes. For a 1-D vector this is 19 bytes; for a 2-D matrix this is 27 bytes.

### 4.2 Dtype Encoding

| Tag (u8) | Dtype    | Element Size | Description                          |
|----------|----------|-------------|--------------------------------------|
| 0        | F16      | 2 bytes     | IEEE 754 half-precision float         |
| 1        | F32      | 4 bytes     | IEEE 754 single-precision float       |
| 2        | F64      | 8 bytes     | IEEE 754 double-precision float       |
| 3        | I8       | 1 byte      | Signed 8-bit integer                  |
| 4        | I16      | 2 bytes     | Signed 16-bit integer                 |
| 5        | I32      | 4 bytes     | Signed 32-bit integer                 |
| 6        | I64      | 8 bytes     | Signed 64-bit integer                 |
| 7        | Bool     | 1 byte      | Boolean (0 = false, nonzero = true)   |
| 8        | Ternary  | 1 byte      | Ternary value: -1, 0, or +1 (as i8)  |
| 9        | Utf8     | 1 byte      | UTF-8 encoded text (per byte)         |

Tags 10-255 are reserved for future dtype extensions.

### 4.3 Precision Guarantees

Binary tensor encoding preserves the exact IEEE 754 bit pattern of every
floating-point element. This is a strict improvement over JSON, which must
convert floats to decimal strings and back, potentially losing the least
significant bits.

Formally: for any `TensorData` value `t`,

```
TensorData::from_wire_bytes(&t.to_wire_bytes()) == Some(t)
```

This identity holds for all dtypes, including special values (NaN, +/-Inf,
negative zero, denormals). JSON serialization does NOT guarantee this property.

### 4.4 Alignment

The wire format does not guarantee any particular alignment of the data payload.
Implementations that require aligned access (e.g., for SIMD processing) MUST
copy the data into aligned buffers after deserialization.

### 4.5 Size Comparison: Binary vs JSON

For a 768-dimensional f32 embedding vector (a common LLM intermediate result):

| Format         | Size (bytes) | Overhead |
|----------------|-------------|----------|
| Raw f32 data   | 3 072       | 0%       |
| QLMS binary    | ~3 091      | 0.6%     |
| JSON (serde)   | ~11 000     | 258%     |

---

## 5. Graph Encoding

### 5.1 JSON Graph Format

In protocol versions 1 and 2, graphs are encoded as JSON objects within the
payload. The JSON schema matches the Rust `Graph` struct in `qlang-core`:

```json
{
  "id": "model_v1",
  "version": "0.1",
  "nodes": [
    {
      "id": 0,
      "op": { "Input": { "name": "x" } },
      "input_types": [],
      "output_types": [
        { "dtype": "F32", "shape": [{ "Fixed": 768 }] }
      ],
      "constraints": [],
      "metadata": {}
    },
    {
      "id": 1,
      "op": "Relu",
      "input_types": [
        { "dtype": "F32", "shape": [{ "Fixed": 768 }] }
      ],
      "output_types": [
        { "dtype": "F32", "shape": [{ "Fixed": 768 }] }
      ],
      "constraints": [],
      "metadata": {}
    }
  ],
  "edges": [
    {
      "id": 0,
      "from_node": 0,
      "from_port": 0,
      "to_node": 1,
      "to_port": 0,
      "tensor_type": { "dtype": "F32", "shape": [{ "Fixed": 768 }] }
    }
  ],
  "constraints": [],
  "metadata": {}
}
```

### 5.2 Graph Invariants

A valid graph MUST satisfy:

1. **DAG property.** The graph MUST be acyclic. Topological sort MUST succeed.
2. **Unique node IDs.** No two nodes may share the same `NodeId`.
3. **Valid edge references.** Every edge's `from_node` and `to_node` MUST
   reference existing nodes.
4. **Type compatibility.** The `tensor_type` on an edge MUST be compatible with
   the output type of the source node's port and the input type of the target
   node's port.
5. **Port connectivity.** All input ports on non-`Input` nodes SHOULD be
   connected (unconnected ports produce a validation warning).

### 5.3 Binary Graph Format (Future)

A future protocol version will define a fully binary graph encoding, eliminating
the JSON overhead entirely. The BINARY_TENSORS flag (Section 3.2) is the first
step: it allows tensor data within otherwise-JSON messages to use the binary
tensor wire format.

### 5.4 Canonical Serialization

For hashing and signing purposes, graphs MUST be serialized to JSON using the
canonical field ordering produced by `serde_json::to_vec()` applied to the
`Graph` struct. This ordering is deterministic because Rust struct fields are
serialized in declaration order, and `HashMap` entries are serialized in
iteration order (which is deterministic for a given build).

Implementations in languages other than Rust MUST reproduce this exact byte
sequence to achieve hash compatibility. The canonical field order is:

- Graph: `id`, `version`, `nodes`, `edges`, `constraints`, `metadata`
- Node: `id`, `op`, `input_types`, `output_types`, `constraints`, `metadata`
- Edge: `id`, `from_node`, `from_port`, `to_node`, `to_port`, `tensor_type`

---

## 6. Message Types

### 6.1 GraphMessage

A `GraphMessage` is the fundamental unit of QLMS communication. It represents
a single request or response between two agents.

```json
{
  "id": 0,
  "from": {
    "name": "agent-alpha",
    "capabilities": ["Execute", "Compress"]
  },
  "to": {
    "name": "agent-beta",
    "capabilities": ["Execute"]
  },
  "graph": { "..." },
  "inputs": {
    "x": {
      "dtype": "F32",
      "shape": [{ "Fixed": 768 }],
      "data": [0, 0, 128, 63, ...]
    }
  },
  "intent": "Execute",
  "in_reply_to": null,
  "signature": null,
  "signer_pubkey": null,
  "graph_hash": null
}
```

| Field          | Type                        | Required | Description |
|----------------|-----------------------------|----------|-------------|
| id             | u64                         | Yes      | Unique message identifier within the conversation. |
| from           | AgentId                     | Yes      | Sender agent identity and capabilities. |
| to             | AgentId                     | Yes      | Intended receiver agent. |
| graph          | Graph                       | Yes      | The computation graph. |
| inputs         | Map<String, TensorData>     | Yes      | Pre-filled input tensors (may be empty). |
| intent         | MessageIntent               | Yes      | What the sender expects the receiver to do. |
| in_reply_to    | u64 or null                 | Yes      | ID of the message this is replying to, or null. |
| signature      | [u8; 64] or null            | No       | Cryptographic signature over the graph hash. |
| signer_pubkey  | [u8; 32] or null            | No       | Signer's public key. |
| graph_hash     | [u8; 32] or null            | No       | SHA-256 hash of the graph at signing time. |

### 6.2 Intent Types

| Intent                        | Description |
|-------------------------------|-------------|
| `Execute`                     | Execute the graph with the provided inputs and return the output tensors. |
| `Optimize`                    | Analyze and optimize the graph structure (e.g., fuse operations, eliminate dead nodes) and return the optimized graph. |
| `Compress { method: String }` | Compress the graph's weights using the specified method (e.g., `"ternary"`, `"lowrank"`, `"sparse"`). Return the compressed graph. |
| `Verify`                      | Verify the mathematical proofs and constraints attached to the graph's nodes. Return a verification report. |
| `Result { original_message_id: u64 }` | This message contains the results of a previously requested operation. The `original_message_id` field links to the request. |
| `Compose`                     | Compose this graph with the receiver's current graph (wire the outputs of one to the inputs of the other). |
| `Train { epochs: usize }`     | Train this model graph on the receiver's local data for the specified number of epochs. Return the trained graph. |

### 6.3 Capability Declaration

Each `AgentId` declares a list of capabilities indicating what the agent can do.
A sender SHOULD check that the receiver's declared capabilities include the
requested intent before sending.

| Capability  | Description |
|-------------|-------------|
| `Execute`   | Can execute computation graphs using an interpreter or JIT compiler. |
| `Compile`   | Can compile graphs to native machine code (e.g., via LLVM). |
| `Optimize`  | Can perform graph-level optimizations (constant folding, fusion, dead code elimination). |
| `Compress`  | Can perform IGQK compression (ternary quantization, low-rank factorization, sparsification). |
| `Train`     | Can train models using local data and compute resources. |
| `Verify`    | Can verify mathematical proofs (constraints, compression bounds, convergence guarantees). |

---

## 7. Security Model

### 7.1 Key Generation

Each agent generates a keypair from a 32-byte seed:

1. The seed is a 32-byte value, ideally produced by a cryptographically secure
   random number generator. For development, deterministic seeds are acceptable.
2. The **public key** is derived as: `SHA-256("qlang-pubkey" || seed)`.
3. The **secret key** is the seed itself.

This derivation is deterministic: the same seed always produces the same keypair.

### 7.2 Signing Process

To sign a graph:

1. **Serialize** the graph to canonical JSON using `serde_json::to_vec(&graph)`.
2. **Hash** the JSON bytes: `hash = SHA-256(json_bytes)`.
3. **Compute r**: `r = HMAC-SHA256(secret_key, hash)` (32 bytes). This value is
   unpredictable without the secret key.
4. **Compute tag**: `tag = SHA-256(r || public_key || hash)` (32 bytes).
5. **Compute s**: `s = SHA-256(tag)` (32 bytes).
6. **Signature** = `r || s` (64 bytes total).
7. Attach `signature`, `public_key`, and `hash` to the message.

### 7.3 Verification

To verify a signature given `(message, signature, public_key)`:

1. Extract `r = signature[0..32]` and `s = signature[32..64]`.
2. Recompute `tag = SHA-256(r || public_key || message)`.
3. Recompute `expected_s = SHA-256(tag)`.
4. Compare `s` with `expected_s` using constant-time comparison.
5. The signature is valid if and only if `s == expected_s`.

Note: Verification does NOT require the secret key. Anyone with the public key
can verify.

### 7.4 Envelope-Level Signing

When the SIGNED flag is set in the envelope (Section 3.1.2):

1. The payload (JSON bytes) is hashed: `payload_hash = SHA-256(payload)`.
2. The payload hash is signed with the sender's keypair.
3. The signature (64 bytes), public key (32 bytes), and payload hash (32 bytes)
   are embedded in the envelope header.
4. The receiver verifies the signature before parsing the payload.

This provides integrity protection for the entire message, including all
`GraphMessage` objects within the payload.

### 7.5 Message-Level Signing

Individual `GraphMessage` objects MAY also carry per-graph signatures in the
`signature`, `signer_pubkey`, and `graph_hash` fields. This allows:

- Different messages within a single envelope to be signed by different agents.
- Signature chains: when agent B receives a signed graph from agent A, modifies
  it, and forwards it to agent C, both A's original signature (in metadata) and
  B's new signature can be verified.

### 7.6 Trust Model

QLMS uses a **peer-to-peer trust model** with no central authority:

- Each agent generates its own keypair independently.
- Public keys are distributed **out-of-band**: configuration files, mDNS
  announcements, or manual exchange.
- An agent trusts a message if and only if the signature verifies against a
  public key in the agent's trust store.
- **Trust-on-first-use (TOFU)** is supported as an optional policy: the first
  time an agent sees a new public key, it is automatically added to the trust
  store.

### 7.7 Threat Model

| Threat                 | Attack Vector                              | Mitigation |
|------------------------|--------------------------------------------|------------|
| Passive eavesdropping  | Attacker reads network traffic              | ENCRYPTED flag (Section 3.2, future). Cleartext transport reveals graph structure and tensor data. |
| Message tampering      | Attacker modifies bytes in transit           | SIGNED flag. SHA-256 hash + HMAC-SHA256 signature detect any modification. |
| Impersonation          | Attacker claims to be agent A               | Signature verification requires A's public key. Attacker cannot forge signatures without A's secret key. |
| Replay attack          | Attacker re-sends a previously valid message | Message IDs and `in_reply_to` fields provide ordering. Agents SHOULD reject duplicate message IDs. |
| Key compromise         | Attacker obtains an agent's secret key       | Rotate the keypair and distribute the new public key. Revoke the old key from all peer trust stores. |

### 7.8 Cryptographic Strength

The current signing scheme (HMAC-SHA256-based) provides:

- **128-bit security** against forgery (birthday bound on SHA-256).
- **Deterministic signatures** (same message + same key = same signature).
- **Constant-time verification** to prevent timing side-channel attacks.

The wire format is designed for seamless upgrade to Ed25519 (RFC 8032), which
provides 128-bit security with stronger properties (existential unforgeability
under chosen-message attack).

---

## 8. Agent Discovery

### 8.1 Static Configuration

In the current implementation, agents discover each other via static
configuration. Each agent maintains a configuration file listing known peers:

```json
{
  "peers": [
    {
      "name": "compressor-1",
      "address": "tcp://192.168.1.10:9900",
      "public_key": "a1b2c3d4..."
    },
    {
      "name": "executor-1",
      "address": "tcp://192.168.1.11:9900",
      "public_key": "e5f6g7h8..."
    }
  ]
}
```

### 8.2 mDNS Discovery (Future)

A future version of this specification will define mDNS-based local network
discovery:

- Service type: `_qlms._tcp`
- TXT records: `version=2`, `capabilities=Execute,Compress`, `pubkey=<hex>`
- Agents announce themselves on the local network and discover peers
  automatically.

### 8.3 Capability-Based Routing (Future)

When multiple agents are available, a routing layer will select the best agent
based on:

1. Required capabilities (does the agent support the requested intent?).
2. Load (how many requests is the agent currently processing?).
3. Latency (how fast can the agent respond?).
4. Trust (is the agent's public key in the trust store?).

---

## 9. Error Codes

QLMS defines a set of numeric error codes for protocol-level failures. These
are returned in error response messages.

| Code | Name                         | Description |
|------|------------------------------|-------------|
| 0    | `SUCCESS`                    | Operation completed successfully. |
| 1    | `INVALID_MAGIC`              | The first 4 bytes of the message are not `0x514C4D53`. |
| 2    | `UNSUPPORTED_VERSION`        | The protocol version is not supported by the receiver. |
| 3    | `SIGNATURE_FAILED`           | The envelope or message signature did not verify. |
| 4    | `GRAPH_VALIDATION_FAILED`    | The graph failed structural validation (cycle detected, type mismatch, etc.). |
| 5    | `UNKNOWN_INTENT`             | The receiver does not recognize the requested `MessageIntent`. |
| 6    | `CAPABILITY_UNAVAILABLE`     | The receiver lacks a capability required by the intent (e.g., `Compress` intent sent to an agent without `Compress` capability). |
| 7    | `TENSOR_FORMAT_ERROR`        | A tensor's wire bytes could not be decoded (unknown dtype, truncated data, size mismatch). |
| 8    | `PAYLOAD_PARSE_ERROR`        | The payload JSON could not be parsed. |
| 9    | `HASH_MISMATCH`              | The computed SHA-256 hash of the payload does not match the stored `payload_hash`. |
| 10   | `DUPLICATE_MESSAGE_ID`       | A message with this ID has already been processed. |
| 11   | `UNKNOWN_FLAGS`              | One or more flags in the envelope are not recognized by this implementation. |
| 12   | `DECOMPRESSION_FAILED`       | The payload could not be decompressed (COMPRESSED flag). Reserved for future use. |
| 13   | `DECRYPTION_FAILED`          | The payload could not be decrypted (ENCRYPTED flag). Reserved for future use. |
| 255  | `INTERNAL_ERROR`             | An unspecified internal error occurred in the receiver. |

Error codes 14-254 are reserved for future use.

---

## 10. Compression

### 10.1 IGQK Ternary Compression

QLMS supports IGQK (Informationsgeometrische Quantenkompression) compression
as a first-class operation. When an agent receives a `Compress { method: "ternary" }`
intent, it applies ternary quantization:

1. Compute the mean absolute value of the weight tensor: `mu = mean(|W|)`.
2. Set threshold: `tau = 0.7 * mu`.
3. Quantize: `W_q[i] = +1 if W[i] > tau, -1 if W[i] < -tau, 0 otherwise`.
4. The resulting tensor has dtype `Ternary` (1 byte per element, values in {-1, 0, +1}).

This achieves up to 32x compression (f32 -> ternary) with bounded distortion.

### 10.2 Compression Proof Annotations

QLANG graphs support `Constraint` and `Proof` annotations on nodes (defined in
`qlang-core/verify.rs`). When a compression operation is applied, the
compressed graph SHOULD include:

- The compression method used.
- The measured distortion D.
- A reference to the theoretical bound (Theorem 5.2).
- The original model's metric tensor information (if available).

### 10.3 Distortion Bounds (Theorem 5.2)

The IGQK theory provides a lower bound on compression distortion. For
compression from an n-dimensional parameter manifold M to a k-dimensional
submanifold N:

```
D >= (n - k) / (2 * beta) * log(1 + beta * sigma_min^2)
```

Where:
- `n` = original dimensionality
- `k` = compressed dimensionality
- `beta` = inverse temperature parameter
- `sigma_min^2` = minimum eigenvalue of the Fisher information metric

For ternary compression (effective dimensionality reduction n -> n/16):

```
D >= (15n / 16) / (2 * beta) * log(1 + beta * sigma_min^2)
```

Agents that support `Verify` capability can check that the measured distortion
of a compressed graph is consistent with these theoretical bounds.

### 10.4 Other Compression Methods

| Method     | Op          | Description |
|------------|-------------|-------------|
| `ternary`  | `ToTernary` | Quantize weights to {-1, 0, +1}. 32x compression. |
| `lowrank`  | `ToLowRank` | Low-rank matrix factorization. Compression ratio depends on target rank. |
| `sparse`   | `ToSparse`  | Sparsification: set small weights to zero. Compression ratio depends on sparsity target. |
| `project`  | `Project`   | General projection onto a submanifold. |

---

## 11. Compatibility

### 11.1 Version Negotiation

QLMS uses a simple version negotiation scheme:

1. The sender includes its protocol version in the envelope header.
2. The receiver checks whether it supports this version.
3. If the version is not supported, the receiver returns error code 2
   (`UNSUPPORTED_VERSION`).
4. Both parties downgrade to the highest mutually supported version.

Currently defined versions:

| Version | Features |
|---------|----------|
| 1       | Basic unsigned messages. No version field in header. |
| 2       | Explicit version field, flags, SIGNED support, BINARY_TENSORS support. |

### 11.2 Backward Compatibility

- A **version 2 reader** encountering a version 1 message (detected by the
  absence of a valid version field at offset 4-5) SHOULD treat it as an
  unsigned message and parse it using the v1 layout.
- A **version 1 reader** encountering a version 2 message will fail to parse
  it (different header layout). Version 1 readers are not expected to handle
  version 2 messages.
- **Unknown flags** in version 2 messages MUST cause the receiver to reject the
  message with error code 11 (`UNKNOWN_FLAGS`). This ensures that future
  flag-dependent features (compression, encryption) are not silently ignored.

### 11.3 Forward Compatibility

Future protocol versions (3+) will maintain the following invariants:

1. The magic bytes (`0x514C4D53`) will never change.
2. The version field will always be at offset 4 (u16 LE).
3. The flags field will always be at offset 6 (u16 LE).
4. New flags will be assigned from the reserved range (bits 4-15).
5. New fields will be appended after existing fixed fields, never inserted.

---

## 12. Transport

### 12.1 TCP

The primary transport for QLMS is TCP. Agents listen on a configurable port
(default: 9900). Each TCP connection carries a sequence of QLMS messages.

Message framing on TCP:

1. Send the complete QLMS envelope as a single write.
2. The receiver reads the magic (4 bytes) to detect the start of a message.
3. The receiver reads the version/flags to determine the header layout.
4. The receiver reads the remaining header fields to determine the payload length.
5. The receiver reads the payload.

Since QLMS v1 does not include an explicit payload length in the header, the
receiver must read until the TCP connection closes or a new magic sequence is
detected. QLMS v2 SHOULD include a payload length field in future revisions.

### 12.2 WebSocket (Future)

For browser-based agents and web integrations, QLMS messages can be transported
over WebSocket. Each WebSocket binary message contains exactly one QLMS
envelope.

### 12.3 In-Process

For agents running in the same process (e.g., a training agent and a
compression agent within the same QLANG runtime), QLMS messages can be
exchanged as in-memory `Vec<u8>` buffers without network overhead.

---

## 13. Implementation Notes

### 13.1 Reference Implementation

The reference implementation is in Rust, distributed across these crates:

| Crate          | Responsibility |
|----------------|---------------|
| `qlang-core`   | Graph, TensorData, Dtype, Shape, crypto (SHA-256, HMAC-SHA256, Keypair, SignedGraph), serial (binary/JSON graph encoding). |
| `qlang-agent`  | GraphMessage, AgentConversation, AgentId, Capability, MessageIntent, TCP server/client, protocol envelope encoding. |
| `qlang-sdk`    | High-level SDK wrapping core + agent for easy integration. |
| `qlang-python` | Python bindings via PyO3/maturin. |

### 13.2 Performance Characteristics

Based on benchmarks (see `examples/benchmark_qlang_vs_json.rs`):

| Operation                          | Typical Performance |
|------------------------------------|---------------------|
| Binary tensor encode (768-dim f32) | < 5 us              |
| Binary tensor decode (768-dim f32) | < 5 us              |
| JSON tensor encode (768-dim f32)   | ~50-100 us          |
| JSON tensor decode (768-dim f32)   | ~50-100 us          |
| SHA-256 hash (100-node graph)      | ~50-200 us          |
| Sign (100-node graph)              | ~100-400 us         |
| Verify signature                   | < 5 us              |

### 13.3 Interoperability

Implementations in other languages MUST:

1. Use the exact same SHA-256 algorithm (FIPS 180-4).
2. Use the exact same HMAC-SHA256 construction (RFC 2104).
3. Use the exact same signing scheme (Section 7.2).
4. Serialize graphs to the exact same canonical JSON (Section 5.4) for hash
   compatibility.
5. Use little-endian byte order for all wire format fields.

---

## 14. References

1. **FIPS 180-4** -- Secure Hash Standard (SHS). National Institute of Standards
   and Technology, 2015. Defines SHA-256.

2. **RFC 2104** -- HMAC: Keyed-Hashing for Message Authentication. H. Krawczyk,
   M. Bellare, R. Canetti, 1997.

3. **RFC 8032** -- Edwards-Curve Digital Signature Algorithm (Ed25519).
   S. Josefsson, I. Liusvaara, 2017. Future signing algorithm for QLMS.

4. **QLANG Language Specification v0.1** -- Graph-based, probabilistic
   programming language for AI-to-AI communication. See `spec/QLANG_SPEC.md`.

5. **IGQK Theory** -- Informationsgeometrische Quantenkompression. Mathematical
   framework for neural network compression combining information geometry,
   quantum mechanics, and compression theory. See project documentation.

6. **RFC 8878** -- Zstandard Compression and the `application/zstd` Media Type.
   Y. Collet, M. Kucherawy, 2021. Future payload compression for QLMS.

7. **NIST SP 800-38D** -- Recommendation for Block Cipher Modes of Operation:
   Galois/Counter Mode (GCM) and GMAC. M. Dworkin, 2007. Future payload
   encryption for QLMS.

---

## Appendix A: Example Message Exchange

### A.1 Unsigned Execution Request (v1)

```
Bytes (hex):
51 4C 4D 53                     -- magic "QLMS"
01 00 00 00                     -- msg_count = 1
5B 7B 22 69 64 22 3A 30 ...     -- payload: [{"id":0, ...}]
```

### A.2 Signed Compression Request (v2)

```
Bytes (hex):
51 4C 4D 53                     -- magic "QLMS"
02 00                           -- version = 2
01 00                           -- flags = SIGNED
XX XX ... (64 bytes)            -- signature
XX XX ... (32 bytes)            -- public key
XX XX ... (32 bytes)            -- payload SHA-256 hash
01 00 00 00                     -- msg_count = 1
5B 7B 22 69 64 22 3A 30 ...     -- payload: [{"id":0, "intent":"Compress", ...}]
```

### A.3 Multi-Agent Signature Chain

```
1. Agent A creates graph G and signs it:
   signed_A = SignedGraph::sign(G, keypair_A)

2. Agent A sends GraphMessage to Agent B:
   msg_1 = { graph: G, signature: signed_A.signature, intent: Compress }

3. Agent B verifies A's signature:
   assert!(signed_A.verify())

4. Agent B compresses G -> G':
   G' = compress_ternary(G)

5. Agent B signs G' and preserves A's signature in metadata:
   G'.metadata["original_signer"] = hex(signed_A.pubkey)
   G'.metadata["original_signature"] = hex(signed_A.signature)
   signed_B = SignedGraph::sign(G', keypair_B)

6. Agent B sends to Agent C:
   msg_2 = { graph: G', signature: signed_B.signature, in_reply_to: msg_1.id }

7. Agent C verifies B's signature and optionally verifies A's original.
```

---

## Appendix B: Comparison with Existing Protocols

| Feature              | QLMS          | gRPC/Protobuf  | REST/JSON     | MCP           |
|----------------------|---------------|----------------|---------------|---------------|
| Binary tensors       | Native        | Bytes field     | Base64 string | N/A           |
| Graph semantics      | Native        | Custom schema   | Custom schema | N/A           |
| Cryptographic signing| Built-in      | TLS only        | JWT/OAuth     | OAuth         |
| Precision            | Bit-exact     | Bit-exact       | Lossy         | Lossy         |
| Compression support  | IGQK native   | External        | External      | N/A           |
| AI-to-AI focus       | Primary       | General         | General       | Tool-calling  |
| Overhead per tensor  | 19 bytes      | ~10 bytes       | ~8000 bytes   | N/A           |

---

*End of specification.*
