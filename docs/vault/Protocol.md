# Protocol

QLMS (QLANG Message Stream) is the binary wire format for AI-to-AI communication. Specified in `spec/QLMS_PROTOCOL_v1.md`.

## Design Goals

1. **Bit-exact fidelity** -- raw bytes, no decimal string conversion
2. **3.5x smaller** than JSON for the same data
3. **Cryptographic provenance** -- every graph can carry a signature

## Wire Formats

### QLMS Envelope (Message Stream)

#### Version 1 (Unsigned)

```
Offset  Size    Field       Description
0       4       magic       0x51 0x4C 0x4D 0x53 ("QLMS")
4       4       msg_count   Number of messages (u32 LE)
8       var     payload     JSON-encoded GraphMessage array
```

#### Version 2 (Signed)

```
Offset  Size    Field           Description
0       4       magic           0x51 0x4C 0x4D 0x53 ("QLMS")
4       2       version         Protocol version: 2 (u16 LE)
6       2       flags           Bit 0: signed (u16 LE)
8       64      signature       HMAC-SHA256 signature
72      32      pubkey          Signer public key
104     32      payload_hash    SHA-256 of payload
136     4       msg_count       Number of messages (u32 LE)
140     var     payload         JSON-encoded GraphMessage array
```

### QLBG (Binary Graph)

Compact binary serialization for individual graphs (`binary.rs`):

```
Offset  Size    Field           Description
0       4       magic           0x51 0x4C 0x42 0x47 ("QLBG")
4       2       version         Format version (u16 LE)
6       var     graph_id        Length-prefixed string
var     var     graph_version   Length-prefixed string
var     4       node_count      u32 LE
var     var     nodes           Each: id + op_tag + input_types + output_types
var     4       edge_count      u32 LE
var     var     edges           Each: from_node + from_port + to_node + to_port + type
last    32      hash            SHA-256 of everything preceding
```

Operations are encoded as single-byte tags (0-41), making graphs extremely compact.

## GraphMessage

The unit of communication between [[Agents]]:

```rust
GraphMessage {
    id: u64,                              // unique message ID
    from: AgentId,                        // sender
    to: AgentId,                          // receiver
    graph: Graph,                         // the computation graph
    inputs: HashMap<String, TensorData>,  // pre-filled data
    intent: MessageIntent,                // what to do with it
    in_reply_to: Option<u64>,            // response to which message
    signature: Option<[u8; 64]>,         // cryptographic signature
    signer_pubkey: Option<[u8; 32]>,     // public key
    graph_hash: Option<[u8; 32]>,        // SHA-256 at signing time
}
```

## Cryptographic Signing

All crypto is pure Rust (no external crates), implemented in `qlang-core/crypto.rs`. See [[Decisions]] for why.

### Signing Flow

1. Compute SHA-256 hash of the graph
2. Sign the hash with the keypair (HMAC-SHA256, Ed25519-compatible wire format)
3. Attach signature (64 bytes), public key (32 bytes), and hash (32 bytes) to message

### Verification

1. Recompute SHA-256 of the current graph
2. Check it matches the stored hash (detects tampering)
3. Verify signature against public key

Unsigned messages are accepted for backward compatibility.

## Merkle Trees

Each graph node gets its own SHA-256 hash. These form a Merkle tree (`merkle.rs`):

```
           Root Hash
          /         \
     Hash(0,1)    Hash(2,3)
      /    \       /    \
  Node0  Node1  Node2  Node3
```

### Use Cases

| Use Case | Description |
|----------|-------------|
| Partial verification | Prove a single node belongs to a signed graph |
| Incremental updates | When a graph changes, only re-hash the changed path |
| Distributed trust | Share proofs without sharing the full graph |

### MerkleProof

```rust
MerkleProof {
    node_id: u32,
    node_hash: [u8; 32],
    siblings: Vec<([u8; 32], bool)>,  // sibling hash + is_on_left
    root: [u8; 32],
}
```

## Size Comparison

```
Format          MNIST model size
JSON            ~50 KB
QLMS binary     ~3 KB
QLBG binary     ~3.2 KB
```

Typical 3.5x reduction for tensor data (no decimal digits, no field names).

## Tensor Wire Format

Tensors are transmitted as raw bytes with a minimal header:

```
[dtype: u8] [ndims: u8] [shape: ndims * u32 LE] [data: raw bytes]
```

Zero-copy on receiving end when dtypes match.

See also: [[Architecture]] for where protocol code lives, [[Glossary]] for term definitions.

#protocol #binary #crypto #merkle
