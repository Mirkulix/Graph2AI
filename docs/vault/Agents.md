# Agents

QLANG is designed as a communication protocol between AI systems. Instead of generating text token by token, agents make 4 structured decisions to build graphs.

## Key Insight

```
Text approach:  47 tokens to describe a computation
Graph approach:  4 decisions (node type, operation, inputs, constraints)
```

Each decision is valid by construction. No syntax errors possible.

## GraphEmitter

The `GraphEmitter` (in `qlang-agent/emitter.rs`) is the structured interface for AI agents:

```rust
let mut emitter = GraphEmitter::new("classifier");
let x = emitter.input("image", Dtype::F32, Shape::matrix(1, 784));
let w = emitter.input("weights", Dtype::F32, Shape::matrix(784, 128));
let h = emitter.matmul(x, w, x_type, w_type, out_type);
let a = emitter.relu(h, out_type);
let y = emitter.output("result", a, out_type);
let graph = emitter.finish();
```

## Agent Identity

Each agent declares its capabilities:

```rust
AgentId {
    name: "trainer",
    capabilities: vec![Execute, Train, Compress],
}
```

### Capabilities

| Capability | Meaning |
|-----------|---------|
| `Execute` | Can run graphs (has a runtime) |
| `Compile` | Can compile to native code (has LLVM) |
| `Optimize` | Can run optimization passes |
| `Compress` | Can perform [[IGQK]] compression |
| `Train` | Can train models (has data access) |
| `Verify` | Can verify proofs |

## Message Intent

When sending a graph, the sender declares what it wants:

| Intent | Meaning |
|--------|---------|
| `Execute` | Run the graph and return results |
| `Optimize` | Optimize the graph and return it |
| `Compress { method }` | Compress weights using specified method |
| `Verify` | Check all proofs in the graph |
| `Result { id }` | This is the result of a previous request |
| `Compose` | Compose this graph with yours |
| `Train { epochs }` | Train this model on your data |

## Conversation Flow

```
Agent A (Trainer)                    Agent B (Compressor)
    |                                     |
    |-- GraphMessage: Compress(ternary) ->|
    |   (graph + weights + intent)        |
    |                                     |-- IGQK compress
    |                                     |-- Verify theorem 5.2
    |<- GraphMessage: Result(0) ----------|
    |   (compressed graph + proof)        |
```

## Auto-Negotiation

When two agents connect, they negotiate capabilities automatically (see `negotiate.rs`):

- Data types both support (f32, f16, int8, ternary)
- Operations both support
- Hardware (GPU type, max tensor size)
- Protocol features (streaming, compression, signing, Merkle proofs)
- Bandwidth estimation

The result is a `NegotiatedProtocol` -- the best common denominator.

## Distributed Training

Three parallelism strategies (see `distributed.rs`):

| Strategy | Description |
|----------|-------------|
| Data Parallel | Each worker trains on different data, gradients averaged |
| Model Parallel | Different workers own different layers |
| Pipeline Parallel | Workers form a pipeline, each processing one stage |

Workers have roles: `Trainer` (produces gradients) or `Aggregator` (combines gradients).

## Autonomous Loop

The `qlang-cli autonomous` command runs a feedback loop where an AI agent iteratively improves a model:

```bash
qlang-cli autonomous --task "classify MNIST" --target 95 --iterations 5
```

1. Agent designs initial architecture
2. Trains the model
3. Evaluates performance
4. If below target, modifies architecture
5. Repeats until target met or iterations exhausted

See [[CLI]] for all agent-related commands.

## Network Server

TCP-based graph exchange (`server.rs`):

- Length-prefixed JSON over TCP
- `Request` types: SubmitGraph, ExecuteGraph, CompressGraph, ListGraphs, GetGraph
- Server stores graphs, executes on demand
- See [[Protocol]] for the wire format

#agent #protocol #ai
