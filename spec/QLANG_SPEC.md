# QLANG Language Specification v0.1

## 1. Philosophy

QLANG is a **graph-based, probabilistic programming language** designed for
AI-to-AI communication. It is not a text-based language. Programs are directed
acyclic graphs (DAGs) that flow directly from one AI agent to another — or to
machine code — without the lossy detour through human-readable text.

### Five Core Principles

1. **GRAPH-FIRST** — No syntax. The AST *is* the program. No parsing errors possible.
2. **PROBABILISTIC** — Values are distributions (density operators ρ), not scalars.
3. **VERIFIABLE** — Every node carries invariants and proofs. Correctness by construction.
4. **TENSOR-NATIVE** — The fundamental data type is the tensor, not int/float.
5. **COMPOSABLE** — Programs are graphs that connect. Composition is edge-wiring.

### Non-Goals

- Human-readability as primary concern (views exist for humans, but the graph is truth)
- Backwards compatibility with existing languages
- General-purpose scripting (QLANG targets computation, not string manipulation)
- Interactive REPL (graphs are submitted whole, not line-by-line)
- Object-oriented programming (no classes, no inheritance)

## 2. Fundamental Types

### 2.1 Tensor

The base data type. All values are tensors.

```
Tensor<dtype>[shape]

dtype  := f16 | f32 | f64 | i8 | i16 | i32 | i64 | bool | ternary
shape  := [d1, d2, ..., dn]  where di ∈ ℕ or di = ? (dynamic)
ternary := {-1, 0, +1}
```

A scalar is `Tensor<f32>[]` (shape = empty = 0-dimensional tensor).
A vector is `Tensor<f32>[n]`.
A matrix is `Tensor<f32>[m, n]`.

### 2.2 Quantum State (ρ)

A density operator representing probabilistic/quantum state over tensors.

```
State<dtype>[shape]

Internally: ρ ∈ ℂ^{d×d}, ρ ≥ 0, Tr(ρ) = 1

Properties (enforced by runtime):
  - Positive semidefinite
  - Trace = 1.0
  - Smooth mapping on statistical manifold
```

### 2.3 Distribution

A classical probability distribution over discrete values.

```
Distribution<T>[n]

Internally: P(x_i) for i = 1..n, Σ P(x_i) = 1
```

### 2.4 Proof

A compile-time certificate that a constraint holds.

```
Proof {
    theorem: reference to formal theorem
    params:  instantiation of theorem parameters
    status:  verified | pending | assumed
}
```

## 3. Graph Structure

A QLANG program is a `Graph`:

```
Graph {
    id:          UniqueId
    nodes:       Vec<Node>
    edges:       Vec<Edge>
    constraints: Vec<Constraint>
    metadata:    Map<String, Value>
}
```

### 3.1 Node

A single computation unit.

```
Node {
    id:          NodeId
    op:          Operation          // what to compute
    inputs:      Vec<PortId>        // input ports
    outputs:     Vec<PortId>        // output ports
    constraints: Vec<Constraint>    // local guarantees
    proof:       Option<Proof>      // correctness certificate
}
```

### 3.2 Edge

A data flow connection between nodes.

```
Edge {
    id:     EdgeId
    from:   (NodeId, PortIndex)    // source node + output port
    to:     (NodeId, PortIndex)    // target node + input port
    dtype:  TensorType             // tensor shape + dtype flowing through
}
```

### 3.3 Port

An input or output slot on a node.

```
Port {
    id:    PortId
    name:  String
    dtype: TensorType
    kind:  Input | Output
}
```

## 4. Operations (Op Catalog)

### 4.1 Tensor Operations

| Op             | Inputs       | Outputs      | Description                |
|----------------|-------------|-------------|----------------------------|
| `add`          | a, b        | result      | Element-wise addition       |
| `mul`          | a, b        | result      | Element-wise multiplication |
| `matmul`       | a, b        | result      | Matrix multiplication       |
| `transpose`    | a           | result      | Tensor transpose            |
| `reshape`      | a, shape    | result      | Reshape tensor              |
| `slice`        | a, range    | result      | Slice subtensor             |
| `concat`       | a, b, axis  | result      | Concatenate tensors         |
| `reduce_sum`   | a, axis     | result      | Sum along axis              |
| `reduce_mean`  | a, axis     | result      | Mean along axis             |

### 4.2 Quantum / IGQK Operations

| Op              | Inputs        | Outputs       | Description                    |
|-----------------|--------------|--------------|--------------------------------|
| `superpose`     | states[]     | ρ            | Create superposition           |
| `evolve`        | ρ, H, γ, dt | ρ'           | Quantum gradient flow          |
| `measure`       | ρ, operators | distribution | Quantum measurement (Born)     |
| `entangle`      | ρ_a, ρ_b    | ρ_ab         | Create entangled state         |
| `project`       | tensor, manifold | projected | Project onto submanifold       |
| `collapse`      | ρ, observation | tensor     | Collapse state to value        |

### 4.3 Compression Operations (IGQK)

| Op              | Inputs         | Outputs     | Description                    |
|-----------------|---------------|------------|--------------------------------|
| `to_ternary`    | tensor        | ternary     | Project to {-1, 0, +1}        |
| `to_lowrank`    | tensor, rank  | lowrank     | Low-rank approximation         |
| `to_sparse`     | tensor, sparsity | sparse   | Sparsification                 |
| `fisher_metric` | model, data   | G           | Compute Fisher information     |

### 4.4 Control Flow

| Op            | Inputs           | Outputs  | Description                    |
|---------------|-----------------|---------|--------------------------------|
| `cond`        | predicate, a, b | result  | Conditional (both evaluated)   |
| `scan`        | init, body, n   | result  | Iterative computation          |
| `subgraph`    | graph, inputs   | outputs | Execute sub-graph              |

Note: There is no `if/else` or `while`. `cond` evaluates both branches
(quantum-style). `scan` is bounded iteration (no unbounded loops).

## 5. Binary Wire Format

QLANG graphs are serialized in a compact binary format for AI-to-AI transfer.
JSON is supported as a human-inspectable fallback.

### 5.1 Binary Format (`.qlg`)

```
Header:
  magic:    [0x51, 0x4C, 0x41, 0x4E]  // "QLAN"
  version:  u16
  flags:    u16
  n_nodes:  u32
  n_edges:  u32

Node Table:
  For each node:
    id:        u32
    op:        u16 (index into op catalog)
    n_inputs:  u8
    n_outputs: u8
    n_constraints: u8
    [constraint data...]

Edge Table:
  For each edge:
    from_node: u32
    from_port: u8
    to_node:   u32
    to_port:   u8
    dtype:     u8
    shape_len: u8
    [shape dims: u32...]

Constant Pool:
  Embedded tensor data (weights, biases)
```

### 5.2 JSON Format (`.qlg.json`)

Human-readable representation for debugging:

```json
{
  "qlang": "0.1",
  "graph": {
    "nodes": [
      {"id": 0, "op": "input", "outputs": [{"name": "x", "dtype": "f32", "shape": [784]}]},
      {"id": 1, "op": "matmul", "inputs": ["0:x", "const:W"], "outputs": [{"name": "y", "dtype": "f32", "shape": [128]}]},
      {"id": 2, "op": "to_ternary", "inputs": ["1:y"], "outputs": [{"name": "t", "dtype": "ternary", "shape": [128]}],
       "proof": {"theorem": "igqk_5_2", "distortion": 0.01}}
    ],
    "edges": [
      {"from": "0:0", "to": "1:0"},
      {"from": "1:0", "to": "2:0"}
    ]
  }
}
```

## 6. Compilation Pipeline

```
QLANG Graph (.qlg)
      │
      ▼
┌─────────────┐
│ Graph Read  │  Deserialize binary/JSON to in-memory graph
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Type Check  │  Verify tensor shapes match across edges
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Verify      │  Check constraint proofs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Optimize    │  Graph transformations (fuse ops, eliminate dead nodes)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Schedule    │  Topological sort, memory planning, device placement
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Codegen     │  Emit LLVM IR (or interpret directly)
└──────┬──────┘
       │
       ▼
  Machine Code (x86-64 / ARM / GPU kernel)
```

## 7. Views (Human Visualization)

The graph is the source of truth. Views are generated FROM the graph:

| View       | Format   | Audience    | Description                           |
|-----------|----------|-------------|---------------------------------------|
| Visual    | SVG/PNG  | Everyone    | Node-and-edge diagram with data flow  |
| Text      | .qlang   | Developers  | Pseudocode approximation              |
| Math      | LaTeX    | Researchers | Mathematical notation                 |
| Debug     | Terminal | Developers  | Live tensor shapes and values         |
| JSON      | .json    | Machines    | Structured but human-readable         |

## 8. Execution Model

### 8.1 Graph Submission

A QLANG program is submitted as a complete graph. There is no line-by-line
execution. The runtime:

1. Receives the graph
2. Validates it (type check + proof check)
3. Schedules node execution (topological order, parallel where possible)
4. Executes nodes, flowing tensors along edges
5. Returns output tensors

### 8.2 Determinism

Classical operations (add, matmul, etc.) are deterministic.
Quantum operations (measure, collapse) are probabilistic — results follow
the Born rule: P(w | ρ) = Tr(ρ M_w).

### 8.3 Memory Model

- Tensors are immutable once produced by a node
- No aliasing, no shared mutable state
- Memory is managed by the runtime (arena allocation per graph execution)
- No garbage collector needed

## 9. Connection to IGQK Theory

QLANG is the natural language for expressing IGQK computations:

| IGQK Concept                  | QLANG Primitive     |
|------------------------------|-------------------|
| Density operator ρ            | `State<T>[shape]`  |
| Fisher information G          | `fisher_metric` op |
| Quantum gradient flow         | `evolve` op        |
| Projection Π: M → N          | `project` op       |
| Measurement {M_w}             | `measure` op       |
| Ternary compression           | `to_ternary` op    |
| Von Neumann entropy S(ρ)      | `entropy` op       |
| Entanglement                  | `entangle` op      |

The IGQK training algorithm (Algorithm 1 from the paper) maps directly to a
QLANG graph with `evolve` → `project` → `measure` nodes.

## 10. Versioning

- Spec version: 0.1 (this document)
- Wire format version: 1
- Op catalog is extensible (new ops can be registered)
- Graphs carry their spec version for forward compatibility
