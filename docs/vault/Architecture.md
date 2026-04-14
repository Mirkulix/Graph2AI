# Architecture

QLANG ist ein Rust-Workspace mit 6 Crates. Programme sind gerichtete azyklische Graphen (DAGs), kein Text. Der Graph ist die Quelle der Wahrheit.

## Crate Map

```
qlang/
├── crates/
│   ├── qlang-core/       Graph, Tensor, Ops, Quantum, Crypto, Merkle, Binary, Cache
│   ├── qlang-compile/    Parser, LLVM, SIMD, AOT, WASM, GPU, Optimizer, REPL, LSP, ScriptJIT
│   ├── qlang-runtime/    Executor, VM, Bytecode, Autograd, Training, Transformer, Diffusion,
│   │                     Hebbian, Swarm, IGQK, Ollama, GPU Compute, Distributed
│   ├── qlang-agent/      Protocol, Emitter, Distributed, Server, Negotiate
│   ├── qlang-python/     PyO3 bindings (import qlang)
│   └── qlang-sdk/        High-level SDK combining core + agent + runtime
├── examples/             21 Dateien (11 Rust, 5 QLANG, 5 weitere)
├── spec/                 Language + Protocol Spezifikationen
├── web/                  Dashboard HTML/CSS/JS
├── editors/
│   ├── vscode/           VS Code Syntax Highlighting Extension
│   └── theia/            Eclipse Theia IDE Integration
└── docs/vault/           Diese Obsidian Knowledge Base
```

## qlang-core (Foundation)

Die Datenstrukturen, auf denen alles aufbaut. Null externe Abhaengigkeiten jenseits von serde.

| Modul | Zweck |
|-------|-------|
| `graph.rs` | `Graph`, `Node`, `Edge`, `NodeId` -- die Programmdarstellung |
| `ops.rs` | 40+ Operationstypen (`Op` enum): Arithmetik, Aktivierungen, Quantum, Transformer, LLM |
| `tensor.rs` | `TensorType`, `TensorData`, `Dtype`, `Shape` -- das Typsystem |
| `quantum.rs` | `DensityMatrix`, `QuantumState` -- probabilistische Werte |
| `binary.rs` | Kompaktes [[BinaryFormat]] (`QLBG` Magic, SHA-256 Content Hash) |
| `crypto.rs` | [[Crypto]]: SHA-256, HMAC-SHA256, `Keypair` -- reines Rust |
| `merkle.rs` | [[Crypto]]: Merkle-Baum ueber Graph-Knoten fuer partielle Verifikation |
| `cache.rs` | Content-addressable Computation Cache mit LRU-Eviction |
| `verify.rs` | `Constraint`, `Proof`, `TheoremRef` -- formale Verifikationsprimitive |
| `serial.rs` | JSON Serialisierung/Deserialisierung |
| `type_check.rs` | Statische Typueberpruefung fuer Graphen |
| `shape_inference.rs` | Automatische Shape-Propagation |
| `stats.rs` | Graph-Statistiken (Knotenanzahl, Tiefe, etc.) |
| `ffi.rs` | C FFI Exporte (`qlang.h` Header) |
| `errors.rs` | Fehlertypen |

## qlang-compile (Compiler)

Transformiert Graphen in ausfuehrbaren Code. LLVM-Module sind hinter `--features llvm` gegateted.

| Modul | Zweck |
|-------|-------|
| `parser.rs` | `.qlang` Text-Syntax zu `Graph` (Error-Recovering, bis zu 50 Diagnostics) |
| `codegen.rs` | LLVM IR Generierung (JIT) fuer Graphen |
| `script_jit.rs` | LLVM JIT fuer **Script-Code** (Variablen, Schleifen, Funktionen) -- Tier 1 in [[Execution]] |
| `matmul_jit.rs` | Spezialisierter JIT fuer Matrix-Multiply |
| `simd.rs` | AVX2 vektorisierte Code-Generierung |
| `aot.rs` | Ahead-of-Time Kompilierung zu `.o` Object Files |
| `wasm.rs` | WebAssembly (`.wat`) Code-Generierung |
| `gpu.rs` | WGSL Compute Shader Generierung (WebGPU) |
| `optimize.rs` | 6-Pass Optimizer (DCE, Constant Folding, CSE, Op Fusion, Identity Elimination) |
| `repl.rs` | Interaktive REPL |
| `lsp.rs` | Language Server Protocol (Diagnostics, Completions, Hover, Goto-Definition) |
| `onnx.rs` | ONNX Import/Export (minimaler Protobuf-Parser, keine externen Deps) |
| `visualize.rs` | Graphviz DOT + ASCII Terminal-Visualisierung |
| `selfhost.rs` | Compiler als QLANG-Graph ausgedrueckt (Foundation) |
| `modules.rs` | Modulsystem fuer Multi-File-Projekte |
| `api.rs` | REST API fuer Graph-Operationen |
| `main.rs` | [[CLI]] Binary Entry Point |

## qlang-runtime (Execution)

Fuehrt Graphen aus. Enthaelt den gesamten ML-Training-Stack.

| Modul | Zweck |
|-------|-------|
| `executor.rs` | Topologisch-sortierender Graph-Executor |
| `vm.rs` | Stack-basierte VM fuer General-Purpose Code (let, if, for, fn, arrays, structs, dicts) |
| `bytecode.rs` | Bytecode-Compiler + Stack-VM -- 10-50x schneller als Tree-Walking. Tier 2 in [[Execution]] |
| `unified.rs` | Bridge zwischen VM (Scripting) und Graph (ML) in einer `.qlang` Datei |
| `autograd.rs` | Reverse-Mode automatische Differenzierung |
| `training.rs` | MLP Training Loop mit Backpropagation. Siehe [[Training]] |
| `transformer.rs` | Multi-Head Attention, LayerNorm, GELU, Positional Encoding |
| `transformer_train.rs` | [[Transformer]]: MiniGPT, RMSNorm, SiLU, BPE Tokenizer, Text-Generierung |
| `tokenizer.rs` | BPE Tokenizer: Train, Encode, Decode, Save/Load (`.qbpe` Format) |
| `diffusion.rs` | [[Diffusion]]: DiffusionSchedule, DDIM Sampler, Cosine/Linear Schedule |
| `hebbian.rs` | [[ParaDiffuse]]: Hebbian Learning fuer ternaere Gewichte |
| `distributed_train.rs` | Multi-GPU Data-Parallel Training, Device Detection |
| `gpu_compute.rs` | wgpu Compute Shaders fuer NVIDIA/AMD/Intel/Apple. Siehe [[GPU]] |
| `graph_train.rs` | Graph-Level Training Integration |
| `optimizers.rs` | SGD (mit Momentum), Adam, Gradient Clipping |
| `conv.rs` | Conv2D, Max Pooling, Causal Masking |
| `mnist.rs` | MNIST Data Loading (IDX Format Parser) |
| `igqk.rs` | [[IGQK]] Full Algorithm 1 Implementation |
| `fisher.rs` | Empirische Fisher Information Metrik, Natural Gradient |
| `quantum_flow.rs` | Quantum Gradient Flow Evolution |
| `linalg.rs` | Matrix-Operationen (Commutator, Anticommutator, Eigenwerte) |
| `theorems.rs` | Theorem 5.1 (Konvergenz), 5.2 (Kompression), 5.3 (Generalisierung) |
| `accel.rs` | Apple Accelerate BLAS / [[GPU]] MLX Backend |
| `ollama.rs` | [[Ollama]] LLM Client |
| `web_server.rs` | [[WebUI]] WebSocket Server (RFC 6455, reines Rust) |
| `stdlib.rs` | 53 Built-in Funktionen (Math, Arrays, Strings, I/O, Random, Tensoren, Typen) |
| `checkpoint.rs` | Model Save/Load in `.qlm` Binaerformat |
| `profiler.rs` | Per-Node Execution Timing |
| `scheduler.rs` | Wavefront Parallelism Detection + Memory Planning |
| `debugger.rs` | Interaktiver Graph-Debugger (Breakpoints, Stepping) |
| `parallel.rs` | Rayon-basierte Wavefront-Parallelausfuehrung |
| `registry.rs` | Model Registry (Save/Load/List/Delete/Compare) |
| `hub.rs` | HTTP Model Hub API Server |
| `gpu_runtime.rs` | GPU Device Abstraction mit CPU Fallback |
| `config.rs` | Runtime-Konfiguration |
| `diagnostics.rs` | Laufzeit-Diagnostik |
| `concurrency.rs` | Concurrency Utilities |
| `types.rs` | Erweiterte Typ-Definitionen |
| `graph_ops.rs` | Graph-Operationen als VM-Funktionen |
| `bench.rs` | Benchmark-Utilities |

## qlang-agent (AI Kommunikation)

Die [[Agents]] und [[Protocol]] Schicht.

| Modul | Zweck |
|-------|-------|
| `emitter.rs` | `GraphEmitter` -- strukturiertes Graph-Building fuer AI Agents |
| `protocol.rs` | `GraphMessage`, `AgentId`, `Capability`, `MessageIntent` |
| `negotiate.rs` | Auto-Negotiation von Capabilities zwischen Agents |
| `compose.rs` | Graph-Komposition (Verdrahtung von Sub-Graphen) |
| `diff.rs` | Graph-Diffing fuer Versionskontrolle |
| `distributed.rs` | Distributed Training (Data Parallel, Model Parallel, Pipeline) |
| `server.rs` | TCP Server fuer Graph-Austausch |
| `packages.rs` | Package Registry mit Standard Library |

## Datenfluss

```
.qlang Text ──► Parser ──► Graph ──► Optimizer ──► Target
                              │                      │
AI Agent ──► Emitter ──────►──┘       ┌──────────────┤
                                      │              │
                              Bytecode VM    LLVM JIT/AOT
                                      │      WASM / GPU
                              Interpreter          │
                                      ▼              │
                                   Results ◄─────────┘
```

Siehe [[Execution]] fuer Details zum 3-Tier System.

#core #architecture
