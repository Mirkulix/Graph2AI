# QLANG

A graph-based programming language and runtime for AI systems, written in Rust.
Programs are directed acyclic graphs (DAGs) of tensor operations that can be
interpreted, JIT-compiled via LLVM, or exchanged as signed binary messages
between agents. QLANG is an experimental research project — not a production
framework.

---

## Current State (measured 2026-04-12)

All numbers below are measured on real datasets on the listed hardware.
Source files are cited so claims can be reproduced.

### Training (what works)

| Method                         | Dataset       | Samples         | Result                    | Source                                    |
|--------------------------------|---------------|-----------------|---------------------------|-------------------------------------------|
| TernaryBrain + refine          | MNIST subset  | 5K / 1K test    | 99.6% accuracy            | `crates/qlang-runtime/tests/real_mnist_brain.rs` |
| TernaryBrain, 15 refine rounds | MNIST full    | 60K / 10K test  | 84.6% accuracy            | `examples/demo.rs` (2026-04-12)           |
| FFNetwork (f32), 20 epochs     | MNIST full    | 60K / 10K test  | 90.83% accuracy           | `examples/demo.rs`                        |
| FFNetwork + QAT on GPU         | MNIST full    | 60K / 10K test  | 84.6% ternary, 30 ep, 24s | `examples/demo.rs`, RTX 2070 Super        |
| IGQK ternary pack              | 570K weights  | —               | 16.0x size reduction      | `examples/demo.rs`                        |

Hardware for GPU runs: 2x RTX 2070 Super (8 GB each), NVLink, CUDA 13.0,
driver 580.x. Training always runs on GPU 1 (`CUDA_VISIBLE_DEVICES=1`).

### JIT Performance (from README benchmark table, verified locally)

| Elements  | Interpreter | LLVM JIT | Speedup |
|-----------|-------------|----------|---------|
| 1,024     | 10.0 µs     | 680 ns   | 14.7x   |
| 65,536    | 703.6 µs    | 44.6 µs  | 15.8x   |
| 1,048,576 | 21.4 ms     | 728.4 µs | 29.4x   |

### Communication

- QLMS binary graph protocol over TCP; HMAC-SHA256 signing
- MessageBus with 6 agents, SSE live-stream
- 384-dim tensor exchange between agents in 188 ms (local)
- 14 protocol tests passing (`crates/qlang-agent`)

### Test suite

- `qlang-core`: 120 tests
- `qlang-compile`: 151 tests
- `qlang-runtime`: 550+ tests
- `qlang-agent`: 77 tests
- Total: ~900 unit/integration tests

Counts are from the `QLANG-STATUS.md` snapshot on 2026-04-12.

---

## What Does NOT Work Yet

Being explicit so nobody builds on false assumptions.

- **Hebbian learning** on MNIST: ~10% (random). Does not learn.
- **Random conv features** on CIFAR-10: ~10%. Useless as-is.
- **Random ViT features** on CIFAR-10: ~12%. Useless.
- **Hand-crafted Gabor/DCT features** on CIFAR-10: ~25%. Too weak.
- **Spiking-MNIST**: ~10% (random). STDP loop needs fixing.
- **Mamba tokenizer**: BPE vocab not persisted in `.qlmb`; generation emits
  `<unk>`. Training in `mamba_train.rs` was aborted at step 50/10000.
- **Forward-Forward on CIFAR-10**: not tested, expected around 35%.
- **CLI training on full 60K MNIST**: too slow; batch parallelization missing.
- **Quantum Gradient Flow**: the executor currently runs a simplified gradient
  step, not the full IGQK flow described in the theory PDFs.
- **Transformer backprop**: uses random perturbation, not true backpropagation.
- **Security audit**: none. HMAC-SHA256 / Ed25519 used are standard but have
  not been externally reviewed.
- **Community**: single-person project, no published papers, no tutorials
  beyond this repo.

See `docs/vault/STRATEGIC_VISION.md` §5 for the full risk list.

---

## Quick Start

Requires Rust 1.70+ (tested with 1.93). LLVM 18 is optional.

```bash
git clone https://github.com/Mirkulix/qland.git
cd qland/qlang

# Build (interpreter only, no LLVM required)
cargo build --release --workspace --no-default-features

# Build with LLVM JIT/AOT backends
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo build --release --workspace --features llvm

# Run the MNIST demo (measured numbers above come from here)
cargo run --release --example demo
```

### Ubuntu/Debian LLVM dependencies

```bash
sudo apt install llvm-18-dev libpolly-18-dev libzstd-dev
```

### Run the QO server + Web UI

```bash
QO_PORT=4747 ./target/release/qo
# → http://localhost:4747
```

### CLI

```bash
./target/release/qlang train --data data/mnist --epochs 10 --output model.qlbg
./target/release/qlang info model.qlbg
./target/release/qlang infer --model model.qlbg --input 3
./target/release/qlang bench model.qlbg
```

---

## Architecture

```
qlang/
├── crates/
│   ├── qlang-core/        # Graph, Tensor, Binary format, Crypto, Quantum ops
│   ├── qlang-compile/     # LLVM JIT/AOT, SIMD, WASM, WGSL, parser, CLI
│   ├── qlang-runtime/     # Executor, autograd, training, ternary, BitNet math
│   │   ├── forward_forward.rs   # FF ternary training
│   │   ├── ternary_brain.rs     # Statistical init + competitive Hebbian
│   │   ├── ternary_ops.rs       # Zero-multiply inference
│   │   ├── bitnet_math.rs       # Absmean, RMSNorm, LoRA, annealing
│   │   ├── gpu_train.rs         # CUDA dual-GPU training
│   │   ├── spiking.rs           # Spiking/STDP module (not yet working)
│   │   └── organism.rs          # Specialist-swarm orchestration
│   └── qlang-agent/       # QLMS protocol, MessageBus, TCP bridge
├── qo/
│   ├── qo-embed/          # candle embeddings + ResNet-18
│   ├── qo-server/         # Axum HTTP + WebSocket + SSE
│   └── qo-agents/         # Specialist agents + executor
├── frontend/              # React UI (10 tabs)
├── examples/              # Runnable examples incl. demo.rs
└── spec/                  # Language specification
```

Design principles (from `spec/QLANG_SPEC.md`):
1. Graph-first — AST is the program, no text parsing required.
2. Tensor-native — base type is `Tensor<dtype>[shape]`.
3. Verifiable — nodes can carry proof annotations linking to theorems.
4. Composable — programs are graphs; composition is edge-wiring.

---

## Language Targets

| Target       | Format      | Status                                  |
|--------------|-------------|-----------------------------------------|
| Interpreter  | direct      | works, 1x baseline                      |
| LLVM JIT     | native      | works, ~29x at 1M elements              |
| LLVM AOT     | .o          | works                                   |
| SIMD         | AVX2        | works (requires `llvm` feature)         |
| WebAssembly  | .wat        | codegen works, browser runtime untested |
| GPU          | WGSL        | codegen + wgpu execution works          |
| Binary wire  | .qlg / QLMS | works, 14 protocol tests pass           |
| ONNX         | JSON        | import/export works for MLP subset      |

---

## Comparison to Alternatives

Source: `docs/vault/STRATEGIC_VISION.md` §5.

| Feature                      | PyTorch   | JAX      | ONNX    | HuggingFace | LangChain | QLANG   |
|------------------------------|-----------|----------|---------|-------------|-----------|---------|
| Community                    | 100K+     | 20K+     | 50K+    | 200K+       | 80K+      | <10     |
| Docs                         | excellent | good     | ok      | excellent   | good      | thin    |
| Tutorials                    | thousands | hundreds | many    | thousands   | many      | none    |
| Published papers             | thousands | hundreds | many    | many        | ok        | 0       |
| Cloud integrations           | all       | all      | many    | many        | many      | none    |
| Large models (>1B params)    | yes       | yes      | yes     | yes         | via API   | no      |
| Signed binary graph protocol | no        | no       | partial | no          | no        | yes     |
| Ternary + IGQK compression   | no        | no       | no      | no          | no        | yes     |
| Graph-native language        | no        | partial  | yes     | no          | no        | yes     |

QLANG is technically differentiated (graph-as-language, signed wire format,
ternary compression, 3-tier execution in one codebase), but ecosystem-wise it
is at hobby scale.

---

## Roadmap

Full version in `docs/vault/STRATEGIC_VISION.md` §4.

### Short-term (2 weeks) — P0
- **G1** — MNIST ≥95% end-to-end on the 60K/10K split, reproducible from Web UI,
  ternary-compressed, `.qlm` under 1 MB.
- **G2** — Fix Mamba tokenizer: persist BPE vocab in `.qlmb`, finish a training
  run, PPL <200 on WikiText-2 valid.
- **G3** — Spiking-MNIST ≥85% (currently 10%). Fix STDP loop, add surrogate
  gradient fallback.

### Mid-term (2 months) — P1
- **G4** — QLMS interop with 2+ external LLMs via a signing proxy.
- **G5** — Continuous evolution daemon (24/7 swarm, persisted generations).
- **G6** — Implement real quantum gradient flow (Padé matrix exponential,
  low-rank density matrices).

### Long-term (6+ months) — P2
- **G7** — Signed model hub with Ed25519 auth.
- **G8** — FPGA / neuromorphic backend (Verilog generator for ternary MLP).
- **G9** — Self-improving organism (agent writes QLANG graphs to improve
  itself).

---

## Contributing

This is currently a single-maintainer research project. Bus factor = 1.
Contributions that help most:

1. Reproducing the measured numbers above on different hardware.
2. Fixing listed "does not work" items (especially spiking-MNIST and
   Mamba tokenizer).
3. Writing tutorials or minimal examples for the `spec/` language features.
4. External review of the crypto code (`qlang-core::crypto`).
5. Independent evaluation of the IGQK theory (`docs/` PDFs).

Before submitting code:
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --release` (pass `--features llvm` if LLVM is set up)

No CLA. Keep PRs small and focused.

---

## License

MIT — see `LICENSE`.

## Author

Aleksandar Barisic ([@Mirkulix](https://github.com/Mirkulix))

Theory references: the IGQK (Information-Geometric Quantum Compression) PDFs
in the parent directory. Not peer-reviewed.
