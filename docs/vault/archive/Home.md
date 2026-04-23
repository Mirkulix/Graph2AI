# QLANG Knowledge Base

> AI-to-AI Programming Language | 53K lines Rust | Graph-based | Zero external deps (core)

## Quick Links

### Kernkonzepte
- [[Architecture]] - System-Uebersicht, 6 Crates, alle Module
- [[Language]] - Syntax-Referenz (Graph + VM)
- [[Execution]] - 3-Tier Ausfuehrung (LLVM JIT, Bytecode VM, Interpreter)
- [[BinaryFormat]] - QLBG Binaerformat (3.5x kleiner als JSON)

### Machine Learning
- [[Training]] - MLP, Transformer, Swarm, Neuroevolution, autonome Feedback-Loops
- [[Transformer]] - MiniGPT Architektur, BPE Tokenizer, RMSNorm, SiLU
- [[Diffusion]] - ParaDiffuse Port, DDIM Sampling, Cosine/Linear Schedule
- [[Swarm]] - Evolutionaere Architektursuche fuer Language Models
- [[ParaDiffuse]] - Diffusion Engine + Hebbian Learning, aus Python portiert

### Infrastruktur
- [[CLI]] - 30+ Befehle mit Beispielen
- [[WebUI]] - Live-Feed Dashboard
- [[GPU]] - wgpu Compute Shaders, Apple MLX, Accelerate BLAS
- [[Crypto]] - SHA-256, HMAC, Merkle Trees, signierte Graphen

### Theorie & Vergleiche
- [[IGQK]] - Informationsgeometrische Quantenkompression
- [[TSLM]] - Bezug zum TSLM-Projekt
- [[Comparison]] - QLANG vs Python/PyTorch, MCP, AutoGPT
- [[Vision]] - Wohin das Ganze fuehrt

### Referenz
- [[Protocol]] - QLMS Wire Format
- [[Agents]] - AI-Agent-System
- [[Ollama]] - LLM-Integration
- [[Glossary]] - Alle Begriffe A-Z
- [[Roadmap]] - Status und naechste Schritte
- [[HowTo]] - Schritt-fuer-Schritt Anleitungen
- [[Decisions]] - Architektur-Entscheidungen

## Stats

| Metrik | Wert |
|--------|------|
| **Rust-Code** | ~53.000 Zeilen |
| **Tests** | ~855 (1.319 test-Annotationen) |
| **Crates** | 6 (core, compile, runtime, agent, python, sdk) |
| **Module** | 80+ |
| **Graph Ops** | 40+ |
| **Stdlib** | 53 Built-in Funktionen |
| **Compilation Targets** | 9 (LLVM JIT, SIMD, AOT, WASM, GPU, Interpreter, Binary, ONNX, Assembly) |
| **Execution Tiers** | 3 (LLVM JIT, Bytecode VM, Tree-Walking Interpreter) |
| **CLI Befehle** | 30+ |
| **Lizenz** | MIT |
| **Autor** | Aleksandar Barisic (@Mirkulix) |

## Build

```bash
cargo build --release                        # Full Build (mit LLVM)
cargo build --release --no-default-features  # Ohne LLVM
cargo test --workspace                       # Alle Tests
cargo run --release --example train_autograd # Neural Network trainieren
```

## Repository

- GitHub: https://github.com/Mirkulix/qland
- Spec: `spec/QLANG_SPEC.md`, `spec/QLMS_PROTOCOL_v1.md`
- Examples: `examples/` (21 Dateien: 11 Rust, 5 QLANG, 5 weitere)

#qlang #home #overview
