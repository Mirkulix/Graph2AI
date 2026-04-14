# Roadmap

## Was existiert (Implementiert)

### Core Sprache
- [x] Graph-Datenstruktur (DAG mit getypten Knoten und Kanten)
- [x] 40+ Operationen (Tensor, Aktivierung, Quantum, Transformer, LLM, Kontrollfluss)
- [x] Typsystem mit Shape Inference
- [x] `.qlang` Text-Parser mit Error Recovery (bis zu 50 Diagnostics)
- [x] JSON + binaere Serialisierung ([[BinaryFormat]])
- [x] SHA-256 Content Hashing ([[Crypto]])
- [x] Merkle Tree Verifikation ([[Crypto]])
- [x] Content-Addressable Computation Cache
- [x] Alle Operatoren (Arithmetik, Vergleich, Logik, Bitweise, Compound Assignment)
- [x] 53 Built-in Funktionen in der Stdlib

### Ausfuehrung (3 Tiers) -- Siehe [[Execution]]
- [x] Tier 1: LLVM JIT fuer Scripts (native Geschwindigkeit)
- [x] Tier 2: Bytecode VM (10-50x schneller als Interpreter)
- [x] Tier 3: Tree-Walking Interpreter (alle Features)
- [x] Graph Executor (topologisch sortiert)
- [x] Unified Runtime (Scripting + Graphen in einer Datei)

### Kompilierung (9 Targets)
- [x] LLVM JIT (29x schneller als Interpreter)
- [x] LLVM SIMD (AVX2 Vektorisierung)
- [x] AOT Kompilierung zu `.o` Object Files
- [x] WebAssembly (WAT)
- [x] GPU Compute Shaders (WGSL)
- [x] Interpreter
- [x] Binaerformat (`.qlg`, `.qlb`)
- [x] ONNX Import/Export
- [x] Assembly Output

### Machine Learning -- Siehe [[Training]]
- [x] Reverse-Mode Autograd
- [x] Training: SGD (mit Momentum), Adam, Gradient Clipping
- [x] LR Schedules: Constant, Step Decay, Cosine Annealing, Linear Warmup
- [x] MLP, Transformer ([[Transformer]]), Conv2D Architekturen
- [x] MNIST Data Loading (IDX Format)
- [x] Cross-Entropy Loss, Softmax, Batch Processing
- [x] Checkpoints in `.qlm`, `.qgpt`, `.qbpe` Format
- [x] [[IGQK]] Ternaere Kompression (4-16x)
- [x] MiniGPT: GPT-style Decoder-only Transformer
- [x] BPE Tokenizer (Train, Encode, Decode, Save/Load)
- [x] RMSNorm + SiLU (moderne Architektur-Optionen)
- [x] Evolutionaere Architektursuche ([[Swarm]])
- [x] Hebbian Learning fuer ternaere Gewichte ([[ParaDiffuse]])
- [x] Diffusion Engine mit DDIM Sampling ([[Diffusion]])

### GPU & Hardware -- Siehe [[GPU]]
- [x] Apple Accelerate BLAS (macOS Standard)
- [x] Apple MLX GPU Backend (optional)
- [x] wgpu Compute Shaders (NVIDIA, AMD, Intel, Apple)
- [x] Multi-GPU Data Parallel Training
- [x] Device Detection (CPU, NVIDIA, AMD, Intel, MLX)

### Agent System
- [x] [[Protocol]] (QLMS binaeres Format, signiert + unsigniert)
- [x] Strukturierter Graph Emitter fuer [[Agents]]
- [x] Auto-Negotiation von Capabilities
- [x] TCP Netzwerk-Server fuer Graph-Austausch
- [x] Distributed Training (Data Parallel, Model Parallel, Pipeline)
- [x] Graph-Diffing fuer Versionskontrolle

### Infrastruktur
- [x] [[CLI]] mit 30+ Befehlen
- [x] Interaktive REPL
- [x] [[WebUI]] mit WebSocket Streaming (neues minimalistisches Design)
- [x] [[Ollama]] LLM Integration
- [x] LSP Server (Diagnostics, Completions, Hover, Goto-Definition)
- [x] REST API Server
- [x] Model Registry + HTTP Model Hub
- [x] Python Bindings (PyO3)
- [x] C FFI (`qlang.h`)
- [x] Docker + docker-compose
- [x] GitHub Actions CI/CD
- [x] VS Code Syntax Highlighting
- [x] Eclipse Theia IDE Integration
- [x] HTTP-zu-QLMS Signing Proxy
- [x] One-Command Install Script (macOS + Linux)

## Was Arbeit braucht

### P1 -- Sollte gebaut werden
- [ ] End-to-End echtes MNIST mit >95% Accuracy auf 60K Bildern
- [ ] Autograd integriert in Graph Executor (aktuell separate Systeme)
- [ ] WebAssembly Runtime Execution (via wasmtime)
- [ ] Voller Quantum Gradient Flow im Executor (aktuell vereinfachter Gradient Step)
- [ ] Transformer Backpropagation (statt Random Perturbation)

### P2 -- Sollte hinzugefuegt werden
- [ ] Performance Regression Alerts in CI
- [ ] Model Hub Authentifizierung (Token-basiert)
- [ ] Distributed Training End-to-End Testing mit echtem TCP
- [ ] WikiText-2 Language Model mit Swarm Training
- [ ] Kontinuierliche Evolution (24/7 Swarm)
- [ ] API Dokumentation (API.md, TUTORIAL.md)

### P3 -- Enterprise
- [ ] SOC2 Compliance, Security Audit
- [ ] Enterprise SSO, Audit Logging
- [ ] SLA Dashboard, Uptime Monitoring
- [ ] Produktion-Haertung des Netzwerk-Servers
- [ ] Quantum Hardware Integration
- [ ] Neuromorphe Hardware Unterstuetzung

## Offene Forschungsfragen

Aus der [[IGQK]] Theorie:

1. **Optimales hbar** -- Wie waehlt man die Quantum-Unsicherheit optimal?
2. **Entanglement-Struktur** -- Welche Muster sind optimal ueber Layer hinweg?
3. **Quantum-Vorteil** -- Gibt es echten Speedup gegenueber klassischen Methoden?
4. **Hardware Quantum** -- Kann IGQK auf echten Quantencomputern laufen?
5. **Universalitaet** -- Funktioniert IGQK fuer alle Architekturen (CNNs, Transformers, etc.)?

## Versionsgeschichte

### v0.3.0 (2026-04-04)
- MiniGPT Transformer mit RMSNorm + SiLU
- BPE Tokenizer (Train/Encode/Decode/Save/Load)
- Swarm Training (evolutionaere Architektursuche)
- Diffusion Engine (Cosine/Linear, DDIM)
- Hebbian Learning (gradient-frei, ternaer)
- wgpu Compute Shaders (NVIDIA/AMD/Intel)
- Bytecode VM (Tier 2, 10-50x schneller)
- LLVM Script JIT (Tier 1, native Geschwindigkeit)
- Distributed Training mit Device Detection
- Content-Addressable Cache
- Neues minimalistisches WebUI Design
- 53K Zeilen Rust, 855 Tests

### v0.2.0 (2026-04-03)
- IGQK Implementation, Binary Protocol, Crypto, Web Dashboard

### v0.1.0 (2026-04-02)
- Initiales Release mit allen Core Features

Siehe [[Vision]] fuer die langfristige Richtung, [[Comparison]] fuer Positionierung.

#roadmap #status #planning
