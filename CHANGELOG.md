# Changelog

All notable changes to QLANG are documented in this file.
Dates use ISO format. Measured numbers cite their source file.

## [Unreleased] — 2026-04-12

### Added
- **Honest status audit** (`QLANG-STATUS.md`, commit `2386657`). All training
  numbers re-measured on real MNIST 60K/10K; synthetic-data claims removed.
  Explicit list of what does not work yet (Hebbian, random conv features,
  spiking-MNIST, Mamba tokenizer, full CIFAR-10 pipeline).
- **GPU QAT + Master Agent + MNIST+IGQK demo** (commit `467f1e0`).
  Forward-Forward with Quantization-Aware Training on GPU reaches 84.6%
  ternary accuracy on MNIST full (10K test set, 30 epochs, 24 s on a single
  RTX 2070 Super). Master agent orchestrates the demo pipeline end-to-end.
  Source: `crates/qlang-runtime/src/gpu_train.rs`, `examples/demo.rs`.
- **Dual-GPU CUDA training + spiking module + live dashboards**
  (commit `b7f2cf2`). Training pipeline uses `CUDA_VISIBLE_DEVICES=1` to keep
  GPU 0 free for display (GPU 0 training causes Xid 109). Spiking module
  scaffolding added (`crates/qlang-runtime/src/spiking.rs`) — currently at
  random-chance accuracy, included for iteration. New live dashboards:
  `frontend/src/GpuTrainingView.tsx`, `frontend/src/SpikingView.tsx`.

### Known issues
- Hebbian learning: ~10% on MNIST (random).
- Spiking-MNIST: ~10% (random). STDP loop and rate-coding need fixing.
- Random conv / ViT features on CIFAR-10: 10–12%.
- Mamba training aborted at step 50/10000; tokenizer vocab missing from
  `.qlmb` checkpoints.
- CLI training on full 60K MNIST is too slow; batch parallelization missing.
- Transformer uses random-perturbation updates instead of true backprop.
- Quantum gradient flow in the executor is a simplified gradient step, not
  the full IGQK flow.

## [0.1.0] — 2026-04-02

### Added
- **Core language**: graph-based DAG representation, 36+ operations.
- **Type system**: `Tensor<dtype>[shape]` with shape inference
  (f32, f64, i32, ternary, etc.).
- **Parser**: `.qlang` text syntax with `@proof theorem_*` annotations.
- **Compiler**: LLVM JIT (measured 29.4x over interpreter at 1M elements,
  release mode).
- **SIMD**: AVX2 vectorization with aligned allocator.
- **AOT**: compilation to native `.o` object files.
- **WebAssembly**: WAT codegen for browser deployment.
- **GPU**: WGSL compute shader generation (WebGPU / wgpu).
- **Runtime**: graph executor with 20+ tensor operations.
- **Autograd**: reverse-mode automatic differentiation.
- **Training**: MLP with SGD, Adam, gradient clipping.
- **Transformer**: multi-head attention, LayerNorm, GELU, positional encoding.
- **Conv2D**: 2D convolution, max pooling, causal attention masking.
- **IGQK**: ternary compression (4–16x), quantum-state operations.
- **ONNX**: import/export for PyTorch/TensorFlow interop (MLP subset).
- **Agent protocol**: QLMS binary format for AI-to-AI exchange.
- **Network server**: TCP-based graph exchange and remote execution.
- **Package system**: local registry with a small standard library.
- **Graph diff**: version-control primitives for graphs.
- **Distributed training**: data-parallel with gradient aggregation.
- **CLI**: REPL, parser, optimizer, visualizer, profiler.
- **LSP**: language-server foundation for IDE integration.
- **CI/CD**: GitHub Actions pipeline.
- **Optimizer**: 6 passes (DCE, constant folding, CSE, op fusion,
  identity elimination, and scheduling).
- **Diagnostics**: deep graph validation with error recovery.
- **Scheduler**: wavefront parallelism detection and memory planning.
- **Benchmarks**: performance measurement suite.
- **Checkpoints**: `.qlm` binary model save/load.
- **Visualization**: Graphviz DOT + ASCII terminal output.
- **Self-hosting**: compiler expressed as QLANG graph (foundation).

### Performance (measured 2026-04-02)
- JIT compilation: 29.4x over interpreter at 1M elements.
- Training: 100% accuracy on toy dataset in 70 ms.
- MNIST 784→128→10 MLP: converges in 15.9 s.
- IGQK compression: 4x–16x with accuracy retention on the toy set.
- Binary graph format: 3.2 KB for the complete MNIST model.
