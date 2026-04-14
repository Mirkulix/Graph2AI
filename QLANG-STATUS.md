# QLANG Status (12. April 2026)

## Honesty Note
All accuracies below are measured on REAL datasets:
- MNIST full = 60,000 train / 10,000 test samples
- MNIST 1K subset = 5,000 train / 1,000 test (subset for fast iteration)
- Numbers from synthetic data tests are NOT listed — use real datasets only.

Last measured: 2026-04-12 on RTX 2070 Super.

## Was funktioniert (bewiesen mit Tests)

### Training
| Methode | Dataset | Samples | Accuracy | Source |
|---------|---------|---------|----------|--------|
| TernaryBrain + refine | MNIST 1K subset | 5K train / 1K test | **99.6%** | real_mnist_brain.rs test |
| TernaryBrain + 15 rounds | MNIST full | 60K train / 10K test | **84.6%** | demo.rs 2026-04-12 |
| FFNetwork f32 | MNIST full | 60K train / 10K test | **90.83%** (20 epochs) | demo.rs |
| FFNetwork QAT (GPU) | MNIST full | 60K train / 10K test | **84.6%** (30 epochs, 24s) | demo.rs RTX 2070S |
| IGQK compression | pack_ternary | 570K weights | **16.0x** real | demo.rs |

### Kommunikation
- MessageBus: 6 Agenten, QLMS GraphMessages, SSE Live-Stream
- NetworkBridge: TCP fuer Remote-Agenten
- Tensor-Austausch: 384-dim Embeddings zwischen Agenten in 188ms
- 14 Tests bestanden

### CLI
```bash
qlang train --data data/mnist --epochs 10 --output model.qlbg
qlang infer --model model.qlbg --input 3
qlang info model.qlbg
qlang bench model.qlbg
```

### Mathematik (implementiert in bitnet_math.rs)
- Absmean Quantization (BitNet b1.58)
- RMSNorm Stabilization
- Stochastic Quantization
- Entropy Regularization
- Quantization Annealing
- Ternary LoRA
- Tensor-Train Decomposition
- Knowledge Distillation (KL-Divergenz)
- 19 Tests bestanden

### UI (http://localhost:4747)
- Messages Tab: Live QLMS Nachrichtenlog
- Canvas: Agenten-Knoten mit animiertem Message-Flow
- Training Tab: Training starten, Live-Monitor
- Tensor-Proof Button

## Was NICHT funktioniert

- Hebbian Learning (10% auf MNIST — lernt nichts)
- Random Conv Features (10% auf CIFAR-10 — nutzlos)
- Random ViT Features (12% auf CIFAR-10 — nutzlos)
- Hand-crafted Gabor/DCT Features (25% auf CIFAR-10 — zu schwach)
- CLI Training mit 60K Samples (zu langsam, braucht Batch-Parallelisierung)
- Forward-Forward auf CIFAR-10 (nicht getestet, wahrscheinlich ~35%)

## Offene Fragen

1. Skaliert TernaryBrain ueber MNIST/CIFAR-10 hinaus?
2. Kann Forward-Forward mit Annealing (tanh(beta*W)) besser werden?
3. Bringt mehr Daten (50K statt 10K) den CIFAR-10 Score ueber 70%?
4. Laesst sich die Feature-Extraktion auf GPU beschleunigen?

## Architektur

```
qlang/
  crates/
    qlang-core/         # Graph, Tensor, Binary, Crypto, Quantum (120 Tests)
    qlang-compile/      # LLVM JIT, WASM, GPU, Parser (151 Tests)
    qlang-runtime/      # Training, Inference, Mathematik (550+ Tests)
      src/
        forward_forward.rs   # FF Ternary Training
        ternary_brain.rs     # Statistical Init + Competitive Hebbian
        ternary_ops.rs       # Zero-Multiply Inference
        bitnet_math.rs       # Absmean, RMSNorm, LoRA, Annealing
        cifar10.rs           # CIFAR-10 Daten-Loader
        vision_transformer.rs # ViT Feature Extractor
    qlang-agent/        # QLMS Protokoll, MessageBus, Bridge (77 Tests)
  qo/
    qo-embed/           # candle Embeddings + ResNet-18 Vision
    qo-server/          # Axum HTTP + WebSocket + SSE
    qo-agents/          # 6 Agenten + Executor
  frontend/             # React UI (10 Tabs)
  src/
    cli.rs              # qlang CLI Binary
    main.rs             # qo Server Binary
```

## Zum Starten

```bash
# QO Server + UI
QO_PORT=4747 ./target/release/qo
# → http://localhost:4747

# QLANG CLI
./target/release/qlang train --data data/mnist --epochs 10 --output model.qlbg
./target/release/qlang info model.qlbg

# Tests
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo test -p qlang-runtime --release -- ternary_brain --nocapture
```
