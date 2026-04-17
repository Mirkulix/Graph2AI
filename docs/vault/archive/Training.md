# Training

QLANG implementiert das gesamte ML-Training in reinem Rust -- ohne PyTorch, ohne TensorFlow, ohne externe ML-Frameworks.

## Trainingsarten

### 1. MLP Training (Backpropagation)

Klassisches Multilayer-Perceptron Training in `training.rs`:

- Forward Pass mit matmul + Aktivierungsfunktionen
- Cross-Entropy Loss mit Softmax
- Reverse-Mode Autograd (automatische Differenzierung)
- Hardware-beschleunigt via Apple Accelerate BLAS / [[GPU]]

```bash
qlang-cli train-mnist --epochs 100 --port 8081
```

**Optimierer:**

| Optimierer | Features |
|-----------|----------|
| SGD | Lernrate, Momentum |
| Adam | Beta1/Beta2, Epsilon, Weight Decay |
| Gradient Clipping | Max-Norm Clipping |

**LR Schedules:**
- Constant
- Step Decay
- Cosine Annealing
- Linear Warmup

### 2. Transformer Training (MiniGPT)

Decoder-only Transformer Language Model. Siehe [[Transformer]] fuer Details.

```bash
qlang-cli train-lm --data text.txt --d-model 128 --layers 4 --epochs 10
```

- BPE Tokenizer (trainiert auf den Daten)
- Token + Positional Embeddings
- Multi-Head Causal Self-Attention
- Pre-Norm Architektur (RMSNorm oder LayerNorm)
- SiLU oder GELU Aktivierung
- Random-Perturbation Gradient Estimation
- Autoregressive Text-Generierung

### 3. Swarm Training (Neuroevolution)

Evolutionaere Architektursuche -- viele Modelle konkurrieren. Siehe [[Swarm]] fuer Details.

```bash
qlang-cli swarm-train --quick --population 10 --generations 5
```

- Population von Modellen mit verschiedenen Architekturen
- Fitness-basierte Selektion
- Mutation (Gewichte perturbieren)
- Crossover (beste Modelle kombinieren)
- Automatische Architekturwahl (d_model, n_heads, n_layers)

### 4. Autonome Feedback-Loop

Ein LLM (via [[Ollama]]) entscheidet iterativ, wie ein Modell verbessert werden soll:

```bash
qlang-cli autonomous --task "classify MNIST" --target 95 --iterations 5
```

1. LLM designed initiale Architektur
2. QLANG trainiert das Modell
3. LLM evaluiert Ergebnisse und schlaegt Verbesserungen vor
4. Wiederholung bis Ziel erreicht oder Iterationen aufgebraucht

### 5. Hebbian Learning

Gradient-freies, bio-inspiriertes Lernen fuer ternaere Gewichte. Siehe [[ParaDiffuse]].

- "Neurons that fire together wire together"
- Kein Gradient noetig -- O(1) pro Gewicht pro Sample
- Produziert natuerlich ternaere Outputs {-1, 0, +1}
- Kompatibel mit [[IGQK]] Quantum Measurement

```rust
use qlang_runtime::hebbian::{HebbianState, hebbian_train_step};

let mut state = HebbianState::new(784, 128);
let mut weights = random_ternary_weights(784 * 128, 42);

for sample in data {
    hebbian_train_step(&mut weights, &sample, &mut state);
}
```

### 6. IGQK Compression als Training

Quantum Gradient Flow trainiert und komprimiert gleichzeitig. Siehe [[IGQK]].

```
d_rho/dt = -i[H, rho] - gamma * {G^-1 * grad_L, rho}
```

- Unitary Evolution (Quantum Exploration) + Dissipative Evolution (Gradient Descent)
- Measurement Collapse zu ternaeren Gewichten
- 16x Kompression bei 100% Accuracy-Erhalt

## Distributed Training

Multi-GPU/Multi-Worker Training mit Gradient Averaging. Siehe [[GPU]].

```rust
let config = DistributedConfig {
    n_workers: 4,
    strategy: ParallelStrategy::DataParallel,
    batch_size_per_worker: 32,
    gradient_accumulation_steps: 1,
};
```

| Strategie | Beschreibung |
|-----------|-------------|
| Data Parallel | Jeder Worker trainiert auf anderen Daten, Gradienten werden gemittelt |
| Model Parallel | Modell wird ueber Worker aufgeteilt (fuer zu grosse Modelle) |
| Pipeline Parallel | Worker bilden eine Pipeline, jeder verarbeitet eine Stufe |

## Checkpoint-System

Modelle werden im `.qlm` Binaerformat gespeichert:

```rust
// Speichern
model.save("model.qlm")?;

// Laden
let model = Model::load("model.qlm")?;
```

Transformer-Modelle verwenden `.qgpt`, Tokenizer `.qbpe`.

## Model Registry

```bash
# Modell registrieren
qlang-cli registry save --name my_model --version 1.0

# Modelle auflisten
qlang-cli registry list

# Modelle vergleichen
qlang-cli registry compare v1 v2
```

## Curriculum

Typischer Trainingsablauf:

1. **Daten vorbereiten** -- MNIST laden oder eigene Textdaten
2. **Architektur waehlen** -- MLP, Transformer, oder via Swarm
3. **Training starten** -- via CLI mit Live-Dashboard
4. **Evaluieren** -- Loss/Accuracy im [[WebUI]] beobachten
5. **Komprimieren** -- Via [[IGQK]] zu ternaeren Gewichten
6. **Deployen** -- Binaeformat, WASM, oder nativer Code

Siehe [[HowTo]] fuer Schritt-fuer-Schritt-Anleitungen.

#training #ml #backpropagation #autograd
