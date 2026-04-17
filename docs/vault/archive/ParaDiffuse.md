# ParaDiffuse

ParaDiffuse ist ein Python/PyTorch-Projekt fuer parallele Diffusion-basierte Generierung. Zentrale Konzepte wurden nach Rust portiert und in QLANG integriert.

## Was wurde portiert

### 1. Diffusion Engine (`diffusion.rs`)

Die komplette Diffusion Pipeline:

- **DiffusionSchedule**: Noise-Schedules (Cosine, Linear)
- **Forward Diffusion**: Rauschen zu sauberen Daten hinzufuegen
- **DDIM Sampler**: Deterministische schnelle Generierung (10-20 Steps)
- **Random Normal**: Box-Muller Transform mit Xorshift64 PRNG

Siehe [[Diffusion]] fuer technische Details.

### 2. Hebbian Learning (`hebbian.rs`)

Bio-inspiriertes, gradient-freies Lernen fuer ternaere Gewichte:

- **HebbianState**: Verfolgt Korrelationen zwischen Pre- und Post-Aktivierungen
- **Salience Signal**: Zentrierte Vorzeichen-Korrelation
- **Ternary Weight Updates**: Gewichte werden basierend auf Salience zu {-1, 0, +1} geflippt
- **Decay**: Salience zerfaellt nach jedem Update

```rust
use qlang_runtime::hebbian::{HebbianState, hebbian_train_step, random_ternary_weights};

// Initialisiere ternaere Gewichte
let mut weights = random_ternary_weights(in_dim * out_dim, 42);
let mut state = HebbianState::new(in_dim, out_dim);

// Training: kein Gradient noetig!
let output = hebbian_train_step(&mut weights, &input, &mut state);
```

### 3. Quantum Scheduler Integration

ParaDiffuse's Idee eines "Quantum Schedulers" wird durch [[IGQK]] realisiert:

- Noise Schedule als Quantum Evolution
- Diffusion Steps als Measurement Operators
- Optimale Schedule via Fisher Information

## Wie Hebbian Learning funktioniert

### Prinzip: "Neurons that fire together wire together"

1. **Forward Pass**: `output = weights * input` (ternaere Matrix-Multiplikation)
2. **Running Means**: Exponential Moving Average von Pre- und Post-Aktivierungen
3. **Salience Update**: Zentrierte Vorzeichen-Korrelation akkumulieren

```
delta_salience[i,j] = sign(post[i] - post_mean[i]) * sign(pre[j] - pre_mean[j])
```

4. **Weight Update**: Wenn Salience den Threshold ueberschreitet, Gewicht flippen

```
if salience > threshold:  weight = +1
if salience < -threshold: weight = -1
otherwise:                weight unchanged
```

5. **Salience Decay**: Nach jedem Apply wird die Salience mit `salience_decay` (z.B. 0.95) multipliziert

### Konfigurierbare Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `threshold` | 0.1 | Schwellwert fuer Weight-Flip |
| `momentum` | 0.9 | EMA-Momentum fuer Running Means |
| `salience_decay` | 0.95 | Zerfallsrate der Salience nach Apply |

### Vorteile gegenueber Backpropagation

| Eigenschaft | Hebbian | Backprop |
|-------------|---------|----------|
| Gradient noetig | Nein | Ja |
| Aufwand pro Gewicht | O(1) | O(n) |
| Ternaere Outputs | Natuerlich | Quantisierung noetig |
| Kompatibel mit IGQK | Ja | Teilweise |

## Was QLANG anders macht

Im Vergleich zum originalen ParaDiffuse:

| Aspekt | ParaDiffuse (Python) | QLANG (Rust) |
|--------|---------------------|-------------|
| Sprache | Python + PyTorch | Reines Rust |
| Abhaengigkeiten | PyTorch, NumPy | Null (core) |
| Performance | GPU via CUDA | CPU BLAS + wgpu |
| Diffusion | DDPM + DDIM | DDIM (deterministisch) |
| Hebbian | PyTorch Tensors | Flache Vektoren |
| Integration | Standalone | Teil des QLANG Ecosystems |
| Ternaere Gewichte | Post-Training | Natuerlich (Hebbian) |

## Zusammenspiel im QLANG Stack

```
Diffusion Engine ──► Erzeugt Embeddings/Gewichte
         │
         ▼
Hebbian Learning ──► Verfeinert ternaere Gewichte
         │
         ▼
IGQK Compression ──► Formal verifizierte Kompression
         │
         ▼
Binary Format ──── ► Kompaktes Deployment
```

## Tests

- **Diffusion**: 27 Tests (Schedules, Sampling, Roundtrip, Edge Cases)
- **Hebbian**: 20 Tests (State, Update, Convergence, Weight Flips, Determinismus)

Siehe [[Diffusion]] fuer die Diffusion Engine, [[Training]] fuer den Gesamt-Ueberblick, [[IGQK]] fuer die theoretische Grundlage.

#paradiffuse #hebbian #diffusion #bio-inspired
