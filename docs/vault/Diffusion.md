# Diffusion

QLANG enthaelt eine Diffusion Engine, portiert von ParaDiffuse (Python/PyTorch) nach Rust. Siehe auch [[ParaDiffuse]].

## Konzept

Diffusion behandelt Generierung als iteratives Denoising:

1. Starte mit reinem Rauschen
2. In N Schritten schrittweise Rauschen entfernen via gelernten Denoiser
3. Ende mit kohaerenten Embeddings/Daten

## DiffusionSchedule

Die Noise-Schedule bestimmt, wie viel Rauschen bei jedem Zeitschritt hinzugefuegt wird.

### Forward Diffusion

```
q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

Wobei `alpha_bar_t` das kumulative Produkt von `alpha_t = 1 - beta_t` ist.

### Cosine Schedule (Nichol & Dhariwal, 2021)

```rust
let schedule = DiffusionSchedule::cosine(1000);
```

- Besser als Linear fuer Text/Embeddings
- Haelt hohes Signal-to-Noise Ratio laenger aufrecht
- Berechnet via: `alpha_bar_t = cos((t/T + s) / (1+s) * pi/2)^2`
- `s = 0.008` (kleiner Offset gegen Singularitaeten)

### Linear Schedule

```rust
let schedule = DiffusionSchedule::linear(1000, 0.0001, 0.02);
```

- Linearer Anstieg von `beta_start` zu `beta_end`
- Einfacher, aber weniger gut fuer Text

### Operationen

```rust
// Forward: Rauschen hinzufuegen
let noisy = schedule.add_noise(&x0, &noise, timestep);

// Predict: x_0 aus noisy x_t und predicted noise schaetzen
let x0_pred = schedule.predict_x0(&xt, &predicted_noise, timestep);
```

## DDIM Sampler

Deterministic Denoising Diffusion Implicit Models (Song, Meng & Ermon, 2020).

Im Gegensatz zu DDPM, das ~1000 Steps braucht, kann DDIM in nur 10-20 Steps hochqualitative Outputs generieren.

```rust
let schedule = DiffusionSchedule::cosine(1000);
let sampler = DdimSampler::new(schedule, 10); // 10 Sampling Steps

let result = sampler.sample(embedding_dim, |noisy_input, timestep| {
    // Denoiser: vorhersage des Rauschens
    denoiser_model.predict_noise(noisy_input, timestep)
});
```

### DDIM Update-Regel

```
x_0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
x_{t-1} = sqrt(alpha_prev) * x_0_pred + sqrt(1-alpha_prev) * eps
```

### Features

- **Deterministisch**: Gleicher Seed = gleiche Ausgabe
- **Schnell**: 10-20 Steps statt 1000
- **Gleichmaessig verteilt**: Timestep-Indizes werden ueber die Schedule verteilt
- **Denoiser als Closure**: Jeder Denoiser-Typ kann verwendet werden

## Random Number Generation

Eigene Implementierung, keine externen Crates:

- **Xorshift64 PRNG**: Schneller Zufallszahlengenerator
- **Box-Muller Transform**: Erzeugung normalverteilter Zufallszahlen

```rust
// Normalverteilte Zufallszahlen
let noise = random_normal(1024);

// Reproduzierbar mit Seed
let noise = random_normal_seeded(1024, Some(42));
```

## Zusammenspiel mit QLANG

Die Diffusion Engine arbeitet mit anderen QLANG-Komponenten zusammen:

- **[[Transformer]]**: Denoiser kann ein Transformer sein
- **Hebbian Learning** ([[ParaDiffuse]]): Gradient-freier Denoiser fuer ternaere Gewichte
- **[[IGQK]]**: Quantum Scheduler fuer optimale Noise-Schedules
- **[[BinaryFormat]]**: Modelle im kompakten Binaerformat speichern

## Tests

27 Tests decken ab:
- Schedule-Erstellung (Cosine, Linear)
- Alpha-Bars absteigend, Betas in validem Bereich
- Forward Diffusion Roundtrip
- DDIM Sampling mit verschiedenen Denoisern
- Zufallszahlengenerierung (Statistik, Determinismus)
- Edge Cases (einzelner Timestep)

#diffusion #generative #ddim #paradiffuse
