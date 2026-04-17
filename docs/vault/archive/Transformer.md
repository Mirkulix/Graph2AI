# Transformer

QLANG implementiert ein GPT-style Decoder-only Transformer Language Model ("MiniGPT") vollstaendig in Rust.

## MiniGPT Architektur

```
Token IDs ──► Token Embedding ──► + Positional Embedding
                                          │
                                          ▼
                              ┌── Transformer Block ──┐
                              │  Pre-Norm (RMS/Layer)  │
                              │  Multi-Head Attention   │  x N layers
                              │  + Residual             │
                              │  Pre-Norm (RMS/Layer)  │
                              │  FFN (Up → Act → Down)  │
                              │  + Residual             │
                              └────────────────────────┘
                                          │
                                          ▼
                              Final Norm ──► LM Head ──► Logits
```

## TransformerConfig

```rust
pub struct TransformerConfig {
    pub vocab_size: usize,    // z.B. 500 (aus BPE Tokenizer)
    pub d_model: usize,       // z.B. 64, 128, 256
    pub n_heads: usize,       // z.B. 4, 8
    pub n_layers: usize,      // z.B. 2, 4, 6
    pub max_seq_len: usize,   // z.B. 64, 128, 256
    pub dropout: f32,         // z.B. 0.0 (reserviert)
    pub use_rms_norm: bool,   // true = RMSNorm (15% schneller)
    pub use_silu: bool,       // true = SiLU statt GELU
}
```

## Komponenten

### RMSNorm

Root Mean Square Layer Normalization -- ~15% schneller als LayerNorm, weil kein Mean-Centering:

```
RMSNorm(x) = x / RMS(x) * gamma
RMS(x) = sqrt(mean(x^2) + eps)
```

Verwendet von LLaMA, Mistral und anderen modernen Architekturen.

### SiLU (Sigmoid Linear Unit)

Auch bekannt als Swish-Aktivierung:

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Alternative zu GELU, verwendet in LLaMA und Mistral fuer potenziell schnellere Konvergenz.

### Multi-Head Causal Self-Attention

- QKV-Projektion: `[seq_len, d_model] x [d_model, 3*d_model]`
- Getrennte Q, K, V Matrizen pro Head
- Skalierte Dot-Product Attention mit kausaler Maske (Zukunft wird auf -inf gesetzt)
- Softmax ueber die Key-Dimension
- Output-Projektion zurueck auf `[seq_len, d_model]`

### Feed-Forward Network

```
FFN(x) = Down(Act(Up(x)))
  Up:   [d_model, 4*d_model]
  Act:  SiLU oder GELU
  Down: [4*d_model, d_model]
```

### Positional Encoding

Gelernte Positional Embeddings (nicht sinusoidal):
- Embedding-Tabelle: `[max_seq_len, d_model]`
- Addiert zum Token Embedding

## BPE Tokenizer

Byte-Pair Encoding Tokenizer in reinem Rust (`tokenizer.rs`):

1. Initialisiere Vokabular mit 256 Byte-Token
2. Finde wiederholt das haeufigste Paar und merge es
3. Stoppe bei gewuenschter `vocab_size`
4. 3 Spezial-Token: `<BOS>`, `<EOS>`, `<PAD>`

```rust
let tokenizer = BpeTokenizer::train("training text...", 500);
let tokens = tokenizer.encode("Hello World");
let text = tokenizer.decode(&tokens);

// Save/Load im QBPE Binaerformat
tokenizer.save("tokenizer.qbpe")?;
let loaded = BpeTokenizer::load("tokenizer.qbpe")?;
```

### QBPE Format

```
[4 bytes] Magic: "QBPE"
[4 bytes] Merge Count (u32 LE)
[var]     Merges: je (u32, u32)
[4 bytes] Vocab Size (u32 LE)
[var]     Vocab: je (u32 length + bytes)
[12 bytes] bos_id, eos_id, pad_id (je u32 LE)
```

## Training

Training verwendet Random-Perturbation Gradient Estimation (kein vollstaendiger Backprop fuer Transformer):

1. Berechne Base-Loss
2. Perturbiere zufaellige Parameter
3. Messe Loss-Aenderung
4. Schaetze Gradientenrichtung
5. Update Parameters

```bash
qlang-cli train-lm --data wiki.txt \
  --d-model 128 --layers 4 --heads 4 \
  --epochs 10 --lr 0.001 \
  --out-model model.qgpt --out-tokenizer tokenizer.qbpe
```

## Text-Generierung

Autoregressive Generierung mit Temperature Sampling:

```rust
let model = MiniGPT::new(config);
// ... training ...
let text = model.generate(&tokenizer, "The", 100, 0.8);
```

1. Encode Prompt zu Token IDs
2. Forward Pass durch alle Layer
3. Logits des letzten Tokens nehmen
4. Temperature anwenden + Sampling
5. Neuen Token anhaengen
6. Wiederholen bis max_length

## Modell speichern/laden

```rust
// Save (QGPT Binaerformat)
model.save("model.qgpt")?;

// Load
let model = MiniGPT::load("model.qgpt", &config)?;
```

## Typische Konfigurationen

| Name | d_model | heads | layers | Parameter |
|------|---------|-------|--------|-----------|
| Tiny | 32 | 2 | 2 | ~50K |
| Small | 64 | 4 | 2 | ~200K |
| Medium | 128 | 4 | 4 | ~2M |
| Large | 256 | 8 | 6 | ~15M |

Alle Matmul-Operationen nutzen `accel::matmul` fuer Hardware-Beschleunigung (Apple Accelerate, [[GPU]]).

Siehe [[Swarm]] fuer evolutionaere Architektursuche, [[Training]] fuer den Gesamt-Ueberblick.

#transformer #minigpt #nlp #language-model
