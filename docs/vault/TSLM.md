# TSLM

TSLM (Transformer-based Small Language Model) ist ein verwandtes Projekt, das einige architektonische Entscheidungen fuer QLANG's [[Transformer]]-Implementierung beeinflusst hat.

## Was QLANG von TSLM uebernommen hat

### RMSNorm

Root Mean Square Layer Normalization -- ~15% schneller als Standard LayerNorm:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

- Kein Mean-Centering (nur Scale, kein Shift)
- Weniger Berechnungen pro Forward Pass
- Standard in modernen Architekturen (LLaMA, Mistral)
- In QLANG konfigurierbar via `use_rms_norm: bool` in `TransformerConfig`

### SiLU (Swish) Aktivierung

```
SiLU(x) = x * sigmoid(x)
```

- Smoother als GELU
- Bessere Gradienten-Eigenschaften
- Verwendet in LLaMA, Mistral
- In QLANG konfigurierbar via `use_silu: bool` in `TransformerConfig`

### Pre-Norm Architektur

Layer Normalization VOR der Attention/FFN, nicht danach:

```
x → Norm → Attention → + Residual → Norm → FFN → + Residual
```

Statt Post-Norm:
```
x → Attention → + Residual → Norm → FFN → + Residual → Norm
```

Pre-Norm ist stabiler beim Training (besonders fuer tiefe Modelle).

## Was QLANG anders macht

| Aspekt | TSLM | QLANG |
|--------|------|-------|
| Sprache | Python/PyTorch | Rust (kein PyTorch) |
| Training | Standard Backprop | Random Perturbation + Swarm |
| Tokenizer | SentencePiece | Eigener BPE in Rust |
| Deployment | Python Runtime | Native Binary / WASM |
| Kompression | Post-Training Quantisierung | [[IGQK]] (waehrend Training) |
| Architektursuche | Manuell | [[Swarm]] (evolutionaer) |
| Graph-System | Keines | Graph-first Design |
| Diffusion | Nein | [[Diffusion]] Engine integriert |
| Hebbian | Nein | [[ParaDiffuse]] Hebbian Learning |
| Binary Protocol | Nein | [[BinaryFormat]] (3.5x kleiner als JSON) |

## Konzeptuelle Verbindung

QLANG's Vision geht ueber TSLM hinaus:

1. **TSLM**: Ein einzelnes kleines Modell trainieren und deployen
2. **QLANG**: Viele kleine Modelle als **Schwarm von Gehirnen** -- via [[Swarm]] optimiert, via [[Protocol]] kommunizierend, via [[IGQK]] komprimiert

TSLM war ein Baustein -- QLANG baut das Oekosystem drumherum:

```
TSLM (ein Modell) ──────────────────── QLANG (Oekosystem)
                                           │
        ┌──────────────────────────────────┤
        │                                  │
   Transformer Engine          +  Swarm Training
   (RMSNorm, SiLU)            +  Graph Protocol
                               +  Binary Format
                               +  IGQK Compression
                               +  Diffusion Engine
                               +  Multi-GPU Training
                               +  WebUI Dashboard
                               +  AI Agent System
```

## Warum Rust statt Python?

Siehe [[Decisions]] fuer die vollstaendige Begruendung. Kurzfassung:

- Single Binary Deployment (kein Python-Runtime)
- Memory Safety ohne Garbage Collection
- Cross-Compilation zu WASM, ARM, x86
- LLVM JIT Integration fuer native Geschwindigkeit
- Zero-Copy Tensor Transport im [[Protocol]]

Siehe [[Transformer]] fuer technische Details, [[Comparison]] fuer ausfuehrliche Vergleiche.

#tslm #transformer #architecture #comparison
