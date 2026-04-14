# CLI

Die `qlang-cli` Binary (aus dem `qlang-compile` Crate) bietet alle Kommandozeilen-Tools. Version 0.3.

## Installation

```bash
cargo build --release -p qlang-compile
# Binary unter: target/release/qlang-cli
```

## Befehle

### Interaktiv

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli repl` | Interaktive REPL fuer QLANG-Ausdruecke |
| `qlang-cli exec <file.qlang>` | QLANG-Programm ausfuehren (VM-Modus, Unified: Scripting + Graphen) |

### Graph-Operationen

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli info <file.qlg.json>` | Graph-Struktur anzeigen (Knoten, Kanten, Typen) |
| `qlang-cli verify <file.qlg.json>` | Alle Constraints und Proofs pruefen |
| `qlang-cli optimize <file.qlg.json> -o <out.json>` | 6 Optimierungspaesse ausfuehren |
| `qlang-cli stats <file.qlg.json>` | Graph-Statistiken (Tiefe, Breite, etc.) |
| `qlang-cli schedule <file.qlg.json>` | Execution-Plan mit Parallelismus anzeigen |
| `qlang-cli parse <file.qlang>` | `.qlang` Text zu Graph parsen |

### Ausfuehrung

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli run <file.qlg.json>` | Via Interpreter ausfuehren |
| `qlang-cli jit <file.qlg.json>` | Via LLVM JIT ausfuehren (29x schneller) |

Siehe [[Execution]] fuer das 3-Tier System (LLVM JIT, Bytecode VM, Interpreter).

### Kompilierung

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli compile <file.qlg.json> -o <out.o>` | AOT-Kompilierung zu Object File |
| `qlang-cli asm <file.qlg.json>` | Native Assembly anzeigen |
| `qlang-cli llvm-ir <file.qlg.json>` | LLVM IR Output anzeigen |
| `qlang-cli wasm <file.qlg.json>` | WebAssembly (WAT) generieren |
| `qlang-cli gpu <file.qlg.json>` | WGSL Compute Shader generieren |

### Visualisierung

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli dot <file.qlg.json>` | Graphviz DOT Format ausgeben |
| `qlang-cli ascii <file.qlg.json>` | ASCII Terminal-Visualisierung |

### Binaerformat

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli binary encode <file.qlg.json>` | Graph zu binaerer `.qlb` Datei enkodieren |
| `qlang-cli binary decode <file.qlb>` | Binaere `.qlb` zu JSON dekodieren |

Siehe [[BinaryFormat]] fuer Details zum QLBG-Format.

### AI / ML Training

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli train-mnist [--port 8081] [--epochs 50]` | MNIST mit live [[WebUI]] Dashboard trainieren |
| `qlang-cli train-lm --data <txt> [Optionen]` | [[Transformer]] Language Model trainieren (MiniGPT) |
| `qlang-cli swarm-train [--data <txt>] [--population 10] [--generations 5] [--quick]` | [[Swarm]]: Evolutionaere Architektursuche |
| `qlang-cli ai-train [--model M] [--quick]` | AI-designed Training Pipeline (via [[Ollama]]) |
| `qlang-cli autonomous [--task T] [--target 95] [--iterations 5]` | Autonome AI Feedback-Loop |
| `qlang-cli devices` | Verfuegbare Compute-Geraete anzeigen (CPU, GPU, MLX) |

#### train-lm Optionen

```
--data <text_file>    Trainingsdaten (Pflicht)
--vocab-size 1000     BPE Vokabulargroesse (default: 500)
--d-model 128         Embedding-Dimension (default: 64)
--layers 4            Transformer-Layer (default: 2)
--heads 4             Attention-Heads (default: 4)
--epochs 10           Training-Epochen (default: 5)
--seq-len 128         Max. Sequenzlaenge (default: 64)
--lr 0.001            Lernrate (default: 0.001)
--out-model M         Modell speichern als (default: model.qgpt)
--out-tokenizer T     Tokenizer speichern als (default: tokenizer.qbpe)
```

#### swarm-train Optionen

```
--data <text_file>    Trainingsdaten (oder --quick fuer eingebaute Daten)
--population 10       Anzahl Modelle in der Population
--generations 5       Anzahl Generationen
--quick               Eingebauter Sample-Text statt Datei
```

### Ollama / LLM

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli ollama health` | Ollama Server Status pruefen |
| `qlang-cli ollama models` | Verfuegbare Modelle auflisten |
| `qlang-cli ollama generate` | Text generieren |
| `qlang-cli ollama chat` | Chat Completion |

Siehe [[Ollama]] fuer Details.

### Server

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli web [--port 8081]` | [[WebUI]] Dashboard-Server starten |
| `qlang-cli proxy [--port 9100] [--upstream URL]` | HTTP-zu-QLMS Signing Proxy |
| `qlang-cli lsp` | Language Server Protocol Server starten (stdin/stdout) |
| `qlang-cli ide [--port 3000]` | QLANG IDE starten (Eclipse Theia) |

### Cache

| Befehl | Beschreibung |
|--------|-------------|
| `qlang-cli cache stats` | Cache Hit/Miss Statistiken anzeigen |
| `qlang-cli cache clear` | Computation Cache leeren |

Siehe [[Crypto]] fuer die Content-Addressable Cache Architektur.

## Beispiele

```bash
# Ein .qlang Programm ausfuehren
qlang-cli exec hello.qlang

# Parse und validiere eine .qlang Datei
qlang-cli parse model.qlang

# Optimieren und kompilieren
qlang-cli optimize model.qlg.json -o optimized.json
qlang-cli compile optimized.json -o model.o

# WASM fuer Browser generieren
qlang-cli wasm model.qlg.json > model.wat

# MNIST trainieren mit Live Dashboard
qlang-cli train-mnist --epochs 100 --port 8081
# Oeffne http://localhost:8081 zum Zuschauen

# Transformer Language Model trainieren
qlang-cli train-lm --data wiki.txt --d-model 128 --layers 4 --epochs 10

# Swarm: Evolution findet die beste Architektur
qlang-cli swarm-train --quick --population 10 --generations 5

# Autonome AI Feedback-Loop
qlang-cli autonomous --task "classify MNIST" --target 95 --iterations 5

# Compute-Geraete erkennen
qlang-cli devices
```

#cli #tools
