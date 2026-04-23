# HowTo

Schritt-fuer-Schritt Anleitungen fuer die wichtigsten QLANG Workflows.

## Erstes QLANG Programm

### 1. Build

```bash
cd qlang
cargo build --release -p qlang-compile
```

### 2. Hello World

Erstelle `hello.qlang`:

```qlang
print("Hello from QLANG!")

let name = "World"
print("Hello, " + name + "!")

let x = 42
let y = x * 2 + 1
print("x = " + str(x) + ", y = " + str(y))
print("sqrt(16) = " + str(sqrt(16)))
```

### 3. Ausfuehren

```bash
qlang-cli exec hello.qlang
```

Output:
```
Hello from QLANG!
Hello, World!
x = 42, y = 85
sqrt(16) = 4
```

### 4. Fibonacci als Beispiel

```qlang
fn fib(n) {
  if n <= 1 {
    return n
  }
  return fib(n - 1) + fib(n - 2)
}

for i in 0..10 {
  print("fib(" + str(i) + ") = " + str(fib(i)))
}
```

---

## ML Pipeline mit Graph

### 1. Graph definieren

Erstelle `model.qlang`:

```qlang
let lr = 0.01
print("=== QLANG ML Pipeline ===")

graph classifier {
  input x: f32[4]
  input W1: f32[4, 8]
  input W2: f32[8, 2]
  node h = matmul(x, W1)
  node a = relu(h)
  node out = matmul(a, W2)
  node pred = softmax(out)
  output predictions = pred
}

print("Model defined: 4 -> 8 -> 2")
```

### 2. Ausfuehren

```bash
qlang-cli exec model.qlang
```

---

## Swarm von Modellen trainieren

### 1. Quick-Start (eingebaute Daten)

```bash
qlang-cli swarm-train --quick --population 10 --generations 5
```

### 2. Mit eigenen Daten

```bash
# Textdatei vorbereiten
echo "Your training text here..." > data.txt

# Swarm starten
qlang-cli swarm-train --data data.txt --population 20 --generations 10
```

### 3. Output verstehen

```
[1/4] Training tokenizer...
  Vocab: 500 tokens, Text: 1234 tokens

[2/4] Creating 10 models...
  Model 1: d=32 h=2 L=2 (50000 params)
  Model 2: d=48 h=3 L=2 (85000 params)
  ...

[3/4] Evolutionary training...
  Gen 1: best_loss=4.23, worst_loss=6.11
  Gen 2: best_loss=3.87, worst_loss=5.44
  ...

[4/4] Winner: Model 5 (d=64 h=4 L=3)
  Generated text: "The quick brown ..."
```

Siehe [[Swarm]] fuer Details.

---

## Transformer Language Model trainieren

### 1. Daten vorbereiten

```bash
# Jede Textdatei funktioniert
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt
```

### 2. Training starten

```bash
qlang-cli train-lm --data shakespeare.txt \
  --d-model 128 --layers 4 --heads 4 \
  --epochs 10 --lr 0.001 \
  --out-model shakespeare.qgpt \
  --out-tokenizer shakespeare.qbpe
```

### 3. Output

```
Training tokenizer (BPE, vocab=500)...
Model: d=128, heads=4, layers=4 (2.1M params)
Epoch 1/10: loss=5.23
Epoch 2/10: loss=4.11
...
Generated: "To be, or not to be..."
Model saved: shakespeare.qgpt
Tokenizer saved: shakespeare.qbpe
```

Siehe [[Transformer]] fuer Architektur-Details.

---

## Autonomous Mode

### 1. Ollama installieren und starten

```bash
# macOS
brew install ollama
ollama serve

# In einem neuen Terminal: Modell laden
ollama pull llama3
```

### 2. Autonomen Loop starten

```bash
qlang-cli autonomous --task "classify MNIST" --target 95 --iterations 5 --model llama3
```

### 3. Was passiert

1. LLM designed eine Architektur
2. QLANG trainiert das Modell
3. LLM evaluiert die Ergebnisse
4. LLM schlaegt Verbesserungen vor
5. Wiederholung bis Ziel erreicht

Siehe [[Agents]] und [[Ollama]] fuer Details.

---

## KI Pipeline (AI-Train)

### 1. Sicherstellen dass Ollama laeuft

```bash
qlang-cli ollama health
# => Ollama is running
```

### 2. AI-designed Training starten

```bash
qlang-cli ai-train --model llama3 --quick
```

Das LLM designed die komplette Netzwerk-Architektur und QLANG fuehrt das Training aus.

---

## WebUI Dashboard

### 1. Nur Dashboard (Demo-Modus)

```bash
qlang-cli web --port 8081
# Oeffne http://localhost:8081
```

### 2. Live Training mit Dashboard

```bash
qlang-cli train-mnist --port 8081 --epochs 100
# Oeffne http://localhost:8081 zum Zuschauen
```

### 3. Dashboard Features nutzen

- **Live Feed**: Automatisch scrollende Nachrichten
- **Run Code**: QLANG Code im Browser schreiben und ausfuehren
- **Train**: MNIST Training mit konfigurierbaren Parametern starten
- **Compress**: Trainiertes Modell via IGQK komprimieren
- **Autonomous**: AI Feedback-Loop direkt aus dem Dashboard

Siehe [[WebUI]] fuer alle Details.

---

## GPU Support aktivieren

### 1. wgpu (NVIDIA/AMD/Intel)

```bash
cargo build --release --features gpu
```

### 2. Apple MLX (Metal)

```bash
cargo build --release --features mlx
```

### 3. Geraete pruefen

```bash
qlang-cli devices
```

Output:
```
CPU (10 cores) - 10 threads
Apple M2 Pro (Metal, integrated)
```

### 4. Acceleration verifizieren

Der Backend wird automatisch gewaehlt. Alle `matmul` Operationen nutzen die schnellste verfuegbare Hardware.

Siehe [[GPU]] fuer alle Backends.

---

## Binaerformat verwenden

### 1. Graph zu Binary encodieren

```bash
qlang-cli binary encode model.qlg.json
# => model.qlb (3.5x kleiner)
```

### 2. Binary zu JSON dekodieren

```bash
qlang-cli binary decode model.qlb
# => JSON auf stdout
```

### 3. Cache nutzen

```bash
# Statistiken anzeigen
qlang-cli cache stats

# Cache leeren
qlang-cli cache clear
```

Siehe [[BinaryFormat]] und [[Crypto]] fuer Details.

#howto #tutorial #getting-started
