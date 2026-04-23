# Swarm

Evolutionaere Architektursuche fuer Language Models. Viele Modelle konkurrieren, die besten ueberleben und mutieren.

## Konzept

Anstatt eine Architektur von Hand zu waehlen, laesst der Swarm Trainer die Evolution die beste finden:

1. **Population**: N Modelle mit verschiedenen Architekturen
2. **Training**: Jedes Modell wird fuer einige Steps trainiert
3. **Fitness**: Loss bestimmt die Fitness
4. **Selektion**: Die besten ueberleben
5. **Mutation**: Gewichte werden perturbiert
6. **Naechste Generation**: Wiederholen

## CLI

```bash
# Quick-Modus mit eingebauten Sample-Daten
qlang-cli swarm-train --quick --population 10 --generations 5

# Mit eigenen Daten
qlang-cli swarm-train --data wiki.txt --population 20 --generations 10
```

## Architektur-Diversitaet

Die initiale Population wird mit 10 verschiedenen Architekturen initialisiert:

| Architektur | d_model | heads | layers | Profil |
|-------------|---------|-------|--------|--------|
| Tiny | 32 | 2 | 2 | Minimal |
| Small-Wide | 48 | 3 | 2 | Breit, flach |
| Small-Deep | 32 | 2 | 4 | Schmal, tief |
| Medium-Wide | 64 | 4 | 2 | Standard breit |
| Medium | 64 | 4 | 3 | Balanced |
| Medium-Deep | 48 | 3 | 4 | Medium tief |
| Large-Wide | 96 | 4 | 2 | Gross breit |
| Large-Balanced | 64 | 4 | 4 | Gross balanced |
| Very-Deep | 32 | 2 | 6 | Sehr tief |
| Widest | 128 | 4 | 2 | Breiteste |

## Ablauf

```
[1/4] Tokenizer trainieren
  ► Gemeinsamer BPE Tokenizer fuer alle Modelle
  ► Vocab: ~500 Tokens

[2/4] Population erstellen
  ► N Modelle mit diversen Architekturen
  ► Parameter-Count pro Modell ausgeben

[3/4] Evolutionary Training
  ► Fuer jede Generation:
    1. Jedes Modell trainiert fuer einige Steps
    2. Fitness (Loss) messen
    3. Ranking nach Fitness
    4. Beste Modelle ueberleben (Elitismus)
    5. Mutation: Gewichte der Ueberlebenden perturbieren
    6. Neue Modelle fuer freie Slots erzeugen

[4/4] Bestes Modell auswaehlen
  ► Text-Generierung mit dem Gewinner
  ► Modell + Tokenizer speichern
```

## Mutation

Gewichte werden perturbiert, um neue Varianten zu erzeugen:

```
gewicht_neu = gewicht_alt + epsilon * random_normal
```

Wobei `epsilon` die Mutationsrate ist (typisch 0.01-0.1).

## Fitness

Die Fitness basiert auf dem Loss (Cross-Entropy fuer Next-Token-Prediction):

```
fitness = -loss
```

Niedrigerer Loss = hoehere Fitness = besser.

## Zusammenspiel

- **[[Transformer]]**: Jedes Swarm-Mitglied ist ein MiniGPT
- **BPE Tokenizer**: Ein gemeinsamer Tokenizer wird einmal trainiert
- **[[Training]]**: Verwendet Random-Perturbation Gradient Estimation
- **[[GPU]]**: Matmul via Apple Accelerate / wgpu

## Konfiguration

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `--population` | 10 | Anzahl Modelle |
| `--generations` | 5 | Anzahl Generationen |
| `--data` | -- | Pfad zur Textdatei |
| `--quick` | -- | Eingebauter Sample-Text |

## Quick-Modus Daten

Der Quick-Modus verwendet einen eingebauten Text mit 20 Saetzen ueber ML-Themen:

```
"The quick brown fox jumps over the lazy dog.
Machine learning models can recognize patterns in data.
Neural networks are inspired by the human brain.
..."
```

## Warum Neuroevolution?

| Vorteil | Beschreibung |
|---------|-------------|
| Kein Gradient noetig | Funktioniert auch ohne differenzierbare Funktionen |
| Architektursuche | Findet automatisch d_model, heads, layers |
| Parallelisierbar | Jedes Modell ist unabhaengig |
| Robust | Entkommt lokalen Minima durch Exploration |
| Keine Hyperparameter-Suche | Evolution findet die besten von selbst |

Siehe [[Training]] fuer alle Trainingsarten, [[Vision]] fuer die groessere Idee hinter dem Swarm.

#swarm #neuroevolution #architecture-search #evolutionary
