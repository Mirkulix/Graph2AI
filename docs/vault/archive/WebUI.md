# WebUI

QLANG enthaelt ein Web-Dashboard fuer Live-Monitoring von Training, Graph-Ausfuehrung und Agent-Kommunikation. Gebaut mit reinem HTML/CSS/JS -- keine Framework-Abhaengigkeiten.

## Starten

```bash
# Statisches Dashboard
qlang-cli web --port 8081

# Live MNIST Training mit Dashboard
qlang-cli train-mnist --port 8081 --epochs 50
```

Oeffne `http://localhost:8081` im Browser.

## Design

Das neue Design ist minimalistisch: ein **Live-Feed** links und eine **Sidebar mit 4 Aktionen** rechts.

### Layout

```
+----------------------------------------------------------+
| QLANG        [Status-Dot]  Live                   [Docs] |
+-----------------------------------+----------------------+
|                                   |                      |
|  LIVE FEED                        |  ACTIONS SIDEBAR     |
|                                   |                      |
|  [time] [TAG] Message...          |  ┌ Run QLANG Code ┐  |
|  [time] [TAG] Message...          |  │ Code Editor     │  |
|  [time] [TAG] Message...          |  │ [Run]           │  |
|  [time] [TAG] Message...          |  └─────────────────┘  |
|  [time] [TAG] Message...          |                      |
|  [time] [TAG] Message...          |  ┌ Train MNIST ────┐  |
|                                   |  │ Epochs: [50]    │  |
|  Scrollt automatisch nach unten   |  │ LR:     [0.01]  │  |
|                                   |  │ [Start]         │  |
|                                   |  └─────────────────┘  |
|                                   |                      |
|                                   |  ┌ IGQK Compress ──┐  |
|                                   |  │ [Compress]      │  |
|                                   |  └─────────────────┘  |
|                                   |                      |
|                                   |  ┌ Autonomous ──────┐ |
|                                   |  │ Task: [...]      │  |
|                                   |  │ [Start]          │  |
|                                   |  └──────────────────┘  |
+-----------------------------------+----------------------+
```

### Feed-Nachrichten

Jede Nachricht im Feed hat:
- Zeitstempel (monospace, grau)
- Farbcodierter Tag
- Nachrichtentext

| Tag | Farbe | Bedeutung |
|-----|-------|-----------|
| `SYS` | Blau | System-Meldungen (Startup, Config) |
| `AI` | Orange | AI/Ollama Kommunikation |
| `TRAIN` | Gruen | Training-Fortschritt |
| `IGQK` | Lila | Kompression, Quantum State |
| `STEP` | Gelb | Einzelne Training-Steps |
| `OK` | Gruen (Hintergrund) | Erfolg, Ziel erreicht |
| `ERR` | Rot | Fehler |

### Sidebar-Aktionen

1. **Run QLANG Code** (Blau) -- Code-Editor mit Syntax-Highlighting, Code direkt ausfuehren
2. **Train MNIST** (Gruen) -- Epochs und LR einstellen, Training mit Live-Updates starten
3. **IGQK Compress** (Lila) -- Trainiertes Modell komprimieren (Ternary/Low-Rank/Sparse)
4. **Autonomous** (Orange) -- AI Feedback-Loop starten (Task und Ziel angeben)

## Architektur

Der Server (`web_server.rs`) implementiert:

- **HTTP File Serving** fuer das Dashboard HTML/CSS/JS
- **WebSocket Protocol** (RFC 6455) fuer Real-Time Event Streaming
- Alles nur mit `std::net` -- keine externen HTTP- oder WebSocket-Crates
- Eigene SHA-1 Implementierung fuer den WebSocket-Handshake

## WebSocket Events

Der Server broadcastet `WebEvent` Nachrichten an alle verbundenen Clients:

| Event | Felder | Zweck |
|-------|--------|-------|
| `GraphNodeExecuted` | node_id, op, shape, time_us, values | Per-Node Execution Tracking |
| `TrainingEpoch` | epoch, loss, accuracy | Training-Fortschritt |
| `AgentMessage` | from, to, message | Agent-Kommunikationslog |
| `CompressionResult` | method, ratio, accuracy_before/after | [[IGQK]] Kompressionsergebnisse |
| `SystemLog` | level, message | System-Level Log |
| `GraphLoaded` | name, num_nodes, num_edges | Graph-Visualisierungsdaten |
| `ModelSaved` | name, version | Checkpoint-Save Events |

## Interaktive Vorhersage

Nach dem Training unterstuetzt das Dashboard interaktive Ziffern-Vorhersage:

1. Ziffer auf dem Canvas zeichnen
2. Zeichnung wird via WebSocket an den Server gesendet
3. Server fuehrt Inferenz auf dem trainierten Modell aus
4. Vorhersage-Ergebnis wird in Echtzeit angezeigt

## Styling

Dunkles Theme mit GitHub-inspirierten Design Tokens:

| Token | Farbe | Verwendung |
|-------|-------|------------|
| `--bg` | `#000000` | Haupthintergrund |
| `--s1` | `#0d1117` | Sidebar, Header |
| `--s2` | `#161b22` | Karten, Felder |
| `--tx` | `#e6edf3` | Text |
| `--bl` | `#58a6ff` | Akzent Blau |
| `--gn` | `#3fb950` | Erfolg Gruen |
| `--or` | `#f97316` | AI Orange |
| `--pu` | `#bc8cff` | IGQK Lila |
| `--rd` | `#f85149` | Fehler Rot |

Siehe auch: [[CLI]] fuer Start-Befehle, [[Agents]] fuer Agent-Message-Monitoring.

#webui #dashboard #monitoring
