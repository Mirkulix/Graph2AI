# QO — Aktueller Status (2026-04-08)

## Projekt

**QO = QLANG + Orbit** — Persönlicher KI-Companion als ein Rust-Binary.

**Verzeichnis:** `/home/mirkulix/neoqlang/qlang`
**Starten:** `QO_PORT=4747 ./target/release/qo` (aus dem qlang Verzeichnis)
**UI:** http://localhost:4747
**Branch:** `main` (142 Commits)
**Binary:** 9.6 MB, Port 4747

## Starten

```bash
cd /home/mirkulix/neoqlang/qlang
QO_PORT=4747 ./target/release/qo
# → http://localhost:4747
```

Falls neu bauen nötig:
```bash
cd /home/mirkulix/neoqlang/qlang
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo build --bin qo --release
cd frontend && npm run build && cd ..
```

## Was funktioniert (verifiziert)

### QLANG wird ECHT genutzt für:
- **Chat:** `qlang_runtime::executor::execute()` mit `Op::OllamaChat` (4 Nodes, 178 Bytes .qlbg)
- **Intent Classifier:** `qlang_runtime::training::MlpWeights` + Autograd (99.2% Accuracy, <1ms)
- **Goal-Graphen:** Echte QLANG-Graphen als .qlbg Binary (339 Bytes, 6 Nodes)
- **Quantum Evolution:** `qlang_core::quantum::DensityMatrix` + `quantum_flow::evolve_step`
- **IGQK Kompression:** `qlang_runtime::igqk_compress` — 16x Kompression in purem Rust

### Embeddings (candle, kein Ollama nötig):
- **Modell:** all-MiniLM-L6-v2 via candle (22MB, 384-dim)
- **Laden:** 64ms beim ersten Aufruf, dann gecached
- **Semantik:** rust-python=0.64, rust-kochen=0.33 (echtes Verständnis)

### Agenten:
- 6 Agenten: CEO, Researcher, Developer, Guardian, Strategist, Artisan
- Guardian ist **deterministisch** (kein LLM)
- Researcher hat **Web-Search** (DuckDuckGo)
- Subtasks laufen **parallel** (tokio::spawn)
- Simulation vor Goal-Execution (MiroFish-inspiriert)
- Quantum State steuert CEO-Strategie

### Frontend (8 Tabs):
- Chat, Ziele, Agenten, Bewusstsein, Provider, Evolution, QLANG, Historie
- Dark/Light Mode Toggle
- Keyboard Shortcuts (Ctrl+1-8, Ctrl+K, Ctrl+Enter)
- Export (Markdown, JSON)
- Activity Feed (SSE)
- Responsive Mobile Layout

### Persistenz:
- Goals, Agents, Patterns, Proposals, Quantum State in redb
- Chat-History in redb
- QLANG Graphen als .qlbg Binary in redb
- Embeddings in redb
- Provider-Konfiguration in redb
- Alles überlebt Restart

### Provider:
- 8 Templates: Groq, DeepSeek, OpenRouter, Gemini, Ollama, Mistral, OpenAI, Anthropic
- Hinzufügen/Bearbeiten/Testen/Löschen aus UI
- Tier-basiertes Routing (Lokal → Free → Paid)
- Cost-Tracking pro Provider

### Sonstiges:
- Telegram Bot (QO_TELEGRAM_TOKEN)
- Auth Middleware (QO_AUTH_TOKEN)
- Systemd Service (scripts/install-service.sh)
- Orbit Migration (scripts/migrate-orbit.sh)
- Obsidian Vault: ~/Dokumente/Obsidian Vault/QO/

## Was NICHT funktioniert / Offen

### QLANG wird NICHT genutzt für:
- Memory-Suche (eigener brute-force Code statt QLANG MatMul)
- Embeddings (candle statt QLANG-Graph)
- JIT Compilation (nie genutzt, alles Interpreter)

### Bekannte Probleme:
- Bewusstsein-Tab zeigt erst Daten nach 5s Polling (kein Heartbeat mehr)
- Intent Classifier: "Rust ist eine Programmiersprache" wird als Question statt Chat erkannt
- Embedding-Classifier auf echten Daten nur 50.7% (BoW-Classifier mit 99.2% bleibt besser)
- `data/qo.redb` wird oft mit `rm -f` gelöscht bei Tests — Daten gehen verloren
- Frontend ist seit vielen Änderungen nicht komplett neu getestet

### Noch nicht gebaut:
- 5-Tier Routing (Code existiert, nicht vollständig verdrahtet)
- IGQK-komprimiertes eigenes Modell als Tier 0
- QLANG-Graphen mit echten Tensor-Ops (MatMul, Attention, etc.)
- JIT-kompilierte Graphen für Performance
- Zweites spezialisiertes Modell (Code-Classifier)

## Crate-Übersicht

| Crate | Zweck |
|-------|-------|
| `qlang-core` | Graph, Tensor, Quantum, Ops, Binary Format |
| `qlang-compile` | LLVM JIT, WASM, GPU |
| `qlang-runtime` | Executor, Autograd, Training, IGQK |
| `qlang-agent` | QLMS Binary Protocol |
| `qo-server` | Axum HTTP + WebSocket + Static Files |
| `qo-agents` | 6 Hybrid-Agenten + Intent Classifier |
| `qo-consciousness` | State Machine + Broadcast Stream |
| `qo-memory` | redb + Obsidian + HNSW + Embeddings |
| `qo-llm` | Tiered LLM Routing + Ollama Client |
| `qo-values` | 5-Werte Scoring |
| `qo-evolution` | Patterns + Proposals + Quantum State |
| `qo-simulation` | MiroFish Szenario-Vorhersage |
| `qo-telegram` | Telegram Bot |
| `qo-embed` | candle all-MiniLM-L6-v2 Embedding |

## Tests

```bash
cd /home/mirkulix/neoqlang/qlang
LLVM_SYS_180_PREFIX=/opt/llvm18 cargo test -p qo-values -p qo-consciousness -p qo-memory -p qo-llm -p qo-agents -p qo-evolution -p qo-simulation -p qo-telegram -p qo-embed
```

Letzter Stand: 113+ Tests bestanden.

## Trainingsdaten

- **Intent Classifier:** 145 synthetische deutsche Sätze (via Groq generiert), 120 Wörter Vokabular
- **Große Daten:** 10.200 aus HuggingFace (german-conversations + evol-instruct-deutsch) — Classifier nur 50.7% damit
- **Orbit Training Data:** 13.968 Sätze — 84% Heartbeat-Logs, unbrauchbar für Intent
- **Obsidian:** 2.655 Markdown-Dateien, 2.223 Pattern-Dateien

## Umgebungsvariablen

```bash
QO_PORT=4747              # Server Port
GROQ_API_KEY=...          # Groq Free Tier (aus ~/.openclaw/.env)
OLLAMA_URL=http://localhost:11434  # Ollama für Chat
OLLAMA_MODEL=qwen2.5:3b  # Default Chat-Modell
QO_AUTH_TOKEN=...         # Optional: Bearer Token Auth
QO_TELEGRAM_TOKEN=...    # Optional: Telegram Bot
QO_TELEGRAM_CHAT_ID=...  # Optional: Erlaubte Chat-ID
LLVM_SYS_180_PREFIX=/opt/llvm18  # LLVM für JIT (Build)
```

## Nächste Schritte (empfohlen)

1. **QO benutzen und echte Probleme finden** statt neue Features
2. **5-Tier Routing fertig verdrahten** (orbit-companion-ft für Chat, qwen für Fragen, Groq für Goals)
3. **Toten Code löschen** (MiroFish-Frontend-Placeholder, etc.)
4. **End-to-End Test** der ganzen Kette verifizieren
5. **Zweites QLANG-Modell** (Code-Classifier)

## Design Spec

Vollständige Architektur-Spec: `docs/superpowers/specs/2026-04-07-qo-design.md`
