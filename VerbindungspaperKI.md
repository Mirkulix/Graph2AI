# Verbindungspaper KI — OrbitQLang Simplification

Dieses Dokument dient als Orientierungshilfe für KI-Coding-Agenten, die an diesem Repository arbeiten. Es beschreibt den aktuellen Stand nach der radikalen Systemvereinfachung (April 2026).

## 1. System-Status: "Deterministic Control Plane"

Das Projekt wurde von einer experimentellen, ML-lastigen Architektur zu einer schlanken, deterministischen **AI-to-AI Control Plane** umgebaut.

- **Kern-Technologie**: QLANG (Graph-basierte Sprache) und QLMS (Signed Executable Graphs).
- **Zustand**: Alle "schweren" KI-Komponenten (Candle, Mamba, MNIST-Training, Evolution-Loops) wurden entfernt.
- **Ziel**: Ein effizienter, sicherer Übertragungsweg für Agenten-Befehle mittels Graphen.

## 2. Wichtige Änderungen (Historie für Agenten)

Falls du in älteren Logs oder Dokumenten Hinweise auf folgende Dinge findest: **IGNORIERE SIE.** Sie wurden gelöscht, um das Projekt "leicht" zu machen:
- `candle-core` & `mamba-tokenizer`: Entfernt.
- `MnistData` & MLP-Training: Entfernt.
- `qlang-python` & `qo-embed`: Entfernt.
- `WebServer` (alt/komplex): Ersetzt durch eine minimalistische Implementierung in `crates/qlang-runtime/src/web_server.rs`.

## 3. Anweisungen für KI-Systeme (Guidelines)

Folge diesen Regeln, um die Integrität der neuen Architektur zu wahren:

1.  **Keine spekulativen ML-Refactorings**: Baue keine großen neuronalen Netze oder Trainings-Loops wieder ein. Das System soll deterministisch bleiben.
2.  **Protokoll-Fokus**: Alle Agenten-Interaktionen laufen über den `MessageBus` und verwenden QLANG-Graphen.
3.  **Deterministische Klassifizierung**: Intent-Klassifizierung findet jetzt regelbasiert (Keyword-Scores) in `qo-agents/src/qlang_model.rs` statt. Verändere dies nicht zurück zu einem MLP, ohne explizite Aufforderung.
4.  **Minimalistischer Web-Server**: Die `web_server.rs` in `qlang-runtime` ist absichtlich puristisch gehalten (kein `tokio`, kein `hyper`). Erweitere sie nur mit Standard-Bibliotheken (`std`), wenn möglich.
5.  **Status-Check**: Bevor du Änderungen vornimmst, prüfe immer zuerst `QLANG-STATUS.md` für den aktuell gültigen Funktionsumfang.

## 4. Aktuelle Architektur-Map

- `/crates/qlang-runtime`: Enthält den Bytecode-Interpreter und den minimalen WebSocket-Server.
- `/qo/qo-agents`: Verwaltet die Rollen (CEO, Researcher, etc.) und deren Intent-Scoring.
- `/qo/qo-server`: Das Axum-Backend für die Supervisor-API (ohne Federation/ML-Demos).
- `/frontend`: Das Dashboard (Vite/React), das via WebSocket (`/ws`) Events empfängt.

---
> Dieses Dokument ist die "Wahrheitsquelle" für die Kooperation zwischen KI-Agenten in diesem Repository. Halte dich an den "Lean & Deterministic"-Kurs.
