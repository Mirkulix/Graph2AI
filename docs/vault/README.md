# docs/vault — Index

Tiefe Design-Notizen fuer QLANG / QLMS / QO. Diese Dateien beschreiben das
*Warum* und *Wie* der Plattform; was tatsaechlich auf realen Daten laeuft,
steht in der Repo-Wurzel in `QLANG-STATUS.md` (Single Source of Truth).

Jede Datei traegt am Anfang einen `> Status:`-Marker:

- **stable** — entspricht implementiertem, getestetem Code.
- **planned** — Vision, Roadmap oder Spezifikation; noch nicht vollstaendig
  realisiert.
- **experimental** — beschreibt Code, der hinter `--features experimental-ml`
  liegt oder noch nicht produktiv ist.

Bei Widerspruch gewinnt `QLANG-STATUS.md`.

## Stable — implementiert

| Datei | Inhalt |
|-------|--------|
| `Architecture.md` | Crate-Layout des Rust-Workspaces (qlang-core, -compile, -runtime, -agent). |
| `Protocol.md` | QLMS Wire Format v1 (Envelope, GraphMessage, Signing, Merkle). |
| `BinaryFormat.md` | QLBG Binary-Graph-Format (Magic, Op-Tags, SHA-256 Content Hash). |
| `Crypto.md` | Pure-Rust SHA-256, HMAC, Merkle, Signing-Flow, Content-Cache. |
| `Decisions.md` | Architektur-Entscheidungen mit Begruendung (warum Rust, warum Binary, warum eigenes Crypto). |
| `Glossary.md` | A-Z Begriffsreferenz fuer das gesamte Vault. |
| `Language.md` | Graph-Sprache + VM-Scripting-Syntax (`.qlang`). |
| `Execution.md` | 3-Tier-Ausfuehrung (LLVM JIT / Bytecode VM / Tree-Walking Interpreter). |
| `GPU.md` | Hardware-Backends (wgpu, MLX, Accelerate) und Multi-GPU-Training. |
| `QLMS_BENCHMARK.md` | Reproduzierbare QLMS-vs-MCP-Messung (Ryzen 9, 2026-04-12). |

## Planned — Vision & Submission

| Datei | Inhalt |
|-------|--------|
| `STRATEGIC_VISION.md` | Single Vision-/Roadmap-Dokument: Why, What, UVP, Goals P0-P2, Risks. |
| `QLMS_SUBMISSION_CHECKLIST.md` | AAIF-(Linux Foundation)-Submission-Tracker fuer QLMS v1.1. |
| `Comparison.md` | Positionierung vs PyTorch / MCP / ONNX / TF-Lite — Marketing-Werte gegen QLANG-STATUS pruefen. |

## Experimental — hinter `--features experimental-ml`

| Datei | Inhalt |
|-------|--------|
| `IGQK.md` | Theoretisches Framework (Information-Geometric Quantum Compression). Ternary-Pack ist real (16x), voller Quantum-Gradient-Flow ist vereinfacht. |

## Archive

`archive/` enthaelt veraltete oder konsolidierte Dokumente. Nichts wurde
geloescht — bei Bedarf koennen Inhalte zurueckgeholt werden.

Konsolidierungs-Notizen:

- **Vision-Track:** `Vision.md`, `QLANG_v2_Architecture.md`, `Roadmap.md`,
  `MASTER_AGENT.md` -> ersetzt durch `STRATEGIC_VISION.md`.
- **Index:** `Home.md` -> ersetzt durch dieses `README.md`.
- **Outdated CLI/UI:** `CLI.md`, `HowTo.md`, `WebUI.md`, `Agents.md`,
  `Ollama.md`, `Training.md` -> referenzieren das alte `qlang-cli`-Tooling
  und Port 8081, beides ueberholt (siehe QLANG-STATUS fuer aktuelles
  `qlang`-Binary und QO auf Port 4747).
- **Experimental ML:** `Transformer.md`, `Swarm.md`, `Diffusion.md`,
  `ParaDiffuse.md`, `TSLM.md`, `GPU_QAT_PROFILING.md` -> Code lebt unter
  `--features experimental-ml`; die Dokumente beschreiben Komponenten, die
  laut QLANG-STATUS heute nicht produktiv sind (Hebbian, Spiking, Random-
  Perturbation Transformer-Training, Random-Conv-Features etc.).

Die Archiv-Dateien tragen *keinen* aktualisierten Status-Marker; sie sind
historisch und gelten nur fuer den Stand zum Zeitpunkt ihres letzten Edits.
