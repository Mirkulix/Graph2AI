# Comparison

Wie QLANG sich von bestehenden Tools und Frameworks unterscheidet.

## QLANG vs Python/PyTorch

| Dimension | Python/PyTorch | QLANG |
|-----------|---------------|-------|
| **Sprache** | Python | Rust |
| **Deployment** | Python Runtime (~100MB) | Single Binary (~10MB) |
| **Abhaengigkeiten** | PyTorch, NumPy, etc. | Zero (core) |
| **GIL** | Ja (limitiert Parallelismus) | Nein (echter Multithreading) |
| **Speicher** | GC Overhead | Zero-Cost Abstractions |
| **Startup** | ~1s (Python Interpreter) | ~1ms (native) |
| **ML Training** | Vollstaendig | MLP, Transformer, Swarm |
| **Ecosystem** | Riesig | Wachsend |
| **GPU** | CUDA (NVIDIA only) | wgpu (NVIDIA + AMD + Intel + Apple) |
| **Kompression** | Post-Training | [[IGQK]] (waehrend Training) |
| **Protokoll** | Kein Standard | [[Protocol]] (binaeruebertragung) |
| **Graph-Repr.** | Implizit (eager) | Explizit (DAG first) |

**Wann QLANG waehlen**: Edge Deployment, AI-to-AI Kommunikation, Binary Protocol, Multi-GPU ohne CUDA, Single Binary Distribution.

**Wann PyTorch waehlen**: Forschung, schnelles Prototyping, riesiges Ecosystem, State-of-the-Art Modelle.

## QLANG vs MCP (Model Context Protocol)

| Dimension | MCP | QLANG |
|-----------|-----|-------|
| **Format** | JSON-RPC | Binaer ([[BinaryFormat]]) |
| **Groesse** | ~3.5x groesser | 1x (Basis) |
| **Inhalt** | Tool Calls + Responses | Computation Graphen |
| **Semantik** | "Rufe dieses Tool auf" | "Fuehre diesen Graphen aus" |
| **Verifikation** | Keine | SHA-256 + Merkle Trees |
| **Signierung** | Keine | HMAC-SHA256 |
| **ML Training** | Nein | Ja (komplett) |
| **Kompression** | Nein | [[IGQK]] (16x) |

**QLANG's Vorteil**: MCP beschreibt WAS getan werden soll. QLANG beschreibt WIE es berechnet wird -- als verifizierbarer, signierbarer Graph.

## QLANG vs AutoGPT / AI Agent Frameworks

| Dimension | AutoGPT | QLANG Agents |
|-----------|---------|-------------|
| **Entscheidungen** | Token-by-Token Text | 4 strukturierte Entscheidungen |
| **Fehler** | Syntaxfehler moeglich | Valid by Construction |
| **Overhead** | 47 Tokens fuer eine Berechnung | 4 Entscheidungen |
| **Verifizierbar** | Nein | SHA-256 + [[Crypto]] Proofs |
| **Kommunikation** | Natuerliche Sprache | Binaeres [[Protocol]] |
| **ML Training** | Delegiert an externe Tools | Eingebaut |
| **Autonomie** | Prompt-basiert | Feedback-Loop mit Ollama |
| **Skalierung** | Ein Agent | [[Swarm]] von Agents |

**QLANG's Vorteil**: Strukturierte, verifizierbare AI-to-AI Kommunikation statt unstrukturiertem Text.

## QLANG vs TensorFlow Lite

| Dimension | TF Lite | QLANG |
|-----------|---------|-------|
| **Zielgeraet** | Mobile / Edge | Ueberall (Desktop, Server, Edge, Browser) |
| **Format** | FlatBuffer (.tflite) | QLBG + QLMS ([[BinaryFormat]]) |
| **Training** | Nur Inferenz | Training + Inferenz |
| **Sprache** | C++ Kern + Python Tools | Reines Rust |
| **Quantisierung** | Int8/Float16 | Ternaer {-1,0,+1} via [[IGQK]] |
| **Kompressionsrate** | 2-4x | 16x (ternary) |
| **Graph-Exchange** | Export from TF | Native binaer + signiert |
| **Abhaengigkeiten** | TensorFlow | Zero (core) |

**QLANG's Vorteil**: 16x Kompression vs 2-4x, plus Training auf dem Geraet, plus kryptographische Verifikation.

## QLANG vs ONNX Runtime

| Dimension | ONNX Runtime | QLANG |
|-----------|-------------|-------|
| **Zweck** | Cross-Framework Inferenz | Vollstaendiges ML System |
| **Format** | ONNX Protobuf | QLBG Binary |
| **Parser** | Google Protobuf (20+ Deps) | Eigener Minimaler (~200 Zeilen) |
| **Ops** | 150+ | 40+ |
| **Training** | Nur via separate Tools | Eingebaut (MLP, Transformer, Swarm) |
| **Kompression** | Nicht eingebaut | [[IGQK]] (waehrend Training) |
| **Interop** | ONNX Import/Export | ONNX Import/Export + natives Format |
| **AI Protocol** | Keins | [[Protocol]] (QLMS) |

**QLANG's Vorteil**: Nicht nur Inferenz, sondern komplettes Oekosystem inkl. Training, Kompression, Kommunikation.

## Feature-Matrix

| Feature | PyTorch | TF Lite | ONNX RT | MCP | AutoGPT | **QLANG** |
|---------|---------|---------|---------|-----|---------|-----------|
| ML Training | Ja | Nein | Nein | Nein | Nein | **Ja** |
| Edge Deploy | Nein | Ja | Ja | n/a | Nein | **Ja** |
| Binary Protocol | Nein | Ja | Ja | Nein | Nein | **Ja** |
| Krypto-Signierung | Nein | Nein | Nein | Nein | Nein | **Ja** |
| Kompression | Post-hoc | Int8 | Nein | n/a | Nein | **16x** |
| Zero-Deps Core | Nein | Nein | Nein | Nein | Nein | **Ja** |
| AI-to-AI Protocol | Nein | Nein | Nein | Ja | Nein | **Ja** |
| Swarm Training | Nein | Nein | Nein | Nein | Nein | **Ja** |
| WASM Target | Nein | Nein | Ja | n/a | Nein | **Ja** |
| Single Binary | Nein | Ja | Nein | n/a | Nein | **Ja** |

## Fazit

QLANG ist kein Ersatz fuer PyTorch in der Forschung. Es ist ein **Produktions-Oekosystem** fuer:

1. AI-to-AI Kommunikation via Graph-Protocol
2. Training + Kompression + Deployment in einem Tool
3. Edge Deployment als Single Binary
4. Kryptographisch verifizierbare Berechnungen
5. Evolutionaere Architekturoptimierung

Siehe [[Vision]] fuer die langfristige Richtung.

#comparison #alternatives #positioning
