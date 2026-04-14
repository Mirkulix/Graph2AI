# Glossary

Alle QLANG-spezifischen Begriffe und Abkuerzungen.

## A

**Agent**
Ein AI-System, das das QLMS [[Protocol]] spricht. Agents senden und empfangen `GraphMessage` Werte mit Berechnungsgraphen. Siehe [[Agents]].

**AOT (Ahead-of-Time)**
Kompilierung eines QLANG-Graphen zu einer nativen Object-Datei (`.o`) vor der Ausfuehrung. Kontrast zu JIT. Siehe [[GPU]].

**Autograd**
Reverse-Mode automatische Differenzierung. Zeichnet Operationen auf einem Tape waehrend des Forward Pass auf, spielt sie dann rueckwaerts ab um Gradienten zu berechnen. Siehe [[Training]].

## B

**Born Rule**
Die Quantenmessformel `P(w|rho) = Tr(rho * M_w)`, die eine kontinuierliche Dichtematrix zu diskreten Gewichten kollabiert. Siehe [[IGQK]].

**BPE (Byte-Pair Encoding)**
Tokenisierungsverfahren. Startet mit 256 Byte-Token, mergt iterativ die haeufigsten Paare bis zur gewuenschten Vokabulargroesse. Implementiert in `tokenizer.rs`. Siehe [[Transformer]].

**Bytecode VM**
Tier 2 im [[Execution]]-System. Kompiliert AST zu flachem Bytecode, fuehrt auf Stack-Maschine aus. 10-50x schneller als der Tree-Walking Interpreter. Gleiches Prinzip wie CPython.

## C

**Capability**
Eine Funktion, die ein Agent deklariert: Execute, Compile, Optimize, Compress, Train, Verify. Verwendet bei [[Agents]] Negotiation.

**Commutator**
`[A, B] = AB - BA`. Repraesentiert unitaere (Quantum) Evolution im [[IGQK]] Gradientenfluss.

**Anticommutator**
`{A, B} = AB + BA`. Repraesentiert dissipative (Gradient Descent) Evolution im [[IGQK]] Gradientenfluss.

**Content-Addressable Cache**
Cache in `cache.rs`, der Berechnungsergebnisse ueber SHA-256 Hashes der Inputs adressiert. Siehe [[Crypto]].

**Cosine Schedule**
Noise-Schedule fuer [[Diffusion]] (Nichol & Dhariwal, 2021). Haelt hohes Signal-to-Noise Ratio laenger aufrecht als Linear.

## D

**DAG (Directed Acyclic Graph)**
Die Programmdarstellung in QLANG. Jedes Programm ist ein DAG aus getypten Knoten, verbunden durch Kanten. Siehe [[Architecture]].

**DDIM (Denoising Diffusion Implicit Models)**
Deterministische Sampling-Methode fuer [[Diffusion]] Modelle (Song et al., 2020). Generiert in 10-20 Steps statt 1000.

**Density Matrix (rho)**
Eine positiv semidefinite Matrix mit `Tr(rho) = 1`, die einen Quanten-/probabilistischen Zustand ueber neuronale Netzwerkgewichte darstellt. Siehe [[IGQK]].

**Dtype**
Datentyp von Tensor-Elementen: `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `bool`, `ternary`.

## E

**Edge**
Eine gerichtete Datenfluss-Verbindung zwischen zwei Knoten in einem Graphen, die einen getypten Tensor traegt.

**Emitter**
Der `GraphEmitter` in `qlang-agent` -- eine strukturierte API fuer AI Agents, um Graphen ohne Text-Syntax zu bauen. Siehe [[Agents]].

## F

**FCHL (Fractional Calculus Hebbian Learning)**
Eine der drei Theorien, die durch [[IGQK]] vereint werden. Verwendet fraktionalen Laplace-Beltrami Operator.

**Fisher Information Metric**
Die Riemannsche Metrik auf der statistischen Mannigfaltigkeit der Gewichte: `g_ij(theta) = E[d_i log p * d_j log p]`. Siehe [[IGQK]].

## G

**GELU (Gaussian Error Linear Unit)**
Aktivierungsfunktion: `x * Phi(x)`. Verwendet in GPT und BERT. In QLANG via `use_silu: false` in TransformerConfig. Siehe [[Transformer]].

**GpuContext**
Persistent GPU-Kontext in `gpu_compute.rs`. Enthaelt wgpu Device, Queue und vorkompilierte Matmul-Pipeline. Siehe [[GPU]].

**GradientBuffer**
Puffer fuer Gradient-Akkumulation und -Mittelung im Multi-Worker [[Training]]. Siehe [[GPU]].

**Graph**
Die fundamentale Programmeinheit in QLANG. Enthaelt Knoten, Kanten, Constraints und Metadaten. Siehe [[Language]].

**GraphMessage**
Die Kommunikationseinheit im QLMS [[Protocol]]. Enthaelt einen Graphen, Inputs, Intent und optionale kryptographische Signatur.

## H

**Hebbian Learning**
Bio-inspiriertes, gradient-freies Lernen: "Neurons that fire together wire together." Implementiert in `hebbian.rs`. Siehe [[ParaDiffuse]].

**HebbianState**
Zustand fuer Hebbian Learning: Salience-Matrix, Running Means, Threshold, Momentum. Siehe [[ParaDiffuse]].

**HLWT (Hybrid Laplace-Wavelet Transformation)**
Eine der drei Theorien, die durch [[IGQK]] vereint werden. HLWT ist die Fourier-Transformation des Quantum Gradient Flow.

## I

**IGQK (Informationsgeometrische Quantenkompression)**
Information-Geometric Quantum Compression. Das theoretische Framework hinter QLANG's Kompression. Siehe [[IGQK]].

**Intent (MessageIntent)**
Was der Sender erwartet: Execute, Optimize, Compress, Verify, Result, Compose, Train. Siehe [[Agents]].

## J

**JIT (Just-In-Time)**
Kompilierung zu nativem Code zur Laufzeit. QLANG hat zwei JIT-Pfade: Graph JIT (`codegen.rs`) und Script JIT (`script_jit.rs`). Siehe [[Execution]].

## K

**Keypair**
Kryptographisches Schluesselpaar fuer [[Protocol]]-Signaturen. 32-Byte Public Key + Signing Key. Siehe [[Crypto]].

## L

**Laplace-Beltrami Operator**
`H = -Delta_M`, die Verallgemeinerung des Laplace-Operators auf Riemannsche Mannigfaltigkeiten. Hamiltonoperator in [[IGQK]].

**LSP (Language Server Protocol)**
IDE-Integrationsserver mit Diagnostics, Completions, Hover und Goto-Definition fuer `.qlang` Dateien.

## M

**Manifold**
Ziel-Untermannigfaltigkeit fuer Kompression: Ternary ({-1,0,+1}), LowRank (rank <= r), Sparse (||W||_0 <= s). Siehe [[IGQK]].

**Merkle Tree**
Hash-Baum ueber Graph-Knoten fuer partielle Verifikation. Jeder Knoten bekommt einen SHA-256 Hash. Siehe [[Crypto]].

**MiniGPT**
QLANG's GPT-style Decoder-only Transformer Language Model. Konfigurierbar via `TransformerConfig`. Siehe [[Transformer]].

**MLX**
Apples Machine Learning Framework fuer Metal GPU. Optionales Backend via `--features mlx`. Siehe [[GPU]].

## N

**Node**
Eine einzelne Berechnung in einem Graphen, identifiziert durch `NodeId` (u32). Hat Operationstyp (`Op`), getypte Input-Ports und Output-Ports.

**NodeId**
Ein `u32` Identifier fuer einen Knoten innerhalb eines Graphen.

## O

**Op (Operation)**
Der Berechnungstyp eines Graph-Knotens. 40+ Varianten im `Op` Enum. Siehe [[Language]].

**ONNX**
Open Neural Network Exchange Format. QLANG kann ONNX Modelle importieren und exportieren.

## P

**ParaDiffuse**
Python/PyTorch-Projekt fuer parallele Diffusion-basierte Generierung. Kernkonzepte portiert nach Rust. Siehe [[ParaDiffuse]].

**Proof**
Ein Compile-Time Zertifikat, dass ein Constraint gilt. Hat `TheoremRef`, Parameter und Status. Siehe [[IGQK]].

## Q

**QBPE**
Binaerformat fuer BPE Tokenizer. Magic Bytes: `"QBPE"`. Siehe [[Transformer]].

**QLBG**
Binary Graph Format mit Magic Bytes `0x51 0x4C 0x42 0x47`. Kompakte Serialisierung mit SHA-256 Content Hash. Siehe [[BinaryFormat]].

**QLMS**
QLANG Message Stream. Binaeres Wire Format fuer Agent-Kommunikation mit Magic Bytes `0x51 0x4C 0x4D 0x53`. Siehe [[Protocol]].

**Quantum Gradient Flow**
Die Evolutionsgleichung `d_rho/dt = -i[H, rho] - gamma{G^-1 * grad_L, rho}` die das [[IGQK]] Training antreibt.

## R

**REPL**
Read-Eval-Print Loop. Interaktive Shell fuer QLANG-Ausdruecke via `qlang-cli repl`. Siehe [[CLI]].

**RMSNorm (Root Mean Square Normalization)**
Layer-Normalisierung nur mit RMS, ohne Mean-Centering. ~15% schneller als LayerNorm. Verwendet in LLaMA, Mistral. Siehe [[Transformer]].

## S

**Salience**
Akkumuliertes Korrelationssignal im [[ParaDiffuse]] Hebbian Learning. Wenn Salience den Threshold ueberschreitet, wird ein Gewicht geflippt.

**Shape**
Die Dimensionen eines Tensors: `[]` (Skalar), `[784]` (Vektor), `[784, 128]` (Matrix), `[?, 784]` (dynamisch).

**SiLU (Sigmoid Linear Unit)**
Aktivierungsfunktion: `x * sigmoid(x)`. Auch bekannt als Swish. Verwendet in LLaMA, Mistral. Siehe [[Transformer]].

**Statistical Manifold**
Die Riemannsche Mannigfaltigkeit der Gewichte theta mit Fisher-Information Metrik. Siehe [[IGQK]].

**Swarm**
Evolutionaere Architektursuche fuer Language Models. Population von Modellen konkurriert, mutiert, evolviert. Siehe [[Swarm]].

## T

**TensorData**
Ein konkreter Tensor mit Dtype, Shape und rohen Datenbytes.

**TensorType**
Eine Tensor-Typ-Deklaration: Dtype + Shape. Z.B. `f32[784, 128]`.

**Ternary**
Gewichtswerte beschraenkt auf {-1, 0, +1}. 16x Kompression gegenueber f32. Siehe [[IGQK]].

**TLGT (Ternary Lie Group Theory)**
Eine der drei Theorien, die durch [[IGQK]] vereint werden. Die ternaere Gruppe G3 ist eine diskrete Untergruppe der Quantensymmetriegruppe.

**TransformerConfig**
Konfiguration fuer MiniGPT: vocab_size, d_model, n_heads, n_layers, max_seq_len, use_rms_norm, use_silu. Siehe [[Transformer]].

## U

**Unified Runtime**
Bridge zwischen VM (Scripting) und Graph (ML) in `unified.rs`. Eine `.qlang` Datei kann beides. Siehe [[Execution]].

## V

**VM (Virtual Machine)**
Der Stack-basierte Interpreter in `vm.rs`. Fuegt General-Purpose Programmierung ueber Graphen hinzu. Siehe [[Language]], [[Execution]].

**Von Neumann Entropy**
`S(rho) = -Tr(rho log rho)`. Misst Quantum-Unsicherheit einer Dichtematrix. Debugging-Tool fuer [[IGQK]].

## W

**WASM (WebAssembly)**
Kompilierungsziel fuer Browser-Deployment. QLANG generiert WAT Textformat. Siehe [[GPU]].

**WGSL**
WebGPU Shading Language. QLANG generiert WGSL Compute Shader fuer GPU-Ausfuehrung. Siehe [[GPU]].

**WebSocket**
Real-Time Kommunikationsprotokoll des [[WebUI]] Dashboards. Implementiert von Grund auf mit `std::net`.

**wgpu**
Cross-platform GPU Compute Bibliothek. QLANG nutzt sie fuer Matmul auf NVIDIA/AMD/Intel/Apple GPUs. Siehe [[GPU]].

## X

**Xorshift64**
Schneller PRNG in der [[Diffusion]] Engine und [[ParaDiffuse]] Hebbian Learning. Keine externen Abhaengigkeiten.

#glossary #reference
