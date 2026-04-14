# Execution

QLANG hat ein 3-Tier Ausfuehrungs-System. Je nach Code-Typ wird automatisch die schnellste verfuegbare Engine gewaehlt.

## Die 3 Tiers

```
Tier 1: LLVM JIT ──── Nativgeschwindigkeit (wie C)
Tier 2: Bytecode VM ── 10-50x schneller als Interpreter
Tier 3: Interpreter ── Alle Features, langsamster
```

### Tier 1: LLVM JIT (`script_jit.rs`)

**Geschwindigkeit**: Native (wie hand-geschriebenes C)

Kompiliert numerische QLANG-Scripts direkt zu nativem Maschinencode via LLVM.

**Unterstuetzt:**
- Variablen (numerisch)
- Arithmetik (`+`, `-`, `*`, `/`, `%`, `**`)
- Vergleiche (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- Kontrollfluss (`if`/`else`, `while`, `for`)
- Funktionen (`fn`, `return`)
- `print` (via FFI Callback)

**Nicht unterstuetzt (Fallback auf Tier 2):**
- Strings, Arrays, Dicts
- Imports
- Index-Ausdruecke
- Built-in Funktionen (ausser print)

```rust
use qlang_compile::script_jit::ScriptJit;
use inkwell::context::Context;

let ctx = Context::create();
let mut jit = ScriptJit::new(&ctx)?;
let (result, output) = jit.compile_and_run(&statements)?;
```

### Tier 2: Bytecode VM (`bytecode.rs`)

**Geschwindigkeit**: 10-50x schneller als Tree-Walking Interpreter

Kompiliert AST zu flachem Bytecode, fuehrt ihn auf einer Stack-Maschine aus. Gleiches Prinzip wie CPython, Lua, Ruby -- keine externen Abhaengigkeiten.

**Unterstuetzt:**
- Alles was der Interpreter kann (Variablen, Strings, Arrays, Dicts, Funktionen)
- Vollstaendige Operator-Menge (Arithmetik, Logik, Bitweise)
- Kontrolfluss (if/else, while, for)
- Built-in Funktionen

**Bytecode Instruction Set:**

| Kategorie | Opcodes |
|-----------|---------|
| Stack | `Const`, `Pop` |
| Variablen | `LoadLocal`, `StoreLocal`, `LoadGlobal`, `StoreGlobal` |
| Arithmetik | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `Neg` |
| Bitweise | `BitAnd`, `BitOr`, `BitXor`, `BitNot`, `Shl`, `Shr` |
| Vergleich | `Eq`, `Ne`, `Lt`, `Gt`, `Le`, `Ge` |
| Logik | `And`, `Or`, `Not` |
| Kontrollfluss | `Jump`, `JumpIfFalse`, `JumpIfTrue`, `Loop` |
| Funktionen | `Call`, `Return` |
| Built-ins | `Print`, `Len`, `TypeOf`, `Str`, `Int`, `Sqrt`, `Abs`, `Floor`, `Min`, `Max`, `Push` |
| Datenstrukturen | `MakeArray`, `MakeDict`, `Index` |
| Halt | `Halt` |

```rust
use qlang_runtime::bytecode::run_bytecode;

let (value, output) = run_bytecode(source_code)?;
```

### Tier 3: Tree-Walking Interpreter (`vm.rs`)

**Geschwindigkeit**: Langsamstes, aber vollstaendigstes Tier

Traversiert den AST direkt -- keine Kompilierung. Unterstuetzt alle QLANG-Features inklusive Graphen.

**Unterstuetzt:**
- Alles aus Tier 2
- Graph-Definitionen und -Ausfuehrung (Unified Mode)
- Alle 53 [[Language]] Stdlib-Funktionen
- Import-System
- Interaktive REPL

```rust
use qlang_runtime::vm::run;

let (value, output) = run(source_code)?;
```

## Wann wird was verwendet?

| Szenario | Tier | Grund |
|----------|------|-------|
| Numerischer Loop (Fibonacci) | Tier 1 (LLVM JIT) | Nur Zahlen + Kontrollfluss |
| Script mit Strings/Arrays | Tier 2 (Bytecode) | Strings nicht in JIT |
| Script mit Graph-Blöcken | Tier 3 (Interpreter) | Braucht Unified Runtime |
| `qlang-cli exec` | Auto-Detect | Versucht JIT, dann Bytecode, dann Interpreter |
| `qlang-cli repl` | Tier 3 | Interaktiv, braucht alles |
| `qlang-cli run <graph>` | Executor | Graph-Executor, kein VM |
| `qlang-cli jit <graph>` | LLVM Graph JIT | Anderer Codepath als Script JIT |

## Auto-Detection

Die `exec` Funktion probiert automatisch:

1. **Analysiere den Code** -- enthaelt er Graph-Bloecke?
2. **Ja**: Unified Mode (Tier 3 + Graph Executor)
3. **Nein, nur numerisch**: Tier 1 (LLVM JIT, falls LLVM verfuegbar)
4. **Nein, mit Strings/Arrays**: Tier 2 (Bytecode VM)

## Geschwindigkeitsvergleich

```
Benchmark: Fibonacci(30)

Tier                Zeit         Relativer Speedup
Tier 1 (LLVM JIT)  ~5ms         1x (Basis)
Tier 2 (Bytecode)  ~50ms        10x langsamer
Tier 3 (Interpr.)  ~500ms       100x langsamer

Benchmark: relu(a + b), 1M Elemente

Tier                Zeit
Tier 1 (LLVM JIT)  ~728us       29x schneller als Interpreter
Tier 3 (Interpr.)  ~21.4ms      (Basis)
```

## Graph Execution (Separate Pipeline)

Fuer Graphen gibt es eine separate Ausfuehrungs-Pipeline:

```
Graph ──► Topologische Sortierung ──► Knotenweise Ausfuehrung
                                           │
                                    ┌──────┼──────┐
                                    │      │      │
                              Interpreter  JIT   GPU
```

- **Interpreter**: `executor.rs` -- fuehrt Ops direkt aus
- **LLVM JIT**: `codegen.rs` -- kompiliert Graph zu nativem Code
- **GPU**: `gpu.rs` -- generiert WGSL Compute Shaders

Siehe [[Architecture]] fuer alle Module, [[GPU]] fuer Hardware-Beschleunigung.

#execution #jit #bytecode #vm #interpreter
