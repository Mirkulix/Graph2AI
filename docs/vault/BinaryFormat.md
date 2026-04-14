# BinaryFormat

QLANG verwendet das QLBG (QLANG Binary Graph) Format fuer kompakte Graph-Serialisierung. 3.5x kleiner und 244x schneller als JSON.

## QLBG Wire Format

```
Offset  Groesse   Feld            Beschreibung
0       4         magic           0x51 0x4C 0x42 0x47 ("QLBG")
4       2         version         Format-Version (u16 LE)
6       var       graph_id        Length-prefixed String
var     var       graph_version   Length-prefixed String
var     4         node_count      u32 LE
var     var       nodes           Je: id + op_tag + input_types + output_types
var     4         edge_count      u32 LE
var     var       edges           Je: from_node + from_port + to_node + to_port + type
last    32        hash            SHA-256 Content Hash von allem davor
```

## Op Tags

Operationen werden als einzelne Bytes (0-41) kodiert:

| Tag | Op | Tag | Op |
|-----|-----|-----|-----|
| 0 | Input | 21 | Evolve |
| 1 | Output | 22 | Measure |
| 2 | Constant | 23 | Entangle |
| 3 | Add | 24 | Collapse |
| 4 | Sub | 25 | Entropy |
| 5 | Mul | 26 | ToTernary |
| 6 | Div | 27 | ToLowRank |
| 7 | Neg | 28 | ToSparse |
| 8 | MatMul | 29 | FisherMetric |
| 9 | Transpose | 30 | Project |
| 10 | Reshape | 31 | LayerNorm |
| 11 | Slice | 32 | Attention |
| 12 | Concat | 33 | Embedding |
| 13 | ReduceSum | 34 | Residual |
| 14 | ReduceMean | 35 | Dropout |
| 15 | ReduceMax | 36 | SubGraph |
| 16 | Relu | 37 | OllamaGenerate |
| 17 | Sigmoid | 38 | OllamaChat |
| 18 | Tanh | 39 | Cond |
| 19 | Softmax | 40 | Scan |
| 20 | Superpose | 41 | GELU |

## Tensor Wire Format

Tensoren werden mit minimalem Header uebertragen:

```
[dtype: u8] [ndims: u8] [shape: ndims * u32 LE] [data: raw bytes]
```

Zero-copy auf Empfaengerseite wenn Dtypes uebereinstimmen.

## Groessenvergleich

```
Format          MNIST Modell    Faktor
JSON            ~50 KB          1x (Basis)
QLBG Binary     ~3.2 KB         15.6x kleiner
QLMS Binary     ~3 KB           16.7x kleiner
```

### Warum so viel kleiner?

1. **Keine Feld-Namen**: JSON wiederholt `"op":`, `"dtype":`, etc. -- QLBG nicht
2. **Keine Dezimal-Strings**: Ein f32 ist 4 Bytes, nicht 8-15 ASCII-Zeichen
3. **Op als 1 Byte**: Statt `"MatMul"` (6 Bytes) nur `0x08` (1 Byte)
4. **Kein Escaping**: Keine JSON-Escape-Sequenzen, keine Unicode-Edge-Cases

### Performance-Vergleich

```
Operation           JSON            Binary          Speedup
Serialize           ~12ms           ~0.05ms         244x
Deserialize         ~8ms            ~0.03ms         267x
Hash Check          n/a             ~0.01ms         (nur binary)
```

## Content Hashing

Jeder binaer-encodierte Graph endet mit einem SHA-256 Content Hash:

1. Alles vor dem Hash wird gehasht
2. Hash wird an den Stream angehaengt
3. Beim Dekodieren: Hash neu berechnen und vergleichen
4. Bei Mismatch: `BinaryError::HashMismatch` -- Daten korrumpiert

Siehe [[Crypto]] fuer Details zur SHA-256 Implementierung.

## CLI

```bash
# Graph zu Binaer encodieren
qlang-cli binary encode model.qlg.json
# -> erzeugt model.qlb

# Binaer zu JSON dekodieren
qlang-cli binary decode model.qlb
# -> gibt JSON auf stdout aus
```

## Fehlerbehandlung

```rust
pub enum BinaryError {
    TooShort,                 // Daten zu kurz
    InvalidMagic,             // Keine QLBG Magic Bytes
    UnsupportedVersion(u16),  // Unbekannte Version
    UnexpectedEof(usize),     // Unerwartetes Datenende
    InvalidOpTag(u8),         // Unbekannter Op-Code
    InvalidDtypeTag(u8),      // Unbekannter Dtype
    InvalidUtf8,              // Ungueltige UTF-8 Zeichenkette
    HashMismatch,             // Daten korrumpiert
}
```

## Zusammenhang mit QLMS

QLBG ist das Format fuer einzelne Graphen. QLMS ([[Protocol]]) ist das Envelope-Format fuer Nachrichten zwischen [[Agents]]:

```
QLMS Envelope
├── Magic: "QLMS"
├── Version, Flags
├── Signatur (optional)
├── Public Key (optional)
├── Payload Hash
├── Message Count
└── Payload (enthaelt GraphMessages, die QLBG-encodierte Graphen enthalten koennen)
```

Siehe [[Protocol]] fuer das vollstaendige QLMS Wire Format.

#binary #format #serialization #performance
