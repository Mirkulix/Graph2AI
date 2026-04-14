# Language

QLANG hat zwei Schichten: die **Graph-Sprache** (primaer, fuer AI Agents) und die **VM-Sprache** (Scripting, fuer Menschen).

## Graph-Sprache (.qlang)

Der Graph ist die Quelle der Wahrheit. Die Text-Syntax ist eine menschenlesbare Ansicht.

### Grundstruktur

```qlang
graph my_model {
  input x: f32[784]
  input W1: f32[784, 128]
  input W2: f32[128, 10]

  node hidden = matmul(x, W1)
  node activated = relu(hidden)
  node logits = matmul(activated, W2)
  node predictions = softmax(logits)

  output result = predictions
}
```

### Typen

Alle Werte sind Tensoren:

| Typ | Bedeutung |
|-----|-----------|
| `f16` | 16-bit Float |
| `f32` | 32-bit Float (Standard) |
| `f64` | 64-bit Float |
| `i8` | 8-bit Integer |
| `i16` | 16-bit Integer |
| `i32` | 32-bit Integer |
| `i64` | 64-bit Integer |
| `bool` | Boolean |
| `ternary` | {-1, 0, +1} (fuer [[IGQK]] Kompression) |

### Shapes

```
f32[]           # Skalar (0-dim Tensor)
f32[784]        # Vektor
f32[784, 128]   # Matrix
f32[1, 784]     # Batch mit 1 Vektor
f32[?, 784]     # Dynamische Batch-Dimension
```

### Operationen

40+ Operationen, organisiert nach Kategorie:

**Tensor-Ops:** `add`, `sub`, `mul`, `div`, `neg`, `matmul`, `transpose`, `reshape`, `slice`, `concat`, `reduce_sum`, `reduce_mean`, `reduce_max`

**Aktivierungen:** `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`

**Quantum/IGQK:** `superpose`, `evolve`, `measure`, `entangle`, `collapse`, `entropy`, `to_ternary`, `to_low_rank`, `to_sparse`, `fisher_metric`, `project`

**Transformer:** `layer_norm`, `attention`, `embedding`, `residual`, `dropout`

**LLM:** `ollama_generate`, `ollama_chat` (siehe [[Ollama]])

**Kontrolle:** `cond`, `scan`, `sub_graph`

### Proof-Annotationen

Kompressions-Operationen koennen formale Proof-Referenzen tragen:

```qlang
node compressed = to_ternary(W1) @proof theorem_5_2
```

Verknuepft mit [[IGQK]] Theorem 5.2 (Kompressionsschranke).

## VM-Sprache (Scripting)

Die VM fuegt General-Purpose-Programmierung ueber Graphen hinzu. Ausgefuehrt via [[CLI]] `qlang-cli exec`.

### Variablen

```qlang
let x = 42
let name = "QLANG"
let arr = [1, 2, 3, 4, 5]
let flag = true
```

### Statische Typ-Annotationen

```qlang
let x: int = 5
let y: float = 3.14
let name: string = "QLANG"
```

Typ-Annotationen sind optional -- der Typ wird automatisch inferiert.

### Kontrollfluss

```qlang
if x > 10 {
  print("gross")
} else {
  print("klein")
}

for i in range(0, 10) {
  print(i)
}

for i in 0..10 {
  print(i)
}

while x > 0 {
  x = x - 1
}
```

### Funktionen

```qlang
fn factorial(n) {
  if n <= 1 {
    return 1
  }
  return n * factorial(n - 1)
}

fn add(a, b) {
  return a + b
}
```

### Datenstrukturen

```qlang
// Arrays
let a = [1, 2, 3]
let first = a[0]

// Dicts
let point = { x: 1.0, y: 2.0 }
let px = point["x"]

// Verschachtelt
let model = {
  name: "classifier",
  layers: [784, 128, 10],
  lr: 0.01
}
```

### Import-System

```qlang
import "math_utils.qlang"
import "models/classifier.qlang"
```

### Alle Operatoren

#### Arithmetik
| Operator | Bedeutung | Beispiel |
|----------|-----------|----------|
| `+` | Addition | `10 + 3` -> `13` |
| `-` | Subtraktion | `10 - 3` -> `7` |
| `*` | Multiplikation | `10 * 3` -> `30` |
| `/` | Division | `10 / 3` -> `3.333...` |
| `%` | Modulo | `10 % 3` -> `1` |
| `**` | Potenz | `2 ** 10` -> `1024` |
| `-x` | Negation | `-5` |

#### Vergleich
| Operator | Bedeutung |
|----------|-----------|
| `==` | Gleich |
| `!=` | Ungleich |
| `<` | Kleiner |
| `>` | Groesser |
| `<=` | Kleiner oder gleich |
| `>=` | Groesser oder gleich |

#### Logik
| Operator | Bedeutung |
|----------|-----------|
| `and` / `&&` | Logisches UND |
| `or` / `\|\|` | Logisches ODER |
| `not` / `!` | Logische Negation |

#### Bitweise
| Operator | Bedeutung | Beispiel |
|----------|-----------|----------|
| `&` | Bitweises UND | `5 & 3` -> `1` |
| `\|` | Bitweises ODER | `5 \| 3` -> `7` |
| `^` | Bitweises XOR | `5 ^ 3` -> `6` |
| `~` | Bitweises NOT | `~0` -> `-1` |
| `<<` | Links-Shift | `1 << 4` -> `16` |
| `>>` | Rechts-Shift | `16 >> 2` -> `4` |

#### Compound Assignment
| Operator | Aequivalent |
|----------|-------------|
| `+=` | `x = x + y` |
| `-=` | `x = x - y` |
| `*=` | `x = x * y` |
| `/=` | `x = x / y` |
| `%=` | `x = x % y` |

### Stdlib (53 Funktionen)

#### Math (12)
`abs`, `sqrt`, `pow`, `min`, `max`, `floor`, `ceil`, `round`, `sin`, `cos`, `log`, `exp`

#### Arrays (11)
`len`, `sum`, `mean`, `max_val`, `min_val`, `sort`, `reverse`, `range`, `zeros`, `ones`, `linspace`

#### Strings (10)
`str`, `concat`, `split`, `trim`, `contains`, `replace`, `to_upper`, `to_lower`, `starts_with`, `ends_with`

#### I/O (4)
`print`, `println`, `read_file`, `write_file`

#### Typen (6)
`type_of`, `is_number`, `is_array`, `is_string`, `to_number`, `to_string`

#### Tensor (5)
`shape`, `reshape`, `transpose`, `dot`, `matmul`

#### Random (3)
`random`, `random_range`, `random_array`

#### Zeit (1)
`clock`

#### Weitere
`push` (Array-Element anhaengen)

### Graph-Ops als VM-Funktionen

18 Graph-Operationen sind als VM-Funktionen aufrufbar:

```qlang
let result = matmul(a, b)
let activated = relu(hidden)
let probs = softmax(logits)
```

### Unified Mode

Eine `.qlang` Datei kann BEIDES: Scripting und ML-Graphen.

```qlang
let lr = 0.01

graph classifier {
  input x: f32[4]
  node r = relu(x)
  output y = r
}

let result = run_graph("classifier", {"x": [1.0, -2.0, 3.0, -4.0]})
print(result)
```

Siehe [[Execution]] fuer die verschiedenen Ausfuehrungs-Modi.

#language #core #syntax
