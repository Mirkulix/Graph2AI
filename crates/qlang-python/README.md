# qlang

**QLANG** is a graph-based AI programming language with built-in IGQK
(Information-Geometric Quantum Compression) support. This package provides
Python bindings powered by PyO3 and maturin.

## Installation

```bash
pip install qlang
```

## Quick Start

### Build and Execute a Computation Graph

```python
import qlang

# Create a graph
g = qlang.Graph("my_model")

# Add inputs (name, dtype, shape)
x = g.add_input("x", "f32", [4])
y = g.add_input("y", "f32", [4])

# Build the computation: relu(x + y)
s = g.add_add(x, y)
r = g.add_relu(s)
g.add_output("result", r)

# Execute
result = g.execute({"x": [1, -2, 3, -4], "y": [10, 20, 30, 40]})
print(result)  # {"result": [11.0, 18.0, 33.0, 36.0]}
```

### IGQK Ternary Compression

Compress floating-point weights to ternary values {-1, 0, +1} using
threshold-based projection (IGQK Theorem 5.2):

```python
import qlang

weights = [0.9, -0.8, 0.01, -0.02, 0.7, -0.6, 0.0, 0.5]
compressed = qlang.compress_ternary(weights)
print(compressed)  # [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 1.0]
```

### Train a Simple MLP

```python
import qlang

# 4 samples, 2 features -> 2 classes
x_train = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
y_train = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
targets = [0, 1, 0, 1]

result = qlang.train_mlp(x_train, y_train, targets, epochs=50, lr=0.1)
print(f"Loss: {result['final_loss'][0]:.4f}")
print(f"Accuracy: {result['final_accuracy'][0]:.2%}")
```

## API Reference

### `qlang.Graph(name: str)`

Create a computation graph.

**Methods:**

| Method | Description |
|--------|-------------|
| `add_input(name, dtype, shape)` | Add an input node. Returns node ID. |
| `add_matmul(a, b)` | Matrix multiplication of two 2-D nodes. |
| `add_relu(input)` | ReLU activation. |
| `add_add(a, b)` | Element-wise addition. |
| `add_softmax(input)` | Softmax (axis 0). |
| `add_to_ternary(input)` | IGQK ternary compression to {-1, 0, +1}. |
| `add_output(name, source)` | Mark a node as graph output. |
| `execute(inputs)` | Run the graph. `inputs`: dict of name to list of floats. |
| `to_json()` | Serialize graph to JSON string. |
| `to_qlang_text()` | Human-readable text representation. |
| `num_nodes()` | Number of nodes in the graph. |
| `verify()` | Validate graph structure and types. |

### `qlang.compress_ternary(weights: list[float]) -> list[float]`

Compress weights to ternary {-1.0, 0.0, +1.0} using threshold = mean(|w|) * 0.7.

### `qlang.train_mlp(x, y, targets, epochs, lr) -> dict`

Train a simple MLP with numerical gradient descent. Returns a dict with keys:
`final_loss`, `final_accuracy`, `weights_w1`, `weights_w2`, `param_count`.

## Supported Dtypes

`f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `bool`, `ternary`

## License

MIT
