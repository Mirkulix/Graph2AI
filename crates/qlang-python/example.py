"""Example usage of the QLANG Python bindings."""

import qlang

# ---------------------------------------------------------------
# 1. Build and execute a simple computation graph
# ---------------------------------------------------------------

g = qlang.Graph("demo")

# Add input nodes
x = g.add_input("x", "f32", [4])
y = g.add_input("y", "f32", [4])

# Element-wise add, then ReLU
s = g.add_add(x, y)
r = g.add_relu(s)

# Mark output
g.add_output("result", r)

# Inspect
print(g)
print(f"Nodes: {g.num_nodes()}")
print(f"Valid: {g.verify()}")

# Execute
result = g.execute({"x": [1, -2, 3, -4], "y": [10, 20, 30, 40]})
print("Result:", result)  # {"result": [11.0, 18.0, 33.0, 36.0]}

# Serialization
print("\nJSON (first 200 chars):")
print(g.to_json()[:200], "...")

print("\nQLANG text:")
print(g.to_qlang_text())

# ---------------------------------------------------------------
# 2. IGQK ternary compression
# ---------------------------------------------------------------

weights = [0.9, -0.8, 0.01, -0.02, 0.7, -0.6, 0.0, 0.5]
compressed = qlang.compress_ternary(weights)
print("Original weights:", weights)
print("Ternary weights: ", compressed)

# ---------------------------------------------------------------
# 3. Train a tiny MLP
# ---------------------------------------------------------------

# 4 samples, 2 features each -> 2 classes
x_train = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
y_train = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]  # one-hot-ish (2 classes)
targets = [0, 1, 0, 1]

result = qlang.train_mlp(x_train, y_train, targets, epochs=50, lr=0.1)
print(f"\nTraining result:")
print(f"  Final loss:     {result['final_loss'][0]:.4f}")
print(f"  Final accuracy: {result['final_accuracy'][0]:.2%}")
print(f"  Param count:    {int(result['param_count'][0])}")
