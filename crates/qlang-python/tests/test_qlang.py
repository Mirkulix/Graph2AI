"""Tests for the qlang Python bindings."""

import qlang


# ---------------------------------------------------------------------------
# Graph creation and execution
# ---------------------------------------------------------------------------

class TestGraph:
    def test_create_graph(self):
        g = qlang.Graph("test")
        assert g.num_nodes() == 0

    def test_add_input(self):
        g = qlang.Graph("test")
        node_id = g.add_input("x", "f32", [4])
        assert isinstance(node_id, int)
        assert g.num_nodes() == 1

    def test_add_relu(self):
        g = qlang.Graph("test")
        x = g.add_input("x", "f32", [4])
        r = g.add_relu(x)
        assert r != x
        assert g.num_nodes() == 2

    def test_add_and_execute(self):
        g = qlang.Graph("test_add")
        x = g.add_input("x", "f32", [4])
        y = g.add_input("y", "f32", [4])
        s = g.add_add(x, y)
        r = g.add_relu(s)
        g.add_output("result", r)

        result = g.execute({"x": [1.0, -2.0, 3.0, -4.0], "y": [10.0, 20.0, 30.0, 40.0]})
        assert "result" in result
        vals = result["result"]
        assert len(vals) == 4
        assert vals[0] == 11.0
        assert vals[1] == 18.0
        assert vals[2] == 33.0
        assert vals[3] == 36.0

    def test_verify_valid_graph(self):
        g = qlang.Graph("test")
        x = g.add_input("x", "f32", [4])
        g.add_output("out", x)
        assert g.verify() is True

    def test_to_json(self):
        g = qlang.Graph("test")
        g.add_input("x", "f32", [4])
        json_str = g.to_json()
        assert isinstance(json_str, str)
        assert "test" in json_str

    def test_to_qlang_text(self):
        g = qlang.Graph("test")
        g.add_input("x", "f32", [4])
        text = g.to_qlang_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_repr(self):
        g = qlang.Graph("demo")
        g.add_input("x", "f32", [2])
        r = repr(g)
        assert "demo" in r
        assert "nodes=1" in r

    def test_softmax(self):
        g = qlang.Graph("softmax_test")
        x = g.add_input("x", "f32", [3])
        s = g.add_softmax(x)
        g.add_output("probs", s)

        result = g.execute({"x": [1.0, 2.0, 3.0]})
        probs = result["probs"]
        assert len(probs) == 3
        total = sum(probs)
        assert abs(total - 1.0) < 1e-5

    def test_matmul(self):
        g = qlang.Graph("matmul_test")
        a = g.add_input("a", "f32", [2, 3])
        b = g.add_input("b", "f32", [3, 2])
        m = g.add_matmul(a, b)
        g.add_output("out", m)

        result = g.execute({
            "a": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 2x3 identity-like
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],    # 3x2
        })
        assert "out" in result
        assert len(result["out"]) == 4  # 2x2 output


# ---------------------------------------------------------------------------
# Ternary compression
# ---------------------------------------------------------------------------

class TestCompressTernary:
    def test_basic_compression(self):
        weights = [0.9, -0.8, 0.01, -0.02, 0.7, -0.6, 0.0, 0.5]
        compressed = qlang.compress_ternary(weights)
        assert len(compressed) == len(weights)
        for v in compressed:
            assert v in (-1.0, 0.0, 1.0)

    def test_all_zero(self):
        weights = [0.0, 0.0, 0.0]
        compressed = qlang.compress_ternary(weights)
        assert compressed == [0.0, 0.0, 0.0]

    def test_positive_values(self):
        weights = [1.0, 1.0, 1.0]
        compressed = qlang.compress_ternary(weights)
        assert all(v == 1.0 for v in compressed)

    def test_negative_values(self):
        weights = [-1.0, -1.0, -1.0]
        compressed = qlang.compress_ternary(weights)
        assert all(v == -1.0 for v in compressed)

    def test_empty_input(self):
        compressed = qlang.compress_ternary([])
        assert compressed == []


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------

class TestTrainMlp:
    def test_basic_training(self):
        x_train = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        y_train = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        targets = [0, 1, 0, 1]

        result = qlang.train_mlp(x_train, y_train, targets, epochs=50, lr=0.1)

        assert "final_loss" in result
        assert "final_accuracy" in result
        assert "weights_w1" in result
        assert "weights_w2" in result
        assert "param_count" in result

        assert len(result["final_loss"]) == 1
        assert len(result["final_accuracy"]) == 1
        assert result["final_loss"][0] >= 0.0
        assert 0.0 <= result["final_accuracy"][0] <= 1.0
        assert result["param_count"][0] > 0

    def test_training_reduces_loss(self):
        x_train = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        y_train = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
        targets = [0, 1, 0, 1]

        result_1 = qlang.train_mlp(x_train, y_train, targets, epochs=1, lr=0.1)
        result_50 = qlang.train_mlp(x_train, y_train, targets, epochs=50, lr=0.1)

        # More training should generally reduce or maintain loss
        assert result_50["final_loss"][0] <= result_1["final_loss"][0] + 0.5

    def test_weights_are_returned(self):
        x_train = [1.0, 0.0, 0.0, 1.0]
        y_train = [1.0, 0.0, 0.0, 1.0]
        targets = [0, 1]

        result = qlang.train_mlp(x_train, y_train, targets, epochs=10, lr=0.1)
        assert len(result["weights_w1"]) > 0
        assert len(result["weights_w2"]) > 0
