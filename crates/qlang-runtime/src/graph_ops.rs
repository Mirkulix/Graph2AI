//! Graph operations callable from VM scripts.
//!
//! This module bridges graph-level tensor operations (matmul, relu, etc.)
//! into the VM, so users can write:
//!
//! ```qlang
//! let a = [1.0, 2.0, 3.0, 4.0]
//! let b = [10.0, 20.0, 30.0, 40.0]
//! let c = add(a, b)          // element-wise add
//! let r = relu([-1.0, 2.0])  // element-wise relu
//! ```
//!
//! Arrays map to 1-D f32 tensors. Tensors (with shape) map to N-D tensors.
//! Results are returned as VM Arrays or Tensors.

use crate::vm::{Value, VmError};
use qlang_core::tensor::{Shape, TensorData};

// ─── Conversion helpers ────────────────────────────────────────────────────

#[allow(dead_code)]
/// Convert a VM Value to a TensorData. Arrays become 1-D vectors;
/// Tensors keep their shape.
fn value_to_tensor(val: &Value) -> Result<(TensorData, Vec<usize>), VmError> {
    match val {
        Value::Array(arr) => {
            let f32_vals: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
            let shape = vec![f32_vals.len()];
            let td = TensorData::from_f32(Shape::vector(f32_vals.len()), &f32_vals);
            Ok((td, shape))
        }
        Value::Tensor(data, shape) => {
            let f32_vals: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let core_shape = match shape.len() {
                0 => Shape::scalar(),
                1 => Shape::vector(shape[0]),
                2 => Shape::matrix(shape[0], shape[1]),
                _ => {
                    // Build a generic shape for higher dims
                    use qlang_core::tensor::Dim;
                    Shape(shape.iter().map(|&d| Dim::Fixed(d)).collect())
                }
            };
            let td = TensorData::from_f32(core_shape, &f32_vals);
            Ok((td, shape.clone()))
        }
        Value::Number(n) => {
            let td = TensorData::from_f32(Shape::scalar(), &[*n as f32]);
            Ok((td, vec![]))
        }
        other => Err(VmError::TypeError(format!(
            "expected array or tensor, got {}",
            other.type_name_static()
        ))),
    }
}

#[allow(dead_code)]
/// Convert TensorData back to a VM Value.
fn tensor_to_value(td: &TensorData, shape: &[usize]) -> Result<Value, VmError> {
    let vals = td
        .as_f32_slice()
        .ok_or_else(|| VmError::RuntimeError("graph op returned non-f32 tensor".into()))?;
    let f64_vals: Vec<f64> = vals.iter().map(|&v| v as f64).collect();

    if shape.len() <= 1 {
        // Return as Array for 0-D and 1-D
        Ok(Value::Array(f64_vals))
    } else {
        Ok(Value::Tensor(f64_vals, shape.to_vec()))
    }
}

#[allow(dead_code)]
/// Get the output shape dims from a TensorData's shape.
fn td_shape_dims(td: &TensorData) -> Vec<usize> {
    use qlang_core::tensor::Dim;
    td.shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => *n,
            Dim::Dynamic => 0,
        })
        .collect()
}

// ─── Arity check helper ────────────────────────────────────────────────────

fn check_arity(_name: &str, args: &[Value], expected: usize) -> Result<(), VmError> {
    if args.len() != expected {
        return Err(VmError::ArityMismatch {
            expected,
            got: args.len(),
        });
    }
    Ok(())
}

// ─── Public dispatch ───────────────────────────────────────────────────────

/// Try to execute a graph op by name. Returns `Ok(Some(value))` if the name
/// matched a graph op, `Ok(None)` if it didn't match (fall through to other
/// dispatch), or `Err` on execution failure.
pub fn try_call_graph_op(name: &str, args: &[Value]) -> Result<Option<Value>, VmError> {
    match name {
        // ── Element-wise binary ops ──
        "add" => {
            check_arity("add", args, 2)?;
            binary_elementwise(&args[0], &args[1], |a, b| a + b, "add")
        }
        "sub" => {
            check_arity("sub", args, 2)?;
            binary_elementwise(&args[0], &args[1], |a, b| a - b, "sub")
        }
        "mul" => {
            check_arity("mul", args, 2)?;
            binary_elementwise(&args[0], &args[1], |a, b| a * b, "mul")
        }
        "div" => {
            check_arity("div", args, 2)?;
            binary_elementwise(&args[0], &args[1], |a, b| a / b, "div")
        }

        // ── Matrix multiplication ──
        "matmul" => {
            check_arity("matmul", args, 2)?;
            graph_matmul(&args[0], &args[1])
        }

        // ── Unary activations ──
        "relu" => {
            check_arity("relu", args, 1)?;
            unary_elementwise(&args[0], |x| x.max(0.0), "relu")
        }
        "sigmoid" => {
            check_arity("sigmoid", args, 1)?;
            unary_elementwise(&args[0], |x| 1.0 / (1.0 + (-x).exp()), "sigmoid")
        }
        "tanh" => {
            check_arity("tanh", args, 1)?;
            unary_elementwise(&args[0], f64::tanh, "tanh")
        }
        "neg" => {
            check_arity("neg", args, 1)?;
            unary_elementwise(&args[0], |x| -x, "neg")
        }

        // ── Softmax ──
        "softmax" => {
            check_arity("softmax", args, 1)?;
            graph_softmax(&args[0])
        }

        // ── Transpose ──
        "transpose" => {
            check_arity("transpose", args, 1)?;
            graph_transpose(&args[0])
        }

        // ── Reductions ──
        "reduce_sum" => {
            check_arity("reduce_sum", args, 1)?;
            graph_reduce(&args[0], |acc, x| acc + x, 0.0, "reduce_sum")
        }
        "reduce_mean" => {
            check_arity("reduce_mean", args, 1)?;
            graph_reduce_mean(&args[0])
        }
        "reduce_max" => {
            check_arity("reduce_max", args, 1)?;
            graph_reduce(&args[0], f64::max, f64::NEG_INFINITY, "reduce_max")
        }

        // ── Tensor creation helpers ──
        "zeros" => {
            // zeros(n) -> array of n zeros
            check_arity("zeros", args, 1)?;
            let n = args[0].as_number()? as usize;
            Ok(Some(Value::Array(vec![0.0; n])))
        }
        "ones" => {
            check_arity("ones", args, 1)?;
            let n = args[0].as_number()? as usize;
            Ok(Some(Value::Array(vec![1.0; n])))
        }

        // ── Tensor with shape: tensor(data_array, shape_array) ──
        "tensor" => {
            check_arity("tensor", args, 2)?;
            let data = args[0].as_array()?.clone();
            let shape_arr = args[1].as_array()?;
            let shape: Vec<usize> = shape_arr.iter().map(|&v| v as usize).collect();
            let expected_numel: usize = shape.iter().product();
            if expected_numel != data.len() {
                return Err(VmError::RuntimeError(format!(
                    "tensor: shape {:?} requires {} elements but got {}",
                    shape, expected_numel, data.len()
                )));
            }
            Ok(Some(Value::Tensor(data, shape)))
        }

        // ── Shape query ──
        "shape" => {
            check_arity("shape", args, 1)?;
            match &args[0] {
                Value::Array(a) => {
                    Ok(Some(Value::Array(vec![a.len() as f64])))
                }
                Value::Tensor(_, s) => {
                    Ok(Some(Value::Array(s.iter().map(|&d| d as f64).collect())))
                }
                other => Err(VmError::TypeError(format!(
                    "shape: expected array or tensor, got {}",
                    other.type_name_static()
                ))),
            }
        }

        _ => Ok(None), // Not a graph op — fall through
    }
}

// ─── Implementation helpers ────────────────────────────────────────────────

fn get_f64_data(val: &Value) -> Result<(Vec<f64>, Vec<usize>), VmError> {
    match val {
        Value::Array(arr) => Ok((arr.clone(), vec![arr.len()])),
        Value::Tensor(data, shape) => Ok((data.clone(), shape.clone())),
        Value::Number(n) => Ok((vec![*n], vec![])),
        other => Err(VmError::TypeError(format!(
            "expected array or tensor, got {}",
            other.type_name_static()
        ))),
    }
}

fn binary_elementwise(
    a: &Value,
    b: &Value,
    op: fn(f64, f64) -> f64,
    name: &str,
) -> Result<Option<Value>, VmError> {
    let (va, sa) = get_f64_data(a)?;
    let (vb, _sb) = get_f64_data(b)?;

    if va.len() != vb.len() {
        return Err(VmError::RuntimeError(format!(
            "{name}: operand length mismatch: {} vs {}",
            va.len(),
            vb.len()
        )));
    }

    let result: Vec<f64> = va.iter().zip(vb.iter()).map(|(&x, &y)| op(x, y)).collect();

    // Preserve shape from first operand; use Tensor if 2D+
    if sa.len() >= 2 {
        Ok(Some(Value::Tensor(result, sa)))
    } else {
        Ok(Some(Value::Array(result)))
    }
}

fn unary_elementwise(
    a: &Value,
    op: fn(f64) -> f64,
    _name: &str,
) -> Result<Option<Value>, VmError> {
    let (va, sa) = get_f64_data(a)?;
    let result: Vec<f64> = va.iter().map(|&x| op(x)).collect();

    if sa.len() >= 2 {
        Ok(Some(Value::Tensor(result, sa)))
    } else {
        Ok(Some(Value::Array(result)))
    }
}

fn graph_matmul(a: &Value, b: &Value) -> Result<Option<Value>, VmError> {
    let (va, sa) = get_f64_data(a)?;
    let (vb, sb) = get_f64_data(b)?;

    if sa.len() != 2 || sb.len() != 2 {
        return Err(VmError::RuntimeError(
            "matmul: both arguments must be 2D tensors (use tensor(data, [rows, cols]))".into(),
        ));
    }

    let (m, k) = (sa[0], sa[1]);
    let (k2, n) = (sb[0], sb[1]);

    if k != k2 {
        return Err(VmError::RuntimeError(format!(
            "matmul: inner dimensions mismatch: [{m},{k}] x [{k2},{n}]"
        )));
    }

    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += va[i * k + p] * vb[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(Some(Value::Tensor(result, vec![m, n])))
}

fn graph_softmax(a: &Value) -> Result<Option<Value>, VmError> {
    let (va, sa) = get_f64_data(a)?;
    let max_val = va.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = va.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let result: Vec<f64> = exps.iter().map(|&e| e / sum).collect();

    if sa.len() >= 2 {
        Ok(Some(Value::Tensor(result, sa)))
    } else {
        Ok(Some(Value::Array(result)))
    }
}

fn graph_transpose(a: &Value) -> Result<Option<Value>, VmError> {
    let (va, sa) = get_f64_data(a)?;
    if sa.len() != 2 {
        return Err(VmError::RuntimeError(
            "transpose: argument must be a 2D tensor".into(),
        ));
    }
    let (m, n) = (sa[0], sa[1]);
    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = va[i * n + j];
        }
    }
    Ok(Some(Value::Tensor(result, vec![n, m])))
}

fn graph_reduce(
    a: &Value,
    op: fn(f64, f64) -> f64,
    init: f64,
    _name: &str,
) -> Result<Option<Value>, VmError> {
    let (va, _) = get_f64_data(a)?;
    let result = va.iter().fold(init, |acc, &x| op(acc, x));
    Ok(Some(Value::Number(result)))
}

fn graph_reduce_mean(a: &Value) -> Result<Option<Value>, VmError> {
    let (va, _) = get_f64_data(a)?;
    if va.is_empty() {
        return Ok(Some(Value::Number(0.0)));
    }
    let sum: f64 = va.iter().sum();
    Ok(Some(Value::Number(sum / va.len() as f64)))
}

#[cfg(test)]
mod tests {
    
    use crate::vm::{run_qlang_script, Value};

    fn run(src: &str) -> (Value, Vec<String>) {
        run_qlang_script(src).expect("script should succeed")
    }

    #[test]
    fn test_add_arrays() {
        let (_, out) = run(r#"
            let a = [1.0, 2.0, 3.0]
            let b = [10.0, 20.0, 30.0]
            let c = add(a, b)
            print(c)
        "#);
        assert_eq!(out, vec!["[11, 22, 33]"]);
    }

    #[test]
    fn test_sub_arrays() {
        let (_, out) = run(r#"
            let a = [10.0, 20.0, 30.0]
            let b = [1.0, 2.0, 3.0]
            let c = sub(a, b)
            print(c)
        "#);
        assert_eq!(out, vec!["[9, 18, 27]"]);
    }

    #[test]
    fn test_mul_arrays() {
        let (_, out) = run(r#"
            let a = [2.0, 3.0, 4.0]
            let b = [5.0, 6.0, 7.0]
            let c = mul(a, b)
            print(c)
        "#);
        assert_eq!(out, vec!["[10, 18, 28]"]);
    }

    #[test]
    fn test_relu_array() {
        let (_, out) = run(r#"
            let a = [-1.0, 0.0, 2.0, -3.0, 5.0]
            let r = relu(a)
            print(r)
        "#);
        assert_eq!(out, vec!["[0, 0, 2, 0, 5]"]);
    }

    #[test]
    fn test_sigmoid_values() {
        let (_, out) = run(r#"
            let a = [0.0]
            let s = sigmoid(a)
            print(s)
        "#);
        // sigmoid(0) = 0.5
        assert_eq!(out, vec!["[0.5]"]);
    }

    #[test]
    fn test_neg_array() {
        let (_, out) = run(r#"
            let a = [1.0, -2.0, 3.0]
            let n = neg(a)
            print(n)
        "#);
        assert_eq!(out, vec!["[-1, 2, -3]"]);
    }

    #[test]
    fn test_softmax() {
        let (_, out) = run(r#"
            let a = [1.0, 1.0, 1.0]
            let s = softmax(a)
            // All equal inputs => all outputs ~0.333...
            // Just check length via reduce_sum ≈ 1.0
            let total = reduce_sum(s)
            print(total)
        "#);
        // Softmax outputs sum to 1.0
        let val: f64 = out[0].parse().unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_2x2() {
        let (_, out) = run(r#"
            let a = tensor([1.0, 2.0, 3.0, 4.0], [2.0, 2.0])
            let b = tensor([5.0, 6.0, 7.0, 8.0], [2.0, 2.0])
            let c = matmul(a, b)
            print(c)
        "#);
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(out, vec!["tensor([19, 22, 43, 50], shape=[2, 2])"]);
    }

    #[test]
    fn test_transpose() {
        let (_, out) = run(r#"
            let a = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0])
            let t = transpose(a)
            print(shape(t))
        "#);
        // shape of transpose([2,3]) should be [3,2]
        assert_eq!(out, vec!["[3, 2]"]);
    }

    #[test]
    fn test_reduce_sum() {
        let (_, out) = run(r#"
            let a = [1.0, 2.0, 3.0, 4.0, 5.0]
            let s = reduce_sum(a)
            print(s)
        "#);
        assert_eq!(out, vec!["15"]);
    }

    #[test]
    fn test_reduce_mean() {
        let (_, out) = run(r#"
            let a = [2.0, 4.0, 6.0, 8.0]
            let m = reduce_mean(a)
            print(m)
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn test_reduce_max() {
        let (_, out) = run(r#"
            let a = [3.0, 1.0, 7.0, 2.0, 5.0]
            let m = reduce_max(a)
            print(m)
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_zeros_and_ones() {
        let (_, out) = run(r#"
            let z = zeros(3.0)
            let o = ones(4.0)
            print(z)
            print(o)
        "#);
        assert_eq!(out, vec!["[0, 0, 0]", "[1, 1, 1, 1]"]);
    }

    #[test]
    fn test_tensor_creation_and_shape() {
        let (_, out) = run(r#"
            let t = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0])
            let s = shape(t)
            print(s)
        "#);
        assert_eq!(out, vec!["[2, 3]"]);
    }

    #[test]
    fn test_chained_graph_ops() {
        let (_, out) = run(r#"
            let x = [-2.0, -1.0, 0.0, 1.0, 2.0]
            let activated = relu(x)
            let doubled = add(activated, activated)
            print(doubled)
        "#);
        // relu: [0,0,0,1,2], doubled: [0,0,0,2,4]
        assert_eq!(out, vec!["[0, 0, 0, 2, 4]"]);
    }

    #[test]
    fn test_graph_op_in_function() {
        let (_, out) = run(r#"
            fn apply_relu_and_sum(x) {
                let activated = relu(x)
                return reduce_sum(activated)
            }
            let data = [-3.0, -1.0, 2.0, 5.0]
            let result = apply_relu_and_sum(data)
            print(result)
        "#);
        // relu: [0,0,2,5], sum = 7
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_matmul_non_square() {
        let (_, out) = run(r#"
            let a = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0])
            let b = tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3.0, 2.0])
            let c = matmul(a, b)
            print(c)
        "#);
        // [[1,2,3],[4,5,6]] x [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        assert_eq!(out, vec!["tensor([4, 5, 10, 11], shape=[2, 2])"]);
    }

    #[test]
    fn test_div_arrays() {
        let (_, out) = run(r#"
            let a = [10.0, 20.0, 30.0]
            let b = [2.0, 4.0, 5.0]
            let c = div(a, b)
            print(c)
        "#);
        assert_eq!(out, vec!["[5, 5, 6]"]);
    }

    #[test]
    fn test_tanh_array() {
        let (_, out) = run(r#"
            let a = [0.0]
            let t = tanh(a)
            print(t)
        "#);
        // tanh(0) = 0
        assert_eq!(out, vec!["[0]"]);
    }
}
