//! Benchmark module — measures performance across execution backends.
//!
//! Provides a standard suite of benchmarks for element-wise ops, matrix
//! multiplication, MLP forward passes, and IGQK ternary compression.
//! All timing uses `std::time::Instant` (no external deps).
//!
//! This module lives in qlang-runtime and does NOT use inkwell/LLVM.
//! JIT timing fields are provided for future use by qlang-compile.

use std::collections::HashMap;
use std::time::Instant;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

use crate::executor;
use crate::training::MlpWeights;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable name of the benchmark.
    pub name: String,
    /// Input size (total number of elements processed).
    pub input_size: usize,
    /// Interpreter execution time in nanoseconds.
    pub interpreter_ns: u64,
    /// JIT execution time in nanoseconds (0 when JIT is unavailable).
    pub jit_ns: u64,
    /// Speedup ratio: interpreter_ns / jit_ns (0.0 when JIT is unavailable).
    pub speedup: f64,
    /// Throughput in GFLOPS (based on interpreter timing).
    pub throughput_gflops: f64,
}

/// A configurable benchmark suite that runs benchmarks across different configs.
pub struct BenchmarkSuite {
    /// Sizes to test for element-wise benchmarks.
    pub element_wise_sizes: Vec<usize>,
    /// (m, k, n) triples for matrix multiplication benchmarks.
    pub matmul_sizes: Vec<(usize, usize, usize)>,
    /// (input_dim, hidden_dim, output_dim, batch_size) for MLP benchmarks.
    pub mlp_configs: Vec<(usize, usize, usize, usize)>,
    /// Sizes for ternary compression benchmarks.
    pub ternary_sizes: Vec<usize>,
    /// Number of warmup iterations before timing.
    pub warmup_iters: usize,
    /// Number of timed iterations (result is the median).
    pub bench_iters: usize,
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self {
            element_wise_sizes: vec![256, 1024, 4096],
            matmul_sizes: vec![(32, 32, 32), (64, 64, 64), (128, 128, 128)],
            mlp_configs: vec![
                (64, 32, 10, 8),
                (128, 64, 10, 16),
            ],
            ternary_sizes: vec![256, 1024, 4096],
            warmup_iters: 2,
            bench_iters: 5,
        }
    }
}

impl BenchmarkSuite {
    /// Run the full suite of benchmarks and return all results.
    pub fn run(&self) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();

        // Element-wise benchmarks
        let ops = ["Add", "Mul", "Relu", "Sigmoid"];
        for &n in &self.element_wise_sizes {
            for op_name in &ops {
                results.push(self.run_with_iters(|| bench_element_wise(n, op_name)));
            }
        }

        // Matrix multiplication
        for &(m, k, n) in &self.matmul_sizes {
            results.push(self.run_with_iters(|| bench_matmul(m, k, n)));
        }

        // MLP forward
        for &(i, h, o, b) in &self.mlp_configs {
            results.push(self.run_with_iters(|| bench_mlp_forward(i, h, o, b)));
        }

        // Ternary compression
        for &n in &self.ternary_sizes {
            results.push(self.run_with_iters(|| bench_ternary_compression(n)));
        }

        results
    }

    /// Run a benchmark function multiple times, take the median timing.
    fn run_with_iters(&self, f: impl Fn() -> BenchmarkResult) -> BenchmarkResult {
        // Warmup
        for _ in 0..self.warmup_iters {
            let _ = f();
        }

        // Collect timings
        let mut timings: Vec<BenchmarkResult> = (0..self.bench_iters).map(|_| f()).collect();
        timings.sort_by_key(|r| r.interpreter_ns);

        // Return median
        timings.swap_remove(timings.len() / 2)
    }
}

// ---------------------------------------------------------------------------
// Individual benchmarks
// ---------------------------------------------------------------------------

/// Benchmark an element-wise operation (Add, Mul, Relu, Sigmoid) on vectors of size `n`.
pub fn bench_element_wise(n: usize, op_name: &str) -> BenchmarkResult {
    let vtype = TensorType::f32_vector(n);

    // Build the graph based on the operation
    let (graph, inputs) = match op_name {
        "Add" | "Mul" => {
            let op = if op_name == "Add" { Op::Add } else { Op::Mul };
            let mut g = Graph::new(format!("bench_{op_name}_{n}"));
            let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![vtype.clone()]);
            let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![vtype.clone()]);
            let op_node = g.add_node(op, vec![vtype.clone(), vtype.clone()], vec![vtype.clone()]);
            let out = g.add_node(Op::Output { name: "y".into() }, vec![vtype.clone()], vec![]);
            g.add_edge(a, 0, op_node, 0, vtype.clone());
            g.add_edge(b, 0, op_node, 1, vtype.clone());
            g.add_edge(op_node, 0, out, 0, vtype.clone());

            let data_a = make_f32_vector(n, 1.0);
            let data_b = make_f32_vector(n, 2.0);
            let mut map = HashMap::new();
            map.insert("a".to_string(), data_a);
            map.insert("b".to_string(), data_b);
            (g, map)
        }
        "Relu" | "Sigmoid" => {
            let op = if op_name == "Relu" { Op::Relu } else { Op::Sigmoid };
            let mut g = Graph::new(format!("bench_{op_name}_{n}"));
            let a = g.add_node(Op::Input { name: "x".into() }, vec![], vec![vtype.clone()]);
            let op_node = g.add_node(op, vec![vtype.clone()], vec![vtype.clone()]);
            let out = g.add_node(Op::Output { name: "y".into() }, vec![vtype.clone()], vec![]);
            g.add_edge(a, 0, op_node, 0, vtype.clone());
            g.add_edge(op_node, 0, out, 0, vtype.clone());

            let data = make_f32_vector_alternating(n);
            let mut map = HashMap::new();
            map.insert("x".to_string(), data);
            (g, map)
        }
        _ => {
            return BenchmarkResult {
                name: format!("unsupported_op_{op_name}"),
                input_size: n,
                interpreter_ns: 0,
                jit_ns: 0,
                speedup: 0.0,
                throughput_gflops: 0.0,
            };
        }
    };

    // Time the interpreter execution
    let start = Instant::now();
    let _ = executor::execute(&graph, inputs);
    let elapsed = start.elapsed();
    let interpreter_ns = elapsed.as_nanos() as u64;

    // FLOPs: 1 per element for Add/Mul/Relu, ~4 for Sigmoid
    let flops_per_elem: u64 = match op_name {
        "Sigmoid" => 4,
        _ => 1,
    };
    let total_flops = n as u64 * flops_per_elem;
    let throughput_gflops = if interpreter_ns > 0 {
        total_flops as f64 / interpreter_ns as f64 // flops / ns = GFLOPS
    } else {
        0.0
    };

    BenchmarkResult {
        name: format!("{op_name}[{n}]"),
        input_size: n,
        interpreter_ns,
        jit_ns: 0,
        speedup: 0.0,
        throughput_gflops,
    }
}

/// Benchmark matrix multiplication: A[m,k] x B[k,n].
/// Interpreter only (JIT matmul lives in qlang-compile).
pub fn bench_matmul(m: usize, k: usize, n: usize) -> BenchmarkResult {
    let atype = TensorType::f32_matrix(m, k);
    let btype = TensorType::f32_matrix(k, n);
    let ctype = TensorType::f32_matrix(m, n);

    let mut g = Graph::new(format!("bench_matmul_{m}x{k}x{n}"));
    let na = g.add_node(Op::Input { name: "a".into() }, vec![], vec![atype.clone()]);
    let nb = g.add_node(Op::Input { name: "b".into() }, vec![], vec![btype.clone()]);
    let mm = g.add_node(Op::MatMul, vec![atype.clone(), btype.clone()], vec![ctype.clone()]);
    let out = g.add_node(Op::Output { name: "y".into() }, vec![ctype.clone()], vec![]);
    g.add_edge(na, 0, mm, 0, atype.clone());
    g.add_edge(nb, 0, mm, 1, btype.clone());
    g.add_edge(mm, 0, out, 0, ctype.clone());

    let data_a = make_f32_vector(m * k, 0.5);
    let data_b = make_f32_vector(k * n, 0.5);
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), TensorData::from_f32(Shape::matrix(m, k), &data_a.as_f32_slice().unwrap()));
    inputs.insert("b".to_string(), TensorData::from_f32(Shape::matrix(k, n), &data_b.as_f32_slice().unwrap()));

    let start = Instant::now();
    let _ = executor::execute(&g, inputs);
    let elapsed = start.elapsed();
    let interpreter_ns = elapsed.as_nanos() as u64;

    // FLOPs for matmul: 2*m*n*k (multiply-add)
    let total_flops = 2 * m as u64 * n as u64 * k as u64;
    let throughput_gflops = if interpreter_ns > 0 {
        total_flops as f64 / interpreter_ns as f64
    } else {
        0.0
    };

    BenchmarkResult {
        name: format!("MatMul[{m}x{k}x{n}]"),
        input_size: m * k + k * n,
        interpreter_ns,
        jit_ns: 0,
        speedup: 0.0,
        throughput_gflops,
    }
}

/// Benchmark a full MLP forward pass (using the training module's MlpWeights).
pub fn bench_mlp_forward(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    batch_size: usize,
) -> BenchmarkResult {
    let mlp = MlpWeights::new(input_dim, hidden_dim, output_dim);

    // Create input batch
    let input: Vec<f32> = (0..batch_size * input_dim)
        .map(|i| (i as f32 * 0.1).sin())
        .collect();

    let start = Instant::now();
    let _ = mlp.forward(&input);
    let elapsed = start.elapsed();
    let interpreter_ns = elapsed.as_nanos() as u64;

    // FLOPs estimate: batch*(input*hidden + hidden + hidden*output + output + softmax)
    // ~= batch * (2*input*hidden + 2*hidden*output)
    let total_flops = batch_size as u64
        * (2 * input_dim as u64 * hidden_dim as u64
            + 2 * hidden_dim as u64 * output_dim as u64);
    let throughput_gflops = if interpreter_ns > 0 {
        total_flops as f64 / interpreter_ns as f64
    } else {
        0.0
    };

    BenchmarkResult {
        name: format!("MLP[{input_dim}->{hidden_dim}->{output_dim} x{batch_size}]"),
        input_size: batch_size * input_dim,
        interpreter_ns,
        jit_ns: 0,
        speedup: 0.0,
        throughput_gflops,
    }
}

/// Benchmark IGQK ternary compression on a vector of `n` weights.
pub fn bench_ternary_compression(n: usize) -> BenchmarkResult {
    let vtype = TensorType::f32_vector(n);

    let mut g = Graph::new(format!("bench_ternary_{n}"));
    let inp = g.add_node(Op::Input { name: "w".into() }, vec![], vec![vtype.clone()]);
    let tern = g.add_node(
        Op::ToTernary,
        vec![vtype.clone()],
        vec![TensorType::new(Dtype::Ternary, Shape::vector(n))],
    );
    let out = g.add_node(
        Op::Output { name: "compressed".into() },
        vec![TensorType::new(Dtype::Ternary, Shape::vector(n))],
        vec![],
    );
    g.add_edge(inp, 0, tern, 0, vtype.clone());
    g.add_edge(tern, 0, out, 0, TensorType::new(Dtype::Ternary, Shape::vector(n)));

    let weights = make_f32_vector_alternating(n);
    let mut inputs = HashMap::new();
    inputs.insert("w".to_string(), weights);

    let start = Instant::now();
    let _ = executor::execute(&g, inputs);
    let elapsed = start.elapsed();
    let interpreter_ns = elapsed.as_nanos() as u64;

    // FLOPs: ~3n (abs + compare + threshold computation)
    let total_flops = 3 * n as u64;
    let throughput_gflops = if interpreter_ns > 0 {
        total_flops as f64 / interpreter_ns as f64
    } else {
        0.0
    };

    BenchmarkResult {
        name: format!("Ternary[{n}]"),
        input_size: n,
        interpreter_ns,
        jit_ns: 0,
        speedup: 0.0,
        throughput_gflops,
    }
}

/// Run a standard suite of benchmarks with default configuration.
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    BenchmarkSuite::default().run()
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format benchmark results as an aligned ASCII table.
pub fn format_benchmark_table(results: &[BenchmarkResult]) -> String {
    if results.is_empty() {
        return String::from("(no benchmark results)");
    }

    // Column headers
    let headers = [
        "Benchmark",
        "Input Size",
        "Interp (us)",
        "JIT (us)",
        "Speedup",
        "GFLOPS",
    ];

    // Compute column widths from headers first
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

    // Format each row and track max widths
    let rows: Vec<[String; 6]> = results
        .iter()
        .map(|r| {
            let row = [
                r.name.clone(),
                format!("{}", r.input_size),
                format!("{:.1}", r.interpreter_ns as f64 / 1_000.0),
                if r.jit_ns > 0 {
                    format!("{:.1}", r.jit_ns as f64 / 1_000.0)
                } else {
                    "N/A".to_string()
                },
                if r.speedup > 0.0 {
                    format!("{:.2}x", r.speedup)
                } else {
                    "N/A".to_string()
                },
                format!("{:.4}", r.throughput_gflops),
            ];
            for (i, cell) in row.iter().enumerate() {
                if cell.len() > widths[i] {
                    widths[i] = cell.len();
                }
            }
            row
        })
        .collect();

    let mut out = String::new();

    // Header
    let header_line: String = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!("{:>width$}", h, width = widths[i]))
        .collect::<Vec<_>>()
        .join("  |  ");
    out.push_str(&header_line);
    out.push('\n');

    // Separator
    let sep: String = widths
        .iter()
        .map(|&w| "-".repeat(w))
        .collect::<Vec<_>>()
        .join("--+--");
    out.push_str(&sep);
    out.push('\n');

    // Rows
    for row in &rows {
        let line: String = row
            .iter()
            .enumerate()
            .map(|(i, cell)| format!("{:>width$}", cell, width = widths[i]))
            .collect::<Vec<_>>()
            .join("  |  ");
        out.push_str(&line);
        out.push('\n');
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an f32 vector tensor filled with a constant value.
fn make_f32_vector(n: usize, value: f32) -> TensorData {
    let data = vec![value; n];
    TensorData::from_f32(Shape::vector(n), &data)
}

/// Create an f32 vector with alternating positive/negative values.
fn make_f32_vector_alternating(n: usize) -> TensorData {
    let data: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    TensorData::from_f32(Shape::vector(n), &data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_element_wise_add_runs() {
        let result = bench_element_wise(128, "Add");
        assert_eq!(result.name, "Add[128]");
        assert_eq!(result.input_size, 128);
        assert!(result.interpreter_ns > 0, "timing should be nonzero");
        assert!(result.throughput_gflops > 0.0, "throughput should be positive");
    }

    #[test]
    fn bench_element_wise_all_ops() {
        for op in &["Add", "Mul", "Relu", "Sigmoid"] {
            let result = bench_element_wise(64, op);
            assert!(result.interpreter_ns > 0, "{op} should have nonzero time");
            assert!(
                result.throughput_gflops.is_finite(),
                "{op} throughput should be finite"
            );
        }
    }

    #[test]
    fn bench_matmul_runs() {
        let result = bench_matmul(16, 16, 16);
        assert_eq!(result.name, "MatMul[16x16x16]");
        assert!(result.interpreter_ns > 0);
        assert!(result.throughput_gflops > 0.0);
    }

    #[test]
    fn bench_mlp_forward_runs() {
        let result = bench_mlp_forward(32, 16, 4, 4);
        assert!(result.name.contains("MLP"));
        assert_eq!(result.input_size, 4 * 32);
        assert!(result.interpreter_ns > 0);
        assert!(result.throughput_gflops > 0.0);
    }

    #[test]
    fn bench_ternary_compression_runs() {
        let result = bench_ternary_compression(256);
        assert_eq!(result.name, "Ternary[256]");
        assert_eq!(result.input_size, 256);
        assert!(result.interpreter_ns > 0);
    }

    #[test]
    fn bench_format_table_not_empty() {
        let results = vec![
            bench_element_wise(64, "Add"),
            bench_matmul(8, 8, 8),
        ];
        let table = format_benchmark_table(&results);
        assert!(table.contains("Benchmark"));
        assert!(table.contains("Add[64]"));
        assert!(table.contains("MatMul[8x8x8]"));
        assert!(table.contains("GFLOPS"));
    }

    #[test]
    fn bench_format_empty() {
        let table = format_benchmark_table(&[]);
        assert_eq!(table, "(no benchmark results)");
    }

    #[test]
    fn bench_suite_runs_all() {
        let suite = BenchmarkSuite {
            element_wise_sizes: vec![64],
            matmul_sizes: vec![(8, 8, 8)],
            mlp_configs: vec![(16, 8, 4, 2)],
            ternary_sizes: vec![64],
            warmup_iters: 1,
            bench_iters: 3,
        };
        let results = suite.run();
        // 4 element-wise ops * 1 size + 1 matmul + 1 mlp + 1 ternary = 7
        assert_eq!(results.len(), 7);
        for r in &results {
            assert!(r.interpreter_ns > 0, "all benchmarks should have timing: {}", r.name);
        }
    }

    #[test]
    fn bench_unsupported_op() {
        let result = bench_element_wise(64, "Nonexistent");
        assert!(result.name.contains("unsupported"));
        assert_eq!(result.interpreter_ns, 0);
    }
}
