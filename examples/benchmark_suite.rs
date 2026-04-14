//! Comprehensive Benchmark Suite — Measure QLANG performance across operations.
//!
//! Run with: cargo run --release --example benchmark_suite
//!
//! Outputs results as a formatted table and optional JSON.
//! Compare with the companion `benchmark_pytorch.py` script.

use std::collections::HashMap;
use std::time::Instant;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};

fn f32_vec(n: usize) -> TensorType {
    TensorType::new(Dtype::F32, Shape::vector(n))
}

fn f32_mat(m: usize, n: usize) -> TensorType {
    TensorType::new(Dtype::F32, Shape::matrix(m, n))
}

struct BenchResult {
    name: String,
    size: usize,
    iterations: usize,
    total_ms: f64,
    mean_us: f64,
    min_us: f64,
    max_us: f64,
    throughput_mops: f64,
}

fn bench<F: FnMut()>(name: &str, size: usize, iterations: usize, mut f: F) -> BenchResult {
    // Warmup
    for _ in 0..3 {
        f();
    }

    let mut times = Vec::with_capacity(iterations);
    let total_start = Instant::now();

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_micros() as f64);
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let mean_us = times.iter().sum::<f64>() / times.len() as f64;
    let min_us = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_us = times.iter().cloned().fold(0.0, f64::max);
    let throughput_mops = (size as f64) / mean_us; // M elements/sec

    BenchResult {
        name: name.into(),
        size,
        iterations,
        total_ms,
        mean_us,
        min_us,
        max_us,
        throughput_mops,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              QLANG Comprehensive Benchmark Suite               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let mut results = Vec::new();

    // ---- Element-wise operations ----
    println!("▸ Element-wise Operations");
    for &size in &[1024, 16384, 262144, 1048576] {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();
        let iters = if size > 100000 { 50 } else { 200 };

        let a_c = a.clone();
        let b_c = b.clone();
        results.push(bench(&format!("add_{size}"), size, iters, || {
            let _: Vec<f32> = a_c.iter().zip(&b_c).map(|(x, y)| x + y).collect();
        }));

        let a_c = a.clone();
        results.push(bench(&format!("relu_{size}"), size, iters, || {
            let _: Vec<f32> = a_c.iter().map(|&x| x.max(0.0)).collect();
        }));

        let a_c = a.clone();
        results.push(bench(&format!("sigmoid_{size}"), size, iters, || {
            let _: Vec<f32> = a_c.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        }));
    }

    // ---- Matrix multiplication ----
    println!("▸ Matrix Multiplication");
    for &(m, k, n) in &[(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32).cos()).collect();
        let iters = if m > 128 { 10 } else { 50 };

        results.push(bench(&format!("matmul_{m}x{k}x{n}"), m * n, iters, || {
            let mut c = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..k {
                        sum += a[i * k + p] * b[p * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }));
    }

    // ---- Softmax ----
    println!("▸ Softmax");
    for &size in &[100, 1000, 10000] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin()).collect();
        results.push(bench(&format!("softmax_{size}"), size, 100, || {
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = data.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let _: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
        }));
    }

    // ---- IGQK Ternary Compression ----
    println!("▸ IGQK Ternary Compression");
    for &size in &[1024, 65536, 1048576] {
        let weights: Vec<f32> = (0..size).map(|i| (i as f32 * 0.0001).sin()).collect();
        results.push(bench(&format!("ternary_{size}"), size, 100, || {
            let mean_abs: f32 =
                weights.iter().map(|x| x.abs()).sum::<f32>() / weights.len() as f32;
            let threshold = mean_abs * 0.7;
            let _: Vec<f32> = weights
                .iter()
                .map(|&x| {
                    if x > threshold {
                        1.0
                    } else if x < -threshold {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
        }));
    }

    // ---- Graph execution ----
    println!("▸ Graph Execution (interpreter)");
    {
        let mut g = Graph::new("bench_graph");
        let inp_a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![f32_vec(1024)]);
        let inp_b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![f32_vec(1024)]);
        let add = g.add_node(
            Op::Add,
            vec![f32_vec(1024), f32_vec(1024)],
            vec![f32_vec(1024)],
        );
        let relu = g.add_node(Op::Relu, vec![f32_vec(1024)], vec![f32_vec(1024)]);
        let out = g.add_node(
            Op::Output { name: "y".into() },
            vec![f32_vec(1024)],
            vec![],
        );
        g.add_edge(inp_a, 0, add, 0, f32_vec(1024));
        g.add_edge(inp_b, 0, add, 1, f32_vec(1024));
        g.add_edge(add, 0, relu, 0, f32_vec(1024));
        g.add_edge(relu, 0, out, 0, f32_vec(1024));

        let a_data = TensorData::from_f32(Shape::vector(1024), &vec![1.0f32; 1024]);
        let b_data = TensorData::from_f32(Shape::vector(1024), &vec![-0.5f32; 1024]);

        results.push(bench("graph_add_relu_1024", 1024, 200, || {
            let mut inputs = HashMap::new();
            inputs.insert("a".into(), a_data.clone());
            inputs.insert("b".into(), b_data.clone());
            let _ = qlang_runtime::executor::execute(&g, inputs);
        }));
    }

    // ---- Print results table ----
    println!("\n╔═══════════════════════════╦═════════╦══════════╦══════════╦══════════╦═══════════╗");
    println!(
        "║ {:25} ║ {:>7} ║ {:>8} ║ {:>8} ║ {:>8} ║ {:>9} ║",
        "Benchmark", "Size", "Mean(µs)", "Min(µs)", "Max(µs)", "M elem/s"
    );
    println!("╠═══════════════════════════╬═════════╬══════════╬══════════╬══════════╬═══════════╣");
    for r in &results {
        println!(
            "║ {:25} ║ {:>7} ║ {:>8.1} ║ {:>8.1} ║ {:>8.1} ║ {:>9.2} ║",
            r.name, r.size, r.mean_us, r.min_us, r.max_us, r.throughput_mops
        );
    }
    println!("╚═══════════════════════════╩═════════╩══════════╩══════════╩══════════╩═══════════╝");

    // ---- JSON output ----
    if std::env::args().any(|a| a == "--json") {
        println!("\n--- JSON ---");
        print!("[");
        for (i, r) in results.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!(
                r#"{{"name":"{}","size":{},"iters":{},"mean_us":{:.1},"min_us":{:.1},"max_us":{:.1},"throughput_mops":{:.2}}}"#,
                r.name, r.size, r.iterations, r.mean_us, r.min_us, r.max_us, r.throughput_mops
            );
        }
        println!("]");
    }

    println!("\nDone. {} benchmarks completed.", results.len());
}
