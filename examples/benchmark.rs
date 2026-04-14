//! Benchmark: Compare Interpreter vs JIT vs SIMD execution.
//!
//! Shows the performance progression:
//!   Interpreter (Rust) → JIT (LLVM scalar) → SIMD (LLVM AVX2)
//!
//! This demonstrates why QLANG compiles to machine code
//! instead of interpreting graphs like Python.


fn main() {
    #[cfg(not(feature = "llvm"))]
    {
        eprintln!("LLVM not available. Build with: cargo run --example benchmark --features llvm");
        return;
    }

    #[cfg(feature = "llvm")]
    run();
}

#[cfg(feature = "llvm")]
fn run() {
    use std::time::Instant;
    println!("=== QLANG Performance Benchmark ===\n");

    use inkwell::context::Context;
    use inkwell::OptimizationLevel;
    use qlang_core::tensor::{Dtype, Shape, TensorType as TT};

    let sizes = [1024, 4096, 16384, 65536, 262144, 1048576];
    let n_warmup = 3;
    let n_runs = 10;

    println!("Warming up JIT compiler...\n");

    println!("{:>10} {:>14} {:>14} {:>14} {:>10} {:>10}",
        "Elements", "Interpreter", "JIT Scalar", "JIT SIMD", "JIT/Int", "SIMD/Int");
    println!("{}", "-".repeat(88));

    for &n in &sizes {
        // Build graph: y = relu(a + b)
        let mut e = qlang_agent::emitter::GraphEmitter::new("bench");
        let a = e.input("a", Dtype::F32, Shape::vector(n));
        let b = e.input("b", Dtype::F32, Shape::vector(n));
        let sum = e.add(a, b, TT::f32_vector(n));
        let activated = e.relu(sum, TT::f32_vector(n));
        e.output("y", activated, TT::f32_vector(n));
        let graph = e.build();

        let input_a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - (n as f32 * 0.005)).collect();
        let input_b: Vec<f32> = (0..n).map(|i| i as f32 * 0.005).collect();

        // === Compile ONCE (not measured) ===
        let context = Context::create();
        let compiled = qlang_compile::codegen::compile_graph(&context, &graph, OptimizationLevel::Aggressive).unwrap();
        let simd_compiled = qlang_compile::simd::compile_graph_simd(&context, &graph).unwrap();

        // === Warmup ===
        for _ in 0..n_warmup {
            let _ = qlang_compile::codegen::execute_compiled(&compiled, &input_a, &input_b);
            let _ = qlang_compile::aligned::execute_aligned(&simd_compiled, &input_a, &input_b);
        }

        // === Interpreter (best of n_runs) ===
        let mut best_interp = std::time::Duration::from_secs(999);
        for _ in 0..n_runs {
            let mut inputs = std::collections::HashMap::new();
            inputs.insert("a".into(), qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_a));
            inputs.insert("b".into(), qlang_core::tensor::TensorData::from_f32(Shape::vector(n), &input_b));
            let start = Instant::now();
            let _ = qlang_runtime::executor::execute(&graph, inputs).unwrap();
            best_interp = best_interp.min(start.elapsed());
        }

        // === JIT Scalar (best of n_runs) ===
        let mut best_jit = std::time::Duration::from_secs(999);
        for _ in 0..n_runs {
            let start = Instant::now();
            let _ = qlang_compile::codegen::execute_compiled(&compiled, &input_a, &input_b).unwrap();
            best_jit = best_jit.min(start.elapsed());
        }

        // === JIT SIMD (best of n_runs, aligned memory) ===
        let mut best_simd = std::time::Duration::from_secs(999);
        for _ in 0..n_runs {
            let start = Instant::now();
            let _ = qlang_compile::aligned::execute_aligned(&simd_compiled, &input_a, &input_b).unwrap();
            best_simd = best_simd.min(start.elapsed());
        }

        let jit_speedup = best_interp.as_nanos() as f64 / best_jit.as_nanos().max(1) as f64;
        let simd_speedup = best_interp.as_nanos() as f64 / best_simd.as_nanos().max(1) as f64;

        println!("{n:>10} {best_interp:>14.2?} {best_jit:>14.2?} {best_simd:>14.2?} {jit_speedup:>9.1}x {simd_speedup:>9.1}x");
    }

    println!("\n=== Legend ===");
    println!("  Interpreter: Pure Rust, no LLVM (like Python)");
    println!("  JIT Scalar:  LLVM compiled, scalar float ops");
    println!("  JIT SIMD:    LLVM compiled, <8 x float> AVX2 vector ops");
    println!("  JIT/Int:     Speedup of JIT over Interpreter");
    println!("  SIMD/Int:    Speedup of SIMD over Interpreter");
}
