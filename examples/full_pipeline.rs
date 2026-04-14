//! QLANG Full Pipeline Demo — Everything in one run.
//!
//! Demonstrates the complete QLANG ecosystem:
//! 1. Parse .qlang text → Graph
//! 2. Train with autograd
//! 3. IGQK ternary compression
//! 4. Save checkpoint (.qlm)
//! 5. Export to all targets: LLVM IR, WASM, GPU shader, .o file
//! 6. Visualize (ASCII, DOT)
//! 7. Distributed training simulation

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  QLANG Full Pipeline Demo v0.4               ║");
    println!("║  Graph-based AI-to-AI Programming Language   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    use qlang_runtime::autograd::train_mlp_autograd;
    use qlang_runtime::checkpoint::{Checkpoint, WeightTensor, CompressionState};
    
    use qlang_runtime::training::MlpWeights;
    use qlang_agent::distributed;
    use std::time::Instant;

    let total_start = Instant::now();

    // ═══ 1. Parse QLANG text ═══
    println!("━━━ [1/7] PARSE .qlang text ━━━");
    let qlang_source = r#"
graph classifier {
  input x: f32[1, 64]
  input W1: f32[64, 32]
  input W2: f32[32, 10]

  node h = matmul(x, W1)
  node a = relu(h)
  node logits = matmul(a, W2)
  node probs = softmax(logits)
  node comp = to_ternary(W1) @proof theorem_5_2

  output predictions = probs
  output compressed = comp
}
"#;

    let graph = qlang_compile::parser::parse(qlang_source).unwrap();
    println!("  Parsed: {} nodes, {} edges ✓", graph.nodes.len(), graph.edges.len());
    let verify = qlang_core::verify::verify_graph(&graph);
    println!("  Verified: {} checks passed ✓\n", verify.passed.len());

    // ═══ 2. Train with autograd ═══
    println!("━━━ [2/7] TRAIN with autograd ━━━");
    let input_dim = 64;
    let hidden_dim = 32;
    let output_dim = 10;

    let data = qlang_runtime::mnist::MnistData::synthetic(500, input_dim);

    let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt() as f32;
    let scale2 = (2.0 / (hidden_dim + output_dim) as f64).sqrt() as f32;
    let mut w1: Vec<f32> = (0..input_dim * hidden_dim).map(|i| (i as f32 * 0.371).sin() * scale1).collect();
    let mut b1 = vec![0.0f32; hidden_dim];
    let mut w2: Vec<f32> = (0..hidden_dim * output_dim).map(|i| (i as f32 * 0.529).sin() * scale2).collect();
    let mut b2 = vec![0.0f32; output_dim];

    let train_start = Instant::now();
    let epochs = 20;

    for epoch in 0..epochs {
        let n_batches = data.n_train / 25;
        for batch in 0..n_batches {
            let (x, y) = data.train_batch(batch * 25, 25);
            train_mlp_autograd(&mut w1, &mut b1, &mut w2, &mut b2, x, y,
                input_dim, hidden_dim, output_dim, 0.03);
        }
        if epoch == 0 || epoch == epochs - 1 {
            let (loss, acc) = train_mlp_autograd(
                &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
                &data.test_images, &data.test_labels,
                input_dim, hidden_dim, output_dim, 0.0);
            println!("  Epoch {:>2}/{}: loss={:.4}, acc={:.1}%", epoch + 1, epochs, loss, acc * 100.0);
        }
    }
    let train_time = train_start.elapsed();
    println!("  Training: {:?} ✓\n", train_time);

    // ═══ 3. IGQK Compression ═══
    println!("━━━ [3/7] IGQK ternary compression ━━━");
    let mlp = MlpWeights { w1: w1.clone(), b1: b1.clone(), w2: w2.clone(), b2: b2.clone(),
        input_dim, hidden_dim, output_dim };
    let compressed = mlp.compress_ternary();

    let (_, orig_acc) = train_mlp_autograd(
        &mut w1.clone(), &mut b1.clone(), &mut w2.clone(), &mut b2.clone(),
        &data.test_images, &data.test_labels, input_dim, hidden_dim, output_dim, 0.0);
    let comp_probs = compressed.forward(&data.test_images);
    let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

    println!("  Original:   {:.1}% accuracy", orig_acc * 100.0);
    println!("  Compressed: {:.1}% accuracy", comp_acc * 100.0);
    println!("  Ratio:      {:.1}x ✓\n", mlp.param_count() as f64 * 4.0 /
        ((w1.len() + w2.len()) as f64 + (b1.len() + b2.len()) as f64 * 4.0));

    // ═══ 4. Save checkpoint ═══
    println!("━━━ [4/7] SAVE checkpoint ━━━");
    let mut ckpt = Checkpoint::new(graph.clone());
    ckpt.add_weight(WeightTensor::from_f32("W1", vec![input_dim, hidden_dim], &w1));
    ckpt.add_weight(WeightTensor::from_f32("b1", vec![hidden_dim], &b1));
    ckpt.add_weight(WeightTensor::from_f32("W2", vec![hidden_dim, output_dim], &w2));
    ckpt.add_weight(WeightTensor::from_f32("b2", vec![output_dim], &b2));
    ckpt.metadata.epochs_trained = epochs;
    ckpt.metadata.final_accuracy = orig_acc;
    ckpt.metadata.training_time_ms = train_time.as_millis() as u64;
    ckpt.compression = Some(CompressionState {
        method: "ternary".into(),
        compression_ratio: 4.0,
        distortion: 0.01,
        accuracy_before: orig_acc,
        accuracy_after: comp_acc,
    });

    let ckpt_path = "/tmp/qlang_model.qlm";
    let ckpt_size = ckpt.save_binary(ckpt_path).unwrap();
    println!("  Saved: {} ({} bytes)", ckpt_path, ckpt_size);
    println!("  Params: {} ({:.1} KB)", ckpt.param_count(), ckpt.total_bytes() as f64 / 1024.0);
    println!("  ✓\n");

    // ═══ 5. Export to all targets ═══
    println!("━━━ [5/7] EXPORT to all compilation targets ━━━");

    // LLVM IR (element-wise ops only — matmul needs separate codegen)
    #[cfg(feature = "llvm")]
    {
        // Build a simple graph for LLVM demo
        let mut llvm_graph = qlang_core::graph::Graph::new("llvm_demo");
        let la = llvm_graph.add_node(qlang_core::ops::Op::Input { name: "a".into() }, vec![], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let lb = llvm_graph.add_node(qlang_core::ops::Op::Input { name: "b".into() }, vec![], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let ladd = llvm_graph.add_node(qlang_core::ops::Op::Add, vec![qlang_core::tensor::TensorType::f32_vector(64); 2], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let lrelu = llvm_graph.add_node(qlang_core::ops::Op::Relu, vec![qlang_core::tensor::TensorType::f32_vector(64)], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let lout = llvm_graph.add_node(qlang_core::ops::Op::Output { name: "y".into() }, vec![qlang_core::tensor::TensorType::f32_vector(64)], vec![]);
        llvm_graph.add_edge(la, 0, ladd, 0, qlang_core::tensor::TensorType::f32_vector(64));
        llvm_graph.add_edge(lb, 0, ladd, 1, qlang_core::tensor::TensorType::f32_vector(64));
        llvm_graph.add_edge(ladd, 0, lrelu, 0, qlang_core::tensor::TensorType::f32_vector(64));
        llvm_graph.add_edge(lrelu, 0, lout, 0, qlang_core::tensor::TensorType::f32_vector(64));

        let context = inkwell::context::Context::create();
        let compiled = qlang_compile::codegen::compile_graph(&context, &llvm_graph,
            inkwell::OptimizationLevel::Aggressive).unwrap();
        println!("  LLVM IR:       {} bytes ✓", compiled.llvm_ir.len());
    }
    #[cfg(not(feature = "llvm"))]
    println!("  LLVM IR:       skipped (no LLVM)");

    // WebAssembly
    let wat = qlang_compile::wasm::to_wat(&graph);
    let js = qlang_compile::wasm::to_js_loader(&graph);
    println!("  WebAssembly:   {} bytes (WAT) ✓", wat.len());
    println!("  JS loader:     {} bytes ✓", js.len());

    // GPU shader
    let wgsl = qlang_compile::gpu::to_wgsl(&graph);
    let matmul_shader = qlang_compile::gpu::matmul_wgsl(1, input_dim, hidden_dim);
    println!("  GPU WGSL:      {} bytes ✓", wgsl.len());
    println!("  MatMul shader: {} bytes ✓", matmul_shader.len());

    // Object file (using a simple graph for element-wise ops)
    #[cfg(feature = "llvm")]
    {
        let mut aot_graph = qlang_core::graph::Graph::new("aot_demo");
        let aa = aot_graph.add_node(qlang_core::ops::Op::Input { name: "a".into() }, vec![], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let ab = aot_graph.add_node(qlang_core::ops::Op::Input { name: "b".into() }, vec![], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let aadd = aot_graph.add_node(qlang_core::ops::Op::Add, vec![qlang_core::tensor::TensorType::f32_vector(64); 2], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let arelu = aot_graph.add_node(qlang_core::ops::Op::Relu, vec![qlang_core::tensor::TensorType::f32_vector(64)], vec![qlang_core::tensor::TensorType::f32_vector(64)]);
        let aout = aot_graph.add_node(qlang_core::ops::Op::Output { name: "y".into() }, vec![qlang_core::tensor::TensorType::f32_vector(64)], vec![]);
        aot_graph.add_edge(aa, 0, aadd, 0, qlang_core::tensor::TensorType::f32_vector(64));
        aot_graph.add_edge(ab, 0, aadd, 1, qlang_core::tensor::TensorType::f32_vector(64));
        aot_graph.add_edge(aadd, 0, arelu, 0, qlang_core::tensor::TensorType::f32_vector(64));
        aot_graph.add_edge(arelu, 0, aout, 0, qlang_core::tensor::TensorType::f32_vector(64));

        match qlang_compile::aot::compile_to_object(&aot_graph, "/tmp/qlang_graph.o",
            inkwell::OptimizationLevel::Aggressive) {
            Ok(aot) => {
                println!("  Native .o:     {} bytes ({}) ✓", aot.file_size, aot.target_triple);
                let _ = std::fs::remove_file("/tmp/qlang_graph.o");
            }
            Err(e) => println!("  Native .o:     error: {e}"),
        }
    }
    #[cfg(not(feature = "llvm"))]
    println!("  Native .o:     skipped (no LLVM)");

    // Binary graph
    let binary = qlang_core::serial::to_binary(&graph).unwrap();
    println!("  Binary .qlg:   {} bytes ✓", binary.len());

    // JSON
    let json = qlang_core::serial::to_json(&graph).unwrap();
    println!("  JSON:          {} bytes ✓", json.len());

    // .qlang text
    let text = qlang_compile::parser::to_qlang_text(&graph);
    println!("  .qlang text:   {} bytes ✓\n", text.len());

    // ═══ 6. Visualize ═══
    println!("━━━ [6/7] VISUALIZE ━━━");
    println!("{}", qlang_compile::visualize::to_ascii(&graph));

    // ═══ 7. Distributed training simulation ═══
    println!("━━━ [7/7] DISTRIBUTED training simulation ━━━");
    let mut job = distributed::create_data_parallel_job(
        "demo_job", 4, distributed::Hyperparams::default());
    println!("  Job: {} workers", job.workers.len());

    let mut worker_grads = std::collections::HashMap::new();
    for i in 0..4 {
        let grad: Vec<f32> = (0..100).map(|j| ((i * 100 + j) as f32 * 0.01).sin()).collect();
        worker_grads.insert(format!("trainer_{i}"), grad);
    }

    let aggregated = distributed::simulate_distributed_step(&mut job, &worker_grads);
    println!("  Aggregated gradients: {} values", aggregated.len());
    println!("  All workers done: {} ✓",
        job.workers.iter().all(|w| w.status == distributed::WorkerStatus::Done));

    // ═══ Summary ═══
    let total_time = total_start.elapsed();
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Pipeline complete in {:?}", total_time);
    println!("║");
    println!("║  Parse → Train → Compress → Save → Export");
    println!("║  → Visualize → Distribute");
    println!("║");
    println!("║  7 compilation targets:");
    println!("║    LLVM IR, Native .o, WebAssembly, GPU WGSL,");
    println!("║    Binary .qlg, JSON, .qlang text");
    println!("║");
    println!("║  No Python. No PyTorch. Pure Rust + QLANG.");
    println!("╚══════════════════════════════════════════════╝");

    let _ = std::fs::remove_file(ckpt_path);
}
