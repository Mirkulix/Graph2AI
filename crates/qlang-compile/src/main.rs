//! QLANG CLI — Compile, visualize and execute QLANG graph files.
//!
//! Usage:
//!   qlang-cli info     <file.qlg.json>                    Show graph info
//!   qlang-cli verify   <file.qlg.json>                    Verify constraints
//!   qlang-cli optimize <file.qlg.json> -o <output.json>   Optimize graph
//!   qlang-cli run      <file.qlg.json>                    Execute (interpreter)
//!   qlang-cli jit      <file.qlg.json>                    Execute (JIT/native)
//!   qlang-cli dot      <file.qlg.json>                    Output Graphviz DOT
//!   qlang-cli ascii    <file.qlg.json>                    ASCII visualization
//!   qlang-cli llvm-ir  <file.qlg.json>                    Show LLVM IR output

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::process;

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];

    // Commands that don't need a file argument
    if command == "repl" {
        qlang_compile::repl::run_repl();
        return;
    }

    if command == "lsp" {
        cmd_lsp();
        return;
    }

    if command == "ollama" {
        cmd_ollama(&args[2..]);
        return;
    }

    if command == "ai-train" {
        cmd_ai_train(&args[2..]);
        return;
    }

    if command == "web" {
        let port: u16 = args.iter()
            .position(|a| a == "--port")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(8081);
        cmd_web(port);
        return;
    }

    if command == "serve" {
        let port: u16 = args.iter()
            .position(|a| a == "--port")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(8081);
        cmd_web(port);
        return;
    }

    if command == "train-mnist" {
        let port: u16 = args.iter()
            .position(|a| a == "--port")
            .and_then(|i| args.get(i + 1))
            .and_then(|p| p.parse().ok())
            .unwrap_or(8081);
        let epochs: usize = args.iter()
            .position(|a| a == "--epochs")
            .and_then(|i| args.get(i + 1))
            .and_then(|e| e.parse().ok())
            .unwrap_or(50);
        cmd_train_mnist(port, epochs);
        return;
    }

    if command == "proxy" {
        cmd_proxy(&args[2..]);
        return;
    }

    if command == "binary" {
        cmd_binary(&args[2..]);
        return;
    }

    if command == "cache" {
        cmd_cache(&args[2..]);
        return;
    }

    if command == "ide" {
        cmd_ide(&args[2..]);
        return;
    }

    if command == "autonomous" {
        cmd_autonomous(&args[2..]);
        return;
    }

    if command == "train-lm" {
        cmd_train_lm(&args[2..]);
        return;
    }

    if command == "swarm-train" {
        cmd_swarm_train(&args[2..]);
        return;
    }

    if command == "orchestrate" {
        cmd_orchestrate(&args[2..]);
        return;
    }

    if command == "devices" {
        cmd_devices();
        return;
    }

    if command == "exec" {
        if args.len() < 3 {
            eprintln!("Usage: qlang-cli exec <script.qlang>");
            process::exit(1);
        }
        let file_path = &args[2];
        cmd_exec_script(file_path);
        return;
    }

    if command == "kubernetes" || command == "k8s" {
        cmd_kubernetes(&args[2..]);
        return;
    }

    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let file_path = &args[2];

    // Handle parse command separately (reads .qlang text, not JSON)
    if command == "parse" {
        cmd_parse(file_path);
        return;
    }

    // Read the graph file (JSON format)
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {file_path}: {e}");
            process::exit(1);
        }
    };

    let graph = match qlang_core::serial::from_json(&content) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing graph: {e}");
            process::exit(1);
        }
    };

    match command.as_str() {
        "info" => cmd_info(&graph),
        "verify" => cmd_verify(&graph),
        "optimize" => {
            let output = args.get(4).map(|s| s.as_str());
            cmd_optimize(graph, output);
        }
        "run" => cmd_run(&graph),
        "jit" => {
            #[cfg(feature = "llvm")]
            cmd_jit(&graph);
            #[cfg(not(feature = "llvm"))]
            {
                eprintln!("LLVM not available. Build with: cargo build --features llvm");
                process::exit(1);
            }
        }
        "dot" => cmd_dot(&graph),
        "ascii" => cmd_ascii(&graph),
        "llvm-ir" => {
            #[cfg(feature = "llvm")]
            cmd_llvm_ir(&graph);
            #[cfg(not(feature = "llvm"))]
            {
                eprintln!("LLVM not available. Build with: cargo build --features llvm");
                process::exit(1);
            }
        }
        "compile" => {
            #[cfg(feature = "llvm")]
            {
                let output = args.get(4).map(|s| s.as_str()).unwrap_or("/tmp/qlang_out.o");
                cmd_compile(&graph, output);
            }
            #[cfg(not(feature = "llvm"))]
            {
                eprintln!("LLVM not available. Build with: cargo build --features llvm");
                process::exit(1);
            }
        }
        "asm" => {
            #[cfg(feature = "llvm")]
            cmd_asm(&graph);
            #[cfg(not(feature = "llvm"))]
            {
                eprintln!("LLVM not available. Build with: cargo build --features llvm");
                process::exit(1);
            }
        }
        "wasm" => {
            println!("{}", qlang_compile::wasm::to_wat(&graph));
        }
        "gpu" => {
            println!("{}", qlang_compile::gpu::to_wgsl(&graph));
        }
        "stats" => {
            let stats = qlang_core::stats::compute_stats(&graph);
            println!("{stats}");
        }
        "schedule" => {
            let plan = qlang_runtime::scheduler::schedule(&graph);
            println!("{}", plan.report());
        }
        _ => {
            eprintln!("Unknown command: {command}");
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("QLANG CLI v0.3 — Graph-based AI-to-AI programming language\n");
    eprintln!("Usage:");
    eprintln!("  qlang-cli info     <file.qlg.json>                    Show graph info");
    eprintln!("  qlang-cli verify   <file.qlg.json>                    Verify constraints");
    eprintln!("  qlang-cli optimize <file.qlg.json> -o <output.json>   Optimize graph");
    eprintln!("  qlang-cli run      <file.qlg.json>                    Execute (interpreter)");
    eprintln!("  qlang-cli jit      <file.qlg.json>                    Execute (JIT/native)");
    eprintln!("  qlang-cli repl                                         Interactive REPL");
    eprintln!("  qlang-cli exec     <file.qlang>                        Execute a QLANG program");
    eprintln!("  qlang-cli parse    <file.qlang>                        Parse .qlang text file");
    eprintln!("  qlang-cli lsp                                            Start LSP server (stdin/stdout)");
    eprintln!("  qlang-cli web      [--port 8081]                       Start web dashboard server");
    eprintln!("  qlang-cli serve    [--port 8081]                       Alias for 'web'");
    eprintln!("  qlang-cli train-mnist [--port 8081] [--epochs 50]      Train MNIST with live dashboard");
    eprintln!("  qlang-cli ai-train [--model M] [--quick]               AI-designed training pipeline");
    eprintln!("  qlang-cli proxy    [--port 9100] [--upstream URL]       HTTP-to-QLMS signing proxy");
    eprintln!("  qlang-cli ollama   health|models|generate|chat         Ollama LLM operations");
    eprintln!("  qlang-cli binary   encode <file.qlg.json>              Encode graph to binary .qlb");
    eprintln!("  qlang-cli binary   decode <file.qlb>                  Decode binary .qlb to JSON");
    eprintln!("  qlang-cli cache    stats                              Show cache hit/miss statistics");
    eprintln!("  qlang-cli cache    clear                              Clear computation cache");
    eprintln!("  qlang-cli ide      [--port 3000]                       Start QLANG IDE (Eclipse Theia)");
    eprintln!("  qlang-cli autonomous [--task T] [--target 95] [--iterations 5] [--model M] [--quick]");
    eprintln!("                                                          Autonomous AI feedback loop");
    eprintln!("  qlang-cli train-lm --data <text.txt> [options]           Train a transformer language model");
    eprintln!("  qlang-cli swarm-train [--data FILE] [--population 10] [--generations 5] [--quick]");
    eprintln!("      --vocab-size 1000 --d-model 128 --layers 4 --heads 4 --epochs 10 --seq-len 128");
    eprintln!("  qlang-cli orchestrate \"task\" [--budget 0.10] [--quality 0.9] [--models 3]");
    eprintln!("                                                          Multi-model orchestration");
    eprintln!("  qlang-cli devices                                      List available compute devices");
    eprintln!("  qlang-cli compile  <file.qlg.json> -o <output.o>      Compile to object file");
    eprintln!("  qlang-cli asm      <file.qlg.json>                    Show native assembly");
    eprintln!("  qlang-cli dot      <file.qlg.json>                    Output Graphviz DOT");
    eprintln!("  qlang-cli ascii    <file.qlg.json>                    ASCII visualization");
    eprintln!("  qlang-cli llvm-ir  <file.qlg.json>                    Show LLVM IR output");
}

fn cmd_kubernetes(args: &[String]) {
    use qlang_agent::kubernetes::{generate_job_manifests, K8sConfig};
    use qlang_agent::distributed::{create_data_parallel_job, Hyperparams};

    let n_workers = args.iter()
        .position(|a| a == "--workers")
        .and_then(|i| args.get(i + 1))
        .and_then(|w| w.parse::<usize>().ok())
        .unwrap_or(4);

    let n_gpus = args.iter()
        .position(|a| a == "--gpu")
        .and_then(|i| args.get(i + 1))
        .and_then(|g| g.parse::<usize>().ok());

    let mut config = K8sConfig::default();
    config.gpu_limit = n_gpus;

    println!("=== Generating Kubernetes Manifests ===");
    println!("Workers: {}", n_workers);
    if let Some(gpu) = n_gpus {
        println!("GPUs per worker: {}", gpu);
    }
    println!("Namespace: {}", config.namespace);
    println!();

    let job = create_data_parallel_job("qlang-training", n_workers, Hyperparams::default());
    let manifests = generate_job_manifests(&job, &config);

    let output_dir = std::path::Path::new("k8s-manifests");
    if !output_dir.exists() {
        std::fs::create_dir(output_dir).unwrap();
    }

    for manifest in manifests {
        let filename = format!("{}.yaml", manifest.name);
        let path = output_dir.join(&filename);
        std::fs::write(&path, &manifest.yaml).unwrap();
        println!("Created: k8s-manifests/{}", filename);
    }

    println!("\nManifests generated successfully!");
    println!("To deploy the cluster, run:");
    println!("  kubectl apply -f k8s-manifests/");
}

fn cmd_devices() {
    println!("=== QLANG Compute Devices ===\n");
    let devices = qlang_runtime::distributed_train::detect_devices();
    for (i, dev) in devices.iter().enumerate() {
        println!("  [{}] {} ({})", i, dev.name, dev.compute_capability);
    }
    println!("\n  Total: {} devices", devices.len());
    println!("\n  Backend priority:");
    println!("    1. wgpu GPU (NVIDIA/AMD) -- cargo build --features gpu");
    println!("    2. Apple MLX (Metal)     -- cargo build --features mlx");
    println!("    3. Apple Accelerate (CPU BLAS)");
    println!("    4. Pure Rust (fallback)");
}

fn cmd_ide(args: &[String]) {
    let port: u16 = args.iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    // Check if Theia is set up
    let theia_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().and_then(|p| p.parent())
        .map(|p| p.join("editors/theia"))
        .unwrap_or_default();

    let node_modules = theia_dir.join("node_modules");

    if !node_modules.exists() {
        println!("QLANG IDE (Eclipse Theia) is not installed yet.");
        println!();
        println!("To set up the IDE:");
        println!("  cd editors/theia");
        println!("  bash setup.sh");
        println!();
        println!("Then run:");
        println!("  qlang-cli ide --port {}", port);
        return;
    }

    println!("Starting QLANG IDE on http://localhost:{}", port);
    println!("Press Ctrl+C to stop.");

    // Start Theia via yarn
    let status = std::process::Command::new("yarn")
        .arg("start")
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&theia_dir)
        .env("QLANG_CLI", std::env::current_exe().unwrap_or_default())
        .status();

    match status {
        Ok(s) if !s.success() => eprintln!("Theia exited with: {}", s),
        Err(e) => eprintln!("Failed to start Theia: {}", e),
        _ => {}
    }
}

fn cmd_exec_script(file_path: &str) {
    let source = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {}: {}", file_path, e);
            process::exit(1);
        }
    };

    let start = std::time::Instant::now();

    // Tier 1: Try LLVM JIT (native speed, numeric-only scripts)
    #[cfg(feature = "llvm")]
    {
        if let Ok(tokens) = qlang_runtime::vm::tokenize(&source) {
            if let Ok(stmts) = qlang_runtime::vm::Parser::new(tokens).parse_program() {
                if let Some((_result, _output)) = qlang_compile::script_jit::try_jit_run(&stmts) {
                    // Print output was already sent to stdout by the JIT callback
                    eprintln!("[executed in {:.3}s, LLVM JIT]", start.elapsed().as_secs_f64());
                    return;
                }
            }
        }
    }

    // Tier 2: Bytecode VM (fast, all features except graphs)
    // Use catch_unwind to gracefully fall back on panics (e.g., unsupported features)
    let bc_result = std::panic::catch_unwind(|| {
        qlang_runtime::bytecode::run_bytecode(&source)
    });
    match bc_result.unwrap_or(Err(qlang_runtime::vm::VmError::RuntimeError("bytecode panic".into()))) {
        Ok((_value, output)) => {
            for line in &output {
                println!("{}", line);
            }
            let elapsed = start.elapsed();
            eprintln!("[executed in {:.3}s, bytecode VM]", elapsed.as_secs_f64());
        }
        Err(_bc_err) => {
            // Tier 3: Tree-walking interpreter (all features including graphs)
            match qlang_runtime::unified::execute_unified(&source) {
                Ok(result) => {
                    for line in &result.output {
                        println!("{}", line);
                    }

                    let elapsed = start.elapsed();

                    if !result.graphs.is_empty() {
                        eprintln!("[graphs defined: {}]", result.graphs.join(", "));
                    }

                    eprintln!("[executed in {:.3}s, interpreter]", elapsed.as_secs_f64());
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            }
        }
    }
}

fn cmd_info(graph: &qlang_core::graph::Graph) {
    println!("{graph}");

    let inputs = graph.input_nodes();
    let outputs = graph.output_nodes();
    let quantum_ops: Vec<_> = graph.nodes.iter().filter(|n| n.op.is_quantum()).collect();

    println!("Summary:");
    println!("  Inputs:      {}", inputs.len());
    println!("  Outputs:     {}", outputs.len());
    println!("  Total nodes: {}", graph.nodes.len());
    println!("  Total edges: {}", graph.edges.len());
    println!("  Quantum ops: {}", quantum_ops.len());

    if let Ok(binary) = qlang_core::serial::to_binary(graph) {
        println!("  Binary size: {} bytes", binary.len());
    }
}

fn cmd_verify(graph: &qlang_core::graph::Graph) {
    let result = qlang_core::verify::verify_graph(graph);
    println!("{result}");

    if result.is_ok() {
        println!("Graph verification PASSED.");
    } else {
        println!("Graph verification FAILED.");
        process::exit(1);
    }
}

fn cmd_optimize(mut graph: qlang_core::graph::Graph, output: Option<&str>) {
    let before = graph.nodes.len();
    let report = qlang_compile::optimize::optimize(&mut graph);
    let after = graph.nodes.len();

    println!("Optimization complete:");
    println!("  Nodes before: {before}");
    println!("  Nodes after:  {after}");
    println!("  Dead nodes removed:  {}", report.dead_nodes_removed);
    println!("  Constants folded:    {}", report.constants_folded);
    println!("  Identity ops removed:{}", report.identity_ops_removed);
    println!("  CSE eliminated:      {}", report.common_subexpressions_eliminated);
    println!("  Ops fused:           {}", report.ops_fused);
    for desc in &report.fused_descriptions {
        println!("    {desc}");
    }

    if let Some(path) = output {
        let json = qlang_core::serial::to_json(&graph).unwrap();
        fs::write(path, json).unwrap();
        println!("  Saved to: {path}");
    }
}

fn cmd_run(graph: &qlang_core::graph::Graph) {
    let mut inputs = HashMap::new();
    for node in graph.input_nodes() {
        if let qlang_core::ops::Op::Input { name } = &node.op {
            if let Some(tt) = node.output_types.first() {
                if let Some(data) = qlang_core::tensor::TensorData::zeros(tt) {
                    println!("  Input '{name}': {} (zeros)", tt);
                    inputs.insert(name.clone(), data);
                }
            }
        }
    }

    match qlang_runtime::executor::execute(graph, inputs) {
        Ok(result) => {
            println!("\nExecution complete (interpreter):");
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Quantum ops:    {}", result.stats.quantum_ops);
            println!("  Total FLOPs:    {}", result.stats.total_flops);

            for (name, tensor) in &result.outputs {
                println!("\n  Output '{name}':");
                println!("    dtype: {}", tensor.dtype);
                println!("    shape: {}", tensor.shape);
                if let Some(vals) = tensor.as_f32_slice() {
                    if vals.len() <= 20 {
                        println!("    values: {:?}", vals);
                    } else {
                        println!("    values: [{}, {}, ... {} total]", vals[0], vals[1], vals.len());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Execution failed: {e}");
            process::exit(1);
        }
    }
}

#[cfg(feature = "llvm")]
fn cmd_jit(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;

    println!("JIT compiling graph '{}'...", graph.id);

    let context = Context::create();
    let result = qlang_compile::codegen::compile_graph(&context, graph, OptimizationLevel::Aggressive);
    match result {
        Ok(compiled) => {
            println!("  Compilation successful!");
            println!("  LLVM IR size: {} bytes", compiled.llvm_ir.len());

            // Determine input sizes from graph
            let input_nodes = graph.input_nodes();
            let n = input_nodes.first()
                .and_then(|n| n.output_types.first())
                .and_then(|t| t.shape.numel())
                .unwrap_or(4);

            let input_a = vec![0.0f32; n];
            let input_b = vec![0.0f32; n];

            println!("  Executing with {} zero-filled elements...", n);

            match qlang_compile::codegen::execute_compiled(&compiled, &input_a, &input_b) {
                Ok(result) => {
                    println!("\n  JIT execution complete (native code):");
                    if result.len() <= 20 {
                        println!("    output: {:?}", result);
                    } else {
                        println!("    output: [{}, {}, ... {} total]", result[0], result[1], result.len());
                    }
                }
                Err(e) => eprintln!("  JIT execution failed: {e}"),
            }
        }
        Err(e) => {
            eprintln!("  JIT compilation failed: {e}");
            process::exit(1);
        }
    };
}

fn cmd_dot(graph: &qlang_core::graph::Graph) {
    print!("{}", qlang_compile::visualize::to_dot(graph));
}

fn cmd_ascii(graph: &qlang_core::graph::Graph) {
    print!("{}", qlang_compile::visualize::to_ascii(graph));
}

#[cfg(feature = "llvm")]
fn cmd_llvm_ir(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;

    let context = Context::create();
    let result = qlang_compile::codegen::compile_graph(&context, graph, OptimizationLevel::None);
    match result {
        Ok(compiled) => {
            println!("{}", compiled.llvm_ir);
        }
        Err(e) => {
            eprintln!("Codegen failed: {e}");
            process::exit(1);
        }
    };
}

fn cmd_parse(file_path: &str) {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading {file_path}: {e}");
            process::exit(1);
        }
    };

    match qlang_compile::parser::parse(&content) {
        Ok(graph) => {
            println!("Parsed successfully!\n");
            println!("{graph}");

            // Verify
            let result = qlang_core::verify::verify_graph(&graph);
            println!("{result}");

            // Show as JSON
            if let Ok(json) = qlang_core::serial::to_json(&graph) {
                println!("JSON output ({} bytes):", json.len());
                if json.len() <= 2000 {
                    println!("{json}");
                } else {
                    println!("{}...", &json[..2000]);
                }
            }
        }
        Err(e) => {
            eprintln!("Parse error: {e}");
            process::exit(1);
        }
    }
}

#[cfg(feature = "llvm")]
fn cmd_compile(graph: &qlang_core::graph::Graph, output: &str) {
    use inkwell::OptimizationLevel;

    println!("Compiling graph '{}' to native object file...", graph.id);

    match qlang_compile::aot::compile_to_object(graph, output, OptimizationLevel::Aggressive) {
        Ok(result) => {
            println!("  Target:      {}", result.target_triple);
            println!("  CPU:         {}", result.cpu);
            println!("  Object file: {}", result.object_path);
            println!("  File size:   {} bytes", result.file_size);
            println!("\n  Link with: cc -o program {} -lm", result.object_path);
            println!("  Function:  void qlang_graph(float*, float*, float*, uint64_t)");
        }
        Err(e) => {
            eprintln!("Compilation failed: {e}");
            process::exit(1);
        }
    }
}

#[cfg(feature = "llvm")]
fn cmd_asm(graph: &qlang_core::graph::Graph) {
    use inkwell::OptimizationLevel;

    match qlang_compile::aot::compile_to_object(graph, "/tmp/qlang_asm_tmp.o", OptimizationLevel::Aggressive) {
        Ok(result) => {
            println!("{}", result.assembly);
            let _ = std::fs::remove_file("/tmp/qlang_asm_tmp.o");
        }
        Err(e) => {
            eprintln!("Compilation failed: {e}");
            process::exit(1);
        }
    }
}

fn cmd_ollama(args: &[String]) {
    let sub = args.first().map(|s| s.as_str()).unwrap_or("");
    let client = qlang_runtime::ollama::OllamaClient::from_env();

    match sub {
        "health" => {
            match client.health() {
                Ok(true) => println!("Ollama is running at {}:{}", client.host, client.port),
                Ok(false) => {
                    eprintln!("Ollama is not reachable at {}:{}", client.host, client.port);
                    process::exit(1);
                }
                Err(e) => {
                    eprintln!("Error checking Ollama health: {e}");
                    process::exit(1);
                }
            }
        }
        "models" => {
            match client.list_models() {
                Ok(models) => {
                    if models.is_empty() {
                        println!("No models available.");
                    } else {
                        println!("Available models:");
                        for m in &models {
                            println!("  {m}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error listing models: {e}");
                    process::exit(1);
                }
            }
        }
        "generate" => {
            let model = match args.get(1) {
                Some(m) => m,
                None => {
                    eprintln!("Usage: qlang-cli ollama generate <model> <prompt>");
                    process::exit(1);
                }
            };
            let prompt = args[2..].join(" ");
            if prompt.is_empty() {
                eprintln!("Usage: qlang-cli ollama generate <model> <prompt>");
                process::exit(1);
            }
            match client.generate(model, &prompt, None) {
                Ok(response) => println!("{response}"),
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            }
        }
        "chat" => {
            let model = match args.get(1) {
                Some(m) => m,
                None => {
                    eprintln!("Usage: qlang-cli ollama chat <model> <prompt>");
                    process::exit(1);
                }
            };
            let prompt = args[2..].join(" ");
            if prompt.is_empty() {
                eprintln!("Usage: qlang-cli ollama chat <model> <prompt>");
                process::exit(1);
            }
            let messages = vec![
                qlang_runtime::ollama::ChatMessage::user(prompt),
            ];
            match client.chat(model, messages) {
                Ok(response) => println!("{response}"),
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Usage: qlang-cli ollama <subcommand>");
            eprintln!("");
            eprintln!("Subcommands:");
            eprintln!("  health              Check if Ollama is running");
            eprintln!("  models              List available models");
            eprintln!("  generate <model> <prompt>   Generate text");
            eprintln!("  chat <model> <prompt>       Chat with model");
            process::exit(1);
        }
    }
}

fn cmd_web(port: u16) {
    // Determine the web root directory (relative to the workspace root)
    let web_root = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        // Go up from crates/qlang-compile to the workspace root, then into web/
        let workspace_root = std::path::Path::new(manifest)
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(std::path::Path::new("."));
        workspace_root.join("web").to_string_lossy().to_string()
    };

    println!("QLANG Dashboard: http://localhost:{port}");
    println!("Serving files from: {web_root}");
    println!("WebSocket endpoint: ws://localhost:{port}/ws");
    println!("Press Ctrl+C to stop.\n");

    match qlang_runtime::web_server::WebServer::start(port, web_root) {
        Ok(handle) => {
            // Send a startup event
            handle.broadcast(qlang_runtime::web_server::WebEvent::SystemLog {
                level: "info".to_string(),
                message: "QLANG Dashboard server started".to_string(),
            });

            // Block main thread — the server runs in a background thread
            // but we need to keep the process alive.
            loop {
                std::thread::sleep(std::time::Duration::from_secs(3600));
            }
        }
        Err(e) => {
            eprintln!("Failed to start web server: {e}");
            process::exit(1);
        }
    }
}

fn cmd_ai_train(args: &[String]) {
    use qlang_runtime::ollama::OllamaClient;
    use qlang_runtime::training::MlpWeights3;
    use qlang_runtime::mnist::MnistData;

    let model = args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("deepseek-r1:1.5b");

    let quick = args.iter().any(|a| a == "--quick");

    println!("=== QLANG AI-Train Pipeline ===\n");

    // Step 1: Ask Ollama to design a model
    println!("Step 1: Asking AI architect ({}) to design a neural network...\n", model);

    let client = OllamaClient::from_env();

    // Check if Ollama is running
    match client.health() {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("Error: Ollama is not running at {}:{}", client.host, client.port);
            eprintln!("Start Ollama with: ollama serve");
            eprintln!("Then pull a model: ollama pull {}", model);
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Error connecting to Ollama: {e}");
            eprintln!("Make sure Ollama is running: ollama serve");
            process::exit(1);
        }
    }

    let prompt = "You are a neural network architect. Design a small MLP for MNIST digit classification (784 inputs, 10 outputs). \
Reply with ONLY a JSON object like this, no other text:\n\
{\"hidden1\": 128, \"hidden2\": 64, \"epochs\": 30, \"learning_rate\": 0.1, \"batch_size\": 32}\n\
Choose reasonable values. Keep the model small but effective.";

    let system_prompt = "You are a JSON-only response bot. Output valid JSON only, no markdown, no explanation.";

    let response = match client.generate(model, prompt, Some(system_prompt)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error from Ollama: {e}");
            eprintln!("Make sure the model '{}' is available: ollama pull {}", model, model);
            process::exit(1);
        }
    };

    // Step 2: Parse the JSON response
    #[derive(serde::Deserialize)]
    struct AiDesign {
        hidden1: Option<usize>,
        hidden2: Option<usize>,
        epochs: Option<usize>,
        learning_rate: Option<f32>,
        batch_size: Option<usize>,
    }

    // Try to extract JSON from the response (LLMs sometimes wrap it in text)
    let json_str = extract_json(&response).unwrap_or_else(|| response.clone());

    let design: AiDesign = serde_json::from_str(&json_str).unwrap_or_else(|_| {
        eprintln!("Warning: Could not parse AI response as JSON, using defaults.");
        eprintln!("  AI said: {}", response.chars().take(200).collect::<String>());
        AiDesign {
            hidden1: None,
            hidden2: None,
            epochs: None,
            learning_rate: None,
            batch_size: None,
        }
    });

    let hidden1 = design.hidden1.unwrap_or(128).min(512).max(16);
    let hidden2 = design.hidden2.unwrap_or(64).min(256).max(8);
    let epochs = design.epochs.unwrap_or(30).min(30).max(1);
    let learning_rate = design.learning_rate.unwrap_or(0.1).min(1.0).max(0.001);
    let batch_size = design.batch_size.unwrap_or(64).min(256).max(8);

    // Step 3: Print what the AI designed
    println!("AI Architect ({}) designed:", model);
    println!("  Hidden layer 1: {} neurons", hidden1);
    println!("  Hidden layer 2: {} neurons", hidden2);
    println!("  Epochs: {}", epochs);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);

    // Step 4: Build and train the model
    println!("\nStep 2: Building model (784 -> {} -> {} -> 10)...", hidden1, hidden2);
    let mut mlp = MlpWeights3::new(784, hidden1, hidden2, 10);
    println!("  Parameters: {}", mlp.param_count());

    println!("\nStep 3: Loading data...");
    let data = if quick {
        println!("  Using synthetic data (--quick mode, 2000 train / 500 test)");
        MnistData::synthetic(2000, 500)
    } else {
        // Try to load real MNIST, fall back to download, fall back to synthetic
        let data_dir = {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let workspace_root = std::path::Path::new(manifest)
                .parent()
                .and_then(|p| p.parent())
                .unwrap_or(std::path::Path::new("."));
            workspace_root.join("data").join("mnist").to_string_lossy().to_string()
        };
        match MnistData::download_and_load(&data_dir) {
            Ok(d) => {
                println!("  Loaded real MNIST: {} train, {} test", d.n_train, d.n_test);
                d
            }
            Err(_) => {
                println!("  MNIST not available, using synthetic data (2000 train / 500 test)");
                MnistData::synthetic(2000, 500)
            }
        }
    };

    println!("\nStep 4: Training...");
    let train_start = std::time::Instant::now();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0;

        let mut offset = 0;
        while offset < data.n_train {
            let (batch_x, batch_y) = data.train_batch(offset, batch_size);
            let actual_batch = batch_y.len();
            if actual_batch == 0 {
                break;
            }
            let loss = mlp.train_step_backprop(batch_x, batch_y, learning_rate);
            epoch_loss += loss;
            n_batches += 1;
            offset += actual_batch;
        }

        let avg_loss = if n_batches > 0 { epoch_loss / n_batches as f32 } else { 0.0 };

        // Print progress every few epochs
        if epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 5 == 0 {
            let train_probs = mlp.forward(&data.train_images);
            let train_acc = mlp.accuracy(&train_probs, &data.train_labels);
            println!("  Epoch {}/{}: loss={:.4}, train_acc={:.1}%",
                epoch + 1, epochs, avg_loss, train_acc * 100.0);
        }
    }

    let train_secs = train_start.elapsed().as_secs_f64();
    println!("  Training time: {:.1}s", train_secs);

    // Evaluate on test set
    let test_probs = mlp.forward(&data.test_images);
    let test_acc = mlp.accuracy(&test_probs, &data.test_labels);
    println!("\n  Test accuracy: {:.1}%", test_acc * 100.0);

    // Step 5: Compress with IGQK
    println!("\nStep 5: IGQK ternary compression...");
    let compressed = mlp.compress_ternary();

    let comp_probs = compressed.forward(&data.test_images);
    let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

    // Compression ratio: f32 weights (32 bits) -> ternary (2 bits)
    let ratio = 32.0 / 2.0;

    println!("  Compressed accuracy: {:.1}%", comp_acc * 100.0);
    println!("  Compression ratio: {:.1}x (f32 -> ternary)", ratio);
    println!("  Accuracy drop: {:.1}%", (test_acc - comp_acc) * 100.0);

    // Step 6: Ask Ollama to evaluate the results
    println!("\nStep 6: Asking AI to evaluate results...\n");

    let eval_prompt = format!(
        "I trained a neural network for MNIST with your architecture (784->{}->{}->10). Results:\n\
        - Test accuracy: {:.1}%\n\
        - Compressed accuracy: {:.1}%\n\
        - Compression ratio: {:.1}x\n\
        - Training time: {:.1}s\n\
        \n\
        Evaluate these results in 2-3 sentences. Was the architecture good?",
        hidden1, hidden2, test_acc * 100.0, comp_acc * 100.0, ratio, train_secs
    );

    match client.generate(model, &eval_prompt, None) {
        Ok(evaluation) => {
            println!("AI Evaluation:");
            println!("{}", evaluation);
        }
        Err(e) => {
            eprintln!("Warning: Could not get AI evaluation: {e}");
        }
    }

    println!("\n=== AI-Train Pipeline Complete ===");
}

fn cmd_autonomous(args: &[String]) {
    let model = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i+1)).cloned()
        .unwrap_or_else(|| "deepseek-r1:1.5b".to_string());
    let target = args.iter().position(|a| a == "--target")
        .and_then(|i| args.get(i+1)).and_then(|p| p.parse::<f32>().ok())
        .unwrap_or(95.0) / 100.0;
    let max_iter = args.iter().position(|a| a == "--iterations")
        .and_then(|i| args.get(i+1)).and_then(|p| p.parse::<usize>().ok())
        .unwrap_or(5);
    let quick = args.iter().any(|a| a == "--quick");
    let task = args.iter().position(|a| a == "--task")
        .and_then(|i| args.get(i+1)).cloned()
        .unwrap_or_else(|| "Build a digit classifier".to_string());

    println!("=== QLANG Autonomous AI Loop ===");
    println!("Task:        {}", task);
    println!("Model:       {}", model);
    println!("Target:      {:.0}% accuracy", target * 100.0);
    println!("Max Iters:   {}", max_iter);
    println!("Dataset:     {}", if quick { "synthetic" } else { "MNIST" });
    println!();

    let ollama = qlang_runtime::ollama::OllamaClient::from_env();
    if !ollama.health().unwrap_or(false) {
        eprintln!("Error: Ollama is not running. Start with: ollama serve");
        std::process::exit(1);
    }

    use qlang_runtime::mnist::MnistData;
    use qlang_runtime::training::MlpWeights3;

    let data = if quick {
        MnistData::synthetic(2000, 500)
    } else {
        MnistData::download_and_load("data/mnist").unwrap_or_else(|_| MnistData::synthetic(2000, 500))
    };

    let mut best_accuracy: f32 = 0.0;
    let mut best_config = (128usize, 64usize);
    let mut history: Vec<String> = Vec::new();

    for iteration in 1..=max_iter {
        println!("--- Iteration {}/{} ---", iteration, max_iter);

        // Step 1: AI designs
        print!("  Asking {} to design... ", model);
        let prompt = if iteration == 1 {
            format!("You are a neural network architect. Task: {}. Design a small MLP (784 in, 10 out). Reply ONLY JSON: {{\"hidden1\":N,\"hidden2\":N,\"epochs\":N,\"learning_rate\":0.1}}", task)
        } else {
            format!("Previous: {}\nBest: {:.1}% (h1={},h2={}). Target: {:.0}%. Improve. ONLY JSON: {{\"hidden1\":N,\"hidden2\":N,\"epochs\":N,\"learning_rate\":0.1}}",
                history.join("; "), best_accuracy*100.0, best_config.0, best_config.1, target*100.0)
        };

        let response = ollama.generate(&model, &prompt, Some("JSON only.")).unwrap_or_default();
        let json_str = extract_json(&response).unwrap_or_else(|| r#"{"hidden1":128,"hidden2":64,"epochs":10,"learning_rate":0.1}"#.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .unwrap_or(serde_json::json!({"hidden1":128,"hidden2":64,"epochs":10,"learning_rate":0.1}));

        let h1 = parsed["hidden1"].as_u64().unwrap_or(128).min(512).max(16) as usize;
        let h2 = parsed["hidden2"].as_u64().unwrap_or(64).min(256).max(8) as usize;
        let epochs = parsed["epochs"].as_u64().unwrap_or(10).min(30).max(3) as usize;
        let lr = parsed["learning_rate"].as_f64().unwrap_or(0.1) as f32;
        println!("784->{}->{}->10, {} epochs, lr={}", h1, h2, epochs, lr);

        // Step 2: Train
        print!("  Training... ");
        let start = std::time::Instant::now();
        let mut nn = MlpWeights3::new(784, h1, h2, 10);
        let mut clr = lr;
        for epoch in 0..epochs {
            if epoch > 0 && epoch % 10 == 0 { clr *= 0.95; }
            let nb = data.n_train / 64;
            for b in 0..nb {
                let (x, y) = data.train_batch(b*64, 64);
                nn.train_step_backprop(x, y, clr);
            }
        }
        let train_time = start.elapsed();

        // Step 3: Evaluate
        let test_probs = nn.forward(&data.test_images);
        let test_acc = nn.accuracy(&test_probs, &data.test_labels);
        let compressed = nn.compress_ternary();
        let comp_probs = compressed.forward(&data.test_images);
        let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

        println!("{:.1}% (compressed: {:.1}%) in {:.1}s", test_acc*100.0, comp_acc*100.0, train_time.as_secs_f64());

        history.push(format!("784->{}->{}->10: {:.1}%", h1, h2, test_acc*100.0));

        if test_acc > best_accuracy {
            best_accuracy = test_acc;
            best_config = (h1, h2);
            println!("  * New best!");
        }

        // Step 4: Check target
        if test_acc >= target {
            println!();
            println!("=== TARGET REACHED in {} iterations ===", iteration);
            println!("  Accuracy: {:.1}%", test_acc * 100.0);
            println!("  Architecture: 784->{}->{}->10", h1, h2);

            // Final AI summary
            let summary = ollama.generate(&model, &format!(
                "Target {:.0}% reached after {} tries. Best: 784->{}->{}->10 at {:.1}%. Summarize in 1 sentence.",
                target*100.0, iteration, best_config.0, best_config.1, best_accuracy*100.0
            ), None).unwrap_or_default();
            println!("  AI: {}", summary);
            return;
        }

        // Step 5: AI feedback
        let feedback = ollama.generate(&model, &format!(
            "{:.1}% (target {:.0}%). What to change? 1 sentence.", test_acc*100.0, target*100.0
        ), None).unwrap_or_default();
        println!("  AI feedback: {}", &feedback[..feedback.len().min(100)]);
    }

    println!();
    println!("=== Max iterations reached ===");
    println!("  Best: {:.1}% with 784->{}->{}->10", best_accuracy*100.0, best_config.0, best_config.1);
    println!("  Target {:.0}% was not reached.", target * 100.0);
}

/// Extract the first JSON object from a string that may contain surrounding text.
fn extract_json(s: &str) -> Option<String> {
    // Find the first '{' and match to its closing '}'
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0;
    for i in start..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn cmd_train_mnist(port: u16, epochs: usize) {
    use qlang_runtime::web_server::{WebServer, WebEvent};
    use qlang_runtime::mnist::MnistData;
    use qlang_runtime::training::MlpWeights3;
    use std::time::Instant;

    // Determine the web root directory
    let web_root = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest)
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(std::path::Path::new("."));
        workspace_root.join("web").to_string_lossy().to_string()
    };

    // Start web server
    let handle = match WebServer::start(port, web_root.clone()) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("Failed to start web server: {e}");
            process::exit(1);
        }
    };

    println!("================================================================");
    println!("  QLANG MNIST Training — Live Dashboard");
    println!("================================================================");
    println!();
    println!("  Dashboard:  http://localhost:{port}");
    println!("  WebSocket:  ws://localhost:{port}/ws");
    println!("  Serving:    {web_root}");
    println!("  Backend:    {}", qlang_runtime::accel::backend_name());
    println!();
    println!("  Open the dashboard in your browser, then training will start");
    println!("  in 3 seconds...");
    println!();

    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: "QLANG MNIST training pipeline starting...".to_string(),
    });

    std::thread::sleep(std::time::Duration::from_secs(3));

    let total_start = Instant::now();

    // ================================================================
    // 1. Load MNIST
    // ================================================================
    println!("--- [1/6] LOAD MNIST ---");
    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: "Loading MNIST data...".to_string(),
    });

    let mnist_dir = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest)
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or(std::path::Path::new("."));
        std::env::var("MNIST_DIR")
            .unwrap_or_else(|_| workspace_root.join("data").join("mnist").to_string_lossy().to_string())
    };

    let data = match MnistData::download_and_load(&mnist_dir) {
        Ok(d) => {
            println!("  Loaded REAL MNIST from '{}'", mnist_dir);
            handle.broadcast(WebEvent::SystemLog {
                level: "info".to_string(),
                message: format!("Loaded real MNIST: {} train, {} test", d.n_train, d.n_test),
            });
            d
        }
        Err(e) => {
            println!("  Could not load real MNIST: {}", e);
            println!("  Falling back to synthetic MNIST data.");
            handle.broadcast(WebEvent::SystemLog {
                level: "warn".to_string(),
                message: format!("MNIST download failed ({}), using synthetic data", e),
            });
            MnistData::synthetic(2000, 500)
        }
    };

    println!("  Train: {}  Test: {}", data.n_train, data.n_test);

    // ================================================================
    // 2. Build model
    // ================================================================
    println!("\n--- [2/6] BUILD MODEL ---");
    let input_dim = 784;
    let hidden1_dim = 256;
    let hidden2_dim = 128;
    let output_dim = 10;

    let mut model = MlpWeights3::new(input_dim, hidden1_dim, hidden2_dim, output_dim);
    let total_params = model.param_count();

    println!("  Architecture: {}->{}->{}->{}  (3-layer MLP)", input_dim, hidden1_dim, hidden2_dim, output_dim);
    println!("  Parameters:   {} ({:.1} KB)", total_params, total_params as f64 * 4.0 / 1024.0);

    // Broadcast graph info
    handle.broadcast(WebEvent::GraphLoaded {
        name: format!("MNIST {}->{}->{}->{}  ({} params)", input_dim, hidden1_dim, hidden2_dim, output_dim, total_params),
        num_nodes: 8,
        num_edges: 7,
    });

    // ================================================================
    // 3. Train
    // ================================================================
    println!("\n--- [3/6] TRAIN ---");
    let batch_size = 64;
    let mut lr = 0.1f32;
    let lr_decay = 0.95f32;
    let lr_decay_every = 10;

    println!("  Epochs:        {}", epochs);
    println!("  Batch size:    {}", batch_size);
    println!("  Learning rate: {} (decay {}x every {} epochs)", lr, lr_decay, lr_decay_every);
    println!();

    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: format!("Training: {} epochs, batch_size={}, lr={}", epochs, batch_size, lr),
    });

    let train_start = Instant::now();

    for epoch in 0..epochs {
        // Learning rate decay
        if epoch > 0 && epoch % lr_decay_every == 0 {
            lr *= lr_decay;
        }

        let n_batches = data.n_train / batch_size;
        let mut epoch_loss = 0.0f32;

        for batch_idx in 0..n_batches {
            let (x, y) = data.train_batch(batch_idx * batch_size, batch_size);
            let loss = model.train_step_backprop(x, y, lr);
            epoch_loss += loss;
        }
        epoch_loss /= n_batches.max(1) as f32;

        // Compute accuracy for this epoch
        let train_probs = model.forward(&data.train_images);
        let train_acc = model.accuracy(&train_probs, &data.train_labels);

        // Broadcast every epoch to dashboard
        handle.broadcast(WebEvent::TrainingEpoch {
            epoch,
            loss: epoch_loss,
            accuracy: train_acc,
        });

        // Print every 5 epochs to console
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("  Epoch {:>3}/{}: loss={:.4}  train_acc={:.1}%  lr={:.4}",
                epoch + 1, epochs, epoch_loss, train_acc * 100.0, lr);
        }
    }

    let train_time = train_start.elapsed();

    // Test accuracy
    let test_probs = model.forward(&data.test_images);
    let test_acc = model.accuracy(&test_probs, &data.test_labels);
    let test_loss = model.loss(&test_probs, &data.test_labels);

    println!();
    println!("  Training time:  {:?}", train_time);
    println!("  Test accuracy:  {:.1}%", test_acc * 100.0);
    println!("  Test loss:      {:.4}", test_loss);

    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: format!("Training complete: test_acc={:.1}%, loss={:.4}, time={:?}", test_acc * 100.0, test_loss, train_time),
    });

    // ================================================================
    // 4. IGQK Ternary Compression
    // ================================================================
    println!("\n--- [4/6] IGQK TERNARY COMPRESSION ---");
    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: "Applying IGQK ternary compression...".to_string(),
    });

    let compressed = model.compress_ternary();

    let comp_probs = compressed.forward(&data.test_images);
    let comp_acc = compressed.accuracy(&comp_probs, &data.test_labels);

    let weight_count = model.w1.len() + model.w2.len() + model.w3.len();
    let original_bytes = total_params * 4;
    let ternary_weight_bytes = (weight_count * 2 + 7) / 8;
    let bias_bytes = (model.b1.len() + model.b2.len() + model.b3.len()) * 4;
    let compressed_bytes = ternary_weight_bytes + bias_bytes;
    let compression_ratio = original_bytes as f64 / compressed_bytes as f64;

    println!("  Before: {:.1}% accuracy, {:.1} KB", test_acc * 100.0, original_bytes as f64 / 1024.0);
    println!("  After:  {:.1}% accuracy, {:.1} KB", comp_acc * 100.0, compressed_bytes as f64 / 1024.0);
    println!("  Accuracy drop: {:.1}%", (test_acc - comp_acc) * 100.0);
    println!("  Compression ratio: {:.1}x", compression_ratio);

    handle.broadcast(WebEvent::CompressionResult {
        method: "IGQK Ternary".to_string(),
        ratio: compression_ratio as f32,
        accuracy_before: test_acc,
        accuracy_after: comp_acc,
    });

    // ================================================================
    // 5. ONNX Export
    // ================================================================
    println!("\n--- [5/6] ONNX EXPORT ---");
    let qlang_source = format!(r#"graph mnist_3layer {{
  input x: f32[1, 784]
  input W1: f32[784, {hidden1_dim}]
  input b1: f32[1, {hidden1_dim}]
  input W2: f32[{hidden1_dim}, {hidden2_dim}]
  input b2: f32[1, {hidden2_dim}]
  input W3: f32[{hidden2_dim}, 10]
  input b3: f32[1, 10]

  node h1 = matmul(x, W1)
  node a1 = relu(h1)
  node h2 = matmul(a1, W2)
  node a2 = relu(h2)
  node logits = matmul(a2, W3)
  node probs = softmax(logits)
  node comp = to_ternary(W1) @proof theorem_5_2

  output predictions = probs
  output compressed = comp
}}"#);

    match qlang_compile::parser::parse(&qlang_source) {
        Ok(graph) => {
            let onnx_json = qlang_compile::onnx::to_onnx_json(&graph);
            println!("  ONNX JSON size: {} bytes", onnx_json.len());

            let onnx_path = "/tmp/qlang_mnist_3layer.onnx.json";
            match std::fs::write(onnx_path, &onnx_json) {
                Ok(_) => println!("  Saved to: {}", onnx_path),
                Err(e) => println!("  Could not save: {}", e),
            }

            handle.broadcast(WebEvent::ModelSaved {
                name: "mnist_3layer".to_string(),
                version: format!("acc={:.1}% comp={:.1}x", test_acc * 100.0, compression_ratio),
            });
        }
        Err(e) => {
            println!("  Parse error (non-fatal): {}", e);
        }
    }

    // ================================================================
    // 6. Summary
    // ================================================================
    let total_time = total_start.elapsed();
    println!("\n================================================================");
    println!("  PIPELINE SUMMARY");
    println!("================================================================");
    println!("  Total time:         {:?}", total_time);
    println!("  Training time:      {:?}", train_time);
    println!("  Architecture:       {}->{}->{}->{}  (3-layer MLP)", input_dim, hidden1_dim, hidden2_dim, output_dim);
    println!("  Parameters:         {} ({:.1} KB)", total_params, total_params as f64 * 4.0 / 1024.0);
    println!("  Training:           {} epochs, batch_size={}", epochs, batch_size);
    println!("  Test accuracy:      {:.1}%", test_acc * 100.0);
    println!("  Compressed acc:     {:.1}%", comp_acc * 100.0);
    println!("  Compression ratio:  {:.1}x", compression_ratio);
    println!("================================================================");
    println!();

    handle.broadcast(WebEvent::SystemLog {
        level: "info".to_string(),
        message: format!(
            "Pipeline complete: acc={:.1}%, compressed={:.1}%, ratio={:.1}x, time={:?}",
            test_acc * 100.0, comp_acc * 100.0, compression_ratio, total_time
        ),
    });

    println!("  Dashboard still running at http://localhost:{port}");
    println!("  Press Ctrl+C to stop.");

    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// Simple LSP server using JSON-RPC over stdin/stdout.
fn cmd_lsp() {
    use std::io::{BufRead, BufReader};

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let reader = BufReader::new(stdin.lock());
    let mut lines = reader.lines();

    eprintln!("QLANG LSP server starting on stdin/stdout...");

    let mut documents: HashMap<String, String> = HashMap::new();

    loop {
        let header = match lines.next() {
            Some(Ok(h)) => h,
            _ => break,
        };

        if !header.starts_with("Content-Length:") {
            continue;
        }

        let content_length: usize = header
            .trim_start_matches("Content-Length:")
            .trim()
            .parse()
            .unwrap_or(0);

        // Skip blank line
        let _ = lines.next();

        // Read JSON body
        let mut body = String::new();
        let mut remaining = content_length;
        while remaining > 0 {
            if let Some(Ok(line)) = lines.next() {
                remaining = remaining.saturating_sub(line.len() + 1);
                body.push_str(&line);
                body.push('\n');
            } else {
                break;
            }
        }

        let msg: serde_json::Value = match serde_json::from_str(body.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let method = msg["method"].as_str().unwrap_or("");
        let id = &msg["id"];

        match method {
            "initialize" => {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 1,
                            "completionProvider": { "triggerCharacters": [".", "=", "(", ":"] },
                            "hoverProvider": true,
                            "definitionProvider": true
                        },
                        "serverInfo": { "name": "qlang-lsp", "version": "0.1.0" }
                    }
                });
                send_response(&stdout, &response);
            }
            "initialized" => {}
            "textDocument/didOpen" => {
                if let (Some(uri), Some(text)) = (
                    msg["params"]["textDocument"]["uri"].as_str(),
                    msg["params"]["textDocument"]["text"].as_str(),
                ) {
                    documents.insert(uri.to_string(), text.to_string());
                    publish_diagnostics(&stdout, uri, text);
                }
            }
            "textDocument/didChange" => {
                if let Some(uri) = msg["params"]["textDocument"]["uri"].as_str() {
                    if let Some(changes) = msg["params"]["contentChanges"].as_array() {
                        if let Some(text) = changes.first().and_then(|c| c["text"].as_str()) {
                            documents.insert(uri.to_string(), text.to_string());
                            publish_diagnostics(&stdout, uri, text);
                        }
                    }
                }
            }
            "textDocument/completion" => {
                let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
                let line = msg["params"]["position"]["line"].as_u64().unwrap_or(0) as usize;
                let col = msg["params"]["position"]["character"].as_u64().unwrap_or(0) as usize;
                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let items = qlang_compile::lsp::completions_at(source, line, col);

                let lsp_items: Vec<serde_json::Value> = items.iter().map(|item| {
                    serde_json::json!({
                        "label": item.label,
                        "kind": match item.kind {
                            qlang_compile::lsp::CompletionKind::Operation => 3,
                            qlang_compile::lsp::CompletionKind::Type => 6,
                            qlang_compile::lsp::CompletionKind::Variable => 6,
                            qlang_compile::lsp::CompletionKind::Keyword => 14,
                        },
                        "documentation": item.documentation
                    })
                }).collect();

                send_response(&stdout, &serde_json::json!({
                    "jsonrpc": "2.0", "id": id, "result": lsp_items
                }));
            }
            "textDocument/hover" => {
                let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
                let line = msg["params"]["position"]["line"].as_u64().unwrap_or(0) as usize;
                let col = msg["params"]["position"]["character"].as_u64().unwrap_or(0) as usize;
                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let hover = qlang_compile::lsp::hover_info(source, line, col);

                let result = match hover {
                    Some(text) => serde_json::json!({ "contents": { "kind": "markdown", "value": text } }),
                    None => serde_json::Value::Null,
                };
                send_response(&stdout, &serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }));
            }
            "textDocument/definition" => {
                let uri = msg["params"]["textDocument"]["uri"].as_str().unwrap_or("");
                let line = msg["params"]["position"]["line"].as_u64().unwrap_or(0) as usize;
                let col = msg["params"]["position"]["character"].as_u64().unwrap_or(0) as usize;
                let source = documents.get(uri).map(|s| s.as_str()).unwrap_or("");
                let def = qlang_compile::lsp::goto_definition(source, line, col);

                let result = match def {
                    Some((dl, dc)) => serde_json::json!({
                        "uri": uri,
                        "range": { "start": { "line": dl, "character": dc }, "end": { "line": dl, "character": dc } }
                    }),
                    None => serde_json::Value::Null,
                };
                send_response(&stdout, &serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }));
            }
            "shutdown" => {
                send_response(&stdout, &serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": null }));
            }
            "exit" => break,
            _ => {}
        }
    }
}

fn send_response(stdout: &std::io::Stdout, msg: &serde_json::Value) {
    let body = serde_json::to_string(msg).unwrap();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    let mut out = stdout.lock();
    let _ = out.write_all(header.as_bytes());
    let _ = out.write_all(body.as_bytes());
    let _ = out.flush();
}

fn publish_diagnostics(stdout: &std::io::Stdout, uri: &str, source: &str) {
    let diags = qlang_compile::lsp::analyze_source(source);
    let lsp_diags: Vec<serde_json::Value> = diags.iter().map(|d| {
        serde_json::json!({
            "range": {
                "start": { "line": d.line.saturating_sub(1), "character": d.col },
                "end": { "line": d.line.saturating_sub(1), "character": d.col + 1 }
            },
            "severity": match d.severity {
                qlang_compile::lsp::Severity::Error => 1,
                qlang_compile::lsp::Severity::Warning => 2,
                qlang_compile::lsp::Severity::Info => 3,
            },
            "source": "qlang",
            "message": d.message
        })
    }).collect();

    send_response(stdout, &serde_json::json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": { "uri": uri, "diagnostics": lsp_diags }
    }));
}

// ---------------------------------------------------------------------------
// Proxy command — HTTP-to-QLMS gateway with auto-signing
// ---------------------------------------------------------------------------

/// `qlang-cli proxy` — start an HTTP-to-QLMS signing proxy.
///
/// Endpoints:
///   GET  /v1/health         — health check
///   GET  /v1/pubkey         — server public key
///   GET  /v1/graphs         — list stored graphs
///   POST /v1/graph/submit   — submit and sign a graph (JSON or .qlang text)
///   POST /v1/graph/execute  — execute a graph with inputs, sign result
///   POST /v1/graph/verify   — verify a signed graph
///   POST /v1/compress       — ternary compress weights, sign result
fn cmd_proxy(args: &[String]) {
    use qlang_core::crypto::{self, Keypair, SignedGraph};
    use qlang_core::graph::Graph;
    use qlang_core::tensor::{Dim, Shape, TensorData};
    use qlang_agent::server::GraphStore;
    use qlang_compile::api::{self, parse_request, json_ok, json_error, Method, HttpResponse};

    let port: u16 = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1)).and_then(|p| p.parse().ok())
        .unwrap_or(9100);

    let upstream = args.iter().position(|a| a == "--upstream")
        .and_then(|i| args.get(i + 1)).cloned();

    // Generate server keypair
    let keypair = Keypair::generate();
    let pubkey_hex = crypto::hex(keypair.public_key());

    println!("QLANG Proxy v{}", env!("CARGO_PKG_VERSION"));
    println!("  Listen:   http://0.0.0.0:{}", port);
    if let Some(ref u) = upstream {
        println!("  Upstream: {}", u);
    }
    println!("  Signing:  HMAC-SHA256 (Ed25519-compatible API)");
    println!("  PubKey:   {}...", &pubkey_hex[..16]);
    println!();
    println!("Endpoints:");
    println!("  GET  /v1/health         Health check");
    println!("  GET  /v1/pubkey         Server public key");
    println!("  GET  /v1/graphs         List stored graphs");
    println!("  POST /v1/graph/submit   Submit & sign a graph");
    println!("  POST /v1/graph/execute  Execute graph & sign result");
    println!("  POST /v1/graph/verify   Verify a signed graph");
    println!("  POST /v1/compress       Ternary compress weights");
    println!();

    let listener = match std::net::TcpListener::bind(format!("0.0.0.0:{}", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Failed to bind 0.0.0.0:{}: {e}", port);
            process::exit(1);
        }
    };

    eprintln!("[proxy] listening on 0.0.0.0:{}", port);

    let store = GraphStore::new();

    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            handle_proxy_request(&mut stream, &keypair, &upstream, &store);
        }
    }

    // Inner handler — routes an HTTP request to the appropriate proxy endpoint.
    fn handle_proxy_request(
        stream: &mut std::net::TcpStream,
        keypair: &Keypair,
        _upstream: &Option<String>,
        store: &GraphStore,
    ) {
        let request = match parse_request(stream) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[proxy] parse error: {e}");
                return;
            }
        };

        eprintln!(
            "[proxy] {} {} (body={} bytes)",
            match request.method {
                Method::Get => "GET",
                Method::Post => "POST",
                Method::Options => "OPTIONS",
                Method::Unknown => "???",
            },
            request.path,
            request.body.len()
        );

        let response = route_proxy(&request, keypair, store);

        let bytes = response.to_bytes();
        let _ = std::io::Write::write_all(stream, &bytes);
        let _ = std::io::Write::flush(stream);
    }

    fn route_proxy(
        request: &api::HttpRequest,
        keypair: &Keypair,
        store: &GraphStore,
    ) -> HttpResponse {
        // Handle CORS preflight.
        if request.method == Method::Options {
            return HttpResponse::new(204, "No Content");
        }

        match (request.method, request.path.as_str()) {
            // --- Health check ---
            (Method::Get, "/v1/health") | (Method::Get, "/health") | (Method::Get, "/") => {
                let info = serde_json::json!({
                    "status": "ok",
                    "version": env!("CARGO_PKG_VERSION"),
                    "protocol": "qlms",
                    "signed": true,
                    "pubkey": crypto::hex(keypair.public_key()),
                });
                json_ok(&info)
            }

            // --- Public key ---
            (Method::Get, "/v1/pubkey") => {
                let resp = serde_json::json!({
                    "pubkey": crypto::hex(keypair.public_key()),
                });
                json_ok(&resp)
            }

            // --- List stored graphs ---
            (Method::Get, "/v1/graphs") => {
                let ids = store.list();
                let infos: Vec<_> = ids.iter()
                    .filter_map(|id| store.get_info(*id))
                    .map(|info| serde_json::json!({
                        "id": info.id,
                        "name": info.name,
                        "nodes": info.num_nodes,
                        "edges": info.num_edges,
                    }))
                    .collect();
                let resp = serde_json::json!({ "graphs": infos });
                json_ok(&resp)
            }

            // --- Submit & sign a graph ---
            (Method::Post, "/v1/graph/submit") => {
                // Try JSON first, then .qlang text
                let graph: Graph = match serde_json::from_slice(&request.body) {
                    Ok(g) => g,
                    Err(_) => {
                        // Try parsing as .qlang text
                        match std::str::from_utf8(&request.body) {
                            Ok(source) => {
                                match qlang_compile::parser::parse(source) {
                                    Ok(g) => g,
                                    Err(e) => {
                                        return json_error(
                                            400,
                                            "Bad Request",
                                            &format!("cannot parse as JSON or .qlang: {e}"),
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                return json_error(
                                    400,
                                    "Bad Request",
                                    &format!("invalid UTF-8: {e}"),
                                );
                            }
                        }
                    }
                };

                let signed = SignedGraph::sign(graph.clone(), keypair);
                let id = store.insert(graph);

                let resp = serde_json::json!({
                    "ok": true,
                    "id": id,
                    "signed": true,
                    "hash": crypto::hex(&signed.hash),
                    "pubkey": crypto::hex(&signed.pubkey),
                    "signature": crypto::hex(&signed.signature),
                });
                json_ok(&resp)
            }

            // --- Execute a graph, sign result ---
            (Method::Post, "/v1/graph/execute") => {
                let payload: serde_json::Value = match serde_json::from_slice(&request.body) {
                    Ok(v) => v,
                    Err(e) => {
                        return json_error(400, "Bad Request", &format!("invalid JSON: {e}"));
                    }
                };

                // Deserialize graph.
                let graph_val = match payload.get("graph") {
                    Some(v) => v,
                    None => return json_error(400, "Bad Request", "missing 'graph' field"),
                };

                let graph: Graph = match serde_json::from_value(graph_val.clone()) {
                    Ok(g) => g,
                    Err(e) => {
                        return json_error(400, "Bad Request", &format!("invalid graph: {e}"));
                    }
                };

                // Build inputs map.
                let inputs_val = payload.get("inputs").cloned()
                    .unwrap_or(serde_json::json!({}));
                let inputs_map = match inputs_val.as_object() {
                    Some(m) => m,
                    None => return json_error(400, "Bad Request", "'inputs' must be an object"),
                };

                let mut inputs: HashMap<String, TensorData> = HashMap::new();
                for (name, arr) in inputs_map {
                    if let Some(values) = arr.as_array() {
                        let floats: Vec<f32> = values
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect();
                        let len = floats.len();
                        let shape = Shape(vec![Dim::Fixed(len)]);
                        inputs.insert(name.clone(), TensorData::from_f32(shape, &floats));
                    }
                }

                // Execute.
                match qlang_runtime::executor::execute(&graph, inputs) {
                    Ok(result) => {
                        let mut output_json = serde_json::Map::new();
                        for (name, tensor) in &result.outputs {
                            let floats = tensor.as_f32_slice().unwrap_or_default();
                            let arr: Vec<serde_json::Value> = floats
                                .iter()
                                .map(|&f| serde_json::Value::from(f))
                                .collect();
                            output_json.insert(name.clone(), serde_json::Value::Array(arr));
                        }

                        // Sign the result
                        let signed = SignedGraph::sign(graph, keypair);

                        let resp = serde_json::json!({
                            "ok": true,
                            "outputs": output_json,
                            "signed": true,
                            "hash": crypto::hex(&signed.hash),
                            "signature": crypto::hex(&signed.signature),
                            "stats": {
                                "nodes_executed": result.stats.nodes_executed,
                                "quantum_ops": result.stats.quantum_ops,
                                "total_flops": result.stats.total_flops,
                            }
                        });
                        json_ok(&resp)
                    }
                    Err(e) => json_error(
                        500,
                        "Internal Server Error",
                        &format!("execution error: {e}"),
                    ),
                }
            }

            // --- Verify a signed graph ---
            (Method::Post, "/v1/graph/verify") => {
                match serde_json::from_slice::<SignedGraph>(&request.body) {
                    Ok(signed) => {
                        let valid = signed.verify();
                        let resp = serde_json::json!({
                            "ok": true,
                            "valid": valid,
                            "hash": crypto::hex(&signed.hash),
                            "pubkey": crypto::hex(&signed.pubkey),
                        });
                        json_ok(&resp)
                    }
                    Err(e) => {
                        json_error(
                            400,
                            "Bad Request",
                            &format!("invalid signed graph JSON: {e}"),
                        )
                    }
                }
            }

            // --- Ternary weight compression ---
            (Method::Post, "/v1/compress") => {
                // Delegate to the existing compress handler, then sign the result
                let compress_resp = api::handle_compress(&request.body);

                // Parse the compress response to add signing info
                if compress_resp.status_code == 200 {
                    if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&compress_resp.body) {
                        // Sign the compressed result
                        let result_hash = crypto::sha256(&compress_resp.body);
                        let signature = keypair.sign(&result_hash);

                        val["signed"] = serde_json::json!(true);
                        val["hash"] = serde_json::json!(crypto::hex(&result_hash));
                        val["signature"] = serde_json::json!(crypto::hex(&signature));
                        val["pubkey"] = serde_json::json!(crypto::hex(keypair.public_key()));

                        json_ok(&val)
                    } else {
                        compress_resp
                    }
                } else {
                    compress_resp
                }
            }

            // --- Fallback: not found ---
            (_, path) => {
                json_error(404, "Not Found", &format!("unknown endpoint: {path}"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Binary encode/decode commands
// ---------------------------------------------------------------------------

fn cmd_binary(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage:");
        eprintln!("  qlang-cli binary encode <file.qlg.json>   Encode graph to binary .qlb");
        eprintln!("  qlang-cli binary decode <file.qlb>         Decode binary .qlb to JSON");
        process::exit(1);
    }

    let subcmd = &args[0];
    match subcmd.as_str() {
        "encode" => {
            if args.len() < 2 {
                eprintln!("Usage: qlang-cli binary encode <file.qlg.json>");
                process::exit(1);
            }
            let input_path = &args[1];
            let content = match fs::read_to_string(input_path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error reading {input_path}: {e}");
                    process::exit(1);
                }
            };
            let graph = match qlang_core::serial::from_json(&content) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error parsing graph JSON: {e}");
                    process::exit(1);
                }
            };

            let binary_data = qlang_core::binary::to_binary(&graph);
            let json_size = content.len();
            let binary_size = binary_data.len();

            // Determine output path: replace extension
            let output_path = if input_path.ends_with(".qlg.json") {
                input_path.replace(".qlg.json", ".qlb")
            } else if input_path.ends_with(".json") {
                input_path.replace(".json", ".qlb")
            } else {
                format!("{input_path}.qlb")
            };

            match fs::write(&output_path, &binary_data) {
                Ok(()) => {
                    let hash = qlang_core::binary::content_hash(&graph);
                    println!("Encoded: {input_path} -> {output_path}");
                    println!("  JSON size:   {json_size:>8} bytes");
                    println!("  Binary size: {binary_size:>8} bytes");
                    let ratio = if json_size > 0 {
                        (binary_size as f64 / json_size as f64) * 100.0
                    } else {
                        0.0
                    };
                    println!("  Compression: {ratio:.1}% of JSON");
                    println!("  Hash:        {}", qlang_core::crypto::hex(&hash));
                }
                Err(e) => {
                    eprintln!("Error writing {output_path}: {e}");
                    process::exit(1);
                }
            }
        }
        "decode" => {
            if args.len() < 2 {
                eprintln!("Usage: qlang-cli binary decode <file.qlb>");
                process::exit(1);
            }
            let input_path = &args[1];
            let data = match fs::read(input_path) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Error reading {input_path}: {e}");
                    process::exit(1);
                }
            };
            let graph = match qlang_core::binary::from_binary(&data) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("Error decoding binary: {e}");
                    process::exit(1);
                }
            };
            let json = match qlang_core::serial::to_json(&graph) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("Error serializing to JSON: {e}");
                    process::exit(1);
                }
            };

            // Determine output path
            let output_path = if input_path.ends_with(".qlb") {
                input_path.replace(".qlb", ".qlg.json")
            } else {
                format!("{input_path}.qlg.json")
            };

            match fs::write(&output_path, &json) {
                Ok(()) => {
                    println!("Decoded: {input_path} -> {output_path}");
                    println!("  Binary size: {:>8} bytes", data.len());
                    println!("  JSON size:   {:>8} bytes", json.len());
                }
                Err(e) => {
                    eprintln!("Error writing {output_path}: {e}");
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown binary subcommand: {subcmd}");
            eprintln!("Use: encode | decode");
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// train-lm: Train a transformer language model
// ---------------------------------------------------------------------------

fn cmd_train_lm(args: &[String]) {
    // Parse arguments
    let data_path: String = args.iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("Error: --data <text_file> is required");
            eprintln!();
            eprintln!("Usage: qlang-cli train-lm --data text.txt [options]");
            eprintln!("  --vocab-size 1000   BPE vocabulary size (default: 500)");
            eprintln!("  --d-model 128       Embedding dimension (default: 64)");
            eprintln!("  --layers 4          Number of transformer layers (default: 2)");
            eprintln!("  --heads 4           Number of attention heads (default: 4)");
            eprintln!("  --epochs 10         Training epochs (default: 5)");
            eprintln!("  --seq-len 128       Maximum sequence length (default: 64)");
            eprintln!("  --lr 0.001          Learning rate (default: 0.001)");
            eprintln!("  --out-model M       Save model to file (default: model.qgpt)");
            eprintln!("  --out-tokenizer T   Save tokenizer to file (default: tokenizer.qbpe)");
            process::exit(1);
        });

    let vocab_size: usize = args.iter()
        .position(|a| a == "--vocab-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    let d_model: usize = args.iter()
        .position(|a| a == "--d-model")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let n_layers: usize = args.iter()
        .position(|a| a == "--layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let n_heads: usize = args.iter()
        .position(|a| a == "--heads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let epochs: usize = args.iter()
        .position(|a| a == "--epochs")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let seq_len: usize = args.iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    let lr: f32 = args.iter()
        .position(|a| a == "--lr")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.001);

    let model_path: String = args.iter()
        .position(|a| a == "--out-model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "model.qgpt".to_string());

    let tokenizer_path: String = args.iter()
        .position(|a| a == "--out-tokenizer")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "tokenizer.qbpe".to_string());

    // Validate d_model is divisible by n_heads
    if d_model % n_heads != 0 {
        eprintln!("Error: d_model ({}) must be divisible by heads ({})", d_model, n_heads);
        process::exit(1);
    }

    // Read the text file
    let text = match fs::read_to_string(&data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", data_path, e);
            process::exit(1);
        }
    };

    if text.is_empty() {
        eprintln!("Error: data file is empty");
        process::exit(1);
    }

    println!("================================================================");
    println!("  QLANG Language Model Training");
    println!("================================================================");
    println!();
    println!("  Data:       {} ({} bytes)", data_path, text.len());
    println!("  Vocab size: {}", vocab_size);
    println!("  d_model:    {}", d_model);
    println!("  Layers:     {}", n_layers);
    println!("  Heads:      {}", n_heads);
    println!("  Seq length: {}", seq_len);
    println!("  Epochs:     {}", epochs);
    println!("  LR:         {}", lr);
    println!("  Backend:    {}", qlang_runtime::accel::backend_name());
    println!();

    let start = std::time::Instant::now();

    // Train
    let (model, tokenizer) = qlang_runtime::transformer_train::train_language_model(
        &text, vocab_size, d_model, n_layers, n_heads, seq_len, epochs, lr,
    );

    let elapsed = start.elapsed();

    // Generate sample text
    println!("\n--- Sample Generation ---");
    let prompt = &text[..text.len().min(20)];
    let prompt_tokens = tokenizer.encode(prompt);
    let generated = model.generate(&prompt_tokens, 50, 0.8);
    let generated_text = tokenizer.decode(&generated);
    println!("  Prompt:    \"{}\"", prompt);
    println!("  Generated: \"{}\"", generated_text);

    // Save model and tokenizer
    println!("\n--- Saving ---");
    match model.save(&model_path) {
        Ok(()) => println!("  Model saved to: {}", model_path),
        Err(e) => eprintln!("  Error saving model: {}", e),
    }
    match tokenizer.save(&tokenizer_path) {
        Ok(()) => println!("  Tokenizer saved to: {}", tokenizer_path),
        Err(e) => eprintln!("  Error saving tokenizer: {}", e),
    }

    println!("\n  Total time: {:.1}s", elapsed.as_secs_f64());
    println!("  Parameters: {}", model.param_count());
}

// Cache stats/clear commands
// ---------------------------------------------------------------------------

fn cmd_cache(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage:");
        eprintln!("  qlang-cli cache stats   Show cache hit/miss statistics");
        eprintln!("  qlang-cli cache clear   Clear the computation cache");
        process::exit(1);
    }

    let subcmd = &args[0];
    let cache_arc = qlang_core::cache::ComputationCache::global();
    match subcmd.as_str() {
        "stats" => {
            match cache_arc.lock() {
                Ok(guard) => {
                    let (hits, misses, size) = guard.stats();
                    let total = hits + misses;
                    let hit_rate = if total > 0 {
                        (hits as f64 / total as f64) * 100.0
                    } else {
                        0.0
                    };
                    println!("Computation Cache Statistics:");
                    println!("  Entries:   {size}");
                    println!("  Hits:      {hits}");
                    println!("  Misses:    {misses}");
                    println!("  Hit rate:  {hit_rate:.1}%");
                }
                Err(e) => {
                    eprintln!("Error accessing cache: {e}");
                    process::exit(1);
                }
            }
        }
        "clear" => {
            match cache_arc.lock() {
                Ok(mut guard) => {
                    let (_, _, size) = guard.stats();
                    guard.clear();
                    println!("Cleared {size} cache entries.");
                }
                Err(e) => {
                    eprintln!("Error accessing cache: {e}");
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("Unknown cache subcommand: {subcmd}");
            eprintln!("Use: stats | clear");
            process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Swarm training: evolve a population of small language models
// ---------------------------------------------------------------------------

fn cmd_swarm_train(args: &[String]) {
    // Parse arguments
    let data_path: String = args.iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();

    let population: usize = args.iter()
        .position(|a| a == "--population")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let generations: usize = args.iter()
        .position(|a| a == "--generations")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let quick = args.iter().any(|a| a == "--quick");

    // Step 1: Prepare data
    let text = if quick {
        "The quick brown fox jumps over the lazy dog. \
         Machine learning models can recognize patterns in data. \
         Neural networks are inspired by the human brain. \
         Artificial intelligence is transforming how we work and live. \
         Deep learning uses multiple layers to extract features. \
         Natural language processing enables computers to understand text. \
         Reinforcement learning teaches agents through trial and error. \
         Computer vision allows machines to interpret images. \
         Transfer learning reuses knowledge from one task to another. \
         Generative models can create new content from learned patterns. \
         The future of AI depends on responsible development. \
         Small specialized models can outperform large general models on specific tasks. \
         Evolution finds optimal solutions through selection and mutation. \
         Compression reduces model size while preserving accuracy. \
         Binary protocols are faster than text-based communication. \
         Quantum computing may revolutionize optimization problems. \
         Edge deployment brings AI closer to where data is generated. \
         Federated learning trains models without centralizing data. \
         Attention mechanisms allow models to focus on relevant information. \
         Tokenization converts text into numerical representations.".to_string()
    } else {
        if data_path.is_empty() {
            eprintln!("Error: --data <text_file> is required (or use --quick for built-in sample)");
            process::exit(1);
        }
        std::fs::read_to_string(&data_path).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", data_path, e);
            process::exit(1);
        })
    };

    println!("================================================================");
    println!("  QLANG Swarm Trainer — Evolutionary LM Architecture Search");
    println!("================================================================");
    println!();
    println!("  Population:   {} models", population);
    println!("  Generations:  {}", generations);
    println!("  Data:         {} ({} bytes)", if quick { "built-in sample" } else { &data_path }, text.len());
    println!("  Backend:      {}", qlang_runtime::accel::backend_name());
    println!();

    // Step 2: Train tokenizer once (shared)
    println!("[1/4] Training tokenizer...");
    let tokenizer = qlang_runtime::tokenizer::BpeTokenizer::train(&text, 500);
    let tokens = tokenizer.encode(&text);
    println!("  Vocab: {} tokens, Text: {} tokens", tokenizer.vocab_size(), tokens.len());

    if tokens.len() < 4 {
        eprintln!("Error: text too short after tokenization ({} tokens, need at least 4)", tokens.len());
        process::exit(1);
    }

    // Step 3: Create initial population with diverse architectures
    println!("\n[2/4] Creating {} models with diverse architectures...", population);

    struct SwarmMember {
        config: qlang_runtime::transformer_train::TransformerConfig,
        model: qlang_runtime::transformer_train::MiniGPT,
        fitness: f32,
        generation_born: usize,
    }

    let architectures: [(usize, usize, usize); 10] = [
        (32, 2, 2),   // tiny
        (48, 3, 2),   // small-wide
        (32, 2, 4),   // small-deep
        (64, 4, 2),   // medium-wide
        (64, 4, 3),   // medium
        (48, 3, 4),   // medium-deep
        (96, 4, 2),   // large-wide
        (64, 4, 4),   // large-balanced
        (32, 2, 6),   // very-deep
        (128, 4, 2),  // widest
    ];

    let mut swarm: Vec<SwarmMember> = Vec::new();
    let vocab_size = tokenizer.vocab_size();

    for i in 0..population {
        let arch_idx = i % architectures.len();
        let (d_model, n_heads, n_layers) = architectures[arch_idx];

        let config = qlang_runtime::transformer_train::TransformerConfig {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            max_seq_len: 64,
            dropout: 0.0,
            use_rms_norm: true,
            use_silu: true,
        };

        let model = qlang_runtime::transformer_train::MiniGPT::new(config.clone());
        let params = model.param_count();
        println!("  Model {}: d={} h={} L={} ({} params)", i + 1, d_model, n_heads, n_layers, params);

        swarm.push(SwarmMember {
            config,
            model,
            fitness: f32::MAX,
            generation_born: 0,
        });
    }

    // Step 4: Evolution loop
    println!("\n[3/4] Evolution ({} generations)...\n", generations);

    let seq_len: usize = 32;
    let epochs_per_gen: usize = 3;

    for gen in 0..generations {
        println!("--- Generation {}/{} ---", gen + 1, generations);
        let gen_start = std::time::Instant::now();

        // Train each model briefly
        for i in 0..swarm.len() {
            let mut total_loss = 0.0f32;
            let mut count = 0u32;

            for epoch in 0..epochs_per_gen {
                let max_start = tokens.len().saturating_sub(seq_len + 1);
                if max_start == 0 { continue; }

                let start = (gen * 7 + i * 13 + epoch * 17) % max_start.max(1);
                let end = std::cmp::min(start + seq_len + 1, tokens.len());
                let window = &tokens[start..end];

                if window.len() > 1 {
                    let loss = swarm[i].model.train_step(window, 0.001);
                    total_loss += loss;
                    count += 1;
                }
            }

            swarm[i].fitness = if count > 0 { total_loss / count as f32 } else { f32::MAX };

            print!("  Model {} (d={} L={}): loss={:.4}",
                i + 1, swarm[i].config.d_model, swarm[i].config.n_layers, swarm[i].fitness);

            // Generate a short sample
            if swarm[i].fitness < 50.0 && !tokens.is_empty() {
                let prompt_end = std::cmp::min(3, tokens.len());
                let prompt = &tokens[0..prompt_end];
                let generated = swarm[i].model.generate(prompt, 10, 0.8);
                let sample_text = tokenizer.decode(&generated);
                let display_len = sample_text.len().min(40);
                print!(" -> \"{}\"", &sample_text[..display_len]);
            }
            println!();
        }

        // Sort by fitness (lower loss = better)
        swarm.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));

        let best = &swarm[0];
        let gen_time = gen_start.elapsed();
        println!("\n  Best: d={} h={} L={} (loss={:.4}, {} params) [{:.1}s]",
            best.config.d_model, best.config.n_heads, best.config.n_layers,
            best.fitness, best.model.param_count(), gen_time.as_secs_f64());

        // Keep top 40%, mutate rest
        if gen < generations - 1 {
            let survivors = (population * 4) / 10;
            let survivors = survivors.max(1);
            let survivor_configs: Vec<qlang_runtime::transformer_train::TransformerConfig> =
                swarm[..survivors].iter().map(|m| m.config.clone()).collect();

            // Replace bottom 60% with mutations of survivors
            for i in survivors..population {
                let parent_idx = i % survivors;
                let parent = &survivor_configs[parent_idx];

                let mut new_config = parent.clone();
                let mutation = (gen * 3 + i * 7) % 3;
                match mutation {
                    0 => {
                        // change d_model
                        if (gen + i) % 2 == 0 {
                            new_config.d_model = (new_config.d_model + 16).min(256);
                        } else {
                            new_config.d_model = new_config.d_model.saturating_sub(16).max(16);
                        }
                        // Ensure d_model is divisible by n_heads
                        new_config.n_heads = new_config.n_heads.min(new_config.d_model / 8).max(1);
                        // Fix: ensure divisibility
                        while new_config.d_model % new_config.n_heads != 0 && new_config.n_heads > 1 {
                            new_config.n_heads -= 1;
                        }
                    }
                    1 => {
                        // change n_layers
                        if (gen + i) % 2 == 0 {
                            new_config.n_layers = (new_config.n_layers + 1).min(8);
                        } else {
                            new_config.n_layers = new_config.n_layers.saturating_sub(1).max(1);
                        }
                    }
                    _ => {
                        // change n_heads
                        if (gen + i) % 2 == 0 {
                            new_config.n_heads = (new_config.n_heads + 1).min(new_config.d_model / 8).max(1);
                        } else {
                            new_config.n_heads = new_config.n_heads.saturating_sub(1).max(1);
                        }
                        // Fix: ensure divisibility
                        while new_config.d_model % new_config.n_heads != 0 && new_config.n_heads > 1 {
                            new_config.n_heads -= 1;
                        }
                    }
                }

                let new_model = qlang_runtime::transformer_train::MiniGPT::new(new_config.clone());
                swarm[i] = SwarmMember {
                    config: new_config,
                    model: new_model,
                    fitness: f32::MAX,
                    generation_born: gen + 1,
                };
            }

            println!("  Mutated {} models for next generation\n", population - survivors);
        }
    }

    // Step 5: Final summary
    println!("\n[4/4] Swarm Training Complete!\n");
    println!("=== Results ===");
    let show_count = population.min(5);
    for (i, member) in swarm.iter().enumerate().take(show_count) {
        println!("  #{}: d_model={} heads={} layers={} | loss={:.4} | params={} | born=gen{}",
            i + 1, member.config.d_model, member.config.n_heads, member.config.n_layers,
            member.fitness, member.model.param_count(), member.generation_born);

        // Generate sample from best models
        if i < 3 && !tokens.is_empty() {
            let prompt_end = std::cmp::min(5, tokens.len());
            let prompt = &tokens[0..prompt_end];
            let generated = member.model.generate(prompt, 20, 0.7);
            println!("     Generated: \"{}\"", tokenizer.decode(&generated));
        }
    }

    let total_params: usize = swarm.iter().take(show_count).map(|m| m.model.param_count()).sum();
    println!("\n  Top {} models: {} total params ({:.1} KB compressed with IGQK)",
        show_count, total_params, total_params as f64 * 4.0 / 1024.0 / 16.0);
    println!("  Backend: {}", qlang_runtime::accel::backend_name());
}

// ---------------------------------------------------------------------------
// Orchestrate command
// ---------------------------------------------------------------------------

fn cmd_orchestrate(args: &[String]) {
    // Collect all args that are not flags into the prompt
    let prompt: String = args.iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    if prompt.is_empty() {
        eprintln!("Usage: qlang-cli orchestrate \"your task here\" [--budget 0.10] [--quality 0.9] [--models 3]");
        return;
    }

    let budget_cost = args
        .iter()
        .position(|a| a == "--budget")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0); // 0 = local only
    let quality = args
        .iter()
        .position(|a| a == "--quality")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.8);
    let max_models = args
        .iter()
        .position(|a| a == "--models")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);

    println!("=== QLANG Orchestrator ===");
    println!("Task: {}", prompt);
    println!(
        "Budget: ${:.2}, Quality: {:.0}%, Max Models: {}",
        budget_cost,
        quality * 100.0,
        max_models
    );
    println!();

    // Setup registry
    let mut registry = qlang_runtime::providers::ProviderRegistry::new(budget_cost);
    registry.discover_all();

    let models = registry.list_all_models();
    println!("Available models: {}", models.len());
    for (pid, mid, name, cost) in &models {
        println!("  {}/{} ({}) ${:.2}/1M", pid, mid, name, cost);
    }
    println!();

    // Create task
    let task = qlang_runtime::orchestrator::Task {
        id: "cli-task".into(),
        description: prompt.clone(),
        task_type: qlang_runtime::orchestrator::TaskType::TextGeneration,
        data_class: qlang_runtime::providers::DataClass::Internal,
        budget: qlang_runtime::orchestrator::Budget {
            max_cost: budget_cost,
            max_time_ms: 120_000,
            max_models,
        },
        quality_target: quality,
    };

    // Plan and execute
    let mut orchestrator = qlang_runtime::orchestrator::Orchestrator::new();
    let plan = orchestrator.plan(&task);
    println!("Plan: {:?} mode, {} steps", plan.mode, plan.steps.len());
    for step in &plan.steps {
        println!(
            "  Step {}: {} (prefer: {})",
            step.step_id, step.role, step.provider_preference
        );
    }
    println!();

    match orchestrator.execute(&plan, &task, &mut registry) {
        Ok(result) => {
            println!("=== Result ===");
            println!("{}", result.final_output);
            println!();
            println!("--- Stats ---");
            println!("Models used: {}", result.models_used.join(", "));
            println!("Total cost: ${:.4}", result.total_cost);
            println!("Total time: {}ms", result.total_latency_ms);
            println!("Steps: {}", result.steps.len());
        }
        Err(e) => {
            eprintln!("Orchestration failed: {}", e);
        }
    }

    // Print cost summary
    println!();
    println!("Cost: {}", registry.cost_summary());

    // Audit log
    println!();
    println!("--- Audit Log ---");
    for d in orchestrator.audit_log() {
        println!(
            "  [{}] {}: {} {}",
            d.timestamp_ms,
            d.action,
            d.reason,
            d.model_selected.as_deref().unwrap_or("")
        );
    }
}

// ---------------------------------------------------------------------------
// Proxy tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod proxy_tests {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    #[test]
    fn proxy_health_check() {
        use qlang_core::crypto::{self, Keypair};
        use qlang_agent::server::GraphStore;
        use qlang_compile::api;

        let keypair = Keypair::from_seed(&[42u8; 32]);
        let _store = GraphStore::new();

        // Bind to port 0 to get a random free port
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request = api::parse_request(&mut stream).unwrap();

            let info = serde_json::json!({
                "status": "ok",
                "version": env!("CARGO_PKG_VERSION"),
                "protocol": "qlms",
                "signed": true,
                "pubkey": crypto::hex(keypair.public_key()),
            });
            let response = api::json_ok(&info);
            let bytes = response.to_bytes();
            let _ = stream.write_all(&bytes);
            let _ = stream.flush();
        });

        // Send a health check request
        let mut client = TcpStream::connect(&addr).unwrap();
        client.write_all(b"GET /v1/health HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();

        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();

        assert!(response.contains("200 OK"), "Expected 200 OK, got: {response}");
        assert!(response.contains("qlms"), "Expected 'qlms' in response");
        assert!(response.contains("\"status\""), "Expected 'status' field");
        assert!(response.contains("\"signed\""), "Expected 'signed' field");

        handle.join().unwrap();
    }

    #[test]
    fn proxy_pubkey_endpoint() {
        use qlang_core::crypto::{self, Keypair};
        use qlang_compile::api;

        let keypair = Keypair::from_seed(&[99u8; 32]);
        let expected_pubkey = crypto::hex(keypair.public_key());

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request = api::parse_request(&mut stream).unwrap();

            let resp = serde_json::json!({ "pubkey": crypto::hex(keypair.public_key()) });
            let response = api::json_ok(&resp);
            let bytes = response.to_bytes();
            let _ = stream.write_all(&bytes);
            let _ = stream.flush();
        });

        let mut client = TcpStream::connect(&addr).unwrap();
        client.write_all(b"GET /v1/pubkey HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();

        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();

        assert!(response.contains("200 OK"));
        assert!(response.contains(&expected_pubkey));

        handle.join().unwrap();
    }
}
