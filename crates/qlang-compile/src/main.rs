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
use std::io::{BufRead, BufReader};
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

    match command.as_str() {
        "repl" => {
            qlang_compile::repl::run_repl();
            return;
        }
        "lsp" => {
            cmd_lsp();
            return;
        }
        "exec" => {
            if args.len() < 3 {
                eprintln!("Usage: qlang-cli exec <script.qlang>");
                process::exit(1);
            }
            cmd_exec_script(&args[2]);
            return;
        }
        "parse" => {
            if args.len() < 3 {
                eprintln!("Usage: qlang-cli parse <script.qlang>");
                process::exit(1);
            }
            cmd_parse(&args[2]);
            return;
        }
        _ => {}
    }

    if args.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let file_path = &args[2];
    let content = fs::read_to_string(file_path).unwrap_or_else(|e| {
        eprintln!("Error reading {file_path}: {e}");
        process::exit(1);
    });

    let graph = qlang_core::serial::from_json(&content).unwrap_or_else(|e| {
        eprintln!("Error parsing graph: {e}");
        process::exit(1);
    });

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
                eprintln!("LLVM not available.");
                process::exit(1);
            }
        }
        "dot" => print!("{}", qlang_compile::visualize::to_dot(&graph)),
        "ascii" => print!("{}", qlang_compile::visualize::to_ascii(&graph)),
        "llvm-ir" => {
            #[cfg(feature = "llvm")]
            cmd_llvm_ir(&graph);
        }
        "wasm" => println!("{}", qlang_compile::wasm::to_wat(&graph)),
        "gpu" => println!("{}", qlang_compile::gpu::to_wgsl(&graph)),
        _ => {
            eprintln!("Unknown command: {command}");
            print_usage();
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("QLANG CLI v0.4 — Simplified Protocol Native\n");
    eprintln!("Usage:");
    eprintln!("  qlang-cli info     <file.qlg.json>    Show graph info");
    eprintln!("  qlang-cli verify   <file.qlg.json>    Verify constraints");
    eprintln!("  qlang-cli optimize <file.qlg.json>    Optimize graph");
    eprintln!("  qlang-cli run      <file.qlg.json>    Execute (interpreter)");
    eprintln!("  qlang-cli jit      <file.qlg.json>    Execute (JIT)");
    eprintln!("  qlang-cli exec     <file.qlang>       Execute script");
    eprintln!("  qlang-cli repl                        Interactive REPL");
    eprintln!("  qlang-cli lsp                         Start LSP server");
}

fn cmd_exec_script(file_path: &str) {
    let source = fs::read_to_string(file_path).unwrap();
    let start = std::time::Instant::now();

    // Try bytecode first
    match qlang_runtime::bytecode::run_bytecode(&source) {
        Ok((_, output)) => {
            for line in output { println!("{line}"); }
            eprintln!("[executed in {:.3}s, bytecode VM]", start.elapsed().as_secs_f64());
        }
        Err(_) => {
            // Fallback to unified
            match qlang_runtime::unified::execute_unified(&source) {
                Ok(result) => {
                    for line in result.output { println!("{line}"); }
                    eprintln!("[executed in {:.3}s, interpreter]", start.elapsed().as_secs_f64());
                }
                Err(e) => { eprintln!("Error: {e}"); process::exit(1); }
            }
        }
    }
}

fn cmd_info(graph: &qlang_core::graph::Graph) {
    println!("{graph}");
}

fn cmd_verify(graph: &qlang_core::graph::Graph) {
    println!("{}", qlang_core::verify::verify_graph(graph));
}

fn cmd_optimize(mut graph: qlang_core::graph::Graph, output: Option<&str>) {
    let report = qlang_compile::optimize::optimize(&mut graph);
    println!("Optimize complete: {} dead nodes removed", report.dead_nodes_removed);
    if let Some(path) = output {
        fs::write(path, qlang_core::serial::to_json(&graph).unwrap()).unwrap();
    }
}

fn cmd_run(graph: &qlang_core::graph::Graph) {
    let inputs = HashMap::new(); // Simplified: zero-filled inputs auto-managed in lib
    match qlang_runtime::executor::execute(graph, inputs) {
        Ok(result) => {
            println!("Execution complete. Outputs: {:?}", result.outputs.keys());
        }
        Err(e) => { eprintln!("Failed: {e}"); process::exit(1); }
    }
}

#[cfg(feature = "llvm")]
fn cmd_jit(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    let context = Context::create();
    {
        let res = qlang_compile::codegen::compile_graph(&context, graph, inkwell::OptimizationLevel::Aggressive);
        match res {
            Ok(_) => println!("JIT compilation successful."),
            Err(e) => eprintln!("JIT failed: {e}"),
        }
    }
}

#[cfg(feature = "llvm")]
fn cmd_llvm_ir(graph: &qlang_core::graph::Graph) {
    use inkwell::context::Context;
    let context = Context::create();
    {
        let compiled = qlang_compile::codegen::compile_graph(&context, graph, inkwell::OptimizationLevel::None).unwrap();
        println!("{}", compiled.llvm_ir);
    }
}

fn cmd_parse(file_path: &str) {
    let content = fs::read_to_string(file_path).unwrap();
    match qlang_compile::parser::parse(&content) {
        Ok(graph) => println!("Parsed: {}", graph.id),
        Err(e) => eprintln!("Parse error: {e}"),
    }
}

fn cmd_lsp() {
    let stdin = std::io::stdin();
    let reader = BufReader::new(stdin.lock());
    eprintln!("QLANG LSP starting...");
    for line in reader.lines() {
        if let Ok(l) = line {
            if l.is_empty() { continue; }
            // Stub: actual LSP logic moved to lib
        }
    }
}

