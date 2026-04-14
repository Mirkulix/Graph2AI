//! QLANG REPL — Interactive graph construction and execution.
//!
//! Provides a command-line interface for building and running QLANG graphs
//! interactively. Useful for experimentation and debugging.
//!
//! Commands:
//!   > input x: f32[4]                    Define an input
//!   > node sum = add(x, y)               Add a computation node
//!   > output result = sum                Define output
//!   > run                                Execute the graph
//!   > show                               Show graph structure
//!   > dot                                Show Graphviz DOT
//!   > llvm                               Show LLVM IR
//!   > gpu                                Show WGSL shader
//!   > clear                              Reset graph
//!   > load <file.qlang>                  Load a .qlang file
//!   > save <file.qlg.json>               Save as JSON
//!   > exit                               Quit

use qlang_core::graph::Graph;
use qlang_core::tensor::TensorData;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Run the QLANG REPL.
pub fn run_repl() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!("QLANG REPL v0.3 — Interactive graph programming");
    println!("Type 'help' for commands, 'exit' to quit.\n");

    let mut graph = Graph::new("repl");
    let mut node_names: HashMap<String, (u32, qlang_core::tensor::TensorType)> = HashMap::new();
    let mut input_data: HashMap<String, Vec<f32>> = HashMap::new();

    loop {
        print!("qlang> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() || line.is_empty() {
            break;
        }
        let line = line.trim();

        if line.is_empty() { continue; }

        match line {
            "exit" | "quit" | "q" => break,
            "help" | "h" | "?" => print_help(),
            "show" | "s" => println!("{graph}"),
            "clear" | "c" => {
                graph = Graph::new("repl");
                node_names.clear();
                input_data.clear();
                println!("Graph cleared.");
            }
            "run" | "r" => run_graph(&graph, &node_names, &input_data),
            "verify" | "v" => {
                let result = qlang_core::verify::verify_graph(&graph);
                println!("{result}");
            }
            "dot" | "d" => {
                println!("{}", crate::visualize::to_dot(&graph));
            }
            "ascii" | "a" => {
                println!("{}", crate::visualize::to_ascii(&graph));
            }
            "llvm" | "l" => {
                #[cfg(feature = "llvm")]
                show_llvm(&graph);
                #[cfg(not(feature = "llvm"))]
                println!("LLVM not available. Build with: cargo build --features llvm");
            }
            "gpu" | "g" => {
                println!("{}", crate::gpu::to_wgsl(&graph));
            }
            "stats" => {
                println!("Nodes: {}", graph.nodes.len());
                println!("Edges: {}", graph.edges.len());
                println!("Inputs: {}", graph.input_nodes().len());
                println!("Outputs: {}", graph.output_nodes().len());
                let quantum = graph.nodes.iter().filter(|n| n.op.is_quantum()).count();
                println!("Quantum ops: {quantum}");
                if let Ok(bin) = qlang_core::serial::to_binary(&graph) {
                    println!("Binary size: {} bytes", bin.len());
                }
            }
            _ => {
                // Try parsing as .qlang statement
                if line.starts_with("load ") {
                    load_file(&mut graph, &mut node_names, &line[5..].trim());
                } else if line.starts_with("save ") {
                    save_file(&graph, &line[5..].trim());
                } else if line.starts_with("set ") {
                    set_input_data(&mut input_data, &line[4..].trim());
                } else if line.starts_with("input ") || line.starts_with("node ") || line.starts_with("output ") {
                    // Parse as inline graph statement
                    let full_source = format!("graph repl {{\n  {line}\n}}");
                    match crate::parser::parse(&full_source) {
                        Ok(parsed) => {
                            // Merge parsed nodes into current graph
                            for node in &parsed.nodes {
                                match &node.op {
                                    qlang_core::ops::Op::Input { name } => {
                                        if !node_names.contains_key(name) {
                                            let tt = node.output_types.first().cloned()
                                                .unwrap_or(qlang_core::tensor::TensorType::f32_scalar());
                                            let id = graph.add_node(node.op.clone(), vec![], vec![tt.clone()]);
                                            node_names.insert(name.clone(), (id, tt));
                                            println!("  + input '{name}' (id={id})");
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            // For node/output statements, re-parse with all existing names
                            if line.starts_with("node ") || line.starts_with("output ") {
                                let mut full = String::from("graph repl {\n");
                                for (name, (_, tt)) in &node_names {
                                    full.push_str(&format!("  input {}: {}\n", name, crate::parser::format_type_pub(tt)));
                                }
                                full.push_str(&format!("  {line}\n}}\n"));

                                match crate::parser::parse(&full) {
                                    Ok(reparsed) => {
                                        // Find the new node(s) and add to our graph
                                        for node in &reparsed.nodes {
                                            match &node.op {
                                                qlang_core::ops::Op::Input { .. } => {} // skip, already added
                                                qlang_core::ops::Op::Output { name } => {
                                                    // Find source
                                                    let incoming = reparsed.incoming_edges(node.id);
                                                    if let Some(edge) = incoming.first() {
                                                        // Find the source node in our graph
                                                        let source_node = reparsed.node(edge.from_node);
                                                        if let Some(_src) = source_node {
                                                            // Look up the actual node name from reparsed context
                                                            // The source is likely the last node we added
                                                            if let Some((src_id, src_tt)) = find_source_in_graph(&node_names, &reparsed, edge.from_node) {
                                                                let out_id = graph.add_node(
                                                                    qlang_core::ops::Op::Output { name: name.clone() },
                                                                    vec![src_tt.clone()],
                                                                    vec![],
                                                                );
                                                                graph.add_edge(src_id, 0, out_id, 0, src_tt);
                                                                println!("  + output '{name}' (id={out_id})");
                                                            }
                                                        }
                                                    }
                                                }
                                                other_op => {
                                                    // Computation node
                                                    let tt = node.output_types.first().cloned()
                                                        .unwrap_or(qlang_core::tensor::TensorType::f32_scalar());
                                                    let id = graph.add_node(
                                                        other_op.clone(),
                                                        node.input_types.clone(),
                                                        vec![tt.clone()],
                                                    );

                                                    // Wire edges from reparsed graph
                                                    let incoming = reparsed.incoming_edges(node.id);
                                                    for (port, edge) in incoming.iter().enumerate() {
                                                        if let Some(src) = reparsed.node(edge.from_node) {
                                                            if let qlang_core::ops::Op::Input { name: src_name } = &src.op {
                                                                if let Some((src_id, src_tt)) = node_names.get(src_name) {
                                                                    graph.add_edge(*src_id, 0, id, port as u8, src_tt.clone());
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // Extract node name from line
                                                    if let Some(var_name) = extract_node_name(line) {
                                                        node_names.insert(var_name.clone(), (id, tt));
                                                        println!("  + node '{var_name}' = {} (id={id})", other_op);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => println!("  Error: {e}"),
                                }
                            }
                        }
                        Err(e) => println!("  Error: {e}"),
                    }
                } else {
                    // Try executing as VM script (variables, expressions, print, etc.)
                    match qlang_runtime::vm::run_qlang_script(line) {
                        Ok((_value, output)) => {
                            for out_line in &output {
                                println!("{}", out_line);
                            }
                        }
                        Err(e) => {
                            eprintln!("  Error: {}", e);
                        }
                    }
                }
            }
        }
    }

    println!("Goodbye.");
}

fn print_help() {
    println!("QLANG REPL Commands:");
    println!("  input <name>: <type>       Define input tensor");
    println!("  node <name> = <op>(args)   Add computation node");
    println!("  output <name> = <node>     Define output");
    println!("  set <input> = [values]     Set input data");
    println!("  run                        Execute graph");
    println!("  show                       Show graph structure");
    println!("  ascii                      ASCII visualization");
    println!("  dot                        Graphviz DOT output");
    println!("  llvm                       Show LLVM IR");
    println!("  gpu                        Show WGSL shader");
    println!("  verify                     Check constraints");
    println!("  stats                      Graph statistics");
    println!("  load <file>                Load .qlang file");
    println!("  save <file>                Save as JSON");
    println!("  clear                      Reset graph");
    println!("  exit                       Quit");
}

fn run_graph(
    graph: &Graph,
    _node_names: &HashMap<String, (u32, qlang_core::tensor::TensorType)>,
    input_data: &HashMap<String, Vec<f32>>,
) {
    let mut inputs = HashMap::new();

    for node in graph.input_nodes() {
        if let qlang_core::ops::Op::Input { name } = &node.op {
            if let Some(tt) = node.output_types.first() {
                let data = if let Some(values) = input_data.get(name) {
                    TensorData::from_f32(tt.shape.clone(), values)
                } else if let Some(zeros) = TensorData::zeros(tt) {
                    println!("  Using zeros for '{name}'");
                    zeros
                } else {
                    continue;
                };
                inputs.insert(name.clone(), data);
            }
        }
    }

    match qlang_runtime::executor::execute(graph, inputs) {
        Ok(result) => {
            println!("  Executed {} nodes ({} quantum ops, {} FLOPs)",
                result.stats.nodes_executed, result.stats.quantum_ops, result.stats.total_flops);
            for (name, tensor) in &result.outputs {
                if let Some(vals) = tensor.as_f32_slice() {
                    if vals.len() <= 16 {
                        println!("  {name} = {:?}", vals);
                    } else {
                        println!("  {name} = [{}, {}, ... {} elements]", vals[0], vals[1], vals.len());
                    }
                } else {
                    println!("  {name}: {} {}", tensor.dtype, tensor.shape);
                }
            }
        }
        Err(e) => println!("  Error: {e}"),
    }
}

#[cfg(feature = "llvm")]
fn show_llvm(graph: &Graph) {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;

    let context = Context::create();
    let result = crate::codegen::compile_graph(&context, graph, OptimizationLevel::None);
    match result {
        Ok(compiled) => println!("{}", compiled.llvm_ir),
        Err(e) => println!("  Codegen error: {e}"),
    };
}

fn load_file(graph: &mut Graph, node_names: &mut HashMap<String, (u32, qlang_core::tensor::TensorType)>, path: &str) {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            match crate::parser::parse(&content) {
                Ok(g) => {
                    *graph = g;
                    node_names.clear();
                    // Rebuild node names
                    for node in &graph.nodes {
                        if let qlang_core::ops::Op::Input { name } = &node.op {
                            if let Some(tt) = node.output_types.first() {
                                node_names.insert(name.clone(), (node.id, tt.clone()));
                            }
                        }
                    }
                    println!("  Loaded: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
                }
                Err(e) => println!("  Parse error: {e}"),
            }
        }
        Err(e) => println!("  File error: {e}"),
    }
}

fn save_file(graph: &Graph, path: &str) {
    match qlang_core::serial::to_json(graph) {
        Ok(json) => {
            match std::fs::write(path, &json) {
                Ok(_) => println!("  Saved: {} bytes", json.len()),
                Err(e) => println!("  Write error: {e}"),
            }
        }
        Err(e) => println!("  Serialize error: {e}"),
    }
}

fn set_input_data(input_data: &mut HashMap<String, Vec<f32>>, s: &str) {
    // "x = [1.0, 2.0, 3.0]"
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        println!("  Usage: set <name> = [values...]");
        return;
    }
    let name = parts[0].trim().to_string();
    let values_str = parts[1].trim().trim_matches(|c| c == '[' || c == ']');
    let values: Vec<f32> = values_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    println!("  Set '{name}' = {} values", values.len());
    input_data.insert(name, values);
}

fn extract_node_name(line: &str) -> Option<String> {
    // "node foo = add(x, y)" → "foo"
    if line.starts_with("node ") {
        let rest = &line[5..];
        let eq = rest.find('=')?;
        Some(rest[..eq].trim().to_string())
    } else {
        None
    }
}

fn find_source_in_graph(
    node_names: &HashMap<String, (u32, qlang_core::tensor::TensorType)>,
    reparsed: &Graph,
    reparsed_node_id: u32,
) -> Option<(u32, qlang_core::tensor::TensorType)> {
    // Look through node_names to find a match
    if let Some(node) = reparsed.node(reparsed_node_id) {
        if let qlang_core::ops::Op::Input { name } = &node.op {
            return node_names.get(name).cloned();
        }
    }
    // Return the last node added
    node_names.values().last().cloned()
}
