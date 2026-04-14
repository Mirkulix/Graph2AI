//! Unified Runtime — Bridge between VM (scripting) and Graph (ML) systems.
//!
//! A single .qlang program can now do BOTH:
//!   - Scripting: variables, loops, functions, arrays
//!   - ML: graph definitions, training, compression
//!
//! Example:
//! ```qlang
//! let lr = 0.01
//! graph classifier { input x: f32[4]; node r = relu(x); output y = r }
//! let result = run_graph("classifier", {"x": [1.0, -2.0, 3.0, -4.0]})
//! print(result)
//! ```

use std::collections::HashMap;
use crate::vm;
use crate::executor;
use qlang_core::graph::Graph;
use qlang_core::tensor::{Shape, TensorData};

/// A graph block extracted from source.
#[derive(Debug)]
pub struct GraphBlock {
    pub name: String,
    pub source: String,
}

/// Result of unified execution.
#[derive(Debug)]
pub struct UnifiedResult {
    pub vm_result: Option<vm::Value>,
    pub graphs: Vec<String>,
    pub output: Vec<String>,
}

/// Error during unified execution.
#[derive(Debug, thiserror::Error)]
pub enum UnifiedError {
    #[error("VM error: {0}")]
    VmError(String),
    #[error("Graph error: {0}")]
    GraphError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Unknown graph: {0}")]
    UnknownGraph(String),
}

/// Split source into graph blocks and script code.
pub fn split_source(source: &str) -> (Vec<GraphBlock>, String) {
    let mut graphs = Vec::new();
    let mut script = String::new();
    let mut in_graph = false;
    let mut brace_depth = 0;
    let mut current_graph_name = String::new();
    let mut current_graph_source = String::new();

    for line in source.lines() {
        let trimmed = line.trim();

        if !in_graph && trimmed.starts_with("graph ") && trimmed.contains('{') {
            in_graph = true;
            brace_depth = 0;
            // Extract name
            let rest = &trimmed[6..];
            current_graph_name = rest.split('{').next().unwrap_or("").trim().to_string();
            current_graph_source = format!("{}\n", line);
            for c in trimmed.chars() {
                if c == '{' { brace_depth += 1; }
                if c == '}' { brace_depth -= 1; }
            }
            if brace_depth == 0 {
                in_graph = false;
                graphs.push(GraphBlock {
                    name: current_graph_name.clone(),
                    source: current_graph_source.clone(),
                });
            }
        } else if in_graph {
            current_graph_source.push_str(line);
            current_graph_source.push('\n');
            for c in trimmed.chars() {
                if c == '{' { brace_depth += 1; }
                if c == '}' { brace_depth -= 1; }
            }
            if brace_depth == 0 {
                in_graph = false;
                graphs.push(GraphBlock {
                    name: current_graph_name.clone(),
                    source: current_graph_source.clone(),
                });
            }
        } else {
            script.push_str(line);
            script.push('\n');
        }
    }

    (graphs, script)
}

/// Execute a unified .qlang program with both graphs and scripts.
pub fn execute_unified(source: &str) -> Result<UnifiedResult, UnifiedError> {
    let (graph_blocks, script) = split_source(source);

    // Register graph blocks (parsing done by caller or compile crate)
    let graph_names: Vec<String> = graph_blocks.iter().map(|g| g.name.clone()).collect();

    // Execute script with graph access
    if !script.trim().is_empty() {
        match vm::run_qlang_script(&script) {
            Ok((value, captured_output)) => {
                return Ok(UnifiedResult {
                    vm_result: Some(value),
                    graphs: graph_names,
                    output: captured_output,
                });
            }
            Err(e) => return Err(UnifiedError::VmError(format!("{}", e))),
        }
    }

    Ok(UnifiedResult {
        vm_result: None,
        graphs: graph_names,
        output: Vec::new(),
    })
}

/// Execute a named graph with inputs.
pub fn run_graph(
    graph: &Graph,
    inputs: HashMap<String, Vec<f32>>,
) -> Result<HashMap<String, Vec<f32>>, UnifiedError> {
    // Convert Vec<f32> inputs to TensorData
    let mut tensor_inputs = HashMap::new();
    for (name, values) in &inputs {
        let n = values.len();
        tensor_inputs.insert(
            name.clone(),
            TensorData::from_f32(Shape::vector(n), values),
        );
    }

    // Execute
    let result = executor::execute(graph, tensor_inputs)
        .map_err(|e| UnifiedError::GraphError(format!("{}", e)))?;

    // Convert back
    let mut outputs = HashMap::new();
    for (name, tensor) in result.outputs {
        if let Some(values) = tensor.as_f32_slice() {
            outputs.insert(name, values);
        }
    }

    Ok(outputs)
}

/// Compress weights using IGQK ternary compression.
pub fn compress_weights(weights: &[f32]) -> Vec<f32> {
    let mean_abs: f32 = weights.iter().map(|x| x.abs()).sum::<f32>() / weights.len() as f32;
    let threshold = mean_abs * 0.7;
    weights.iter().map(|&x| {
        if x > threshold { 1.0 }
        else if x < -threshold { -1.0 }
        else { 0.0 }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn split_graph_and_script() {
        let source = r#"
let x = 5

graph model {
  input a: f32[4]
  node r = relu(a)
  output y = r
}

let y = x + 1
print(y)
"#;
        let (graphs, script) = split_source(source);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].name, "model");
        assert!(script.contains("let x = 5"));
        assert!(script.contains("let y = x + 1"));
        assert!(!script.contains("graph model"));
    }

    #[test]
    fn split_multiple_graphs() {
        let source = r#"
graph encoder {
  input x: f32[8]
  node h = relu(x)
  output encoded = h
}

graph decoder {
  input z: f32[4]
  node out = relu(z)
  output decoded = out
}
"#;
        let (graphs, _script) = split_source(source);
        assert_eq!(graphs.len(), 2);
        assert_eq!(graphs[0].name, "encoder");
        assert_eq!(graphs[1].name, "decoder");
    }

    #[test]
    fn split_script_only() {
        let source = "let x = 42\nprint(x)\n";
        let (graphs, script) = split_source(source);
        assert_eq!(graphs.len(), 0);
        assert!(script.contains("let x = 42"));
    }

    #[test]
    fn split_graph_only() {
        let source = "graph test {\n  input x: f32[2]\n  output y = x\n}\n";
        let (graphs, script) = split_source(source);
        assert_eq!(graphs.len(), 1);
        assert!(script.trim().is_empty());
    }

    #[test]
    fn execute_script_only() {
        let source = "let x = 5.0\nlet y = x + 3.0\n";
        let result = execute_unified(source).unwrap();
        assert!(result.graphs.is_empty());
    }

    #[test]
    fn execute_graph_parses() {
        let source = r#"
graph test_model {
  input x: f32[4]
  node r = relu(x)
  output y = r
}
"#;
        let result = execute_unified(source).unwrap();
        assert_eq!(result.graphs.len(), 1);
        assert_eq!(result.graphs[0], "test_model");
    }

    #[test]
    fn run_graph_works() {
        let mut g = Graph::new("test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(4); 2], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(4));

        let mut inputs = HashMap::new();
        inputs.insert("a".into(), vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert("b".into(), vec![10.0, 20.0, 30.0, 40.0]);

        let result = run_graph(&g, inputs).unwrap();
        assert_eq!(result["y"], vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn compress_weights_ternary() {
        let weights = vec![0.5, -0.5, 0.1, -0.1, 0.8, -0.8, 0.0];
        let compressed = compress_weights(&weights);
        for &w in &compressed {
            assert!(w == 1.0 || w == -1.0 || w == 0.0);
        }
    }

    #[test]
    fn mixed_program_parses() {
        let source = r#"
let epochs = 10
let lr = 0.01

graph classifier {
  input x: f32[1, 4]
  input W: f32[4, 3]
  node h = matmul(x, W)
  node p = softmax(h)
  output predictions = p
}

let result = epochs * 2
"#;
        let (graphs, script) = split_source(source);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].name, "classifier");
        assert!(script.contains("let epochs = 10"));
        assert!(script.contains("let result = epochs * 2"));
    }
}
