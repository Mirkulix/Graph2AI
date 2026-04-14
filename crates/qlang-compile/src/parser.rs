//! QLANG Text Parser — Human-readable .qlang syntax → Graph
//!
//! This is a VIEW for humans. The graph is the source of truth.
//! The text format exists so humans can READ and WRITE QLANG programs.
//! AI agents should use the GraphEmitter directly.
//!
//! Syntax:
//! ```qlang
//! graph my_model {
//!   input x: f32[784]
//!   input W: f32[784, 128]
//!
//!   node h = matmul(x, W)
//!   node a = relu(h)
//!   node c = to_ternary(a) @proof theorem_5_2
//!
//!   output y = c
//! }
//! ```

use qlang_core::graph::Graph;
use qlang_core::ops::{Manifold, Op};
use qlang_core::tensor::{Dim, Dtype, Shape, TensorType};
use qlang_core::verify::{Constraint, ConstraintKind, Proof, ProofStatus, TheoremRef};
use std::collections::HashMap;

/// Parse errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    #[error("line {line}: {message}")]
    SyntaxError { line: usize, message: String },

    #[error("undefined node '{0}'")]
    UndefinedNode(String),

    #[error("unknown type '{0}'")]
    UnknownType(String),

    #[error("unknown operation '{0}'")]
    UnknownOp(String),
}

/// Severity level for parser diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

/// A single diagnostic emitted during parsing, with location info.
#[derive(Debug, Clone)]
pub struct ParseDiagnostic {
    pub line: usize,
    pub col: usize,
    pub message: String,
    pub severity: DiagnosticSeverity,
    /// Optional suggestion for how to fix the error.
    pub suggestion: Option<String>,
}

impl std::fmt::Display for ParseDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sev = match self.severity {
            DiagnosticSeverity::Error => "error",
            DiagnosticSeverity::Warning => "warning",
        };
        write!(f, "{}:{}:{}: {}", sev, self.line, self.col, self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " ({})", suggestion)?;
        }
        Ok(())
    }
}

/// Result of error-recovering parse: partial AST + all diagnostics.
#[derive(Debug)]
pub struct ParseResult {
    /// The (possibly partial) graph. `None` only if the header could not be parsed at all.
    pub graph: Option<Graph>,
    /// All diagnostics collected during parsing (up to `MAX_ERRORS`).
    pub diagnostics: Vec<ParseDiagnostic>,
}

impl ParseResult {
    /// Returns `true` when no errors were reported (warnings are ok).
    pub fn is_ok(&self) -> bool {
        !self.diagnostics.iter().any(|d| d.severity == DiagnosticSeverity::Error)
    }

    /// Convenience: returns the graph if there were zero errors.
    pub fn into_result(self) -> Result<Graph, Vec<ParseDiagnostic>> {
        if self.is_ok() {
            Ok(self.graph.unwrap_or_else(|| Graph::new("empty")))
        } else {
            Err(self.diagnostics)
        }
    }

    /// Number of errors (not warnings).
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.severity == DiagnosticSeverity::Error).count()
    }
}

/// Maximum number of errors before the parser gives up.
const MAX_ERRORS: usize = 50;

/// Parse a .qlang text file into a Graph.
///
/// Returns the first error encountered (backwards-compatible API).
pub fn parse(source: &str) -> Result<Graph, ParseError> {
    let result = parse_recovering(source);
    if result.is_ok() {
        Ok(result.graph.unwrap_or_else(|| Graph::new("empty")))
    } else {
        // Return the first error for backwards compatibility.
        let first = result.diagnostics.into_iter()
            .find(|d| d.severity == DiagnosticSeverity::Error)
            .expect("is_ok was false but no error diagnostic found");
        Err(ParseError::SyntaxError {
            line: first.line,
            message: first.message,
        })
    }
}

/// Parse a .qlang text file with error recovery.
///
/// Collects up to [`MAX_ERRORS`] errors and returns a partial AST even when
/// errors are present. This is the preferred entry-point for IDE / LSP usage.
pub fn parse_recovering(source: &str) -> ParseResult {
    let mut parser = Parser::new(source);
    parser.parse_graph_recovering()
}

struct Parser<'a> {
    lines: Vec<(usize, &'a str)>, // (line_number, content)
    pos: usize,
    /// Maps node names to (node_id, tensor_type)
    nodes: HashMap<String, (u32, TensorType)>,
    /// Collected diagnostics during error-recovering parse.
    diagnostics: Vec<ParseDiagnostic>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        let lines: Vec<(usize, &str)> = source
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.trim()))
            .filter(|(_, l)| !l.is_empty() && !l.starts_with("//") && !l.starts_with('#'))
            .collect();

        Self {
            lines,
            pos: 0,
            nodes: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }

    fn current(&self) -> Option<(usize, &'a str)> {
        self.lines.get(self.pos).copied()
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    fn err(&self, line: usize, msg: impl Into<String>) -> ParseError {
        ParseError::SyntaxError {
            line,
            message: msg.into(),
        }
    }

    /// Error-recovering variant of `parse_graph`. Skips bad lines instead of
    /// returning early, collecting diagnostics along the way.
    fn parse_graph_recovering(&mut self) -> ParseResult {
        // Parse "graph <name> {"
        let name = match self.current() {
            Some((_line, text)) if text.starts_with("graph ") => {
                let rest = &text[6..];
                let n = rest.trim_end_matches('{').trim().to_string();
                self.advance();
                n
            }
            Some((line, _text)) => {
                self.diagnostics.push(ParseDiagnostic {
                    line,
                    col: 1,
                    message: "expected 'graph <name> {'".into(),
                    severity: DiagnosticSeverity::Error,
                    suggestion: None,
                });
                return ParseResult { graph: None, diagnostics: std::mem::take(&mut self.diagnostics) };
            }
            None => {
                self.diagnostics.push(ParseDiagnostic {
                    line: 0,
                    col: 1,
                    message: "empty file".into(),
                    severity: DiagnosticSeverity::Error,
                    suggestion: None,
                });
                return ParseResult { graph: None, diagnostics: std::mem::take(&mut self.diagnostics) };
            }
        };

        let mut graph = Graph::new(&name);

        while let Some((line, text)) = self.current() {
            if self.diagnostics.len() >= MAX_ERRORS {
                break;
            }
            if text == "}" {
                self.advance();
                break;
            }

            let result = if text.starts_with("input ") {
                self.parse_input(&mut graph, line, text)
            } else if text.starts_with("output ") {
                self.parse_output(&mut graph, line, text)
            } else if text.starts_with("node ") {
                self.parse_node(&mut graph, line, text)
            } else {
                Err(self.err(line, format!("unexpected: '{text}'")))
            };

            if let Err(e) = result {
                let (eline, emsg) = match &e {
                    ParseError::SyntaxError { line, message } => (*line, message.clone()),
                    _ => (line, format!("{e}")),
                };
                self.diagnostics.push(ParseDiagnostic {
                    line: eline,
                    col: 1,
                    message: emsg,
                    severity: DiagnosticSeverity::Error,
                    suggestion: None,
                });
            }

            self.advance();
        }

        ParseResult {
            graph: Some(graph),
            diagnostics: std::mem::take(&mut self.diagnostics),
        }
    }

    #[allow(dead_code)]
    fn parse_graph(&mut self) -> Result<Graph, ParseError> {
        // Parse "graph <name> {"
        let (line, text) = self.current().ok_or(self.err(0, "empty file"))?;

        let name = if text.starts_with("graph ") {
            let rest = &text[6..];
            let name = rest.trim_end_matches('{').trim();
            name.to_string()
        } else {
            return Err(self.err(line, "expected 'graph <name> {'"));
        };

        self.advance();

        let mut graph = Graph::new(&name);

        // Parse body
        while let Some((line, text)) = self.current() {
            if text == "}" {
                self.advance();
                break;
            }

            if text.starts_with("input ") {
                self.parse_input(&mut graph, line, text)?;
            } else if text.starts_with("output ") {
                self.parse_output(&mut graph, line, text)?;
            } else if text.starts_with("node ") {
                self.parse_node(&mut graph, line, text)?;
            } else {
                return Err(self.err(line, format!("unexpected: '{text}'")));
            }

            self.advance();
        }

        Ok(graph)
    }

    fn parse_input(&mut self, graph: &mut Graph, line: usize, text: &str) -> Result<(), ParseError> {
        // "input <name>: <type>"
        let rest = &text[6..]; // skip "input "
        let parts: Vec<&str> = rest.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(self.err(line, "expected 'input <name>: <type>'"));
        }

        let name = parts[0].trim();
        let type_str = parts[1].trim();
        let tt = self.parse_type(type_str, line)?;

        let id = graph.add_node(
            Op::Input { name: name.to_string() },
            vec![],
            vec![tt.clone()],
        );
        self.nodes.insert(name.to_string(), (id, tt));
        Ok(())
    }

    fn parse_output(&mut self, graph: &mut Graph, line: usize, text: &str) -> Result<(), ParseError> {
        // "output <name> = <source_node>"
        let rest = &text[7..]; // skip "output "
        let parts: Vec<&str> = rest.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(self.err(line, "expected 'output <name> = <node>'"));
        }

        let name = parts[0].trim();
        let source_name = parts[1].trim();

        let (source_id, source_type) = self.nodes.get(source_name)
            .ok_or(ParseError::UndefinedNode(source_name.to_string()))?
            .clone();

        let out_id = graph.add_node(
            Op::Output { name: name.to_string() },
            vec![source_type.clone()],
            vec![],
        );
        graph.add_edge(source_id, 0, out_id, 0, source_type);
        Ok(())
    }

    fn parse_node(&mut self, graph: &mut Graph, line: usize, text: &str) -> Result<(), ParseError> {
        // "node <name> = <op>(<args>) [@proof <theorem>]"
        let rest = &text[5..]; // skip "node "
        let parts: Vec<&str> = rest.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(self.err(line, "expected 'node <name> = <op>(<args>)'"));
        }

        let name = parts[0].trim();
        let rhs = parts[1].trim();

        // Check for @proof annotation
        let (op_str, proof) = if let Some(idx) = rhs.find("@proof") {
            let theorem_str = rhs[idx + 6..].trim();
            let theorem = match theorem_str {
                "theorem_5_1" => TheoremRef::IgqkConvergence,
                "theorem_5_2" => TheoremRef::IgqkCompressionBound,
                "theorem_5_3" => TheoremRef::IgqkEntanglementGeneralization,
                other => TheoremRef::External { name: other.to_string() },
            };
            (
                rhs[..idx].trim(),
                Some(Constraint {
                    kind: ConstraintKind::DistortionBound { max_distortion: 0.01 },
                    proof: Some(Proof {
                        theorem,
                        status: ProofStatus::Assumed,
                        parameters: vec![],
                    }),
                }),
            )
        } else {
            (rhs, None)
        };

        // Parse "op(arg1, arg2)"
        let paren_start = op_str.find('(')
            .ok_or(self.err(line, "expected '(' in operation"))?;
        let paren_end = op_str.rfind(')')
            .ok_or(self.err(line, "expected ')' in operation"))?;

        let op_name = op_str[..paren_start].trim();
        let args_str = &op_str[paren_start + 1..paren_end];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();

        // Resolve argument nodes
        let mut arg_ids = Vec::new();
        let mut arg_types = Vec::new();
        for arg in &args {
            let (id, tt) = self.nodes.get(*arg)
                .ok_or(ParseError::UndefinedNode(arg.to_string()))?
                .clone();
            arg_ids.push(id);
            arg_types.push(tt);
        }

        // Determine operation and output type
        let (op, output_type) = self.resolve_op(op_name, &arg_types, line)?;

        let node_id = graph.add_node(
            op,
            arg_types.clone(),
            vec![output_type.clone()],
        );

        // Add proof constraint if present
        if let Some(constraint) = proof {
            graph.nodes.last_mut().unwrap().constraints.push(constraint);
        }

        // Wire edges
        for (port, &source_id) in arg_ids.iter().enumerate() {
            graph.add_edge(
                source_id,
                0,
                node_id,
                port as u8,
                arg_types[port].clone(),
            );
        }

        self.nodes.insert(name.to_string(), (node_id, output_type));
        Ok(())
    }

    fn resolve_op(&self, name: &str, args: &[TensorType], _line: usize) -> Result<(Op, TensorType), ParseError> {
        let first_type = args.first().cloned().unwrap_or(TensorType::f32_scalar());

        match name {
            "add" => Ok((Op::Add, first_type)),
            "sub" => Ok((Op::Sub, first_type)),
            "mul" => Ok((Op::Mul, first_type)),
            "div" => Ok((Op::Div, first_type)),
            "neg" => Ok((Op::Neg, first_type)),
            "matmul" => {
                // Output shape: [m, n] from [m, k] × [k, n]
                if args.len() >= 2 {
                    let a_shape = &args[0].shape;
                    let b_shape = &args[1].shape;
                    let m = a_shape.0.first().cloned().unwrap_or(Dim::Dynamic);
                    let n = b_shape.0.last().cloned().unwrap_or(Dim::Dynamic);
                    let out_type = TensorType::new(Dtype::F32, Shape(vec![m, n]));
                    Ok((Op::MatMul, out_type))
                } else {
                    Ok((Op::MatMul, first_type))
                }
            }
            "relu" => Ok((Op::Relu, first_type)),
            "sigmoid" => Ok((Op::Sigmoid, first_type)),
            "tanh" => Ok((Op::Tanh, first_type)),
            "softmax" => Ok((Op::Softmax { axis: first_type.shape.rank().saturating_sub(1) }, first_type)),
            "transpose" => {
                let mut new_shape = first_type.shape.0.clone();
                new_shape.reverse();
                Ok((Op::Transpose, TensorType::new(first_type.dtype, Shape(new_shape))))
            }
            "to_ternary" => {
                let out = TensorType::new(Dtype::Ternary, first_type.shape.clone());
                Ok((Op::ToTernary, out))
            }
            "to_lowrank" => Ok((Op::ToLowRank { rank: 16 }, first_type)),
            "superpose" => Ok((Op::Superpose, first_type)),
            "measure" => Ok((Op::Measure, first_type)),
            "entangle" => Ok((Op::Entangle, first_type)),
            "collapse" => Ok((Op::Collapse, first_type)),
            "entropy" => Ok((Op::Entropy, TensorType::f32_scalar())),
            "evolve" => Ok((Op::Evolve { gamma: 0.01, dt: 0.001 }, first_type)),
            "project_ternary" => Ok((Op::Project { manifold: Manifold::Ternary }, first_type)),
            "layer_norm" => Ok((Op::LayerNorm { eps: 1e-5 }, first_type)),
            "gelu" => Ok((Op::Gelu, first_type)),
            "residual" => Ok((Op::Residual, first_type)),
            "dropout" => Ok((Op::Dropout { rate: 0.1 }, first_type)),
            s if s.starts_with("ollama_generate:") => {
                let model = s.trim_start_matches("ollama_generate:").to_string();
                let out = TensorType::new(Dtype::Utf8, Shape(vec![Dim::Dynamic]));
                Ok((Op::OllamaGenerate { model }, out))
            }
            "ollama_generate" => {
                let out = TensorType::new(Dtype::Utf8, Shape(vec![Dim::Dynamic]));
                Ok((Op::OllamaGenerate { model: "llama3".to_string() }, out))
            }
            s if s.starts_with("ollama_chat:") => {
                let model = s.trim_start_matches("ollama_chat:").to_string();
                let out = TensorType::new(Dtype::Utf8, Shape(vec![Dim::Dynamic]));
                Ok((Op::OllamaChat { model }, out))
            }
            "ollama_chat" => {
                let out = TensorType::new(Dtype::Utf8, Shape(vec![Dim::Dynamic]));
                Ok((Op::OllamaChat { model: "llama3".to_string() }, out))
            }
            _ => Err(ParseError::UnknownOp(name.to_string())),
        }
    }

    fn parse_type(&self, s: &str, _line: usize) -> Result<TensorType, ParseError> {
        // Parse "f32[784]", "f32[28, 28]", "ternary[128]", "f32" (scalar)
        let bracket_start = s.find('[');

        let (dtype_str, shape) = if let Some(idx) = bracket_start {
            let dtype_s = s[..idx].trim();
            let shape_str = &s[idx + 1..s.len() - 1]; // strip [ ]
            let dims: Vec<Dim> = shape_str
                .split(',')
                .map(|d| {
                    let d = d.trim();
                    if d == "?" {
                        Dim::Dynamic
                    } else {
                        Dim::Fixed(d.parse().unwrap_or(0))
                    }
                })
                .collect();
            (dtype_s, Shape(dims))
        } else {
            (s.trim(), Shape::scalar())
        };

        let dtype = match dtype_str {
            "f16" => Dtype::F16,
            "f32" => Dtype::F32,
            "f64" => Dtype::F64,
            "i8" => Dtype::I8,
            "i16" => Dtype::I16,
            "i32" => Dtype::I32,
            "i64" => Dtype::I64,
            "bool" => Dtype::Bool,
            "ternary" => Dtype::Ternary,
            "utf8" => Dtype::Utf8,
            other => return Err(ParseError::UnknownType(other.to_string())),
        };

        Ok(TensorType::new(dtype, shape))
    }
}

/// Generate .qlang text from a Graph (the reverse direction).
pub fn to_qlang_text(graph: &Graph) -> String {
    let mut out = String::new();
    out.push_str(&format!("graph {} {{\n", graph.id));

    // Collect node names
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut next_var = 0;

    // Inputs
    for node in &graph.nodes {
        match &node.op {
            Op::Input { name } => {
                if let Some(tt) = node.output_types.first() {
                    out.push_str(&format!("  input {}: {}\n", name, format_type(tt)));
                    names.insert(node.id, name.clone());
                }
            }
            _ => {}
        }
    }

    out.push('\n');

    // Computation nodes
    if let Ok(order) = graph.topological_sort() {
        for &id in &order {
            if let Some(node) = graph.node(id) {
                match &node.op {
                    Op::Input { .. } | Op::Output { .. } => continue,
                    _ => {}
                }

                let var_name = format!("v{next_var}");
                next_var += 1;

                let op_name = match &node.op {
                    Op::Add => "add",
                    Op::Sub => "sub",
                    Op::Mul => "mul",
                    Op::Div => "div",
                    Op::Neg => "neg",
                    Op::MatMul => "matmul",
                    Op::Relu => "relu",
                    Op::Sigmoid => "sigmoid",
                    Op::Tanh => "tanh",
                    Op::Softmax { .. } => "softmax",
                    Op::Transpose => "transpose",
                    Op::ToTernary => "to_ternary",
                    Op::ToLowRank { .. } => "to_lowrank",
                    Op::Superpose => "superpose",
                    Op::Measure => "measure",
                    Op::Entangle => "entangle",
                    Op::Collapse => "collapse",
                    Op::Entropy => "entropy",
                    Op::Evolve { .. } => "evolve",
                    Op::Project { .. } => "project_ternary",
                    Op::LayerNorm { .. } => "layer_norm",
                    Op::Attention { .. } => "attention",
                    Op::Embedding { .. } => "embedding",
                    Op::Residual => "residual",
                    Op::Gelu => "gelu",
                    Op::Dropout { .. } => "dropout",
                    Op::OllamaGenerate { model } => {
                        let incoming = graph.incoming_edges(id);
                        let args: Vec<String> = incoming
                            .iter()
                            .map(|e| names.get(&e.from_node).cloned().unwrap_or_else(|| format!("?{}", e.from_node)))
                            .collect();
                        out.push_str(&format!("  node {} = ollama_generate:{}({})\n",
                            var_name, model, args.join(", ")));
                        names.insert(id, var_name);
                        continue;
                    }
                    Op::OllamaChat { model } => {
                        let incoming = graph.incoming_edges(id);
                        let args: Vec<String> = incoming
                            .iter()
                            .map(|e| names.get(&e.from_node).cloned().unwrap_or_else(|| format!("?{}", e.from_node)))
                            .collect();
                        out.push_str(&format!("  node {} = ollama_chat:{}({})\n",
                            var_name, model, args.join(", ")));
                        names.insert(id, var_name);
                        continue;
                    }
                    other => {
                        out.push_str(&format!("  // unsupported: {other}\n"));
                        names.insert(id, var_name);
                        continue;
                    }
                };

                // Find inputs via edges
                let incoming = graph.incoming_edges(id);
                let args: Vec<String> = incoming
                    .iter()
                    .map(|e| names.get(&e.from_node).cloned().unwrap_or_else(|| format!("?{}", e.from_node)))
                    .collect();

                let proof_str = if !node.constraints.is_empty() {
                    " @proof theorem_5_2"
                } else {
                    ""
                };

                out.push_str(&format!("  node {} = {}({}){}\n",
                    var_name, op_name, args.join(", "), proof_str));

                names.insert(id, var_name);
            }
        }
    }

    out.push('\n');

    // Outputs
    for node in &graph.nodes {
        if let Op::Output { name } = &node.op {
            let incoming = graph.incoming_edges(node.id);
            if let Some(edge) = incoming.first() {
                let source_name = names.get(&edge.from_node).cloned().unwrap_or("?".into());
                out.push_str(&format!("  output {} = {}\n", name, source_name));
            }
        }
    }

    out.push_str("}\n");
    out
}

/// Format a TensorType as .qlang text.
pub fn format_type_pub(tt: &TensorType) -> String {
    format_type(tt)
}

fn format_type(tt: &TensorType) -> String {
    let dtype = match tt.dtype {
        Dtype::F16 => "f16",
        Dtype::F32 => "f32",
        Dtype::F64 => "f64",
        Dtype::I8 => "i8",
        Dtype::I16 => "i16",
        Dtype::I32 => "i32",
        Dtype::I64 => "i64",
        Dtype::Bool => "bool",
        Dtype::Ternary => "ternary",
        Dtype::Utf8 => "utf8",
    };

    if tt.shape.0.is_empty() {
        dtype.to_string()
    } else {
        let dims: Vec<String> = tt.shape.0.iter().map(|d| match d {
            Dim::Fixed(n) => n.to_string(),
            Dim::Dynamic => "?".to_string(),
        }).collect();
        format!("{}[{}]", dtype, dims.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_graph() {
        let source = r#"
graph hello {
  input x: f32[4]
  input y: f32[4]

  node sum = add(x, y)
  node activated = relu(sum)

  output result = activated
}
"#;
        let graph = parse(source).unwrap();
        assert_eq!(graph.id, "hello");
        assert_eq!(graph.nodes.len(), 5); // 2 inputs + add + relu + output
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn parse_mlp() {
        let source = r#"
graph mlp {
  input x: f32[1, 784]
  input W1: f32[784, 128]
  input W2: f32[128, 10]

  node h = matmul(x, W1)
  node a = relu(h)
  node logits = matmul(a, W2)

  output y = logits
}
"#;
        let graph = parse(source).unwrap();
        assert_eq!(graph.id, "mlp");
        assert_eq!(graph.input_nodes().len(), 3);
        assert_eq!(graph.output_nodes().len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn parse_igqk_compression() {
        let source = r#"
graph compress {
  input weights: f32[768, 768]

  node compressed = to_ternary(weights) @proof theorem_5_2

  output out = compressed
}
"#;
        let graph = parse(source).unwrap();

        // Check proof annotation
        let compress_node = graph.nodes.iter().find(|n| matches!(n.op, Op::ToTernary)).unwrap();
        assert!(!compress_node.constraints.is_empty());
    }

    #[test]
    fn parse_with_comments() {
        let source = r#"
// This is a comment
graph test {
  # This is also a comment
  input x: f32[8]
  node r = relu(x)
  output y = r
}
"#;
        let graph = parse(source).unwrap();
        assert_eq!(graph.nodes.len(), 3);
    }

    #[test]
    fn roundtrip_parse_emit() {
        let source = r#"
graph roundtrip {
  input a: f32[4]
  input b: f32[4]

  node sum = add(a, b)
  node out = relu(sum)

  output result = out
}
"#;
        let graph = parse(source).unwrap();
        let emitted = to_qlang_text(&graph);

        // Parse the emitted text
        let graph2 = parse(&emitted).unwrap();

        assert_eq!(graph.id, graph2.id);
        assert_eq!(graph.nodes.len(), graph2.nodes.len());
        assert_eq!(graph.edges.len(), graph2.edges.len());
    }

    #[test]
    fn parse_quantum_ops() {
        let source = r#"
graph quantum {
  input state: f32[16]
  input gradient: f32[16]
  input partner: f32[16]

  node evolved = evolve(state, gradient)
  node measured = measure(evolved)
  node entangled = entangle(measured, partner)
  node collapsed = collapse(entangled)

  output result = collapsed
}
"#;
        let graph = parse(source).unwrap();
        assert_eq!(graph.nodes.len(), 8);

        let evolve_node = graph.nodes.iter().find(|n| matches!(n.op, Op::Evolve { .. })).unwrap();
        assert!(evolve_node.op.is_quantum());
        assert!(graph.nodes.iter().any(|n| matches!(n.op, Op::Entangle)));
        assert!(graph.nodes.iter().any(|n| matches!(n.op, Op::Collapse)));
    }

    #[test]
    fn roundtrip_quantum_emit_includes_entangle_and_collapse() {
        let source = r#"
graph quantum_roundtrip {
  input state: f32[4]
  input partner: f32[4]

  node entangled = entangle(state, partner)
  node collapsed = collapse(entangled)

  output result = collapsed
}
"#;
        let graph = parse(source).unwrap();
        let emitted = to_qlang_text(&graph);
        let reparsed = parse(&emitted).unwrap();

        assert!(emitted.contains("entangle("));
        assert!(emitted.contains("collapse("));
        assert!(reparsed.nodes.iter().any(|n| matches!(n.op, Op::Entangle)));
        assert!(reparsed.nodes.iter().any(|n| matches!(n.op, Op::Collapse)));
    }
}
