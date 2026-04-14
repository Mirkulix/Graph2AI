//! LSP foundation for .qlang files — data structures and analysis.
//!
//! Provides diagnostics, completions, and hover info without any external
//! LSP dependencies. This module can be wired into a real LSP server later.

use std::collections::HashMap;

use crate::parser::{parse, ParseError};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Severity level for a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A diagnostic message attached to a source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: usize,
    pub col: usize,
    pub message: String,
    pub severity: Severity,
}

/// The kind of a completion item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Operation,
    Type,
    Variable,
    Keyword,
}

/// A single completion suggestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub documentation: String,
}

// ---------------------------------------------------------------------------
// Known operations and types
// ---------------------------------------------------------------------------

/// All recognised operations with short docs.
const KNOWN_OPS: &[(&str, &str)] = &[
    ("add", "Element-wise addition of two tensors"),
    ("sub", "Element-wise subtraction of two tensors"),
    ("mul", "Element-wise multiplication of two tensors"),
    ("div", "Element-wise division of two tensors"),
    ("neg", "Element-wise negation"),
    ("matmul", "Matrix multiplication: [m,k] x [k,n] -> [m,n]"),
    ("relu", "Rectified linear unit activation: max(0, x)"),
    ("sigmoid", "Sigmoid activation: 1/(1+exp(-x))"),
    ("tanh", "Hyperbolic tangent activation"),
    ("softmax", "Softmax along last axis"),
    ("transpose", "Transpose (reverses dimensions)"),
    ("to_ternary", "Quantize weights to ternary {-1, 0, +1} (IGQK)"),
    ("to_lowrank", "Low-rank approximation (default rank 16)"),
    ("superpose", "Create quantum superposition state"),
    ("measure", "Quantum measurement — collapse to classical"),
    ("entropy", "Compute von Neumann entropy (scalar output)"),
    ("evolve", "Quantum gradient flow evolution step"),
    ("project_ternary", "Project onto ternary manifold"),
    ("layer_norm", "Layer normalization (eps=1e-5)"),
    ("gelu", "Gaussian Error Linear Unit activation"),
    ("residual", "Residual (skip) connection"),
    ("dropout", "Dropout regularization (rate=0.1)"),
];

/// All recognised dtype names with short docs.
const KNOWN_TYPES: &[(&str, &str)] = &[
    ("f16", "16-bit floating point"),
    ("f32", "32-bit floating point"),
    ("f64", "64-bit floating point"),
    ("i8", "8-bit signed integer"),
    ("i16", "16-bit signed integer"),
    ("i32", "32-bit signed integer"),
    ("i64", "64-bit signed integer"),
    ("bool", "Boolean"),
    ("ternary", "Ternary {-1, 0, +1} — core to IGQK compression"),
];

/// Top-level keywords.
const KEYWORDS: &[(&str, &str)] = &[
    ("graph", "Define a new computation graph"),
    ("input", "Declare an input tensor"),
    ("node", "Declare a computation node"),
    ("output", "Declare a graph output"),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the raw lines of the source (0-indexed).
fn source_lines(source: &str) -> Vec<&str> {
    source.lines().collect()
}

/// Extract defined node/input names from the source, together with the line
/// they were defined on and a short description.
fn defined_names(source: &str) -> Vec<(String, usize, String)> {
    let mut names: Vec<(String, usize, String)> = Vec::new();
    for (idx, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.starts_with("input ") {
            if let Some(name) = extract_input_name(line) {
                let desc = format!("input defined on line {}", idx + 1);
                names.push((name, idx, desc));
            }
        } else if line.starts_with("node ") {
            if let Some(name) = extract_node_name(line) {
                let desc = format!("node defined on line {}", idx + 1);
                names.push((name, idx, desc));
            }
        }
    }
    names
}

fn extract_input_name(line: &str) -> Option<String> {
    // "input <name>: <type>"
    let rest = line.strip_prefix("input ")?;
    let name = rest.split(':').next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn extract_node_name(line: &str) -> Option<String> {
    // "node <name> = ..."
    let rest = line.strip_prefix("node ")?;
    let name = rest.split('=').next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Find the word (identifier) under the cursor at `(line, col)`.
fn word_at(source: &str, line: usize, col: usize) -> Option<String> {
    let lines = source_lines(source);
    let text = lines.get(line)?;
    if col > text.len() {
        return None;
    }

    let bytes = text.as_bytes();
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

    // Walk backwards from col to find start.
    let mut start = col;
    while start > 0 && is_ident(bytes[start - 1]) {
        start -= 1;
    }
    // Walk forwards from col to find end.
    let mut end = col;
    while end < bytes.len() && is_ident(bytes[end]) {
        end += 1;
    }

    if start == end {
        None
    } else {
        Some(text[start..end].to_string())
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse `.qlang` source text and return diagnostics.
///
/// Uses `crate::parser::parse` and translates any `ParseError` into a
/// `Diagnostic`. Also performs lightweight additional checks (warnings).
pub fn analyze_source(source: &str) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // 1. Run the real parser and collect errors.
    if let Err(e) = parse(source) {
        let (line, msg) = match &e {
            ParseError::SyntaxError { line, message } => (*line, message.clone()),
            ParseError::UndefinedNode(name) => (0, format!("undefined node '{name}'")),
            ParseError::UnknownType(name) => (0, format!("unknown type '{name}'")),
            ParseError::UnknownOp(name) => (0, format!("unknown operation '{name}'")),
        };
        diags.push(Diagnostic {
            line,
            col: 0,
            message: msg,
            severity: Severity::Error,
        });
    }

    // 2. Lightweight warnings (line-level scan).
    let mut has_graph = false;
    let mut has_output = false;
    let mut defined: HashMap<String, usize> = HashMap::new();
    let mut used: HashMap<String, bool> = HashMap::new();

    for (idx, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        let line_num = idx + 1;

        if line.starts_with("graph ") {
            has_graph = true;
        }

        if line.starts_with("output ") {
            has_output = true;
        }

        // Track definitions.
        if let Some(name) = line
            .strip_prefix("input ")
            .and_then(|r| r.split(':').next())
            .map(|n| n.trim().to_string())
        {
            if !name.is_empty() {
                if defined.contains_key(&name) {
                    diags.push(Diagnostic {
                        line: line_num,
                        col: 0,
                        message: format!("duplicate definition of '{name}'"),
                        severity: Severity::Warning,
                    });
                }
                defined.insert(name.clone(), line_num);
                used.insert(name, false);
            }
        }

        if let Some(name) = line
            .strip_prefix("node ")
            .and_then(|r| r.split('=').next())
            .map(|n| n.trim().to_string())
        {
            if !name.is_empty() {
                // Mark uses inside the RHS.
                if let Some(rhs) = line.strip_prefix("node ").and_then(|r| r.splitn(2, '=').nth(1))
                {
                    for key in used.keys().cloned().collect::<Vec<_>>() {
                        if rhs.contains(&key) {
                            used.insert(key, true);
                        }
                    }
                }
                if defined.contains_key(&name) {
                    diags.push(Diagnostic {
                        line: line_num,
                        col: 0,
                        message: format!("duplicate definition of '{name}'"),
                        severity: Severity::Warning,
                    });
                }
                defined.insert(name.clone(), line_num);
                used.insert(name, false);
            }
        }

        // Output references.
        if let Some(rhs) = line
            .strip_prefix("output ")
            .and_then(|r| r.splitn(2, '=').nth(1))
        {
            let ref_name = rhs.trim().to_string();
            if used.contains_key(&ref_name) {
                used.insert(ref_name, true);
            }
        }
    }

    if !source.trim().is_empty() && !has_graph {
        diags.push(Diagnostic {
            line: 1,
            col: 0,
            message: "missing 'graph' declaration".into(),
            severity: Severity::Warning,
        });
    }

    if has_graph && !has_output {
        diags.push(Diagnostic {
            line: 1,
            col: 0,
            message: "graph has no 'output' declaration".into(),
            severity: Severity::Info,
        });
    }

    diags
}

/// Return completion items appropriate for the cursor position.
///
/// Context heuristics (based on the text to the left of the cursor):
/// - After `node <name> = ` → operation completions
/// - After `input <name>: ` → type completions
/// - After an operation name and `(` → variable/node name completions
/// - Beginning of a line → keyword completions
pub fn completions_at(source: &str, line: usize, col: usize) -> Vec<CompletionItem> {
    let lines = source_lines(source);
    let text = match lines.get(line) {
        Some(t) => *t,
        None => return Vec::new(),
    };

    // Clamp col to line length.
    let col = col.min(text.len());
    let prefix = &text[..col];
    let trimmed = prefix.trim_start();

    // 1. After "node <name> = " → operations.
    if trimmed.starts_with("node ") {
        if let Some(after_eq) = trimmed.splitn(2, '=').nth(1) {
            let after_eq = after_eq.trim_start();
            // If we already have an op name and are inside parens → variable completions.
            if after_eq.contains('(') {
                return variable_completions(source, line);
            }
            // Otherwise → operation completions.
            return op_completions(after_eq);
        }
    }

    // 2. After "input <name>: " → type completions.
    if trimmed.starts_with("input ") {
        if trimmed.contains(':') {
            let after_colon = trimmed.splitn(2, ':').nth(1).unwrap_or("").trim_start();
            return type_completions(after_colon);
        }
    }

    // 3. After "output <name> = " → variable completions.
    if trimmed.starts_with("output ") && trimmed.contains('=') {
        return variable_completions(source, line);
    }

    // 4. At the beginning of a non-empty line inside a graph → keywords.
    keyword_completions(trimmed)
}

/// Return hover documentation for the symbol at `(line, col)`.
pub fn hover_info(source: &str, line: usize, col: usize) -> Option<String> {
    let word = word_at(source, line, col)?;

    // Check keywords.
    for &(kw, doc) in KEYWORDS {
        if word == kw {
            return Some(format!("**{kw}** (keyword)\n\n{doc}"));
        }
    }

    // Check operations.
    for &(op, doc) in KNOWN_OPS {
        if word == op {
            return Some(format!("**{op}** (operation)\n\n{doc}"));
        }
    }

    // Check types.
    for &(ty, doc) in KNOWN_TYPES {
        if word == ty {
            return Some(format!("**{ty}** (type)\n\n{doc}"));
        }
    }

    // Check user-defined names.
    let names = defined_names(source);
    for (name, def_line, desc) in &names {
        if word == *name {
            // Try to extract the full definition line for richer info.
            let lines_vec = source_lines(source);
            let def_text = lines_vec
                .get(*def_line)
                .map(|l| l.trim())
                .unwrap_or("");
            return Some(format!("**{name}** — {desc}\n\n```qlang\n{def_text}\n```"));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Completion helpers
// ---------------------------------------------------------------------------

fn op_completions(prefix: &str) -> Vec<CompletionItem> {
    KNOWN_OPS
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, doc)| CompletionItem {
            label: name.to_string(),
            kind: CompletionKind::Operation,
            documentation: doc.to_string(),
        })
        .collect()
}

fn type_completions(prefix: &str) -> Vec<CompletionItem> {
    KNOWN_TYPES
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, doc)| CompletionItem {
            label: name.to_string(),
            kind: CompletionKind::Type,
            documentation: doc.to_string(),
        })
        .collect()
}

fn variable_completions(source: &str, before_line: usize) -> Vec<CompletionItem> {
    defined_names(source)
        .into_iter()
        .filter(|(_, def_line, _)| *def_line < before_line)
        .map(|(name, _, desc)| CompletionItem {
            label: name,
            kind: CompletionKind::Variable,
            documentation: desc,
        })
        .collect()
}

fn keyword_completions(prefix: &str) -> Vec<CompletionItem> {
    KEYWORDS
        .iter()
        .filter(|(kw, _)| kw.starts_with(prefix))
        .map(|(kw, doc)| CompletionItem {
            label: kw.to_string(),
            kind: CompletionKind::Keyword,
            documentation: doc.to_string(),
        })
        .collect()
}

/// Return the definition location (line, col) for the symbol at `(line, col)`.
///
/// Looks up user-defined names (inputs, nodes) and returns the line they were
/// defined on.
pub fn goto_definition(source: &str, line: usize, col: usize) -> Option<(usize, usize)> {
    let word = word_at(source, line, col)?;

    let names = defined_names(source);
    for (name, def_line, _) in &names {
        if word == *name {
            // Return the column where the name starts on the definition line.
            let lines = source_lines(source);
            let def_text = lines.get(*def_line)?;
            let col_offset = def_text.find(&word).unwrap_or(0);
            return Some((*def_line, col_offset));
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
graph demo {
  input x: f32[4]
  input W: f32[4, 8]

  node h = matmul(x, W)
  node a = relu(h)
  node c = to_ternary(a) @proof theorem_5_2

  output y = c
}
"#;

    // ----- analyze_source ---------------------------------------------------

    #[test]
    fn analyze_valid_source_no_errors() {
        let diags = analyze_source(SAMPLE);
        let errors: Vec<_> = diags.iter().filter(|d| d.severity == Severity::Error).collect();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn analyze_syntax_error() {
        let bad = "graph broken {\n  node x = unknown_op()\n  output y = x\n}\n";
        let diags = analyze_source(bad);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Error),
            "expected an error diagnostic"
        );
    }

    #[test]
    fn analyze_missing_graph() {
        let bad = "input x: f32[4]\n";
        let diags = analyze_source(bad);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Warning
                && d.message.contains("graph")),
            "expected warning about missing graph: {diags:?}"
        );
    }

    #[test]
    fn analyze_no_output_info() {
        let src = "graph g {\n  input x: f32[4]\n}\n";
        let diags = analyze_source(src);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Info
                && d.message.contains("output")),
            "expected info about missing output: {diags:?}"
        );
    }

    #[test]
    fn analyze_duplicate_definition() {
        let src = "graph g {\n  input x: f32[4]\n  input x: f32[8]\n  output y = x\n}\n";
        let diags = analyze_source(src);
        assert!(
            diags.iter().any(|d| d.severity == Severity::Warning
                && d.message.contains("duplicate")),
            "expected duplicate warning: {diags:?}"
        );
    }

    // ----- completions_at ---------------------------------------------------

    #[test]
    fn completions_ops_after_node_eq() {
        // "  node z = " — cursor at end of line
        let src = "graph g {\n  input x: f32[4]\n  node z = \n}\n";
        let items = completions_at(src, 2, 11);
        assert!(!items.is_empty(), "expected operation completions");
        assert!(items.iter().all(|c| c.kind == CompletionKind::Operation));
        let labels: Vec<_> = items.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"relu"), "expected relu in completions");
        assert!(labels.contains(&"matmul"), "expected matmul in completions");
    }

    #[test]
    fn completions_ops_prefix_filter() {
        // "  node z = re" — partial prefix
        let src = "graph g {\n  input x: f32[4]\n  node z = re\n}\n";
        let items = completions_at(src, 2, 13);
        assert!(!items.is_empty());
        for c in &items {
            assert!(
                c.label.starts_with("re"),
                "completion '{}' should start with 're'",
                c.label
            );
        }
    }

    #[test]
    fn completions_types_after_input_colon() {
        // "  input y: " — cursor after colon
        let src = "graph g {\n  input y: \n}\n";
        let items = completions_at(src, 1, 12);
        assert!(!items.is_empty(), "expected type completions");
        assert!(items.iter().all(|c| c.kind == CompletionKind::Type));
        let labels: Vec<_> = items.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"f32"));
        assert!(labels.contains(&"ternary"));
    }

    #[test]
    fn completions_variables_inside_parens() {
        // "  node h = matmul(" — cursor inside parens
        let src = "graph g {\n  input x: f32[4]\n  input W: f32[4, 8]\n  node h = matmul(\n}\n";
        let items = completions_at(src, 3, 19);
        assert!(!items.is_empty(), "expected variable completions");
        assert!(items.iter().all(|c| c.kind == CompletionKind::Variable));
        let labels: Vec<_> = items.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"x"));
        assert!(labels.contains(&"W"));
    }

    #[test]
    fn completions_keywords_at_line_start() {
        let src = "graph g {\n  \n}\n";
        let items = completions_at(src, 1, 2);
        assert!(!items.is_empty(), "expected keyword completions");
        let labels: Vec<_> = items.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"input"));
        assert!(labels.contains(&"node"));
        assert!(labels.contains(&"output"));
    }

    // ----- hover_info -------------------------------------------------------

    #[test]
    fn hover_keyword() {
        let info = hover_info(SAMPLE, 1, 0).unwrap();
        assert!(info.contains("graph"), "expected 'graph' hover: {info}");
        assert!(info.contains("keyword"));
    }

    #[test]
    fn hover_operation() {
        // Line 5: "  node h = matmul(x, W)"
        let info = hover_info(SAMPLE, 5, 13).unwrap();
        assert!(info.contains("matmul"), "expected 'matmul' hover: {info}");
        assert!(info.contains("operation"));
    }

    #[test]
    fn hover_type() {
        // Line 2: "  input x: f32[4]"
        let info = hover_info(SAMPLE, 2, 13).unwrap();
        assert!(info.contains("f32"), "expected 'f32' hover: {info}");
        assert!(info.contains("type"));
    }

    #[test]
    fn hover_user_defined_name() {
        // "x" on line 5: "  node h = matmul(x, W)"
        let info = hover_info(SAMPLE, 5, 19).unwrap();
        assert!(info.contains("x"), "expected 'x' hover: {info}");
        assert!(info.contains("input"));
    }

    #[test]
    fn hover_no_symbol() {
        // Cursor on whitespace.
        let info = hover_info(SAMPLE, 0, 0);
        assert!(info.is_none(), "expected None for empty line");
    }
}
