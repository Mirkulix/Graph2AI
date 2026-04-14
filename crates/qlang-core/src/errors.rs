//! Production-quality error system for QLANG.
//!
//! Provides rich, actionable error messages with source context,
//! colored output, intelligent fix suggestions via Levenshtein distance,
//! and multi-error collection for batch reporting.

use crate::graph::NodeId;
use std::fmt;

// ── Known operation names for typo suggestions ────────────────────────────────

/// All valid operation names in QLANG, used for suggesting fixes on typos.
const KNOWN_OPS: &[&str] = &[
    "input", "output", "const", "add", "sub", "mul", "div", "neg", "matmul",
    "transpose", "reshape", "slice", "concat", "reduce_sum", "reduce_mean",
    "reduce_max", "relu", "sigmoid", "tanh", "softmax", "superpose", "evolve",
    "measure", "entangle", "collapse", "entropy", "to_ternary", "to_lowrank",
    "to_sparse", "fisher_metric", "project", "layer_norm", "attention",
    "embedding", "residual", "gelu", "dropout", "cond", "scan", "subgraph",
];

/// Known dtype names for type-mismatch suggestions.
const KNOWN_DTYPES: &[&str] = &[
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "bool", "ternary",
];

// ── Core error type ──────────────────────────────────────────────────────────

/// Comprehensive error type covering every failure mode in QLANG.
#[derive(Debug, Clone)]
pub enum QlangError {
    /// Syntax / parse error with precise source location.
    ParseError {
        line: usize,
        col: usize,
        message: String,
        source_line: Option<String>,
        suggestion: Option<String>,
    },

    /// Type system violation.
    TypeError {
        node_id: Option<NodeId>,
        expected: String,
        got: String,
        suggestion: Option<String>,
    },

    /// Runtime failure during graph execution.
    RuntimeError {
        message: String,
        node_id: Option<NodeId>,
        stack_trace: Vec<String>,
    },

    /// Failure during a compilation phase.
    CompilationError {
        phase: String,
        message: String,
    },

    /// Filesystem / IO failure.
    IoError {
        path: String,
        operation: String,
        message: String,
    },

    /// Tensor shape mismatch.
    ShapeError {
        expected_shape: Vec<usize>,
        got_shape: Vec<usize>,
        node_id: Option<NodeId>,
    },

    /// Out-of-memory condition.
    MemoryError {
        requested_bytes: u64,
        available_bytes: u64,
    },

    /// Operation exceeded its time budget.
    TimeoutError {
        elapsed_ms: u64,
        limit_ms: u64,
    },
}

impl fmt::Display for QlangError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QlangError::ParseError { line, col, message, .. } => {
                write!(f, "parse error at {}:{}: {}", line, col, message)
            }
            QlangError::TypeError { expected, got, .. } => {
                write!(f, "type error: expected {}, got {}", expected, got)
            }
            QlangError::RuntimeError { message, .. } => {
                write!(f, "runtime error: {}", message)
            }
            QlangError::CompilationError { phase, message } => {
                write!(f, "compilation error in {}: {}", phase, message)
            }
            QlangError::IoError { path, operation, message } => {
                write!(f, "I/O error: {} on '{}': {}", operation, path, message)
            }
            QlangError::ShapeError { expected_shape, got_shape, .. } => {
                write!(
                    f,
                    "shape error: expected {:?}, got {:?}",
                    expected_shape, got_shape
                )
            }
            QlangError::MemoryError { requested_bytes, available_bytes } => {
                write!(
                    f,
                    "out of memory: requested {} bytes but only {} available",
                    requested_bytes, available_bytes
                )
            }
            QlangError::TimeoutError { elapsed_ms, limit_ms } => {
                write!(
                    f,
                    "timeout: operation took {}ms, limit was {}ms",
                    elapsed_ms, limit_ms
                )
            }
        }
    }
}

impl std::error::Error for QlangError {}

// ── Levenshtein distance ─────────────────────────────────────────────────────

/// Compute the Levenshtein (edit) distance between two strings.
///
/// This is the minimum number of single-character insertions, deletions,
/// or substitutions required to transform `a` into `b`.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let a_len = a_bytes.len();
    let b_len = b_bytes.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    // Use a single row + one previous value to keep memory O(min(m, n)).
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row: Vec<usize> = vec![0; b_len + 1];

    for i in 1..=a_len {
        curr_row[0] = i;
        for j in 1..=b_len {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] { 0 } else { 1 };
            curr_row[j] = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// Find the closest match for `input` among `candidates` using Levenshtein
/// distance. Returns `None` if the best match is too distant (more than half
/// the input length + 2, capped at a max distance of 3).
fn closest_match<'a>(input: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let max_allowed = (input.len() / 2 + 2).min(3);

    candidates
        .iter()
        .map(|c| (*c, levenshtein_distance(&input.to_lowercase(), &c.to_lowercase())))
        .filter(|(_, d)| *d <= max_allowed && *d > 0)
        .min_by_key(|(_, d)| *d)
        .map(|(c, _)| c)
}

// ── Intelligent suggestion engine ────────────────────────────────────────────

/// Generate an intelligent fix suggestion for the given error.
///
/// Strategies:
/// - **Typos**: Levenshtein distance against known operation/type names.
/// - **Shape mismatches**: Suggest inserting a reshape node.
/// - **Type mismatches**: Suggest appropriate cast.
pub fn suggest_fix(error: &QlangError) -> Option<String> {
    match error {
        QlangError::ParseError { message, suggestion, .. } => {
            // If the error already carries a suggestion, return it.
            if let Some(s) = suggestion {
                return Some(s.clone());
            }
            // Try to extract an unknown identifier from the message and match it.
            if let Some(unknown) = extract_unknown_name(message) {
                if let Some(closest) = closest_match(&unknown, KNOWN_OPS) {
                    return Some(format!("Did you mean: {} ?", closest));
                }
            }
            None
        }

        QlangError::TypeError { expected, got, suggestion, .. } => {
            if let Some(s) = suggestion {
                return Some(s.clone());
            }
            // Suggest a cast if both are known dtypes.
            let exp_lower = expected.to_lowercase();
            let got_lower = got.to_lowercase();
            if KNOWN_DTYPES.contains(&exp_lower.as_str())
                && KNOWN_DTYPES.contains(&got_lower.as_str())
            {
                return Some(format!(
                    "Insert a cast from {} to {}: cast({}, dtype={})",
                    got, expected, got, expected
                ));
            }
            // Typo in type name?
            if let Some(closest) = closest_match(got, KNOWN_DTYPES) {
                return Some(format!("Did you mean type: {} ?", closest));
            }
            None
        }

        QlangError::ShapeError { expected_shape, got_shape, .. } => {
            Some(format!(
                "Insert a reshape to convert {:?} -> {:?}: reshape(x, target_shape={:?})",
                got_shape, expected_shape, expected_shape
            ))
        }

        QlangError::MemoryError { requested_bytes, .. } => {
            let mb = *requested_bytes as f64 / (1024.0 * 1024.0);
            Some(format!(
                "Reduce memory usage ({:.1} MB requested). Consider: \
                 low-rank approximation (to_lowrank), sparsification (to_sparse), \
                 or processing in smaller batches.",
                mb
            ))
        }

        QlangError::TimeoutError { limit_ms, .. } => {
            Some(format!(
                "Operation exceeded the {}ms limit. Consider breaking the \
                 computation into smaller sub-graphs or increasing the timeout.",
                limit_ms
            ))
        }

        // No automatic suggestion for these.
        QlangError::RuntimeError { .. }
        | QlangError::CompilationError { .. }
        | QlangError::IoError { .. } => None,
    }
}

/// Try to pull an unknown name out of common error message patterns.
fn extract_unknown_name(message: &str) -> Option<String> {
    // Pattern: "unknown operation 'foo'"
    if let Some(start) = message.find('\'') {
        if let Some(end) = message[start + 1..].find('\'') {
            return Some(message[start + 1..start + 1 + end].to_string());
        }
    }
    None
}

// ── Pretty error formatter ───────────────────────────────────────────────────

/// ANSI color codes used for terminal output.
const RED: &str = "\x1b[31;1m";
const YELLOW: &str = "\x1b[33;1m";
const CYAN: &str = "\x1b[36m";
const GRAY: &str = "\x1b[90m";
const RESET: &str = "\x1b[0m";

/// Pretty-print a `QlangError` with source context, color, and suggestions.
///
/// If `source` is provided, the relevant source line is extracted and an
/// underline arrow points to the error column.
///
/// # Example output
///
/// ```text
/// Error at line 7, column 15:
///   node h = matmol(x, W)
///                ^^^^^^ unknown operation 'matmol'
///
/// Did you mean: matmul ?
/// ```
pub fn format_error(error: &QlangError, source: Option<&str>) -> String {
    let mut out = String::new();

    match error {
        QlangError::ParseError { line, col, message, source_line, .. } => {
            out.push_str(&format!(
                "{}error{} at line {}{}{}, column {}{}{}:\n",
                RED, RESET, CYAN, line, RESET, CYAN, col, RESET
            ));
            // Show the source line (from the error struct or the provided source).
            let display_line = source_line
                .as_deref()
                .or_else(|| {
                    source.and_then(|s| s.lines().nth(line.saturating_sub(1)))
                });
            if let Some(src) = display_line {
                out.push_str(&format!("  {}{}{}\n", GRAY, src, RESET));
                // Underline caret.
                if *col > 0 && *col <= src.len() + 1 {
                    let padding = " ".repeat(col - 1 + 2); // +2 for leading "  "
                    let word_len = token_length_at(src, *col);
                    let carets = "^".repeat(word_len.max(1));
                    out.push_str(&format!(
                        "{}{}{} {}{}\n",
                        padding, RED, carets, message, RESET
                    ));
                }
            } else {
                out.push_str(&format!("  {}{}{}\n", RED, message, RESET));
            }
        }

        QlangError::TypeError { node_id, expected, got, .. } => {
            out.push_str(&format!("{}error{}: type mismatch", RED, RESET));
            if let Some(id) = node_id {
                out.push_str(&format!(" at node {}{}{}", CYAN, id, RESET));
            }
            out.push('\n');
            out.push_str(&format!(
                "  expected {}{}{}, got {}{}{}\n",
                CYAN, expected, RESET, RED, got, RESET
            ));
        }

        QlangError::RuntimeError { message, node_id, stack_trace } => {
            out.push_str(&format!("{}runtime error{}", RED, RESET));
            if let Some(id) = node_id {
                out.push_str(&format!(" at node {}{}{}", CYAN, id, RESET));
            }
            out.push_str(&format!(": {}\n", message));
            if !stack_trace.is_empty() {
                out.push_str(&format!("{}stack trace:{}\n", GRAY, RESET));
                for (i, frame) in stack_trace.iter().enumerate() {
                    out.push_str(&format!("  {}{}. {}{}\n", GRAY, i, frame, RESET));
                }
            }
        }

        QlangError::CompilationError { phase, message } => {
            out.push_str(&format!(
                "{}compilation error{} in phase '{}{}{}':\n  {}\n",
                RED, RESET, CYAN, phase, RESET, message
            ));
        }

        QlangError::IoError { path, operation, message } => {
            out.push_str(&format!(
                "{}I/O error{}: {} on '{}{}{}': {}\n",
                RED, RESET, operation, CYAN, path, RESET, message
            ));
        }

        QlangError::ShapeError { expected_shape, got_shape, node_id } => {
            out.push_str(&format!("{}shape error{}", RED, RESET));
            if let Some(id) = node_id {
                out.push_str(&format!(" at node {}{}{}", CYAN, id, RESET));
            }
            out.push('\n');
            out.push_str(&format!(
                "  expected shape {}{:?}{}, got {}{:?}{}\n",
                CYAN, expected_shape, RESET, RED, got_shape, RESET
            ));
        }

        QlangError::MemoryError { requested_bytes, available_bytes } => {
            let req_mb = *requested_bytes as f64 / (1024.0 * 1024.0);
            let avail_mb = *available_bytes as f64 / (1024.0 * 1024.0);
            out.push_str(&format!(
                "{}out of memory{}: requested {:.1} MB but only {:.1} MB available\n",
                RED, RESET, req_mb, avail_mb
            ));
        }

        QlangError::TimeoutError { elapsed_ms, limit_ms } => {
            out.push_str(&format!(
                "{}timeout{}: operation took {}{}ms{} (limit: {}ms)\n",
                YELLOW, RESET, RED, elapsed_ms, RESET, limit_ms
            ));
        }
    }

    // Append suggestion if available.
    if let Some(suggestion) = suggest_fix(error) {
        out.push_str(&format!("\n{}{}{}\n", YELLOW, suggestion, RESET));
    }

    out
}

/// Estimate the length of the token starting at column `col` (1-based) in `line`.
fn token_length_at(line: &str, col: usize) -> usize {
    if col == 0 || col > line.len() {
        return 1;
    }
    let bytes = line.as_bytes();
    let start = col - 1;
    let mut end = start;
    while end < bytes.len() && !bytes[end].is_ascii_whitespace() && bytes[end] != b'(' && bytes[end] != b')' && bytes[end] != b',' {
        end += 1;
    }
    if end == start { 1 } else { end - start }
}

// ── Multi-error collection ───────────────────────────────────────────────────

/// Collects multiple errors so that processing can continue past the first
/// failure and present all problems at once.
#[derive(Debug, Default)]
pub struct ErrorRecovery {
    errors: Vec<QlangError>,
}

impl ErrorRecovery {
    /// Create a new, empty error collector.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Record an error and continue processing.
    pub fn continue_after_error(&mut self, error: QlangError) {
        self.errors.push(error);
    }

    /// All collected errors.
    pub fn errors(&self) -> &[QlangError] {
        &self.errors
    }

    /// Whether any errors were recorded.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// The first recorded error, if any.
    pub fn first_error(&self) -> Option<&QlangError> {
        self.errors.first()
    }

    /// Number of collected errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Format all collected errors into a single report string.
    pub fn format_all(&self, source: Option<&str>) -> String {
        let mut out = String::new();
        for (i, err) in self.errors.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format_error(err, source));
        }
        if self.errors.len() > 1 {
            out.push_str(&format!(
                "\n{}{} errors total.{}\n",
                RED,
                self.errors.len(),
                RESET
            ));
        }
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(levenshtein_distance("matmul", "matmul"), 0);
    }

    #[test]
    fn test_levenshtein_one_substitution() {
        assert_eq!(levenshtein_distance("matmol", "matmul"), 1);
    }

    #[test]
    fn test_levenshtein_empty_strings() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", "xyz"), 3);
    }

    #[test]
    fn test_levenshtein_insertion_deletion() {
        // "kitten" -> "sitting" = 3 (classic example)
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_suggest_fix_typo_in_parse_error() {
        let err = QlangError::ParseError {
            line: 7,
            col: 15,
            message: "unknown operation 'matmol'".to_string(),
            source_line: Some("node h = matmol(x, W)".to_string()),
            suggestion: None,
        };
        let suggestion = suggest_fix(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("matmul"));
    }

    #[test]
    fn test_suggest_fix_unknown_op() {
        let err = QlangError::ParseError {
            line: 1,
            col: 1,
            message: "unknown operation 'sofmax'".to_string(),
            source_line: None,
            suggestion: None,
        };
        let suggestion = suggest_fix(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("softmax"));
    }

    #[test]
    fn test_format_error_with_source_line() {
        let err = QlangError::ParseError {
            line: 7,
            col: 10,
            message: "unknown operation 'matmol'".to_string(),
            source_line: None,
            suggestion: None,
        };
        let source = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nnode h = matmol(x, W)\nline 8";
        let formatted = format_error(&err, Some(source));
        assert!(formatted.contains("matmol"));
        assert!(formatted.contains("matmul")); // suggestion
        // Line number is present (possibly surrounded by ANSI codes).
        assert!(formatted.contains("7"));
    }

    #[test]
    fn test_format_error_no_source() {
        let err = QlangError::ParseError {
            line: 1,
            col: 1,
            message: "unexpected token".to_string(),
            source_line: None,
            suggestion: None,
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("unexpected token"));
    }

    #[test]
    fn test_shape_error_formatting() {
        let err = QlangError::ShapeError {
            expected_shape: vec![32, 64],
            got_shape: vec![32, 128],
            node_id: Some(5),
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("[32, 64]"));
        assert!(formatted.contains("[32, 128]"));
        // Node id is present (possibly surrounded by ANSI codes).
        assert!(formatted.contains("5"));
        assert!(formatted.contains("reshape")); // suggestion
    }

    #[test]
    fn test_multiple_error_collection() {
        let mut recovery = ErrorRecovery::new();
        assert!(!recovery.has_errors());
        assert!(recovery.first_error().is_none());

        recovery.continue_after_error(QlangError::ParseError {
            line: 1,
            col: 1,
            message: "first error".to_string(),
            source_line: None,
            suggestion: None,
        });
        recovery.continue_after_error(QlangError::ParseError {
            line: 2,
            col: 1,
            message: "second error".to_string(),
            source_line: None,
            suggestion: None,
        });

        assert!(recovery.has_errors());
        assert_eq!(recovery.len(), 2);
        assert_eq!(recovery.errors().len(), 2);

        let first = recovery.first_error().unwrap();
        match first {
            QlangError::ParseError { message, .. } => {
                assert_eq!(message, "first error");
            }
            _ => panic!("expected ParseError"),
        }

        let report = recovery.format_all(None);
        assert!(report.contains("first error"));
        assert!(report.contains("second error"));
        assert!(report.contains("2 errors total"));
    }

    #[test]
    fn test_runtime_error_with_stack_trace() {
        let err = QlangError::RuntimeError {
            message: "division by zero".to_string(),
            node_id: Some(42),
            stack_trace: vec![
                "node 42: div(a, b)".to_string(),
                "node 30: subgraph(inner)".to_string(),
            ],
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("division by zero"));
        assert!(formatted.contains("node 42"));
        assert!(formatted.contains("stack trace"));
        assert!(formatted.contains("node 30: subgraph(inner)"));
    }

    #[test]
    fn test_timeout_error_message() {
        let err = QlangError::TimeoutError {
            elapsed_ms: 5000,
            limit_ms: 3000,
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("5000"));
        assert!(formatted.contains("3000"));
        assert!(formatted.contains("timeout"));
    }

    #[test]
    fn test_memory_error_message() {
        let err = QlangError::MemoryError {
            requested_bytes: 2_147_483_648, // 2 GB
            available_bytes: 1_073_741_824, // 1 GB
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("2048.0 MB"));
        assert!(formatted.contains("1024.0 MB"));
        assert!(formatted.contains("out of memory"));
        // Should also have a suggestion
        let suggestion = suggest_fix(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("low-rank"));
    }

    #[test]
    fn test_type_error_cast_suggestion() {
        let err = QlangError::TypeError {
            node_id: Some(10),
            expected: "f32".to_string(),
            got: "f64".to_string(),
            suggestion: None,
        };
        let suggestion = suggest_fix(&err);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("cast"));
    }

    #[test]
    fn test_io_error_display() {
        let err = QlangError::IoError {
            path: "/tmp/model.qlang".to_string(),
            operation: "read".to_string(),
            message: "file not found".to_string(),
        };
        let s = format!("{}", err);
        assert!(s.contains("/tmp/model.qlang"));
        assert!(s.contains("read"));
        assert!(s.contains("file not found"));
    }

    #[test]
    fn test_compilation_error_display() {
        let err = QlangError::CompilationError {
            phase: "optimization".to_string(),
            message: "cycle detected in graph".to_string(),
        };
        let formatted = format_error(&err, None);
        assert!(formatted.contains("optimization"));
        assert!(formatted.contains("cycle detected"));
    }
}
