//! QLANG Standard Library
//!
//! Provides built-in functions for math, arrays, strings, I/O, types,
//! tensors, random numbers, and time.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// Runtime value type for the QLANG interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Array(Vec<f64>),
    String(String),
    Bool(bool),
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    Null,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{n}"),
            Value::Array(a) => write!(f, "{a:?}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Tensor { data, shape } => write!(f, "Tensor(shape={shape:?}, data={data:?})"),
            Value::Null => write!(f, "null"),
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors returned by standard-library functions.
#[derive(Debug, Clone, PartialEq)]
pub enum StdLibError {
    /// Unknown function name.
    UnknownFunction(String),
    /// Wrong number of arguments.
    ArityMismatch { name: String, expected: String, got: usize },
    /// Argument has the wrong type.
    TypeError { name: String, message: String },
    /// Runtime error (e.g. I/O failure).
    RuntimeError(String),
}

impl fmt::Display for StdLibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StdLibError::UnknownFunction(name) => write!(f, "unknown function: {name}"),
            StdLibError::ArityMismatch { name, expected, got } => {
                write!(f, "{name}: expected {expected} argument(s), got {got}")
            }
            StdLibError::TypeError { name, message } => write!(f, "{name}: {message}"),
            StdLibError::RuntimeError(msg) => write!(f, "runtime error: {msg}"),
        }
    }
}

impl std::error::Error for StdLibError {}

// ---------------------------------------------------------------------------
// Helper macros / functions
// ---------------------------------------------------------------------------

fn arity(name: &str, args: &[Value], n: usize) -> Result<(), StdLibError> {
    if args.len() != n {
        Err(StdLibError::ArityMismatch {
            name: name.to_string(),
            expected: n.to_string(),
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

fn arity_range(name: &str, args: &[Value], lo: usize, hi: usize) -> Result<(), StdLibError> {
    if args.len() < lo || args.len() > hi {
        Err(StdLibError::ArityMismatch {
            name: name.to_string(),
            expected: format!("{lo}..{hi}"),
            got: args.len(),
        })
    } else {
        Ok(())
    }
}

fn as_number(name: &str, v: &Value) -> Result<f64, StdLibError> {
    match v {
        Value::Number(n) => Ok(*n),
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: format!("expected Number, got {}", type_name(v)),
        }),
    }
}

fn as_array(name: &str, v: &Value) -> Result<Vec<f64>, StdLibError> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: format!("expected Array, got {}", type_name(v)),
        }),
    }
}

fn as_string(name: &str, v: &Value) -> Result<String, StdLibError> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: format!("expected String, got {}", type_name(v)),
        }),
    }
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Number(_) => "Number",
        Value::Array(_) => "Array",
        Value::String(_) => "String",
        Value::Bool(_) => "Bool",
        Value::Tensor { .. } => "Tensor",
        Value::Null => "Null",
    }
}

// ---------------------------------------------------------------------------
// Simple LCG random (no external deps)
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        // Use upper 32 bits for quality.
        let upper = (self.state >> 33) as f64;
        upper / (1u64 << 31) as f64
    }
}

// ---------------------------------------------------------------------------
// StdLib
// ---------------------------------------------------------------------------

type BuiltinFn = fn(&str, &[Value], &mut Lcg) -> Result<Value, StdLibError>;

/// The QLANG standard library.
pub struct StdLib {
    fns: HashMap<&'static str, BuiltinFn>,
    rng: Lcg,
}

impl StdLib {
    /// Create a new standard library instance with all built-in functions.
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::with_seed(seed)
    }

    /// Create with a fixed RNG seed (useful for tests).
    pub fn with_seed(seed: u64) -> Self {
        let mut fns = HashMap::<&'static str, BuiltinFn>::new();

        // Math
        fns.insert("abs", stdlib_abs);
        fns.insert("sqrt", stdlib_sqrt);
        fns.insert("pow", stdlib_pow);
        fns.insert("min", stdlib_min);
        fns.insert("max", stdlib_max);
        fns.insert("floor", stdlib_floor);
        fns.insert("ceil", stdlib_ceil);
        fns.insert("round", stdlib_round);
        fns.insert("sin", stdlib_sin);
        fns.insert("cos", stdlib_cos);
        fns.insert("log", stdlib_log);
        fns.insert("exp", stdlib_exp);

        // Array
        fns.insert("len", stdlib_len);
        fns.insert("sum", stdlib_sum);
        fns.insert("mean", stdlib_mean);
        fns.insert("max_val", stdlib_max_val);
        fns.insert("min_val", stdlib_min_val);
        fns.insert("sort", stdlib_sort);
        fns.insert("reverse", stdlib_reverse);
        fns.insert("range", stdlib_range);
        fns.insert("zeros", stdlib_zeros);
        fns.insert("ones", stdlib_ones);
        fns.insert("linspace", stdlib_linspace);

        // String
        fns.insert("str", stdlib_str);
        fns.insert("concat", stdlib_concat);
        fns.insert("split", stdlib_split_fn);
        fns.insert("trim", stdlib_trim);
        fns.insert("contains", stdlib_contains);
        fns.insert("replace", stdlib_replace);
        fns.insert("to_upper", stdlib_to_upper);
        fns.insert("to_lower", stdlib_to_lower);
        fns.insert("starts_with", stdlib_starts_with);
        fns.insert("ends_with", stdlib_ends_with);

        // I/O
        fns.insert("print", stdlib_print);
        fns.insert("println", stdlib_println);
        fns.insert("read_file", stdlib_read_file);
        fns.insert("write_file", stdlib_write_file);

        // Type
        fns.insert("type_of", stdlib_type_of);
        fns.insert("is_number", stdlib_is_number);
        fns.insert("is_array", stdlib_is_array);
        fns.insert("is_string", stdlib_is_string);
        fns.insert("to_number", stdlib_to_number);
        fns.insert("to_string", stdlib_to_string);

        // Tensor
        fns.insert("shape", stdlib_shape);
        fns.insert("reshape", stdlib_reshape);
        fns.insert("transpose", stdlib_transpose);
        fns.insert("dot", stdlib_dot);
        fns.insert("matmul", stdlib_matmul);

        // Random
        fns.insert("random", stdlib_random);
        fns.insert("random_range", stdlib_random_range);
        fns.insert("random_array", stdlib_random_array);

        // Time
        fns.insert("clock", stdlib_clock);

        Self { fns, rng: Lcg::new(seed) }
    }

    /// Call a built-in function by name.
    pub fn call(&mut self, name: &str, args: &[Value]) -> Result<Value, StdLibError> {
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| StdLibError::UnknownFunction(name.to_string()))?;
        f(name, args, &mut self.rng)
    }

    /// Return a list of all registered functions with short descriptions.
    pub fn list_functions() -> Vec<(&'static str, &'static str)> {
        vec![
            // Math
            ("abs", "absolute value"),
            ("sqrt", "square root"),
            ("pow", "exponentiation"),
            ("min", "minimum of two numbers"),
            ("max", "maximum of two numbers"),
            ("floor", "floor"),
            ("ceil", "ceiling"),
            ("round", "round to nearest integer"),
            ("sin", "sine"),
            ("cos", "cosine"),
            ("log", "natural logarithm"),
            ("exp", "e^x"),
            // Array
            ("len", "length of array or string"),
            ("sum", "sum of array"),
            ("mean", "mean of array"),
            ("max_val", "maximum value in array"),
            ("min_val", "minimum value in array"),
            ("sort", "sort array ascending"),
            ("reverse", "reverse array"),
            ("range", "range(start, stop[, step])"),
            ("zeros", "array of zeros"),
            ("ones", "array of ones"),
            ("linspace", "linearly spaced values"),
            // String
            ("str", "convert value to string"),
            ("concat", "concatenate strings"),
            ("split", "split string by delimiter"),
            ("trim", "trim whitespace"),
            ("contains", "check if string contains substring"),
            ("replace", "replace occurrences in string"),
            ("to_upper", "uppercase"),
            ("to_lower", "lowercase"),
            ("starts_with", "check prefix"),
            ("ends_with", "check suffix"),
            // I/O
            ("print", "print without newline"),
            ("println", "print with newline"),
            ("read_file", "read file to string"),
            ("write_file", "write string to file"),
            // Type
            ("type_of", "type name as string"),
            ("is_number", "check if Number"),
            ("is_array", "check if Array"),
            ("is_string", "check if String"),
            ("to_number", "convert to Number"),
            ("to_string", "convert to String"),
            // Tensor
            ("shape", "tensor shape"),
            ("reshape", "reshape tensor"),
            ("transpose", "transpose 2-D tensor"),
            ("dot", "dot product"),
            ("matmul", "matrix multiplication"),
            // Random
            ("random", "random f64 in [0,1)"),
            ("random_range", "random f64 in [lo,hi)"),
            ("random_array", "array of random f64"),
            // Time
            ("clock", "seconds since epoch"),
        ]
    }
}

impl Default for StdLib {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Built-in function implementations
// ===========================================================================

// ---- Math -----------------------------------------------------------------

fn stdlib_abs(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.abs()))
}

fn stdlib_sqrt(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.sqrt()))
}

fn stdlib_pow(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let base = as_number(name, &args[0])?;
    let exp = as_number(name, &args[1])?;
    Ok(Value::Number(base.powf(exp)))
}

fn stdlib_min(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let a = as_number(name, &args[0])?;
    let b = as_number(name, &args[1])?;
    Ok(Value::Number(a.min(b)))
}

fn stdlib_max(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let a = as_number(name, &args[0])?;
    let b = as_number(name, &args[1])?;
    Ok(Value::Number(a.max(b)))
}

fn stdlib_floor(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.floor()))
}

fn stdlib_ceil(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.ceil()))
}

fn stdlib_round(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.round()))
}

fn stdlib_sin(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.sin()))
}

fn stdlib_cos(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.cos()))
}

fn stdlib_log(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.ln()))
}

fn stdlib_exp(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Number(as_number(name, &args[0])?.exp()))
}

// ---- Array ----------------------------------------------------------------

fn stdlib_len(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    match &args[0] {
        Value::Array(a) => Ok(Value::Number(a.len() as f64)),
        Value::String(s) => Ok(Value::Number(s.len() as f64)),
        Value::Tensor { shape, .. } => {
            Ok(Value::Number(shape.iter().product::<usize>() as f64))
        }
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: "expected Array, String, or Tensor".to_string(),
        }),
    }
}

fn stdlib_sum(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let a = as_array(name, &args[0])?;
    Ok(Value::Number(a.iter().sum()))
}

fn stdlib_mean(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let a = as_array(name, &args[0])?;
    if a.is_empty() {
        return Ok(Value::Number(f64::NAN));
    }
    Ok(Value::Number(a.iter().sum::<f64>() / a.len() as f64))
}

fn stdlib_max_val(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let a = as_array(name, &args[0])?;
    if a.is_empty() {
        return Ok(Value::Null);
    }
    Ok(Value::Number(
        a.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    ))
}

fn stdlib_min_val(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let a = as_array(name, &args[0])?;
    if a.is_empty() {
        return Ok(Value::Null);
    }
    Ok(Value::Number(
        a.iter().cloned().fold(f64::INFINITY, f64::min),
    ))
}

fn stdlib_sort(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let mut a = as_array(name, &args[0])?;
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Value::Array(a))
}

fn stdlib_reverse(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let mut a = as_array(name, &args[0])?;
    a.reverse();
    Ok(Value::Array(a))
}

fn stdlib_range(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity_range(name, args, 1, 3)?;
    let (start, stop, step) = match args.len() {
        1 => (0.0, as_number(name, &args[0])?, 1.0),
        2 => (as_number(name, &args[0])?, as_number(name, &args[1])?, 1.0),
        _ => (
            as_number(name, &args[0])?,
            as_number(name, &args[1])?,
            as_number(name, &args[2])?,
        ),
    };
    if step == 0.0 {
        return Err(StdLibError::RuntimeError("range: step cannot be 0".into()));
    }
    let mut out = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < stop {
            out.push(v);
            v += step;
        }
    } else {
        while v > stop {
            out.push(v);
            v += step;
        }
    }
    Ok(Value::Array(out))
}

fn stdlib_zeros(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let n = as_number(name, &args[0])? as usize;
    Ok(Value::Array(vec![0.0; n]))
}

fn stdlib_ones(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let n = as_number(name, &args[0])? as usize;
    Ok(Value::Array(vec![1.0; n]))
}

fn stdlib_linspace(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 3)?;
    let start = as_number(name, &args[0])?;
    let stop = as_number(name, &args[1])?;
    let n = as_number(name, &args[2])? as usize;
    if n == 0 {
        return Ok(Value::Array(vec![]));
    }
    if n == 1 {
        return Ok(Value::Array(vec![start]));
    }
    let step = (stop - start) / (n - 1) as f64;
    let out: Vec<f64> = (0..n).map(|i| start + step * i as f64).collect();
    Ok(Value::Array(out))
}

// ---- String ---------------------------------------------------------------

fn stdlib_str(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::String(format!("{}", args[0])))
}

fn stdlib_concat(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let a = as_string(name, &args[0])?;
    let b = as_string(name, &args[1])?;
    Ok(Value::String(format!("{a}{b}")))
}

fn stdlib_split_fn(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let s = as_string(name, &args[0])?;
    let delim = as_string(name, &args[1])?;
    // Return an array of strings packed as a single Value::Array is not possible
    // since Array holds f64. We return a String with parts joined by '\0' for now,
    // but a better approach: return the first part as demonstration or use Null.
    // For practical use we return a string with JSON-like representation.
    let parts: Vec<&str> = s.split(&delim).collect();
    // Return as a string representation (QLANG arrays are f64 only).
    let joined = parts.join("\n");
    Ok(Value::String(joined))
}

fn stdlib_trim(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let s = as_string(name, &args[0])?;
    Ok(Value::String(s.trim().to_string()))
}

fn stdlib_contains(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let s = as_string(name, &args[0])?;
    let sub = as_string(name, &args[1])?;
    Ok(Value::Bool(s.contains(&sub)))
}

fn stdlib_replace(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 3)?;
    let s = as_string(name, &args[0])?;
    let from = as_string(name, &args[1])?;
    let to = as_string(name, &args[2])?;
    Ok(Value::String(s.replace(&from, &to)))
}

fn stdlib_to_upper(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::String(as_string(name, &args[0])?.to_uppercase()))
}

fn stdlib_to_lower(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::String(as_string(name, &args[0])?.to_lowercase()))
}

fn stdlib_starts_with(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let s = as_string(name, &args[0])?;
    let prefix = as_string(name, &args[1])?;
    Ok(Value::Bool(s.starts_with(&prefix)))
}

fn stdlib_ends_with(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let s = as_string(name, &args[0])?;
    let suffix = as_string(name, &args[1])?;
    Ok(Value::Bool(s.ends_with(&suffix)))
}

// ---- I/O ------------------------------------------------------------------

fn stdlib_print(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    print!("{}", args[0]);
    Ok(Value::Null)
}

fn stdlib_println(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    println!("{}", args[0]);
    Ok(Value::Null)
}

fn stdlib_read_file(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let path = as_string(name, &args[0])?;
    let contents = std::fs::read_to_string(&path)
        .map_err(|e| StdLibError::RuntimeError(format!("read_file({path}): {e}")))?;
    Ok(Value::String(contents))
}

fn stdlib_write_file(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let path = as_string(name, &args[0])?;
    let contents = as_string(name, &args[1])?;
    std::fs::write(&path, &contents)
        .map_err(|e| StdLibError::RuntimeError(format!("write_file({path}): {e}")))?;
    Ok(Value::Null)
}

// ---- Type -----------------------------------------------------------------

fn stdlib_type_of(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::String(type_name(&args[0]).to_string()))
}

fn stdlib_is_number(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Bool(matches!(args[0], Value::Number(_))))
}

fn stdlib_is_array(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Bool(matches!(args[0], Value::Array(_))))
}

fn stdlib_is_string(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::Bool(matches!(args[0], Value::String(_))))
}

fn stdlib_to_number(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    match &args[0] {
        Value::Number(n) => Ok(Value::Number(*n)),
        Value::String(s) => {
            let n: f64 = s.parse().map_err(|_| StdLibError::TypeError {
                name: name.to_string(),
                message: format!("cannot parse '{s}' as number"),
            })?;
            Ok(Value::Number(n))
        }
        Value::Bool(b) => Ok(Value::Number(if *b { 1.0 } else { 0.0 })),
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: format!("cannot convert {} to Number", type_name(&args[0])),
        }),
    }
}

fn stdlib_to_string(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    Ok(Value::String(format!("{}", args[0])))
}

// ---- Tensor ---------------------------------------------------------------

fn stdlib_shape(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    match &args[0] {
        Value::Tensor { shape, .. } => {
            Ok(Value::Array(shape.iter().map(|&s| s as f64).collect()))
        }
        Value::Array(a) => Ok(Value::Array(vec![a.len() as f64])),
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: "expected Tensor or Array".to_string(),
        }),
    }
}

fn stdlib_reshape(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let (data, _old_shape) = match &args[0] {
        Value::Tensor { data, shape } => (data.clone(), shape.clone()),
        Value::Array(a) => (a.clone(), vec![a.len()]),
        _ => {
            return Err(StdLibError::TypeError {
                name: name.to_string(),
                message: "first argument must be Tensor or Array".to_string(),
            })
        }
    };
    let new_shape = as_array(name, &args[1])?;
    let new_shape: Vec<usize> = new_shape.iter().map(|&x| x as usize).collect();
    let total: usize = new_shape.iter().product();
    if total != data.len() {
        return Err(StdLibError::RuntimeError(format!(
            "reshape: cannot reshape {} elements into shape {new_shape:?}",
            data.len()
        )));
    }
    Ok(Value::Tensor { data, shape: new_shape })
}

fn stdlib_transpose(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    match &args[0] {
        Value::Tensor { data, shape } => {
            if shape.len() != 2 {
                return Err(StdLibError::RuntimeError(
                    "transpose: only 2-D tensors supported".to_string(),
                ));
            }
            let (rows, cols) = (shape[0], shape[1]);
            let mut out = vec![0.0; data.len()];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
            Ok(Value::Tensor {
                data: out,
                shape: vec![cols, rows],
            })
        }
        _ => Err(StdLibError::TypeError {
            name: name.to_string(),
            message: "expected Tensor".to_string(),
        }),
    }
}

fn stdlib_dot(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let a = as_array(name, &args[0])?;
    let b = as_array(name, &args[1])?;
    if a.len() != b.len() {
        return Err(StdLibError::RuntimeError(format!(
            "dot: length mismatch ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    let s: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok(Value::Number(s))
}

fn stdlib_matmul(name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let (a_data, a_shape) = match &args[0] {
        Value::Tensor { data, shape } => (data, shape),
        _ => {
            return Err(StdLibError::TypeError {
                name: name.to_string(),
                message: "first argument must be Tensor".to_string(),
            })
        }
    };
    let (b_data, b_shape) = match &args[1] {
        Value::Tensor { data, shape } => (data, shape),
        _ => {
            return Err(StdLibError::TypeError {
                name: name.to_string(),
                message: "second argument must be Tensor".to_string(),
            })
        }
    };
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(StdLibError::RuntimeError(
            "matmul: only 2-D tensors supported".to_string(),
        ));
    }
    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    if k1 != k2 {
        return Err(StdLibError::RuntimeError(format!(
            "matmul: inner dimensions mismatch ({k1} vs {k2})"
        )));
    }
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k1 {
                s += a_data[i * k1 + p] * b_data[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
    Ok(Value::Tensor {
        data: out,
        shape: vec![m, n],
    })
}

// ---- Random ---------------------------------------------------------------

fn stdlib_random(_name: &str, args: &[Value], rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity("random", args, 0)?;
    Ok(Value::Number(rng.next_f64()))
}

fn stdlib_random_range(name: &str, args: &[Value], rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity(name, args, 2)?;
    let lo = as_number(name, &args[0])?;
    let hi = as_number(name, &args[1])?;
    let v = lo + (hi - lo) * rng.next_f64();
    Ok(Value::Number(v))
}

fn stdlib_random_array(
    name: &str,
    args: &[Value],
    rng: &mut Lcg,
) -> Result<Value, StdLibError> {
    arity(name, args, 1)?;
    let n = as_number(name, &args[0])? as usize;
    let out: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    Ok(Value::Array(out))
}

// ---- Time -----------------------------------------------------------------

fn stdlib_clock(_name: &str, args: &[Value], _rng: &mut Lcg) -> Result<Value, StdLibError> {
    arity("clock", args, 0)?;
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    Ok(Value::Number(secs))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> StdLib {
        StdLib::with_seed(12345)
    }

    // 1. Math: abs, sqrt, pow
    #[test]
    fn test_math_abs_sqrt_pow() {
        let mut sl = lib();
        assert_eq!(
            sl.call("abs", &[Value::Number(-3.0)]).unwrap(),
            Value::Number(3.0)
        );
        assert_eq!(
            sl.call("sqrt", &[Value::Number(9.0)]).unwrap(),
            Value::Number(3.0)
        );
        assert_eq!(
            sl.call("pow", &[Value::Number(2.0), Value::Number(10.0)]).unwrap(),
            Value::Number(1024.0)
        );
    }

    // 2. Math: sin, cos, exp, log
    #[test]
    fn test_math_trig_exp_log() {
        let mut sl = lib();
        let sin_val = sl.call("sin", &[Value::Number(0.0)]).unwrap();
        assert_eq!(sin_val, Value::Number(0.0));

        let cos_val = sl.call("cos", &[Value::Number(0.0)]).unwrap();
        assert_eq!(cos_val, Value::Number(1.0));

        let exp_val = sl.call("exp", &[Value::Number(0.0)]).unwrap();
        assert_eq!(exp_val, Value::Number(1.0));

        if let Value::Number(v) = sl.call("log", &[Value::Number(1.0)]).unwrap() {
            assert!((v - 0.0).abs() < 1e-12);
        } else {
            panic!("expected Number");
        }
    }

    // 3. Array: sum, mean
    #[test]
    fn test_array_sum_mean() {
        let mut sl = lib();
        let arr = Value::Array(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sl.call("sum", &[arr.clone()]).unwrap(), Value::Number(10.0));
        assert_eq!(sl.call("mean", &[arr]).unwrap(), Value::Number(2.5));
    }

    // 4. Array: sort, range
    #[test]
    fn test_array_sort_range() {
        let mut sl = lib();
        let sorted = sl
            .call("sort", &[Value::Array(vec![3.0, 1.0, 2.0])])
            .unwrap();
        assert_eq!(sorted, Value::Array(vec![1.0, 2.0, 3.0]));

        let r = sl.call("range", &[Value::Number(5.0)]).unwrap();
        assert_eq!(r, Value::Array(vec![0.0, 1.0, 2.0, 3.0, 4.0]));

        let r2 = sl
            .call("range", &[Value::Number(1.0), Value::Number(4.0)])
            .unwrap();
        assert_eq!(r2, Value::Array(vec![1.0, 2.0, 3.0]));
    }

    // 5. String: concat
    #[test]
    fn test_string_concat() {
        let mut sl = lib();
        let result = sl
            .call(
                "concat",
                &[
                    Value::String("hello ".into()),
                    Value::String("world".into()),
                ],
            )
            .unwrap();
        assert_eq!(result, Value::String("hello world".into()));
    }

    // 6. String: split, contains
    #[test]
    fn test_string_split_contains() {
        let mut sl = lib();
        let split = sl
            .call(
                "split",
                &[Value::String("a,b,c".into()), Value::String(",".into())],
            )
            .unwrap();
        assert_eq!(split, Value::String("a\nb\nc".into()));

        let c = sl
            .call(
                "contains",
                &[
                    Value::String("hello world".into()),
                    Value::String("world".into()),
                ],
            )
            .unwrap();
        assert_eq!(c, Value::Bool(true));
    }

    // 7. Type checking
    #[test]
    fn test_type_functions() {
        let mut sl = lib();
        assert_eq!(
            sl.call("type_of", &[Value::Number(1.0)]).unwrap(),
            Value::String("Number".into())
        );
        assert_eq!(
            sl.call("is_number", &[Value::Number(1.0)]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            sl.call("is_number", &[Value::String("x".into())]).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            sl.call("is_array", &[Value::Array(vec![])]).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            sl.call("is_string", &[Value::String("hi".into())]).unwrap(),
            Value::Bool(true)
        );
    }

    // 8. Random number generation
    #[test]
    fn test_random() {
        let mut sl = lib();
        if let Value::Number(v) = sl.call("random", &[]).unwrap() {
            assert!((0.0..1.0).contains(&v));
        } else {
            panic!("expected Number");
        }

        if let Value::Number(v) = sl
            .call(
                "random_range",
                &[Value::Number(10.0), Value::Number(20.0)],
            )
            .unwrap()
        {
            assert!((10.0..20.0).contains(&v));
        } else {
            panic!("expected Number");
        }

        if let Value::Array(a) = sl.call("random_array", &[Value::Number(5.0)]).unwrap() {
            assert_eq!(a.len(), 5);
            for v in &a {
                assert!((0.0..1.0).contains(v));
            }
        } else {
            panic!("expected Array");
        }
    }

    // 9. Tensor: reshape
    #[test]
    fn test_tensor_reshape() {
        let mut sl = lib();
        let t = Value::Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
        };
        let reshaped = sl
            .call("reshape", &[t, Value::Array(vec![3.0, 2.0])])
            .unwrap();
        assert_eq!(
            reshaped,
            Value::Tensor {
                data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape: vec![3, 2],
            }
        );
    }

    // 10. Tensor: transpose
    #[test]
    fn test_tensor_transpose() {
        let mut sl = lib();
        let t = Value::Tensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
        };
        let tr = sl.call("transpose", &[t]).unwrap();
        assert_eq!(
            tr,
            Value::Tensor {
                data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
                shape: vec![3, 2],
            }
        );
    }

    // 11. Error: wrong argument type
    #[test]
    fn test_error_wrong_type() {
        let mut sl = lib();
        let result = sl.call("sqrt", &[Value::String("x".into())]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StdLibError::TypeError { name, .. } => assert_eq!(name, "sqrt"),
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    // 12. Error: wrong argument count
    #[test]
    fn test_error_wrong_arity() {
        let mut sl = lib();
        let result = sl.call("abs", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StdLibError::ArityMismatch { name, expected, got } => {
                assert_eq!(name, "abs");
                assert_eq!(expected, "1");
                assert_eq!(got, 0);
            }
            other => panic!("expected ArityMismatch, got {other:?}"),
        }
    }

    // 13. Unknown function
    #[test]
    fn test_unknown_function() {
        let mut sl = lib();
        let result = sl.call("nonexistent", &[]);
        assert!(matches!(result, Err(StdLibError::UnknownFunction(_))));
    }

    // 14. Tensor: matmul
    #[test]
    fn test_matmul() {
        let mut sl = lib();
        // 2x2 identity * 2x2 matrix = same matrix
        let identity = Value::Tensor {
            data: vec![1.0, 0.0, 0.0, 1.0],
            shape: vec![2, 2],
        };
        let mat = Value::Tensor {
            data: vec![5.0, 6.0, 7.0, 8.0],
            shape: vec![2, 2],
        };
        let result = sl.call("matmul", &[identity, mat.clone()]).unwrap();
        assert_eq!(result, mat);
    }

    // 15. list_functions returns entries
    #[test]
    fn test_list_functions() {
        let fns = StdLib::list_functions();
        assert!(fns.len() >= 40);
        assert!(fns.iter().any(|(name, _)| *name == "abs"));
        assert!(fns.iter().any(|(name, _)| *name == "matmul"));
        assert!(fns.iter().any(|(name, _)| *name == "clock"));
    }
}
