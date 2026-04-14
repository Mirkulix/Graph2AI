//! QLANG Virtual Machine — Stack-based interpreter for general-purpose programming.
//!
//! This module turns QLANG from a graph-only ML language into a complete
//! programming language with variables, arithmetic, conditionals, loops,
//! functions, arrays, and more.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Value type ─────────────────────────────────────────────────────────────

/// Runtime value in the QLANG VM.
#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    String(String),
    Array(Vec<f64>),
    Tensor(Vec<f64>, Vec<usize>),
    Dict(Vec<(String, Value)>),
    Null,
}

impl Value {
    pub fn as_number(&self) -> Result<f64, VmError> {
        match self {
            Value::Number(n) => Ok(*n),
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            other => Err(VmError::TypeError(format!("expected number, got {}", other.type_name()))),
        }
    }

    pub fn as_bool(&self) -> Result<bool, VmError> {
        match self {
            Value::Bool(b) => Ok(*b),
            Value::Number(n) => Ok(*n != 0.0),
            other => Err(VmError::TypeError(format!("expected bool, got {}", other.type_name()))),
        }
    }

    pub fn as_array(&self) -> Result<&Vec<f64>, VmError> {
        match self {
            Value::Array(a) => Ok(a),
            other => Err(VmError::TypeError(format!("expected array, got {}", other.type_name()))),
        }
    }

    fn type_name(&self) -> &'static str {
        self.type_name_static()
    }

    /// Public version of type_name for use by other modules.
    pub fn type_name_static(&self) -> &'static str {
        match self {
            Value::Number(_) => "number",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Tensor(_, _) => "tensor",
            Value::Dict(_) => "dict",
            Value::Null => "null",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Number(n) => {
                if *n == (*n as i64) as f64 && n.abs() < 1e15 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
            Value::Bool(b) => write!(f, "{b}"),
            Value::String(s) => write!(f, "{s}"),
            Value::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if *v == (*v as i64) as f64 && v.abs() < 1e15 {
                        write!(f, "{}", *v as i64)?;
                    } else {
                        write!(f, "{v}")?;
                    }
                }
                write!(f, "]")
            }
            Value::Tensor(data, shape) => {
                write!(f, "tensor([")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if *v == (*v as i64) as f64 && v.abs() < 1e15 {
                        write!(f, "{}", *v as i64)?;
                    } else {
                        write!(f, "{v}")?;
                    }
                }
                write!(f, "], shape={shape:?})")
            }
            Value::Dict(entries) => {
                write!(f, "{{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Null => write!(f, "null"),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Null, Value::Null) => true,
            _ => false,
        }
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum VmError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    TypeError(String),
    DivisionByZero,
    IndexOutOfBounds { index: isize, len: usize },
    ArityMismatch { expected: usize, got: usize },
    ParseError(String),
    RuntimeError(String),
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VmError::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            VmError::UndefinedFunction(name) => write!(f, "undefined function: {name}"),
            VmError::TypeError(msg) => write!(f, "type error: {msg}"),
            VmError::DivisionByZero => write!(f, "division by zero"),
            VmError::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for array of length {len}")
            }
            VmError::ArityMismatch { expected, got } => {
                write!(f, "expected {expected} arguments, got {got}")
            }
            VmError::ParseError(msg) => write!(f, "parse error: {msg}"),
            VmError::RuntimeError(msg) => write!(f, "runtime error: {msg}"),
        }
    }
}

impl std::error::Error for VmError {}

// ─── Tokens ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    NumberLit(f64),
    StringLit(String),
    Ident(String),
    BoolLit(bool),

    // Keywords
    Let,
    Fn,
    If,
    Else,
    While,
    For,
    In,
    Return,
    Print,
    And,
    Or,
    Not,
    Import,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,   // %
    StarStar,  // **
    Eq,        // =
    PlusEq,    // +=
    MinusEq,   // -=
    StarEq,    // *=
    SlashEq,   // /=
    PercentEq, // %=
    EqEq,      // ==
    BangEq,    // !=
    Bang,      // !
    Lt,        // <
    Gt,        // >
    LtEq,      // <=
    GtEq,      // >=
    Ampersand, // &
    Pipe,      // |
    Caret,     // ^
    Tilde,     // ~
    LtLt,      // <<
    GtGt,      // >>
    AmpAmp,    // &&
    PipePipe,  // ||

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    DotDot,    // ..
    Semicolon,
    Arrow,     // ->

    Eof,
}

// ─── Lexer ──────────────────────────────────────────────────────────────────

pub fn tokenize(source: &str) -> Result<Vec<Token>, VmError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        // Skip whitespace
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // Skip line comments
        if ch == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        // Numbers
        if ch.is_ascii_digit() || (ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            // Consume decimal point only if followed by a digit (not '..' range)
            if i < chars.len() && chars[i] == '.'
                && i + 1 < chars.len() && chars[i + 1] != '.' && chars[i + 1].is_ascii_digit()
            {
                i += 1; // consume '.'
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s: String = chars[start..i].iter().collect();
            let n: f64 = s.parse().map_err(|_| VmError::ParseError(format!("invalid number: {s}")))?;
            tokens.push(Token::NumberLit(n));
            continue;
        }

        // Strings
        if ch == '"' {
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            if i >= chars.len() {
                return Err(VmError::ParseError("unterminated string".into()));
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push(Token::StringLit(s));
            i += 1; // skip closing "
            continue;
        }

        // Identifiers and keywords
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let tok = match word.as_str() {
                "let" => Token::Let,
                "fn" => Token::Fn,
                "if" => Token::If,
                "else" => Token::Else,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "return" => Token::Return,
                "print" => Token::Print,
                "and" => Token::And,
                "or" => Token::Or,
                "not" => Token::Not,
                "import" => Token::Import,
                "true" => Token::BoolLit(true),
                "false" => Token::BoolLit(false),
                _ => Token::Ident(word),
            };
            tokens.push(tok);
            continue;
        }

        // Two-character operators
        if i + 1 < chars.len() {
            let two: String = chars[i..i + 2].iter().collect();
            match two.as_str() {
                "**" => { tokens.push(Token::StarStar); i += 2; continue; }
                "==" => { tokens.push(Token::EqEq); i += 2; continue; }
                "!=" => { tokens.push(Token::BangEq); i += 2; continue; }
                "+=" => { tokens.push(Token::PlusEq); i += 2; continue; }
                "->" => { tokens.push(Token::Arrow); i += 2; continue; }
                "-=" => { tokens.push(Token::MinusEq); i += 2; continue; }
                "*=" => { tokens.push(Token::StarEq); i += 2; continue; }
                "/=" => { tokens.push(Token::SlashEq); i += 2; continue; }
                "%=" => { tokens.push(Token::PercentEq); i += 2; continue; }
                "<=" => { tokens.push(Token::LtEq); i += 2; continue; }
                ">=" => { tokens.push(Token::GtEq); i += 2; continue; }
                "<<" => { tokens.push(Token::LtLt); i += 2; continue; }
                ">>" => { tokens.push(Token::GtGt); i += 2; continue; }
                "&&" => { tokens.push(Token::AmpAmp); i += 2; continue; }
                "||" => { tokens.push(Token::PipePipe); i += 2; continue; }
                ".." => { tokens.push(Token::DotDot); i += 2; continue; }
                _ => {}
            }
        }

        // Single-character tokens
        let tok = match ch {
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '!' => Token::Bang,
            '&' => Token::Ampersand,
            '|' => Token::Pipe,
            '^' => Token::Caret,
            '~' => Token::Tilde,
            '=' => Token::Eq,
            '<' => Token::Lt,
            '>' => Token::Gt,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '[' => Token::LBracket,
            ']' => Token::RBracket,
            ',' => Token::Comma,
            ':' => Token::Colon,
            ';' => Token::Semicolon,
            _ => return Err(VmError::ParseError(format!("unexpected character: '{ch}'"))),
        };
        tokens.push(tok);
        i += 1;
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

// ─── Static type annotations ────────────────────────────────────────────────

/// Optional static type annotation for variables and function signatures.
#[derive(Debug, Clone, PartialEq)]
pub enum QType {
    Int,
    Float,
    String,
    Bool,
    Array(Box<QType>),
    Dict(Box<QType>, Box<QType>),
    Tensor(std::string::String),
    Any,
    Void,
}

impl fmt::Display for QType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QType::Int => write!(f, "int"),
            QType::Float => write!(f, "float"),
            QType::String => write!(f, "string"),
            QType::Bool => write!(f, "bool"),
            QType::Array(inner) => write!(f, "[{}]", inner),
            QType::Dict(k, v) => write!(f, "{{{}: {}}}", k, v),
            QType::Tensor(s) => write!(f, "tensor({})", s),
            QType::Any => write!(f, "any"),
            QType::Void => write!(f, "void"),
        }
    }
}

/// A function parameter with an optional type annotation.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: std::string::String,
    pub type_ann: Option<QType>,
}

// ─── AST ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Expr {
    NumberLit(f64),
    BoolLit(bool),
    StringLit(String),
    ArrayLit(Vec<Expr>),
    Var(String),
    BinOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    UnaryOp { op: UnaryOp, operand: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
    Index { array: Box<Expr>, index: Box<Expr> },
    DictLit(Vec<(String, Expr)>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg, Not, BitNot,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { name: String, type_ann: Option<QType>, value: Expr },
    Assign { name: String, value: Expr },
    If { cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt> },
    While { cond: Expr, body: Vec<Stmt> },
    For { var: String, start: Expr, end: Expr, body: Vec<Stmt> },
    FnDef { name: String, params: Vec<Param>, return_type: Option<QType>, body: Vec<Stmt> },
    Return(Expr),
    Print(Expr),
    ExprStmt(Expr),
    Import(String),
}

// ─── Parser ─────────────────────────────────────────────────────────────────

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), VmError> {
        let tok = self.advance();
        if &tok == expected {
            Ok(())
        } else {
            Err(VmError::ParseError(format!("expected {expected:?}, got {tok:?}")))
        }
    }

    fn at(&self, expected: &Token) -> bool {
        self.peek() == expected
    }

    /// Parse a type annotation (e.g. `int`, `string`, `[float]`, `{string: int}`).
    fn parse_type(&mut self) -> Result<QType, VmError> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.advance();
                match name.as_str() {
                    "int" => Ok(QType::Int),
                    "float" => Ok(QType::Float),
                    "string" => Ok(QType::String),
                    "bool" => Ok(QType::Bool),
                    "any" => Ok(QType::Any),
                    "void" => Ok(QType::Void),
                    "tensor" => Ok(QType::Tensor(std::string::String::new())),
                    other => Err(VmError::ParseError(format!("unknown type: {other}"))),
                }
            }
            Token::LBracket => {
                self.advance(); // consume '['
                let inner = self.parse_type()?;
                self.expect(&Token::RBracket)?;
                Ok(QType::Array(Box::new(inner)))
            }
            Token::LBrace => {
                self.advance(); // consume '{'
                let key_type = self.parse_type()?;
                self.expect(&Token::Colon)?;
                let val_type = self.parse_type()?;
                self.expect(&Token::RBrace)?;
                Ok(QType::Dict(Box::new(key_type), Box::new(val_type)))
            }
            other => Err(VmError::ParseError(format!("expected type, got {other:?}"))),
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Stmt>, VmError> {
        let mut stmts = Vec::new();
        while !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, VmError> {
        match self.peek().clone() {
            Token::Let => self.parse_let(),
            Token::If => self.parse_if(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::Fn => self.parse_fn_def(),
            Token::Return => self.parse_return(),
            Token::Print => self.parse_print(),
            Token::Import => self.parse_import(),
            Token::Ident(_) => {
                // Could be assignment (x = ...), compound (x += ...) or expression (foo(...))
                if self.pos + 1 < self.tokens.len() {
                    let next = &self.tokens[self.pos + 1];
                    let is_assign = match next {
                        Token::Eq => {
                            // Check it's not == (comparison)
                            self.pos + 2 >= self.tokens.len() || self.tokens[self.pos + 2] != Token::Eq
                        }
                        Token::PlusEq | Token::MinusEq | Token::StarEq | Token::SlashEq | Token::PercentEq => true,
                        _ => false,
                    };
                    if is_assign {
                        return self.parse_assign();
                    }
                }
                let expr = self.parse_expr()?;
                Ok(Stmt::ExprStmt(expr))
            }
            _ => {
                let expr = self.parse_expr()?;
                Ok(Stmt::ExprStmt(expr))
            }
        }
    }

    fn parse_let(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'let'
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier after 'let', got {other:?}"))),
        };
        // Optional type annotation: let x: int = ...
        let type_ann = if self.at(&Token::Colon) {
            self.advance(); // consume ':'
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Let { name, type_ann, value })
    }

    fn parse_assign(&mut self) -> Result<Stmt, VmError> {
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier, got {other:?}"))),
        };
        // Check for compound assignment: +=, -=, *=, /=, %=
        let compound_op = match self.peek() {
            Token::PlusEq => Some(BinOp::Add),
            Token::MinusEq => Some(BinOp::Sub),
            Token::StarEq => Some(BinOp::Mul),
            Token::SlashEq => Some(BinOp::Div),
            Token::PercentEq => Some(BinOp::Mod),
            _ => None,
        };
        if let Some(op) = compound_op {
            self.advance(); // consume +=, -= etc.
            let rhs = self.parse_expr()?;
            // x += 5  becomes  x = x + 5
            let value = Expr::BinOp {
                op,
                left: Box::new(Expr::Var(name.clone())),
                right: Box::new(rhs),
            };
            return Ok(Stmt::Assign { name, value });
        }
        self.expect(&Token::Eq)?;
        let value = self.parse_expr()?;
        Ok(Stmt::Assign { name, value })
    }

    fn parse_if(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'if'
        let cond = self.parse_expr()?;
        let then_body = self.parse_block()?;
        let else_body = if self.at(&Token::Else) {
            self.advance();
            if self.at(&Token::If) {
                vec![self.parse_if()?]
            } else {
                self.parse_block()?
            }
        } else {
            vec![]
        };
        Ok(Stmt::If { cond, then_body, else_body })
    }

    fn parse_while(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'while'
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'for'
        let var = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected identifier after 'for', got {other:?}"))),
        };
        self.expect(&Token::In)?;
        let start = self.parse_unary()?;
        self.expect(&Token::DotDot)?;
        let end = self.parse_unary()?;
        let body = self.parse_block()?;
        Ok(Stmt::For { var, start, end, body })
    }

    fn parse_fn_def(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'fn'
        let name = match self.advance() {
            Token::Ident(n) => n,
            other => return Err(VmError::ParseError(format!("expected function name, got {other:?}"))),
        };
        self.expect(&Token::LParen)?;
        let mut params: Vec<Param> = Vec::new();
        while !self.at(&Token::RParen) {
            if !params.is_empty() {
                self.expect(&Token::Comma)?;
            }
            let pname = match self.advance() {
                Token::Ident(p) => p,
                other => return Err(VmError::ParseError(format!("expected parameter name, got {other:?}"))),
            };
            // Optional type annotation: fn foo(a: int, b: float)
            let type_ann = if self.at(&Token::Colon) {
                self.advance(); // consume ':'
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push(Param { name: pname, type_ann });
        }
        self.expect(&Token::RParen)?;
        // Optional return type: fn foo(a, b) -> int { ... }
        let return_type = if self.at(&Token::Arrow) {
            self.advance(); // consume '->'
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(Stmt::FnDef { name, params, return_type, body })
    }

    fn parse_return(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'return'
        let value = self.parse_expr()?;
        Ok(Stmt::Return(value))
    }

    fn parse_print(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'print'
        self.expect(&Token::LParen)?;
        let value = self.parse_expr()?;
        self.expect(&Token::RParen)?;
        Ok(Stmt::Print(value))
    }

    fn parse_import(&mut self) -> Result<Stmt, VmError> {
        self.advance(); // consume 'import'
        match self.advance() {
            Token::StringLit(path) => Ok(Stmt::Import(path)),
            other => Err(VmError::ParseError(format!("expected string after 'import', got {other:?}"))),
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, VmError> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    // Expression parsing with precedence climbing

    fn parse_expr(&mut self) -> Result<Expr, VmError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_and()?;
        while self.at(&Token::Or) || self.at(&Token::PipePipe) {
            self.advance();
            let right = self.parse_and()?;
            left = Expr::BinOp { op: BinOp::Or, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_bitor()?;
        while self.at(&Token::And) || self.at(&Token::AmpAmp) {
            self.advance();
            let right = self.parse_bitor()?;
            left = Expr::BinOp { op: BinOp::And, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_bitxor()?;
        while self.at(&Token::Pipe) {
            self.advance();
            let right = self.parse_bitxor()?;
            left = Expr::BinOp { op: BinOp::BitOr, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_bitand()?;
        while self.at(&Token::Caret) {
            self.advance();
            let right = self.parse_bitand()?;
            left = Expr::BinOp { op: BinOp::BitXor, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_equality()?;
        while self.at(&Token::Ampersand) {
            self.advance();
            let right = self.parse_equality()?;
            left = Expr::BinOp { op: BinOp::BitAnd, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_comparison()?;
        loop {
            let op = match self.peek() {
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Ne,
                _ => break,
            };
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_shift()?;
        loop {
            let op = match self.peek() {
                Token::Lt => BinOp::Lt,
                Token::Gt => BinOp::Gt,
                Token::LtEq => BinOp::Le,
                Token::GtEq => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let right = self.parse_shift()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::LtLt => BinOp::Shl,
                Token::GtGt => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_multiplicative()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, VmError> {
        let mut left = self.parse_power()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_power()?;
            left = Expr::BinOp { op, left: Box::new(left), right: Box::new(right) };
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Expr, VmError> {
        let base = self.parse_unary()?;
        if self.at(&Token::StarStar) {
            self.advance();
            let exp = self.parse_power()?; // right-associative
            Ok(Expr::BinOp { op: BinOp::Pow, left: Box::new(base), right: Box::new(exp) })
        } else {
            Ok(base)
        }
    }

    fn parse_unary(&mut self) -> Result<Expr, VmError> {
        match self.peek().clone() {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp { op: UnaryOp::Neg, operand: Box::new(operand) })
            }
            Token::Not | Token::Bang => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp { op: UnaryOp::Not, operand: Box::new(operand) })
            }
            Token::Tilde => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::UnaryOp { op: UnaryOp::BitNot, operand: Box::new(operand) })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, VmError> {
        let mut expr = self.parse_primary()?;
        // Handle indexing: expr[index]
        while self.at(&Token::LBracket) {
            self.advance();
            let index = self.parse_expr()?;
            self.expect(&Token::RBracket)?;
            expr = Expr::Index { array: Box::new(expr), index: Box::new(index) };
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, VmError> {
        match self.peek().clone() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(Expr::NumberLit(n))
            }
            Token::BoolLit(b) => {
                self.advance();
                Ok(Expr::BoolLit(b))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Expr::StringLit(s))
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !self.at(&Token::RBracket) && !self.at(&Token::Eof) {
                    if !elems.is_empty() {
                        self.expect(&Token::Comma)?;
                    }
                    elems.push(self.parse_expr()?);
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::ArrayLit(elems))
            }
            Token::LBrace => {
                // Dict literal: {"key": value, ...}
                self.advance();
                let mut entries = Vec::new();
                while !self.at(&Token::RBrace) && !self.at(&Token::Eof) {
                    if !entries.is_empty() {
                        self.expect(&Token::Comma)?;
                    }
                    // Key must be a string literal
                    let key = match self.advance() {
                        Token::StringLit(s) => s,
                        other => return Err(VmError::ParseError(format!("expected string key in dict, got {other:?}"))),
                    };
                    self.expect(&Token::Colon)?;
                    let value = self.parse_expr()?;
                    entries.push((key, value));
                }
                self.expect(&Token::RBrace)?;
                Ok(Expr::DictLit(entries))
            }
            Token::Ident(name) => {
                self.advance();
                // Function call?
                if self.at(&Token::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !self.at(&Token::RParen) && !self.at(&Token::Eof) {
                        if !args.is_empty() {
                            self.expect(&Token::Comma)?;
                        }
                        args.push(self.parse_expr()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Expr::Call { name, args })
                } else {
                    Ok(Expr::Var(name))
                }
            }
            other => Err(VmError::ParseError(format!("unexpected token: {other:?}"))),
        }
    }
}

pub fn parse_program(tokens: &[Token]) -> Result<Vec<Stmt>, VmError> {
    let mut parser = Parser::new(tokens.to_vec());
    parser.parse_program()
}

// ─── VM State ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FnDef {
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub return_type: Option<QType>,
}

/// Signal used to propagate return values up the call stack.
pub enum ExecSignal {
    None,
    Return(Value),
}

#[derive(Clone)]
pub struct VmState {
    /// Stack of variable scopes (innermost last).
    pub scopes: Vec<HashMap<String, Value>>,
    /// User-defined functions.
    pub functions: HashMap<String, FnDef>,
    /// Captured print output (for testing).
    pub output: Vec<String>,
}

impl VmState {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            output: Vec::new(),
        }
    }

    pub fn get_var(&self, name: &str) -> Result<Value, VmError> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Ok(val.clone());
            }
        }
        Err(VmError::UndefinedVariable(name.to_string()))
    }

    pub fn set_var(&mut self, name: &str, value: Value) {
        // Set in the innermost scope that already has it, or the current scope
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return;
            }
        }
        // New variable in current scope
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    pub fn declare_var(&mut self, name: &str, value: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), value);
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn exec_stmts(&mut self, stmts: &[Stmt]) -> Result<ExecSignal, VmError> {
        for stmt in stmts {
            match self.exec_stmt(stmt)? {
                ExecSignal::Return(v) => return Ok(ExecSignal::Return(v)),
                ExecSignal::None => {}
            }
        }
        Ok(ExecSignal::None)
    }

    pub fn exec_stmt(&mut self, stmt: &Stmt) -> Result<ExecSignal, VmError> {
        match stmt {
            Stmt::Let { name, type_ann: _, value } => {
                let val = self.eval_expr(value)?;
                self.declare_var(name, val);
                Ok(ExecSignal::None)
            }
            Stmt::Assign { name, value } => {
                // Verify variable exists somewhere
                let _ = self.get_var(name)?;
                let val = self.eval_expr(value)?;
                self.set_var(name, val);
                Ok(ExecSignal::None)
            }
            Stmt::If { cond, then_body, else_body } => {
                let c = self.eval_expr(cond)?;
                if c.as_bool()? {
                    self.push_scope();
                    let sig = self.exec_stmts(then_body)?;
                    self.pop_scope();
                    Ok(sig)
                } else {
                    self.push_scope();
                    let sig = self.exec_stmts(else_body)?;
                    self.pop_scope();
                    Ok(sig)
                }
            }
            Stmt::While { cond, body } => {
                let mut iterations = 0u64;
                loop {
                    let c = self.eval_expr(cond)?;
                    if !c.as_bool()? {
                        break;
                    }
                    self.push_scope();
                    match self.exec_stmts(body)? {
                        ExecSignal::Return(v) => {
                            self.pop_scope();
                            return Ok(ExecSignal::Return(v));
                        }
                        ExecSignal::None => {}
                    }
                    self.pop_scope();
                    iterations += 1;
                    if iterations > 1_000_000 {
                        return Err(VmError::RuntimeError("loop exceeded 1000000 iterations".into()));
                    }
                }
                Ok(ExecSignal::None)
            }
            Stmt::For { var, start, end, body } => {
                let s = self.eval_expr(start)?.as_number()? as i64;
                let e = self.eval_expr(end)?.as_number()? as i64;
                for i in s..e {
                    self.push_scope();
                    self.declare_var(var, Value::Number(i as f64));
                    match self.exec_stmts(body)? {
                        ExecSignal::Return(v) => {
                            self.pop_scope();
                            return Ok(ExecSignal::Return(v));
                        }
                        ExecSignal::None => {}
                    }
                    self.pop_scope();
                }
                Ok(ExecSignal::None)
            }
            Stmt::FnDef { name, params, return_type, body } => {
                self.functions.insert(name.clone(), FnDef {
                    params: params.clone(),
                    body: body.clone(),
                    return_type: return_type.clone(),
                });
                Ok(ExecSignal::None)
            }
            Stmt::Return(expr) => {
                let val = self.eval_expr(expr)?;
                Ok(ExecSignal::Return(val))
            }
            Stmt::Print(expr) => {
                let val = self.eval_expr(expr)?;
                let s = format!("{val}");
                self.output.push(s);
                Ok(ExecSignal::None)
            }
            Stmt::ExprStmt(expr) => {
                self.eval_expr(expr)?;
                Ok(ExecSignal::None)
            }
            Stmt::Import(path) => {
                let source = std::fs::read_to_string(path).map_err(|e| {
                    VmError::RuntimeError(format!("cannot import '{}': {}", path, e))
                })?;
                let tokens = tokenize(&source)?;
                let stmts = parse_program(&tokens)?;
                self.exec_stmts(&stmts)?;
                Ok(ExecSignal::None)
            }
        }
    }

    pub fn eval_expr(&mut self, expr: &Expr) -> Result<Value, VmError> {
        match expr {
            Expr::NumberLit(n) => Ok(Value::Number(*n)),
            Expr::BoolLit(b) => Ok(Value::Bool(*b)),
            Expr::StringLit(s) => Ok(Value::String(s.clone())),
            Expr::ArrayLit(elems) => {
                let mut vals = Vec::with_capacity(elems.len());
                for e in elems {
                    vals.push(self.eval_expr(e)?.as_number()?);
                }
                Ok(Value::Array(vals))
            }
            Expr::Var(name) => self.get_var(name),
            Expr::BinOp { op, left, right } => {
                let lv = self.eval_expr(left)?;
                // Short-circuit for logical ops
                match op {
                    BinOp::And => {
                        if !lv.as_bool()? {
                            return Ok(Value::Bool(false));
                        }
                        let rv = self.eval_expr(right)?;
                        return Ok(Value::Bool(rv.as_bool()?));
                    }
                    BinOp::Or => {
                        if lv.as_bool()? {
                            return Ok(Value::Bool(true));
                        }
                        let rv = self.eval_expr(right)?;
                        return Ok(Value::Bool(rv.as_bool()?));
                    }
                    _ => {}
                }
                let rv = self.eval_expr(right)?;
                // String concatenation with +
                if let BinOp::Add = op {
                    if matches!(&lv, Value::String(_)) || matches!(&rv, Value::String(_)) {
                        let ls = format!("{}", lv);
                        let rs = format!("{}", rv);
                        return Ok(Value::String(format!("{}{}", ls, rs)));
                    }
                }
                // Equality/inequality works on all types
                if let BinOp::Eq = op {
                    return Ok(Value::Bool(lv == rv));
                }
                if let BinOp::Ne = op {
                    return Ok(Value::Bool(lv != rv));
                }
                let a = lv.as_number()?;
                let b = rv.as_number()?;
                match op {
                    BinOp::Add => Ok(Value::Number(a + b)),
                    BinOp::Sub => Ok(Value::Number(a - b)),
                    BinOp::Mul => Ok(Value::Number(a * b)),
                    BinOp::Div => {
                        if b == 0.0 {
                            return Err(VmError::DivisionByZero);
                        }
                        Ok(Value::Number(a / b))
                    }
                    BinOp::Mod => {
                        if b == 0.0 {
                            return Err(VmError::DivisionByZero);
                        }
                        Ok(Value::Number(a % b))
                    }
                    BinOp::Pow => Ok(Value::Number(a.powf(b))),
                    BinOp::Eq | BinOp::Ne => unreachable!(),
                    BinOp::Lt => Ok(Value::Bool(a < b)),
                    BinOp::Gt => Ok(Value::Bool(a > b)),
                    BinOp::Le => Ok(Value::Bool(a <= b)),
                    BinOp::Ge => Ok(Value::Bool(a >= b)),
                    BinOp::And | BinOp::Or => unreachable!(),
                    BinOp::BitAnd => Ok(Value::Number(((a as i64) & (b as i64)) as f64)),
                    BinOp::BitOr => Ok(Value::Number(((a as i64) | (b as i64)) as f64)),
                    BinOp::BitXor => Ok(Value::Number(((a as i64) ^ (b as i64)) as f64)),
                    BinOp::Shl => Ok(Value::Number(((a as i64) << (b as u32)) as f64)),
                    BinOp::Shr => Ok(Value::Number(((a as i64) >> (b as u32)) as f64)),
                }
            }
            Expr::UnaryOp { op, operand } => {
                let v = self.eval_expr(operand)?;
                match op {
                    UnaryOp::Neg => Ok(Value::Number(-v.as_number()?)),
                    UnaryOp::Not => Ok(Value::Bool(!v.as_bool()?)),
                    UnaryOp::BitNot => Ok(Value::Number(!(v.as_number()? as i64) as f64)),
                }
            }
            Expr::Call { name, args } => {
                let evaluated_args: Vec<Value> = args
                    .iter()
                    .map(|a| self.eval_expr(a))
                    .collect::<Result<_, _>>()?;

                // Built-in functions
                match name.as_str() {
                    "len" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return match &evaluated_args[0] {
                            Value::Array(a) => Ok(Value::Number(a.len() as f64)),
                            Value::String(s) => Ok(Value::Number(s.len() as f64)),
                            Value::Dict(entries) => Ok(Value::Number(entries.len() as f64)),
                            other => Err(VmError::TypeError(format!("len() expects string, array, or dict, got {}", other.type_name()))),
                        };
                    }
                    "type" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::String(evaluated_args[0].type_name_static().to_string()));
                    }
                    "str" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::String(format!("{}", evaluated_args[0])));
                    }
                    "int" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.floor()));
                    }
                    "sqrt" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.sqrt()));
                    }
                    "abs" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.abs()));
                    }
                    "floor" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let n = evaluated_args[0].as_number()?;
                        return Ok(Value::Number(n.floor()));
                    }
                    "min" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let a = evaluated_args[0].as_number()?;
                        let b = evaluated_args[1].as_number()?;
                        return Ok(Value::Number(a.min(b)));
                    }
                    "max" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let a = evaluated_args[0].as_number()?;
                        let b = evaluated_args[1].as_number()?;
                        return Ok(Value::Number(a.max(b)));
                    }
                    // ── Math functions (new) ──────────────────────
                    "ceil" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.ceil()));
                    }
                    "round" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.round()));
                    }
                    "sin" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.sin()));
                    }
                    "cos" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.cos()));
                    }
                    "tan" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.tan()));
                    }
                    "log" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.ln()));
                    }
                    "log2" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.log2()));
                    }
                    "log10" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.log10()));
                    }
                    "exp" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(evaluated_args[0].as_number()?.exp()));
                    }
                    "pow" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let a = evaluated_args[0].as_number()?;
                        let b = evaluated_args[1].as_number()?;
                        return Ok(Value::Number(a.powf(b)));
                    }
                    "pi" => {
                        if !evaluated_args.is_empty() {
                            return Err(VmError::ArityMismatch { expected: 0, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(std::f64::consts::PI));
                    }
                    "e" => {
                        if !evaluated_args.is_empty() {
                            return Err(VmError::ArityMismatch { expected: 0, got: evaluated_args.len() });
                        }
                        return Ok(Value::Number(std::f64::consts::E));
                    }
                    "clamp" => {
                        if evaluated_args.len() != 3 {
                            return Err(VmError::ArityMismatch { expected: 3, got: evaluated_args.len() });
                        }
                        let x = evaluated_args[0].as_number()?;
                        let lo = evaluated_args[1].as_number()?;
                        let hi = evaluated_args[2].as_number()?;
                        return Ok(Value::Number(x.max(lo).min(hi)));
                    }

                    // ── String functions ──────────────────────────────
                    "upper" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::String(s) => return Ok(Value::String(s.to_uppercase())),
                            other => return Err(VmError::TypeError(format!("upper() expects string, got {}", other.type_name()))),
                        }
                    }
                    "lower" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::String(s) => return Ok(Value::String(s.to_lowercase())),
                            other => return Err(VmError::TypeError(format!("lower() expects string, got {}", other.type_name()))),
                        }
                    }
                    "trim" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::String(s) => return Ok(Value::String(s.trim().to_string())),
                            other => return Err(VmError::TypeError(format!("trim() expects string, got {}", other.type_name()))),
                        }
                    }
                    "split" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("split() expects string, got {}", other.type_name()))),
                        };
                        let sep = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("split() expects string separator, got {}", other.type_name()))),
                        };
                        let parts: Vec<(String, Value)> = s.split(&sep)
                            .enumerate()
                            .map(|(i, part)| (i.to_string(), Value::String(part.to_string())))
                            .collect();
                        return Ok(Value::Dict(parts));
                    }
                    "join" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let sep = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("join() expects string separator, got {}", other.type_name()))),
                        };
                        match &evaluated_args[0] {
                            Value::Array(arr) => {
                                let strs: Vec<String> = arr.iter().map(|n| {
                                    if *n == (*n as i64) as f64 && n.abs() < 1e15 {
                                        format!("{}", *n as i64)
                                    } else {
                                        format!("{n}")
                                    }
                                }).collect();
                                return Ok(Value::String(strs.join(&sep)));
                            }
                            Value::Dict(entries) => {
                                let strs: Vec<String> = entries.iter().map(|(_, v)| format!("{v}")).collect();
                                return Ok(Value::String(strs.join(&sep)));
                            }
                            other => return Err(VmError::TypeError(format!("join() expects array or dict, got {}", other.type_name()))),
                        }
                    }
                    "contains" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("contains() expects string, got {}", other.type_name()))),
                        };
                        let sub = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("contains() expects string substring, got {}", other.type_name()))),
                        };
                        return Ok(Value::Bool(s.contains(&sub)));
                    }
                    "starts_with" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("starts_with() expects string, got {}", other.type_name()))),
                        };
                        let prefix = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("starts_with() expects string prefix, got {}", other.type_name()))),
                        };
                        return Ok(Value::Bool(s.starts_with(&prefix)));
                    }
                    "ends_with" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("ends_with() expects string, got {}", other.type_name()))),
                        };
                        let suffix = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("ends_with() expects string suffix, got {}", other.type_name()))),
                        };
                        return Ok(Value::Bool(s.ends_with(&suffix)));
                    }
                    "replace" => {
                        if evaluated_args.len() != 3 {
                            return Err(VmError::ArityMismatch { expected: 3, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("replace() expects string, got {}", other.type_name()))),
                        };
                        let old = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("replace() expects string old, got {}", other.type_name()))),
                        };
                        let new_s = match &evaluated_args[2] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("replace() expects string new, got {}", other.type_name()))),
                        };
                        return Ok(Value::String(s.replace(&old, &new_s)));
                    }
                    "substr" => {
                        if evaluated_args.len() != 3 {
                            return Err(VmError::ArityMismatch { expected: 3, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("substr() expects string, got {}", other.type_name()))),
                        };
                        let start = evaluated_args[1].as_number()? as usize;
                        let length = evaluated_args[2].as_number()? as usize;
                        let chars: Vec<char> = s.chars().collect();
                        let end = (start + length).min(chars.len());
                        let start_clamped = start.min(chars.len());
                        let result: String = chars[start_clamped..end].iter().collect();
                        return Ok(Value::String(result));
                    }
                    "char_at" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("char_at() expects string, got {}", other.type_name()))),
                        };
                        let idx_num = evaluated_args[1].as_number()?;
                        let idx = idx_num as usize;
                        let chars: Vec<char> = s.chars().collect();
                        if idx >= chars.len() {
                            return Err(VmError::IndexOutOfBounds { index: idx_num as isize, len: chars.len() });
                        }
                        return Ok(Value::String(chars[idx].to_string()));
                    }
                    "parse_int" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::String(s) => match s.trim().parse::<i64>() {
                                Ok(n) => return Ok(Value::Number(n as f64)),
                                Err(e) => return Err(VmError::RuntimeError(format!("parse_int: cannot parse '{}': {}", s, e))),
                            },
                            Value::Number(n) => return Ok(Value::Number((*n as i64) as f64)),
                            other => return Err(VmError::TypeError(format!("parse_int() expects string, got {}", other.type_name()))),
                        }
                    }
                    "parse_float" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::String(s) => match s.trim().parse::<f64>() {
                                Ok(n) => return Ok(Value::Number(n)),
                                Err(e) => return Err(VmError::RuntimeError(format!("parse_float: cannot parse '{}': {}", s, e))),
                            },
                            Value::Number(n) => return Ok(Value::Number(*n)),
                            other => return Err(VmError::TypeError(format!("parse_float() expects string, got {}", other.type_name()))),
                        }
                    }
                    "format" => {
                        if evaluated_args.is_empty() {
                            return Err(VmError::ArityMismatch { expected: 1, got: 0 });
                        }
                        let template = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("format() expects string template, got {}", other.type_name()))),
                        };
                        let mut result = template;
                        for (i, arg) in evaluated_args[1..].iter().enumerate() {
                            let placeholder = format!("{{{}}}", i);
                            result = result.replace(&placeholder, &format!("{arg}"));
                        }
                        for arg in &evaluated_args[1..] {
                            if let Some(pos) = result.find("{}") {
                                let (before, after) = result.split_at(pos);
                                result = format!("{}{}{}", before, arg, &after[2..]);
                            }
                        }
                        return Ok(Value::String(result));
                    }

                    // ── Array functions ───────────────────────────────
                    "push" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let mut arr = evaluated_args[0].as_array()?.clone();
                        let val = evaluated_args[1].as_number()?;
                        arr.push(val);
                        return Ok(Value::Array(arr));
                    }
                    "pop" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let mut arr = evaluated_args[0].as_array()?.clone();
                        if arr.is_empty() {
                            return Err(VmError::RuntimeError("pop() on empty array".into()));
                        }
                        let last = arr.pop().unwrap();
                        return Ok(Value::Dict(vec![
                            ("array".to_string(), Value::Array(arr)),
                            ("value".to_string(), Value::Number(last)),
                        ]));
                    }
                    "reverse" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let mut arr = evaluated_args[0].as_array()?.clone();
                        arr.reverse();
                        return Ok(Value::Array(arr));
                    }
                    "sort" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let mut arr = evaluated_args[0].as_array()?.clone();
                        arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        return Ok(Value::Array(arr));
                    }
                    "range" => {
                        if evaluated_args.len() < 2 || evaluated_args.len() > 3 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let start = evaluated_args[0].as_number()?;
                        let end = evaluated_args[1].as_number()?;
                        let step = if evaluated_args.len() == 3 {
                            evaluated_args[2].as_number()?
                        } else {
                            1.0
                        };
                        if step == 0.0 {
                            return Err(VmError::RuntimeError("range() step cannot be zero".into()));
                        }
                        let mut arr = Vec::new();
                        let mut current = start;
                        if step > 0.0 {
                            while current < end {
                                arr.push(current);
                                current += step;
                            }
                        } else {
                            while current > end {
                                arr.push(current);
                                current += step;
                            }
                        }
                        return Ok(Value::Array(arr));
                    }
                    "sum" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?;
                        let total: f64 = arr.iter().sum();
                        return Ok(Value::Number(total));
                    }
                    "avg" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?;
                        if arr.is_empty() {
                            return Err(VmError::RuntimeError("avg() on empty array".into()));
                        }
                        let total: f64 = arr.iter().sum();
                        return Ok(Value::Number(total / arr.len() as f64));
                    }
                    "flatten" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?.clone();
                        return Ok(Value::Array(arr));
                    }
                    "zip" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let a = evaluated_args[0].as_array()?;
                        let b = evaluated_args[1].as_array()?;
                        let pairs: Vec<(String, Value)> = a.iter().zip(b.iter())
                            .enumerate()
                            .map(|(i, (x, y))| {
                                (i.to_string(), Value::Array(vec![*x, *y]))
                            })
                            .collect();
                        return Ok(Value::Dict(pairs));
                    }
                    "enumerate" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?;
                        let pairs: Vec<(String, Value)> = arr.iter()
                            .enumerate()
                            .map(|(i, v)| {
                                (i.to_string(), Value::Array(vec![i as f64, *v]))
                            })
                            .collect();
                        return Ok(Value::Dict(pairs));
                    }
                    "slice" => {
                        if evaluated_args.len() != 3 {
                            return Err(VmError::ArityMismatch { expected: 3, got: evaluated_args.len() });
                        }
                        let arr = evaluated_args[0].as_array()?;
                        let start = evaluated_args[1].as_number()? as usize;
                        let end = evaluated_args[2].as_number()? as usize;
                        let start_clamped = start.min(arr.len());
                        let end_clamped = end.min(arr.len());
                        if start_clamped > end_clamped {
                            return Ok(Value::Array(vec![]));
                        }
                        return Ok(Value::Array(arr[start_clamped..end_clamped].to_vec()));
                    }

                    // ── I/O functions ─────────────────────────────────
                    "read_file" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let path = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("read_file() expects string path, got {}", other.type_name()))),
                        };
                        match std::fs::read_to_string(&path) {
                            Ok(content) => return Ok(Value::String(content)),
                            Err(err) => return Err(VmError::RuntimeError(format!("read_file: {}", err))),
                        }
                    }
                    "write_file" => {
                        if evaluated_args.len() != 2 {
                            return Err(VmError::ArityMismatch { expected: 2, got: evaluated_args.len() });
                        }
                        let path = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("write_file() expects string path, got {}", other.type_name()))),
                        };
                        let content = match &evaluated_args[1] {
                            Value::String(s) => s.clone(),
                            other => format!("{other}"),
                        };
                        match std::fs::write(&path, &content) {
                            Ok(_) => return Ok(Value::Null),
                            Err(err) => return Err(VmError::RuntimeError(format!("write_file: {}", err))),
                        }
                    }
                    "file_exists" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let path = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("file_exists() expects string path, got {}", other.type_name()))),
                        };
                        return Ok(Value::Bool(std::path::Path::new(&path).exists()));
                    }
                    "read_lines" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let path = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("read_lines() expects string path, got {}", other.type_name()))),
                        };
                        match std::fs::read_to_string(&path) {
                            Ok(content) => {
                                let lines: Vec<(String, Value)> = content.lines()
                                    .enumerate()
                                    .map(|(i, line)| (i.to_string(), Value::String(line.to_string())))
                                    .collect();
                                return Ok(Value::Dict(lines));
                            }
                            Err(err) => return Err(VmError::RuntimeError(format!("read_lines: {}", err))),
                        }
                    }

                    // ── System functions ──────────────────────────────
                    "time" => {
                        if !evaluated_args.is_empty() {
                            return Err(VmError::ArityMismatch { expected: 0, got: evaluated_args.len() });
                        }
                        let secs = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs_f64();
                        return Ok(Value::Number(secs));
                    }
                    "sleep" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let ms = evaluated_args[0].as_number()?;
                        std::thread::sleep(std::time::Duration::from_millis(ms as u64));
                        return Ok(Value::Null);
                    }
                    "exit" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let code = evaluated_args[0].as_number()? as i32;
                        std::process::exit(code);
                    }
                    "env" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let var_name = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("env() expects string, got {}", other.type_name()))),
                        };
                        match std::env::var(&var_name) {
                            Ok(val) => return Ok(Value::String(val)),
                            Err(_) => return Ok(Value::Null),
                        }
                    }
                    "args" => {
                        if !evaluated_args.is_empty() {
                            return Err(VmError::ArityMismatch { expected: 0, got: evaluated_args.len() });
                        }
                        let argv: Vec<(String, Value)> = std::env::args()
                            .enumerate()
                            .map(|(i, a)| (i.to_string(), Value::String(a)))
                            .collect();
                        return Ok(Value::Dict(argv));
                    }

                    // ── Type functions ────────────────────────────────
                    "float" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        match &evaluated_args[0] {
                            Value::Number(n) => return Ok(Value::Number(*n)),
                            Value::Bool(b) => return Ok(Value::Number(if *b { 1.0 } else { 0.0 })),
                            Value::String(s) => match s.trim().parse::<f64>() {
                                Ok(n) => return Ok(Value::Number(n)),
                                Err(err) => return Err(VmError::RuntimeError(format!("float(): cannot convert '{}': {}", s, err))),
                            },
                            other => return Err(VmError::TypeError(format!("float() cannot convert {}", other.type_name()))),
                        }
                    }
                    "bool" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let result = match &evaluated_args[0] {
                            Value::Bool(b) => *b,
                            Value::Number(n) => *n != 0.0,
                            Value::String(s) => !s.is_empty(),
                            Value::Array(a) => !a.is_empty(),
                            Value::Null => false,
                            _ => true,
                        };
                        return Ok(Value::Bool(result));
                    }
                    "is_int" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let result = match &evaluated_args[0] {
                            Value::Number(n) => *n == (*n as i64) as f64,
                            _ => false,
                        };
                        return Ok(Value::Bool(result));
                    }
                    "is_float" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Bool(matches!(&evaluated_args[0], Value::Number(_))));
                    }
                    "is_string" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Bool(matches!(&evaluated_args[0], Value::String(_))));
                    }
                    "is_array" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Bool(matches!(&evaluated_args[0], Value::Array(_))));
                    }
                    "is_null" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        return Ok(Value::Bool(matches!(&evaluated_args[0], Value::Null)));
                    }

                    // ── JSON functions ────────────────────────────────
                    "json_parse" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let s = match &evaluated_args[0] {
                            Value::String(s) => s.clone(),
                            other => return Err(VmError::TypeError(format!("json_parse() expects string, got {}", other.type_name()))),
                        };
                        match serde_json::from_str::<serde_json::Value>(&s) {
                            Ok(json) => return Ok(json_to_value(json)),
                            Err(err) => return Err(VmError::RuntimeError(format!("json_parse: {}", err))),
                        }
                    }
                    "json_stringify" => {
                        if evaluated_args.len() != 1 {
                            return Err(VmError::ArityMismatch { expected: 1, got: evaluated_args.len() });
                        }
                        let json = value_to_json(&evaluated_args[0]);
                        return Ok(Value::String(json.to_string()));
                    }

                    _ => {}
                }

                // User-defined functions (checked before graph ops so users can override)
                if let Some(fndef) = self.functions.get(name).cloned() {
                    if evaluated_args.len() != fndef.params.len() {
                        return Err(VmError::ArityMismatch {
                            expected: fndef.params.len(),
                            got: evaluated_args.len(),
                        });
                    }

                    self.push_scope();
                    for (param, arg) in fndef.params.iter().zip(evaluated_args.iter()) {
                        self.declare_var(&param.name, arg.clone());
                    }
                    let signal = self.exec_stmts(&fndef.body)?;
                    self.pop_scope();

                    return match signal {
                        ExecSignal::Return(v) => Ok(v),
                        ExecSignal::None => Ok(Value::Null),
                    };
                }

                // Graph operations (matmul, relu, add, etc.)
                if let Some(result) = crate::graph_ops::try_call_graph_op(name, &evaluated_args)? {
                    return Ok(result);
                }

                Err(VmError::UndefinedFunction(name.clone()))
            }
            Expr::Index { array, index } => {
                let arr_val = self.eval_expr(array)?;
                let idx_val = self.eval_expr(index)?;
                // Handle dict indexing with string keys first (before as_number)
                if let Value::Dict(entries) = &arr_val {
                    let key = match &idx_val {
                        Value::String(s) => s.clone(),
                        other => return Err(VmError::TypeError(format!("dict key must be string, got {}", other.type_name()))),
                    };
                    for (k, v) in entries {
                        if k == &key {
                            return Ok(v.clone());
                        }
                    }
                    return Err(VmError::RuntimeError(format!("key '{}' not found in dict", key)));
                }
                let idx_num = idx_val.as_number()?;
                if idx_num < 0.0 {
                    return Err(VmError::IndexOutOfBounds {
                        index: idx_num as isize,
                        len: 0
                    });
                }
                let idx = idx_num as usize;
                match &arr_val {
                    Value::Array(arr) => {
                        if idx >= arr.len() {
                            return Err(VmError::IndexOutOfBounds { index: idx_num as isize, len: arr.len() });
                        }
                        Ok(Value::Number(arr[idx]))
                    }
                    Value::String(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        if idx >= chars.len() {
                            return Err(VmError::IndexOutOfBounds { index: idx_num as isize, len: chars.len() });
                        }
                        Ok(Value::String(chars[idx].to_string()))
                    }
                    other => Err(VmError::TypeError(format!("cannot index into {}", other.type_name()))),
                }
            }
            Expr::DictLit(entries) => {
                let mut result = Vec::with_capacity(entries.len());
                for (key, val_expr) in entries {
                    let val = self.eval_expr(val_expr)?;
                    result.push((key.clone(), val));
                }
                Ok(Value::Dict(result))
            }
        }
    }
}

// ─── JSON conversion helpers ────────────────────────────────────────────────

fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => Value::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            // Try to convert to Vec<f64> if all elements are numbers
            let all_numbers = arr.iter().all(|v| v.is_number());
            if all_numbers && !arr.is_empty() {
                let nums: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                Value::Array(nums)
            } else {
                let entries: Vec<(String, Value)> = arr
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), json_to_value(v)))
                    .collect();
                Value::Dict(entries)
            }
        }
        serde_json::Value::Object(map) => {
            let entries: Vec<(String, Value)> = map
                .into_iter()
                .map(|(k, v)| (k, json_to_value(v)))
                .collect();
            Value::Dict(entries)
        }
    }
}

fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Number(n) => serde_json::json!(*n),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(arr) => {
            let json_arr: Vec<serde_json::Value> = arr.iter().map(|n| serde_json::json!(*n)).collect();
            serde_json::Value::Array(json_arr)
        }
        Value::Tensor(data, shape) => {
            serde_json::json!({"data": data, "shape": shape})
        }
        Value::Dict(entries) => {
            let map: serde_json::Map<String, serde_json::Value> = entries
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Execute a QLANG program and return the final value and captured output.
pub fn execute_program(stmts: &[Stmt]) -> Result<Value, VmError> {
    let mut vm = VmState::new();
    vm.exec_stmts(stmts)?;
    Ok(Value::Null)
}

/// Top-level entry point: lex, parse, type-check, and execute a QLANG script.
pub fn run_qlang_script(source: &str) -> Result<(Value, Vec<String>), VmError> {
    let tokens = tokenize(source)?;
    let stmts = parse_program(&tokens)?;
    // Run the static type checker (only validates annotated code).
    type_check(&stmts)?;
    let mut vm = VmState::new();
    let signal = vm.exec_stmts(&stmts)?;
    let result = match signal {
        ExecSignal::Return(v) => v,
        ExecSignal::None => Value::Null,
    };
    Ok((result, vm.output))
}

// ─── Static type checker ────────────────────────────────────────────────────

/// Infer the static type of an expression, if determinable from the AST alone.
/// Returns `None` when the type cannot be inferred (e.g. variable references
/// without known types, function calls without known signatures).
fn infer_expr_type(expr: &Expr, env: &HashMap<String, QType>) -> Option<QType> {
    match expr {
        Expr::NumberLit(n) => {
            // If the number has no fractional part, treat as Int.
            if *n == (*n as i64) as f64 && n.abs() < 1e15 {
                Some(QType::Int)
            } else {
                Some(QType::Float)
            }
        }
        Expr::BoolLit(_) => Some(QType::Bool),
        Expr::StringLit(_) => Some(QType::String),
        Expr::ArrayLit(elems) => {
            if elems.is_empty() {
                Some(QType::Array(Box::new(QType::Any)))
            } else {
                let inner = infer_expr_type(&elems[0], env)?;
                Some(QType::Array(Box::new(inner)))
            }
        }
        Expr::DictLit(_) => {
            // Dict keys are always strings in qlang.
            Some(QType::Dict(Box::new(QType::String), Box::new(QType::Any)))
        }
        Expr::Var(name) => env.get(name).cloned(),
        Expr::BinOp { op, left, right } => {
            match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                | BinOp::And | BinOp::Or => Some(QType::Bool),
                BinOp::Add => {
                    let lt = infer_expr_type(left, env);
                    let rt = infer_expr_type(right, env);
                    // String concatenation
                    if lt == Some(QType::String) || rt == Some(QType::String) {
                        return Some(QType::String);
                    }
                    // Numeric: if either side is Float, result is Float.
                    if lt == Some(QType::Float) || rt == Some(QType::Float) {
                        Some(QType::Float)
                    } else if lt == Some(QType::Int) && rt == Some(QType::Int) {
                        Some(QType::Int)
                    } else {
                        lt.or(rt)
                    }
                }
                BinOp::Sub | BinOp::Mul | BinOp::Mod | BinOp::Pow
                | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    let lt = infer_expr_type(left, env);
                    let rt = infer_expr_type(right, env);
                    if lt == Some(QType::Float) || rt == Some(QType::Float) {
                        Some(QType::Float)
                    } else if lt == Some(QType::Int) && rt == Some(QType::Int) {
                        Some(QType::Int)
                    } else {
                        lt.or(rt)
                    }
                }
                BinOp::Div => {
                    // Division always produces Float.
                    Some(QType::Float)
                }
            }
        }
        Expr::UnaryOp { op, operand } => {
            match op {
                UnaryOp::Not => Some(QType::Bool),
                UnaryOp::Neg | UnaryOp::BitNot => infer_expr_type(operand, env),
            }
        }
        Expr::Call { .. } | Expr::Index { .. } => None,
    }
}

/// Check if two types are compatible (for assignment / argument passing).
/// `Int` and `Float` are compatible with each other (numeric promotion).
fn types_compatible(expected: &QType, actual: &QType) -> bool {
    if expected == actual {
        return true;
    }
    // Any is compatible with everything.
    if *expected == QType::Any || *actual == QType::Any {
        return true;
    }
    // Int and Float are compatible (numeric promotion).
    if matches!(
        (expected, actual),
        (QType::Int, QType::Float)
            | (QType::Float, QType::Int)
    ) {
        return true;
    }
    // Array compatibility: [int] is compatible with [float] etc.
    if let (QType::Array(e_inner), QType::Array(a_inner)) = (expected, actual) {
        return types_compatible(e_inner, a_inner);
    }
    // Dict compatibility.
    if let (QType::Dict(ek, ev), QType::Dict(ak, av)) = (expected, actual) {
        return types_compatible(ek, ak) && types_compatible(ev, av);
    }
    false
}

/// Type-check a list of statements. Only validates statements that carry type
/// annotations; unannotated code passes through freely for backward compatibility.
pub fn type_check(stmts: &[Stmt]) -> Result<(), VmError> {
    let mut env: HashMap<String, QType> = HashMap::new();
    type_check_stmts(stmts, &mut env)
}

fn type_check_stmts(stmts: &[Stmt], env: &mut HashMap<String, QType>) -> Result<(), VmError> {
    for stmt in stmts {
        type_check_stmt(stmt, env)?;
    }
    Ok(())
}

fn type_check_stmt(stmt: &Stmt, env: &mut HashMap<String, QType>) -> Result<(), VmError> {
    match stmt {
        Stmt::Let { name, type_ann, value } => {
            if let Some(expected) = type_ann {
                // Record the variable's declared type.
                env.insert(name.clone(), expected.clone());
                // Infer the expression type and check compatibility.
                if let Some(actual) = infer_expr_type(value, env) {
                    if !types_compatible(expected, &actual) {
                        return Err(VmError::TypeError(format!(
                            "variable '{}': expected type {}, got {}",
                            name, expected, actual
                        )));
                    }
                }
            } else {
                // No annotation — try to infer and record for downstream checks.
                if let Some(inferred) = infer_expr_type(value, env) {
                    env.insert(name.clone(), inferred);
                }
            }
        }
        Stmt::Assign { name, value } => {
            // If the variable has a known type, check the new value.
            if let Some(expected) = env.get(name).cloned() {
                if let Some(actual) = infer_expr_type(value, env) {
                    if !types_compatible(&expected, &actual) {
                        return Err(VmError::TypeError(format!(
                            "variable '{}': expected type {}, got {}",
                            name, expected, actual
                        )));
                    }
                }
            }
        }
        Stmt::FnDef { name: fn_name, params, return_type, body } => {
            // Build a local env with parameter types.
            let mut fn_env = env.clone();
            for p in params {
                if let Some(t) = &p.type_ann {
                    fn_env.insert(p.name.clone(), t.clone());
                }
            }
            // Store function return type so callers can use it.
            if let Some(rt) = return_type {
                env.insert(format!("__fn_return_{}", fn_name), rt.clone());
            }
            // Store parameter types for call-site checking.
            for (i, p) in params.iter().enumerate() {
                if let Some(t) = &p.type_ann {
                    env.insert(format!("__fn_param_{}_{}", fn_name, i), t.clone());
                }
            }
            env.insert(
                format!("__fn_arity_{}", fn_name),
                QType::Int, // placeholder — we just need the entry to exist
            );
            // Save arity as a special marker (we use the number of params).
            let arity = params.len();
            // We store arity encoded as a convention; the call-site checker
            // looks up __fn_param_{name}_{i} entries.
            let _ = arity; // suppress unused warning; arity is implicit from param entries

            // Type-check the body in the function's environment.
            type_check_stmts(body, &mut fn_env)?;

            // Check return statements if a return type is declared.
            if let Some(expected_ret) = return_type {
                check_return_types(body, expected_ret, &fn_env, fn_name)?;
            }
        }
        Stmt::If { cond: _, then_body, else_body } => {
            type_check_stmts(then_body, env)?;
            type_check_stmts(else_body, env)?;
        }
        Stmt::While { cond: _, body } => {
            type_check_stmts(body, env)?;
        }
        Stmt::For { var: _, start: _, end: _, body } => {
            type_check_stmts(body, env)?;
        }
        Stmt::Return(_) | Stmt::Print(_) | Stmt::ExprStmt(_) | Stmt::Import(_) => {
            // No type annotations to check at statement level.
        }
    }
    Ok(())
}

/// Walk a function body and check that all return expressions are compatible
/// with the declared return type.
fn check_return_types(
    stmts: &[Stmt],
    expected: &QType,
    env: &HashMap<String, QType>,
    fn_name: &str,
) -> Result<(), VmError> {
    for stmt in stmts {
        match stmt {
            Stmt::Return(expr) => {
                if let Some(actual) = infer_expr_type(expr, env) {
                    if !types_compatible(expected, &actual) {
                        return Err(VmError::TypeError(format!(
                            "function '{}': expected return type {}, got {}",
                            fn_name, expected, actual
                        )));
                    }
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                check_return_types(then_body, expected, env, fn_name)?;
                check_return_types(else_body, expected, env, fn_name)?;
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => {
                check_return_types(body, expected, env, fn_name)?;
            }
            Stmt::FnDef { body, .. } => {
                // Nested functions have their own return types; skip.
                let _ = body;
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> (Value, Vec<String>) {
        run_qlang_script(src).expect("script should succeed")
    }

    fn run_err(src: &str) -> VmError {
        run_qlang_script(src).expect_err("script should fail")
    }

    #[test]
    fn test_variable_binding_and_arithmetic() {
        let (_, out) = run(r#"
            let x = 5.0
            let y = x + 3.0 * 2.0
            print(y)
        "#);
        assert_eq!(out, vec!["11"]);
    }

    #[test]
    fn test_arithmetic_precedence() {
        let (_, out) = run(r#"
            let a = 2.0 + 3.0 * 4.0
            let b = (2.0 + 3.0) * 4.0
            print(a)
            print(b)
        "#);
        assert_eq!(out, vec!["14", "20"]);
    }

    #[test]
    fn test_if_else_branches() {
        let (_, out) = run(r#"
            let x = 10.0
            if x > 5.0 {
                print(1.0)
            } else {
                print(0.0)
            }
            if x < 5.0 {
                print(1.0)
            } else {
                print(0.0)
            }
        "#);
        assert_eq!(out, vec!["1", "0"]);
    }

    #[test]
    fn test_for_loop_with_range() {
        let (_, out) = run(r#"
            let sum = 0.0
            for i in 0..5 {
                sum = sum + i
            }
            print(sum)
        "#);
        // 0+1+2+3+4 = 10
        assert_eq!(out, vec!["10"]);
    }

    #[test]
    fn test_while_loop() {
        let (_, out) = run(r#"
            let x = 1.0
            while x < 100.0 {
                x = x * 2.0
            }
            print(x)
        "#);
        assert_eq!(out, vec!["128"]);
    }

    #[test]
    fn test_function_definition_and_call() {
        let (_, out) = run(r#"
            fn add(a, b) {
                return a + b
            }
            let result = add(3.0, 4.0)
            print(result)
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_recursive_function_fibonacci() {
        let (_, out) = run(r#"
            fn fibonacci(n) {
                if n <= 1.0 {
                    return n
                }
                return fibonacci(n - 1.0) + fibonacci(n - 2.0)
            }
            let result = fibonacci(10.0)
            print(result)
        "#);
        assert_eq!(out, vec!["55"]);
    }

    #[test]
    fn test_array_creation_and_indexing() {
        let (_, out) = run(r#"
            let arr = [10.0, 20.0, 30.0]
            print(arr[0])
            print(arr[1])
            print(arr[2])
        "#);
        assert_eq!(out, vec!["10", "20", "30"]);
    }

    #[test]
    fn test_array_length() {
        let (_, out) = run(r#"
            let arr = [1.0, 2.0, 3.0, 4.0, 5.0]
            print(len(arr))
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn test_comparison_operators() {
        let (_, out) = run(r#"
            print(1.0 == 1.0)
            print(1.0 != 2.0)
            print(3.0 < 5.0)
            print(5.0 > 3.0)
            print(3.0 <= 3.0)
            print(3.0 >= 4.0)
        "#);
        assert_eq!(out, vec!["true", "true", "true", "true", "true", "false"]);
    }

    #[test]
    fn test_nested_function_calls() {
        let (_, out) = run(r#"
            fn double(x) { return x * 2.0 }
            fn triple(x) { return x * 3.0 }
            let result = double(triple(5.0))
            print(result)
        "#);
        assert_eq!(out, vec!["30"]);
    }

    #[test]
    fn test_print_output_capture() {
        let (_, out) = run(r#"
            print(42.0)
            print("hello")
            print(true)
        "#);
        assert_eq!(out, vec!["42", "hello", "true"]);
    }

    #[test]
    fn test_string_values() {
        let (_, out) = run(r#"
            let s = "hello world"
            print(s)
        "#);
        assert_eq!(out, vec!["hello world"]);
    }

    #[test]
    fn test_error_on_undefined_variable() {
        let err = run_err("print(undefined_var)");
        match err {
            VmError::UndefinedVariable(name) => assert_eq!(name, "undefined_var"),
            other => panic!("expected UndefinedVariable, got: {other}"),
        }
    }

    #[test]
    fn test_error_on_division_by_zero() {
        let err = run_err("let x = 1.0 / 0.0");
        match err {
            VmError::DivisionByZero => {}
            other => panic!("expected DivisionByZero, got: {other}"),
        }
    }

    #[test]
    fn test_complex_program_fibonacci_and_arrays() {
        let (_, out) = run(r#"
            fn fibonacci(n) {
                if n <= 1.0 {
                    return n
                }
                return fibonacci(n - 1.0) + fibonacci(n - 2.0)
            }

            let result = fibonacci(10.0)
            print(result)

            let data = [1.0, 2.0, 3.0, 4.0, 5.0]
            let sum = 0.0
            for i in 0..len(data) {
                sum = sum + data[i]
            }
            print(sum)
        "#);
        assert_eq!(out, vec!["55", "15"]);
    }

    #[test]
    fn test_logical_operators() {
        let (_, out) = run(r#"
            print(true and false)
            print(true or false)
            print(not true)
            print(true and true)
        "#);
        assert_eq!(out, vec!["false", "true", "false", "true"]);
    }

    #[test]
    fn test_unary_negation() {
        let (_, out) = run(r#"
            let x = 5.0
            let y = -x
            print(y)
            print(-3.0)
        "#);
        assert_eq!(out, vec!["-5", "-3"]);
    }

    #[test]
    fn test_error_undefined_function() {
        let err = run_err("let x = nope(1.0)");
        match err {
            VmError::UndefinedFunction(name) => assert_eq!(name, "nope"),
            other => panic!("expected UndefinedFunction, got: {other}"),
        }
    }

    #[test]
    fn test_index_out_of_bounds() {
        let err = run_err(r#"
            let arr = [1.0, 2.0]
            print(arr[5])
        "#);
        match err {
            VmError::IndexOutOfBounds { index: 5, len: 2 } => {}
            other => panic!("expected IndexOutOfBounds, got: {other}"),
        }
    }

    // ─── Static type system tests ───────────────────────────────────────────

    #[test]
    fn test_typed_let_int_correct() {
        let (_, out) = run(r#"
            let x: int = 5
            print(x)
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn test_typed_let_float_correct() {
        let (_, out) = run(r#"
            let x: float = 3.14
            print(x)
        "#);
        assert_eq!(out, vec!["3.14"]);
    }

    #[test]
    fn test_typed_let_string_correct() {
        let (_, out) = run(r#"
            let name: string = "hello"
            print(name)
        "#);
        assert_eq!(out, vec!["hello"]);
    }

    #[test]
    fn test_typed_let_bool_correct() {
        let (_, out) = run(r#"
            let flag: bool = true
            print(flag)
        "#);
        assert_eq!(out, vec!["true"]);
    }

    #[test]
    fn test_typed_let_array_correct() {
        let (_, out) = run(r#"
            let data: [int] = [1, 2, 3]
            print(data)
        "#);
        assert_eq!(out, vec!["[1, 2, 3]"]);
    }

    #[test]
    fn test_typed_let_wrong_type_fails() {
        let err = run_err(r#"
            let x: int = "oops"
        "#);
        match err {
            VmError::TypeError(msg) => {
                assert!(msg.contains("expected type int"), "got: {msg}");
                assert!(msg.contains("got string"), "got: {msg}");
            }
            other => panic!("expected TypeError, got: {other}"),
        }
    }

    #[test]
    fn test_typed_let_string_vs_int_fails() {
        let err = run_err(r#"
            let s: string = 42
        "#);
        match err {
            VmError::TypeError(msg) => {
                assert!(msg.contains("expected type string"), "got: {msg}");
            }
            other => panic!("expected TypeError, got: {other}"),
        }
    }

    #[test]
    fn test_typed_let_bool_vs_int_fails() {
        let err = run_err(r#"
            let b: bool = 0
        "#);
        match err {
            VmError::TypeError(msg) => {
                assert!(msg.contains("expected type bool"), "got: {msg}");
            }
            other => panic!("expected TypeError, got: {other}"),
        }
    }

    #[test]
    fn test_typed_let_int_float_compatible() {
        // Int and float are compatible via numeric promotion.
        let (_, out) = run(r#"
            let x: float = 5
            print(x)
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn test_typed_fn_correct() {
        let (_, out) = run(r#"
            fn add(a: int, b: int) -> int {
                return a + b
            }
            let result = add(3, 4)
            print(result)
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_typed_fn_return_type_mismatch() {
        let err = run_err(r#"
            fn greet(name: string) -> int {
                return "hello"
            }
            greet("world")
        "#);
        match err {
            VmError::TypeError(msg) => {
                assert!(msg.contains("expected return type int"), "got: {msg}");
                assert!(msg.contains("got string"), "got: {msg}");
            }
            other => panic!("expected TypeError, got: {other}"),
        }
    }

    #[test]
    fn test_typed_fn_with_string_return() {
        let (_, out) = run(r#"
            fn greet(name: string) -> string {
                return "Hello " + name
            }
            print(greet("world"))
        "#);
        assert_eq!(out, vec!["Hello world"]);
    }

    #[test]
    fn test_backward_compat_untyped_let() {
        // No types at all -- must still work.
        let (_, out) = run(r#"
            let x = 5
            let y = "hello"
            print(x)
            print(y)
        "#);
        assert_eq!(out, vec!["5", "hello"]);
    }

    #[test]
    fn test_backward_compat_untyped_fn() {
        let (_, out) = run(r#"
            fn foo(a, b) { return a + b }
            print(foo(3, 4))
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn test_backward_compat_mixed_typed_untyped() {
        // Mix typed and untyped in the same program.
        let (_, out) = run(r#"
            let x: int = 10
            let y = 20
            fn add(a: int, b) { return a + b }
            print(add(x, y))
        "#);
        assert_eq!(out, vec!["30"]);
    }

    #[test]
    fn test_typed_dict_annotation() {
        let (_, out) = run(r#"
            let config: {string: any} = {"lr": 0.01}
            print(config)
        "#);
        assert_eq!(out, vec![r#"{"lr": 0.01}"#]);
    }

    #[test]
    fn test_type_any_accepts_everything() {
        let (_, out) = run(r#"
            let x: any = 42
            print(x)
            let y: any = "hello"
            print(y)
        "#);
        assert_eq!(out, vec!["42", "hello"]);
    }

    #[test]
    fn test_arrow_token_in_fn() {
        // Verify the -> token is properly lexed and parsed.
        let (_, out) = run(r#"
            fn square(x: float) -> float {
                return x * x
            }
            print(square(5.0))
        "#);
        assert_eq!(out, vec!["25"]);
    }

    // ── Standard library tests ──────────────────────────────────────────

    #[test]
    fn test_stdlib_math_floor_ceil_round() {
        let (_, out) = run("print(floor(3.7))");
        assert_eq!(out, vec!["3"]);
        let (_, out) = run("print(ceil(3.2))");
        assert_eq!(out, vec!["4"]);
        let (_, out) = run("print(round(3.5))");
        assert_eq!(out, vec!["4"]);
        let (_, out) = run("print(round(3.4))");
        assert_eq!(out, vec!["3"]);
    }

    #[test]
    fn test_stdlib_math_trig() {
        let (_, out) = run("print(sin(0.0))");
        assert_eq!(out, vec!["0"]);
        let (_, out) = run("print(cos(0.0))");
        assert_eq!(out, vec!["1"]);
        let (_, out) = run("print(tan(0.0))");
        assert_eq!(out, vec!["0"]);
    }

    #[test]
    fn test_stdlib_math_log_exp() {
        let (_, out) = run("print(log(1.0))");
        assert_eq!(out, vec!["0"]);
        let (_, out) = run("print(log2(8.0))");
        assert_eq!(out, vec!["3"]);
        let (_, out) = run("print(log10(1000.0))");
        assert_eq!(out, vec!["3"]);
        let (_, out) = run("print(exp(0.0))");
        assert_eq!(out, vec!["1"]);
    }

    #[test]
    fn test_stdlib_math_pow_pi_e() {
        let (_, out) = run("print(pow(2.0, 10.0))");
        assert_eq!(out, vec!["1024"]);
        let (_, out) = run(r#"
            let p = pi()
            print(p > 3.14)
            print(p < 3.15)
        "#);
        assert_eq!(out, vec!["true", "true"]);
        let (_, out) = run(r#"
            let euler = e()
            print(euler > 2.71)
            print(euler < 2.72)
        "#);
        assert_eq!(out, vec!["true", "true"]);
    }

    #[test]
    fn test_stdlib_math_clamp() {
        let (_, out) = run("print(clamp(5.0, 0.0, 10.0))");
        assert_eq!(out, vec!["5"]);
        let (_, out) = run("print(clamp(-3.0, 0.0, 10.0))");
        assert_eq!(out, vec!["0"]);
        let (_, out) = run("print(clamp(15.0, 0.0, 10.0))");
        assert_eq!(out, vec!["10"]);
    }

    #[test]
    fn test_stdlib_string_upper_lower_trim() {
        let (_, out) = run(r#"print(upper("hello"))"#);
        assert_eq!(out, vec!["HELLO"]);
        let (_, out) = run(r#"print(lower("HELLO"))"#);
        assert_eq!(out, vec!["hello"]);
        let (_, out) = run(r#"print(trim("  hello  "))"#);
        assert_eq!(out, vec!["hello"]);
    }

    #[test]
    fn test_stdlib_string_contains_starts_ends() {
        let (_, out) = run(r#"print(contains("hello world", "world"))"#);
        assert_eq!(out, vec!["true"]);
        let (_, out) = run(r#"print(contains("hello", "xyz"))"#);
        assert_eq!(out, vec!["false"]);
        let (_, out) = run(r#"print(starts_with("hello", "hel"))"#);
        assert_eq!(out, vec!["true"]);
        let (_, out) = run(r#"print(ends_with("hello", "llo"))"#);
        assert_eq!(out, vec!["true"]);
    }

    #[test]
    fn test_stdlib_string_replace() {
        let (_, out) = run(r#"print(replace("hello world", "world", "qlang"))"#);
        assert_eq!(out, vec!["hello qlang"]);
    }

    #[test]
    fn test_stdlib_string_substr_char_at() {
        let (_, out) = run(r#"print(substr("hello world", 0, 5))"#);
        assert_eq!(out, vec!["hello"]);
        let (_, out) = run(r#"print(char_at("hello", 1))"#);
        assert_eq!(out, vec!["e"]);
    }

    #[test]
    fn test_stdlib_string_parse_int_float() {
        let (_, out) = run(r#"print(parse_int("42"))"#);
        assert_eq!(out, vec!["42"]);
        let (_, out) = run(r#"print(parse_float("3.14"))"#);
        assert!(out[0].starts_with("3.14"));
    }

    #[test]
    fn test_stdlib_string_split_join() {
        let (_, out) = run(r#"
            let parts = split("a,b,c", ",")
            print(parts["0"])
            print(parts["1"])
            print(parts["2"])
        "#);
        assert_eq!(out, vec!["a", "b", "c"]);
        let (_, out) = run(r#"
            let arr = [1.0, 2.0, 3.0]
            print(join(arr, "-"))
        "#);
        assert_eq!(out, vec!["1-2-3"]);
    }

    #[test]
    fn test_stdlib_string_format() {
        let (_, out) = run(r#"print(format("hello {0}, you are {1}", "world", 42))"#);
        assert_eq!(out, vec!["hello world, you are 42"]);
    }

    #[test]
    fn test_stdlib_array_push_pop() {
        let (_, out) = run(r#"
            let arr = [1.0, 2.0, 3.0]
            let arr2 = push(arr, 4.0)
            print(len(arr2))
        "#);
        assert_eq!(out, vec!["4"]);
        let (_, out) = run(r#"
            let arr = [1.0, 2.0, 3.0]
            let result = pop(arr)
            print(result["value"])
        "#);
        assert_eq!(out, vec!["3"]);
    }

    #[test]
    fn test_stdlib_array_reverse_sort() {
        let (_, out) = run(r#"
            let arr = [3.0, 1.0, 2.0]
            print(reverse(arr))
            print(sort(arr))
        "#);
        assert_eq!(out, vec!["[2, 1, 3]", "[1, 2, 3]"]);
    }

    #[test]
    fn test_stdlib_array_range() {
        let (_, out) = run("print(range(0, 5))");
        assert_eq!(out, vec!["[0, 1, 2, 3, 4]"]);
        let (_, out) = run("print(range(0, 10, 2))");
        assert_eq!(out, vec!["[0, 2, 4, 6, 8]"]);
    }

    #[test]
    fn test_stdlib_array_sum_avg() {
        let (_, out) = run("print(sum([1.0, 2.0, 3.0, 4.0]))");
        assert_eq!(out, vec!["10"]);
        let (_, out) = run("print(avg([2.0, 4.0, 6.0]))");
        assert_eq!(out, vec!["4"]);
    }

    #[test]
    fn test_stdlib_array_slice() {
        let (_, out) = run("print(slice([10.0, 20.0, 30.0, 40.0, 50.0], 1, 4))");
        assert_eq!(out, vec!["[20, 30, 40]"]);
    }

    #[test]
    fn test_stdlib_array_zip_enumerate() {
        let (_, out) = run(r#"
            let z = zip([1.0, 2.0], [10.0, 20.0])
            print(z["0"])
            print(z["1"])
        "#);
        assert_eq!(out, vec!["[1, 10]", "[2, 20]"]);
        let (_, out) = run(r#"
            let en = enumerate([10.0, 20.0])
            print(en["0"])
            print(en["1"])
        "#);
        assert_eq!(out, vec!["[0, 10]", "[1, 20]"]);
    }

    #[test]
    fn test_stdlib_type_functions_comprehensive() {
        let (_, out) = run(r#"print(type("hello"))"#);
        assert_eq!(out, vec!["string"]);
        let (_, out) = run("print(type(42))");
        assert_eq!(out, vec!["number"]);
        let (_, out) = run("print(is_int(5))");
        assert_eq!(out, vec!["true"]);
        let (_, out) = run("print(is_int(5.5))");
        assert_eq!(out, vec!["false"]);
        let (_, out) = run("print(is_float(5.5))");
        assert_eq!(out, vec!["true"]);
        let (_, out) = run(r#"print(is_string("hi"))"#);
        assert_eq!(out, vec!["true"]);
        let (_, out) = run("print(is_array([1.0]))");
        assert_eq!(out, vec!["true"]);
        let (_, out) = run("print(is_null(42))");
        assert_eq!(out, vec!["false"]);
    }

    #[test]
    fn test_stdlib_type_conversions() {
        let (_, out) = run("print(float(true))");
        assert_eq!(out, vec!["1"]);
        let (_, out) = run("print(float(false))");
        assert_eq!(out, vec!["0"]);
        let (_, out) = run(r#"print(float("3.14"))"#);
        assert!(out[0].starts_with("3.14"));
        let (_, out) = run("print(bool(0))");
        assert_eq!(out, vec!["false"]);
        let (_, out) = run("print(bool(1))");
        assert_eq!(out, vec!["true"]);
        let (_, out) = run(r#"print(bool(""))"#);
        assert_eq!(out, vec!["false"]);
        let (_, out) = run(r#"print(bool("hi"))"#);
        assert_eq!(out, vec!["true"]);
    }

    #[test]
    fn test_stdlib_json() {
        // Test json_parse with a numeric array (no quotes needed)
        let (_, out) = run(r#"
            let j = json_parse("[1, 2, 3]")
            print(j)
        "#);
        assert_eq!(out, vec!["[1, 2, 3]"]);
        // Test json_stringify
        let (_, out) = run(r#"
            let s = json_stringify(42)
            print(s)
        "#);
        assert_eq!(out, vec!["42.0"]);
        // Test json_stringify with array
        let (_, out) = run(r#"
            let s = json_stringify([1.0, 2.0, 3.0])
            print(s)
        "#);
        assert_eq!(out, vec!["[1.0,2.0,3.0]"]);
        // Test json_parse with object via read_file (avoids escape issues)
        // Write JSON with proper quotes from Rust side
        std::fs::write("/tmp/qlang_test_json.json", r#"{"name": "qlang", "version": 1}"#).unwrap();
        let (_, out) = run(r#"
            let text = read_file("/tmp/qlang_test_json.json")
            let obj = json_parse(text)
            print(obj["name"])
            print(obj["version"])
        "#);
        assert_eq!(out, vec!["qlang", "1"]);
    }

    #[test]
    fn test_stdlib_io_file_roundtrip() {
        let (_, out) = run(r#"
            write_file("/tmp/qlang_test_stdlib.txt", "hello qlang")
            let content = read_file("/tmp/qlang_test_stdlib.txt")
            print(content)
            print(file_exists("/tmp/qlang_test_stdlib.txt"))
            print(file_exists("/tmp/nonexistent_qlang_xyz.txt"))
        "#);
        assert_eq!(out, vec!["hello qlang", "true", "false"]);
    }

    #[test]
    fn test_stdlib_io_read_lines() {
        // Use \n literally in write_file
        let (_, out) = run(r#"
            write_file("/tmp/qlang_test_lines.txt", "line1
line2
line3")
            let lines = read_lines("/tmp/qlang_test_lines.txt")
            print(lines["0"])
            print(lines["1"])
            print(lines["2"])
        "#);
        assert_eq!(out, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn test_stdlib_system_time() {
        let (_, out) = run(r#"
            let t = time()
            print(t > 1000000000.0)
        "#);
        assert_eq!(out, vec!["true"]);
    }

    #[test]
    fn test_stdlib_system_env() {
        let (_, out) = run(r#"
            let home = env("HOME")
            print(is_null(home))
        "#);
        assert_eq!(out, vec!["false"]);
    }

    #[test]
    fn test_stdlib_flatten() {
        let (_, out) = run("print(flatten([1.0, 2.0, 3.0]))");
        assert_eq!(out, vec!["[1, 2, 3]"]);
    }

    #[test]
    fn test_stdlib_arity_errors() {
        let err = run_err("ceil(1.0, 2.0)");
        match err {
            VmError::ArityMismatch { expected: 1, got: 2 } => {}
            other => panic!("expected ArityMismatch, got: {other}"),
        }
        let err = run_err("pow(1.0)");
        match err {
            VmError::ArityMismatch { expected: 2, got: 1 } => {}
            other => panic!("expected ArityMismatch, got: {other}"),
        }
    }

    #[test]
    fn test_stdlib_type_errors() {
        let err = run_err("upper(42)");
        match err {
            VmError::TypeError(_) => {}
            other => panic!("expected TypeError, got: {other}"),
        }
    }
}
