//! Bytecode compiler + stack-based VM for QLANG scripts.
//!
//! This is 10-50x faster than the tree-walking interpreter in vm.rs.
//! The approach: compile AST to flat bytecode, then execute on a stack machine.
//! No external dependencies (no LLVM). Same idea as CPython, Lua, Ruby.
//!
//! Usage:
//!   let (value, output) = run_bytecode(source)?;

use std::collections::HashMap;

use crate::vm::{
    tokenize, type_check, BinOp, Expr, Parser, Stmt, UnaryOp, Value, VmError,
};

// ─── Opcodes ────────────────────────────────────────────────────────────────

/// Bytecode instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // Stack operations
    Const = 0,    // push constant from pool (u16 index)
    Pop,          // discard top of stack

    // Variables
    LoadLocal,    // push local variable (u8 slot)
    StoreLocal,   // pop and store to local (u8 slot)
    LoadGlobal,   // push global variable (u16 constant index = name)
    StoreGlobal,  // pop and store to global (u16 constant index = name)

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Neg, // unary negate

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // Comparison (pushes bool)
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,

    // Logic
    And,
    Or,
    Not,

    // Control flow
    Jump,         // unconditional jump (u16 offset)
    JumpIfFalse,  // pop, jump if false (u16 offset)
    JumpIfTrue,   // pop, jump if true (u16 offset)
    Loop,         // jump backward (u16 offset)

    // Functions
    Call,   // call user function (u16 name index, u8 arg count)
    Return, // return from function

    // Built-in functions
    Print,    // pop and print
    Len,      // pop, push length
    TypeOf,   // pop, push type name as string
    Str,      // pop, push string representation
    Int,      // pop, push floored integer
    Sqrt,     // pop, push sqrt
    Abs,      // pop, push abs
    Floor,    // pop, push floor
    Min,      // pop 2, push min
    Max,      // pop 2, push max
    Push,     // pop value, pop array, push array with value appended

    // Arrays
    MakeArray, // pop N items, push array (u16 count)
    Index,     // pop index, pop array/dict/string, push element

    // Dicts
    MakeDict, // pop N key-value pairs, push dict (u16 count = number of entries)

    // Halt
    Halt,
}

impl OpCode {
    fn from_u8(byte: u8) -> Option<OpCode> {
        use OpCode::*;
        match byte {
            0 => Some(Const), 1 => Some(Pop),
            2 => Some(LoadLocal), 3 => Some(StoreLocal),
            4 => Some(LoadGlobal), 5 => Some(StoreGlobal),
            6 => Some(Add), 7 => Some(Sub), 8 => Some(Mul), 9 => Some(Div),
            10 => Some(Mod), 11 => Some(Pow), 12 => Some(Neg),
            13 => Some(BitAnd), 14 => Some(BitOr), 15 => Some(BitXor),
            16 => Some(BitNot), 17 => Some(Shl), 18 => Some(Shr),
            19 => Some(Eq), 20 => Some(Ne), 21 => Some(Lt),
            22 => Some(Gt), 23 => Some(Le), 24 => Some(Ge),
            25 => Some(And), 26 => Some(Or), 27 => Some(Not),
            28 => Some(Jump), 29 => Some(JumpIfFalse),
            30 => Some(JumpIfTrue), 31 => Some(Loop),
            32 => Some(Call), 33 => Some(Return),
            34 => Some(Print), 35 => Some(Len), 36 => Some(TypeOf),
            37 => Some(Str), 38 => Some(Int), 39 => Some(Sqrt),
            40 => Some(Abs), 41 => Some(Floor), 42 => Some(Min),
            43 => Some(Max), 44 => Some(Push),
            45 => Some(MakeArray), 46 => Some(Index),
            47 => Some(MakeDict),
            48 => Some(Halt),
            _ => None,
        }
    }
}

// ─── Chunk ──────────────────────────────────────────────────────────────────

/// A compiled chunk of bytecode.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<u8>,
    pub constants: Vec<Value>,
}

impl Chunk {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
        }
    }

    fn emit(&mut self, op: OpCode) {
        self.code.push(op as u8);
    }

    fn emit_byte(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn emit_u16(&mut self, val: u16) {
        self.code.push((val >> 8) as u8);
        self.code.push((val & 0xff) as u8);
    }

    fn add_constant(&mut self, val: Value) -> u16 {
        // Reuse existing constant if it matches
        for (i, c) in self.constants.iter().enumerate() {
            if values_identical(c, &val) {
                return i as u16;
            }
        }
        let idx = self.constants.len();
        self.constants.push(val);
        idx as u16
    }

}

/// Check if two Values are identical (used for constant deduplication).
fn values_identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.to_bits() == y.to_bits(),
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

// ─── Compiled function ──────────────────────────────────────────────────────

/// A user-defined function compiled to bytecode.
#[derive(Debug, Clone)]
struct CompiledFunction {
    params: Vec<String>,
    chunk: Chunk,
}

// ─── Local variable tracking ────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Local {
    name: String,
    depth: usize,
}

// ─── Compiler ───────────────────────────────────────────────────────────────

/// Compiles AST (Vec<Stmt>) into bytecode (Chunk).
struct Compiler {
    chunk: Chunk,
    locals: Vec<Local>,
    scope_depth: usize,
    functions: HashMap<String, CompiledFunction>,
}

impl Compiler {
    fn new() -> Self {
        Self {
            chunk: Chunk::new(),
            locals: Vec::new(),
            scope_depth: 0,
            functions: HashMap::new(),
        }
    }

    /// Compile a full program.
    fn compile(stmts: &[Stmt]) -> Result<(Chunk, HashMap<String, CompiledFunction>), VmError> {
        let mut compiler = Compiler::new();
        for stmt in stmts {
            compiler.compile_stmt(stmt)?;
        }
        compiler.chunk.emit(OpCode::Halt);
        Ok((compiler.chunk, compiler.functions))
    }

    // ── Emit helpers ──

    fn emit_jump(&mut self, op: OpCode) -> usize {
        self.chunk.emit(op);
        let offset = self.chunk.code.len();
        self.chunk.emit_u16(0xffff); // placeholder
        offset
    }

    fn patch_jump(&mut self, offset: usize) {
        let jump_target = self.chunk.code.len();
        let distance = jump_target - offset - 2; // -2 for the u16 placeholder
        self.chunk.code[offset] = (distance >> 8) as u8;
        self.chunk.code[offset + 1] = (distance & 0xff) as u8;
    }

    fn emit_loop(&mut self, loop_start: usize) {
        self.chunk.emit(OpCode::Loop);
        // +2 for the u16 operand we are about to emit
        let distance = self.chunk.code.len() - loop_start + 2;
        self.chunk.emit_u16(distance as u16);
    }

    // ── Scope helpers ──

    fn begin_scope(&mut self) {
        self.scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.scope_depth -= 1;
        // Pop locals that went out of scope
        while let Some(local) = self.locals.last() {
            if local.depth > self.scope_depth {
                self.locals.pop();
                self.chunk.emit(OpCode::Pop);
            } else {
                break;
            }
        }
    }

    fn add_local(&mut self, name: &str) {
        self.locals.push(Local {
            name: name.to_string(),
            depth: self.scope_depth,
        });
    }

    fn resolve_local(&self, name: &str) -> Option<usize> {
        for (i, local) in self.locals.iter().enumerate().rev() {
            if local.name == name {
                return Some(i);
            }
        }
        None
    }

    // ── Statement compilation ──

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), VmError> {
        match stmt {
            Stmt::Let { name, type_ann: _, value } => {
                self.compile_expr(value)?;
                self.add_local(name);
                // The value is on the stack, which IS the local variable slot
            }
            Stmt::Assign { name, value } => {
                self.compile_expr(value)?;
                if let Some(idx) = self.resolve_local(name) {
                    self.chunk.emit(OpCode::StoreLocal);
                    self.chunk.emit_byte(idx as u8);
                } else {
                    let idx = self.chunk.add_constant(Value::String(name.clone()));
                    self.chunk.emit(OpCode::StoreGlobal);
                    self.chunk.emit_u16(idx);
                }
            }
            Stmt::Print(expr) => {
                self.compile_expr(expr)?;
                self.chunk.emit(OpCode::Print);
            }
            Stmt::ExprStmt(expr) => {
                self.compile_expr(expr)?;
                self.chunk.emit(OpCode::Pop);
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                self.compile_expr(cond)?;
                let jump_false = self.emit_jump(OpCode::JumpIfFalse);

                // then branch
                self.begin_scope();
                for s in then_body {
                    self.compile_stmt(s)?;
                }
                self.end_scope();

                if else_body.is_empty() {
                    self.patch_jump(jump_false);
                } else {
                    let jump_over = self.emit_jump(OpCode::Jump);
                    self.patch_jump(jump_false);

                    // else branch
                    self.begin_scope();
                    for s in else_body {
                        self.compile_stmt(s)?;
                    }
                    self.end_scope();

                    self.patch_jump(jump_over);
                }
            }
            Stmt::While { cond, body } => {
                let loop_start = self.chunk.code.len();
                self.compile_expr(cond)?;
                let exit_jump = self.emit_jump(OpCode::JumpIfFalse);

                self.begin_scope();
                for s in body {
                    self.compile_stmt(s)?;
                }
                self.end_scope();

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);
            }
            Stmt::For {
                var,
                start,
                end,
                body,
            } => {
                // Push start value as the loop variable
                self.compile_expr(start)?;
                self.add_local(var);
                let local_idx = self.locals.len() - 1;

                let loop_start = self.chunk.code.len();

                // Load var, load end, compare: var < end
                self.chunk.emit(OpCode::LoadLocal);
                self.chunk.emit_byte(local_idx as u8);
                self.compile_expr(end)?;
                self.chunk.emit(OpCode::Lt);

                let exit_jump = self.emit_jump(OpCode::JumpIfFalse);

                // Body (in a new scope for body-local variables)
                self.begin_scope();
                for s in body {
                    self.compile_stmt(s)?;
                }
                self.end_scope();

                // Increment: var = var + 1
                self.chunk.emit(OpCode::LoadLocal);
                self.chunk.emit_byte(local_idx as u8);
                let one = self.chunk.add_constant(Value::Number(1.0));
                self.chunk.emit(OpCode::Const);
                self.chunk.emit_u16(one);
                self.chunk.emit(OpCode::Add);
                self.chunk.emit(OpCode::StoreLocal);
                self.chunk.emit_byte(local_idx as u8);

                self.emit_loop(loop_start);
                self.patch_jump(exit_jump);

                // Pop the loop variable
                self.locals.pop();
                self.chunk.emit(OpCode::Pop);
            }
            Stmt::FnDef { name, params, return_type: _, body } => {
                // Compile the function body into a separate chunk
                let mut fn_compiler = Compiler::new();
                // Add params as locals
                for p in params {
                    fn_compiler.add_local(&p.name);
                }
                for s in body {
                    fn_compiler.compile_stmt(s)?;
                }
                // Implicit return null
                let null_idx = fn_compiler.chunk.add_constant(Value::Null);
                fn_compiler.chunk.emit(OpCode::Const);
                fn_compiler.chunk.emit_u16(null_idx);
                fn_compiler.chunk.emit(OpCode::Return);

                self.functions.insert(
                    name.clone(),
                    CompiledFunction {
                        params: params.iter().map(|p| p.name.clone()).collect(),
                        chunk: fn_compiler.chunk,
                    },
                );
                // Also collect sub-functions defined inside this function
                for (k, v) in fn_compiler.functions {
                    self.functions.insert(k, v);
                }
            }
            Stmt::Return(expr) => {
                self.compile_expr(expr)?;
                self.chunk.emit(OpCode::Return);
            }
            Stmt::Import(_path) => {
                // Import is not supported in bytecode VM; fall back to tree-walker
                return Err(VmError::RuntimeError(
                    "import not supported in bytecode VM".into(),
                ));
            }
        }
        Ok(())
    }

    // ── Expression compilation ──

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), VmError> {
        match expr {
            Expr::NumberLit(n) => {
                let idx = self.chunk.add_constant(Value::Number(*n));
                self.chunk.emit(OpCode::Const);
                self.chunk.emit_u16(idx);
            }
            Expr::StringLit(s) => {
                let idx = self.chunk.add_constant(Value::String(s.clone()));
                self.chunk.emit(OpCode::Const);
                self.chunk.emit_u16(idx);
            }
            Expr::BoolLit(b) => {
                let idx = self.chunk.add_constant(Value::Bool(*b));
                self.chunk.emit(OpCode::Const);
                self.chunk.emit_u16(idx);
            }
            Expr::Var(name) => {
                if let Some(idx) = self.resolve_local(name) {
                    self.chunk.emit(OpCode::LoadLocal);
                    self.chunk.emit_byte(idx as u8);
                } else {
                    let idx = self.chunk.add_constant(Value::String(name.clone()));
                    self.chunk.emit(OpCode::LoadGlobal);
                    self.chunk.emit_u16(idx);
                }
            }
            Expr::BinOp { op, left, right } => {
                // Short-circuit for And/Or
                match op {
                    BinOp::And => {
                        self.compile_expr(left)?;
                        let jump = self.emit_jump(OpCode::JumpIfFalse);
                        self.chunk.emit(OpCode::Pop); // discard true left
                        self.compile_expr(right)?;
                        let end = self.emit_jump(OpCode::Jump);
                        self.patch_jump(jump);
                        // Left was false, replace with false
                        self.chunk.emit(OpCode::Pop); // pop the false left
                        let f = self.chunk.add_constant(Value::Bool(false));
                        self.chunk.emit(OpCode::Const);
                        self.chunk.emit_u16(f);
                        self.patch_jump(end);
                        return Ok(());
                    }
                    BinOp::Or => {
                        self.compile_expr(left)?;
                        let jump = self.emit_jump(OpCode::JumpIfTrue);
                        self.chunk.emit(OpCode::Pop); // discard false left
                        self.compile_expr(right)?;
                        let end = self.emit_jump(OpCode::Jump);
                        self.patch_jump(jump);
                        // Left was true, replace with true
                        self.chunk.emit(OpCode::Pop); // pop the true left
                        let t = self.chunk.add_constant(Value::Bool(true));
                        self.chunk.emit(OpCode::Const);
                        self.chunk.emit_u16(t);
                        self.patch_jump(end);
                        return Ok(());
                    }
                    _ => {}
                }

                self.compile_expr(left)?;
                self.compile_expr(right)?;
                match op {
                    BinOp::Add => self.chunk.emit(OpCode::Add),
                    BinOp::Sub => self.chunk.emit(OpCode::Sub),
                    BinOp::Mul => self.chunk.emit(OpCode::Mul),
                    BinOp::Div => self.chunk.emit(OpCode::Div),
                    BinOp::Mod => self.chunk.emit(OpCode::Mod),
                    BinOp::Pow => self.chunk.emit(OpCode::Pow),
                    BinOp::Eq => self.chunk.emit(OpCode::Eq),
                    BinOp::Ne => self.chunk.emit(OpCode::Ne),
                    BinOp::Lt => self.chunk.emit(OpCode::Lt),
                    BinOp::Gt => self.chunk.emit(OpCode::Gt),
                    BinOp::Le => self.chunk.emit(OpCode::Le),
                    BinOp::Ge => self.chunk.emit(OpCode::Ge),
                    BinOp::BitAnd => self.chunk.emit(OpCode::BitAnd),
                    BinOp::BitOr => self.chunk.emit(OpCode::BitOr),
                    BinOp::BitXor => self.chunk.emit(OpCode::BitXor),
                    BinOp::Shl => self.chunk.emit(OpCode::Shl),
                    BinOp::Shr => self.chunk.emit(OpCode::Shr),
                    BinOp::And | BinOp::Or => unreachable!(),
                }
            }
            Expr::UnaryOp { op, operand } => {
                self.compile_expr(operand)?;
                match op {
                    UnaryOp::Neg => self.chunk.emit(OpCode::Neg),
                    UnaryOp::Not => self.chunk.emit(OpCode::Not),
                    UnaryOp::BitNot => self.chunk.emit(OpCode::BitNot),
                }
            }
            Expr::Call { name, args } => {
                // Compile arguments first
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let argc = args.len() as u8;
                // Check for built-in single-arg functions
                match name.as_str() {
                    "len" => {
                        self.chunk.emit(OpCode::Len);
                    }
                    "type" => {
                        self.chunk.emit(OpCode::TypeOf);
                    }
                    "str" => {
                        self.chunk.emit(OpCode::Str);
                    }
                    "int" => {
                        self.chunk.emit(OpCode::Int);
                    }
                    "sqrt" => {
                        self.chunk.emit(OpCode::Sqrt);
                    }
                    "abs" => {
                        self.chunk.emit(OpCode::Abs);
                    }
                    "floor" => {
                        self.chunk.emit(OpCode::Floor);
                    }
                    "min" => {
                        self.chunk.emit(OpCode::Min);
                    }
                    "max" => {
                        self.chunk.emit(OpCode::Max);
                    }
                    "push" => {
                        self.chunk.emit(OpCode::Push);
                    }
                    _ => {
                        // User-defined or graph-op function call
                        let name_idx = self.chunk.add_constant(Value::String(name.clone()));
                        self.chunk.emit(OpCode::Call);
                        self.chunk.emit_u16(name_idx);
                        self.chunk.emit_byte(argc);
                    }
                }
            }
            Expr::ArrayLit(elems) => {
                for e in elems {
                    self.compile_expr(e)?;
                }
                self.chunk.emit(OpCode::MakeArray);
                self.chunk.emit_u16(elems.len() as u16);
            }
            Expr::Index { array, index } => {
                self.compile_expr(array)?;
                self.compile_expr(index)?;
                self.chunk.emit(OpCode::Index);
            }
            Expr::DictLit(entries) => {
                // Push key-value pairs: key (string const), then value expr
                for (key, val_expr) in entries {
                    let key_idx = self.chunk.add_constant(Value::String(key.clone()));
                    self.chunk.emit(OpCode::Const);
                    self.chunk.emit_u16(key_idx);
                    self.compile_expr(val_expr)?;
                }
                self.chunk.emit(OpCode::MakeDict);
                self.chunk.emit_u16(entries.len() as u16);
            }
        }
        Ok(())
    }
}

// ─── Bytecode VM ────────────────────────────────────────────────────────────

/// Stack-based bytecode virtual machine.
pub struct BytecodeVm {
    stack: Vec<Value>,
    globals: HashMap<String, Value>,
    output: Vec<String>,
}

impl BytecodeVm {
    fn new() -> Self {
        Self {
            stack: Vec::with_capacity(256),
            globals: HashMap::new(),
            output: Vec::new(),
        }
    }

    /// Execute the main chunk, returning the final value and captured output.
    fn run(
        chunk: &Chunk,
        functions: &HashMap<String, CompiledFunction>,
    ) -> Result<(Value, Vec<String>), VmError> {
        let mut vm = BytecodeVm::new();
        vm.execute(chunk, functions)?;
        let result = vm.stack.last().cloned().unwrap_or(Value::Null);
        Ok((result, vm.output))
    }

    fn execute(
        &mut self,
        chunk: &Chunk,
        functions: &HashMap<String, CompiledFunction>,
    ) -> Result<(), VmError> {
        let mut ip: usize = 0;

        macro_rules! read_u16 {
            () => {{
                let hi = chunk.code[ip] as u16;
                let lo = chunk.code[ip + 1] as u16;
                ip += 2;
                (hi << 8) | lo
            }};
        }

        macro_rules! read_u8 {
            () => {{
                let v = chunk.code[ip];
                ip += 1;
                v
            }};
        }

        loop {
            if ip >= chunk.code.len() {
                break;
            }

            let op_byte = chunk.code[ip];
            ip += 1;

            let op = OpCode::from_u8(op_byte).ok_or_else(|| {
                VmError::RuntimeError(format!("invalid opcode: {}", op_byte))
            })?;

            match op {
                OpCode::Const => {
                    let idx = read_u16!() as usize;
                    self.stack.push(chunk.constants[idx].clone());
                }
                OpCode::Pop => {
                    self.stack.pop();
                }
                OpCode::LoadLocal => {
                    let slot = read_u8!() as usize;
                    self.stack.push(self.stack[slot].clone());
                }
                OpCode::StoreLocal => {
                    let slot = read_u8!() as usize;
                    let val = self.stack.last().cloned().unwrap_or(Value::Null);
                    self.stack[slot] = val;
                    // StoreLocal does NOT pop — the Assign statement handler
                    // does not emit an extra Pop because the value is consumed
                    // by being stored. But we must pop here to match the
                    // compile_stmt for Assign which does not add a local.
                    self.stack.pop();
                }
                OpCode::LoadGlobal => {
                    let idx = read_u16!() as usize;
                    let name = match &chunk.constants[idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid global name constant".into(),
                            ))
                        }
                    };
                    let val = self
                        .globals
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| VmError::UndefinedVariable(name))?;
                    self.stack.push(val);
                }
                OpCode::StoreGlobal => {
                    let idx = read_u16!() as usize;
                    let name = match &chunk.constants[idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid global name constant".into(),
                            ))
                        }
                    };
                    let val = self.stack.pop().unwrap_or(Value::Null);
                    self.globals.insert(name, val);
                }

                // Arithmetic
                OpCode::Add => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    // String concatenation
                    if matches!(&a, Value::String(_)) || matches!(&b, Value::String(_)) {
                        self.stack
                            .push(Value::String(format!("{}{}", a, b)));
                    } else {
                        self.stack
                            .push(Value::Number(a.as_number()? + b.as_number()?));
                    }
                }
                OpCode::Sub => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a - b));
                }
                OpCode::Mul => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a * b));
                }
                OpCode::Div => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    if b == 0.0 {
                        return Err(VmError::DivisionByZero);
                    }
                    self.stack.push(Value::Number(a / b));
                }
                OpCode::Mod => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    if b == 0.0 {
                        return Err(VmError::DivisionByZero);
                    }
                    self.stack.push(Value::Number(a % b));
                }
                OpCode::Pow => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.powf(b)));
                }
                OpCode::Neg => {
                    let v = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(-v));
                }

                // Bitwise
                OpCode::BitAnd => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a & b) as f64));
                }
                OpCode::BitOr => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a | b) as f64));
                }
                OpCode::BitXor => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a ^ b) as f64));
                }
                OpCode::BitNot => {
                    let v = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((!v) as f64));
                }
                OpCode::Shl => {
                    let b = self.stack.pop().unwrap().as_number()? as u32;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a << b) as f64));
                }
                OpCode::Shr => {
                    let b = self.stack.pop().unwrap().as_number()? as u32;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a >> b) as f64));
                }

                // Comparison
                OpCode::Eq => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(Value::Bool(a == b));
                }
                OpCode::Ne => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(Value::Bool(a != b));
                }
                OpCode::Lt => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a < b));
                }
                OpCode::Gt => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a > b));
                }
                OpCode::Le => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a <= b));
                }
                OpCode::Ge => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a >= b));
                }

                // Logic
                OpCode::And => {
                    let b = self.stack.pop().unwrap().as_bool()?;
                    let a = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(a && b));
                }
                OpCode::Or => {
                    let b = self.stack.pop().unwrap().as_bool()?;
                    let a = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(a || b));
                }
                OpCode::Not => {
                    let v = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(!v));
                }

                // Control flow
                OpCode::Jump => {
                    let offset = read_u16!() as usize;
                    ip += offset;
                }
                OpCode::JumpIfFalse => {
                    let offset = read_u16!() as usize;
                    let val = self.stack.pop().unwrap();
                    if !val.as_bool()? {
                        ip += offset;
                    }
                }
                OpCode::JumpIfTrue => {
                    let offset = read_u16!() as usize;
                    let val = self.stack.pop().unwrap();
                    if val.as_bool()? {
                        ip += offset;
                    }
                }
                OpCode::Loop => {
                    let offset = read_u16!() as usize;
                    ip -= offset;
                }

                // Functions
                OpCode::Call => {
                    let name_idx = read_u16!() as usize;
                    let argc = read_u8!() as usize;

                    let name = match &chunk.constants[name_idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid function name constant".into(),
                            ))
                        }
                    };

                    // Collect args from the stack
                    let stack_len = self.stack.len();
                    if stack_len < argc {
                        return Err(VmError::RuntimeError(format!(
                            "stack underflow calling {}",
                            name
                        )));
                    }
                    let args: Vec<Value> =
                        self.stack.drain(stack_len - argc..).collect();

                    // User-defined functions first (so users can override builtins)
                    if let Some(func) = functions.get(&name) {
                        if args.len() != func.params.len() {
                            return Err(VmError::ArityMismatch {
                                expected: func.params.len(),
                                got: args.len(),
                            });
                        }
                        let result =
                            self.execute_function(func, &args, functions)?;
                        self.stack.push(result);
                    } else if let Some(result) =
                        crate::graph_ops::try_call_graph_op(&name, &args)?
                    {
                        self.stack.push(result);
                    } else {
                        return Err(VmError::UndefinedFunction(name));
                    }
                }
                OpCode::Return => {
                    // Return handled by execute_function — in the main chunk
                    // this is a no-op (the value is on the stack).
                    break;
                }

                // Built-in functions
                OpCode::Print => {
                    let v = self.stack.pop().unwrap_or(Value::Null);
                    self.output.push(format!("{}", v));
                }
                OpCode::Len => {
                    let v = self.stack.pop().unwrap();
                    match &v {
                        Value::Array(a) => {
                            self.stack.push(Value::Number(a.len() as f64))
                        }
                        Value::String(s) => {
                            self.stack.push(Value::Number(s.len() as f64))
                        }
                        Value::Dict(entries) => {
                            self.stack
                                .push(Value::Number(entries.len() as f64))
                        }
                        other => {
                            return Err(VmError::TypeError(format!(
                                "len() expects string, array, or dict, got {}",
                                other.type_name_static()
                            )))
                        }
                    }
                }
                OpCode::TypeOf => {
                    let v = self.stack.pop().unwrap();
                    self.stack.push(Value::String(
                        v.type_name_static().to_string(),
                    ));
                }
                OpCode::Str => {
                    let v = self.stack.pop().unwrap();
                    self.stack.push(Value::String(format!("{}", v)));
                }
                OpCode::Int => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.floor()));
                }
                OpCode::Sqrt => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.sqrt()));
                }
                OpCode::Abs => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.abs()));
                }
                OpCode::Floor => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.floor()));
                }
                OpCode::Min => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.min(b)));
                }
                OpCode::Max => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.max(b)));
                }
                OpCode::Push => {
                    let val = self.stack.pop().unwrap().as_number()?;
                    let arr_val = self.stack.pop().unwrap();
                    let mut arr = arr_val.as_array()?.clone();
                    arr.push(val);
                    self.stack.push(Value::Array(arr));
                }

                // Arrays
                OpCode::MakeArray => {
                    let count = read_u16!() as usize;
                    let stack_len = self.stack.len();
                    let elems: Vec<f64> = self
                        .stack
                        .drain(stack_len - count..)
                        .map(|v| v.as_number())
                        .collect::<Result<_, _>>()?;
                    self.stack.push(Value::Array(elems));
                }

                // Index
                OpCode::Index => {
                    let idx_val = self.stack.pop().unwrap();
                    let arr_val = self.stack.pop().unwrap();
                    // Handle dict indexing with string keys before as_number()
                    if let Value::Dict(entries) = &arr_val {
                        let key = match &idx_val {
                            Value::String(s) => s.clone(),
                            other => {
                                return Err(VmError::TypeError(format!(
                                    "dict key must be string, got {}",
                                    other.type_name_static()
                                )))
                            }
                        };
                        let found = entries
                            .iter()
                            .find(|(k, _)| k == &key)
                            .map(|(_, v)| v.clone());
                        match found {
                            Some(v) => self.stack.push(v),
                            None => {
                                return Err(VmError::RuntimeError(
                                    format!(
                                        "key '{}' not found in dict",
                                        key
                                    ),
                                ))
                            }
                        }
                    } else {
                        let idx_num = idx_val.as_number()?;
                        let idx = idx_num as usize;
                        match &arr_val {
                            Value::Array(arr) => {
                                if idx >= arr.len() {
                                    return Err(VmError::IndexOutOfBounds {
                                        index: idx_num as isize,
                                        len: arr.len(),
                                    });
                                }
                                self.stack.push(Value::Number(arr[idx]));
                            }
                            Value::String(s) => {
                                let chars: Vec<char> = s.chars().collect();
                                if idx >= chars.len() {
                                    return Err(VmError::IndexOutOfBounds {
                                        index: idx_num as isize,
                                        len: chars.len(),
                                    });
                                }
                                self.stack
                                    .push(Value::String(chars[idx].to_string()));
                            }
                            other => {
                                return Err(VmError::TypeError(format!(
                                    "cannot index into {}",
                                    other.type_name_static()
                                )))
                            }
                        }
                    }
                }

                // Dicts
                OpCode::MakeDict => {
                    let count = read_u16!() as usize;
                    let pairs_count = count * 2; // key + value per entry
                    let stack_len = self.stack.len();
                    let items: Vec<Value> =
                        self.stack.drain(stack_len - pairs_count..).collect();
                    let mut entries = Vec::with_capacity(count);
                    for pair in items.chunks(2) {
                        let key = match &pair[0] {
                            Value::String(s) => s.clone(),
                            _ => {
                                return Err(VmError::RuntimeError(
                                    "dict key must be string".into(),
                                ))
                            }
                        };
                        entries.push((key, pair[1].clone()));
                    }
                    self.stack.push(Value::Dict(entries));
                }

                OpCode::Halt => break,
            }
        }

        Ok(())
    }

    /// Execute a compiled function with given arguments.
    /// This runs the function's chunk in isolation, sharing globals and output.
    fn execute_function(
        &mut self,
        func: &CompiledFunction,
        args: &[Value],
        all_functions: &HashMap<String, CompiledFunction>,
    ) -> Result<Value, VmError> {
        // Save current stack
        let saved_stack_len = self.stack.len();

        // Push args as locals (they become slot 0, 1, 2, ...)
        for arg in args {
            self.stack.push(arg.clone());
        }

        // Execute the function's chunk
        let mut ip: usize = 0;
        let chunk = &func.chunk;
        let base = saved_stack_len;

        // The function's local slot 0 = base+0, slot 1 = base+1, etc.
        // We need a small inner interpreter that uses base-relative addressing.

        macro_rules! read_u16 {
            () => {{
                let hi = chunk.code[ip] as u16;
                let lo = chunk.code[ip + 1] as u16;
                ip += 2;
                (hi << 8) | lo
            }};
        }

        macro_rules! read_u8 {
            () => {{
                let v = chunk.code[ip];
                ip += 1;
                v
            }};
        }

        loop {
            if ip >= chunk.code.len() {
                break;
            }

            let op_byte = chunk.code[ip];
            ip += 1;

            let op = OpCode::from_u8(op_byte).ok_or_else(|| {
                VmError::RuntimeError(format!("invalid opcode: {}", op_byte))
            })?;

            match op {
                OpCode::Const => {
                    let idx = read_u16!() as usize;
                    self.stack.push(chunk.constants[idx].clone());
                }
                OpCode::Pop => {
                    self.stack.pop();
                }
                OpCode::LoadLocal => {
                    let slot = read_u8!() as usize;
                    self.stack.push(self.stack[base + slot].clone());
                }
                OpCode::StoreLocal => {
                    let slot = read_u8!() as usize;
                    let val =
                        self.stack.last().cloned().unwrap_or(Value::Null);
                    self.stack[base + slot] = val;
                    self.stack.pop();
                }
                OpCode::LoadGlobal => {
                    let idx = read_u16!() as usize;
                    let name = match &chunk.constants[idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid global name constant".into(),
                            ))
                        }
                    };
                    let val = self
                        .globals
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| VmError::UndefinedVariable(name))?;
                    self.stack.push(val);
                }
                OpCode::StoreGlobal => {
                    let idx = read_u16!() as usize;
                    let name = match &chunk.constants[idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid global name constant".into(),
                            ))
                        }
                    };
                    let val = self.stack.pop().unwrap_or(Value::Null);
                    self.globals.insert(name, val);
                }

                // Arithmetic — same as main loop
                OpCode::Add => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    if matches!(&a, Value::String(_))
                        || matches!(&b, Value::String(_))
                    {
                        self.stack
                            .push(Value::String(format!("{}{}", a, b)));
                    } else {
                        self.stack.push(Value::Number(
                            a.as_number()? + b.as_number()?,
                        ));
                    }
                }
                OpCode::Sub => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a - b));
                }
                OpCode::Mul => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a * b));
                }
                OpCode::Div => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    if b == 0.0 {
                        return Err(VmError::DivisionByZero);
                    }
                    self.stack.push(Value::Number(a / b));
                }
                OpCode::Mod => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    if b == 0.0 {
                        return Err(VmError::DivisionByZero);
                    }
                    self.stack.push(Value::Number(a % b));
                }
                OpCode::Pow => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.powf(b)));
                }
                OpCode::Neg => {
                    let v = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(-v));
                }
                OpCode::BitAnd => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a & b) as f64));
                }
                OpCode::BitOr => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a | b) as f64));
                }
                OpCode::BitXor => {
                    let b = self.stack.pop().unwrap().as_number()? as i64;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a ^ b) as f64));
                }
                OpCode::BitNot => {
                    let v = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((!v) as f64));
                }
                OpCode::Shl => {
                    let b = self.stack.pop().unwrap().as_number()? as u32;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a << b) as f64));
                }
                OpCode::Shr => {
                    let b = self.stack.pop().unwrap().as_number()? as u32;
                    let a = self.stack.pop().unwrap().as_number()? as i64;
                    self.stack.push(Value::Number((a >> b) as f64));
                }
                OpCode::Eq => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(Value::Bool(a == b));
                }
                OpCode::Ne => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(Value::Bool(a != b));
                }
                OpCode::Lt => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a < b));
                }
                OpCode::Gt => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a > b));
                }
                OpCode::Le => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a <= b));
                }
                OpCode::Ge => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Bool(a >= b));
                }
                OpCode::And => {
                    let b = self.stack.pop().unwrap().as_bool()?;
                    let a = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(a && b));
                }
                OpCode::Or => {
                    let b = self.stack.pop().unwrap().as_bool()?;
                    let a = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(a || b));
                }
                OpCode::Not => {
                    let v = self.stack.pop().unwrap().as_bool()?;
                    self.stack.push(Value::Bool(!v));
                }

                // Control flow
                OpCode::Jump => {
                    let offset = read_u16!() as usize;
                    ip += offset;
                }
                OpCode::JumpIfFalse => {
                    let offset = read_u16!() as usize;
                    let val = self.stack.pop().unwrap();
                    if !val.as_bool()? {
                        ip += offset;
                    }
                }
                OpCode::JumpIfTrue => {
                    let offset = read_u16!() as usize;
                    let val = self.stack.pop().unwrap();
                    if val.as_bool()? {
                        ip += offset;
                    }
                }
                OpCode::Loop => {
                    let offset = read_u16!() as usize;
                    ip -= offset;
                }

                // Functions
                OpCode::Call => {
                    let name_idx = read_u16!() as usize;
                    let argc = read_u8!() as usize;

                    let name = match &chunk.constants[name_idx] {
                        Value::String(s) => s.clone(),
                        _ => {
                            return Err(VmError::RuntimeError(
                                "invalid function name constant".into(),
                            ))
                        }
                    };

                    let stack_len = self.stack.len();
                    let call_args: Vec<Value> =
                        self.stack.drain(stack_len - argc..).collect();

                    // User-defined first (so users can override builtins)
                    if let Some(f) = all_functions.get(&name) {
                        if call_args.len() != f.params.len() {
                            return Err(VmError::ArityMismatch {
                                expected: f.params.len(),
                                got: call_args.len(),
                            });
                        }
                        let result = self.execute_function(
                            f,
                            &call_args,
                            all_functions,
                        )?;
                        self.stack.push(result);
                    } else if let Some(result) =
                        crate::graph_ops::try_call_graph_op(
                            &name, &call_args,
                        )?
                    {
                        self.stack.push(result);
                    } else {
                        return Err(VmError::UndefinedFunction(name));
                    }
                }

                OpCode::Return => {
                    // Pop the return value
                    let ret_val =
                        self.stack.pop().unwrap_or(Value::Null);
                    // Clean up function locals from the stack
                    self.stack.truncate(saved_stack_len);
                    return Ok(ret_val);
                }

                // Built-ins
                OpCode::Print => {
                    let v = self.stack.pop().unwrap_or(Value::Null);
                    self.output.push(format!("{}", v));
                }
                OpCode::Len => {
                    let v = self.stack.pop().unwrap();
                    match &v {
                        Value::Array(a) => {
                            self.stack.push(Value::Number(a.len() as f64))
                        }
                        Value::String(s) => {
                            self.stack.push(Value::Number(s.len() as f64))
                        }
                        Value::Dict(entries) => {
                            self.stack
                                .push(Value::Number(entries.len() as f64))
                        }
                        other => {
                            return Err(VmError::TypeError(format!(
                                "len() expects string, array, or dict, got {}",
                                other.type_name_static()
                            )))
                        }
                    }
                }
                OpCode::TypeOf => {
                    let v = self.stack.pop().unwrap();
                    self.stack.push(Value::String(
                        v.type_name_static().to_string(),
                    ));
                }
                OpCode::Str => {
                    let v = self.stack.pop().unwrap();
                    self.stack.push(Value::String(format!("{}", v)));
                }
                OpCode::Int => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.floor()));
                }
                OpCode::Sqrt => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.sqrt()));
                }
                OpCode::Abs => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.abs()));
                }
                OpCode::Floor => {
                    let n = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(n.floor()));
                }
                OpCode::Min => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.min(b)));
                }
                OpCode::Max => {
                    let b = self.stack.pop().unwrap().as_number()?;
                    let a = self.stack.pop().unwrap().as_number()?;
                    self.stack.push(Value::Number(a.max(b)));
                }
                OpCode::Push => {
                    let val = self.stack.pop().unwrap().as_number()?;
                    let arr_val = self.stack.pop().unwrap();
                    let mut arr = arr_val.as_array()?.clone();
                    arr.push(val);
                    self.stack.push(Value::Array(arr));
                }

                OpCode::MakeArray => {
                    let count = read_u16!() as usize;
                    let stack_len = self.stack.len();
                    let elems: Vec<f64> = self
                        .stack
                        .drain(stack_len - count..)
                        .map(|v| v.as_number())
                        .collect::<Result<_, _>>()?;
                    self.stack.push(Value::Array(elems));
                }

                OpCode::Index => {
                    let idx_val = self.stack.pop().unwrap();
                    let arr_val = self.stack.pop().unwrap();
                    // Handle dict indexing with string keys before as_number()
                    if let Value::Dict(entries) = &arr_val {
                        let key = match &idx_val {
                            Value::String(s) => s.clone(),
                            other => {
                                return Err(VmError::TypeError(format!(
                                    "dict key must be string, got {}",
                                    other.type_name_static()
                                )))
                            }
                        };
                        let found = entries
                            .iter()
                            .find(|(k, _)| k == &key)
                            .map(|(_, v)| v.clone());
                        match found {
                            Some(v) => self.stack.push(v),
                            None => {
                                return Err(VmError::RuntimeError(
                                    format!(
                                        "key '{}' not found in dict",
                                        key
                                    ),
                                ))
                            }
                        }
                    } else {
                        let idx_num = idx_val.as_number()?;
                        let idx = idx_num as usize;
                        match &arr_val {
                            Value::Array(arr) => {
                                if idx >= arr.len() {
                                    return Err(VmError::IndexOutOfBounds {
                                        index: idx_num as isize,
                                        len: arr.len(),
                                    });
                                }
                                self.stack.push(Value::Number(arr[idx]));
                            }
                            Value::String(s) => {
                                let chars: Vec<char> = s.chars().collect();
                                if idx >= chars.len() {
                                    return Err(VmError::IndexOutOfBounds {
                                        index: idx_num as isize,
                                        len: chars.len(),
                                    });
                                }
                                self.stack.push(Value::String(
                                    chars[idx].to_string(),
                                ));
                            }
                            other => {
                                return Err(VmError::TypeError(format!(
                                    "cannot index into {}",
                                    other.type_name_static()
                                )))
                            }
                        }
                    }
                }

                OpCode::MakeDict => {
                    let count = read_u16!() as usize;
                    let pairs_count = count * 2;
                    let stack_len = self.stack.len();
                    let items: Vec<Value> =
                        self.stack.drain(stack_len - pairs_count..).collect();
                    let mut entries = Vec::with_capacity(count);
                    for pair in items.chunks(2) {
                        let key = match &pair[0] {
                            Value::String(s) => s.clone(),
                            _ => {
                                return Err(VmError::RuntimeError(
                                    "dict key must be string".into(),
                                ))
                            }
                        };
                        entries.push((key, pair[1].clone()));
                    }
                    self.stack.push(Value::Dict(entries));
                }

                OpCode::Halt => break,
            }
        }

        // If we reached here without a Return, return Null
        let ret = self.stack.pop().unwrap_or(Value::Null);
        self.stack.truncate(saved_stack_len);
        Ok(ret)
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Compile and execute a QLANG script using the bytecode VM.
/// Returns the final value and captured output lines.
pub fn run_bytecode(source: &str) -> Result<(Value, Vec<String>), VmError> {
    let tokens = tokenize(source)?;
    let mut parser = Parser::new(tokens);
    let stmts = parser.parse_program()?;
    type_check(&stmts)?;
    let (chunk, functions) = Compiler::compile(&stmts)?;
    BytecodeVm::run(&chunk, &functions)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> (Value, Vec<String>) {
        run_bytecode(src).expect("bytecode script should succeed")
    }

    fn run_output(src: &str) -> Vec<String> {
        run(src).1
    }

    #[test]
    fn arithmetic_basic() {
        let out = run_output("print(2 + 3 * 4)");
        assert_eq!(out, vec!["14"]);
    }

    #[test]
    fn arithmetic_precedence() {
        let out = run_output(r#"
            let a = 2 + 3 * 4
            let b = (2 + 3) * 4
            print(a)
            print(b)
        "#);
        assert_eq!(out, vec!["14", "20"]);
    }

    #[test]
    fn variable_binding() {
        let out = run_output(r#"
            let x = 5
            let y = x + 3
            print(y)
        "#);
        assert_eq!(out, vec!["8"]);
    }

    #[test]
    fn variable_reassignment() {
        let out = run_output(r#"
            let x = 10
            x = x + 5
            print(x)
        "#);
        assert_eq!(out, vec!["15"]);
    }

    #[test]
    fn if_else() {
        let out = run_output(r#"
            let x = 10
            if x > 5 {
                print("big")
            } else {
                print("small")
            }
        "#);
        assert_eq!(out, vec!["big"]);
    }

    #[test]
    fn if_else_false() {
        let out = run_output(r#"
            let x = 2
            if x > 5 {
                print("big")
            } else {
                print("small")
            }
        "#);
        assert_eq!(out, vec!["small"]);
    }

    #[test]
    fn while_loop() {
        let out = run_output(r#"
            let x = 0
            while x < 5 {
                x = x + 1
            }
            print(x)
        "#);
        assert_eq!(out, vec!["5"]);
    }

    #[test]
    fn for_loop() {
        let out = run_output(r#"
            let sum = 0
            for i in 0..5 {
                sum = sum + i
            }
            print(sum)
        "#);
        assert_eq!(out, vec!["10"]); // 0+1+2+3+4 = 10
    }

    #[test]
    fn function_basic() {
        let out = run_output(r#"
            fn add(a, b) {
                return a + b
            }
            print(add(3, 4))
        "#);
        assert_eq!(out, vec!["7"]);
    }

    #[test]
    fn function_recursive_fibonacci() {
        let out = run_output(r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            print(fib(10))
        "#);
        assert_eq!(out, vec!["55"]);
    }

    #[test]
    fn string_concatenation() {
        let out = run_output(r#"
            let greeting = "hello" + " " + "world"
            print(greeting)
        "#);
        assert_eq!(out, vec!["hello world"]);
    }

    #[test]
    fn array_basic() {
        let out = run_output(r#"
            let arr = [10, 20, 30]
            print(arr[1])
            print(len(arr))
        "#);
        assert_eq!(out, vec!["20", "3"]);
    }

    #[test]
    fn dict_basic() {
        let out = run_output(r#"
            let d = {"name": "qlang", "version": 3}
            print(d["name"])
        "#);
        assert_eq!(out, vec!["qlang"]);
    }

    #[test]
    fn builtin_functions() {
        let out = run_output(r#"
            print(abs(-5))
            print(sqrt(16))
            print(min(3, 7))
            print(max(3, 7))
            print(floor(3.7))
        "#);
        assert_eq!(out, vec!["5", "4", "3", "7", "3"]);
    }

    #[test]
    fn boolean_logic() {
        let out = run_output(r#"
            print(true and false)
            print(true or false)
            print(not true)
        "#);
        assert_eq!(out, vec!["false", "true", "false"]);
    }

    #[test]
    fn comparison_ops() {
        let out = run_output(r#"
            print(3 == 3)
            print(3 != 4)
            print(3 < 4)
            print(4 > 3)
            print(3 <= 3)
            print(4 >= 5)
        "#);
        assert_eq!(out, vec!["true", "true", "true", "true", "true", "false"]);
    }

    #[test]
    fn bitwise_ops() {
        let out = run_output(r#"
            print(5 & 3)
            print(5 | 3)
            print(5 ^ 3)
            print(1 << 3)
            print(8 >> 2)
        "#);
        assert_eq!(out, vec!["1", "7", "6", "8", "2"]);
    }

    #[test]
    fn power_operator() {
        let out = run_output(r#"
            print(2 ** 10)
        "#);
        assert_eq!(out, vec!["1024"]);
    }

    #[test]
    fn nested_function_calls() {
        let out = run_output(r#"
            fn square(x) {
                return x * x
            }
            fn sum_squares(a, b) {
                return square(a) + square(b)
            }
            print(sum_squares(3, 4))
        "#);
        assert_eq!(out, vec!["25"]); // 9 + 16
    }

    #[test]
    fn compound_assignment() {
        let out = run_output(r#"
            let x = 10
            x += 5
            print(x)
            x -= 3
            print(x)
            x *= 2
            print(x)
        "#);
        assert_eq!(out, vec!["15", "12", "24"]);
    }

    #[test]
    fn type_function() {
        let out = run_output(r#"
            print(type(42))
            print(type("hello"))
            print(type(true))
            print(type([1, 2, 3]))
        "#);
        assert_eq!(out, vec!["number", "string", "bool", "array"]);
    }

    #[test]
    fn fibonacci_benchmark() {
        // Test that bytecode VM produces correct results for fib(20)
        let out = run_output(r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            print(fib(20))
        "#);
        assert_eq!(out, vec!["6765"]);
    }

    #[test]
    fn speed_comparison_fibonacci() {
        // Run fib(25) on both interpreters and compare speed
        let source = r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            print(fib(25))
        "#;

        // Bytecode VM
        let bc_start = std::time::Instant::now();
        let (_, bc_out) = run_bytecode(source).unwrap();
        let bc_elapsed = bc_start.elapsed();

        // Tree-walking interpreter
        let tw_start = std::time::Instant::now();
        let (_, tw_out) = crate::vm::run_qlang_script(source).unwrap();
        let tw_elapsed = tw_start.elapsed();

        // Both should produce the same result
        assert_eq!(bc_out, vec!["75025"]);
        assert_eq!(tw_out, vec!["75025"]);

        // Print timing for information (visible in test output with --nocapture)
        eprintln!(
            "fib(25) bytecode VM:    {:?}",
            bc_elapsed
        );
        eprintln!(
            "fib(25) tree-walking:   {:?}",
            tw_elapsed
        );
        eprintln!(
            "speedup: {:.1}x",
            tw_elapsed.as_secs_f64() / bc_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn loop_benchmark() {
        // Run a tight loop on both to compare speed
        let source = r#"
            let sum = 0
            let i = 0
            while i < 100000 {
                sum = sum + i
                i = i + 1
            }
            print(sum)
        "#;

        // Bytecode VM
        let bc_start = std::time::Instant::now();
        let (_, bc_out) = run_bytecode(source).unwrap();
        let bc_elapsed = bc_start.elapsed();

        // Tree-walking interpreter
        let tw_start = std::time::Instant::now();
        let (_, tw_out) = crate::vm::run_qlang_script(source).unwrap();
        let tw_elapsed = tw_start.elapsed();

        // Both should produce the same result: sum of 0..99999
        let expected = "4999950000";
        assert_eq!(bc_out, vec![expected]);
        assert_eq!(tw_out, vec![expected]);

        eprintln!(
            "loop(100k) bytecode VM:  {:?}",
            bc_elapsed
        );
        eprintln!(
            "loop(100k) tree-walking: {:?}",
            tw_elapsed
        );
        eprintln!(
            "speedup: {:.1}x",
            tw_elapsed.as_secs_f64() / bc_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn str_function() {
        let out = run_output(r#"
            let n = 42
            let s = str(n)
            print(s)
            print(type(s))
        "#);
        assert_eq!(out, vec!["42", "string"]);
    }

    #[test]
    fn nested_if() {
        let out = run_output(r#"
            let x = 15
            if x > 10 {
                if x > 20 {
                    print("very big")
                } else {
                    print("medium")
                }
            } else {
                print("small")
            }
        "#);
        assert_eq!(out, vec!["medium"]);
    }

    #[test]
    fn while_with_return_in_function() {
        let out = run_output(r#"
            fn find_first_above(threshold) {
                let i = 0
                while i < 100 {
                    if i * i > threshold {
                        return i
                    }
                    i = i + 1
                }
                return -1
            }
            print(find_first_above(50))
        "#);
        assert_eq!(out, vec!["8"]); // 8*8=64 > 50
    }

    #[test]
    fn modulo_operator() {
        let out = run_output(r#"
            print(10 % 3)
            print(17 % 5)
        "#);
        assert_eq!(out, vec!["1", "2"]);
    }

    #[test]
    fn negation() {
        let out = run_output(r#"
            let x = 5
            print(-x)
            print(-(-x))
        "#);
        assert_eq!(out, vec!["-5", "5"]);
    }

    #[test]
    fn elif_chain() {
        let out = run_output(r#"
            let x = 5
            if x > 10 {
                print("big")
            } else if x > 3 {
                print("medium")
            } else {
                print("small")
            }
        "#);
        assert_eq!(out, vec!["medium"]);
    }

    #[test]
    fn for_loop_with_function() {
        let out = run_output(r#"
            fn factorial(n) {
                let result = 1
                for i in 1..n {
                    result = result * i
                }
                return result
            }
            print(factorial(6))
        "#);
        // for i in 1..6 gives i = 1,2,3,4,5 => 1*2*3*4*5 = 120
        assert_eq!(out, vec!["120"]);
    }

    #[test]
    fn string_number_concat() {
        let out = run_output(r#"
            let result = "value: " + 42
            print(result)
        "#);
        assert_eq!(out, vec!["value: 42"]);
    }
}
