//! LLVM JIT Compilation for QLANG Script Code.
//!
//! Compiles numeric QLANG scripts (variables, arithmetic, if/else, loops,
//! functions, print) directly to native machine code via LLVM.
//! Result: same performance as hand-written C.
//!
//! This is "Tier 1" in the execution hierarchy:
//!   1. LLVM JIT  — native speed, numeric-only scripts
//!   2. Bytecode VM — fast, all features except graphs
//!   3. Tree-walking interpreter — all features including graphs
//!
//! Features NOT supported (fall back to VM):
//!   strings, arrays, dicts, imports, index expressions, built-in functions

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module;
use inkwell::types::FloatType;
use inkwell::values::{FloatValue, FunctionValue, PointerValue};
use inkwell::FloatPredicate;
use inkwell::OptimizationLevel;

use std::collections::HashMap;
use std::sync::Mutex;

use qlang_runtime::vm::{BinOp, Expr, Param, Stmt, UnaryOp};

// ─── JIT Compiler ────────────────────────────────────────────────────────────

/// LLVM JIT compiler for QLANG numeric scripts.
///
/// Compiles a flat list of statements into a single LLVM function,
/// JIT-compiles it, and executes at native speed.
pub struct ScriptJit<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    /// Variable name → stack alloca pointer
    variables: HashMap<String, PointerValue<'ctx>>,
    /// User-defined function name → LLVM function value
    functions: HashMap<String, FunctionValue<'ctx>>,
    /// Cached f64 type
    f64_type: FloatType<'ctx>,
}

// Callback for capturing print output from JIT code.
// We use a global Mutex to pass data from the JIT-ed code back to Rust.
static PRINT_BUFFER: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Called from JIT code to print a f64 value.
extern "C" fn qlang_jit_print_f64(val: f64) {
    let line = if val == (val as i64) as f64 && val.abs() < 1e15 {
        format!("{}", val as i64)
    } else {
        format!("{val}")
    };
    if let Ok(mut buf) = PRINT_BUFFER.lock() {
        buf.push(line.clone());
    }
    println!("{line}");
}

impl<'ctx> ScriptJit<'ctx> {
    /// Create a new JIT compiler backed by the given LLVM context.
    pub fn new(context: &'ctx Context) -> Result<Self, String> {
        let module = context.create_module("qlang_script");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| format!("failed to create JIT engine: {e}"))?;
        let builder = context.create_builder();
        let f64_type = context.f64_type();

        // Declare our print helper: void qlang_jit_print_f64(double)
        let void_type = context.void_type();
        let print_fn_type = void_type.fn_type(&[f64_type.into()], false);
        let print_fn = module.add_function(
            "qlang_jit_print_f64",
            print_fn_type,
            Some(inkwell::module::Linkage::External),
        );

        // Register the print function with the execution engine
        execution_engine.add_global_mapping(&print_fn, qlang_jit_print_f64 as *const () as usize);

        // Declare llvm.pow.f64 intrinsic for ** operator
        let pow_fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
        module.add_function("llvm.pow.f64", pow_fn_type, None);

        // Declare fmod for % operator
        let fmod_fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
        let fmod_fn = module.add_function(
            "fmod",
            fmod_fn_type,
            Some(inkwell::module::Linkage::External),
        );
        execution_engine.add_global_mapping(&fmod_fn, libc_fmod as *const () as usize);

        Ok(ScriptJit {
            context,
            module,
            builder,
            execution_engine,
            variables: HashMap::new(),
            functions: HashMap::new(),
            f64_type,
        })
    }

    /// Compile a list of statements and JIT-execute them.
    ///
    /// Returns `(last_value, print_output_lines)`.
    pub fn compile_and_run(&mut self, stmts: &[Stmt]) -> Result<(f64, Vec<String>), String> {
        // First pass: compile user-defined functions
        for stmt in stmts {
            if let Stmt::FnDef { name, params, body, .. } = stmt {
                self.compile_function(name, params, body)?;
            }
        }

        // Create main function: double __qlang_main()
        let main_type = self.f64_type.fn_type(&[], false);
        let main_fn = self.module.add_function("__qlang_main", main_type, None);
        let entry = self.context.append_basic_block(main_fn, "entry");
        self.builder.position_at_end(entry);

        // Clear variables for main scope
        self.variables.clear();

        // Compile all statements into main
        let mut last_val = self.f64_type.const_float(0.0);
        for stmt in stmts {
            // Skip FnDef (already compiled above)
            if matches!(stmt, Stmt::FnDef { .. }) {
                continue;
            }
            if let Some(v) = self.compile_stmt(stmt, main_fn)? {
                last_val = v;
            }
        }

        // Return last value
        self.builder.build_return(Some(&last_val)).unwrap();

        // Verify the module
        if let Err(e) = self.module.verify() {
            return Err(format!("LLVM verification failed: {}", e.to_string()));
        }

        // Clear the print buffer
        if let Ok(mut buf) = PRINT_BUFFER.lock() {
            buf.clear();
        }

        // JIT execute
        let result = unsafe {
            let main_jit = self
                .execution_engine
                .get_function::<unsafe extern "C" fn() -> f64>("__qlang_main")
                .map_err(|e| format!("JIT lookup error: {e}"))?;
            main_jit.call()
        };

        // Collect print output
        let output = PRINT_BUFFER.lock().map(|mut buf| {
            let out = buf.clone();
            buf.clear();
            out
        }).unwrap_or_default();

        Ok((result, output))
    }

    /// Compile a user-defined function to LLVM IR.
    fn compile_function(
        &mut self,
        name: &str,
        params: &[Param],
        body: &[Stmt],
    ) -> Result<(), String> {
        // All parameters and return value are f64
        let param_types: Vec<inkwell::types::BasicMetadataTypeEnum> =
            params.iter().map(|_| self.f64_type.into()).collect();
        let fn_type = self.f64_type.fn_type(&param_types, false);
        let function = self.module.add_function(name, fn_type, None);

        // Save current state
        let saved_vars = self.variables.clone();
        let saved_block = self.builder.get_insert_block();

        // Create entry block
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        // Clear variables for this function scope
        self.variables.clear();

        // Allocate parameters
        for (i, param) in params.iter().enumerate() {
            let ptr = self
                .builder
                .build_alloca(self.f64_type, &param.name)
                .unwrap();
            let param_val = function.get_nth_param(i as u32).unwrap().into_float_value();
            self.builder.build_store(ptr, param_val).unwrap();
            self.variables.insert(param.name.clone(), ptr);
        }

        // Compile body
        let mut last_val = self.f64_type.const_float(0.0);
        let mut has_return = false;
        for stmt in body {
            if has_return {
                break;
            }
            if matches!(stmt, Stmt::Return(_)) {
                has_return = true;
            }
            if let Some(v) = self.compile_stmt(stmt, function)? {
                last_val = v;
            }
        }

        // If no explicit return, return last value
        if !has_return {
            self.builder.build_return(Some(&last_val)).unwrap();
        }

        // Restore state
        self.variables = saved_vars;
        if let Some(block) = saved_block {
            self.builder.position_at_end(block);
        }

        // Register the function
        self.functions.insert(name.to_string(), function);

        Ok(())
    }

    /// Compile a single statement. Returns Some(value) if the statement produces
    /// a value (e.g., expression statement), None otherwise.
    fn compile_stmt(
        &mut self,
        stmt: &Stmt,
        function: FunctionValue<'ctx>,
    ) -> Result<Option<FloatValue<'ctx>>, String> {
        match stmt {
            Stmt::Let { name, value, .. } => {
                let val = self.compile_expr(value, function)?;
                let ptr = self.builder.build_alloca(self.f64_type, name).unwrap();
                self.builder.build_store(ptr, val).unwrap();
                self.variables.insert(name.clone(), ptr);
                Ok(None)
            }

            Stmt::Assign { name, value } => {
                let val = self.compile_expr(value, function)?;
                if let Some(ptr) = self.variables.get(name) {
                    self.builder.build_store(*ptr, val).unwrap();
                    Ok(None)
                } else {
                    Err(format!("JIT: undefined variable in assignment: {name}"))
                }
            }

            Stmt::If {
                cond,
                then_body,
                else_body,
            } => {
                let cond_val = self.compile_expr(cond, function)?;
                let zero = self.f64_type.const_float(0.0);
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, cond_val, zero, "ifcond")
                    .unwrap();

                let then_bb = self.context.append_basic_block(function, "then");
                let else_bb = self.context.append_basic_block(function, "else");
                let merge_bb = self.context.append_basic_block(function, "ifmerge");

                self.builder
                    .build_conditional_branch(cmp, then_bb, else_bb)
                    .unwrap();

                // Then block
                self.builder.position_at_end(then_bb);
                for s in then_body {
                    self.compile_stmt(s, function)?;
                }
                // Only branch to merge if current block is not already terminated
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder.build_unconditional_branch(merge_bb).unwrap();
                }

                // Else block
                self.builder.position_at_end(else_bb);
                for s in else_body {
                    self.compile_stmt(s, function)?;
                }
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder.build_unconditional_branch(merge_bb).unwrap();
                }

                // Continue at merge
                self.builder.position_at_end(merge_bb);
                Ok(None)
            }

            Stmt::While { cond, body } => {
                let loop_cond_bb = self.context.append_basic_block(function, "while_cond");
                let loop_body_bb = self.context.append_basic_block(function, "while_body");
                let after_bb = self.context.append_basic_block(function, "while_end");

                // Jump to condition check
                self.builder
                    .build_unconditional_branch(loop_cond_bb)
                    .unwrap();

                // Condition block
                self.builder.position_at_end(loop_cond_bb);
                let cond_val = self.compile_expr(cond, function)?;
                let zero = self.f64_type.const_float(0.0);
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, cond_val, zero, "whilecond")
                    .unwrap();
                self.builder
                    .build_conditional_branch(cmp, loop_body_bb, after_bb)
                    .unwrap();

                // Body block
                self.builder.position_at_end(loop_body_bb);
                for s in body {
                    self.compile_stmt(s, function)?;
                }
                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder
                        .build_unconditional_branch(loop_cond_bb)
                        .unwrap();
                }

                // Continue after loop
                self.builder.position_at_end(after_bb);
                Ok(None)
            }

            Stmt::For {
                var,
                start,
                end,
                body,
            } => {
                // Compile as: let var = start; while var < end { body; var = var + 1 }
                let start_val = self.compile_expr(start, function)?;
                let end_val = self.compile_expr(end, function)?;

                // Allocate loop variable
                let var_ptr = self.builder.build_alloca(self.f64_type, var).unwrap();
                self.builder.build_store(var_ptr, start_val).unwrap();
                self.variables.insert(var.clone(), var_ptr);

                let loop_cond_bb = self.context.append_basic_block(function, "for_cond");
                let loop_body_bb = self.context.append_basic_block(function, "for_body");
                let after_bb = self.context.append_basic_block(function, "for_end");

                self.builder
                    .build_unconditional_branch(loop_cond_bb)
                    .unwrap();

                // Condition: var < end
                self.builder.position_at_end(loop_cond_bb);
                let current_val = self
                    .builder
                    .build_load(self.f64_type, var_ptr, "for_var")
                    .unwrap()
                    .into_float_value();
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OLT, current_val, end_val, "forcond")
                    .unwrap();
                self.builder
                    .build_conditional_branch(cmp, loop_body_bb, after_bb)
                    .unwrap();

                // Body
                self.builder.position_at_end(loop_body_bb);
                for s in body {
                    self.compile_stmt(s, function)?;
                }

                // Increment: var = var + 1
                let cur = self
                    .builder
                    .build_load(self.f64_type, var_ptr, "for_cur")
                    .unwrap()
                    .into_float_value();
                let one = self.f64_type.const_float(1.0);
                let next = self.builder.build_float_add(cur, one, "for_next").unwrap();
                self.builder.build_store(var_ptr, next).unwrap();

                if self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_none()
                {
                    self.builder
                        .build_unconditional_branch(loop_cond_bb)
                        .unwrap();
                }

                self.builder.position_at_end(after_bb);
                Ok(None)
            }

            Stmt::Print(expr) => {
                let val = self.compile_expr(expr, function)?;
                let print_fn = self
                    .module
                    .get_function("qlang_jit_print_f64")
                    .expect("print function should be declared");
                self.builder
                    .build_call(print_fn, &[val.into()], "print_call")
                    .unwrap();
                Ok(None)
            }

            Stmt::Return(expr) => {
                let val = self.compile_expr(expr, function)?;
                self.builder.build_return(Some(&val)).unwrap();
                Ok(Some(val))
            }

            Stmt::ExprStmt(expr) => {
                let val = self.compile_expr(expr, function)?;
                Ok(Some(val))
            }

            Stmt::FnDef { .. } => {
                // Already compiled in the first pass
                Ok(None)
            }

            _ => Err(format!("JIT: unsupported statement: {stmt:?}")),
        }
    }

    /// Compile an expression to LLVM IR, returning its f64 value.
    fn compile_expr(
        &mut self,
        expr: &Expr,
        function: FunctionValue<'ctx>,
    ) -> Result<FloatValue<'ctx>, String> {
        match expr {
            Expr::NumberLit(n) => Ok(self.f64_type.const_float(*n)),

            Expr::BoolLit(b) => Ok(self.f64_type.const_float(if *b { 1.0 } else { 0.0 })),

            Expr::Var(name) => {
                if let Some(ptr) = self.variables.get(name.as_str()) {
                    Ok(self
                        .builder
                        .build_load(self.f64_type, *ptr, name)
                        .unwrap()
                        .into_float_value())
                } else {
                    Err(format!("JIT: undefined variable: {name}"))
                }
            }

            Expr::BinOp { op, left, right } => {
                let l = self.compile_expr(left, function)?;
                let r = self.compile_expr(right, function)?;
                self.compile_binop(*op, l, r)
            }

            Expr::UnaryOp { op, operand } => {
                let v = self.compile_expr(operand, function)?;
                self.compile_unaryop(*op, v)
            }

            Expr::Call { name, args } => {
                // Compile arguments
                let mut compiled_args: Vec<inkwell::values::BasicMetadataValueEnum> = Vec::new();
                for arg in args {
                    let val = self.compile_expr(arg, function)?;
                    compiled_args.push(val.into());
                }

                // Look up function
                let callee = self
                    .module
                    .get_function(name)
                    .ok_or_else(|| format!("JIT: undefined function: {name}"))?;

                let call_val = self
                    .builder
                    .build_call(callee, &compiled_args, "calltmp")
                    .unwrap();

                // Extract return value (f64)
                Ok(call_val
                    .try_as_basic_value()
                    .left()
                    .ok_or_else(|| format!("JIT: function {name} did not return a value"))?
                    .into_float_value())
            }

            _ => Err(format!("JIT: unsupported expression: {expr:?}")),
        }
    }

    /// Compile a binary operation.
    fn compile_binop(
        &self,
        op: BinOp,
        l: FloatValue<'ctx>,
        r: FloatValue<'ctx>,
    ) -> Result<FloatValue<'ctx>, String> {
        match op {
            BinOp::Add => Ok(self.builder.build_float_add(l, r, "add").unwrap()),
            BinOp::Sub => Ok(self.builder.build_float_sub(l, r, "sub").unwrap()),
            BinOp::Mul => Ok(self.builder.build_float_mul(l, r, "mul").unwrap()),
            BinOp::Div => Ok(self.builder.build_float_div(l, r, "div").unwrap()),

            BinOp::Mod => {
                // Use fmod(l, r)
                let fmod_fn = self
                    .module
                    .get_function("fmod")
                    .expect("fmod should be declared");
                let result = self
                    .builder
                    .build_call(fmod_fn, &[l.into(), r.into()], "mod")
                    .unwrap();
                Ok(result
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value())
            }

            BinOp::Pow => {
                // Use llvm.pow.f64 intrinsic
                let pow_fn = self
                    .module
                    .get_function("llvm.pow.f64")
                    .expect("pow intrinsic should be declared");
                let result = self
                    .builder
                    .build_call(pow_fn, &[l.into(), r.into()], "pow")
                    .unwrap();
                Ok(result
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value())
            }

            // Comparisons: result is i1, convert to f64 (0.0 or 1.0)
            BinOp::Lt => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OLT, l, r, "lt")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "lt_f")
                    .unwrap())
            }
            BinOp::Gt => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OGT, l, r, "gt")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "gt_f")
                    .unwrap())
            }
            BinOp::Le => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OLE, l, r, "le")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "le_f")
                    .unwrap())
            }
            BinOp::Ge => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OGE, l, r, "ge")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "ge_f")
                    .unwrap())
            }
            BinOp::Eq => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::OEQ, l, r, "eq")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "eq_f")
                    .unwrap())
            }
            BinOp::Ne => {
                let cmp = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, l, r, "ne")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(cmp, self.f64_type, "ne_f")
                    .unwrap())
            }

            // Logical operators: treat f64 as bool (0.0 = false, nonzero = true)
            BinOp::And => {
                let zero = self.f64_type.const_float(0.0);
                let l_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, l, zero, "l_bool")
                    .unwrap();
                let r_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, r, zero, "r_bool")
                    .unwrap();
                let result = self.builder.build_and(l_bool, r_bool, "and").unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(result, self.f64_type, "and_f")
                    .unwrap())
            }
            BinOp::Or => {
                let zero = self.f64_type.const_float(0.0);
                let l_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, l, zero, "l_bool")
                    .unwrap();
                let r_bool = self
                    .builder
                    .build_float_compare(FloatPredicate::ONE, r, zero, "r_bool")
                    .unwrap();
                let result = self.builder.build_or(l_bool, r_bool, "or").unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(result, self.f64_type, "or_f")
                    .unwrap())
            }

            // Bitwise ops: convert f64 → i64, operate, convert back
            BinOp::BitAnd => {
                let i64_type = self.context.i64_type();
                let li = self
                    .builder
                    .build_float_to_signed_int(l, i64_type, "li")
                    .unwrap();
                let ri = self
                    .builder
                    .build_float_to_signed_int(r, i64_type, "ri")
                    .unwrap();
                let res = self.builder.build_and(li, ri, "bitand").unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "bitand_f")
                    .unwrap())
            }
            BinOp::BitOr => {
                let i64_type = self.context.i64_type();
                let li = self
                    .builder
                    .build_float_to_signed_int(l, i64_type, "li")
                    .unwrap();
                let ri = self
                    .builder
                    .build_float_to_signed_int(r, i64_type, "ri")
                    .unwrap();
                let res = self.builder.build_or(li, ri, "bitor").unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "bitor_f")
                    .unwrap())
            }
            BinOp::BitXor => {
                let i64_type = self.context.i64_type();
                let li = self
                    .builder
                    .build_float_to_signed_int(l, i64_type, "li")
                    .unwrap();
                let ri = self
                    .builder
                    .build_float_to_signed_int(r, i64_type, "ri")
                    .unwrap();
                let res = self.builder.build_xor(li, ri, "bitxor").unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "bitxor_f")
                    .unwrap())
            }
            BinOp::Shl => {
                let i64_type = self.context.i64_type();
                let li = self
                    .builder
                    .build_float_to_signed_int(l, i64_type, "li")
                    .unwrap();
                let ri = self
                    .builder
                    .build_float_to_signed_int(r, i64_type, "ri")
                    .unwrap();
                let res = self.builder.build_left_shift(li, ri, "shl").unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "shl_f")
                    .unwrap())
            }
            BinOp::Shr => {
                let i64_type = self.context.i64_type();
                let li = self
                    .builder
                    .build_float_to_signed_int(l, i64_type, "li")
                    .unwrap();
                let ri = self
                    .builder
                    .build_float_to_signed_int(r, i64_type, "ri")
                    .unwrap();
                let res = self
                    .builder
                    .build_right_shift(li, ri, true, "shr")
                    .unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "shr_f")
                    .unwrap())
            }
        }
    }

    /// Compile a unary operation.
    fn compile_unaryop(
        &self,
        op: UnaryOp,
        v: FloatValue<'ctx>,
    ) -> Result<FloatValue<'ctx>, String> {
        match op {
            UnaryOp::Neg => Ok(self.builder.build_float_neg(v, "neg").unwrap()),
            UnaryOp::Not => {
                let zero = self.f64_type.const_float(0.0);
                let is_zero = self
                    .builder
                    .build_float_compare(FloatPredicate::OEQ, v, zero, "is_zero")
                    .unwrap();
                Ok(self
                    .builder
                    .build_unsigned_int_to_float(is_zero, self.f64_type, "not_f")
                    .unwrap())
            }
            UnaryOp::BitNot => {
                let i64_type = self.context.i64_type();
                let vi = self
                    .builder
                    .build_float_to_signed_int(v, i64_type, "vi")
                    .unwrap();
                let res = self.builder.build_not(vi, "bitnot").unwrap();
                Ok(self
                    .builder
                    .build_signed_int_to_float(res, self.f64_type, "bitnot_f")
                    .unwrap())
            }
        }
    }
}

/// C library fmod (used for the % operator on f64).
extern "C" fn libc_fmod(a: f64, b: f64) -> f64 {
    a % b
}

// ─── JIT compatibility check ─────────────────────────────────────────────────

/// Check whether a list of statements can be JIT-compiled.
///
/// Returns `true` if ALL statements use only numeric operations
/// (no strings, arrays, dicts, imports, index expressions).
pub fn is_jit_compatible(stmts: &[Stmt]) -> bool {
    stmts.iter().all(is_stmt_jit_compatible)
}

fn is_stmt_jit_compatible(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let { value, .. } => is_expr_numeric(value),
        Stmt::Assign { value, .. } => is_expr_numeric(value),
        Stmt::If {
            cond,
            then_body,
            else_body,
        } => {
            is_expr_numeric(cond)
                && is_jit_compatible(then_body)
                && is_jit_compatible(else_body)
        }
        Stmt::While { cond, body } => is_expr_numeric(cond) && is_jit_compatible(body),
        Stmt::For {
            start, end, body, ..
        } => is_expr_numeric(start) && is_expr_numeric(end) && is_jit_compatible(body),
        Stmt::FnDef { body, .. } => {
            is_jit_compatible(body)
        }
        Stmt::Return(expr) => is_expr_numeric(expr),
        Stmt::Print(expr) => is_expr_numeric(expr),
        Stmt::ExprStmt(expr) => is_expr_numeric(expr),
        Stmt::Import(_) => false,
    }
}

fn is_expr_numeric(expr: &Expr) -> bool {
    match expr {
        Expr::NumberLit(_) | Expr::BoolLit(_) => true,
        Expr::StringLit(_) => false,
        Expr::ArrayLit(_) => false,
        Expr::DictLit(_) => false,
        Expr::Index { .. } => false,
        Expr::Var(_) => true, // assume numeric at check time; will fail at compile if not
        Expr::BinOp { left, right, .. } => is_expr_numeric(left) && is_expr_numeric(right),
        Expr::UnaryOp { operand, .. } => is_expr_numeric(operand),
        Expr::Call { args, .. } => args.iter().all(is_expr_numeric),
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Try to JIT-compile and run a QLANG script.
///
/// Returns `Some((result_value, output_lines))` if the script was
/// successfully JIT-compiled and executed. Returns `None` if the script
/// uses features not supported by the JIT (strings, arrays, etc.),
/// in which case the caller should fall back to the bytecode VM.
pub fn try_jit_run(stmts: &[Stmt]) -> Option<(f64, Vec<String>)> {
    if !is_jit_compatible(stmts) {
        return None;
    }
    let context = Context::create();
    let mut jit = ScriptJit::new(&context).ok()?;
    jit.compile_and_run(stmts).ok()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_runtime::vm::{tokenize, Parser};

    fn jit_run(source: &str) -> (f64, Vec<String>) {
        let tokens = tokenize(source).expect("tokenize failed");
        let stmts = Parser::new(tokens).parse_program().expect("parse failed");
        assert!(is_jit_compatible(&stmts), "script is not JIT-compatible");
        let context = Context::create();
        let mut jit = ScriptJit::new(&context).expect("JIT creation failed");
        jit.compile_and_run(&stmts).expect("JIT execution failed")
    }

    #[test]
    fn test_arithmetic() {
        let (val, _) = jit_run("let x = 2 + 3 * 4\nx");
        assert_eq!(val, 14.0);
    }

    #[test]
    fn test_variables() {
        let (val, _) = jit_run("let a = 10\nlet b = 20\na + b");
        assert_eq!(val, 30.0);
    }

    #[test]
    fn test_assignment() {
        let (val, _) = jit_run("let x = 5\nx = x + 1\nx");
        assert_eq!(val, 6.0);
    }

    #[test]
    fn test_if_else() {
        let (_, output) = jit_run(
            "let x = 10\nif x > 5 {\n  print(1)\n} else {\n  print(0)\n}",
        );
        assert_eq!(output, vec!["1"]);
    }

    #[test]
    fn test_while_loop() {
        let (val, _) = jit_run(
            "let sum = 0\nlet i = 0\nwhile i < 10 {\n  sum = sum + i\n  i = i + 1\n}\nsum",
        );
        assert_eq!(val, 45.0); // 0+1+2+...+9 = 45
    }

    #[test]
    fn test_for_loop() {
        let (val, _) = jit_run("let sum = 0\nfor i in 1..11 {\n  sum = sum + i\n}\nsum");
        assert_eq!(val, 55.0); // 1+2+...+10 = 55
    }

    #[test]
    fn test_function() {
        let (val, _) = jit_run(
            "fn square(x) {\n  return x * x\n}\nsquare(7)",
        );
        assert_eq!(val, 49.0);
    }

    #[test]
    fn test_nested_function_calls() {
        let (val, _) = jit_run(
            "fn add(a, b) {\n  return a + b\n}\nfn double(x) {\n  return x * 2\n}\ndouble(add(3, 4))",
        );
        assert_eq!(val, 14.0);
    }

    #[test]
    fn test_power_operator() {
        let (val, _) = jit_run("2 ** 10");
        assert_eq!(val, 1024.0);
    }

    #[test]
    fn test_modulo() {
        let (val, _) = jit_run("17 % 5");
        assert_eq!(val, 2.0);
    }

    #[test]
    fn test_comparison_operators() {
        let (val, _) = jit_run("let a = (3 < 5) + (3 > 5) + (3 == 3) + (3 != 4)\na");
        // true + false + true + true = 3.0
        assert_eq!(val, 3.0);
    }

    #[test]
    fn test_print_output() {
        let (_, output) = jit_run("print(42)\nprint(3.14)");
        assert_eq!(output, vec!["42", "3.14"]);
    }

    #[test]
    fn test_jit_incompatible() {
        let tokens = tokenize("let s = \"hello\"").unwrap();
        let stmts = Parser::new(tokens).parse_program().unwrap();
        assert!(!is_jit_compatible(&stmts));
    }

    #[test]
    fn test_fibonacci() {
        let (val, _) = jit_run(
            r#"
            fn fib(n) {
                if n <= 1 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            fib(10)
            "#,
        );
        assert_eq!(val, 55.0);
    }

    #[test]
    fn test_try_jit_run_compatible() {
        let tokens = tokenize("let x = 2 + 3\nx").unwrap();
        let stmts = Parser::new(tokens).parse_program().unwrap();
        let result = try_jit_run(&stmts);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 5.0);
    }

    #[test]
    fn test_try_jit_run_incompatible() {
        let tokens = tokenize("let s = \"hello\"").unwrap();
        let stmts = Parser::new(tokens).parse_program().unwrap();
        let result = try_jit_run(&stmts);
        assert!(result.is_none());
    }
}
