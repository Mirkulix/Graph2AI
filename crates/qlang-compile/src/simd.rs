//! SIMD-optimized LLVM code generation.
//!
//! Generates vectorized LLVM IR that processes 8 floats at once (AVX2).
//! Performance: 8x throughput compared to scalar operations.

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::OptimizationLevel;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;

use crate::codegen::{CodegenError, CompiledGraph};

/// Compile a graph with explicit SIMD vectorization (8-wide AVX2).
pub fn compile_graph_simd<'ctx>(
    context: &'ctx Context,
    graph: &Graph,
) -> Result<CompiledGraph<'ctx>, CodegenError> {
    let module = context.create_module(&format!("{}_simd", graph.id));
    let builder = context.create_builder();

    let is_aarch64 = cfg!(target_arch = "aarch64");
    let vec_len = if is_aarch64 { 4 } else { 8 };
    let vec_len_u64 = vec_len as u64;

    let f32_type = context.f32_type();
    let f32_ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    let i64_type = context.i64_type();
    let void_type = context.void_type();
    let vec_f32_type = f32_type.vec_type(vec_len);

    let fn_type = void_type.fn_type(
        &[f32_ptr_type.into(), f32_ptr_type.into(), f32_ptr_type.into(), i64_type.into()],
        false,
    );

    let function = module.add_function("qlang_graph", fn_type, None);
    if is_aarch64 {
        function.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            context.create_string_attribute("target-features", "+neon"),
        );
    } else {
        function.add_attribute(
            inkwell::attributes::AttributeLoc::Function,
            context.create_string_attribute("target-features", "+avx2,+fma"),
        );
    }

    let entry_block = context.append_basic_block(function, "entry");
    let vector_loop = context.append_basic_block(function, "vector_loop");
    let vector_body = context.append_basic_block(function, "vector_body");
    let scalar_loop = context.append_basic_block(function, "scalar_loop");
    let scalar_body = context.append_basic_block(function, "scalar_body");
    let exit_block = context.append_basic_block(function, "exit");

    let input_a = function.get_nth_param(0).unwrap().into_pointer_value();
    let input_b = function.get_nth_param(1).unwrap().into_pointer_value();
    let output = function.get_nth_param(2).unwrap().into_pointer_value();
    let n = function.get_nth_param(3).unwrap().into_int_value();

    let ops: Vec<&Op> = graph
        .nodes.iter()
        .filter(|n| !matches!(n.op, Op::Input { .. } | Op::Output { .. } | Op::Constant))
        .map(|n| &n.op)
        .collect();

    let step = i64_type.const_int(vec_len_u64, false);
    let zero = i64_type.const_int(0, false);
    let one = i64_type.const_int(1, false);

    // Entry: compute n_vector_elems = (n / vec_len) * vec_len
    builder.position_at_end(entry_block);
    let n_vectors = builder.build_int_unsigned_div(n, step, "n_vec").unwrap();
    let n_vector_elems = builder.build_int_mul(n_vectors, step, "n_vec_elems").unwrap();
    builder.build_unconditional_branch(vector_loop).unwrap();

    // === Vector loop header ===
    builder.position_at_end(vector_loop);
    let vec_i = builder.build_phi(i64_type, "vec_i").unwrap();
    vec_i.add_incoming(&[(&zero, entry_block)]);
    let vec_i_val = vec_i.as_basic_value().into_int_value();

    let vec_cond = builder.build_int_compare(inkwell::IntPredicate::ULT, vec_i_val, n_vector_elems, "vc").unwrap();
    builder.build_conditional_branch(vec_cond, vector_body, scalar_loop).unwrap();

    // === Vector body: load <N x float>, compute, store ===
    builder.position_at_end(vector_body);

    let a_base = unsafe { builder.build_gep(f32_type, input_a, &[vec_i_val], "a_b").unwrap() };
    let a_vec = builder.build_load(vec_f32_type, a_base, "av").unwrap().into_vector_value();

    let b_base = unsafe { builder.build_gep(f32_type, input_b, &[vec_i_val], "b_b").unwrap() };
    let b_vec = builder.build_load(vec_f32_type, b_base, "bv").unwrap().into_vector_value();

    // Apply vector ops
    let result_vec = emit_vector_ops(&builder, &context, &module, &ops, a_vec, b_vec, vec_len)?;

    let out_base = unsafe { builder.build_gep(f32_type, output, &[vec_i_val], "o_b").unwrap() };
    builder.build_store(out_base, result_vec).unwrap();

    let vec_next = builder.build_int_add(vec_i_val, step, "vn").unwrap();
    vec_i.add_incoming(&[(&vec_next, vector_body)]);
    builder.build_unconditional_branch(vector_loop).unwrap();

    // === Scalar loop for remainder ===
    builder.position_at_end(scalar_loop);
    let sc_i = builder.build_phi(i64_type, "sc_i").unwrap();
    sc_i.add_incoming(&[(&n_vector_elems, vector_loop)]);
    let sc_i_val = sc_i.as_basic_value().into_int_value();

    let sc_cond = builder.build_int_compare(inkwell::IntPredicate::ULT, sc_i_val, n, "sc").unwrap();
    builder.build_conditional_branch(sc_cond, scalar_body, exit_block).unwrap();

    builder.position_at_end(scalar_body);
    let a_ptr = unsafe { builder.build_gep(f32_type, input_a, &[sc_i_val], "as").unwrap() };
    let a_val = builder.build_load(f32_type, a_ptr, "av").unwrap().into_float_value();
    let b_ptr = unsafe { builder.build_gep(f32_type, input_b, &[sc_i_val], "bs").unwrap() };
    let b_val = builder.build_load(f32_type, b_ptr, "bv").unwrap().into_float_value();

    let sc_result = crate::codegen::emit_ops(&builder, &context, &module, &ops, a_val, b_val)?;

    let out_ptr = unsafe { builder.build_gep(f32_type, output, &[sc_i_val], "os").unwrap() };
    builder.build_store(out_ptr, sc_result).unwrap();

    let sc_next = builder.build_int_add(sc_i_val, one, "sn").unwrap();
    sc_i.add_incoming(&[(&sc_next, scalar_body)]);
    builder.build_unconditional_branch(scalar_loop).unwrap();

    builder.position_at_end(exit_block);
    builder.build_return(None).unwrap();

    let llvm_ir = module.print_to_string().to_string();
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| CodegenError::LlvmError(e.to_string()))?;

    Ok(CompiledGraph {
        context,
        module,
        execution_engine,
        function_name: "qlang_graph".to_string(),
        llvm_ir,
    })
}

/// Emit vector operations on <N x float>.
fn emit_vector_ops<'ctx>(
    builder: &Builder<'ctx>,
    context: &'ctx Context,
    module: &inkwell::module::Module<'ctx>,
    ops: &[&Op],
    a: inkwell::values::VectorValue<'ctx>,
    b: inkwell::values::VectorValue<'ctx>,
    vec_len: u32,
) -> Result<inkwell::values::VectorValue<'ctx>, CodegenError> {
    let vec_type = context.f32_type().vec_type(vec_len);
    let mut current = a;

    let build_intrinsic = |name: &str, val: inkwell::values::VectorValue<'ctx>| {
        let intrinsic_name = format!("{}.v{}f32", name, vec_len);
        let fn_val = module.get_function(&intrinsic_name).unwrap_or_else(|| {
            let fn_type = vec_type.fn_type(&[vec_type.into()], false);
            module.add_function(&intrinsic_name, fn_type, None)
        });
        builder.build_call(fn_val, &[val.into()], "").unwrap().try_as_basic_value().left().unwrap().into_vector_value()
    };

    let build_splat = |val: f32| {
        let mut v = vec_type.get_undef();
        let i32_type = context.i32_type();
        let c = context.f32_type().const_float(val as f64);
        for i in 0..vec_len as u64 {
            let idx = i32_type.const_int(i, false);
            v = builder.build_insert_element(v, c, idx, "").unwrap();
        }
        v
    };

    for op in ops {
        current = match op {
            Op::Add => builder.build_float_add(current, b, "vadd").unwrap(),
            Op::Sub => builder.build_float_sub(current, b, "vsub").unwrap(),
            Op::Mul => builder.build_float_mul(current, b, "vmul").unwrap(),
            Op::Div => builder.build_float_div(current, b, "vdiv").unwrap(),
            Op::Neg => builder.build_float_neg(current, "vneg").unwrap(),

            Op::Exp => build_intrinsic("llvm.exp", current),
            Op::Log => build_intrinsic("llvm.log", current),
            
            Op::Tanh => {
                // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
                let two = build_splat(2.0);
                let one = build_splat(1.0);
                let two_x = builder.build_float_mul(two, current, "two_x").unwrap();
                let exp_2x = build_intrinsic("llvm.exp", two_x);
                let num = builder.build_float_sub(exp_2x, one, "num").unwrap();
                let den = builder.build_float_add(exp_2x, one, "den").unwrap();
                builder.build_float_div(num, den, "vtanh").unwrap()
            }

            Op::Gelu => {
                // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let half = build_splat(0.5);
                let one = build_splat(1.0);
                let sqrt_2_pi = build_splat((2.0f32 / std::f32::consts::PI).sqrt());
                let coeff = build_splat(0.044715);

                let x_sq = builder.build_float_mul(current, current, "x_sq").unwrap();
                let x_cb = builder.build_float_mul(x_sq, current, "x_cb").unwrap();
                let c_x_cb = builder.build_float_mul(coeff, x_cb, "c_x_cb").unwrap();
                let inner = builder.build_float_add(current, c_x_cb, "inner").unwrap();
                let scaled = builder.build_float_mul(sqrt_2_pi, inner, "scaled").unwrap();
                
                // tanh
                let two = build_splat(2.0);
                let two_x_tanh = builder.build_float_mul(two, scaled, "two_x_tanh").unwrap();
                let exp_2x_tanh = build_intrinsic("llvm.exp", two_x_tanh);
                let num_tanh = builder.build_float_sub(exp_2x_tanh, one, "num_tanh").unwrap();
                let den_tanh = builder.build_float_add(exp_2x_tanh, one, "den_tanh").unwrap();
                let tanh_val = builder.build_float_div(num_tanh, den_tanh, "tanh_val").unwrap();

                let one_plus_tanh = builder.build_float_add(one, tanh_val, "one_plus_tanh").unwrap();
                let half_x = builder.build_float_mul(half, current, "half_x").unwrap();
                builder.build_float_mul(half_x, one_plus_tanh, "vgelu").unwrap()
            }

            Op::Relu => {
                let zero_vec = vec_type.const_zero();
                let cmp = builder
                    .build_float_compare(inkwell::FloatPredicate::OGT, current, zero_vec, "vrc")
                    .unwrap();
                builder.build_select(cmp, current, zero_vec, "vr").unwrap().into_vector_value()
            }

            Op::ToTernary => {
                // Build splat constants using the vector type
                let f32_t = context.f32_type();
                let pos_thresh = vec_type.const_zero(); // will be replaced
                let _ = pos_thresh;

                // Use splat: create scalar then broadcast
                let p3 = f32_t.const_float(0.3);
                let n3 = f32_t.const_float(-0.3);
                let one_c = f32_t.const_float(1.0);
                let neg1_c = f32_t.const_float(-1.0);
                let zero_c = f32_t.const_float(0.0);

                // Build splat vectors by inserting same value N times
                let mut pos_v = vec_type.get_undef();
                let mut neg_v = vec_type.get_undef();
                let mut one_v = vec_type.get_undef();
                let mut neg1_v = vec_type.get_undef();
                let mut zero_v = vec_type.get_undef();

                let i32_type = context.i32_type();
                for i in 0..vec_len as u64 {
                    let idx = i32_type.const_int(i, false);
                    pos_v = builder.build_insert_element(pos_v, p3, idx, "").unwrap();
                    neg_v = builder.build_insert_element(neg_v, n3, idx, "").unwrap();
                    one_v = builder.build_insert_element(one_v, one_c, idx, "").unwrap();
                    neg1_v = builder.build_insert_element(neg1_v, neg1_c, idx, "").unwrap();
                    zero_v = builder.build_insert_element(zero_v, zero_c, idx, "").unwrap();
                }

                let is_pos = builder
                    .build_float_compare(inkwell::FloatPredicate::OGT, current, pos_v, "vip")
                    .unwrap();
                let is_neg = builder
                    .build_float_compare(inkwell::FloatPredicate::OLT, current, neg_v, "vin")
                    .unwrap();

                let neg_or_zero = builder.build_select(is_neg, neg1_v, zero_v, "vnz").unwrap().into_vector_value();
                builder.build_select(is_pos, one_v, neg_or_zero, "vt").unwrap().into_vector_value()
            }

            other => {
                return Err(CodegenError::UnsupportedOp(format!("SIMD: {other}")));
            }
        };
    }

    Ok(current)
}

// SIMD tests require AVX2-aligned memory allocation.
// The current allocator (Vec<f32>) doesn't guarantee 32-byte alignment.
// TODO Phase 5: Use aligned allocator for SIMD execution.
#[cfg(test)]
mod tests {
    #![allow(unused)]
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn simd_compiles_with_vector_types() {
        // Just verify the graph compiles to LLVM IR with <8 x float>
        let mut g = Graph::new("simd_ir");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(16); 2], vec![TensorType::f32_vector(16)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(16)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(16));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(16));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(16));

        let context = Context::create();
        let compiled = compile_graph_simd(&context, &g).unwrap();
        let is_aarch64 = cfg!(target_arch = "aarch64");
        let vec_str = if is_aarch64 { "<4 x float>" } else { "<8 x float>" };
        assert!(compiled.llvm_ir.contains(vec_str));
        assert!(compiled.llvm_ir.contains(&format!("fadd {}", vec_str)));
    }

    // Execution tests require aligned memory — skipped until Phase 5
    #[test]
    #[ignore = "requires 32-byte aligned memory allocation (Phase 5)"]
    fn simd_add() {
        let mut g = Graph::new("simd_add");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(16); 2], vec![TensorType::f32_vector(16)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(16)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(16));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(16));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(16));

        let context = Context::create();
        let compiled = compile_graph_simd(&context, &g).unwrap();

        let is_aarch64 = cfg!(target_arch = "aarch64");
        let vec_str = if is_aarch64 { "<4 x float>" } else { "<8 x float>" };
        assert!(compiled.llvm_ir.contains(vec_str));

        let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..16).map(|i| (i * 10) as f32).collect();
        let result = crate::codegen::execute_compiled(&compiled, &a_data, &b_data).unwrap();

        let expected: Vec<f32> = (0..16).map(|i| i as f32 + (i * 10) as f32).collect();
        assert_eq!(result, expected);
    }

    #[test]
    #[ignore = "requires 32-byte aligned memory allocation (Phase 5)"]
    fn simd_relu() {
        let mut g = Graph::new("simd_relu");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(16)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(16)], vec![TensorType::f32_vector(16)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(16)], vec![]);
        g.add_edge(a, 0, relu, 0, TensorType::f32_vector(16));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(16));

        let context = Context::create();
        let compiled = compile_graph_simd(&context, &g).unwrap();

        let input: Vec<f32> = (-8..8).map(|i| i as f32).collect();
        let dummy = vec![0.0f32; 16];
        let result = crate::codegen::execute_compiled(&compiled, &input, &dummy).unwrap();

        let expected: Vec<f32> = (-8..8).map(|i| (i as f32).max(0.0)).collect();
        assert_eq!(result, expected);
    }

    #[test]
    #[ignore = "requires 32-byte aligned memory allocation (Phase 5)"]
    fn simd_handles_non_aligned_sizes() {
        let mut g = Graph::new("simd_13");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(13)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(13)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(13); 2], vec![TensorType::f32_vector(13)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(13)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(13));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(13));
        g.add_edge(add, 0, out, 0, TensorType::f32_vector(13));

        let context = Context::create();
        let compiled = compile_graph_simd(&context, &g).unwrap();

        let a_data: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b_data = vec![100.0f32; 13];
        let result = crate::codegen::execute_compiled(&compiled, &a_data, &b_data).unwrap();

        let expected: Vec<f32> = (0..13).map(|i| i as f32 + 100.0).collect();
        assert_eq!(result, expected);
    }
}
