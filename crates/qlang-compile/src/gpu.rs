//! GPU Compute Shader Generation — Compile QLANG graphs to GPU kernels.
//!
//! Generates WGSL (WebGPU Shading Language) compute shaders from QLANG graphs.
//! WGSL is the standard shader language for WebGPU, supported everywhere:
//! - Chrome, Firefox, Safari (via WebGPU)
//! - Native apps (via wgpu-rs)
//! - Cross-platform: Windows, Linux, macOS, Android, iOS
//!
//! Pipeline:
//!   QLANG Graph → WGSL shader → GPU dispatch → Results
//!
//! The generated shader processes tensors in parallel across GPU workgroups.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;

/// Generate a WGSL compute shader from a QLANG graph.
///
/// The shader processes elements in parallel:
///   @workgroup_size(256)
///   fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///       let i = id.x;
///       output[i] = op(input_a[i], input_b[i]);
///   }
pub fn to_wgsl(graph: &Graph) -> String {
    let mut shader = String::new();

    // Header
    shader.push_str("// QLANG Auto-Generated WGSL Compute Shader\n");
    shader.push_str(&format!("// Graph: {}\n\n", graph.id));

    // Bindings
    shader.push_str("@group(0) @binding(0) var<storage, read> input_a: array<f32>;\n");
    shader.push_str("@group(0) @binding(1) var<storage, read> input_b: array<f32>;\n");
    shader.push_str("@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n");
    shader.push_str("@group(0) @binding(3) var<uniform> params: Params;\n\n");

    shader.push_str("struct Params {\n");
    shader.push_str("    n_elements: u32,\n");
    shader.push_str("};\n\n");

    // Collect operations
    let ops: Vec<&Op> = graph
        .nodes
        .iter()
        .filter(|n| !matches!(n.op, Op::Input { .. } | Op::Output { .. } | Op::Constant))
        .map(|n| &n.op)
        .collect();

    // Main compute kernel
    shader.push_str("@compute @workgroup_size(256)\n");
    shader.push_str("fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n");
    shader.push_str("    let i = global_id.x;\n");
    shader.push_str("    if (i >= params.n_elements) {\n");
    shader.push_str("        return;\n");
    shader.push_str("    }\n\n");
    shader.push_str("    let a = input_a[i];\n");
    shader.push_str("    let b = input_b[i];\n\n");

    // Generate operation chain
    let mut current_var = "a".to_string();
    for (idx, op) in ops.iter().enumerate() {
        let result_var = format!("r{idx}");
        let expr = match op {
            Op::Add => format!("    let {} = {} + b;\n", result_var, current_var),
            Op::Sub => format!("    let {} = {} - b;\n", result_var, current_var),
            Op::Mul => format!("    let {} = {} * b;\n", result_var, current_var),
            Op::Div => format!("    let {} = {} / b;\n", result_var, current_var),
            Op::Neg => format!("    let {} = -{};\n", result_var, current_var),
            Op::Relu => format!("    let {} = max({}, 0.0);\n", result_var, current_var),
            Op::Sigmoid => format!("    let {} = 1.0 / (1.0 + exp(-{}));\n", result_var, current_var),
            Op::Tanh => format!("    let {} = tanh({});\n", result_var, current_var),
            Op::ToTernary => {
                let mut s = String::new();
                s.push_str(&format!("    var {}: f32;\n", result_var));
                s.push_str(&format!("    if ({} > 0.3) {{\n", current_var));
                s.push_str(&format!("        {} = 1.0;\n", result_var));
                s.push_str(&format!("    }} else if ({} < -0.3) {{\n", current_var));
                s.push_str(&format!("        {} = -1.0;\n", result_var));
                s.push_str("    } else {\n");
                s.push_str(&format!("        {} = 0.0;\n", result_var));
                s.push_str("    }\n");
                s
            }
            _ => format!("    let {} = {}; // unsupported: {}\n", result_var, current_var, op),
        };
        shader.push_str(&expr);
        current_var = result_var;
    }

    shader.push_str(&format!("\n    output[i] = {};\n", current_var));
    shader.push_str("}\n");

    shader
}

/// Generate a WGSL shader for matrix multiplication.
///
/// This is a tiled matmul optimized for GPU:
///   - Uses workgroup shared memory for tiling
///   - 16×16 tile size (standard for modern GPUs)
///   - Each thread computes one element of the output
pub fn matmul_wgsl(m: usize, k: usize, n: usize) -> String {
    let tile_size = 16;

    let mut shader = String::new();
    shader.push_str("// QLANG Auto-Generated Matrix Multiplication Shader\n");
    shader.push_str(&format!("// Dimensions: [{}×{}] @ [{}×{}] → [{}×{}]\n\n", m, k, k, n, m, n));

    shader.push_str("@group(0) @binding(0) var<storage, read> A: array<f32>;\n");
    shader.push_str("@group(0) @binding(1) var<storage, read> B: array<f32>;\n");
    shader.push_str("@group(0) @binding(2) var<storage, read_write> C: array<f32>;\n\n");

    shader.push_str("struct Dims {\n");
    shader.push_str("    M: u32, K: u32, N: u32,\n");
    shader.push_str("};\n");
    shader.push_str("@group(0) @binding(3) var<uniform> dims: Dims;\n\n");

    shader.push_str(&format!("const TILE_SIZE: u32 = {}u;\n\n", tile_size));

    shader.push_str("var<workgroup> tile_A: array<array<f32, TILE_SIZE>, TILE_SIZE>;\n");
    shader.push_str("var<workgroup> tile_B: array<array<f32, TILE_SIZE>, TILE_SIZE>;\n\n");

    shader.push_str("@compute @workgroup_size(TILE_SIZE, TILE_SIZE)\n");
    shader.push_str("fn main(\n");
    shader.push_str("    @builtin(global_invocation_id) global_id: vec3<u32>,\n");
    shader.push_str("    @builtin(local_invocation_id) local_id: vec3<u32>,\n");
    shader.push_str("    @builtin(workgroup_id) wg_id: vec3<u32>,\n");
    shader.push_str(") {\n");
    shader.push_str("    let row = global_id.y;\n");
    shader.push_str("    let col = global_id.x;\n");
    shader.push_str("    let local_row = local_id.y;\n");
    shader.push_str("    let local_col = local_id.x;\n\n");

    shader.push_str("    var sum: f32 = 0.0;\n");
    shader.push_str("    let n_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;\n\n");

    shader.push_str("    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {\n");
    shader.push_str("        // Load tiles into shared memory\n");
    shader.push_str("        let a_col = t * TILE_SIZE + local_col;\n");
    shader.push_str("        let b_row = t * TILE_SIZE + local_row;\n\n");

    shader.push_str("        if (row < dims.M && a_col < dims.K) {\n");
    shader.push_str("            tile_A[local_row][local_col] = A[row * dims.K + a_col];\n");
    shader.push_str("        } else {\n");
    shader.push_str("            tile_A[local_row][local_col] = 0.0;\n");
    shader.push_str("        }\n\n");

    shader.push_str("        if (b_row < dims.K && col < dims.N) {\n");
    shader.push_str("            tile_B[local_row][local_col] = B[b_row * dims.N + col];\n");
    shader.push_str("        } else {\n");
    shader.push_str("            tile_B[local_row][local_col] = 0.0;\n");
    shader.push_str("        }\n\n");

    shader.push_str("        workgroupBarrier();\n\n");

    shader.push_str("        // Compute partial dot product\n");
    shader.push_str("        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {\n");
    shader.push_str("            sum = sum + tile_A[local_row][i] * tile_B[i][local_col];\n");
    shader.push_str("        }\n\n");

    shader.push_str("        workgroupBarrier();\n");
    shader.push_str("    }\n\n");

    shader.push_str("    if (row < dims.M && col < dims.N) {\n");
    shader.push_str("        C[row * dims.N + col] = sum;\n");
    shader.push_str("    }\n");
    shader.push_str("}\n");

    shader
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn wgsl_add_relu() {
        let mut g = Graph::new("gpu_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(1024)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(1024)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(1024); 2], vec![TensorType::f32_vector(1024)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(1024)], vec![TensorType::f32_vector(1024)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(1024)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(1024));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(1024));
        g.add_edge(add, 0, relu, 0, TensorType::f32_vector(1024));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(1024));

        let wgsl = to_wgsl(&g);
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("@workgroup_size(256)"));
        assert!(wgsl.contains("a + b"));
        assert!(wgsl.contains("max("));
    }

    #[test]
    fn wgsl_ternary() {
        let mut g = Graph::new("ternary_gpu");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(512)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(512)]);
        let t = g.add_node(Op::ToTernary, vec![TensorType::f32_vector(512)], vec![TensorType::f32_vector(512)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(512)], vec![]);
        g.add_edge(a, 0, t, 0, TensorType::f32_vector(512));
        g.add_edge(t, 0, out, 0, TensorType::f32_vector(512));

        let wgsl = to_wgsl(&g);
        assert!(wgsl.contains("1.0"));
        assert!(wgsl.contains("-1.0"));
        assert!(wgsl.contains("0.3"));
    }

    #[test]
    fn matmul_shader() {
        let wgsl = matmul_wgsl(64, 128, 32);
        assert!(wgsl.contains("TILE_SIZE"));
        assert!(wgsl.contains("workgroupBarrier"));
        assert!(wgsl.contains("tile_A"));
        assert!(wgsl.contains("tile_B"));
        assert!(wgsl.contains("[64×128]"));
    }
}
