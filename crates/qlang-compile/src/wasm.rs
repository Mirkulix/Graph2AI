//! WebAssembly Code Generation — Compile QLANG graphs to WASM.
//!
//! Generates WebAssembly Text Format (WAT) that can be:
//! - Compiled to .wasm binary (via wabt/wat2wasm)
//! - Run in any browser (Chrome, Firefox, Safari)
//! - Embedded in Node.js/Deno/Bun
//! - Used as portable compute module
//!
//! Pipeline: QLANG Graph → WAT text → .wasm binary → Execute anywhere
//!
//! The generated WASM module exports:
//!   (func $qlang_graph (param $input_ptr i32) (param $output_ptr i32) (param $n i32))

use qlang_core::graph::Graph;
use qlang_core::ops::Op;

/// Generate WebAssembly Text Format (WAT) from a QLANG graph.
pub fn to_wat(graph: &Graph) -> String {
    let mut wat = String::new();

    // Module header
    wat.push_str(&format!("(module ;; QLANG Graph: {}\n", graph.id));
    wat.push_str("  ;; Auto-generated WebAssembly from QLANG compiler\n\n");

    // Memory (1 page = 64KB, enough for small tensors)
    wat.push_str("  (memory (export \"memory\") 16) ;; 1MB\n\n");

    // Import math functions (provided by host)
    wat.push_str("  ;; Math imports (provided by JavaScript host)\n");
    wat.push_str("  (import \"math\" \"exp\" (func $exp (param f32) (result f32)))\n");
    wat.push_str("  (import \"math\" \"sqrt\" (func $sqrt (param f32) (result f32)))\n");
    wat.push_str("  (import \"math\" \"tanh\" (func $tanh (param f32) (result f32)))\n\n");

    // Collect operations
    let ops: Vec<&Op> = graph
        .nodes
        .iter()
        .filter(|n| !matches!(n.op, Op::Input { .. } | Op::Output { .. } | Op::Constant))
        .map(|n| &n.op)
        .collect();

    // Main function: process n elements
    wat.push_str("  (func $qlang_graph (export \"qlang_graph\")\n");
    wat.push_str("    (param $input_a i32)  ;; pointer to input array a\n");
    wat.push_str("    (param $input_b i32)  ;; pointer to input array b\n");
    wat.push_str("    (param $output i32)   ;; pointer to output array\n");
    wat.push_str("    (param $n i32)        ;; number of elements\n");
    wat.push_str("    (local $i i32)\n");
    wat.push_str("    (local $a f32)\n");
    wat.push_str("    (local $b f32)\n");
    wat.push_str("    (local $result f32)\n\n");

    // Loop over elements
    wat.push_str("    (local.set $i (i32.const 0))\n");
    wat.push_str("    (block $break\n");
    wat.push_str("      (loop $loop\n");
    wat.push_str("        ;; Check i < n\n");
    wat.push_str("        (br_if $break (i32.ge_u (local.get $i) (local.get $n)))\n\n");

    // Load inputs
    wat.push_str("        ;; Load input_a[i]\n");
    wat.push_str("        (local.set $a (f32.load\n");
    wat.push_str("          (i32.add (local.get $input_a)\n");
    wat.push_str("                   (i32.mul (local.get $i) (i32.const 4)))))\n\n");

    wat.push_str("        ;; Load input_b[i]\n");
    wat.push_str("        (local.set $b (f32.load\n");
    wat.push_str("          (i32.add (local.get $input_b)\n");
    wat.push_str("                   (i32.mul (local.get $i) (i32.const 4)))))\n\n");

    // Apply operations
    wat.push_str("        ;; Compute result\n");
    let mut current = "$a".to_string();

    for (idx, op) in ops.iter().enumerate() {
        let result_var = if idx == ops.len() - 1 { "$result".to_string() } else { format!("$r{idx}") };

        if idx < ops.len() - 1 {
            wat.push_str(&format!("        (local ${} f32)\n", format!("r{idx}")));
        }

        match op {
            Op::Add => {
                wat.push_str(&format!("        (local.set {} (f32.add (local.get {}) (local.get $b)))\n",
                    result_var, current));
            }
            Op::Sub => {
                wat.push_str(&format!("        (local.set {} (f32.sub (local.get {}) (local.get $b)))\n",
                    result_var, current));
            }
            Op::Mul => {
                wat.push_str(&format!("        (local.set {} (f32.mul (local.get {}) (local.get $b)))\n",
                    result_var, current));
            }
            Op::Neg => {
                wat.push_str(&format!("        (local.set {} (f32.neg (local.get {})))\n",
                    result_var, current));
            }
            Op::Relu => {
                wat.push_str(&format!("        (local.set {} (f32.max (local.get {}) (f32.const 0.0)))\n",
                    result_var, current));
            }
            Op::Sigmoid => {
                wat.push_str(&format!(
                    "        (local.set {0} (f32.div (f32.const 1.0)\n\
                     \x20         (f32.add (f32.const 1.0) (call $exp (f32.neg (local.get {1}))))))\n",
                    result_var, current));
            }
            Op::Tanh => {
                wat.push_str(&format!("        (local.set {} (call $tanh (local.get {})))\n",
                    result_var, current));
            }
            Op::ToTernary => {
                wat.push_str(&format!(
                    "        (local.set {0}\n\
                     \x20         (if (result f32) (f32.gt (local.get {1}) (f32.const 0.3))\n\
                     \x20           (then (f32.const 1.0))\n\
                     \x20           (else (if (result f32) (f32.lt (local.get {1}) (f32.const -0.3))\n\
                     \x20             (then (f32.const -1.0))\n\
                     \x20             (else (f32.const 0.0))))))\n",
                    result_var, current));
            }
            _ => {
                wat.push_str(&format!("        ;; unsupported op: {}\n", op));
                wat.push_str(&format!("        (local.set {} (local.get {}))\n", result_var, current));
            }
        }

        current = result_var;
    }

    // Store result
    wat.push_str("\n        ;; Store output[i]\n");
    wat.push_str("        (f32.store\n");
    wat.push_str("          (i32.add (local.get $output)\n");
    wat.push_str("                   (i32.mul (local.get $i) (i32.const 4)))\n");
    wat.push_str("          (local.get $result))\n\n");

    // Increment and loop
    wat.push_str("        (local.set $i (i32.add (local.get $i) (i32.const 1)))\n");
    wat.push_str("        (br $loop)\n");
    wat.push_str("      )\n");
    wat.push_str("    )\n");
    wat.push_str("  )\n");

    // Helper: allocate memory region
    wat.push_str("\n  ;; Allocator: returns pointer to n*4 bytes\n");
    wat.push_str("  (global $heap_ptr (mut i32) (i32.const 0))\n");
    wat.push_str("  (func $alloc (export \"alloc\") (param $bytes i32) (result i32)\n");
    wat.push_str("    (local $ptr i32)\n");
    wat.push_str("    (local.set $ptr (global.get $heap_ptr))\n");
    wat.push_str("    (global.set $heap_ptr (i32.add (global.get $heap_ptr) (local.get $bytes)))\n");
    wat.push_str("    (local.get $ptr)\n");
    wat.push_str("  )\n");

    wat.push_str(")\n");

    wat
}

/// Generate JavaScript glue code to run the WASM module.
pub fn to_js_loader(graph: &Graph) -> String {
    format!(r#"// QLANG Graph: {} — JavaScript WASM Loader
// Auto-generated by QLANG compiler

async function loadQlangModule(wasmPath) {{
  const importObject = {{
    math: {{
      exp: Math.exp,
      sqrt: Math.sqrt,
      tanh: Math.tanh,
    }},
  }};

  const {{ instance }} = await WebAssembly.instantiateStreaming(
    fetch(wasmPath),
    importObject
  );

  return {{
    // Execute the graph on float32 arrays
    execute(inputA, inputB) {{
      const n = inputA.length;
      const mem = new Float32Array(instance.exports.memory.buffer);

      // Allocate memory
      const ptrA = instance.exports.alloc(n * 4);
      const ptrB = instance.exports.alloc(n * 4);
      const ptrOut = instance.exports.alloc(n * 4);

      // Copy inputs to WASM memory
      mem.set(inputA, ptrA / 4);
      mem.set(inputB, ptrB / 4);

      // Execute graph
      instance.exports.qlang_graph(ptrA, ptrB, ptrOut, n);

      // Read output
      return new Float32Array(mem.buffer, ptrOut, n);
    }},
  }};
}}

// Usage:
// const qlang = await loadQlangModule('graph.wasm');
// const result = qlang.execute(new Float32Array([1,2,3,4]), new Float32Array([5,6,7,8]));
// console.log(result); // [6, 8, 10, 12]
"#, graph.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;
    use qlang_core::ops::Op;
    use qlang_core::tensor::TensorType;

    #[test]
    fn wat_add_relu() {
        let mut g = Graph::new("wasm_test");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(4)]);
        let add = g.add_node(Op::Add, vec![TensorType::f32_vector(4); 2], vec![TensorType::f32_vector(4)]);
        let relu = g.add_node(Op::Relu, vec![TensorType::f32_vector(4)], vec![TensorType::f32_vector(4)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(4)], vec![]);
        g.add_edge(a, 0, add, 0, TensorType::f32_vector(4));
        g.add_edge(b, 0, add, 1, TensorType::f32_vector(4));
        g.add_edge(add, 0, relu, 0, TensorType::f32_vector(4));
        g.add_edge(relu, 0, out, 0, TensorType::f32_vector(4));

        let wat = to_wat(&g);
        assert!(wat.contains("(module"));
        assert!(wat.contains("f32.add"));
        assert!(wat.contains("f32.max"));
        assert!(wat.contains("qlang_graph"));
    }

    #[test]
    fn wat_ternary() {
        let mut g = Graph::new("ternary_wasm");
        let a = g.add_node(Op::Input { name: "a".into() }, vec![], vec![TensorType::f32_vector(8)]);
        let _b = g.add_node(Op::Input { name: "b".into() }, vec![], vec![TensorType::f32_vector(8)]);
        let t = g.add_node(Op::ToTernary, vec![TensorType::f32_vector(8)], vec![TensorType::f32_vector(8)]);
        let out = g.add_node(Op::Output { name: "y".into() }, vec![TensorType::f32_vector(8)], vec![]);
        g.add_edge(a, 0, t, 0, TensorType::f32_vector(8));
        g.add_edge(t, 0, out, 0, TensorType::f32_vector(8));

        let wat = to_wat(&g);
        assert!(wat.contains("f32.const 1.0"));
        assert!(wat.contains("f32.const -1.0"));
    }

    #[test]
    fn js_loader() {
        let g = Graph::new("test_js");
        let js = to_js_loader(&g);
        assert!(js.contains("WebAssembly.instantiateStreaming"));
        assert!(js.contains("Float32Array"));
    }
}
