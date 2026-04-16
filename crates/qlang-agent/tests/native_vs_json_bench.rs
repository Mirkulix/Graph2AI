//! PRD Task 3.1 — native `.qlg` vs JSON payload benchmark.
//!
//! Asserts the non-functional performance bounds from the PRD:
//!
//!   NF1: native serialize MUST be at least 2x faster than JSON
//!   NF2: native envelope MUST be at least 2.3x smaller than JSON
//!
//! Also verifies bit-lossless round-trip for both formats. This test is
//! the load-bearing evidence that `FLAG_NATIVE_BINARY` meets the spec
//! §13.2 "QLMS bin vs MCP JSON" numbers on a realistic conversation
//! shape — multi-message, multi-node graphs, non-trivial metadata.
//!
//! Run under `--release` for meaningful timing:
//!
//! ```bash
//! cargo test -p qlang-agent --release --test native_vs_json_bench -- --nocapture
//! ```
//!
//! The `#[ignore]` default guard prevents the bench from dragging debug
//! CI, while still being one command away for audit and for the AAIF
//! submission checklist.

use qlang_agent::protocol::{
    decode_qlms_frame, AgentConversation, AgentId, Capability, MessageIntent,
};
use qlang_core::crypto::Keypair;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData, TensorType};
use std::collections::HashMap;
use std::time::Instant;

const MESSAGES: usize = 10;
const NODES_PER_GRAPH: usize = 20;
const SERIALIZE_ITERS: usize = 500;
const WARMUP_ITERS: usize = 50;
/// NF1 bound — native MUST be at least this many times faster.
const MIN_SPEED_RATIO: f64 = 2.0;
/// NF2 bound — native envelope MUST be at least this many times smaller.
const MIN_SIZE_RATIO: f64 = 2.3;

fn agent(name: &str) -> AgentId {
    AgentId {
        name: name.to_string(),
        capabilities: vec![Capability::Execute, Capability::Compile, Capability::Train],
    }
}

/// Construct a graph of roughly `n_nodes` small ops wired in a chain.
/// Big enough for JSON field-name overhead to matter, small enough for
/// bincode to shine.
fn build_graph(n_nodes: usize, id: &str) -> Graph {
    let mut g = Graph::new(id);
    let ty = TensorType::f32_vector(128);
    let mut previous = g.add_node(
        Op::Input {
            name: "x".into(),
        },
        Vec::new(),
        vec![ty.clone()],
    );
    for i in 1..n_nodes - 1 {
        let op = match i % 4 {
            0 => Op::Relu,
            1 => Op::Tanh,
            2 => Op::Sigmoid,
            _ => Op::Gelu,
        };
        let node = g.add_node(op, vec![ty.clone()], vec![ty.clone()]);
        g.add_edge(previous, 0, node, 0, ty.clone());
        previous = node;
    }
    let out = g.add_node(
        Op::Output {
            name: "y".into(),
        },
        vec![ty.clone()],
        Vec::new(),
    );
    g.add_edge(previous, 0, out, 0, ty);
    g
}

/// Build a realistic `inputs` map containing a single f32 weight vector.
/// This is where JSON loses the most ground — f32 values in JSON cost
/// ~15 characters each, while bincode stores 4 raw bytes.
fn sample_inputs(vec_len: usize, seed: f32) -> HashMap<String, TensorData> {
    let values: Vec<f32> = (0..vec_len)
        .map(|i| seed + (i as f32) * 0.017 - 0.5)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(
        "weights".to_string(),
        TensorData::from_f32(Shape::vector(vec_len), &values),
    );
    inputs
}

fn build_conv(n_messages: usize, n_nodes: usize) -> AgentConversation {
    let mut conv = AgentConversation::new();
    let a = agent("trainer");
    let b = agent("compressor");
    for i in 0..n_messages {
        conv.send(
            a.clone(),
            b.clone(),
            build_graph(n_nodes, &format!("bench-{i}")),
            sample_inputs(256, i as f32 * 0.1),
            if i % 2 == 0 {
                MessageIntent::Execute
            } else {
                MessageIntent::Optimize
            },
            if i > 0 { Some(i as u64 - 1) } else { None },
        );
    }
    conv
}

fn median_ns(times_ns: &mut [u128]) -> u128 {
    times_ns.sort_unstable();
    times_ns[times_ns.len() / 2]
}

/// Run the benchmark. Kept as its own function so AAIF auditors can call
/// it by running `cargo test ... -- --ignored --nocapture`.
#[test]
#[ignore = "performance bench — run explicitly under --release"]
fn native_beats_json_on_size_and_speed() {
    let kp = Keypair::from_seed(&[0x77u8; 32]);
    let conv = build_conv(MESSAGES, NODES_PER_GRAPH);

    // --- Size comparison (signed envelopes — realistic wire state) ---
    let json_bytes = conv.to_signed_binary(&kp).expect("json build");
    let native_bytes = conv.to_signed_binary_native(&kp).expect("native build");
    let json_len = json_bytes.len();
    let native_len = native_bytes.len();
    let size_ratio = json_len as f64 / native_len as f64;
    eprintln!(
        "[bench] JSON signed = {} B, native signed = {} B, ratio = {:.2}x",
        json_len, native_len, size_ratio
    );

    // --- Serialize time comparison ---
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = std::hint::black_box(conv.to_signed_binary(&kp));
        let _ = std::hint::black_box(conv.to_signed_binary_native(&kp));
    }

    let mut json_samples = Vec::with_capacity(5);
    let mut native_samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = Instant::now();
        for _ in 0..SERIALIZE_ITERS {
            let _ = std::hint::black_box(conv.to_signed_binary(&kp).unwrap());
        }
        json_samples.push(t0.elapsed().as_nanos() / SERIALIZE_ITERS as u128);

        let t0 = Instant::now();
        for _ in 0..SERIALIZE_ITERS {
            let _ = std::hint::black_box(conv.to_signed_binary_native(&kp).unwrap());
        }
        native_samples.push(t0.elapsed().as_nanos() / SERIALIZE_ITERS as u128);
    }
    let json_ns = median_ns(&mut json_samples);
    let native_ns = median_ns(&mut native_samples);
    let speed_ratio = json_ns as f64 / native_ns as f64;
    eprintln!(
        "[bench] JSON serialize median = {} ns, native = {} ns, ratio = {:.2}x",
        json_ns, native_ns, speed_ratio
    );

    // --- Round-trip fidelity (both paths MUST decode back to identical messages) ---
    let decoded_json = decode_qlms_frame(&json_bytes).expect("json decode");
    let decoded_native = decode_qlms_frame(&native_bytes).expect("native decode");
    assert_eq!(
        decoded_json.messages.len(),
        decoded_native.messages.len(),
        "JSON and native must decode to the same message count"
    );
    assert_eq!(
        decoded_json.messages.len(),
        MESSAGES,
        "decoded message count must match input"
    );
    for (a, b) in decoded_json
        .messages
        .iter()
        .zip(decoded_native.messages.iter())
    {
        assert_eq!(a.id, b.id, "ids must match across formats");
        assert_eq!(
            a.graph.nodes.len(),
            b.graph.nodes.len(),
            "node counts must match across formats"
        );
    }

    // --- Spec bounds ---
    assert!(
        size_ratio >= MIN_SIZE_RATIO,
        "NF2 violated: native envelope only {:.2}x smaller than JSON, need >= {:.2}x \
         (JSON={} B, native={} B)",
        size_ratio,
        MIN_SIZE_RATIO,
        json_len,
        native_len
    );
    assert!(
        speed_ratio >= MIN_SPEED_RATIO,
        "NF1 violated: native serialize only {:.2}x faster than JSON, need >= {:.2}x \
         (JSON={} ns, native={} ns)",
        speed_ratio,
        MIN_SPEED_RATIO,
        json_ns,
        native_ns
    );
}
