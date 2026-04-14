//! Benchmark: QLANG binary graph format (.qlb) vs JSON (.qlg.json)
//!
//! Run: cargo run --release --no-default-features --example benchmark_binary_format
//!
//! Compares:
//! 1. Serialization size (bytes) for various graph sizes
//! 2. Encode/decode speed (1000 iterations)
//! 3. Content hash computation time
//! 4. Cache hit/miss demonstration

use qlang_core::binary;
use qlang_core::cache::{CacheEntry, ComputationCache};
use qlang_core::crypto::{self, sha256};
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType};
use std::collections::HashMap;
use std::time::Instant;

fn f32_mat(m: usize, n: usize) -> TensorType {
    TensorType::new(Dtype::F32, Shape::matrix(m, n))
}

/// Build a graph with the given number of chained operations
fn build_graph(name: &str, n_layers: usize) -> Graph {
    let mut g = Graph::new(name);

    let inp = g.add_node(
        Op::Input { name: "x".into() },
        vec![],
        vec![f32_mat(128, 64)],
    );

    let mut prev = inp;
    for _i in 0..n_layers {
        let relu = g.add_node(Op::Relu, vec![f32_mat(128, 64)], vec![f32_mat(128, 64)]);
        g.add_edge(prev, 0, relu, 0, f32_mat(128, 64));
        prev = relu;
    }

    let out = g.add_node(
        Op::Output { name: "y".into() },
        vec![f32_mat(128, 64)],
        vec![],
    );
    g.add_edge(prev, 0, out, 0, f32_mat(128, 64));

    g
}

/// Build a more realistic transformer-like graph
fn build_transformer_graph() -> Graph {
    let mut g = Graph::new("transformer-bench");

    let x = g.add_node(
        Op::Input { name: "tokens".into() },
        vec![],
        vec![f32_mat(32, 512)],
    );
    let emb = g.add_node(
        Op::Embedding { vocab_size: 50000, d_model: 512 },
        vec![f32_mat(32, 512)],
        vec![f32_mat(32, 512)],
    );
    g.add_edge(x, 0, emb, 0, f32_mat(32, 512));

    let mut prev = emb;
    // 6 transformer blocks
    for _ in 0..6 {
        let attn = g.add_node(
            Op::Attention { n_heads: 8, d_model: 512 },
            vec![f32_mat(32, 512), f32_mat(32, 512), f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        g.add_edge(prev, 0, attn, 0, f32_mat(32, 512));
        g.add_edge(prev, 0, attn, 1, f32_mat(32, 512));
        g.add_edge(prev, 0, attn, 2, f32_mat(32, 512));

        let res = g.add_node(
            Op::Residual,
            vec![f32_mat(32, 512), f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        g.add_edge(prev, 0, res, 0, f32_mat(32, 512));
        g.add_edge(attn, 0, res, 1, f32_mat(32, 512));

        let ln = g.add_node(
            Op::LayerNorm { eps: 1e-6 },
            vec![f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        g.add_edge(res, 0, ln, 0, f32_mat(32, 512));

        let gelu = g.add_node(
            Op::Gelu,
            vec![f32_mat(32, 512)],
            vec![f32_mat(32, 512)],
        );
        g.add_edge(ln, 0, gelu, 0, f32_mat(32, 512));

        prev = gelu;
    }

    let out = g.add_node(
        Op::Output { name: "logits".into() },
        vec![f32_mat(32, 512)],
        vec![],
    );
    g.add_edge(prev, 0, out, 0, f32_mat(32, 512));

    g
}

fn main() {
    println!("================================================================");
    println!("  QLANG Binary Graph Format (.qlb) vs JSON -- Benchmark");
    println!("================================================================");
    println!();

    // =====================================================================
    // 1. SIZE COMPARISON
    // =====================================================================
    println!("--- 1. Size Comparison ---\n");

    let graphs = vec![
        ("tiny (3 nodes)", build_graph("tiny", 1)),
        ("small (12 nodes)", build_graph("small", 10)),
        ("medium (52 nodes)", build_graph("medium", 50)),
        ("large (202 nodes)", build_graph("large", 200)),
        ("transformer (26 nodes)", build_transformer_graph()),
    ];

    println!(
        "  {:30} {:>10} {:>10} {:>8}",
        "Graph", "JSON", "Binary", "Ratio"
    );
    println!("  {:-<30} {:-<10} {:-<10} {:-<8}", "", "", "", "");

    for (name, graph) in &graphs {
        let json = serde_json::to_vec(graph).unwrap();
        let bin = binary::to_binary(graph);

        let ratio = json.len() as f64 / bin.len() as f64;
        println!(
            "  {:30} {:>8} B {:>8} B {:>6.1}x",
            name,
            json.len(),
            bin.len(),
            ratio
        );
    }
    println!();

    // =====================================================================
    // 2. ENCODE/DECODE SPEED
    // =====================================================================
    println!("--- 2. Encode/Decode Speed (1000 iterations) ---\n");

    let bench_graph = build_transformer_graph();
    let iterations = 1000;

    // JSON encode
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = serde_json::to_vec(&bench_graph).unwrap();
    }
    let json_encode_time = start.elapsed();

    // Binary encode
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = binary::to_binary(&bench_graph);
    }
    let binary_encode_time = start.elapsed();

    // JSON decode
    let json_data = serde_json::to_string(&bench_graph).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        let _: Graph = serde_json::from_str(&json_data).unwrap();
    }
    let json_decode_time = start.elapsed();

    // Binary decode
    let binary_data = binary::to_binary(&bench_graph);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = binary::from_binary(&binary_data).unwrap();
    }
    let binary_decode_time = start.elapsed();

    println!(
        "  {:20} {:>12} {:>12} {:>8}",
        "Operation", "JSON", "Binary", "Speedup"
    );
    println!("  {:-<20} {:-<12} {:-<12} {:-<8}", "", "", "", "");
    println!(
        "  {:20} {:>10.2}ms {:>10.2}ms {:>6.1}x",
        "Encode (total)",
        json_encode_time.as_secs_f64() * 1000.0,
        binary_encode_time.as_secs_f64() * 1000.0,
        json_encode_time.as_secs_f64() / binary_encode_time.as_secs_f64()
    );
    println!(
        "  {:20} {:>10.2}ms {:>10.2}ms {:>6.1}x",
        "Decode (total)",
        json_decode_time.as_secs_f64() * 1000.0,
        binary_decode_time.as_secs_f64() * 1000.0,
        json_decode_time.as_secs_f64() / binary_decode_time.as_secs_f64()
    );
    println!(
        "  {:20} {:>10.2}us {:>10.2}us",
        "Encode (per graph)",
        json_encode_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
        binary_encode_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
    println!(
        "  {:20} {:>10.2}us {:>10.2}us",
        "Decode (per graph)",
        json_decode_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
        binary_decode_time.as_secs_f64() * 1_000_000.0 / iterations as f64,
    );
    println!();

    // =====================================================================
    // 3. CONTENT HASH
    // =====================================================================
    println!("--- 3. Content Hash (1000 iterations) ---\n");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = binary::content_hash(&bench_graph);
    }
    let hash_time = start.elapsed();

    let hash = binary::content_hash(&bench_graph);
    println!(
        "  Hash:         {}",
        crypto::hex(&hash)
    );
    println!(
        "  Time (total): {:.2}ms",
        hash_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Time (per):   {:.2}us",
        hash_time.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!();

    // =====================================================================
    // 4. COMPUTATION CACHE DEMO
    // =====================================================================
    println!("--- 4. Computation Cache Demo ---\n");

    let mut cache = ComputationCache::with_capacity(100);

    let graph_hash = binary::content_hash(&bench_graph);
    let input_data = TensorData::from_f32(Shape::vector(4), &[1.0, 2.0, 3.0, 4.0]);
    let input_hash = sha256(input_data.as_bytes());
    let cache_key =
        ComputationCache::cache_key(&graph_hash, &[input_hash.as_slice()]);

    // First lookup: miss
    let result = cache.get(&cache_key);
    assert!(result.is_none());
    let (hits, misses, _) = cache.stats();
    println!("  After 1st lookup:  hits={hits}, misses={misses} (MISS)");

    // Simulate computation and store
    let mut outputs = HashMap::new();
    outputs.insert(
        "y".to_string(),
        TensorData::from_f32(Shape::vector(4), &[2.0, 4.0, 6.0, 8.0]),
    );
    cache.insert(
        cache_key,
        CacheEntry {
            outputs,
            compute_time_us: 1500,
            created: Instant::now(),
        },
    );

    // Second lookup: hit
    let result = cache.get(&cache_key);
    assert!(result.is_some());
    let (hits, misses, size) = cache.stats();
    println!("  After 2nd lookup:  hits={hits}, misses={misses} (HIT)");
    println!("  Cache entries:     {size}");

    // Benchmark lookup speed
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = cache.get(&cache_key);
    }
    let lookup_time = start.elapsed();
    println!(
        "  Lookup speed:      {:.0}ns per lookup (100k lookups in {:.2}ms)",
        lookup_time.as_nanos() as f64 / 100_000.0,
        lookup_time.as_secs_f64() * 1000.0,
    );
    println!();

    println!("================================================================");
    println!("  Benchmark complete.");
    println!("================================================================");
}
