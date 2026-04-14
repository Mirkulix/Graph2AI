//! Benchmark: QLANG binary wire format vs JSON for AI tensor transport.
//!
//! Run: cargo run --release --no-default-features --example benchmark_qlang_vs_json
//!
//! Compares:
//! 1. Serialization size (bytes) -- 768-dim embedding vector
//! 2. Precision (float roundtrip accuracy)
//! 3. Serialization speed (encode + decode, 10 000 iterations)
//! 4. Graph signing throughput (SHA-256 + HMAC-SHA256)

use qlang_core::crypto;
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData, TensorType};
use std::time::Instant;

fn main() {
    println!("================================================================");
    println!("  QLANG Binary Wire Format vs JSON -- Benchmark");
    println!("================================================================");
    println!();

    // =====================================================================
    // 1. SIZE COMPARISON
    // =====================================================================
    println!("--- 1. Size Comparison (768-dim f32 embedding vector) ---\n");

    let values_768: Vec<f32> = (0..768)
        .map(|i| ((i as f64) * 0.001 * std::f64::consts::PI).sin() as f32)
        .collect();
    let tensor_768 = TensorData::from_f32(Shape::vector(768), &values_768);

    // JSON size
    let json_bytes = serde_json::to_vec(&tensor_768).unwrap();
    let json_size = json_bytes.len();

    // QLANG binary wire size
    let wire_bytes = tensor_768.to_wire_bytes();
    let wire_size = wire_bytes.len();

    let size_ratio = json_size as f64 / wire_size as f64;
    println!("  JSON size:          {:>8} bytes", json_size);
    println!("  QLANG binary size:  {:>8} bytes", wire_size);
    println!("  Ratio:              {:.1}x smaller with QLANG", size_ratio);
    println!();

    // Also show what the theoretical minimum is (just the raw floats)
    let raw_size = 768 * 4; // 768 floats * 4 bytes each
    println!(
        "  Raw f32 data:       {:>8} bytes (theoretical minimum)",
        raw_size
    );
    println!(
        "  QLANG overhead:     {:>8} bytes ({:.1}% header)",
        wire_size - raw_size,
        ((wire_size - raw_size) as f64 / wire_size as f64) * 100.0
    );
    println!();

    // =====================================================================
    // 2. PRECISION COMPARISON
    // =====================================================================
    println!("--- 2. Precision Comparison (float roundtrip accuracy) ---\n");

    let hard_values: Vec<f32> = vec![
        std::f32::consts::PI,
        std::f32::consts::E,
        1.0 / 3.0,
        f32::MIN_POSITIVE,
        f32::MAX,
    ];
    let labels = ["pi", "e", "1/3", "MIN_POSITIVE", "MAX"];
    let tensor_precise = TensorData::from_f32(Shape::vector(hard_values.len()), &hard_values);

    // JSON roundtrip
    let json = serde_json::to_vec(&tensor_precise).unwrap();
    let json_decoded: TensorData = serde_json::from_slice(&json).unwrap();
    let json_values = json_decoded.as_f32_slice().unwrap();

    // QLANG binary roundtrip
    let wire = tensor_precise.to_wire_bytes();
    let wire_decoded = TensorData::from_wire_bytes(&wire).unwrap();
    let wire_values = wire_decoded.as_f32_slice().unwrap();

    println!(
        "  {:>14}  {:>15}  {:>15}  {:>12}  {:>12}",
        "Value", "Original", "JSON roundtrip", "JSON err", "QLANG err"
    );
    println!("  {}", "-".repeat(74));

    let mut json_max_err: f64 = 0.0;
    let mut wire_max_err: f64 = 0.0;

    for i in 0..hard_values.len() {
        let orig = hard_values[i];
        let j_val = json_values[i];
        let w_val = wire_values[i];
        let j_err = ((orig as f64) - (j_val as f64)).abs();
        let w_err = ((orig as f64) - (w_val as f64)).abs();
        json_max_err = json_max_err.max(j_err);
        wire_max_err = wire_max_err.max(w_err);
        println!(
            "  {:>14}  {:>15.8e}  {:>15.8e}  {:>12.2e}  {:>12.2e}",
            labels[i], orig, j_val, j_err, w_err
        );
    }
    println!();
    println!(
        "  JSON max error:  {:.2e}{}",
        json_max_err,
        if json_max_err > 0.0 {
            "  (precision lost in text conversion)"
        } else {
            ""
        }
    );
    println!(
        "  QLANG max error: {:.2e}{}",
        wire_max_err,
        if wire_max_err == 0.0 {
            "  (bit-exact: binary preserves IEEE 754)"
        } else {
            ""
        }
    );
    println!();

    // =====================================================================
    // 3. SPEED COMPARISON
    // =====================================================================
    println!("--- 3. Speed Comparison (10 000 encode+decode cycles, 768-dim) ---\n");

    let iterations = 10_000;

    // -- JSON encode --
    let start = Instant::now();
    for _ in 0..iterations {
        let v = serde_json::to_vec(&tensor_768).unwrap();
        std::hint::black_box(&v);
    }
    let json_encode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // -- JSON decode --
    let start = Instant::now();
    for _ in 0..iterations {
        let t: TensorData = serde_json::from_slice(&json_bytes).unwrap();
        std::hint::black_box(&t);
    }
    let json_decode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // -- QLANG binary encode --
    let start = Instant::now();
    for _ in 0..iterations {
        let v = tensor_768.to_wire_bytes();
        std::hint::black_box(&v);
    }
    let wire_encode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    // -- QLANG binary decode --
    let start = Instant::now();
    for _ in 0..iterations {
        let t = TensorData::from_wire_bytes(&wire_bytes).unwrap();
        std::hint::black_box(&t);
    }
    let wire_decode_us = start.elapsed().as_micros() as f64 / iterations as f64;

    let encode_speedup = json_encode_us / wire_encode_us.max(0.001);
    let decode_speedup = json_decode_us / wire_decode_us.max(0.001);

    println!(
        "  {:>20}  {:>12}  {:>12}  {:>12}",
        "", "JSON", "QLANG binary", "Speedup"
    );
    println!("  {}", "-".repeat(60));
    println!(
        "  {:>20}  {:>10.1} us  {:>10.1} us  {:>10.1}x",
        "Encode (avg)", json_encode_us, wire_encode_us, encode_speedup
    );
    println!(
        "  {:>20}  {:>10.1} us  {:>10.1} us  {:>10.1}x",
        "Decode (avg)", json_decode_us, wire_decode_us, decode_speedup
    );
    println!(
        "  {:>20}  {:>10.1} us  {:>10.1} us  {:>10.1}x",
        "Roundtrip (avg)",
        json_encode_us + json_decode_us,
        wire_encode_us + wire_decode_us,
        (json_encode_us + json_decode_us) / (wire_encode_us + wire_decode_us).max(0.001)
    );
    println!();

    // =====================================================================
    // 4. GRAPH SIGNING OVERHEAD
    // =====================================================================
    println!("--- 4. Graph Signing Overhead (100-node graph, 1000 iterations) ---\n");

    let mut graph = Graph::new("benchmark");
    for i in 0..100 {
        graph.add_node(
            Op::Input {
                name: format!("x{i}"),
            },
            vec![],
            vec![TensorType::f32_vector(768)],
        );
    }

    let keypair = crypto::Keypair::from_seed(&[42u8; 32]);
    let sign_iterations = 1_000;

    // -- Hash --
    let start = Instant::now();
    for _ in 0..sign_iterations {
        let h = crypto::hash_graph(&graph);
        std::hint::black_box(&h);
    }
    let hash_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    // -- Sign --
    let start = Instant::now();
    for _ in 0..sign_iterations {
        let h = crypto::hash_graph(&graph);
        let s = keypair.sign(&h);
        std::hint::black_box(&s);
    }
    let sign_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    // -- Verify --
    let hash = crypto::hash_graph(&graph);
    let sig = keypair.sign(&hash);
    let start = Instant::now();
    for _ in 0..sign_iterations {
        let ok = crypto::Keypair::verify(keypair.public_key(), &hash, &sig);
        std::hint::black_box(&ok);
    }
    let verify_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    // -- SignedGraph (full sign + verify) --
    let start = Instant::now();
    for _ in 0..sign_iterations {
        let sg = crypto::SignedGraph::sign(graph.clone(), &keypair);
        let ok = sg.verify();
        std::hint::black_box(&ok);
    }
    let full_us = start.elapsed().as_micros() as f64 / sign_iterations as f64;

    println!("  {:>28}  {:>10}", "Operation", "Avg time");
    println!("  {}", "-".repeat(42));
    println!("  {:>28}  {:>8.1} us", "SHA-256 hash", hash_us);
    println!("  {:>28}  {:>8.1} us", "Hash + sign", sign_us);
    println!("  {:>28}  {:>8.1} us", "Verify signature", verify_us);
    println!(
        "  {:>28}  {:>8.1} us",
        "Full SignedGraph roundtrip", full_us
    );
    println!();

    let signs_per_sec = 1_000_000.0 / sign_us;
    println!(
        "  Throughput: {:.0} graph sign operations per second",
        signs_per_sec
    );
    println!(
        "  Verdict:    {}",
        if sign_us < 1000.0 {
            "Fast enough for real-time AI-to-AI communication"
        } else {
            "May need optimization for real-time use"
        }
    );
    println!();

    // =====================================================================
    // SUMMARY TABLE
    // =====================================================================
    println!("================================================================");
    println!("  Summary");
    println!("================================================================");
    println!();
    println!("  | Metric                  | JSON          | QLANG Binary    | Winner     |");
    println!("  |-------------------------|---------------|-----------------|------------|");
    println!(
        "  | Size (768-dim f32)      | {:>8} B     | {:>8} B       | QLANG {:.0}x  |",
        json_size, wire_size, size_ratio
    );
    println!(
        "  | Encode speed            | {:>8.1} us   | {:>8.1} us     | QLANG {:.0}x  |",
        json_encode_us, wire_encode_us, encode_speedup
    );
    println!(
        "  | Decode speed            | {:>8.1} us   | {:>8.1} us     | QLANG {:.0}x  |",
        json_decode_us, wire_decode_us, decode_speedup
    );
    println!(
        "  | Precision               | {:>12.2e} | {:>12.2e}    | {}      |",
        json_max_err,
        wire_max_err,
        if wire_max_err == 0.0 {
            "QLANG"
        } else {
            "tie"
        }
    );
    println!(
        "  | Graph signing           | n/a           | {:>8.1} us     | --         |",
        sign_us
    );
    println!();
}
