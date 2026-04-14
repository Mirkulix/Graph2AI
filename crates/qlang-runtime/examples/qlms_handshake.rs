//! QLMS Handshake Demo — two in-process agents exchange a trained TernaryBrain
//! specialist over the QLMS binary wire format (HMAC-SHA256 signed).
//!
//! Run with:
//!   cargo run --release --example qlms_handshake --no-default-features
//!
//! No HTTP, no network — purely in-process byte buffer exchange to isolate
//! the cost of the serialization + signature verification.

use qlang_core::crypto::{ct_eq, hmac_sha256, sha256};
use qlang_runtime::mnist::MnistData;
use qlang_runtime::ternary_brain::TernaryBrain;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---- QLMS wire format constants ----
const QLMS_MAGIC: &[u8; 4] = b"QLMS";
const QLMS_VERSION: u16 = 1;
const QLMS_KIND_MODEL: u16 = 0x0001;
const SHARED_SECRET: &[u8] = b"qlms-handshake-demo-2026";

fn shared_key() -> [u8; 32] {
    sha256(SHARED_SECRET)
}

// ---- Payload (matches qlms_benchmark.rs layout) ----
struct ModelPayload<'a> {
    specialist_id: &'a str,
    image_dim: u32,
    n_classes: u32,
    class_names: &'a [String],
    timestamp_ms: u64,
    weights: &'a [i8],
}

fn qlms_encode_payload(p: &ModelPayload) -> Vec<u8> {
    let total_weights = p.weights.len() as u32;
    let mut buf = Vec::with_capacity(64 + p.specialist_id.len() + p.weights.len());
    let id_bytes = p.specialist_id.as_bytes();
    buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf.extend_from_slice(&p.image_dim.to_le_bytes());
    buf.extend_from_slice(&p.n_classes.to_le_bytes());
    buf.extend_from_slice(&total_weights.to_le_bytes());
    buf.extend_from_slice(&(p.class_names.len() as u16).to_le_bytes());
    for name in p.class_names {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        buf.extend_from_slice(nb);
    }
    buf.extend_from_slice(&p.timestamp_ms.to_le_bytes());
    let wbytes: &[u8] =
        unsafe { std::slice::from_raw_parts(p.weights.as_ptr() as *const u8, p.weights.len()) };
    buf.extend_from_slice(wbytes);
    buf
}

fn qlms_encode_frame(payload: &[u8]) -> (Vec<u8>, [u8; 32]) {
    let sig = hmac_sha256(&shared_key(), payload);
    let mut buf = Vec::with_capacity(4 + 2 + 2 + 32 + 4 + payload.len());
    buf.extend_from_slice(QLMS_MAGIC);
    buf.extend_from_slice(&QLMS_VERSION.to_le_bytes());
    buf.extend_from_slice(&QLMS_KIND_MODEL.to_le_bytes());
    buf.extend_from_slice(&sig);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(payload);
    (buf, sig)
}

struct DecodedPayload {
    specialist_id: String,
    image_dim: u32,
    n_classes: u32,
    #[allow(dead_code)]
    class_names: Vec<String>,
    #[allow(dead_code)]
    timestamp_ms: u64,
    weights: Vec<i8>,
}

fn qlms_decode_frame(data: &[u8]) -> Result<DecodedPayload, String> {
    if data.len() < 4 + 2 + 2 + 32 + 4 {
        return Err("frame too short".into());
    }
    if &data[0..4] != QLMS_MAGIC {
        return Err("bad magic".into());
    }
    let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
    if version != QLMS_VERSION {
        return Err(format!("bad version: {}", version));
    }
    let mut sig = [0u8; 32];
    sig.copy_from_slice(&data[8..40]);
    let payload_len = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
    if data.len() < 44 + payload_len {
        return Err("truncated payload".into());
    }
    let payload = &data[44..44 + payload_len];
    let expected = hmac_sha256(&shared_key(), payload);
    if !ct_eq(&expected, &sig) {
        return Err("HMAC mismatch".into());
    }
    let mut o = 0usize;
    let id_len = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
    o += 2;
    let id = String::from_utf8(payload[o..o + id_len].to_vec()).map_err(|e| e.to_string())?;
    o += id_len;
    let image_dim = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
    o += 4;
    let n_classes = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
    o += 4;
    let total_weights = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
    o += 4;
    let n_names = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
    o += 2;
    let mut class_names = Vec::with_capacity(n_names);
    for _ in 0..n_names {
        let nlen = u16::from_le_bytes(payload[o..o + 2].try_into().unwrap()) as usize;
        o += 2;
        class_names
            .push(String::from_utf8(payload[o..o + nlen].to_vec()).map_err(|e| e.to_string())?);
        o += nlen;
    }
    let timestamp_ms = u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
    o += 8;
    let weights: Vec<i8> = payload[o..o + total_weights].iter().map(|&b| b as i8).collect();
    Ok(DecodedPayload {
        specialist_id: id,
        image_dim,
        n_classes,
        class_names,
        timestamp_ms,
        weights,
    })
}

fn hex_short(bytes: &[u8]) -> String {
    let head: String = bytes[..2].iter().map(|b| format!("{:02x}", b)).collect();
    let tail: String = bytes[bytes.len() - 2..]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    format!("{}...{}", head, tail)
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn main() {
    println!("QLMS Handshake Demo");
    println!("===================");
    println!();

    // ===================================================================
    // Agent A: train a TernaryBrain specialist on a small MNIST subset.
    // ===================================================================
    println!("Agent A: Training TernaryBrain on 500 MNIST samples...");
    let data = MnistData::synthetic(500, 200);
    let mut brain_a = TernaryBrain::init(
        &data.train_images,
        &data.train_labels,
        784,
        data.n_train,
        10,
        6, // 6 neurons/class → 60 neurons × 784 = 47040 ternary weights
    );
    brain_a.refine(&data.train_images, &data.train_labels, data.n_train, 3);
    let acc = brain_a.accuracy(&data.test_images, &data.test_labels, data.n_test);
    println!(
        "Agent A: Specialist trained (accuracy on holdout: {:.1}%)",
        acc * 100.0
    );
    println!();

    // ===================================================================
    // Agent A: encode as QLMS binary frame.
    // ===================================================================
    println!("Agent A: Encoding as QLMS...");
    let weights: Vec<i8> = brain_a.dump_weights_i8();
    let class_names: Vec<String> = (0..10).map(|i| format!("digit_{}", i)).collect();
    let specialist_id = format!("mnist-ternary-{}", now_ms());
    let payload = ModelPayload {
        specialist_id: &specialist_id,
        image_dim: brain_a.image_dim as u32,
        n_classes: brain_a.n_classes as u32,
        class_names: &class_names,
        timestamp_ms: now_ms(),
        weights: &weights,
    };

    let t_enc = Instant::now();
    let payload_bytes = qlms_encode_payload(&payload);
    let (frame, sig) = qlms_encode_frame(&payload_bytes);
    let enc_us = t_enc.elapsed().as_micros();

    println!("  Payload:     {} ternary weights (i8)", weights.len());
    println!(
        "  Frame size:  {} bytes (44B header + {}B payload)",
        frame.len(),
        payload_bytes.len()
    );
    println!("  HMAC:        {} (SHA-256 truncated)", hex_short(&sig));
    println!("  Encode time: {} µs", enc_us);
    println!();

    // ===================================================================
    // Agent B: receive bytes, verify signature, decode.
    // ===================================================================
    println!("Agent B: Receiving...");
    let t_rtt = Instant::now();
    let received: &[u8] = &frame; // in-process byte handoff
    let t_dec = Instant::now();
    let decoded = match qlms_decode_frame(received) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  Decode failed: {}", e);
            std::process::exit(1);
        }
    };
    let dec_us = t_dec.elapsed().as_micros();
    let rtt_us = t_rtt.elapsed().as_micros();

    println!("  Magic verified: QLMS");
    println!("  HMAC verified: YES");
    println!("  Decode time:   {} µs", dec_us);
    println!("  Total RTT:     {} µs (in-process)", rtt_us);
    println!();

    // ===================================================================
    // Agent B: reconstruct brain from template + weights, run inference.
    // ===================================================================
    println!("Agent B: Running inference on test sample...");
    // Build a same-topology template (neurons/class must match Agent A's)
    let template = TernaryBrain::init(
        &data.train_images,
        &data.train_labels,
        784,
        data.n_train,
        decoded.n_classes as usize,
        6,
    );
    let brain_b = TernaryBrain::from_template_and_weights(&template, &decoded.weights)
        .expect("reconstruct brain from received weights");
    assert!(brain_b.verify_ternary(), "received weights must be ternary");

    // Pick first test sample
    let sample_idx = 0usize;
    let x = &data.test_images[sample_idx * 784..(sample_idx + 1) * 784];
    let actual = data.test_labels[sample_idx];
    let scores = brain_b.layer.predict_one(x, brain_b.n_classes);
    let (pred, &top) = scores
        .iter()
        .enumerate()
        .max_by_key(|(_, &s)| s)
        .unwrap();
    let sum: i32 = scores.iter().filter(|&&s| s > 0).sum();
    let confidence = if sum > 0 { top as f32 / sum as f32 } else { 0.0 };
    let correct = pred as u8 == actual;
    println!(
        "  Prediction: {}  (confidence {:.2})",
        pred, confidence
    );
    println!(
        "  Actual:     {}  {}",
        actual,
        if correct { "CORRECT" } else { "WRONG" }
    );
    println!();

    // Sanity: specialist_id round-tripped
    assert_eq!(decoded.specialist_id, specialist_id);
    assert_eq!(decoded.image_dim, 784);

    // ===================================================================
    // Tamper test: flip one bit, confirm Agent B rejects.
    // ===================================================================
    println!("Tamper test:");
    let tamper_offset = 100usize;
    println!("  Flipping bit at offset {}", tamper_offset);
    let mut tampered = frame.clone();
    tampered[tamper_offset] ^= 0x01;
    match qlms_decode_frame(&tampered) {
        Ok(_) => {
            println!("  Agent B verification: ACCEPTED (UNEXPECTED — tamper detection failed)");
            std::process::exit(2);
        }
        Err(e) => {
            println!("  Agent B verification: REJECTED ({})  ✓", e);
        }
    }
}
