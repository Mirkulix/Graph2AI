//! QLMS Handshake Demo — two in-process agents exchange a signed ternary
//! specialist payload over the QLMS binary wire format.
//!
//! The demo stays fully local and deterministic: no network, no MNIST, no
//! legacy training modules. Agent A emits a small ternary weight vector;
//! Agent B verifies the HMAC and uses the received vector for a toy score.

use qlang_core::crypto::{ct_eq, hmac_sha256, sha256};
use qlang_runtime::federation::verify_ternary;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const QLMS_MAGIC: &[u8; 4] = b"QLMS";
const QLMS_VERSION: u16 = 1;
const QLMS_KIND_MODEL: u16 = 0x0001;
const SHARED_SECRET: &[u8] = b"qlms-handshake-demo-2026";

fn shared_key() -> [u8; 32] {
    sha256(SHARED_SECRET)
}

struct ModelPayload<'a> {
    specialist_id: &'a str,
    feature_dim: u32,
    n_classes: u32,
    class_names: &'a [String],
    timestamp_ms: u64,
    weights: &'a [i8],
}

struct DecodedPayload {
    specialist_id: String,
    feature_dim: u32,
    n_classes: u32,
    class_names: Vec<String>,
    timestamp_ms: u64,
    weights: Vec<i8>,
}

fn qlms_encode_payload(p: &ModelPayload) -> Vec<u8> {
    let total_weights = p.weights.len() as u32;
    let mut buf = Vec::with_capacity(64 + p.specialist_id.len() + p.weights.len());
    let id_bytes = p.specialist_id.as_bytes();
    buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(id_bytes);
    buf.extend_from_slice(&p.feature_dim.to_le_bytes());
    buf.extend_from_slice(&p.n_classes.to_le_bytes());
    buf.extend_from_slice(&total_weights.to_le_bytes());
    buf.extend_from_slice(&(p.class_names.len() as u16).to_le_bytes());
    for name in p.class_names {
        let nb = name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        buf.extend_from_slice(nb);
    }
    buf.extend_from_slice(&p.timestamp_ms.to_le_bytes());
    for &w in p.weights {
        buf.push(w as u8);
    }
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

fn qlms_decode_frame(data: &[u8]) -> Result<DecodedPayload, String> {
    if data.len() < 44 {
        return Err("frame too short".into());
    }
    if &data[0..4] != QLMS_MAGIC {
        return Err("bad magic".into());
    }
    let version = u16::from_le_bytes(data[4..6].try_into().unwrap());
    if version != QLMS_VERSION {
        return Err(format!("bad version: {version}"));
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
    let specialist_id =
        String::from_utf8(payload[o..o + id_len].to_vec()).map_err(|e| e.to_string())?;
    o += id_len;
    let feature_dim = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
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
        let name =
            String::from_utf8(payload[o..o + nlen].to_vec()).map_err(|e| e.to_string())?;
        o += nlen;
        class_names.push(name);
    }
    let timestamp_ms = u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
    o += 8;
    let weights = payload[o..o + total_weights].iter().map(|&b| b as i8).collect();

    Ok(DecodedPayload {
        specialist_id,
        feature_dim,
        n_classes,
        class_names,
        timestamp_ms,
        weights,
    })
}

fn hex_short(bytes: &[u8]) -> String {
    let head: String = bytes[..2].iter().map(|b| format!("{:02x}", b)).collect();
    let tail: String = bytes[bytes.len() - 2..].iter().map(|b| format!("{:02x}", b)).collect();
    format!("{head}...{tail}")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn make_demo_weights() -> Vec<i8> {
    vec![1, 0, -1, 1, 1, -1, 0, 1]
}

fn make_demo_sample() -> Vec<f32> {
    vec![0.8, 0.1, 0.2, 0.9, 1.0, 0.0, 0.4, 0.7]
}

fn dot_ternary(weights: &[i8], sample: &[f32]) -> f32 {
    weights
        .iter()
        .zip(sample.iter())
        .map(|(&w, &x)| w as f32 * x)
        .sum()
}

fn main() {
    println!("QLMS Handshake Demo");
    println!("===================");
    println!();

    let weights = make_demo_weights();
    let class_names: Vec<String> = vec!["reject".into(), "accept".into()];
    let specialist_id = format!("toy-ternary-{}", now_ms());
    let payload = ModelPayload {
        specialist_id: &specialist_id,
        feature_dim: weights.len() as u32,
        n_classes: class_names.len() as u32,
        class_names: &class_names,
        timestamp_ms: now_ms(),
        weights: &weights,
    };

    println!("Agent A: preparing ternary specialist payload...");
    println!("  Feature dim: {}", payload.feature_dim);
    println!("  Classes:     {}", payload.n_classes);
    println!("  Ternary OK:  {}", verify_ternary(&weights));

    let t_enc = Instant::now();
    let payload_bytes = qlms_encode_payload(&payload);
    let (frame, sig) = qlms_encode_frame(&payload_bytes);
    let enc_us = t_enc.elapsed().as_micros();

    println!("Agent A: encoded QLMS frame");
    println!("  Payload:     {} weights", weights.len());
    println!("  Frame size:  {} bytes", frame.len());
    println!("  HMAC:        {}", hex_short(&sig));
    println!("  Encode time: {} us", enc_us);
    println!();

    let t_rtt = Instant::now();
    let t_dec = Instant::now();
    let decoded = qlms_decode_frame(&frame).expect("decode verified frame");
    let dec_us = t_dec.elapsed().as_micros();
    let rtt_us = t_rtt.elapsed().as_micros();

    let sample = make_demo_sample();
    let score = dot_ternary(&decoded.weights, &sample);
    let prediction = if score >= 0.0 { 1usize } else { 0usize };
    let label = decoded
        .class_names
        .get(prediction)
        .cloned()
        .unwrap_or_else(|| format!("class_{prediction}"));

    println!("Agent B: verified and decoded specialist");
    println!("  Specialist:  {}", decoded.specialist_id);
    println!("  Timestamp:   {}", decoded.timestamp_ms);
    println!("  Ternary OK:  {}", verify_ternary(&decoded.weights));
    println!("  Decode time: {} us", dec_us);
    println!("  Total RTT:   {} us (in-process)", rtt_us);
    println!();

    println!("Agent B: toy inference with received weights");
    println!("  Score:       {:.3}", score);
    println!("  Prediction:  {} ({})", prediction, label);
    println!();
    println!("Handshake complete.");
}
