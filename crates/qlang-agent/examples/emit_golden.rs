//! Golden-fixture emitter for the Python QLMS reference parser.
//!
//! Produces deterministic binary dumps under
//! `bindings/python/qlms_parser/fixtures/` that the Python side reads to
//! prove language-independent parsing + HMAC-SHA256 verification (spec
//! §16.4 item 6 / Task 1.3 of the A2A-qlang development plan).
//!
//! Run:
//!
//! ```bash
//! cargo run -p qlang-agent --example emit_golden
//! ```
//!
//! Each fixture pairs a `.qlms` binary dump with a `.json` sidecar that
//! records the metadata the Python test asserts on — seed, expected
//! pubkey, payload hash, flags, etc. The sidecars are human-readable and
//! make the test auditable without a hex dump.

use qlang_agent::protocol::{
    AgentConversation, AgentId, Capability, MessageIntent,
};
use qlang_core::crypto::{hex, Keypair};
use qlang_core::graph::Graph;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

fn trainer() -> AgentId {
    AgentId {
        name: "trainer".into(),
        capabilities: vec![Capability::Execute, Capability::Train],
    }
}

fn compressor() -> AgentId {
    AgentId {
        name: "compressor".into(),
        capabilities: vec![Capability::Compress, Capability::Verify],
    }
}

/// Resolve the fixtures directory relative to the workspace root.
///
/// Uses `CARGO_MANIFEST_DIR` of the `qlang-agent` crate (…/crates/qlang-agent)
/// and walks up to the repo root, then down into the bindings tree.
fn fixtures_dir() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = crate_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // repo root
        .expect("qlang-agent must live under <repo>/crates/qlang-agent");
    workspace_root
        .join("bindings")
        .join("python")
        .join("qlms_parser")
        .join("fixtures")
}

/// Emit a single fixture pair: `<name>.qlms` (binary) + `<name>.json` (metadata).
fn emit(dir: &Path, name: &str, frame: &[u8], meta: serde_json::Value) {
    let bin_path = dir.join(format!("{name}.qlms"));
    let meta_path = dir.join(format!("{name}.json"));
    fs::write(&bin_path, frame).expect("write qlms fixture");
    fs::write(
        &meta_path,
        serde_json::to_vec_pretty(&meta).expect("serialize meta"),
    )
    .expect("write meta fixture");
    println!(
        "  wrote {} ({} bytes) + {}",
        bin_path.display(),
        frame.len(),
        meta_path.display()
    );
}

fn main() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");
    println!("emitting QLMS golden fixtures → {}", dir.display());

    // ---- Fixture 1: v1 unsigned envelope (JSON-in-QLMS, no auth block) ----
    {
        let mut conv = AgentConversation::new();
        let g = Graph::new("golden_v1_unsigned");
        conv.send(
            trainer(),
            compressor(),
            g,
            HashMap::new(),
            MessageIntent::Verify,
            None,
        );
        let bin = conv
            .to_binary()
            .expect("v1 serialization cannot fail on well-formed graph");
        let meta = json!({
            "kind": "v1_unsigned",
            "msg_count": 1,
            "note": "v1 wire: [QLMS magic, u32 LE count, JSON payload]",
        });
        emit(&dir, "v1_unsigned_single", &bin, meta);
    }

    // ---- Fixture 2: v2 signed envelope (deterministic keypair) ----
    {
        let seed = [0x2Au8; 32];
        let kp = Keypair::from_seed(&seed);
        let mut conv = AgentConversation::new();
        let g = Graph::new("golden_v2_signed");
        conv.send(
            trainer(),
            compressor(),
            g,
            HashMap::new(),
            MessageIntent::Execute,
            None,
        );
        let bin = conv
            .to_signed_binary(&kp)
            .expect("v2 serialization cannot fail on well-formed graph");
        let meta = json!({
            "kind": "v2_signed",
            "msg_count": 1,
            "seed_hex": hex(&seed),
            "expected_pubkey_hex": hex(&kp.public_key()),
            "flags": 0x0001,
            "version": 2,
            "note": "v2 wire: [QLMS magic, u16 version, u16 flags, 64B sig, 32B pubkey, 32B payload_hash, u32 msg_count, JSON payload]",
        });
        emit(&dir, "v2_signed_single", &bin, meta);
    }

    // ---- Fixture 3: v2 signed envelope with a multi-message conversation ----
    {
        let seed = [0x7Fu8; 32];
        let kp = Keypair::from_seed(&seed);
        let mut conv = AgentConversation::new();
        let g1 = Graph::new("golden_multi_1");
        let g2 = Graph::new("golden_multi_2");
        let id1 = conv.send(
            trainer(),
            compressor(),
            g1,
            HashMap::new(),
            MessageIntent::Compress {
                method: "ternary".into(),
            },
            None,
        );
        conv.send(
            compressor(),
            trainer(),
            g2,
            HashMap::new(),
            MessageIntent::Result {
                original_message_id: id1,
            },
            Some(id1),
        );
        let bin = conv
            .to_signed_binary(&kp)
            .expect("v2 multi-message serialization");
        let meta = json!({
            "kind": "v2_signed",
            "msg_count": 2,
            "seed_hex": hex(&seed),
            "expected_pubkey_hex": hex(&kp.public_key()),
            "flags": 0x0001,
            "version": 2,
            "note": "v2 with two chained messages (Compress → Result)",
        });
        emit(&dir, "v2_signed_two_messages", &bin, meta);
    }

    println!("done — {} fixtures emitted.", 3);
}
