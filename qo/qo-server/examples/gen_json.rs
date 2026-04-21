use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, Dim, TensorType, TensorData};
use qlang_core::crypto::{Keypair};
use qlang_agent::protocol::{GraphMessage, AgentId, MessageIntent, Capability};
use base64::Engine;
use std::collections::HashMap;

fn main() {
    let kp = Keypair::from_seed(&[42u8; 32]); // Deterministic test key
    let mut g = Graph::new("collective_demo");
    let utf8_dynamic = TensorType::new(Dtype::Utf8, Shape(vec![Dim::Dynamic]));
    
    g.add_node(Op::Input { name: "prompt".into() }, vec![], vec![utf8_dynamic.clone()]);
    g.add_node(Op::OllamaChat { model: "mistral".into() }, vec![utf8_dynamic.clone()], vec![utf8_dynamic.clone()]);
    g.add_node(Op::OllamaChat { model: "mistral".into() }, vec![utf8_dynamic.clone()], vec![utf8_dynamic.clone()]);
    g.add_node(Op::Output { name: "result".into() }, vec![utf8_dynamic.clone()], vec![]);
    
    g.add_edge(0, 0, 1, 0, utf8_dynamic.clone());
    g.add_edge(1, 0, 2, 0, utf8_dynamic.clone());
    g.add_edge(2, 0, 3, 0, utf8_dynamic.clone());

    let mut inputs = HashMap::new();
    inputs.insert("prompt".into(), TensorData::from_string("Hallo Welt"));

    let msg = GraphMessage {
        id: 123,
        from: AgentId { name: "user_ide".into(), capabilities: vec![Capability::Execute] },
        to: AgentId { name: "researcher".into(), capabilities: vec![Capability::Execute] },
        graph: g,
        inputs,
        intent: MessageIntent::Execute,
        in_reply_to: None,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    };

    let mut conv = qlang_agent::protocol::AgentConversation::new();
    conv.send(msg.from, msg.to, msg.graph, msg.inputs, msg.intent, msg.in_reply_to);

    // Use signed binary (v2)
    let bin = conv.to_signed_binary(&kp).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bin);

    let req = serde_json::json!({
        "encoding": "base64",
        "frame": b64
    });

    println!("{}", serde_json::to_string_pretty(&req).unwrap());
}
