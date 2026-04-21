pub mod parser;

use qlang_agent::protocol::{AgentId, GraphMessage, MessageIntent};
use qlang_core::crypto::Keypair;
use std::collections::HashMap;

/// Takes a vector of raw Graphs (e.g. decoded from LLM output)
/// and wraps them in valid, signed QLMS Envelopes.
pub fn wrap_and_sign(
    graphs: Vec<qlang_core::graph::Graph>,
    from: AgentId,
    to: AgentId,
    intent: MessageIntent,
    keypair: Option<&Keypair>,
) -> Vec<GraphMessage> {
    graphs
        .into_iter()
        .map(|graph| {
            let mut msg = GraphMessage {
                id: qlang_agent::protocol::next_msg_id(),
                from: from.clone(),
                to: to.clone(),
                graph,
                inputs: HashMap::new(),
                intent: intent.clone(),
                in_reply_to: None,
                signature: None,
                signer_pubkey: None,
                graph_hash: None,
            };

            if let Some(kp) = keypair {
                msg = msg.sign(kp);
            }

            msg
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_agent::protocol::Capability;
    use qlang_core::graph::Graph;

    #[test]
    fn wraps_and_signs() {
        let from = AgentId {
            name: "test_from".into(),
            capabilities: vec![Capability::Execute],
        };
        let to = AgentId {
            name: "test_to".into(),
            capabilities: vec![Capability::Execute],
        };
        let kp = Keypair::from_seed(&[42u8; 32]);

        let graph = Graph::new("my_graph");
        let results = wrap_and_sign(vec![graph], from, to, MessageIntent::Execute, Some(&kp));

        assert_eq!(results.len(), 1);
        assert!(results[0].signature.is_some());
        assert!(results[0].verify_signature());
    }
}
