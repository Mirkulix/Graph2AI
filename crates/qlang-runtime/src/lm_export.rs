//! Export trained LM as QLANG graph for portable inference.
//!
//! Takes a trained TrainableLM and exports it as:
//! 1. A QLANG graph (.qlbg) describing the architecture
//! 2. Weight data (f32 or ternary packed)
//!
//! The exported model can be loaded and executed by the QLANG executor
//! or sent to another agent via the MessageBus.

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dtype, Shape, TensorType};
use qlang_core::binary;
use crate::mamba_train::TrainableLM;
use crate::ternary_ops;

/// Export a trained LM as a QLANG inference graph.
///
/// The graph structure:
///   Input(token_id) → Embedding → MambaLayer1 → MambaLayer2 → OutputHead → Softmax → Output
pub fn export_graph(lm: &TrainableLM) -> Graph {
    let d = lm.d_model;
    let v = lm.vocab_size;

    let mut g = Graph::new(format!("qlang_lm_d{}_v{}", d, v));

    let f32_token = TensorType::new(Dtype::F32, Shape::scalar());
    let f32_embed = TensorType::new(Dtype::F32, Shape::matrix(v, d));
    let f32_seq = TensorType::new(Dtype::F32, Shape::vector(d));
    let f32_logits = TensorType::new(Dtype::F32, Shape::vector(v));

    // Input: token ID
    let input = g.add_node(Op::Input { name: "token".into() }, vec![], vec![f32_token.clone()]);

    // Embedding table (stored as input for now)
    let embed_table = g.add_node(Op::Input { name: "embedding".into() }, vec![], vec![f32_embed.clone()]);

    // Embedding lookup
    let embed = g.add_node(
        Op::Embedding { vocab_size: v, d_model: d },
        vec![f32_token.clone(), f32_embed.clone()],
        vec![f32_seq.clone()],
    );

    // Mamba layers (simplified as MatMul + ReLU chains in the graph)
    let mut prev = embed;
    for (i, _layer) in lm.layers.iter().enumerate() {
        let w_name = format!("mamba_{}_w", i);
        let w_input = g.add_node(
            Op::Input { name: w_name },
            vec![],
            vec![TensorType::new(Dtype::F32, Shape::matrix(d, d))],
        );
        let matmul = g.add_node(
            Op::MatMul,
            vec![f32_seq.clone(), TensorType::new(Dtype::F32, Shape::matrix(d, d))],
            vec![f32_seq.clone()],
        );
        let relu = g.add_node(Op::Relu, vec![f32_seq.clone()], vec![f32_seq.clone()]);
        let residual = g.add_node(Op::Residual, vec![f32_seq.clone(), f32_seq.clone()], vec![f32_seq.clone()]);

        g.add_edge(prev, 0, matmul, 0, f32_seq.clone());
        g.add_edge(w_input, 0, matmul, 1, TensorType::new(Dtype::F32, Shape::matrix(d, d)));
        g.add_edge(matmul, 0, relu, 0, f32_seq.clone());
        g.add_edge(relu, 0, residual, 0, f32_seq.clone());
        g.add_edge(prev, 0, residual, 1, f32_seq.clone());

        prev = residual;
    }

    // Output head
    let w_out = g.add_node(Op::Input { name: "output_head".into() }, vec![], vec![TensorType::new(Dtype::F32, Shape::matrix(d, v))]);
    let logits = g.add_node(
        Op::MatMul,
        vec![f32_seq.clone(), TensorType::new(Dtype::F32, Shape::matrix(d, v))],
        vec![f32_logits.clone()],
    );
    let softmax = g.add_node(Op::Softmax { axis: 0 }, vec![f32_logits.clone()], vec![f32_logits.clone()]);
    let output = g.add_node(Op::Output { name: "probs".into() }, vec![f32_logits.clone()], vec![]);

    g.add_edge(prev, 0, logits, 0, f32_seq.clone());
    g.add_edge(w_out, 0, logits, 1, TensorType::new(Dtype::F32, Shape::matrix(d, v)));
    g.add_edge(logits, 0, softmax, 0, f32_logits.clone());
    g.add_edge(softmax, 0, output, 0, f32_logits.clone());
    g.add_edge(input, 0, embed, 0, f32_token);
    g.add_edge(embed_table, 0, embed, 1, f32_embed);

    // Metadata
    g.metadata.insert("model_type".into(), "qlang_lm".into());
    g.metadata.insert("d_model".into(), d.to_string());
    g.metadata.insert("vocab_size".into(), v.to_string());
    g.metadata.insert("n_layers".into(), lm.layers.len().to_string());
    g.metadata.insert("params".into(), lm.param_count().to_string());

    g
}

/// Export model as .qlbg binary + weights file.
pub fn save_model(lm: &TrainableLM, graph_path: &str, weights_path: &str) -> Result<(), String> {
    // 1. Export graph
    let graph = export_graph(lm);
    let graph_binary = binary::to_binary(&graph);
    std::fs::write(graph_path, &graph_binary).map_err(|e| format!("write graph: {e}"))?;

    // 2. Export weights
    let mut weight_data = Vec::new();

    // Magic + version
    weight_data.extend_from_slice(&[0x51, 0x4C, 0x57, 0x54]); // "QLWT" = QLANG Weights
    weight_data.extend_from_slice(&1u32.to_le_bytes()); // version

    // Embedding
    write_tensor(&mut weight_data, &lm.embedding);
    // Output head
    write_tensor(&mut weight_data, &lm.output_head);
    // Mamba layers
    for layer in &lm.layers {
        write_tensor(&mut weight_data, &layer.w_x);
        write_tensor(&mut weight_data, &layer.w_h);
        write_tensor(&mut weight_data, &layer.w_gate);
        write_tensor(&mut weight_data, &layer.w_out);
        write_tensor(&mut weight_data, &layer.b_h);
        write_tensor(&mut weight_data, &layer.b_gate);
    }

    std::fs::write(weights_path, &weight_data).map_err(|e| format!("write weights: {e}"))?;

    Ok(())
}

/// Export ternary weights (16x smaller).
pub fn save_model_ternary(lm: &mut TrainableLM, graph_path: &str, weights_path: &str) -> Result<(), String> {
    // Ternarize
    for layer in &mut lm.layers { layer.ternarize(); }

    // Export graph
    let graph = export_graph(lm);
    let graph_binary = binary::to_binary(&graph);
    std::fs::write(graph_path, &graph_binary).map_err(|e| format!("write graph: {e}"))?;

    // Export ternary-packed weights
    let mut weight_data = Vec::new();
    weight_data.extend_from_slice(&[0x51, 0x4C, 0x54, 0x57]); // "QLTW" = QLANG Ternary Weights
    weight_data.extend_from_slice(&1u32.to_le_bytes());

    // Embedding stays f32
    write_tensor(&mut weight_data, &lm.embedding);
    // Output head stays f32
    write_tensor(&mut weight_data, &lm.output_head);
    // Mamba layers — ternary packed
    for layer in &lm.layers {
        for w in [&layer.w_x, &layer.w_h, &layer.w_gate, &layer.w_out] {
            let (packed, alpha) = ternary_ops::pack_ternary(w);
            weight_data.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            weight_data.extend_from_slice(&packed);
            weight_data.extend_from_slice(&alpha.to_le_bytes());
        }
        write_tensor(&mut weight_data, &layer.b_h);
        write_tensor(&mut weight_data, &layer.b_gate);
    }

    std::fs::write(weights_path, &weight_data).map_err(|e| format!("write weights: {e}"))?;
    Ok(())
}

fn write_tensor(buf: &mut Vec<u8>, data: &[f32]) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_lm_graph() {
        let text = "the cat sat on the mat the dog ran in the park";
        let lm = TrainableLM::new(text, 32, 64, 2, 20);

        let graph = export_graph(&lm);
        println!("Exported graph: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        println!("Metadata: {:?}", graph.metadata);

        let binary = binary::to_binary(&graph);
        println!("QLBG size: {} bytes", binary.len());

        // Roundtrip
        let restored = binary::from_binary(&binary).unwrap();
        assert_eq!(restored.nodes.len(), graph.nodes.len());
        println!("Roundtrip: OK");
    }

    #[test]
    fn save_and_load_model() {
        let text = "the cat sat on the mat";
        let lm = TrainableLM::new(text, 16, 32, 1, 10);

        let graph_path = "/tmp/test_lm.qlbg";
        let weights_path = "/tmp/test_lm.qlwt";

        save_model(&lm, graph_path, weights_path).unwrap();

        let graph_size = std::fs::metadata(graph_path).unwrap().len();
        let weights_size = std::fs::metadata(weights_path).unwrap().len();

        println!("Graph: {} bytes, Weights: {} bytes", graph_size, weights_size);
        println!("Total: {} bytes ({:.1} KB)", graph_size + weights_size, (graph_size + weights_size) as f64 / 1024.0);

        assert!(graph_size > 0);
        assert!(weights_size > 0);
    }
}
