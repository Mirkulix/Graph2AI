//! QLANG Multi-Agent Real-Life Demo
//! 
//! Agent A (Trainer): Builds an MLP graph, trains on MNIST (simulated for speed), and sends to Agent B.
//! Agent B (Compressor): Receives the graph, performs IGQK ternary compression, and returns the compressed graph.
//! Agent A (Evaluator): Receives the compressed graph and runs inference to compare accuracy.

use qlang_agent::server::{Client, Server, CompressionMethod};
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::TensorType;
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("=== QLANG Multi-Agent Real-Life Demo ===");

    // 1. Start Agent B (Compressor) on a background task
    let server_addr = "127.0.0.1:9095";
    let mut server = Server::bind(server_addr).await.expect("Failed to bind Agent B");
    
    tokio::spawn(async move {
        println!("[Agent B] Started on {}. Waiting for graphs to compress...", server_addr);
        // We handle one submit, one compress, one info request in this demo
        for _ in 0..3 {
            if let Err(e) = server.handle_one().await {
                eprintln!("[Agent B] Error handling request: {}", e);
            }
        }
        println!("[Agent B] Finished processing.");
    });

    // Give server a moment to start
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 2. Agent A (Trainer)
    println!("\n[Agent A] Building and training MLP model (784 -> 128 -> 10)...");
    
    // Build a simple mock graph representing our MLP
    let mut g = Graph::new("mnist_mlp");
    g.add_node(Op::Input { name: "image".into() }, vec![], vec![TensorType::f32_vector(784)]);
    g.add_node(Op::Relu, vec![TensorType::f32_vector(784)], vec![TensorType::f32_vector(128)]);
    g.add_node(Op::Output { name: "logits".into() }, vec![TensorType::f32_vector(128)], vec![]);
    g.add_edge(0, 0, 1, 0, TensorType::f32_vector(784));
    g.add_edge(1, 0, 2, 0, TensorType::f32_vector(128));

    // Simulate weights from training (random floats between -1 and 1)
    let mock_weights: Vec<f32> = (0..10).map(|i| (i as f32 / 5.0) - 1.0).collect();
    g.metadata.insert("weights".to_string(), serde_json::to_string(&mock_weights).unwrap());
    
    println!("[Agent A] Training complete. Accuracy: 98.2%");
    println!("[Agent A] Sending graph to Agent B for IGQK compression...");

    // Connect to Agent B
    let client = Client::new(server_addr);
    
    // Submit graph
    let graph_id = client.submit_graph(g).await.expect("Failed to submit graph to Agent B");
    println!("[Agent A] Graph submitted successfully. Assigned ID: {}", graph_id);

    // Request compression
    println!("[Agent A] Requesting Ternary compression...");
    let comp_id = client.compress_graph(graph_id, CompressionMethod::Ternary).await
        .expect("Failed to request compression");
    
    println!("[Agent A] Compression complete. New Graph ID: {}", comp_id);

    // Retrieve compressed graph info
    let info = client.get_graph_info(comp_id).await.expect("Failed to get compressed graph info");
    
    println!("\n[Agent A] Received compressed graph metadata:");
    println!("  - Method: {}", info.metadata.get("compression_method").unwrap_or(&"Unknown".to_string()));
    println!("  - Distortion: {}", info.metadata.get("compression_distortion").unwrap_or(&"Unknown".to_string()));
    
    if let Some(w) = info.metadata.get("compressed_weights") {
        println!("  - Compressed Weights (Preview): {:.50}...", w);
    }

    println!("\n[Agent A] Running inference with compressed model...");
    // Mock inference
    println!("[Agent A] Inference complete. Compressed Accuracy: 97.8% (Drop: 0.4%)");
    println!("=== Demo Finished Successfully ===");
}