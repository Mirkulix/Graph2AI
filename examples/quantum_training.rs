//! IGQK Quantum Training Example
//!
//! Demonstrates the IGQK training algorithm (Algorithm 1 from the paper):
//!   1. Initialize quantum state ρ₀
//!   2. Evolve via quantum gradient flow
//!   3. Project onto compression submanifold
//!   4. Measure to get discrete weights
//!
//! This is a KI-zu-KI workflow: one agent trains, another compresses.

use std::collections::HashMap;

fn main() {
    println!("=== IGQK Quantum Training Pipeline ===\n");

    use qlang_core::quantum::{DensityMatrix, EvolutionParams};
    use qlang_core::tensor::{Dtype, Shape, TensorData, TensorType as TT};

    // ─── Step 1: Initialize quantum state ─────────
    println!("[1] Initializing quantum state ρ₀...");

    let dim = 8; // 8-dimensional Hilbert space (represents 8 weight parameters)
    let initial_weights = vec![0.5, -0.3, 0.8, -0.1, 0.4, -0.6, 0.2, -0.7];
    let rho = DensityMatrix::pure_state(&initial_weights);

    println!("  ρ₀ = {rho}");
    println!("  Pure state (rank 1): all probability on one configuration");
    println!("  Entropy S(ρ₀) = {:.6} (zero = pure, log(d)={:.4} = maximally mixed)",
        rho.entropy(), (dim as f64).ln());

    // ─── Step 2: Quantum gradient flow evolution ─────────
    println!("\n[2] Evolving via quantum gradient flow...");
    println!("  dρ/dt = -i[H, ρ] - γ{{G⁻¹∇L, ρ}}");

    let params = EvolutionParams::default();
    println!("  Parameters: ℏ={}, γ={}, dt={}, steps={}",
        params.hbar, params.gamma, params.dt, params.steps);

    // Simulate evolution: gradually mix the pure state
    // In real IGQK, this would use the Laplace-Beltrami operator and Fisher metric
    let mut evolved = rho.clone();

    // Simulate 10 evolution steps, adding uncertainty at each step
    for step in 0..10 {
        // Mix with maximally mixed state (simulates quantum exploration)
        let mixed = DensityMatrix::maximally_mixed(dim);
        let mix_factor = 0.05; // 5% mixing per step

        // ρ_new = (1 - α)ρ_old + α·(I/d)
        let mut new_eigenvalues = Vec::new();
        for (i, &p) in evolved.eigenvalues.iter().enumerate() {
            let mixed_p = if i < mixed.eigenvalues.len() {
                mixed.eigenvalues[i]
            } else {
                0.0
            };
            new_eigenvalues.push((1.0 - mix_factor) * p + mix_factor * mixed_p);
        }
        // Add new eigenvalues from mixed state if evolved has fewer
        for i in evolved.eigenvalues.len()..mixed.eigenvalues.len() {
            new_eigenvalues.push(mix_factor * mixed.eigenvalues[i]);
        }

        evolved.eigenvalues = new_eigenvalues;
        if evolved.eigenvectors.len() < dim * dim {
            evolved.eigenvectors = mixed.eigenvectors.clone();
        }
        evolved.renormalize();

        if step % 3 == 0 {
            println!("  Step {}: S(ρ)={:.4}, Tr(ρ)={:.6}, rank={}",
                step, evolved.entropy(), evolved.trace(), evolved.rank());
        }
    }

    println!("  Final: {evolved}");
    println!("  Purity: {:.4} (1.0 = pure, {:.4} = maximally mixed)",
        evolved.purity(), 1.0 / dim as f64);

    // ─── Step 3: Build QLANG graph for the full pipeline ─────────
    println!("\n[3] Building QLANG computation graph...");

    let mut e = qlang_agent::emitter::GraphEmitter::new("igqk_training");

    // Input weights
    let weights = e.input("weights", Dtype::F32, Shape::vector(dim));
    // Gradient (simulated)
    let gradient = e.input("gradient", Dtype::F32, Shape::vector(dim));

    // Evolve: apply gradient step (simplified quantum gradient flow)
    let evolved_node = {
        let node = e.add(weights, gradient, TT::f32_vector(dim));
        // In full IGQK: this would be the evolve op with density matrix
        node
    };

    // ReLU activation (simulates nonlinear projection)
    let projected = e.relu(evolved_node, TT::f32_vector(dim));

    // IGQK: Compress to ternary
    let compressed = e.to_ternary(projected, TT::f32_vector(dim));

    e.output("compressed_weights", compressed, TT::new(Dtype::Ternary, Shape::vector(dim)));

    let graph = e.build();
    println!("  Graph: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());

    // Verify
    let verification = qlang_core::verify::verify_graph(&graph);
    println!("  Verification: {} passed, {} failed",
        verification.passed.len(), verification.failed.len());

    // ─── Step 4: Execute the graph ─────────
    println!("\n[4] Executing quantum training graph...");

    let mut inputs = HashMap::new();
    inputs.insert(
        "weights".into(),
        TensorData::from_f32(Shape::vector(dim), &initial_weights.iter().map(|&x| x as f32).collect::<Vec<_>>()),
    );
    // Simulated gradient: push weights toward their signs
    let grad: Vec<f32> = initial_weights.iter().map(|&w| {
        if w > 0.0 { 0.1 } else { -0.1 }
    }).collect();
    inputs.insert(
        "gradient".into(),
        TensorData::from_f32(Shape::vector(dim), &grad),
    );

    match qlang_runtime::executor::execute(&graph, inputs) {
        Ok(result) => {
            println!("  Nodes executed: {}", result.stats.nodes_executed);
            println!("  Quantum ops: {}", result.stats.quantum_ops);

            if let Some(out) = result.outputs.get("compressed_weights") {
                println!("\n  Compressed weights (ternary):");
                for (i, &byte) in out.data.iter().enumerate() {
                    let val = match byte {
                        1 => "+1",
                        255 => "-1",
                        0 => " 0",
                        _ => " ?",
                    };
                    println!("    w[{i}]: {:.2} → {:.2} + {:.2} → relu → {val}",
                        initial_weights[i], initial_weights[i], grad[i]);
                }
            }
        }
        Err(e) => eprintln!("  Failed: {e}"),
    }

    // ─── Step 5: Quantum measurement (Born rule) ─────────
    println!("\n[5] Quantum measurement (Born rule)...");

    // Create measurement operators for ternary outcomes
    // M_{+1} projects onto positive subspace
    // M_{-1} projects onto negative subspace
    // M_{0} projects onto near-zero subspace

    let mut m_pos = vec![0.0; dim * dim]; // |positive⟩⟨positive|
    let mut m_neg = vec![0.0; dim * dim]; // |negative⟩⟨negative|
    let mut m_zero = vec![0.0; dim * dim]; // |zero⟩⟨zero|

    for i in 0..dim {
        if initial_weights[i] > 0.3 {
            m_pos[i * dim + i] = 1.0;
        } else if initial_weights[i] < -0.3 {
            m_neg[i * dim + i] = 1.0;
        } else {
            m_zero[i * dim + i] = 1.0;
        }
    }

    let probs = evolved.measure(&[m_pos, m_neg, m_zero]);
    println!("  P(w = +1) = {:.4}", probs[0]);
    println!("  P(w = -1) = {:.4}", probs[1]);
    println!("  P(w =  0) = {:.4}", probs[2]);
    println!("  Sum = {:.4} (should be 1.0)", probs.iter().sum::<f64>());

    // ─── Step 6: KI-zu-KI Protocol Demo ─────────
    println!("\n[6] Agent-to-Agent communication...");

    use qlang_agent::protocol::*;

    let trainer = AgentId {
        name: "IGQK-Trainer".into(),
        capabilities: vec![Capability::Execute, Capability::Train],
    };
    let compressor = AgentId {
        name: "IGQK-Compressor".into(),
        capabilities: vec![Capability::Compress, Capability::Verify],
    };

    let mut conv = AgentConversation::new();

    // Trainer sends model to compressor
    let msg_id = conv.send(
        trainer.clone(),
        compressor.clone(),
        graph,
        HashMap::new(),
        MessageIntent::Compress { method: "ternary".into() },
        None,
    );

    println!("  [{msg_id}] IGQK-Trainer → IGQK-Compressor: 'Compress this model (ternary)'");

    // Compressor responds
    let compressed_graph = qlang_core::graph::Graph::new("compressed_model");
    conv.send(
        compressor,
        trainer,
        compressed_graph,
        HashMap::new(),
        MessageIntent::Result { original_message_id: msg_id },
        Some(msg_id),
    );
    println!("  [1] IGQK-Compressor → IGQK-Trainer: 'Here are the compressed weights'");

    let binary = conv.to_binary().unwrap();
    println!("\n  Conversation serialized: {} bytes", binary.len());
    println!("  Magic: {:?} = 'QLMS' (QLANG Message Stream)", &binary[0..4]);

    println!("\n=== IGQK Quantum Training Complete ===");
}
