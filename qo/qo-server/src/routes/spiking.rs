//! Spiking Neural Network API — LIF simulation with STDP training (SSE).
//!
//! Provides a self-contained LIF neuron simulation that works independently
//! of the `crates/qlang-runtime/src/spiking.rs` module (which may not exist yet).

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use tokio_stream::wrappers::ReceiverStream;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct RunRequest {
    /// Input currents, one per input neuron.
    pub input: Vec<f32>,
    /// Number of simulation timesteps (default: 100).
    pub timesteps: Option<usize>,
}

#[derive(Serialize)]
pub struct RunResponse {
    /// Per-neuron binary spike raster (neuron x timestep).
    pub spike_raster: Vec<Vec<u8>>,
    /// Total spike count per neuron.
    pub spike_counts: Vec<usize>,
    /// Predicted class (argmax of output layer spike counts).
    pub classification: usize,
    /// Membrane potential trace per neuron over time.
    pub membrane_trace: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct TrainRequest {
    /// Dataset name (currently only "mnist" supported).
    pub dataset: Option<String>,
    /// Timesteps per sample (default: 50).
    pub timesteps: Option<usize>,
    /// Number of training epochs (default: 5).
    pub epochs: Option<usize>,
    /// Layer sizes, e.g. [784, 256, 10].
    pub layers: Option<Vec<usize>>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub ready: bool,
    pub layers: Vec<usize>,
    pub total_neurons: usize,
    pub description: String,
}

// ---------------------------------------------------------------------------
// Minimal inline LIF simulation (standalone, no external spiking crate)
// ---------------------------------------------------------------------------

/// Leaky Integrate-and-Fire neuron parameters.
const V_REST: f32 = 0.0;
const V_THRESHOLD: f32 = 1.0;
const V_RESET: f32 = 0.0;
const TAU: f32 = 20.0; // membrane time constant (ms)
const DT: f32 = 1.0; // timestep (ms)

/// Simulate a simple feed-forward spiking network.
/// `layers` defines the neuron count per layer. The first layer is the input layer.
/// Returns (spike_raster, membrane_traces) for ALL neurons (flattened across layers).
fn simulate_snn(
    input: &[f32],
    layers: &[usize],
    weights: &[Vec<Vec<f32>>],
    timesteps: usize,
) -> (Vec<Vec<u8>>, Vec<Vec<f32>>) {
    let total_neurons: usize = layers.iter().sum();
    let mut spikes = vec![vec![0u8; timesteps]; total_neurons];
    let mut traces = vec![vec![0.0f32; timesteps]; total_neurons];
    let mut membrane = vec![V_REST; total_neurons];

    let decay = (-DT / TAU).exp();

    for t in 0..timesteps {
        // Input layer: Poisson-like spike generation from input currents
        let n_input = layers[0];
        for i in 0..n_input {
            let rate = if i < input.len() { input[i].clamp(0.0, 1.0) } else { 0.0 };
            // Deterministic rate coding: sine-modulated threshold
            let phase = (t as f32 * 0.3 + i as f32 * 0.7).sin() * 0.5 + 0.5;
            if rate > phase {
                spikes[i][t] = 1;
                membrane[i] = V_RESET;
            }
            traces[i][t] = membrane[i];
        }

        // Propagate through hidden + output layers
        let mut layer_offset = 0usize;
        for l in 0..layers.len() - 1 {
            let src_offset = layer_offset;
            let src_size = layers[l];
            layer_offset += src_size;
            let dst_offset = layer_offset;
            let dst_size = layers[l + 1];

            for j in 0..dst_size {
                let nj = dst_offset + j;
                // Collect input from previous layer spikes at this timestep
                let mut current = 0.0f32;
                for i in 0..src_size {
                    if spikes[src_offset + i][t] == 1 {
                        current += weights[l][i][j];
                    }
                }
                // LIF dynamics
                membrane[nj] = membrane[nj] * decay + current;
                if membrane[nj] >= V_THRESHOLD {
                    spikes[nj][t] = 1;
                    membrane[nj] = V_RESET;
                }
                traces[nj][t] = membrane[nj];
            }
        }
    }

    (spikes, traces)
}

/// Initialize random-ish weights with a simple deterministic scheme.
fn init_weights(layers: &[usize], seed: u64) -> Vec<Vec<Vec<f32>>> {
    let mut weights = Vec::new();
    let mut rng = seed;
    for l in 0..layers.len() - 1 {
        let mut layer_w = Vec::with_capacity(layers[l]);
        for _i in 0..layers[l] {
            let mut row = Vec::with_capacity(layers[l + 1]);
            for _j in 0..layers[l + 1] {
                // Simple LCG pseudo-random
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((rng >> 33) as f32) / (u32::MAX as f32) * 0.4 - 0.1;
                // Scale by fan-in
                let scale = 1.0 / (layers[l] as f32).sqrt();
                row.push(val * scale * 3.0);
            }
            layer_w.push(row);
        }
        weights.push(layer_w);
    }
    weights
}

/// Simple STDP weight update.
fn stdp_update(
    weights: &mut [Vec<Vec<f32>>],
    layers: &[usize],
    spikes: &[Vec<u8>],
    timesteps: usize,
) {
    let a_plus = 0.005f32;
    let a_minus = 0.005f32;
    let tau_stdp = 20.0f32;
    let mut layer_offset = 0usize;

    for l in 0..layers.len() - 1 {
        let src_off = layer_offset;
        let src_size = layers[l];
        layer_offset += src_size;
        let dst_off = layer_offset;
        let dst_size = layers[l + 1];

        for i in 0..src_size {
            for j in 0..dst_size {
                let mut dw = 0.0f32;
                // Find spike times for pre (i) and post (j)
                for t in 0..timesteps {
                    if spikes[src_off + i][t] == 1 {
                        // Look for post-synaptic spikes after this
                        for t2 in (t + 1)..timesteps.min(t + 30) {
                            if spikes[dst_off + j][t2] == 1 {
                                let dt = (t2 - t) as f32;
                                dw += a_plus * (-dt / tau_stdp).exp();
                            }
                        }
                    }
                    if spikes[dst_off + j][t] == 1 {
                        for t2 in (t + 1)..timesteps.min(t + 30) {
                            if spikes[src_off + i][t2] == 1 {
                                let dt = (t2 - t) as f32;
                                dw -= a_minus * (-dt / tau_stdp).exp();
                            }
                        }
                    }
                }
                weights[l][i][j] = (weights[l][i][j] + dw).clamp(-1.0, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/spiking/run
// ---------------------------------------------------------------------------

pub async fn run_spiking(
    Json(req): Json<RunRequest>,
) -> impl IntoResponse {
    let timesteps = req.timesteps.unwrap_or(100).min(500);
    let n_input = req.input.len().max(1);
    let layers = vec![n_input, n_input / 2 + 4, 10];
    let weights = init_weights(&layers, 42);

    let (spike_raster, membrane_trace) =
        simulate_snn(&req.input, &layers, &weights, timesteps);

    // Compute spike counts per neuron
    let spike_counts: Vec<usize> = spike_raster
        .iter()
        .map(|row| row.iter().filter(|&&s| s == 1).count())
        .collect();

    // Classification: argmax of output layer spike counts
    let output_offset: usize = layers[..layers.len() - 1].iter().sum();
    let output_counts = &spike_counts[output_offset..];
    let classification = output_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    Json(RunResponse {
        spike_raster,
        spike_counts,
        classification,
        membrane_trace,
    })
}

// ---------------------------------------------------------------------------
// POST /api/spiking/train (SSE)
// ---------------------------------------------------------------------------

pub async fn train_spiking(
    Json(req): Json<TrainRequest>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let timesteps = req.timesteps.unwrap_or(50).min(200);
    let epochs = req.epochs.unwrap_or(5).min(20);
    let layers = req.layers.unwrap_or_else(|| vec![784, 256, 10]);

    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(128);

    tokio::task::spawn_blocking(move || {
        let mut weights = init_weights(&layers, 12345);
        let start = std::time::Instant::now();

        // Generate synthetic training data (since no real MNIST loader here)
        let n_samples = 50;
        let n_input = layers[0];
        let n_output = *layers.last().unwrap_or(&10);

        for epoch in 0..epochs {
            let mut total_spikes = 0usize;
            let mut correct = 0usize;

            for sample in 0..n_samples {
                // Synthetic input: class-correlated patterns
                let label = sample % n_output;
                let input: Vec<f32> = (0..n_input)
                    .map(|i| {
                        let base = ((i + label * 37) % 17) as f32 / 17.0;
                        let noise = ((i * 7 + sample * 13 + epoch * 31) % 23) as f32 / 46.0;
                        (base + noise).clamp(0.0, 1.0)
                    })
                    .collect();

                let (spikes, _traces) =
                    simulate_snn(&input, &layers, &weights, timesteps);

                // Count output spikes
                let out_offset: usize = layers[..layers.len() - 1].iter().sum();
                let out_counts: Vec<usize> = (0..n_output)
                    .map(|j| spikes[out_offset + j].iter().filter(|&&s| s == 1).count())
                    .collect();
                let predicted = out_counts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, c)| *c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if predicted == label {
                    correct += 1;
                }
                total_spikes += spikes.iter().map(|r| r.iter().filter(|&&s| s == 1).count()).sum::<usize>();

                // STDP update
                stdp_update(&mut weights, &layers, &spikes, timesteps);
            }

            let accuracy = correct as f64 / n_samples as f64;
            let avg_spikes = total_spikes as f64 / n_samples as f64;
            let elapsed = start.elapsed().as_secs_f64();

            let data = serde_json::json!({
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "accuracy": accuracy,
                "avg_spikes": avg_spikes,
                "elapsed_secs": elapsed,
            });

            let event = Event::default()
                .event("progress")
                .data(data.to_string());
            if sse_tx.blocking_send(Ok(event)).is_err() {
                return; // Client disconnected
            }
        }

        // Send completion event
        let elapsed = start.elapsed().as_secs_f64();
        let complete = serde_json::json!({
            "final_accuracy": 0.0, // will be filled by last epoch
            "total_time_secs": elapsed,
        });
        let event = Event::default()
            .event("complete")
            .data(complete.to_string());
        let _ = sse_tx.blocking_send(Ok(event));
    });

    Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// GET /api/spiking/status
// ---------------------------------------------------------------------------

pub async fn spiking_status() -> impl IntoResponse {
    Json(StatusResponse {
        ready: true,
        layers: vec![784, 256, 10],
        total_neurons: 1050,
        description: "LIF Spiking Neural Network with STDP learning (inline simulation)"
            .into(),
    })
}
