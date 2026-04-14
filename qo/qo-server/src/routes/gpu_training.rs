//! GPU Training API — Mamba LM training with SSE progress streaming.

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use qlang_runtime::gpu_train::{GpuTrainConfig, TrainEvent};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct GpuTrainRequest {
    /// "mamba" (default) or "ff" (forward-forward)
    pub model: Option<String>,
    /// "wikitext2" or "mnist"
    pub dataset: Option<String>,
    /// Number of training steps (default: 5000)
    pub n_steps: Option<usize>,
    /// Model dimension (default: 128, CPU-safe)
    pub d_model: Option<usize>,
    /// Number of Mamba layers (default: 2)
    pub n_layers: Option<usize>,
    /// Learning rate (default: 0.001)
    pub lr: Option<f32>,
    /// Log interval (default: 50)
    pub log_every: Option<usize>,
    /// Use GPU (default: false — CPU fallback)
    pub use_gpu: Option<bool>,
}

#[derive(Serialize)]
pub struct GpuTrainStatus {
    pub running: bool,
    pub step: usize,
    pub total_steps: usize,
    pub loss: f32,
    pub ppl: f32,
    pub elapsed_secs: f64,
    pub model: String,
    pub dataset: String,
    /// Full metrics history for chart replay on page load.
    pub history: Vec<MetricsSnapshot>,
}

#[derive(Serialize)]
pub struct StopResponse {
    pub stopped: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Shared training state (stored in AppState)
// ---------------------------------------------------------------------------

/// A single metrics snapshot for history replay.
#[derive(Serialize, Clone)]
pub struct MetricsSnapshot {
    pub step: usize,
    pub loss: f32,
    pub ppl: f32,
    pub steps_per_sec: f64,
    pub eta_secs: u64,
    pub elapsed_secs: f64,
    pub generated: String,
}

/// Shared state for the currently running GPU training job.
pub struct GpuTrainingState {
    pub running: AtomicBool,
    pub stop_flag: Arc<AtomicBool>,
    pub step: std::sync::atomic::AtomicUsize,
    pub total_steps: std::sync::atomic::AtomicUsize,
    pub loss: std::sync::Mutex<f32>,
    pub ppl: std::sync::Mutex<f32>,
    pub elapsed_secs: std::sync::Mutex<f64>,
    pub model: std::sync::Mutex<String>,
    pub dataset: std::sync::Mutex<String>,
    /// Metrics history for UI replay on page reload.
    pub history: std::sync::Mutex<Vec<MetricsSnapshot>>,
    /// SSE broadcast — multiple UI clients can subscribe.
    pub broadcast: tokio::sync::broadcast::Sender<TrainEvent>,
}

impl Default for GpuTrainingState {
    fn default() -> Self {
        let (broadcast, _) = tokio::sync::broadcast::channel(512);
        Self {
            running: AtomicBool::new(false),
            stop_flag: Arc::new(AtomicBool::new(false)),
            step: std::sync::atomic::AtomicUsize::new(0),
            total_steps: std::sync::atomic::AtomicUsize::new(0),
            loss: std::sync::Mutex::new(0.0),
            ppl: std::sync::Mutex::new(0.0),
            elapsed_secs: std::sync::Mutex::new(0.0),
            model: std::sync::Mutex::new(String::new()),
            dataset: std::sync::Mutex::new(String::new()),
            history: std::sync::Mutex::new(Vec::new()),
            broadcast,
        }
    }
}

// ---------------------------------------------------------------------------
// POST /api/training/gpu — Start training with SSE streaming
// ---------------------------------------------------------------------------

pub async fn start_gpu_training(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GpuTrainRequest>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let model_name = req.model.unwrap_or_else(|| "mamba".into());
    let dataset_name = req.dataset.unwrap_or_else(|| "wikitext2".into());
    let n_steps = req.n_steps.unwrap_or(5000);
    let d_model = req.d_model.unwrap_or(128);
    let n_layers = req.n_layers.unwrap_or(2);
    let lr = req.lr.unwrap_or(0.001);
    let log_every = req.log_every.unwrap_or(50);
    let use_gpu = req.use_gpu.unwrap_or(false);

    // Resolve data path
    let data_path = if dataset_name == "mnist" {
        resolve_path(&[
            "data/mnist/train-images-idx3-ubyte",
            "../data/mnist/train-images-idx3-ubyte",
            "/home/mirkulix/AI/neoqlang/qlang/data/mnist/train-images-idx3-ubyte",
        ])
        .unwrap_or_else(|| "data/mnist".into())
    } else {
        resolve_path(&[
            "data/wikitext2/train.txt",
            "../data/wikitext2/train.txt",
            "/home/mirkulix/AI/neoqlang/qlang/data/wikitext2/train.txt",
        ])
        .unwrap_or_else(|| "data/wikitext2/train.txt".into())
    };

    // Check if already running — send error event immediately
    if state.gpu_training.running.load(Ordering::Relaxed) {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(1);
        let _ = tx
            .send(Ok(Event::default()
                .event("error")
                .data(r#"{"type":"error","message":"Training already running"}"#)))
            .await;
        drop(tx);
        return Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default());
    }

    // Build training config
    let config = GpuTrainConfig {
        d_model,
        d_hidden: d_model * 2,
        d_state: 32,
        n_layers,
        vocab_size: 8000,
        seq_len: 128,
        lr,
        n_steps,
        grad_accum: 4,
        log_every,
        save_every: if n_steps >= 2000 { 1000 } else { 0 },
        data_path,
        output_path: "data/mamba_30m".into(),
    };

    // Reset shared state
    state.gpu_training.stop_flag.store(false, Ordering::Relaxed);
    state.gpu_training.running.store(true, Ordering::Relaxed);
    state.gpu_training.history.lock().unwrap().clear();
    state.gpu_training.step.store(0, Ordering::Relaxed);
    state
        .gpu_training
        .total_steps
        .store(n_steps, Ordering::Relaxed);
    *state.gpu_training.model.lock().unwrap() = model_name;
    *state.gpu_training.dataset.lock().unwrap() = dataset_name;

    // SSE event channel (tokio mpsc — async-safe)
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);

    let stop_clone = state.gpu_training.stop_flag.clone();
    let gpu_state = state.gpu_training.clone();

    // Spawn the blocking training in a background task
    tokio::task::spawn_blocking(move || {
        // std::sync channel for the training loop (non-async)
        let (sync_tx, sync_rx) = std::sync::mpsc::channel::<TrainEvent>();

        // Bridge thread: sync_rx -> sse_tx (converts TrainEvent to SSE Event)
        let sse_tx_clone = sse_tx;
        let gpu_state_clone = gpu_state.clone();
        let bridge = std::thread::spawn(move || {
            while let Ok(event) = sync_rx.recv() {
                // Update shared status + history on progress
                if let TrainEvent::Progress {
                    step,
                    total_steps,
                    loss,
                    ppl,
                    steps_per_sec,
                    eta_secs,
                    elapsed_secs,
                    ref generated,
                    ..
                } = &event
                {
                    gpu_state_clone.step.store(*step, Ordering::Relaxed);
                    gpu_state_clone
                        .total_steps
                        .store(*total_steps, Ordering::Relaxed);
                    *gpu_state_clone.loss.lock().unwrap() = *loss;
                    *gpu_state_clone.ppl.lock().unwrap() = *ppl;
                    *gpu_state_clone.elapsed_secs.lock().unwrap() = *elapsed_secs;
                    gpu_state_clone.history.lock().unwrap().push(MetricsSnapshot {
                        step: *step, loss: *loss, ppl: *ppl,
                        steps_per_sec: *steps_per_sec, eta_secs: *eta_secs,
                        elapsed_secs: *elapsed_secs, generated: generated.clone(),
                    });
                }
                // Broadcast to any UI subscribers
                let _ = gpu_state_clone.broadcast.send(event.clone());

                let event_name = match &event {
                    TrainEvent::Progress { .. } => "progress",
                    TrainEvent::Checkpoint { .. } => "checkpoint",
                    TrainEvent::Complete { .. } => "complete",
                    TrainEvent::Error { .. } => "error",
                };
                let is_terminal = matches!(
                    &event,
                    TrainEvent::Complete { .. } | TrainEvent::Error { .. }
                );
                let data = serde_json::to_string(&event).unwrap_or_default();
                let sse_event = Event::default().event(event_name).data(data);

                if sse_tx_clone.blocking_send(Ok(sse_event)).is_err() {
                    break; // Client disconnected
                }
                if is_terminal {
                    break;
                }
            }
        });

        // Run the actual training — CUDA candle path when GPU requested,
        // falls back to wgpu, then CPU.
        let result = if use_gpu {
            // Try candle CUDA first (tensors stay on GPU), fall back to wgpu hybrid
            let r = qlang_runtime::gpu_train::train_candle_with_progress(
                &config,
                sync_tx.clone(),
                stop_clone.clone(),
            );
            match r {
                Ok(()) => Ok(()),
                Err(ref e) if e.contains("CUDA feature not enabled") || e.contains("CUDA init") => {
                    eprintln!("[gpu_training] Candle CUDA unavailable ({}), falling back to wgpu", e);
                    qlang_runtime::gpu_train::train_gpu_with_progress(
                        &config,
                        sync_tx.clone(),
                        stop_clone,
                    )
                }
                Err(e) => Err(e),
            }
        } else {
            qlang_runtime::gpu_train::train_cpu_with_progress(
                &config,
                sync_tx.clone(),
                stop_clone,
            )
        };
        gpu_state.running.store(false, Ordering::Relaxed);
        if let Err(e) = result {
            let _ = sync_tx.send(TrainEvent::Error { message: e });
        }
        drop(sync_tx);
        let _ = bridge.join();
    });

    Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// GET /api/training/gpu/status
// ---------------------------------------------------------------------------

pub async fn gpu_training_status(
    State(state): State<Arc<AppState>>,
) -> Json<GpuTrainStatus> {
    let ts = &state.gpu_training;
    Json(GpuTrainStatus {
        running: ts.running.load(Ordering::Relaxed),
        step: ts.step.load(Ordering::Relaxed),
        total_steps: ts.total_steps.load(Ordering::Relaxed),
        loss: *ts.loss.lock().unwrap(),
        ppl: *ts.ppl.lock().unwrap(),
        elapsed_secs: *ts.elapsed_secs.lock().unwrap(),
        model: ts.model.lock().unwrap().clone(),
        dataset: ts.dataset.lock().unwrap().clone(),
        history: ts.history.lock().unwrap().clone(),
    })
}

// ---------------------------------------------------------------------------
// POST /api/training/gpu/stop
// ---------------------------------------------------------------------------

pub async fn stop_gpu_training(
    State(state): State<Arc<AppState>>,
) -> Json<StopResponse> {
    if state.gpu_training.running.load(Ordering::Relaxed) {
        state.gpu_training.stop_flag.store(true, Ordering::Relaxed);
        Json(StopResponse {
            stopped: true,
            message: "Stop signal sent. Training will stop after current step."
                .into(),
        })
    } else {
        Json(StopResponse {
            stopped: false,
            message: "No training is currently running.".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// GET /api/training/gpu/stream — Subscribe to live training events (SSE)
// ---------------------------------------------------------------------------

pub async fn gpu_training_stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let mut rx = state.gpu_training.broadcast.subscribe();
    let stream = async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let event_name = match &event {
                        TrainEvent::Progress { .. } => "progress",
                        TrainEvent::Checkpoint { .. } => "checkpoint",
                        TrainEvent::Complete { .. } => "complete",
                        TrainEvent::Error { .. } => "error",
                    };
                    let is_terminal = matches!(&event, TrainEvent::Complete { .. } | TrainEvent::Error { .. });
                    let data = serde_json::to_string(&event).unwrap_or_default();
                    yield Ok(Event::default().event(event_name).data(data));
                    if is_terminal { break; }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_path(candidates: &[&str]) -> Option<String> {
    for p in candidates {
        if std::fs::metadata(p).is_ok() {
            return Some(p.to_string());
        }
    }
    None
}
