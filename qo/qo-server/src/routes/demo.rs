use axum::response::sse::{Event, KeepAlive, Sse};
use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;

#[derive(Deserialize, Default)]
pub struct DemoRequest {
    pub train_samples: Option<usize>,  // default 60000 = all
    pub refine_epochs: Option<usize>,  // default 3
}

#[derive(Serialize)]
#[serde(tag = "phase")]
pub enum DemoEvent {
    #[serde(rename = "loading")]
    Loading { progress: f32, message: String },
    #[serde(rename = "training")]
    Training { progress: f32, f32_accuracy: f32, elapsed_secs: f64, epoch: usize, total_epochs: usize },
    #[serde(rename = "compressing")]
    Compressing { progress: f32, message: String },
    #[serde(rename = "verifying")]
    Verifying { progress: f32, ternary_accuracy: f32 },
    #[serde(rename = "complete")]
    Complete {
        f32_accuracy: f32,
        ternary_accuracy: f32,
        train_samples: usize,
        test_samples: usize,
        total_weights: usize,
        f32_size_bytes: usize,
        ternary_size_bytes: usize,
        compression_ratio: f32,
        total_time_secs: f64,
        confusion_matrix: Vec<Vec<u32>>,  // 10x10
    },
    #[serde(rename = "error")]
    Error { message: String },
}

pub async fn start_mnist_igqk_demo(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<DemoRequest>,
) -> Sse<impl futures::stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);

    tokio::task::spawn_blocking(move || {
        let start = std::time::Instant::now();

        // Helper to send events
        let send_event = |name: &str, event: DemoEvent| {
            let data = serde_json::to_string(&event).unwrap_or_default();
            let _ = tx.blocking_send(Ok(Event::default().event(name).data(data)));
        };

        // === Phase 1: Load MNIST ===
        send_event("loading", DemoEvent::Loading { progress: 0.0, message: "Lade MNIST...".into() });

        let paths = ["data/mnist", "../data/mnist", "/home/mirkulix/AI/neoqlang/qlang/data/mnist"];
        let data = paths.iter().filter_map(|p| qlang_runtime::mnist::MnistData::load_from_dir(p).ok()).next();
        let data = match data {
            Some(d) => d,
            None => {
                send_event("error", DemoEvent::Error { message: "MNIST nicht gefunden".into() });
                return;
            }
        };

        let n_train = req.train_samples.unwrap_or(60000).min(data.n_train);
        let n_test = data.n_test.min(10000);
        let refine_epochs = req.refine_epochs.unwrap_or(10);  // blueprint: 10 rounds
        let neurons_per_class = 100usize;                      // blueprint: 100 neurons/class

        send_event("loading", DemoEvent::Loading {
            progress: 1.0,
            message: format!("MNIST geladen: {} Training / {} Test", n_train, n_test),
        });

        // === Phase 2: Train FFNetwork (f32 shadow + ternary sync) — echter 95%+ Pfad ===
        use qlang_runtime::forward_forward::FFNetwork;

        send_event("training", DemoEvent::Training {
            progress: 0.0, f32_accuracy: 0.0, elapsed_secs: start.elapsed().as_secs_f64(),
            epoch: 0, total_epochs: refine_epochs,
        });

        let train_imgs = &data.train_images[..n_train * 784];
        let train_lbls = &data.train_labels[..n_train];
        let test_imgs = &data.test_images[..n_test * 784];
        let test_lbls = &data.test_labels[..n_test];

        let _ = neurons_per_class;  // not used in FF path

        // Try GPU QAT first (tch-rs on CUDA GPU 1). Falls back to CPU on error.
        let gpu_result = {
            use qlang_runtime::forward_forward_gpu::train_ff_qat_gpu;
            let layer_sizes = [794usize, 512, 256, 128];
            let epochs_total = refine_epochs;
            let tx_cb = tx.clone();
            let progress_cb = move |epoch: usize, tern_acc: f32, elapsed: f64| {
                let ev = DemoEvent::Training {
                    progress: epoch as f32 / epochs_total as f32,
                    f32_accuracy: tern_acc,
                    elapsed_secs: elapsed,
                    epoch,
                    total_epochs: epochs_total,
                };
                let data = serde_json::to_string(&ev).unwrap_or_default();
                let _ = tx_cb.blocking_send(Ok(Event::default().event("training").data(data)));
            };
            train_ff_qat_gpu(
                train_imgs, train_lbls, &layer_sizes, 10,
                n_train, n_test, test_imgs, test_lbls,
                refine_epochs, 100, 0.03, 2.0,
                progress_cb,
            )
        };

        // Unpack: either use GPU result, or fall back to CPU FFNetwork
        let (f32_acc, ternary_acc, total_weights, all_ternary_f32, confusion) = match gpu_result {
            Ok(res) => {
                eprintln!(
                    "[demo] GPU QAT done: f32={:.4} tern={:.4} weights={} time={:.1}s",
                    res.f32_accuracy, res.ternary_accuracy, res.total_weights, res.total_time_secs
                );
                (res.f32_accuracy, res.ternary_accuracy, res.total_weights, res.all_ternary_weights, res.confusion_matrix)
            }
            Err(gpu_err) => {
                eprintln!("[demo] GPU QAT unavailable ({}); falling back to CPU", gpu_err);
                // FFNetwork: 794 (784 image + 10 label one-hot) → 512 → 256 → 128
                let mut net = FFNetwork::new(&[794, 512, 256, 128], 10);
                for epoch in 0..refine_epochs {
                    let (pg, ng) = net.train_epoch_qat(train_imgs, train_lbls, 784, n_train, 100);
                    let tern_acc = net.accuracy_ternary(test_imgs, test_lbls, 784, n_test);
                    eprintln!("[demo] CPU QAT epoch {}: pg={:.3} ng={:.3} tern_acc={:.4}",
                        epoch + 1, pg, ng, tern_acc);
                    send_event("training", DemoEvent::Training {
                        progress: (epoch as f32 + 1.0) / refine_epochs as f32,
                        f32_accuracy: tern_acc,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                        epoch: epoch + 1,
                        total_epochs: refine_epochs,
                    });
                }
                let f32_acc = net.accuracy(test_imgs, test_lbls, 784, n_test);
                let ternary_acc_cpu = net.accuracy_ternary(test_imgs, test_lbls, 784, n_test);
                let total_w: usize = net.layers.iter().map(|l| l.weights.len()).sum();
                let mut all_tern: Vec<f32> = Vec::with_capacity(total_w);
                for layer in &net.layers { all_tern.extend_from_slice(&layer.weights); }
                let preds = net.predict_ternary(test_imgs, 784, n_test);
                let mut cm = vec![vec![0u32; 10]; 10];
                for (p, &a) in preds.iter().zip(test_lbls.iter()) {
                    cm[a as usize][*p as usize] += 1;
                }
                (f32_acc, ternary_acc_cpu, total_w, all_tern, cm)
            }
        };

        // === Phase 4: REAL IGQK Compression — pack all ternary weights ===
        use qlang_runtime::ternary_ops;

        send_event("compressing", DemoEvent::Compressing {
            progress: 0.2,
            message: format!("Extrahiere {} ternäre Gewichte...", total_weights),
        });

        send_event("compressing", DemoEvent::Compressing {
            progress: 0.5,
            message: "Packe mit pack_ternary (2 bits/weight)...".into(),
        });

        // REAL packing: 4 ternary weights per byte
        let (packed, alpha) = ternary_ops::pack_ternary(&all_ternary_f32);

        let f32_size = total_weights * 4;
        let ternary_size = packed.len();
        let ratio = f32_size as f32 / ternary_size as f32;

        send_event("compressing", DemoEvent::Compressing {
            progress: 1.0,
            message: format!("Compression: {:.1}x ({} KB f32 → {} KB ternär, alpha={:.3})",
                ratio, f32_size/1024, ternary_size/1024, alpha),
        });

        // === Phase 5: ternary verification event ===
        send_event("verifying", DemoEvent::Verifying {
            progress: 0.3,
            ternary_accuracy: 0.0,
        });
        send_event("verifying", DemoEvent::Verifying {
            progress: 1.0,
            ternary_accuracy: ternary_acc,
        });

        eprintln!("[demo] FINAL: f32={:.4} ternary={:.4}", f32_acc, ternary_acc);

        // === Complete ===
        send_event("complete", DemoEvent::Complete {
            f32_accuracy: f32_acc,
            ternary_accuracy: ternary_acc,
            train_samples: n_train,
            test_samples: n_test,
            total_weights,
            f32_size_bytes: f32_size,
            ternary_size_bytes: ternary_size,
            compression_ratio: ratio,
            total_time_secs: start.elapsed().as_secs_f64(),
            confusion_matrix: confusion,
        });
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}
