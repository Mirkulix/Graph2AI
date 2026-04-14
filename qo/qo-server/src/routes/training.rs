//! QLANG Training API — train ternary models from the browser.

use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use crate::AppState;

#[derive(Deserialize)]
pub struct TrainRequest {
    /// Number of training epochs (default: 10)
    pub epochs: Option<usize>,
    /// Number of training samples to use (default: 10000)
    pub train_samples: Option<usize>,
    /// Layer sizes after input (default: [256, 128])
    pub hidden_layers: Option<Vec<usize>>,
    /// Dataset: "mnist" or "synthetic" (default: mnist)
    pub dataset: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct TrainProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub f32_accuracy: f32,
    pub ternary_accuracy: f32,
    pub pos_goodness: f32,
    pub neg_goodness: f32,
    pub elapsed_secs: f64,
    pub status: String,
}

#[derive(Serialize)]
pub struct TrainResult {
    pub success: bool,
    pub f32_accuracy: f32,
    pub ternary_accuracy: f32,
    pub total_params: usize,
    pub f32_size_kb: f32,
    pub ternary_size_kb: f32,
    pub compression_ratio: f32,
    pub train_time_secs: f64,
    pub model_file: Option<String>,
    pub epochs: Vec<TrainProgress>,
}

/// POST /api/training/qlang — Train a ternary model via Forward-Forward.
///
/// Returns the full result after training completes.
pub async fn train_qlang(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TrainRequest>,
) -> Json<TrainResult> {
    let epochs = req.epochs.unwrap_or(10);
    let train_limit = req.train_samples.unwrap_or(10000);
    let hidden = req.hidden_layers.unwrap_or_else(|| vec![256, 128]);
    let dataset_name = req.dataset.unwrap_or_else(|| "mnist".into());

    // Run training in blocking task (CPU-intensive)
    let result = tokio::task::spawn_blocking(move || {
        run_training(&dataset_name, epochs, train_limit, &hidden)
    }).await;

    match result {
        Ok(Ok(train_result)) => {
            // Publish activity
            state.stream.publish_activity(
                format!(
                    "QLANG Training abgeschlossen: f32={:.1}% ternary={:.1}% ({} Epochen, {:.1}s)",
                    train_result.f32_accuracy * 100.0,
                    train_result.ternary_accuracy * 100.0,
                    train_result.epochs.len(),
                    train_result.train_time_secs,
                ),
                Some("QLANG".to_string()),
                "success",
            );
            Json(train_result)
        }
        Ok(Err(e)) => Json(TrainResult {
            success: false,
            f32_accuracy: 0.0,
            ternary_accuracy: 0.0,
            total_params: 0,
            f32_size_kb: 0.0,
            ternary_size_kb: 0.0,
            compression_ratio: 0.0,
            train_time_secs: 0.0,
            model_file: None,
            epochs: vec![TrainProgress {
                epoch: 0, total_epochs: 0,
                f32_accuracy: 0.0, ternary_accuracy: 0.0,
                pos_goodness: 0.0, neg_goodness: 0.0,
                elapsed_secs: 0.0,
                status: format!("Fehler: {e}"),
            }],
        }),
        Err(e) => Json(TrainResult {
            success: false,
            f32_accuracy: 0.0,
            ternary_accuracy: 0.0,
            total_params: 0,
            f32_size_kb: 0.0,
            ternary_size_kb: 0.0,
            compression_ratio: 0.0,
            train_time_secs: 0.0,
            model_file: None,
            epochs: vec![TrainProgress {
                epoch: 0, total_epochs: 0,
                f32_accuracy: 0.0, ternary_accuracy: 0.0,
                pos_goodness: 0.0, neg_goodness: 0.0,
                elapsed_secs: 0.0,
                status: format!("Task-Fehler: {e}"),
            }],
        }),
    }
}

fn run_training(
    dataset: &str,
    epochs: usize,
    train_limit: usize,
    hidden: &[usize],
) -> Result<TrainResult, String> {
    use qlang_runtime::forward_forward::FFNetwork;
    use qlang_runtime::mnist::MnistData;
    use qlang_runtime::ternary_ops;

    // Load data
    let data = if dataset == "mnist" {
        // Try multiple paths
        let paths = ["data/mnist", "../data/mnist", "/home/mirkulix/neoqlang/qlang/data/mnist"];
        let mut loaded = None;
        for p in &paths {
            if let Ok(d) = MnistData::load_from_dir(p) {
                loaded = Some(d);
                break;
            }
        }
        loaded.unwrap_or_else(|| MnistData::synthetic(train_limit.min(5000), 1000))
    } else {
        MnistData::synthetic(train_limit.min(5000), 1000)
    };

    let actual_train = data.n_train.min(train_limit);
    let actual_test = data.n_test.min(2000);
    let train_images = &data.train_images[..actual_train * 784];
    let train_labels = &data.train_labels[..actual_train];
    let test_images = &data.test_images[..actual_test * 784];
    let test_labels = &data.test_labels[..actual_test];

    // Build network: 794 (784+10) → hidden layers
    let mut layer_sizes = vec![794];
    layer_sizes.extend_from_slice(hidden);
    let mut net = FFNetwork::new(&layer_sizes, 10);

    let total_start = Instant::now();
    let mut epoch_results = Vec::new();

    for epoch in 0..epochs {
        let (pg, ng) = net.train_epoch(train_images, train_labels, 784, actual_train, 100);

        let f32_acc = net.accuracy(test_images, test_labels, 784, actual_test);
        let tern_acc = net.accuracy_ternary(test_images, test_labels, 784, actual_test);

        epoch_results.push(TrainProgress {
            epoch: epoch + 1,
            total_epochs: epochs,
            f32_accuracy: f32_acc,
            ternary_accuracy: tern_acc,
            pos_goodness: pg,
            neg_goodness: ng,
            elapsed_secs: total_start.elapsed().as_secs_f64(),
            status: "training".into(),
        });
    }

    let train_time = total_start.elapsed();
    let f32_acc = net.accuracy(test_images, test_labels, 784, actual_test);
    let tern_acc = net.accuracy_ternary(test_images, test_labels, 784, actual_test);

    let total_params: usize = net.layers.iter().map(|l| l.weights.len() + l.biases.len()).sum();
    let f32_size = total_params * 4;
    let tern_size: usize = net.layers.iter()
        .map(|l| ternary_ops::pack_ternary(&l.weights).0.len() + l.biases.len() * 4)
        .sum();

    // Save model
    let model_path = format!("data/qlang_model_{}.qlbg",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );

    // Try to save (best effort)
    let _ = std::fs::create_dir_all("data");
    let saved_path = if save_model_simple(&net, &model_path, epochs, f32_acc, tern_acc, actual_train, actual_test, train_time.as_secs_f64(), total_params, f32_size, tern_size).is_ok() {
        Some(model_path)
    } else {
        None
    };

    Ok(TrainResult {
        success: true,
        f32_accuracy: f32_acc,
        ternary_accuracy: tern_acc,
        total_params,
        f32_size_kb: f32_size as f32 / 1024.0,
        ternary_size_kb: tern_size as f32 / 1024.0,
        compression_ratio: f32_size as f32 / tern_size as f32,
        train_time_secs: train_time.as_secs_f64(),
        model_file: saved_path,
        epochs: epoch_results,
    })
}

fn save_model_simple(
    net: &qlang_runtime::forward_forward::FFNetwork,
    path: &str,
    epochs: usize,
    f32_acc: f32,
    tern_acc: f32,
    train_samples: usize,
    test_samples: usize,
    train_time: f64,
    total_params: usize,
    f32_size: usize,
    tern_size: usize,
) -> Result<(), String> {
    use qlang_runtime::ternary_ops;

    // Simple format: JSON metadata + packed weights
    let mut file_data = Vec::new();

    // Magic: "QLTN" (QLANG Ternary Network)
    file_data.extend_from_slice(&[0x51, 0x4C, 0x54, 0x4E]);

    // Metadata as JSON
    let meta = serde_json::json!({
        "method": "forward-forward-ternary",
        "epochs": epochs,
        "f32_accuracy": f32_acc,
        "ternary_accuracy": tern_acc,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "train_time_secs": train_time,
        "total_params": total_params,
        "f32_size_bytes": f32_size,
        "ternary_size_bytes": tern_size,
        "layers": net.layers.iter().map(|l| {
            serde_json::json!({
                "in_dim": l.in_dim,
                "out_dim": l.out_dim,
            })
        }).collect::<Vec<_>>(),
    });
    let meta_bytes = serde_json::to_vec(&meta).map_err(|e| e.to_string())?;
    file_data.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    file_data.extend_from_slice(&meta_bytes);

    // Packed weights per layer
    for layer in &net.layers {
        let (packed, _alpha) = ternary_ops::pack_ternary(&layer.weights);
        file_data.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&packed);
        // Biases as f32
        let bias_bytes: Vec<u8> = layer.biases.iter().flat_map(|b| b.to_le_bytes()).collect();
        file_data.extend_from_slice(&(bias_bytes.len() as u32).to_le_bytes());
        file_data.extend_from_slice(&bias_bytes);
    }

    std::fs::write(path, &file_data).map_err(|e| e.to_string())
}
