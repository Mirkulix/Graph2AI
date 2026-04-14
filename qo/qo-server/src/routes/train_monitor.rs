//! Live training monitor — reads the training log file and streams progress.

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use std::sync::Arc;

use crate::AppState;

#[derive(Serialize)]
pub struct TrainMonitorData {
    pub running: bool,
    pub epochs: Vec<EpochData>,
    pub current_epoch: usize,
    pub total_epochs: usize,
    pub best_f32: f32,
    pub best_ternary: f32,
    pub raw_log: String,
}

#[derive(Serialize, Clone)]
pub struct EpochData {
    pub epoch: usize,
    pub f32_accuracy: f32,
    pub ternary_accuracy: f32,
    pub pos_goodness: f32,
    pub neg_goodness: f32,
}

/// GET /api/training/monitor — live training progress from log file.
pub async fn monitor(State(_state): State<Arc<AppState>>) -> Json<TrainMonitorData> {
    let log_paths = [
        "data/training-90pct.log",
        "../data/training-90pct.log",
        "/home/mirkulix/neoqlang/qlang/data/training-90pct.log",
    ];

    let mut log_content = String::new();
    for path in &log_paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            log_content = content;
            break;
        }
    }

    if log_content.is_empty() {
        return Json(TrainMonitorData {
            running: false,
            epochs: vec![],
            current_epoch: 0,
            total_epochs: 0,
            best_f32: 0.0,
            best_ternary: 0.0,
            raw_log: "Kein Training-Log gefunden.".into(),
        });
    }

    // Parse epochs from log
    let mut epochs = Vec::new();
    let mut total_epochs = 0;

    for line in log_content.lines() {
        // Parse: "  Epoch  4/50: f32=89.5%  ternary=82.2%  (pg=3.26 ng=1.33)"
        if let Some(epoch_str) = line.trim().strip_prefix("Epoch") {
            if let Some((epoch_part, rest)) = epoch_str.split_once(':') {
                let parts: Vec<&str> = epoch_part.trim().split('/').collect();
                if parts.len() == 2 {
                    let epoch: usize = parts[0].trim().parse().unwrap_or(0);
                    total_epochs = parts[1].trim().parse().unwrap_or(0);

                    let mut f32_acc = 0.0f32;
                    let mut tern_acc = 0.0f32;
                    let mut pg = 0.0f32;
                    let mut ng = 0.0f32;

                    for segment in rest.split_whitespace() {
                        if let Some(val) = segment.strip_prefix("f32=") {
                            f32_acc = val.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
                        }
                        if let Some(val) = segment.strip_prefix("ternary=") {
                            tern_acc = val.trim_end_matches('%').parse().unwrap_or(0.0) / 100.0;
                        }
                        if let Some(val) = segment.strip_prefix("(pg=") {
                            pg = val.parse().unwrap_or(0.0);
                        }
                        if let Some(val) = segment.strip_prefix("ng=") {
                            ng = val.trim_end_matches(')').parse().unwrap_or(0.0);
                        }
                    }

                    epochs.push(EpochData { epoch, f32_accuracy: f32_acc, ternary_accuracy: tern_acc, pos_goodness: pg, neg_goodness: ng });
                }
            }
        }
    }

    let running = !log_content.contains("Training complete:") && !epochs.is_empty();
    let current_epoch = epochs.last().map(|e| e.epoch).unwrap_or(0);
    let best_f32 = epochs.iter().map(|e| e.f32_accuracy).fold(0.0f32, f32::max);
    let best_ternary = epochs.iter().map(|e| e.ternary_accuracy).fold(0.0f32, f32::max);

    Json(TrainMonitorData {
        running,
        epochs,
        current_epoch,
        total_epochs,
        best_f32,
        best_ternary,
        raw_log: log_content,
    })
}
