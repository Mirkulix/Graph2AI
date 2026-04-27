//! GET /api/hardware — host CPU + memory telemetry via sysinfo crate.
//!
//! GPU info is not exposed on Windows by sysinfo (would need nvml-wrapper or
//! winapi WMI calls); we leave gpu as an empty array so the cockpit shows
//! "No GPU detected" rather than fabricating numbers.

use axum::Json;
use serde::Serialize;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[derive(Serialize)]
pub struct HardwareResponse {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpu: Vec<GpuInfo>,
}

#[derive(Serialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    /// 0.0–1.0
    pub load: f32,
}

#[derive(Serialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
}

#[derive(Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub temp_c: Option<f32>,
    /// 0.0–1.0
    pub util: Option<f32>,
    pub mem_mb: Option<u64>,
}

/// GET /api/hardware
pub async fn hardware() -> Json<HardwareResponse> {
    // Refresh just CPU + memory (cheap). Two refreshes are required to get
    // a meaningful CPU usage delta: the first call seeds the previous-tick
    // counters, the second produces the actual %.
    let mut sys = System::new_with_specifics(
        RefreshKind::new()
            .with_cpu(CpuRefreshKind::new().with_cpu_usage())
            .with_memory(MemoryRefreshKind::everything()),
    );
    sys.refresh_cpu_usage();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let cpus = sys.cpus();
    let cores = cpus.len();
    let model = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let avg_pct: f32 = if cpus.is_empty() {
        0.0
    } else {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
    };

    // sysinfo 0.32 returns bytes from total_memory()/used_memory().
    let total_mb = sys.total_memory() / 1_048_576;
    let used_mb = sys.used_memory() / 1_048_576;

    Json(HardwareResponse {
        cpu: CpuInfo {
            model,
            cores,
            load: (avg_pct / 100.0).clamp(0.0, 1.0),
        },
        memory: MemoryInfo { total_mb, used_mb },
        gpu: Vec::new(),
    })
}
