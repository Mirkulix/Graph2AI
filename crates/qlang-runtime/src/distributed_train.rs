//! Distributed Training — Coordinate training across multiple GPUs/workers.
//!
//! Provides multi-GPU data-parallel training with gradient averaging,
//! plus device detection for NVIDIA, AMD, Apple, and CPU backends.
//!
//! Strategies:
//! - **DataParallel**: Each GPU trains on different data, gradients are averaged.
//! - **ModelParallel**: Model is split across GPUs (for models too large for one GPU).
//! - **PipelineParallel**: Different pipeline stages on different GPUs.
//!
//! Usage:
//! ```rust,no_run
//! use qlang_runtime::distributed_train::*;
//!
//! let config = DistributedConfig {
//!     n_workers: 2,
//!     strategy: ParallelStrategy::DataParallel,
//!     batch_size_per_worker: 32,
//!     gradient_accumulation_steps: 1,
//! };
//!
//! let losses = train_data_parallel(
//!     &config,
//!     |worker_id| {
//!         // each worker computes gradients on its data shard
//!         let grads = vec![vec![0.1; 10]];
//!         let loss = 0.5;
//!         (grads, loss)
//!     },
//!     |avg_grads| { /* apply averaged gradients to model */ },
//!     &[10],
//!     5, // epochs
//! );
//! ```

use std::sync::{Arc, Mutex};
use std::thread;

/// Strategy for distributing work across devices.
#[derive(Clone, Debug)]
pub enum ParallelStrategy {
    /// Each GPU trains on different data, gradients are averaged.
    DataParallel,
    /// Model is split across GPUs (for models too large for one GPU).
    ModelParallel,
    /// Different pipeline stages on different GPUs.
    PipelineParallel,
}

/// Configuration for distributed training.
#[derive(Clone, Debug)]
pub struct DistributedConfig {
    /// Number of worker threads/GPUs.
    pub n_workers: usize,
    /// How to distribute the work.
    pub strategy: ParallelStrategy,
    /// Batch size per worker (effective batch = n_workers * batch_size_per_worker).
    pub batch_size_per_worker: usize,
    /// Accumulate gradients over this many micro-batches before averaging.
    pub gradient_accumulation_steps: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            n_workers: 1,
            strategy: ParallelStrategy::DataParallel,
            batch_size_per_worker: 32,
            gradient_accumulation_steps: 1,
        }
    }
}

/// A gradient buffer for accumulating and averaging gradients from multiple workers.
///
/// Each parameter group has its own gradient vector. Workers call [`accumulate`] to
/// add their computed gradients, then [`average`] produces the mean across workers.
pub struct GradientBuffer {
    /// One gradient vector per parameter group.
    gradients: Vec<Vec<f32>>,
    /// How many workers have contributed so far.
    count: usize,
}

impl GradientBuffer {
    /// Create a new buffer with the given parameter group sizes, initialized to zero.
    pub fn new(param_shapes: &[usize]) -> Self {
        GradientBuffer {
            gradients: param_shapes.iter().map(|&s| vec![0.0; s]).collect(),
            count: 0,
        }
    }

    /// Add gradients from a single worker.
    ///
    /// `worker_grads` must have the same structure as `param_shapes` passed to [`new`].
    pub fn accumulate(&mut self, worker_grads: &[Vec<f32>]) {
        for (buf, grad) in self.gradients.iter_mut().zip(worker_grads) {
            for (b, g) in buf.iter_mut().zip(grad) {
                *b += g;
            }
        }
        self.count += 1;
    }

    /// Average the accumulated gradients by dividing by the worker count.
    pub fn average(&mut self) {
        if self.count > 0 {
            let scale = 1.0 / self.count as f32;
            for buf in &mut self.gradients {
                for b in buf.iter_mut() {
                    *b *= scale;
                }
            }
        }
    }

    /// Reset all gradients to zero and clear the worker count.
    pub fn reset(&mut self) {
        for buf in &mut self.gradients {
            for b in buf.iter_mut() {
                *b = 0.0;
            }
        }
        self.count = 0;
    }

    /// Return a reference to the accumulated (or averaged) gradient vectors.
    pub fn gradients(&self) -> &[Vec<f32>] {
        &self.gradients
    }

    /// Return the number of workers that have contributed so far.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Run data-parallel training across multiple threads.
///
/// Each thread simulates a separate GPU/device and trains on a different data slice.
/// Gradients are collected and averaged after each epoch, then applied to the shared model.
///
/// # Arguments
///
/// * `config`       - Distributed training configuration.
/// * `train_fn`     - Training function: takes a worker ID (0..n_workers) and returns
///                    `(gradients, loss)`. Gradients is `Vec<Vec<f32>>` matching `param_shapes`.
/// * `apply_fn`     - Function to apply averaged gradients to the model.
/// * `param_shapes` - Size of each parameter group (e.g., `&[w1.len(), b1.len(), w2.len(), b2.len()]`).
/// * `epochs`       - Number of training epochs.
///
/// # Returns
///
/// A `Vec<f32>` of per-epoch average losses.
pub fn train_data_parallel(
    config: &DistributedConfig,
    train_fn: impl Fn(usize) -> (Vec<Vec<f32>>, f32) + Send + Sync + 'static,
    apply_fn: impl Fn(&[Vec<f32>]) + Send + Sync + 'static,
    param_shapes: &[usize],
    epochs: usize,
) -> Vec<f32> {
    let train_fn = Arc::new(train_fn);
    let apply_fn = Arc::new(apply_fn);
    let mut losses = Vec::new();

    for epoch in 0..epochs {
        let grad_buf = Arc::new(Mutex::new(GradientBuffer::new(param_shapes)));
        let epoch_loss = Arc::new(Mutex::new(0.0f32));

        // Spawn one thread per worker
        let mut handles = Vec::new();
        for worker_id in 0..config.n_workers {
            let train_fn = Arc::clone(&train_fn);
            let grad_buf = Arc::clone(&grad_buf);
            let epoch_loss = Arc::clone(&epoch_loss);

            handles.push(thread::spawn(move || {
                let (grads, loss) = train_fn(worker_id);
                let mut buf = grad_buf.lock().unwrap();
                buf.accumulate(&grads);
                *epoch_loss.lock().unwrap() += loss;
            }));
        }

        // Wait for all workers to finish
        for h in handles {
            h.join().unwrap();
        }

        // Average gradients and apply to the model
        let mut buf = grad_buf.lock().unwrap();
        buf.average();
        apply_fn(buf.gradients());

        let avg_loss = *epoch_loss.lock().unwrap() / config.n_workers as f32;
        losses.push(avg_loss);

        eprintln!(
            "[distributed] Epoch {}: avg_loss={:.4} ({} workers)",
            epoch + 1,
            avg_loss,
            config.n_workers
        );
    }

    losses
}

// ---------------------------------------------------------------------------
// Device detection
// ---------------------------------------------------------------------------

/// Detect available compute devices on this machine.
///
/// Always includes at least one CPU device. When built with the `gpu` feature,
/// also enumerates wgpu-visible GPU adapters (NVIDIA, AMD, Intel, etc.).
/// When built with the `mlx` feature on macOS, includes Apple MLX.
pub fn detect_devices() -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    // CPU is always available
    let cpu_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    devices.push(DeviceInfo {
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: DeviceType::Cpu,
        memory_mb: 0, // unknown for CPU
        compute_capability: format!("{} threads", cpu_cores),
    });

    // Check for wgpu GPU devices
    #[cfg(feature = "gpu")]
    {
        if let Some(gpu_info) = detect_wgpu_devices() {
            devices.extend(gpu_info);
        }
    }

    // Check for MLX (Apple Silicon)
    #[cfg(all(target_os = "macos", feature = "mlx"))]
    {
        devices.push(DeviceInfo {
            name: "Apple MLX (Metal GPU)".to_string(),
            device_type: DeviceType::AppleMlx,
            memory_mb: 0,
            compute_capability: "Unified Memory".to_string(),
        });
    }

    devices
}

/// Information about a single compute device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Device category.
    pub device_type: DeviceType,
    /// Estimated device memory in megabytes (0 if unknown).
    pub memory_mb: u64,
    /// Backend or capability description.
    pub compute_capability: String,
}

/// Category of compute device.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    /// Standard CPU.
    Cpu,
    /// NVIDIA GPU (CUDA/Vulkan).
    NvidiaGpu,
    /// AMD GPU (Vulkan/ROCm).
    AmdGpu,
    /// Intel integrated or discrete GPU.
    IntelGpu,
    /// Apple Metal via MLX.
    AppleMlx,
    /// Generic wgpu adapter (backend unknown or other).
    Wgpu,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::NvidiaGpu => write!(f, "NVIDIA GPU"),
            DeviceType::AmdGpu => write!(f, "AMD GPU"),
            DeviceType::IntelGpu => write!(f, "Intel GPU"),
            DeviceType::AppleMlx => write!(f, "Apple MLX"),
            DeviceType::Wgpu => write!(f, "wgpu"),
        }
    }
}

/// Enumerate wgpu GPU adapters (only available with the `gpu` feature).
#[cfg(feature = "gpu")]
fn detect_wgpu_devices() -> Option<Vec<DeviceInfo>> {
    let instance = wgpu::Instance::default();
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());

    if adapters.is_empty() {
        return None;
    }

    let mut devices = Vec::new();
    for adapter in &adapters {
        let info = adapter.get_info();
        let name_lower = info.name.to_lowercase();
        let dt = match info.backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 => {
                if name_lower.contains("nvidia") || name_lower.contains("geforce") {
                    DeviceType::NvidiaGpu
                } else if name_lower.contains("amd") || name_lower.contains("radeon") {
                    DeviceType::AmdGpu
                } else if name_lower.contains("intel") {
                    DeviceType::IntelGpu
                } else {
                    DeviceType::Wgpu
                }
            }
            wgpu::Backend::Metal => DeviceType::AppleMlx,
            _ => DeviceType::Wgpu,
        };
        devices.push(DeviceInfo {
            name: info.name.clone(),
            device_type: dt,
            memory_mb: 0,
            compute_capability: format!("{:?}", info.backend),
        });
    }

    Some(devices)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_buffer_accumulate_and_average() {
        let mut buf = GradientBuffer::new(&[3, 2]);

        // Worker 0
        buf.accumulate(&[vec![1.0, 2.0, 3.0], vec![10.0, 20.0]]);
        assert_eq!(buf.count(), 1);

        // Worker 1
        buf.accumulate(&[vec![3.0, 4.0, 5.0], vec![30.0, 40.0]]);
        assert_eq!(buf.count(), 2);

        buf.average();
        let grads = buf.gradients();
        assert_eq!(grads.len(), 2);
        assert!((grads[0][0] - 2.0).abs() < 1e-6);
        assert!((grads[0][1] - 3.0).abs() < 1e-6);
        assert!((grads[0][2] - 4.0).abs() < 1e-6);
        assert!((grads[1][0] - 20.0).abs() < 1e-6);
        assert!((grads[1][1] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn gradient_buffer_reset() {
        let mut buf = GradientBuffer::new(&[2]);
        buf.accumulate(&[vec![5.0, 10.0]]);
        buf.reset();
        assert_eq!(buf.count(), 0);
        assert_eq!(buf.gradients()[0], vec![0.0, 0.0]);
    }

    #[test]
    fn gradient_buffer_average_empty() {
        let mut buf = GradientBuffer::new(&[3]);
        // Averaging with 0 contributors should not divide by zero
        buf.average();
        assert_eq!(buf.gradients()[0], vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn gradient_buffer_single_worker() {
        let mut buf = GradientBuffer::new(&[4]);
        buf.accumulate(&[vec![2.0, 4.0, 6.0, 8.0]]);
        buf.average();
        // With a single worker, average == the original values
        assert_eq!(buf.gradients()[0], vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn data_parallel_mock_training() {
        let applied_grads: Arc<Mutex<Vec<Vec<f32>>>> = Arc::new(Mutex::new(Vec::new()));
        let applied_clone = Arc::clone(&applied_grads);

        let losses = train_data_parallel(
            &DistributedConfig {
                n_workers: 2,
                strategy: ParallelStrategy::DataParallel,
                batch_size_per_worker: 4,
                gradient_accumulation_steps: 1,
            },
            |worker_id| {
                // Worker 0 returns [1, 2], worker 1 returns [3, 4]
                let g = if worker_id == 0 {
                    vec![vec![1.0, 2.0]]
                } else {
                    vec![vec![3.0, 4.0]]
                };
                let loss = worker_id as f32 + 1.0;
                (g, loss)
            },
            move |avg_grads| {
                let mut store = applied_clone.lock().unwrap();
                *store = avg_grads.to_vec();
            },
            &[2],
            1, // 1 epoch
        );

        assert_eq!(losses.len(), 1);
        // avg_loss = (1.0 + 2.0) / 2 = 1.5
        assert!((losses[0] - 1.5).abs() < 1e-6);

        // avg grads = ([1,2] + [3,4]) / 2 = [2, 3]
        let grads = applied_grads.lock().unwrap();
        assert_eq!(grads.len(), 1);
        assert!((grads[0][0] - 2.0).abs() < 1e-6);
        assert!((grads[0][1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn data_parallel_multiple_epochs() {
        let epoch_count = Arc::new(Mutex::new(0usize));
        let ec = Arc::clone(&epoch_count);

        let losses = train_data_parallel(
            &DistributedConfig {
                n_workers: 3,
                strategy: ParallelStrategy::DataParallel,
                batch_size_per_worker: 8,
                gradient_accumulation_steps: 1,
            },
            |_worker_id| {
                (vec![vec![0.5, 0.5]], 0.3)
            },
            move |_avg_grads| {
                let mut count = ec.lock().unwrap();
                *count += 1;
            },
            &[2],
            5,
        );

        assert_eq!(losses.len(), 5);
        // apply_fn should be called once per epoch
        assert_eq!(*epoch_count.lock().unwrap(), 5);
        // Each epoch: avg_loss = (0.3 * 3) / 3 = 0.3
        for &l in &losses {
            assert!((l - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn data_parallel_single_worker() {
        let losses = train_data_parallel(
            &DistributedConfig {
                n_workers: 1,
                strategy: ParallelStrategy::DataParallel,
                batch_size_per_worker: 16,
                gradient_accumulation_steps: 1,
            },
            |_| (vec![vec![1.0]], 0.42),
            |_| {},
            &[1],
            2,
        );
        assert_eq!(losses.len(), 2);
        assert!((losses[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn detect_devices_finds_cpu() {
        let devices = detect_devices();
        assert!(!devices.is_empty(), "Should find at least one device (CPU)");
        assert_eq!(devices[0].device_type, DeviceType::Cpu);
        assert!(devices[0].name.contains("CPU"));
    }

    #[test]
    fn device_info_display() {
        let info = DeviceInfo {
            name: "Test GPU".into(),
            device_type: DeviceType::NvidiaGpu,
            memory_mb: 8192,
            compute_capability: "Vulkan".into(),
        };
        assert_eq!(format!("{}", info), "Test GPU");
        assert_eq!(format!("{}", info.device_type), "NVIDIA GPU");
    }

    #[test]
    fn device_type_display_all_variants() {
        assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
        assert_eq!(format!("{}", DeviceType::NvidiaGpu), "NVIDIA GPU");
        assert_eq!(format!("{}", DeviceType::AmdGpu), "AMD GPU");
        assert_eq!(format!("{}", DeviceType::IntelGpu), "Intel GPU");
        assert_eq!(format!("{}", DeviceType::AppleMlx), "Apple MLX");
        assert_eq!(format!("{}", DeviceType::Wgpu), "wgpu");
    }

    #[test]
    fn distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.n_workers, 1);
        assert_eq!(config.batch_size_per_worker, 32);
        assert_eq!(config.gradient_accumulation_steps, 1);
        assert!(matches!(config.strategy, ParallelStrategy::DataParallel));
    }

    #[test]
    fn gradient_buffer_multiple_param_groups() {
        // Simulate a model with 3 parameter groups (e.g., W1, b1, W2)
        let mut buf = GradientBuffer::new(&[6, 3, 4]);

        buf.accumulate(&[
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0],
        ]);
        buf.accumulate(&[
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ]);

        buf.average();
        let g = buf.gradients();
        assert_eq!(g.len(), 3);
        assert!((g[0][0] - 2.0).abs() < 1e-6);
        assert!((g[1][0] - 3.0).abs() < 1e-6);
        assert!((g[2][0] - 2.0).abs() < 1e-6);
    }
}
