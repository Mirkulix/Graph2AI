//! GPU Runtime — Execute QLANG graphs on GPU via generated WGSL shaders.
//!
//! This module provides GPU execution capabilities using WGSL compute shaders.
//!
//! Two modes:
//! - **`gpu` feature enabled**: Real wgpu-based GPU execution via `wgpu` + `pollster`.
//! - **`gpu` feature disabled** (default): CPU-simulated GPU execution for testing.
//!
//! The public API (`GpuDevice`, `GpuKernel`, `GpuResult`, `GpuError`, `gpu_info`)
//! is identical in both modes.

use std::collections::HashMap;

/// GPU device abstraction.
#[derive(Debug)]
pub struct GpuDevice {
    /// Device name/identifier.
    pub name: String,
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Maximum workgroup size.
    pub max_workgroup_size: u32,
    /// Whether this is a real GPU or CPU fallback.
    pub is_real_gpu: bool,
    /// Inner wgpu handles (only present when `gpu` feature is enabled and a GPU was found).
    #[cfg(feature = "gpu")]
    inner: Option<WgpuInner>,
}

/// Real wgpu device/queue handles.
#[cfg(feature = "gpu")]
#[derive(Debug)]
struct WgpuInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// A compiled GPU kernel ready for execution.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// The WGSL shader source code.
    pub shader_source: String,
    /// Number of elements to process.
    pub n_elements: usize,
    /// Workgroup size.
    pub workgroup_size: u32,
}

/// Result of GPU execution.
#[derive(Debug)]
pub struct GpuResult {
    /// Output data.
    pub data: Vec<f32>,
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Whether executed on real GPU or CPU fallback.
    pub used_gpu: bool,
}

/// Errors during GPU execution.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("GPU not available: {0}")]
    NotAvailable(String),
    #[error("shader compilation failed: {0}")]
    ShaderError(String),
    #[error("buffer too large: {size} bytes (max {max})")]
    BufferTooLarge { size: usize, max: usize },
    #[error("execution error: {0}")]
    ExecutionError(String),
}

// ---------------------------------------------------------------------------
// Real wgpu implementation (feature = "gpu")
// ---------------------------------------------------------------------------
#[cfg(feature = "gpu")]
impl GpuDevice {
    /// Create a CPU fallback device for environments without GPU.
    pub fn cpu_fallback() -> Self {
        Self {
            name: "CPU Fallback (WGSL Simulator)".into(),
            max_buffer_size: usize::MAX,
            max_workgroup_size: 256,
            is_real_gpu: false,
            inner: None,
        }
    }

    /// Try to initialize a real GPU device.
    /// Falls back to CPU simulation if no adapter/device can be obtained.
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(a) => a,
            None => return Self::cpu_fallback(),
        };

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = match adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qlang-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
        {
            Ok(dq) => dq,
            Err(_) => return Self::cpu_fallback(),
        };

        Self {
            name: format!("{} ({})", adapter_info.name, adapter_info.backend.to_str()),
            max_buffer_size: limits.max_buffer_size as usize,
            max_workgroup_size: limits.max_compute_workgroup_size_x,
            is_real_gpu: true,
            inner: Some(WgpuInner { device, queue }),
        }
    }

    /// Check whether a real GPU adapter is available on this machine.
    pub fn is_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_some()
        })
    }

    /// Compile a WGSL shader into a kernel.
    pub fn compile_kernel(
        &self,
        shader_source: &str,
        n_elements: usize,
    ) -> Result<GpuKernel, GpuError> {
        if !shader_source.contains("@compute") {
            return Err(GpuError::ShaderError(
                "shader must contain @compute entry point".into(),
            ));
        }
        Ok(GpuKernel {
            shader_source: shader_source.to_string(),
            n_elements,
            workgroup_size: self.max_workgroup_size,
        })
    }

    /// Execute a kernel with element-wise operation on two input buffers.
    pub fn execute_elementwise(
        &self,
        kernel: &GpuKernel,
        input_a: &[f32],
        input_b: &[f32],
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        // Fall back to CPU simulation when no real GPU is present.
        let inner = match &self.inner {
            Some(i) => i,
            None => {
                let result = simulate_elementwise_shader(
                    &kernel.shader_source,
                    input_a,
                    input_b,
                    kernel.n_elements,
                )?;
                return Ok(GpuResult {
                    data: result,
                    execution_time_us: start.elapsed().as_micros() as u64,
                    used_gpu: false,
                });
            }
        };

        let n = kernel.n_elements.min(input_a.len()).min(input_b.len());
        let buf_bytes = (n * std::mem::size_of::<f32>()) as u64;

        // --- shader module ---
        let module = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qlang_elementwise"),
            source: wgpu::ShaderSource::Wgsl(kernel.shader_source.as_str().into()),
        });

        // --- buffers ---
        use wgpu::util::DeviceExt;

        let buf_a = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_a"),
            contents: bytemuck_cast_slice(&input_a[..n]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_b = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input_b"),
            contents: bytemuck_cast_slice(&input_b[..n]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_out = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // params uniform: n_elements
        let params_data = (n as u32).to_le_bytes();
        let buf_params = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &params_data,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // staging buffer for readback
        let buf_staging = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- pipeline ---
        let pipeline = inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
            ],
        });

        // --- dispatch ---
        let workgroup_count = ((n as u32) + kernel.workgroup_size - 1) / kernel.workgroup_size;
        let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elementwise_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elementwise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, buf_bytes);
        inner.queue.submit(Some(encoder.finish()));

        // --- readback ---
        let slice = buf_staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        inner.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| GpuError::ExecutionError(format!("map recv failed: {e}")))?
            .map_err(|e| GpuError::ExecutionError(format!("map failed: {e}")))?;

        let mapped = slice.get_mapped_range();
        let output: Vec<f32> = mapped
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(mapped);
        buf_staging.unmap();

        Ok(GpuResult {
            data: output,
            execution_time_us: start.elapsed().as_micros() as u64,
            used_gpu: true,
        })
    }

    /// Execute a matmul kernel.
    pub fn execute_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        // Fall back to CPU when no real GPU is present.
        let inner = match &self.inner {
            Some(i) => i,
            None => {
                let result = cpu_matmul(a, b, m, k, n);
                return Ok(GpuResult {
                    data: result,
                    execution_time_us: start.elapsed().as_micros() as u64,
                    used_gpu: false,
                });
            }
        };

        // Generate the tiled matmul shader (reuse the compile crate's generator would be
        // ideal, but we inline the same WGSL format here for self-containedness).
        let shader_source = generate_matmul_wgsl();
        let module = inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("qlang_matmul"),
            source: wgpu::ShaderSource::Wgsl(shader_source.as_str().into()),
        });

        use wgpu::util::DeviceExt;

        let buf_a = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_A"),
            contents: bytemuck_cast_slice(a),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_b = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("matmul_B"),
            contents: bytemuck_cast_slice(b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_size = (m * n * std::mem::size_of::<f32>()) as u64;
        let buf_c = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_C"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dims uniform: M, K, N as u32
        let dims: [u32; 3] = [m as u32, k as u32, n as u32];
        let dims_bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
        // Pad to 16 bytes (uniform buffers need 16-byte alignment on many backends)
        let mut dims_padded = dims_bytes.clone();
        dims_padded.resize(16, 0);
        let buf_dims = inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: &dims_padded,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let buf_staging = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline = inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: buf_dims.as_entire_binding() },
            ],
        });

        let tile = 16u32;
        let wg_x = (n as u32 + tile - 1) / tile;
        let wg_y = (m as u32 + tile - 1) / tile;

        let mut encoder = inner.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_c, 0, &buf_staging, 0, out_size);
        inner.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = buf_staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        inner.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| GpuError::ExecutionError(format!("map recv failed: {e}")))?
            .map_err(|e| GpuError::ExecutionError(format!("map failed: {e}")))?;

        let mapped = slice.get_mapped_range();
        let output: Vec<f32> = mapped
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(mapped);
        buf_staging.unmap();

        Ok(GpuResult {
            data: output,
            execution_time_us: start.elapsed().as_micros() as u64,
            used_gpu: true,
        })
    }
}

/// Cast `&[f32]` to `&[u8]` without pulling in bytemuck as a dependency.
#[cfg(feature = "gpu")]
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f32>())
    }
}

/// Inline matmul WGSL shader (same tiled algorithm as qlang-compile's `matmul_wgsl`).
#[cfg(feature = "gpu")]
fn generate_matmul_wgsl() -> String {
    r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_A: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_B: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: f32 = 0.0;
    let n_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        if (row < dims.M && a_col < dims.K) {
            tile_A[local_row][local_col] = A[row * dims.K + a_col];
        } else {
            tile_A[local_row][local_col] = 0.0;
        }

        if (b_row < dims.K && col < dims.N) {
            tile_B[local_row][local_col] = B[b_row * dims.N + col];
        } else {
            tile_B[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_A[local_row][i] * tile_B[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
"#
    .to_string()
}

#[cfg(feature = "gpu")]
/// Information about available GPU devices.
pub fn gpu_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    if GpuDevice::is_available() {
        let dev = GpuDevice::new();
        info.insert("backend".into(), dev.name.clone());
        info.insert("max_buffer_size".into(), dev.max_buffer_size.to_string());
        info.insert("max_workgroup_size".into(), dev.max_workgroup_size.to_string());
        info.insert("real_gpu".into(), dev.is_real_gpu.to_string());
    } else {
        info.insert("backend".into(), "No GPU adapter found".into());
    }
    info.insert("feature_flag".into(), "gpu (enabled)".into());
    info
}

// ---------------------------------------------------------------------------
// CPU fallback implementation (no "gpu" feature)
// ---------------------------------------------------------------------------
#[cfg(not(feature = "gpu"))]
impl GpuDevice {
    /// Create a CPU fallback device for environments without GPU.
    pub fn cpu_fallback() -> Self {
        Self {
            name: "CPU Fallback (WGSL Simulator)".into(),
            max_buffer_size: usize::MAX,
            max_workgroup_size: 256,
            is_real_gpu: false,
        }
    }

    /// Try to initialize a real GPU device.
    /// Without the `gpu` feature, always returns CPU fallback.
    pub fn new() -> Self {
        Self::cpu_fallback()
    }

    /// Check whether a real GPU adapter is available.
    /// Without the `gpu` feature, always returns `false`.
    pub fn is_available() -> bool {
        false
    }

    /// Compile a WGSL shader into a kernel.
    pub fn compile_kernel(
        &self,
        shader_source: &str,
        n_elements: usize,
    ) -> Result<GpuKernel, GpuError> {
        if !shader_source.contains("@compute") {
            return Err(GpuError::ShaderError(
                "shader must contain @compute entry point".into(),
            ));
        }
        Ok(GpuKernel {
            shader_source: shader_source.to_string(),
            n_elements,
            workgroup_size: self.max_workgroup_size,
        })
    }

    /// Execute a kernel with element-wise operation on two input buffers.
    pub fn execute_elementwise(
        &self,
        kernel: &GpuKernel,
        input_a: &[f32],
        input_b: &[f32],
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        let result = simulate_elementwise_shader(
            &kernel.shader_source,
            input_a,
            input_b,
            kernel.n_elements,
        )?;
        Ok(GpuResult {
            data: result,
            execution_time_us: start.elapsed().as_micros() as u64,
            used_gpu: false,
        })
    }

    /// Execute a matmul kernel.
    pub fn execute_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<GpuResult, GpuError> {
        let start = std::time::Instant::now();

        let result = cpu_matmul(a, b, m, k, n);
        Ok(GpuResult {
            data: result,
            execution_time_us: start.elapsed().as_micros() as u64,
            used_gpu: false,
        })
    }
}

#[cfg(not(feature = "gpu"))]
/// Information about available GPU devices.
pub fn gpu_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("backend".into(), "CPU Fallback (wgpu not enabled)".into());
    info.insert("feature_flag".into(), "gpu (disabled)".into());
    info.insert(
        "instructions".into(),
        "Enable GPU: cargo build --features gpu".into(),
    );
    info
}

// ---------------------------------------------------------------------------
// Shared helpers (used by both feature-gated paths)
// ---------------------------------------------------------------------------

/// CPU matmul: standard O(m*k*n) implementation.
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

/// Simulate WGSL element-wise shader on CPU by parsing the operation.
fn simulate_elementwise_shader(
    shader: &str,
    input_a: &[f32],
    input_b: &[f32],
    n_elements: usize,
) -> Result<Vec<f32>, GpuError> {
    let n = n_elements.min(input_a.len()).min(input_b.len());
    let mut output = vec![0.0f32; n];

    for i in 0..n {
        let a = input_a[i];
        let b = input_b[i];
        output[i] = if shader.contains("a + b") {
            a + b
        } else if shader.contains("a - b") {
            a - b
        } else if shader.contains("a * b") {
            a * b
        } else if shader.contains("a / b") {
            if b != 0.0 { a / b } else { 0.0 }
        } else if shader.contains("max(") && shader.contains("0.0") {
            a.max(0.0) // relu
        } else if shader.contains("1.0 / (1.0 + exp(") {
            1.0 / (1.0 + (-a).exp()) // sigmoid
        } else if shader.contains("tanh(") {
            a.tanh()
        } else {
            a // passthrough
        };
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_creation() {
        let dev = GpuDevice::cpu_fallback();
        assert!(!dev.is_real_gpu);
        assert_eq!(dev.max_workgroup_size, 256);
    }

    #[test]
    fn test_compile_kernel() {
        let dev = GpuDevice::new();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a + b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 1024).unwrap();
        assert_eq!(kernel.n_elements, 1024);
    }

    #[test]
    fn test_compile_kernel_invalid() {
        let dev = GpuDevice::new();
        let result = dev.compile_kernel("invalid shader", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_elementwise_add() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a + b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 4).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
        assert!(!result.used_gpu);
    }

    #[test]
    fn test_elementwise_mul() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = a * b; output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 3).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[2.0, 3.0, 4.0], &[5.0, 6.0, 7.0])
            .unwrap();
        assert_eq!(result.data, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_matmul_cpu() {
        let dev = GpuDevice::cpu_fallback();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dev.execute_matmul(&a, &b, 2, 2, 2).unwrap();
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_info() {
        let info = gpu_info();
        assert!(info.contains_key("backend"));
        assert!(info.contains_key("feature_flag"));
    }

    #[test]
    fn test_relu_shader() {
        let dev = GpuDevice::cpu_fallback();
        let shader = "@compute @workgroup_size(256)\nfn main() { let r0 = max(a, 0.0); output[i] = r0; }";
        let kernel = dev.compile_kernel(shader, 4).unwrap();
        let result = dev
            .execute_elementwise(&kernel, &[-1.0, 2.0, -3.0, 4.0], &[0.0; 4])
            .unwrap();
        assert_eq!(result.data, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_is_available() {
        // Should not panic regardless of feature flag
        let _available = GpuDevice::is_available();
    }
}
