//! GPU compute backend via wgpu -- cross-platform matmul on NVIDIA, AMD, Intel, and Apple GPUs.
//!
//! This module provides a lightweight `GpuContext` that initializes a wgpu device once and
//! exposes `matmul`, `matmul_at_b`, and `matmul_a_bt` functions that match the API in
//! `accel.rs`. All GPU code is gated behind `#[cfg(feature = "gpu")]`.
//!
//! The WGSL compute shader uses tiled matrix multiplication with 16x16 workgroups for
//! good performance across all GPU vendors.

#[cfg(feature = "gpu")]
pub mod gpu_compute {
    use std::sync::OnceLock;

    /// Embedded WGSL tiled matmul shader (16x16 tiles, works on all wgpu backends).
    const MATMUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let lr = lid.y;
    let lc = lid.x;

    var sum: f32 = 0.0;
    let n_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let a_col = t * TILE + lc;
        let b_row = t * TILE + lr;

        if (row < dims.M && a_col < dims.K) {
            tile_a[lr][lc] = A[row * dims.K + a_col];
        } else {
            tile_a[lr][lc] = 0.0;
        }

        if (b_row < dims.K && col < dims.N) {
            tile_b[lr][lc] = B[b_row * dims.N + col];
        } else {
            tile_b[lr][lc] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE; i = i + 1u) {
            sum = sum + tile_a[lr][i] * tile_b[i][lc];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
"#;

    /// Persistent GPU context: device + queue + pre-compiled matmul pipeline.
    pub struct GpuContext {
        device: wgpu::Device,
        queue: wgpu::Queue,
        matmul_pipeline: wgpu::ComputePipeline,
        bind_group_layout: wgpu::BindGroupLayout,
        adapter_name: String,
    }

    impl std::fmt::Debug for GpuContext {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("GpuContext")
                .field("adapter", &self.adapter_name)
                .finish()
        }
    }

    impl GpuContext {
        /// Attempt to create a GPU context.  Returns `None` if no adapter/device is found.
        pub fn new() -> Option<Self> {
            pollster::block_on(Self::new_async())
        }

        async fn new_async() -> Option<Self> {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;

            let adapter_info = adapter.get_info();
            let adapter_name = format!(
                "{} ({}, {})",
                adapter_info.name,
                adapter_info.backend.to_str(),
                match adapter_info.device_type {
                    wgpu::DeviceType::DiscreteGpu => "discrete",
                    wgpu::DeviceType::IntegratedGpu => "integrated",
                    wgpu::DeviceType::VirtualGpu => "virtual",
                    wgpu::DeviceType::Cpu => "cpu",
                    wgpu::DeviceType::Other => "other",
                }
            );

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("qlang-gpu-compute"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .ok()?;

            // Pre-compile the matmul shader and pipeline.
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
            });

            let matmul_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("matmul_pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let bind_group_layout = matmul_pipeline.get_bind_group_layout(0);

            Some(Self {
                device,
                queue,
                matmul_pipeline,
                bind_group_layout,
                adapter_name,
            })
        }

        /// Human-readable adapter description (e.g. "NVIDIA GeForce RTX 4090 (Vulkan, discrete)").
        pub fn adapter_name(&self) -> &str {
            &self.adapter_name
        }

        // ---- internal dispatch helper ------------------------------------------------

        /// Run the pre-compiled matmul pipeline: C = op(A) * op(B).
        ///
        /// `a_data` and `b_data` are the raw row-major f32 buffers to upload.
        /// `m`, `k`, `n` are the logical dimensions of the output C = [m, n].
        fn dispatch_matmul(
            &self,
            a_data: &[f32],
            b_data: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> Vec<f32> {
            use wgpu::util::DeviceExt;

            let buf_a = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("A"),
                    contents: cast_f32_to_u8(a_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let buf_b = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("B"),
                    contents: cast_f32_to_u8(b_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let out_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
            let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("C"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Dims uniform -- padded to 16 bytes for alignment.
            let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
            let dims_bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
            let buf_dims =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("dims"),
                        contents: &dims_bytes,
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf_a.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_c.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf_dims.as_entire_binding(),
                    },
                ],
            });

            let tile: u32 = 16;
            let wg_x = (n as u32 + tile - 1) / tile;
            let wg_y = (m as u32 + tile - 1) / tile;

            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("matmul_enc"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.matmul_pipeline);
                pass.set_bind_group(0, Some(&bind_group), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            encoder.copy_buffer_to_buffer(&buf_c, 0, &buf_staging, 0, out_bytes);
            self.queue.submit(Some(encoder.finish()));

            // Readback.
            let slice = buf_staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .expect("gpu readback channel closed")
                .expect("gpu map failed");

            let mapped = slice.get_mapped_range();
            let output: Vec<f32> = mapped
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            drop(mapped);
            buf_staging.unmap();

            output
        }

        // ---- public matmul API -------------------------------------------------------

        /// C = A * B  where A is [m, k], B is [k, n], result is [m, n]. All row-major.
        pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
            debug_assert_eq!(a.len(), m * k, "gpu matmul: a must be [m,k]");
            debug_assert_eq!(b.len(), k * n, "gpu matmul: b must be [k,n]");
            self.dispatch_matmul(a, b, m, k, n)
        }

        /// C = A^T * B  where A is stored as [k, m], B is [k, n], result is [m, n].
        ///
        /// We transpose A on the CPU (cheap compared to GPU dispatch overhead for the
        /// matrix sizes typical in QLANG training) and then run the standard matmul shader.
        pub fn matmul_at_b(
            &self,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Vec<f32> {
            debug_assert_eq!(a.len(), k * m, "gpu matmul_at_b: a must be [k,m]");
            debug_assert_eq!(b.len(), k * n, "gpu matmul_at_b: b must be [k,n]");
            // Transpose A from [k, m] to [m, k].
            let mut at = vec![0.0f32; m * k];
            for r in 0..k {
                for c in 0..m {
                    at[c * k + r] = a[r * m + c];
                }
            }
            self.dispatch_matmul(&at, b, m, k, n)
        }

        /// C = A * B^T  where A is [m, k], B is stored as [n, k], result is [m, n].
        pub fn matmul_a_bt(
            &self,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k: usize,
        ) -> Vec<f32> {
            debug_assert_eq!(a.len(), m * k, "gpu matmul_a_bt: a must be [m,k]");
            debug_assert_eq!(b.len(), n * k, "gpu matmul_a_bt: b must be [n,k]");
            // Transpose B from [n, k] to [k, n].
            let mut bt = vec![0.0f32; k * n];
            for r in 0..n {
                for c in 0..k {
                    bt[c * n + r] = b[r * k + c];
                }
            }
            self.dispatch_matmul(a, &bt, m, k, n)
        }
    }

    // ---- singleton accessor ----------------------------------------------------------

    static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();

    /// Return a reference to the global GPU context, initializing on first call.
    /// Returns `None` if no GPU adapter is available.
    pub fn get_gpu() -> Option<&'static GpuContext> {
        GPU_CTX.get_or_init(|| GpuContext::new()).as_ref()
    }

    // ---- helpers ---------------------------------------------------------------------

    /// Reinterpret `&[f32]` as `&[u8]`.
    fn cast_f32_to_u8(data: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        }
    }
}

// ---- tests ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::gpu_compute::*;

    /// Helper: CPU reference matmul for correctness checks.
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }

    fn assert_approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: gpu={x}, cpu={y}"
            );
        }
    }

    #[test]
    fn test_gpu_context_creation() {
        // Should not panic -- returns None if no GPU.
        let _ctx = GpuContext::new();
    }

    #[test]
    fn test_matmul_2x2() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_2x2");
                return;
            }
        };
        let a = vec![1.0, 2.0, 3.0, 4.0f32];
        let b = vec![5.0, 6.0, 7.0, 8.0f32];
        let result = gpu.matmul(&a, &b, 2, 2, 2);
        let expected = cpu_matmul(&a, &b, 2, 2, 2);
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_non_square() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_non_square");
                return;
            }
        };
        // A [2,3] * B [3,4] = C [2,4]
        let a: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let result = gpu.matmul(&a, &b, 2, 4, 3);
        let expected = cpu_matmul(&a, &b, 2, 4, 3);
        assert_approx(&result, &expected, 1e-4);
    }

    #[test]
    fn test_matmul_medium() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_medium");
                return;
            }
        };
        // 64x64 -- exercises multiple tiles
        let m = 64;
        let k = 48;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let result = gpu.matmul(&a, &b, m, n, k);
        let expected = cpu_matmul(&a, &b, m, n, k);
        assert_approx(&result, &expected, 0.1);
    }

    #[test]
    fn test_matmul_large() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_large");
                return;
            }
        };
        let m = 128;
        let k = 256;
        let n = 128;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 100) as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 100) as f32) * 0.001).collect();
        let result = gpu.matmul(&a, &b, m, n, k);
        let expected = cpu_matmul(&a, &b, m, n, k);
        assert_approx(&result, &expected, 0.5);
    }

    #[test]
    fn test_matmul_at_b() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_at_b");
                return;
            }
        };
        // A stored [k=3, m=2], A^T = [2,3], B = [3,2] -> C = [2,2]
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0f32];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32];
        let result = gpu.matmul_at_b(&a, &b, 2, 2, 3);
        // Expected: A^T * B = [[1,2,3],[4,5,6]] * [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_a_bt() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_a_bt");
                return;
            }
        };
        // A = [2,3], B stored [n=2, k=3], B^T = [3,2] -> C = [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let b = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0f32];
        let result = gpu.matmul_a_bt(&a, &b, 2, 2, 3);
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        assert_approx(&result, &expected, 1e-3);
    }

    #[test]
    fn test_matmul_identity() {
        let gpu = match get_gpu() {
            Some(g) => g,
            None => {
                eprintln!("No GPU available -- skipping test_matmul_identity");
                return;
            }
        };
        let eye = vec![1.0, 0.0, 0.0, 1.0f32];
        let b = vec![1.0, 2.0, 3.0, 4.0f32];
        let result = gpu.matmul(&eye, &b, 2, 2, 2);
        assert_approx(&result, &b, 1e-5);
    }

    #[test]
    fn test_singleton_accessor() {
        // Should not panic -- the singleton pattern must work.
        let a = get_gpu();
        let b = get_gpu();
        match (a, b) {
            (Some(ga), Some(gb)) => {
                // Same pointer.
                assert!(std::ptr::eq(ga, gb));
            }
            (None, None) => {} // no GPU, fine
            _ => panic!("singleton inconsistency"),
        }
    }
}
