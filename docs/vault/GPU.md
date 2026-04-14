# GPU

QLANG unterstuetzt mehrere Hardware-Beschleunigungsbackends: Apple Silicon, wgpu Compute Shaders (NVIDIA/AMD/Intel), und Pure Rust Fallback.

## Acceleration Stack

```
                    ┌─────────────────────┐
                    │   QLANG Runtime      │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   wgpu Compute         Apple MLX/Accel        Pure Rust
   --features gpu       macOS default          Fallback
   NVIDIA/AMD/Intel     Metal + BLAS           ueberall
```

## wgpu Compute Shaders (gpu_compute.rs)

Cross-platform GPU-Matmul via wgpu -- funktioniert auf NVIDIA, AMD, Intel und Apple GPUs.

```bash
cargo build --release --features gpu
```

### WGSL Tiled Matmul Shader

Der eingebettete Shader verwendet 16x16 Tile-basierte Matrix-Multiplikation:

```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Tiled matmul mit shared memory
}
```

### GpuContext API

```rust
use qlang_runtime::gpu_compute::gpu_compute::{GpuContext, get_gpu};

// Singleton-Zugriff (initialisiert beim ersten Aufruf)
if let Some(gpu) = get_gpu() {
    println!("GPU: {}", gpu.adapter_name());

    // C = A * B
    let c = gpu.matmul(&a, &b, m, n, k);

    // C = A^T * B (fuer Gradient dW = X^T @ dY)
    let c = gpu.matmul_at_b(&a, &b, m, n, k);

    // C = A * B^T (fuer Backprop d_hidden = d_logits @ W^T)
    let c = gpu.matmul_a_bt(&a, &b, m, n, k);
}
```

### Unterstuetzte GPUs

| Hersteller | Backend | Beispiele |
|-----------|---------|-----------|
| NVIDIA | Vulkan/DX12 | RTX 4090, RTX 3080, A100, T4 |
| AMD | Vulkan | RX 7900, RX 6800 |
| Intel | Vulkan | Arc A770, Iris Xe |
| Apple | Metal | M1, M2, M3, M4 |

## Apple Accelerate (Standard auf macOS)

Matrix-Operationen nutzen automatisch Apples Accelerate Framework via `cblas_sgemm`. Hardware-optimiertes BLAS auf M1/M2/M3/M4 mit NEON + AMX Co-Prozessoren.

Drei optimierte Kernel in `accel.rs`:

| Funktion | Operation | Anwendung |
|----------|-----------|-----------|
| `matmul(a, b, m, n, k)` | C = A * B | Forward Pass |
| `matmul_at_b(a, b, m, n, k)` | C = A^T * B | Gradient (dW = X^T @ dY) |
| `matmul_a_bt(a, b, m, n, k)` | C = A * B^T | Backprop (d_hidden = dY @ W^T) |

Keine externen Crates -- direktes FFI zu `cblas_sgemm` via `unsafe extern "C"`.

## Apple MLX (Metal GPU)

Optional, via `--features mlx`:

```bash
cargo build --release --features mlx
```

Routing aller Matmul-Operationen ueber Apples MLX Framework fuer Metal GPU Beschleunigung. Verwendet `mlx-rs` Crate.

## Multi-GPU Distributed Training

Mehrere GPUs koennen fuer Data-Parallel Training verwendet werden. Siehe [[Training]].

```rust
let config = DistributedConfig {
    n_workers: 4,
    strategy: ParallelStrategy::DataParallel,
    batch_size_per_worker: 32,
    gradient_accumulation_steps: 1,
};
```

### GradientBuffer

```rust
let mut buf = GradientBuffer::new(&[w1_size, b1_size, w2_size]);
buf.accumulate(&worker_gradients);
buf.average();
apply_fn(buf.gradients());
```

## Device Detection

```bash
qlang-cli devices
```

```rust
use qlang_runtime::distributed_train::detect_devices;

let devices = detect_devices();
for d in &devices {
    println!("{} ({}) - {}", d.name, d.device_type, d.compute_capability);
}
```

Erkennt automatisch:

| DeviceType | Beschreibung |
|-----------|-------------|
| `Cpu` | Standard CPU |
| `NvidiaGpu` | NVIDIA GPU (CUDA/Vulkan) |
| `AmdGpu` | AMD GPU (Vulkan/ROCm) |
| `IntelGpu` | Intel GPU (integriert oder diskret) |
| `AppleMlx` | Apple Metal via MLX |
| `Wgpu` | Generischer wgpu Adapter |

## Backend Detection

```rust
use qlang_runtime::accel;
println!("{}", accel::backend_name());
// "Apple MLX (Metal GPU)"        -- mit --features mlx auf macOS
// "Apple Accelerate (CPU BLAS)"  -- macOS Standard
// "Pure Rust (fallback)"         -- Linux / Windows
```

## LLVM SIMD

Fuer Nicht-Apple Plattformen generiert der LLVM-Compiler AVX2-vektorisierten Code (`simd.rs`):

- 256-bit breite Operationen (8 x f32 pro Instruktion)
- Aligned Memory Allocator fuer SIMD (`aligned.rs`)
- Automatische Vektorisierung von Element-weisen Operationen

## WGSL Shader Generation

Der Compiler kann WGSL Compute Shaders fuer Browser-basierte GPU-Ausfuehrung generieren:

```bash
qlang-cli gpu model.qlg.json > model.wgsl
```

## Compilation Targets

| Target | Backend | Typischer Speedup |
|--------|---------|-------------------|
| LLVM JIT | Native x86-64 | 29x vs Interpreter |
| LLVM SIMD | AVX2 Vektoren | 29x |
| AOT .o | Native Object File | 29x |
| WASM | WebAssembly | 5-10x |
| GPU WGSL | WebGPU Shader | 100x+ |
| GPU wgpu | Compute Shader (Runtime) | Hardware-abhaengig |
| MLX | Metal GPU (Apple) | Hardware-abhaengig |
| Accelerate | CPU BLAS (Apple) | Hardware-abhaengig |

## Benchmarks

```
Benchmark: relu(a + b), Release Mode

Elements    Interpreter    LLVM JIT     Speedup
   1,024        10.0us     680ns        14.7x
  65,536       703.6us      44.6us      15.8x
1,048,576      21.4ms      728.4us      29.4x
```

MNIST Training:
```
784->128->10 MLP
Training time: 70ms (100% Accuracy, Toy Data)
Real MNIST: ~15.9s fuer Konvergenz
```

Siehe [[Architecture]] fuer die vollstaendige Kompilierungspipeline, [[Execution]] fuer das 3-Tier System.

#gpu #performance #acceleration #wgpu #distributed
