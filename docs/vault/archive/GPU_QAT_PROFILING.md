# GPU QAT Hot-Loop Profiling — `forward_forward_gpu.rs`

**Baseline:** 33.6x speedup vs CPU, ~0.77 s/epoch at batch=100, 60k MNIST, layers [794, 512, 256, 128], 3 FF layers.
**Target file:** `crates/qlang-runtime/src/forward_forward_gpu.rs`
**Method:** static code review of `ff_step_qat_gpu` + `run` hot loop (no training executed).

## Hot Loop Breakdown (per batch)

Per batch (600/epoch at bs=100, 60k train), for each of 3 layers the inner step does:

| Step | Op | Tensor shapes (L1) | Notes |
|---|---|---|---|
| 1 | `shadow.abs().mean()` | [out×in] → scalar | reduction + `double_value` sync |
| 2 | `(shadow/γ).round().clamp` | [out×in] | 3 elementwise ops |
| 3 | `ternary * γ` | [out×in] | 1 elementwise op |
| 4 | `scaled.tr()` | [in×out] view | free |
| 5 | `pos.matmul(w_t) + bias` | [B,in]·[in,out] | **hot matmul #1** |
| 6 | `.relu()` | [B,out] | |
| 7 | `neg.matmul(w_t) + bias` | [B,in]·[in,out] | **hot matmul #2** (same w_t) |
| 8 | `pow(2).sum_dim` × 2 | [B,out] → [B] | goodness pos/neg |
| 9 | `sigmoid(g - θ)` × 2 | [B] | |
| 10 | `(1-p).unsqueeze * act * 2` × 2 | [B,out] | d_pos, d_neg |
| 11 | `d_pos.tr().matmul(pos_input)` | [out,B]·[B,in] | **hot matmul #3** |
| 12 | `d_neg.tr().matmul(neg_input)` | [out,B]·[B,in] | **hot matmul #4** |
| 13 | `sum_dim(0)` × 2 | [out] | bias grads |
| 14 | `shadow + dw*lr`, `bias + db*lr` | [out,in], [out] | STE update |

**Per-step op count:** 4 large matmuls + ~15 elementwise/reduction kernels. At bs=100 the matmuls are small enough that kernel launch overhead and elementwise glue dominate at ~60% of wall time (educated estimate — confirm with `nvprof`).

## Optimization Table

| # | Optimization | Current state | Est. speedup | Effort | Notes |
|---|---|---|---|---|---|
| 1 | **Concat pos+neg into single forward** ([2B,in]·[in,out] then split) | 2 separate matmuls (lines 131, 133) | **1.4–1.7x** on fwd; ~1.2–1.3x epoch | 30 min | GEMM efficiency scales with batch; one kernel launch |
| 2 | **Same for backward** (stack `d_pos`/`d_neg` and `pos_input`/`neg_input`, single matmul, subtract halves) | 2 separate matmuls (line 159) | **1.3–1.5x** on bwd | 45 min | Combine with #1 as a unit |
| 3 | **Bump batch size 100 → 256** | bs=100 hardcoded in bench (line 28) | **1.3–1.5x** | 5 min | Better GPU utilization; watch LR scaling |
| 4 | **Cache ternarize per layer across steps** | recomputed each batch (line 126) | 1.0x | N/A | Must stay per-batch: shadow updates every step |
| 5 | **Avoid `gamma_t.double_value(&[])` host sync** (use tensor-valued gamma, keep on GPU) | host sync per layer per batch (line 79) | **1.1–1.2x** | 20 min | Biggest "hidden" win — 3 syncs × 600 batches = 1800/epoch stalls |
| 6 | **Fuse absmean+quantize into single CUDA graph / scripted op** | 4 kernels | 1.1–1.2x | 2 h | tch-rs doesn't expose CUDA Graphs easily; low ROI |
| 7 | **Half-precision (f16/bf16) shadow + f32 accumulate** | pure f32 | unknown (1.3–1.8x typical on Turing, but 2070S has no real tensor-core bf16 path for GEMM — fp16 yes) | 2–3 h | Requires careful STE stability; Turing fp16 tensor cores active for matmul |
| 8 | **Replace `pow(2).sum_dim` with `(x*x).sum_dim`** | `pow_tensor_scalar(2.0)` | 1.02x | 2 min | pow() dispatches generic kernel; square is cheaper. Minor but free. |
| 9 | **Replace `sum_dim` reduction path for `db` with direct `d_pos.sum(0) - d_neg.sum(0)`** | fine already | 1.0x | — | No change needed |
| 10 | **Parallel CUDA streams** (overlap data fetch + compute) | dataset fully resident on GPU already | **1.0x** | N/A | No upload to overlap; not applicable |
| 11 | **Convert `narrow` slices to `index_select` with shuffled indices** | sequential `narrow` | 1.0x | — | Already zero-copy view; no gain |
| 12 | **Skip `normalize_gpu` recomputing `pow().sum().sqrt()`** → `x.norm(dim=1, keepdim=true)` fused | 3 kernels | 1.05x | 10 min | Marginal |

## Identified Hidden Stalls

- **Line 79**: `gamma_t.double_value(&[])` forces a **GPU→CPU sync** every call. Called 2× per layer per batch (once in training, recomputed in eval), plus inside eval loops. At 600 batches × 3 layers = 1800 host syncs/epoch. This is likely the single largest unnecessary overhead relative to its simplicity.
- **Line 81–82**: `/gamma` where gamma is f64 scalar triggers scalar-tensor broadcast path; if gamma stays on-GPU as a 0-d tensor the sync disappears.

## Top 3 Recommendations (impact / effort)

1. **Eliminate gamma host-sync** (line 79 → keep `gamma_t` as tensor, use `shadow / &gamma_t`) — **~15 min, ~1.15x.** Biggest ROI by far.
2. **Concatenate pos+neg batches for both forward & backward matmuls** — **~45 min total, ~1.4–1.6x.** Collapses 4 matmuls to 2; GPU utilization at bs=100 benefits enormously.
3. **Bump batch_size from 100 to 256** — **~5 min, ~1.3x.** Trivially gated via env var; may need LR retune.

Combined est: 33.6x → **~70x** over CPU (≈0.35 s/epoch). Implementing #2+#3 together is cleanest: concat gives [2·bs, in] so each matmul is effectively bs=512 when bs=256.

## Implemented (this task)

**Bench batch-size env-overridable** — `crates/qlang-runtime/examples/ff_gpu_bench.rs` now reads `BATCH` env var (default 100 unchanged). Run `BATCH=256 EPOCHS=3 cargo run --example ff_gpu_bench --release --features cuda` to measure the #3 win without code changes.

No training executed per constraint; measured delta = not available (bench not run).

## Not Recommended

- **Flash Attention / fused attention kernels** — irrelevant, no attention in FF.
- **F.conv1d in place of matmul** — pure GEMM is already optimal for dense layers.
- **int8/int4 quantization of shadow** — shadow *must* stay high-precision for STE gradient accumulation; only the forward path is ternary.
