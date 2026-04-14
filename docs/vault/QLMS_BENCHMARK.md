# QLMS Binary Protocol vs MCP JSON — Real Measurements

**Task:** T016
**Date:** 2026-04-12
**Benchmark source:** `crates/qlang-runtime/examples/qlms_benchmark.rs`

## Hardware & Environment

- **CPU:** AMD Ryzen 9 3900X (12 cores / 24 threads)
- **Arch:** x86_64
- **Kernel:** Linux 6.19.11-200.fc43.x86_64 (Fedora 43)
- **Build:** `cargo --release`, `--no-default-features`, `LIBTORCH_USE_PYTORCH=1`
- **Run command:**
  ```bash
  LIBTORCH_USE_PYTORCH=1 cargo run --release --no-default-features \
    --example qlms_benchmark -p qlang-runtime
  ```

## Workload

- **TernaryBrain specialist:** 64-dim input × 24 neurons = **1536 ternary weights**
  (each weight ∈ {-1, 0, +1}, stored as i8)
- 1000-iteration averages for serialize/deserialize/RTT
- RTT: length-prefixed TCP loopback echo server (isolates serialize + socket cost
  from any heavyweight server framework; simulates a real MCP/HTTP POST envelope
  lower bound)
- 20-iteration warmup before RTT measurements

## Measured Results

| Method    | Serialized size | Serialize (ns) | Deserialize (ns) | Loopback RTT (µs) |
|-----------|-----------------|----------------|------------------|-------------------|
| QLMS bin  | **1 798 B**     | **7 658**      | **8 249**        | **77**            |
| MCP JSON  | 4 112 B         | 9 393          | 18 900           | 78                |

## Ratios (MCP / QLMS)

| Dimension    | Ratio | Winner           |
|--------------|-------|------------------|
| Payload size | 2.29× | QLMS 2.29× smaller |
| Serialize    | 1.23× | QLMS 1.23× faster  |
| Deserialize  | 2.29× | QLMS 2.29× faster  |
| Loopback RTT | 1.01× | ~tie (syscall bound) |

## Interpretation

- **Size win is real and consistent.** QLMS stores each ternary weight as a
  single raw byte (i8). JSON must encode each weight as an ASCII integer with
  commas — averaging ~2.2 chars/weight plus field overhead. For a 1536-weight
  specialist: 1798 B vs 4112 B (2.29× smaller).
- **Deserialize win is large (2.29×).** JSON parsing has to tokenize every
  weight and reallocate a `Vec<i64>` before downcasting; QLMS does a single
  `memcpy` of the weight slab plus fixed LE integer reads.
- **Serialize win is modest (1.23×).** `serde_json::to_vec` is well-optimized
  and amortizes nicely over the small message; QLMS still wins because it
  avoids per-weight itoa and UTF-8 bookkeeping.
- **RTT is loopback-bound.** Both payloads fit in a single MSS — TCP
  localhost syscall overhead dominates (~77 µs). On a real network with
  bandwidth-bound links, the 2.29× size reduction would translate directly
  to 2.29× faster RTT.

## Caveats

- **Localhost only.** No real network, no TLS, no proxies — a strict lower
  bound. The size advantage will widen once you include gzip/transport framing
  differences (JSON compresses well, but so does the ternary byte slab).
- **Single-shot frames.** No batching or pipelining — a realistic MCP server
  that accumulates multiple tool calls per TCP connection would amortize RTT
  further, equally for both protocols.
- **Simulated MCP envelope.** The JSON shape matches a typical
  `jsonrpc`/`method`/`params` weight transfer but is not negotiated against a
  live MCP peer. The size comparison is apples-to-apples for the payload
  itself.
- **HMAC included on QLMS side.** QLMS frame carries a 32-byte HMAC-SHA256;
  the MCP payload includes an equivalent `hmac_hex` field (64 hex chars). The
  security surface is comparable.
- **Small payload regime.** At 1.5 KB these numbers are nanosecond-scale;
  with large weight tensors (1M+ weights) the 2.29× size ratio compounds into
  multi-MB transfer differences.

## Conclusion

For AI-to-AI ternary weight transfer, **QLMS binary is 2.29× smaller and
2.29× faster to deserialize than JSON-RPC** for the exact same information
content. Serialize speed and localhost RTT are closer but still favor QLMS.
The protocol's advantage scales linearly with weight-array size — the
benefit grows for larger specialists.
