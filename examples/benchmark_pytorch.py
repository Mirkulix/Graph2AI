#!/usr/bin/env python3
"""PyTorch Benchmark Comparison — Run the same ops as benchmark_suite.rs.

Usage:
    pip install torch
    python examples/benchmark_pytorch.py

Compare output with: cargo run --release --example benchmark_suite
"""

import time
import json
import sys

try:
    import torch
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


def bench(name, size, iterations, fn):
    """Run benchmark and return results."""
    # Warmup
    for _ in range(3):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed_us = (time.perf_counter() - start) * 1e6
        times.append(elapsed_us)

    mean_us = sum(times) / len(times)
    min_us = min(times)
    max_us = max(times)
    throughput = size / mean_us  # M elements/sec

    return {
        "name": name,
        "size": size,
        "iters": iterations,
        "mean_us": round(mean_us, 1),
        "min_us": round(min_us, 1),
        "max_us": round(max_us, 1),
        "throughput_mops": round(throughput, 2),
    }


def main():
    device = "cpu"
    results = []

    print("=" * 60)
    print("  PyTorch Benchmark Suite (CPU)")
    print("=" * 60)

    # Element-wise operations
    print("\n▸ Element-wise Operations")
    for size in [1024, 16384, 262144, 1048576]:
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        iters = 50 if size > 100000 else 200

        results.append(bench(f"add_{size}", size, iters, lambda: a + b))
        results.append(bench(f"relu_{size}", size, iters, lambda: torch.relu(a)))
        results.append(bench(f"sigmoid_{size}", size, iters, lambda: torch.sigmoid(a)))

    # Matrix multiplication
    print("▸ Matrix Multiplication")
    for m, k, n in [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]:
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)
        iters = 10 if m > 128 else 50
        results.append(bench(f"matmul_{m}x{k}x{n}", m * n, iters, lambda: a @ b))

    # Softmax
    print("▸ Softmax")
    for size in [100, 1000, 10000]:
        data = torch.randn(size, device=device)
        results.append(
            bench(f"softmax_{size}", size, 100, lambda: torch.softmax(data, dim=0))
        )

    # Print results
    print(f"\n{'Benchmark':<27} {'Size':>7} {'Mean(µs)':>10} {'Min(µs)':>10} {'Max(µs)':>10} {'M elem/s':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<27} {r['size']:>7} {r['mean_us']:>10.1f} {r['min_us']:>10.1f} {r['max_us']:>10.1f} {r['throughput_mops']:>10.2f}"
        )

    if "--json" in sys.argv:
        print("\n--- JSON ---")
        print(json.dumps(results))

    print(f"\nDone. {len(results)} benchmarks completed.")


if __name__ == "__main__":
    main()
