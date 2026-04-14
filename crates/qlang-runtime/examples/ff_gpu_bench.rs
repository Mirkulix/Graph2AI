//! Benchmark GPU Forward-Forward QAT on MNIST.
//!
//! Run with:
//!   LIBTORCH_USE_PYTORCH=1 cargo run --release --no-default-features \
//!     --features cuda --example ff_gpu_bench -p qlang-runtime

use qlang_runtime::forward_forward_gpu::train_ff_qat_gpu;
use qlang_runtime::mnist::MnistData;

fn main() {
    let paths = [
        "data/mnist",
        "../data/mnist",
        "/home/mirkulix/AI/neoqlang/qlang/data/mnist",
    ];
    let data = paths
        .iter()
        .filter_map(|p| MnistData::load_from_dir(p).ok())
        .next()
        .expect("MNIST not found in any expected path");

    let n_train = 60_000.min(data.n_train);
    let n_test = 10_000.min(data.n_test);
    let epochs = std::env::var("EPOCHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10usize);
    let batch_size = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize);

    println!(
        "[bench] n_train={} n_test={} epochs={} batch={}",
        n_train, n_test, epochs, batch_size
    );

    let layer_sizes = [794usize, 512, 256, 128];

    let mut last_elapsed = 0f64;
    let mut per_epoch_times: Vec<f64> = Vec::new();
    let start = std::time::Instant::now();

    let result = train_ff_qat_gpu(
        &data.train_images,
        &data.train_labels,
        &layer_sizes,
        10,
        n_train,
        n_test,
        &data.test_images,
        &data.test_labels,
        epochs,
        batch_size,
        0.03,
        2.0,
        |epoch, tern_acc, elapsed| {
            let dt = elapsed - last_elapsed;
            per_epoch_times.push(dt);
            last_elapsed = elapsed;
            println!(
                "  epoch {:>2}/{}: tern_acc={:.4} ({:.2}%)  epoch_time={:.2}s  total={:.1}s",
                epoch,
                epochs,
                tern_acc,
                tern_acc * 100.0,
                dt,
                elapsed
            );
        },
    )
    .expect("GPU training failed");

    let total = start.elapsed().as_secs_f64();
    let avg_epoch = per_epoch_times.iter().copied().sum::<f64>() / per_epoch_times.len().max(1) as f64;

    println!("\n=== RESULT ===");
    println!("f32_accuracy     = {:.4} ({:.2}%)", result.f32_accuracy, result.f32_accuracy * 100.0);
    println!("ternary_accuracy = {:.4} ({:.2}%)", result.ternary_accuracy, result.ternary_accuracy * 100.0);
    println!("total_weights    = {}", result.total_weights);
    println!("total_time       = {:.2}s", total);
    println!("avg epoch time   = {:.2}s", avg_epoch);
    println!(
        "speedup vs CPU   = {:.1}x  (baseline 26s/epoch)",
        26.0 / avg_epoch.max(1e-3)
    );

    let mut min_conf = u32::MAX;
    let mut max_conf = 0u32;
    for row in &result.confusion_matrix {
        for &v in row {
            if v < min_conf { min_conf = v; }
            if v > max_conf { max_conf = v; }
        }
    }
    println!("confusion_matrix [10x10]: min={} max={}", min_conf, max_conf);
    for row in &result.confusion_matrix {
        for v in row {
            print!("{:>5}", v);
        }
        println!();
    }
}
