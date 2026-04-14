//! GPU Training Binary — run on the RTX 2070 Super machine.
//! Usage: cargo run --release --bin gpu_train

use qlang_runtime::gpu_train::{GpuTrainConfig, train_cpu};

fn main() {
    println!("QLANG GPU Training — Mamba LM\n");

    let config = GpuTrainConfig {
        data_path: std::env::args().nth(1).unwrap_or_else(|| "data/wikitext2/train.txt".into()),
        output_path: std::env::args().nth(2).unwrap_or_else(|| "data/mamba_30m".into()),
        ..Default::default()
    };

    match train_cpu(&config) {
        Ok(()) => println!("\nTraining complete!"),
        Err(e) => eprintln!("Training failed: {e}"),
    }
}
