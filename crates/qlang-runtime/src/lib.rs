//! QLANG Runtime — Graph executor
//!
//! Executes QLANG graphs by:
//! 1. Topologically sorting nodes
//! 2. Executing each node in order
//! 3. Flowing tensor data along edges
//!
//! This is the interpreter backend (Phase 1).
//! Phase 2 will add LLVM JIT compilation.

pub mod accel;
pub mod checkpoint;
pub mod diagnostics;
pub mod executor;
pub mod profiler;
pub mod scheduler;
pub mod bench;
pub mod stdlib;
pub mod vm;
pub mod bytecode;
pub mod debugger;

pub mod mcp_bridge;
pub mod config;
pub mod types;
pub mod unified;
pub mod concurrency;
pub mod graph_ops;
pub mod registry;
pub mod parallel;
pub mod hub;
pub mod ollama;
pub mod providers;
pub mod cloud_http;
pub mod openai_client;
pub mod anthropic_client;
pub mod gemini_client;
pub mod groq_client;
pub mod deepseek_client;
pub mod orchestrator;
pub mod tokenizer;
pub mod web_server;
pub mod hybrid_router;

#[cfg(test)]
pub(crate) mod test_data {
    #[derive(Clone, Debug)]
    pub struct MnistLikeData {
        pub train_images: Vec<f32>,
        pub train_labels: Vec<u8>,
        pub test_images: Vec<f32>,
        pub test_labels: Vec<u8>,
        pub n_train: usize,
        pub n_test: usize,
        pub image_size: usize,
    }

    #[derive(Clone, Copy)]
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
            Self(s)
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }

        fn next_f32(&mut self) -> f32 {
            let v = self.next_u32() as f32 / (u32::MAX as f32);
            v
        }
    }

    pub fn mnist_like(n_train: usize, n_test: usize) -> MnistLikeData {
        generate_classification_data(n_train, n_test, 784, 10, 0xC0FFEEu64)
    }

    fn generate_classification_data(
        n_train: usize,
        n_test: usize,
        dim: usize,
        n_classes: usize,
        seed: u64,
    ) -> MnistLikeData {
        let mut rng = Rng::new(seed);

        let mut protos: Vec<Vec<f32>> = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mut p = vec![0.0f32; dim];
            for k in 0..32usize {
                let idx = ((c * 97 + k * 193) % dim) as usize;
                p[idx] = 1.0;
            }
            protos.push(p);
        }

        let mut train_images = vec![0.0f32; n_train * dim];
        let mut train_labels = vec![0u8; n_train];
        let mut test_images = vec![0.0f32; n_test * dim];
        let mut test_labels = vec![0u8; n_test];

        for i in 0..n_train {
            let cls = (i % n_classes) as u8;
            train_labels[i] = cls;
            let proto = &protos[cls as usize];
            let base = i * dim;
            for j in 0..dim {
                let noise = (rng.next_f32() - 0.5) * 0.15;
                let v = proto[j] + noise;
                train_images[base + j] = v.clamp(0.0, 1.0);
            }
        }

        for i in 0..n_test {
            let cls = (i % n_classes) as u8;
            test_labels[i] = cls;
            let proto = &protos[cls as usize];
            let base = i * dim;
            for j in 0..dim {
                let noise = (rng.next_f32() - 0.5) * 0.15;
                let v = proto[j] + noise;
                test_images[base + j] = v.clamp(0.0, 1.0);
            }
        }

        MnistLikeData {
            train_images,
            train_labels,
            test_images,
            test_labels,
            n_train,
            n_test,
            image_size: dim,
        }
    }
}
