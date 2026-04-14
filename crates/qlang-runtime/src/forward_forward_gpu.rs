//! GPU-accelerated Forward-Forward training with QAT via tch-rs.
//!
//! All operations on CUDA GPU 1 (RTX 2070 Super). Dataset uploaded to GPU once,
//! no per-step CPU<->GPU transfers. Shadow f32 weights + STE-based QAT:
//!   forward uses absmean-ternarized weights, gradient flows to shadow.
//!
//! CRITICAL: Device MUST be Cuda(1) when 2+ GPUs available (GPU 0 drives display).
//!
//! Target: 50-100x speedup vs CPU `train_epoch_qat`.

/// Result of a GPU Forward-Forward QAT run.
pub struct GpuFFResult {
    pub f32_accuracy: f32,
    pub ternary_accuracy: f32,
    /// Flattened ternary weights for all layers (concatenated), values in {-1, 0, +1}.
    pub all_ternary_weights: Vec<f32>,
    pub total_weights: usize,
    pub total_time_secs: f64,
    /// 10x10 confusion matrix built with ternary predictions.
    pub confusion_matrix: Vec<Vec<u32>>,
    /// Layer output dims (for reconstruction if needed).
    pub layer_shapes: Vec<(usize, usize)>,
}

#[cfg(feature = "cuda")]
pub fn train_ff_qat_gpu(
    images: &[f32],
    labels: &[u8],
    layer_sizes: &[usize],
    n_classes: usize,
    n_train: usize,
    n_test: usize,
    test_images: &[f32],
    test_labels: &[u8],
    epochs: usize,
    batch_size: usize,
    lr: f32,
    threshold: f32,
    progress_cb: impl FnMut(usize, f32, f64),
) -> Result<GpuFFResult, String> {
    gpu_impl::run(
        images, labels, layer_sizes, n_classes, n_train, n_test,
        test_images, test_labels, epochs, batch_size, lr, threshold,
        progress_cb,
    )
}

#[cfg(not(feature = "cuda"))]
pub fn train_ff_qat_gpu(
    _images: &[f32],
    _labels: &[u8],
    _layer_sizes: &[usize],
    _n_classes: usize,
    _n_train: usize,
    _n_test: usize,
    _test_images: &[f32],
    _test_labels: &[u8],
    _epochs: usize,
    _batch_size: usize,
    _lr: f32,
    _threshold: f32,
    _progress_cb: impl FnMut(usize, f32, f64),
) -> Result<GpuFFResult, String> {
    Err("CUDA feature not enabled — rebuild with `--features cuda`".into())
}

#[cfg(feature = "cuda")]
mod gpu_impl {
    use super::GpuFFResult;
    use std::time::Instant;
    use tch::{Device, Kind, Tensor};

    /// Quantize shadow weight matrix → ternary {-1, 0, +1} via absmean.
    /// Returns (ternary_tensor, scaled_for_forward, gamma_scalar_f32).
    /// STE: forward uses `scaled = ternary * gamma` but gradient flows to shadow.
    fn absmean_ternarize(shadow: &Tensor) -> (Tensor, Tensor, f64) {
        // gamma = mean(|shadow|), scalar (f64 via double_value)
        let gamma_t = shadow.abs().mean(Kind::Float);
        let gamma = gamma_t.double_value(&[]).max(1e-8);
        // ternary = clamp(round(shadow / gamma), -1, 1)
        let ternary = (shadow / gamma).round().clamp(-1.0, 1.0);
        let scaled = &ternary * gamma;
        (ternary, scaled, gamma)
    }

    fn normalize_gpu(x: &Tensor) -> Tensor {
        // Per-row L2 norm clamp
        let dims = [1i64];
        let norm = x
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&dims[..], true, Kind::Float)
            .sqrt()
            .clamp_min(1e-8);
        x / norm
    }

    fn one_hot(labels: &Tensor, n_classes: i64, device: Device) -> Tensor {
        let n = labels.size()[0];
        let mut oh = Tensor::zeros(&[n, n_classes], (Kind::Float, device));
        oh.scatter_(
            1,
            &labels.unsqueeze(1),
            &Tensor::ones(&[n, 1], (Kind::Float, device)),
        )
    }

    fn tensor_to_f32_vec(t: &Tensor) -> Vec<f32> {
        let flat = t.flatten(0, -1).to_kind(Kind::Float).to(Device::Cpu);
        let numel = flat.numel();
        let mut out = vec![0f32; numel];
        flat.f_copy_data(&mut out, numel).unwrap_or(());
        out
    }

    /// Core FF+QAT step for one layer.
    /// Updates `shadow` and `bias` IN PLACE (returns new Tensors since tch consumes).
    /// Returns (new_shadow, new_bias, pos_out, neg_out) — pos/neg_out are pre-normalized activations.
    fn ff_step_qat_gpu(
        shadow: &Tensor,
        bias: &Tensor,
        pos_input: &Tensor,
        neg_input: &Tensor,
        lr: f64,
        threshold: f64,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let (_ternary, scaled, _gamma) = absmean_ternarize(shadow);

        // Forward: [B, in] @ [in, out] = [B, out]
        // scaled shape = [out, in], so use scaled.tr()
        let w_t = scaled.tr();
        let pos_pre = pos_input.matmul(&w_t) + bias;
        let pos_act = pos_pre.relu();
        let neg_pre = neg_input.matmul(&w_t) + bias;
        let neg_act = neg_pre.relu();

        // Goodness per sample: sum(act^2, dim=1) → [B]
        let dims = [1i64];
        let pos_g = pos_act
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&dims[..], false, Kind::Float);
        let neg_g = neg_act
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&dims[..], false, Kind::Float);

        // sigmoid(pos_g - θ) etc. Use logistic directly.
        // pos_p = sigmoid(pos_g - θ); we want (1 - pos_p) coefficient
        let pos_p = (&pos_g - threshold).sigmoid();
        let neg_p = (&neg_g - threshold).sigmoid();

        // d_pos = (1 - pos_p) * 2 * pos_act  [B, out]
        let one_minus_pos: Tensor = pos_p.ones_like() - &pos_p;
        let d_pos = one_minus_pos.unsqueeze(1) * &pos_act * 2.0;
        let d_neg = neg_p.unsqueeze(1) * &neg_act * 2.0;

        let batch = pos_input.size()[0] as f64;
        let inv_batch = 1.0 / batch;

        // dW = (d_pos^T @ pos_input - d_neg^T @ neg_input) / batch  → [out, in]
        let dw = (d_pos.tr().matmul(pos_input) - d_neg.tr().matmul(neg_input)) * inv_batch;
        // db = sum over batch → [out]
        let zdims = [0i64];
        let db = (d_pos.sum_dim_intlist(&zdims[..], false, Kind::Float)
            - d_neg.sum_dim_intlist(&zdims[..], false, Kind::Float))
            * inv_batch;

        // STE: gradient written directly to SHADOW
        let new_shadow = shadow + dw * lr;
        let new_bias = bias + db * lr;

        (new_shadow, new_bias, pos_act, neg_act)
    }

    /// Evaluate accuracy using ternary (quantized) weights for every class embedding.
    fn eval_ternary_accuracy(
        shadows: &[Tensor],
        biases: &[Tensor],
        test_x: &Tensor,
        test_y: &Tensor,
        n_test: usize,
        n_classes: usize,
        image_dim: usize,
        device: Device,
    ) -> (f32, Vec<Vec<u32>>) {
        let n = n_test as i64;
        let nc = n_classes as i64;

        // Pre-quantize every layer once (avoid redoing per class)
        let scaled_layers: Vec<Tensor> = shadows
            .iter()
            .map(|s| {
                let (_t, scaled, _g) = absmean_ternarize(s);
                scaled.tr() // [in, out] for matmul
            })
            .collect();

        let _ = image_dim; // test_x is already [N, 784]
        // Build [n, n_classes] goodness matrix, then argmax.
        let mut per_class_goodness: Vec<Tensor> = Vec::with_capacity(n_classes);
        for c in 0..nc {
            let labels_c = Tensor::full(&[n], c, (Kind::Int64, device));
            let onehot = one_hot(&labels_c, nc, device);
            let mut x = Tensor::cat(&[test_x.shallow_clone(), onehot], 1);

            let mut total_goodness = Tensor::zeros(&[n], (Kind::Float, device));
            for (layer_idx, w_t) in scaled_layers.iter().enumerate() {
                x = (x.matmul(w_t) + &biases[layer_idx]).relu();
                let dims = [1i64];
                let g = x
                    .pow_tensor_scalar(2.0)
                    .sum_dim_intlist(&dims[..], false, Kind::Float);
                total_goodness = &total_goodness + &g;
                x = normalize_gpu(&x);
            }
            per_class_goodness.push(total_goodness.unsqueeze(1)); // [n, 1]
        }
        let refs: Vec<&Tensor> = per_class_goodness.iter().collect();
        let goodness_matrix = Tensor::cat(&refs, 1); // [n, n_classes]
        let best_class = goodness_matrix.argmax(1, false); // [n], i64

        let correct = best_class
            .eq_tensor(test_y)
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]);
        let acc = (correct / n_test as f64) as f32;

        // Build confusion matrix on CPU
        let preds_cpu: Vec<i64> = {
            let cpu = best_class.to(Device::Cpu);
            let numel = cpu.numel();
            let mut v = vec![0i64; numel];
            cpu.f_copy_data(&mut v, numel).unwrap_or(());
            v
        };
        let actual_cpu: Vec<i64> = {
            let cpu = test_y.to(Device::Cpu);
            let numel = cpu.numel();
            let mut v = vec![0i64; numel];
            cpu.f_copy_data(&mut v, numel).unwrap_or(());
            v
        };
        let mut confusion = vec![vec![0u32; n_classes]; n_classes];
        for (p, a) in preds_cpu.iter().zip(actual_cpu.iter()) {
            let pi = (*p as usize).min(n_classes - 1);
            let ai = (*a as usize).min(n_classes - 1);
            confusion[ai][pi] += 1;
        }

        (acc, confusion)
    }

    /// Evaluate accuracy with f32 shadow weights (no quantization).
    fn eval_f32_accuracy(
        shadows: &[Tensor],
        biases: &[Tensor],
        test_x: &Tensor,
        test_y: &Tensor,
        n_test: usize,
        n_classes: usize,
        device: Device,
    ) -> f32 {
        let n = n_test as i64;
        let nc = n_classes as i64;

        let shadow_t: Vec<Tensor> = shadows.iter().map(|s| s.tr()).collect();

        let mut per_class: Vec<Tensor> = Vec::with_capacity(n_classes);
        for c in 0..nc {
            let labels_c = Tensor::full(&[n], c, (Kind::Int64, device));
            let onehot = one_hot(&labels_c, nc, device);
            let mut x = Tensor::cat(&[test_x.shallow_clone(), onehot], 1);
            let mut total_goodness = Tensor::zeros(&[n], (Kind::Float, device));
            for (layer_idx, w_t) in shadow_t.iter().enumerate() {
                x = (x.matmul(w_t) + &biases[layer_idx]).relu();
                let dims = [1i64];
                let g = x
                    .pow_tensor_scalar(2.0)
                    .sum_dim_intlist(&dims[..], false, Kind::Float);
                total_goodness = &total_goodness + &g;
                x = normalize_gpu(&x);
            }
            per_class.push(total_goodness.unsqueeze(1));
        }
        let refs: Vec<&Tensor> = per_class.iter().collect();
        let goodness_matrix = Tensor::cat(&refs, 1);
        let best_class = goodness_matrix.argmax(1, false);

        let correct = best_class
            .eq_tensor(test_y)
            .to_kind(Kind::Float)
            .sum(Kind::Float)
            .double_value(&[]);
        (correct / n_test as f64) as f32
    }

    pub fn run(
        images: &[f32],
        labels: &[u8],
        layer_sizes: &[usize],
        n_classes: usize,
        n_train: usize,
        n_test: usize,
        test_images: &[f32],
        test_labels: &[u8],
        epochs: usize,
        batch_size: usize,
        lr: f32,
        threshold: f32,
        mut progress_cb: impl FnMut(usize, f32, f64),
    ) -> Result<GpuFFResult, String> {
        if !tch::Cuda::is_available() {
            return Err(
                "No CUDA GPU available (tch::Cuda::is_available()==false). \
                 Run with LD_PRELOAD=<torch-lib>/libtorch_cuda.so or fix torch linkage."
                    .into(),
            );
        }
        let n_gpus = tch::Cuda::device_count();
        let device = if n_gpus >= 2 {
            Device::Cuda(1) // training GPU
        } else {
            Device::Cuda(0)
        };
        eprintln!(
            "[ff-gpu] Training on CUDA device {} ({} GPU(s) visible)",
            match device { Device::Cuda(i) => i as i64, _ => -1 },
            n_gpus
        );

        let image_dim: usize = layer_sizes[0].saturating_sub(n_classes);
        if image_dim == 0 {
            return Err("layer_sizes[0] must be > n_classes".into());
        }

        // Upload dataset to GPU once (full dataset resides on GPU).
        let train_x = Tensor::from_slice(&images[..n_train * image_dim])
            .reshape(&[n_train as i64, image_dim as i64])
            .to(device);
        let train_y_i64: Vec<i64> = labels[..n_train].iter().map(|&l| l as i64).collect();
        let train_y = Tensor::from_slice(&train_y_i64).to(device);

        let test_x = Tensor::from_slice(&test_images[..n_test * image_dim])
            .reshape(&[n_test as i64, image_dim as i64])
            .to(device);
        let test_y_i64: Vec<i64> = test_labels[..n_test].iter().map(|&l| l as i64).collect();
        let test_y = Tensor::from_slice(&test_y_i64).to(device);

        // Initialize shadow weights + biases (no autograd — we do STE manually)
        let mut shadows: Vec<Tensor> = Vec::with_capacity(layer_sizes.len() - 1);
        let mut biases: Vec<Tensor> = Vec::with_capacity(layer_sizes.len() - 1);
        let mut layer_shapes: Vec<(usize, usize)> = Vec::with_capacity(layer_sizes.len() - 1);
        for i in 0..layer_sizes.len() - 1 {
            let in_dim = layer_sizes[i] as i64;
            let out_dim = layer_sizes[i + 1] as i64;
            let scale = (2.0 / (in_dim + out_dim) as f64).sqrt();
            let w =
                Tensor::randn(&[out_dim, in_dim], (Kind::Float, device)) * scale;
            let b = Tensor::zeros(&[out_dim], (Kind::Float, device));
            shadows.push(w);
            biases.push(b);
            layer_shapes.push((layer_sizes[i], layer_sizes[i + 1]));
        }

        let nc_i = n_classes as i64;
        let bs_i = batch_size as i64;
        let n_batches = n_train / batch_size;
        let start = Instant::now();
        let lr64 = lr as f64;
        let th64 = threshold as f64;

        for epoch in 0..epochs {
            for batch_idx in 0..n_batches {
                let off = (batch_idx * batch_size) as i64;
                let batch_x = train_x.narrow(0, off, bs_i);
                let batch_y = train_y.narrow(0, off, bs_i);

                // One-hot: correct label
                let y_oh = one_hot(&batch_y, nc_i, device);
                let pos_input = Tensor::cat(&[batch_x.shallow_clone(), y_oh], 1);

                // Wrong label: (y + 1) mod n_classes
                let wrong_y = (&batch_y + 1).remainder(nc_i);
                let wrong_oh = one_hot(&wrong_y, nc_i, device);
                let neg_input = Tensor::cat(&[batch_x.shallow_clone(), wrong_oh], 1);

                let mut pos_cur = pos_input;
                let mut neg_cur = neg_input;
                for layer_idx in 0..shadows.len() {
                    let (new_w, new_b, pos_out, neg_out) = ff_step_qat_gpu(
                        &shadows[layer_idx],
                        &biases[layer_idx],
                        &pos_cur,
                        &neg_cur,
                        lr64,
                        th64,
                    );
                    shadows[layer_idx] = new_w;
                    biases[layer_idx] = new_b;
                    pos_cur = normalize_gpu(&pos_out);
                    neg_cur = normalize_gpu(&neg_out);
                }
            }

            // Epoch-end evaluation on ternary weights
            let (tern_acc, _cm) = tch::no_grad(|| {
                eval_ternary_accuracy(
                    &shadows, &biases, &test_x, &test_y, n_test, n_classes, image_dim, device,
                )
            });
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "[ff-gpu] epoch {}/{}  tern_acc={:.4}  elapsed={:.1}s",
                epoch + 1,
                epochs,
                tern_acc,
                elapsed
            );
            progress_cb(epoch + 1, tern_acc, elapsed);
        }

        // Final eval: both f32 and ternary, plus confusion matrix
        let f32_acc = tch::no_grad(|| {
            eval_f32_accuracy(&shadows, &biases, &test_x, &test_y, n_test, n_classes, device)
        });
        let (ternary_acc, confusion) = tch::no_grad(|| {
            eval_ternary_accuracy(
                &shadows, &biases, &test_x, &test_y, n_test, n_classes, image_dim, device,
            )
        });

        // Extract ternary weights to CPU for IGQK packing
        let mut all_ternary: Vec<f32> = Vec::new();
        let mut total_weights: usize = 0;
        for s in &shadows {
            let (ternary, _scaled, _g) = absmean_ternarize(s);
            let v = tensor_to_f32_vec(&ternary);
            total_weights += v.len();
            all_ternary.extend_from_slice(&v);
        }

        let total_time = start.elapsed().as_secs_f64();

        Ok(GpuFFResult {
            f32_accuracy: f32_acc,
            ternary_accuracy: ternary_acc,
            all_ternary_weights: all_ternary,
            total_weights,
            total_time_secs: total_time,
            confusion_matrix: confusion,
            layer_shapes,
        })
    }
}
