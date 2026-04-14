//! Convolution and causal attention operations for QLANG.
//!
//! Provides:
//! - 2D convolution (Conv2D)
//! - 2D max pooling
//! - Causal attention masking (for GPT-style autoregressive models)
//! - Masked scaled dot-product attention

use crate::autograd::Tape;

/// 2D Convolution.
///
/// Input:  `[batch, channels_in, height, width]`  (flattened row-major)
/// Kernel: `[channels_out, channels_in, kH, kW]`  (flattened row-major)
/// Output: `[batch, channels_out, out_h, out_w]`
///
/// `out_h = (height + 2*padding - kH) / stride + 1`
/// `out_w = (width  + 2*padding - kW) / stride + 1`
pub fn conv2d(
    input: &[f32],
    input_shape: [usize; 4], // [batch, c_in, h, w]
    kernel: &[f32],
    kernel_shape: [usize; 4], // [c_out, c_in, kh, kw]
    stride: usize,
    padding: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [batch, c_in, h, w] = input_shape;
    let [c_out, kc_in, kh, kw] = kernel_shape;
    assert_eq!(c_in, kc_in, "Input channels must match kernel channels_in");
    assert!(stride > 0, "Stride must be positive");

    let out_h = (h + 2 * padding - kh) / stride + 1;
    let out_w = (w + 2 * padding - kw) / stride + 1;
    let out_shape = [batch, c_out, out_h, out_w];
    let mut output = vec![0.0f32; batch * c_out * out_h * out_w];

    for b in 0..batch {
        for co in 0..c_out {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    for ci in 0..c_in {
                        for fh in 0..kh {
                            for fw in 0..kw {
                                let ih = oh * stride + fh;
                                let iw = ow * stride + fw;
                                // Account for padding
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let input_idx = ((b * c_in + ci) * h + ih) * w + iw;
                                    let kernel_idx = ((co * c_in + ci) * kh + fh) * kw + fw;
                                    sum += input[input_idx] * kernel[kernel_idx];
                                }
                            }
                        }
                    }
                    let out_idx = ((b * c_out + co) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }

    (output, out_shape)
}

/// 2D Max Pooling.
///
/// Input:  `[batch, channels, height, width]` (flattened row-major)
/// Output: `[batch, channels, out_h, out_w]`
///
/// `out_h = (height - pool_size) / stride + 1`
/// `out_w = (width  - pool_size) / stride + 1`
pub fn max_pool2d(
    input: &[f32],
    input_shape: [usize; 4], // [batch, channels, h, w]
    pool_size: usize,
    stride: usize,
) -> (Vec<f32>, [usize; 4]) {
    let [batch, channels, h, w] = input_shape;
    assert!(pool_size > 0 && stride > 0);
    assert!(pool_size <= h && pool_size <= w, "Pool size exceeds spatial dimensions");

    let out_h = (h - pool_size) / stride + 1;
    let out_w = (w - pool_size) / stride + 1;
    let out_shape = [batch, channels, out_h, out_w];
    let mut output = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;
                            let idx = ((b * channels + c) * h + ih) * w + iw;
                            max_val = max_val.max(input[idx]);
                        }
                    }
                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    output[out_idx] = max_val;
                }
            }
        }
    }

    (output, out_shape)
}

/// Generate a causal (upper-triangular) attention mask.
///
/// Returns a `seq_len x seq_len` mask where:
/// - `mask[i][j] = 0.0`        if `j <= i` (allowed to attend)
/// - `mask[i][j] = -infinity`   if `j > i`  (blocked / future token)
///
/// This is added to attention scores before softmax so that future
/// positions receive zero probability.
pub fn causal_attention_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Masked scaled dot-product attention:
///   `softmax(Q * K^T / sqrt(d_k) + mask) * V`
///
/// Q: `[seq_q, d_k]`
/// K: `[seq_k, d_k]`
/// V: `[seq_k, d_v]`
/// mask: `[seq_q, seq_k]` (e.g. from `causal_attention_mask`)
///
/// Output: `[seq_q, d_v]`
pub fn masked_attention(
    tape: &mut Tape,
    q: usize,
    k: usize,
    v: usize,
    mask: &[f32],
    d_k: usize,
) -> usize {
    let q_data = tape.value(q).to_vec();
    let k_data = tape.value(k).to_vec();
    let v_data = tape.value(v).to_vec();

    let seq_q = tape.values[q].shape[0];
    let seq_k = tape.values[k].shape[0];
    let d_v = tape.values[v].shape[1];

    assert_eq!(mask.len(), seq_q * seq_k, "Mask shape must be [seq_q, seq_k]");

    // Compute Q @ K^T / sqrt(d_k) + mask
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut scores = vec![0.0f32; seq_q * seq_k];

    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for d in 0..d_k {
                dot += q_data[i * d_k + d] * k_data[j * d_k + d];
            }
            scores[i * seq_k + j] = dot * scale + mask[i * seq_k + j];
        }
    }

    // Softmax over seq_k dimension
    for i in 0..seq_q {
        let offset = i * seq_k;
        let max = scores[offset..offset + seq_k]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..seq_k {
            scores[offset + j] = (scores[offset + j] - max).exp();
            sum += scores[offset + j];
        }
        if sum > 0.0 {
            for j in 0..seq_k {
                scores[offset + j] /= sum;
            }
        }
    }

    // scores @ V
    let mut output = vec![0.0f32; seq_q * d_v];
    for i in 0..seq_q {
        for j in 0..d_v {
            let mut sum = 0.0f32;
            for k_idx in 0..seq_k {
                sum += scores[i * seq_k + k_idx] * v_data[k_idx * d_v + j];
            }
            output[i * d_v + j] = sum;
        }
    }

    tape.variable(output, vec![seq_q, d_v])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_known_output() {
        // 1 batch, 1 channel, 3x3 input, 1 output channel, 2x2 kernel, stride=1, padding=0
        let input = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let kernel = vec![
            1.0, 0.0,
            0.0, 1.0,
        ];
        // Expected output (2x2):
        //   (1*1 + 2*0 + 4*0 + 5*1) = 6
        //   (2*1 + 3*0 + 5*0 + 6*1) = 8
        //   (4*1 + 5*0 + 7*0 + 8*1) = 12
        //   (5*1 + 6*0 + 8*0 + 9*1) = 14
        let (output, shape) = conv2d(
            &input,
            [1, 1, 3, 3],
            &kernel,
            [1, 1, 2, 2],
            1,
            0,
        );
        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(output, vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // 1 batch, 1 channel, 2x2 input, 1 output channel, 3x3 kernel, stride=1, padding=1
        // With padding=1, output should be same size as input (2x2)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![
            0.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.0,
        ]; // Identity kernel
        let (output, shape) = conv2d(
            &input,
            [1, 1, 2, 2],
            &kernel,
            [1, 1, 3, 3],
            1,
            1,
        );
        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv2d_stride() {
        // 1 batch, 1 channel, 4x4 input, stride=2
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let kernel = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 all-ones kernel
        let (output, shape) = conv2d(
            &input,
            [1, 1, 4, 4],
            &kernel,
            [1, 1, 2, 2],
            2,
            0,
        );
        // out_h = (4 - 2)/2 + 1 = 2, out_w = 2
        assert_eq!(shape, [1, 1, 2, 2]);
        // top-left: 1+2+5+6=14, top-right: 3+4+7+8=22
        // bot-left: 9+10+13+14=46, bot-right: 11+12+15+16=54
        assert_eq!(output, vec![14.0, 22.0, 46.0, 54.0]);
    }

    #[test]
    fn test_conv2d_multi_channel() {
        // 1 batch, 2 input channels, 2x2 input, 1 output channel
        let input = vec![
            // channel 0
            1.0, 2.0,
            3.0, 4.0,
            // channel 1
            5.0, 6.0,
            7.0, 8.0,
        ];
        // kernel: [1, 2, 1, 1] -- single 1x1 kernel across 2 input channels
        let kernel = vec![1.0, 1.0];
        let (output, shape) = conv2d(
            &input,
            [1, 2, 2, 2],
            &kernel,
            [1, 2, 1, 1],
            1,
            0,
        );
        assert_eq!(shape, [1, 1, 2, 2]);
        // Each output pixel = sum of corresponding pixels across channels
        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_max_pool2d_basic() {
        // 1 batch, 1 channel, 4x4 input, pool_size=2, stride=2
        let input = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let (output, shape) = max_pool2d(&input, [1, 1, 4, 4], 2, 2);
        assert_eq!(shape, [1, 1, 2, 2]);
        assert_eq!(output, vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_reduces_dimensions() {
        // 2 batches, 1 channel, 6x6, pool 3, stride 3 -> 2x2
        let input = vec![0.0f32; 2 * 1 * 6 * 6];
        let (_, shape) = max_pool2d(&input, [2, 1, 6, 6], 3, 3);
        assert_eq!(shape, [2, 1, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_overlapping() {
        // pool_size=2, stride=1 on 3x3 -> 2x2
        let input = vec![
            9.0, 1.0, 2.0,
            3.0, 5.0, 4.0,
            7.0, 8.0, 6.0,
        ];
        let (output, shape) = max_pool2d(&input, [1, 1, 3, 3], 2, 1);
        assert_eq!(shape, [1, 1, 2, 2]);
        // [max(9,1,3,5), max(1,2,5,4), max(3,5,7,8), max(5,4,8,6)]
        assert_eq!(output, vec![9.0, 5.0, 8.0, 8.0]);
    }

    #[test]
    fn test_causal_mask_upper_triangular() {
        let mask = causal_attention_mask(4);
        assert_eq!(mask.len(), 16);

        // Diagonal and below should be 0.0
        for i in 0..4 {
            for j in 0..=i {
                assert_eq!(mask[i * 4 + j], 0.0, "mask[{i}][{j}] should be 0.0");
            }
        }

        // Above diagonal should be -inf
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert!(
                    mask[i * 4 + j] == f32::NEG_INFINITY,
                    "mask[{i}][{j}] should be -inf"
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_size_1() {
        let mask = causal_attention_mask(1);
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn test_masked_attention_prevents_future() {
        let mut tape = Tape::new();
        let seq_len = 4;
        let d_k = 2;

        // Q, K: uniform so raw attention would be uniform without mask
        let q = tape.variable(vec![1.0; seq_len * d_k], vec![seq_len, d_k]);
        let k = tape.variable(vec![1.0; seq_len * d_k], vec![seq_len, d_k]);
        // V: each row is distinct so we can verify which positions are attended to
        // Row i has all values = (i+1) as f32
        let v_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| vec![(i + 1) as f32; d_k])
            .collect();
        let v = tape.variable(v_data, vec![seq_len, d_k]);

        let mask = causal_attention_mask(seq_len);
        let out = masked_attention(&mut tape, q, k, v, &mask, d_k);
        let result = tape.value(out);

        // Position 0 can only attend to position 0 -> output should be V[0] = [1, 1]
        assert!(
            (result[0] - 1.0).abs() < 1e-5,
            "Position 0 should only attend to itself, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-5,
            "Position 0 should only attend to itself, got {}",
            result[1]
        );

        // Position 1 attends to positions 0 and 1 -> avg of V[0] and V[1] = 1.5
        assert!(
            (result[2] - 1.5).abs() < 1e-4,
            "Position 1 should attend to [0,1], got {}",
            result[2]
        );

        // Position 3 attends to all 4 -> avg = 2.5
        let last_row_avg = (result[(seq_len - 1) * d_k] + result[(seq_len - 1) * d_k + 1]) / 2.0;
        assert!(
            (last_row_avg - 2.5).abs() < 1e-4,
            "Last position should attend to all, got {}",
            last_row_avg
        );
    }

    #[test]
    fn test_masked_attention_no_mask_matches_uniform() {
        // With a zero mask, masked_attention should behave like standard attention
        let mut tape = Tape::new();
        let seq_len = 3;
        let d_k = 4;

        let q = tape.variable(vec![0.1; seq_len * d_k], vec![seq_len, d_k]);
        let k = tape.variable(vec![0.1; seq_len * d_k], vec![seq_len, d_k]);
        let v = tape.variable(vec![0.5; seq_len * d_k], vec![seq_len, d_k]);

        let zero_mask = vec![0.0f32; seq_len * seq_len];
        let out = masked_attention(&mut tape, q, k, v, &zero_mask, d_k);
        let result = tape.value(out);

        // Uniform Q/K -> uniform attention -> output = V values
        for &val in result {
            assert!((val - 0.5).abs() < 1e-5);
        }
    }
}
