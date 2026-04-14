//! Full System Test: All QLANG modules working together.
//! This is the honest benchmark of everything we built.

use qlang_runtime::ternary_brain::TernaryBrain;
use qlang_runtime::forward_forward::FFNetwork;
use qlang_runtime::noprop::NoPropNet;
use qlang_runtime::hdc::{HdVector, HdMemory};
use qlang_runtime::ttt::TttNetwork;
use qlang_runtime::mamba::MambaModel;
use qlang_runtime::mnist::MnistData;
use std::time::Instant;

#[test]
fn full_system_benchmark() {
    let data = MnistData::synthetic(3000, 1000);

    println!("\n{}", "=".repeat(70));
    println!("QLANG FULL SYSTEM TEST — All Modules on MNIST");
    println!("{}\n", "=".repeat(70));

    // 1. TernaryBrain
    let start = Instant::now();
    let brain = TernaryBrain::init(
        &data.train_images, &data.train_labels, 784, data.n_train, 10, 20,
    );
    let brain_acc = brain.accuracy(&data.test_images, &data.test_labels, data.n_test);
    let brain_time = start.elapsed();
    println!("1. TernaryBrain:    {:.1}%  {:?}  (statistical init, zero gradients)", brain_acc * 100.0, brain_time);

    // 2. NoProp (5 steps, 10 epochs)
    let start = Instant::now();
    let mut noprop = NoPropNet::new(784, 10, 10, 5);
    for _ in 0..10 {
        noprop.train_epoch(&data.train_images, &data.train_labels, data.n_train, 50);
    }
    let noprop_f32 = noprop.accuracy(&data.test_images, &data.test_labels, data.n_test, false);
    let noprop_tern = noprop.accuracy(&data.test_images, &data.test_labels, data.n_test, true);
    let noprop_time = start.elapsed();
    println!("2. NoProp:          f32={:.1}% tern={:.1}%  {:?}  (denoising, no backprop)", noprop_f32 * 100.0, noprop_tern * 100.0, noprop_time);

    // 3. Forward-Forward (2 layers, 10 epochs)
    let start = Instant::now();
    let mut ff = FFNetwork::new(&[794, 128, 64], 10);
    for _ in 0..10 {
        ff.train_epoch(&data.train_images, &data.train_labels, 784, data.n_train, 50);
    }
    let ff_f32 = ff.accuracy(&data.test_images, &data.test_labels, 784, data.n_test);
    let ff_tern = ff.accuracy_ternary(&data.test_images, &data.test_labels, 784, data.n_test);
    let ff_time = start.elapsed();
    println!("3. Forward-Forward: f32={:.1}% tern={:.1}%  {:?}  (layer-local, no backprop)", ff_f32 * 100.0, ff_tern * 100.0, ff_time);

    // 4. Mamba forward (architecture test, not trained)
    let start = Instant::now();
    let mamba = MambaModel::new(32, 64, 16, 2);
    let mamba_input = vec![0.1f32; 8 * 32]; // 8 timesteps, 32 dims
    let mamba_out = mamba.forward(&mamba_input, 8);
    let mamba_time = start.elapsed();
    let mamba_finite = mamba_out.iter().all(|x| x.is_finite());
    println!("4. Mamba SSM:       {} params, output finite: {}  {:?}", mamba.param_count(), mamba_finite, mamba_time);

    // 5. HDC
    let start = Instant::now();
    let dim = 784;
    let mut mem = HdMemory::new(dim);
    // Encode digits 0-9 as HD vectors
    for digit in 0..10u8 {
        let mut digit_vec = HdVector::zero(dim);
        let mut count = 0;
        for i in 0..data.n_train.min(500) {
            if data.train_labels[i] == digit {
                let feat = HdVector::from_f32(&data.train_images[i * 784..(i + 1) * 784]);
                digit_vec = HdVector::bundle(&[&digit_vec, &feat]);
                count += 1;
            }
        }
        mem.store(&format!("{}", digit), digit_vec);
    }
    // Test HDC classification
    let mut hdc_correct = 0;
    let hdc_test_n = data.n_test.min(200);
    for i in 0..hdc_test_n {
        let query = HdVector::from_f32(&data.test_images[i * 784..(i + 1) * 784]);
        if let Some((name, _)) = mem.query(&query) {
            if let Ok(predicted) = name.parse::<u8>() {
                if predicted == data.test_labels[i] { hdc_correct += 1; }
            }
        }
    }
    let hdc_acc = hdc_correct as f32 / hdc_test_n as f32;
    let hdc_time = start.elapsed();
    println!("5. HDC (784-dim):   {:.1}%  {:?}  (ternary vectors, no training)", hdc_acc * 100.0, hdc_time);

    // 6. TTT
    let start = Instant::now();
    let mut ttt = TttNetwork::new(&[784, 128, 64, 10]);
    // Test: standard vs TTT-adapted inference
    let sample = &data.test_images[..784];
    let out_standard = ttt.forward(sample);
    let out_ttt = ttt.forward_with_ttt(sample);
    let ttt_diff: f32 = out_standard.iter().zip(out_ttt.iter()).map(|(a, b)| (a - b).abs()).sum();
    let ttt_time = start.elapsed();
    println!("6. TTT:             adaptation diff={:.4}  {:?}  (self-supervised at inference)", ttt_diff, ttt_time);

    // Summary
    println!("\n{}", "=".repeat(70));
    println!("SUMMARY");
    println!("{}", "=".repeat(70));
    println!("{:<25} {:>10} {:>10} {:>10}", "Method", "f32 Acc", "Tern Acc", "Time");
    println!("{}", "-".repeat(55));
    println!("{:<25} {:>10} {:>9.1}% {:>10?}", "TernaryBrain", "-", brain_acc * 100.0, brain_time);
    println!("{:<25} {:>9.1}% {:>9.1}% {:>10?}", "NoProp", noprop_f32 * 100.0, noprop_tern * 100.0, noprop_time);
    println!("{:<25} {:>9.1}% {:>9.1}% {:>10?}", "Forward-Forward", ff_f32 * 100.0, ff_tern * 100.0, ff_time);
    println!("{:<25} {:>10} {:>9.1}% {:>10?}", "HDC", "-", hdc_acc * 100.0, hdc_time);
    println!("{:<25} {:>10} {:>10} {:>10?}", "Mamba", format!("{}p", mamba.param_count()), if mamba_finite {"ok"} else {"NaN"}, mamba_time);
    println!("{:<25} {:>10} {:>10} {:>10?}", "TTT", "-", format!("d={:.3}", ttt_diff), ttt_time);
    println!();

    // Assertions
    assert!(brain_acc > 0.60, "TernaryBrain must >60%");
    assert!(noprop_f32 > 0.70, "NoProp f32 must >70%");
    assert!(ff_f32 > 0.15, "FF f32 must >15%");
    assert!(hdc_acc > 0.10, "HDC must beat random");
    assert!(mamba_finite, "Mamba must produce finite output");
    assert!(ttt_diff > 0.0001, "TTT must change output");
}
