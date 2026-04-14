//! Spiking MNIST integration test.
//!
//! Verifies that the spiking network beats random (10%) on a real
//! MNIST subset or synthetic fallback via supervised STDP training.

use qlang_runtime::mnist::MnistData;
use qlang_runtime::spiking::SpikingNetwork;

#[test]
fn spiking_mnist_beats_random() {
    let data = MnistData::load_from_dir("data/mnist")
        .or_else(|_| MnistData::load_from_dir("../../data/mnist"))
        .unwrap_or_else(|_| MnistData::synthetic(2000, 500));

    // Clip dataset sizes to what's available.
    let n_train = 2000.min(data.train_labels.len());
    let n_test = 500.min(data.test_labels.len());
    let input_dim = 784;

    let mut net = SpikingNetwork::new(&[input_dim, 128, 10]);

    net.train_stdp(
        &data.train_images[..n_train * input_dim],
        &data.train_labels[..n_train],
        n_train,
        input_dim,
        100,
        5,
    );

    let acc = net.accuracy(
        &data.test_images[..n_test * input_dim],
        &data.test_labels[..n_test],
        n_test,
        input_dim,
        150,
    );

    println!("Spiking MNIST: {:.1}% ({} train / {} test)", acc * 100.0, n_train, n_test);
    assert!(
        acc > 0.35,
        "Should beat random 10% significantly (got {:.1}%)",
        acc * 100.0
    );
}
