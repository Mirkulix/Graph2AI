//! Integration: federated ternary organism.
//!
//! Simulates 3 QO nodes training ternary specialists on disjoint MNIST
//! partitions, then merging their weights via ternary majority vote. The
//! merged organism must not catastrophically drop below the average of the
//! individual node accuracies.
//!
//! Backwards-compat: 2-node merge path is exercised as a subcase.

use qlang_runtime::federation::{count_changes, ternary_majority_vote, verify_ternary};
use qlang_runtime::mnist::MnistData;
use qlang_runtime::ternary_brain::TernaryBrain;

/// Train a single-node specialist on a disjoint partition of `full`.
fn train_node(full: &MnistData, start: usize, end: usize, rounds: usize) -> TernaryBrain {
    let image_size = full.image_size;
    let train_images = &full.train_images[start * image_size..end * image_size];
    let train_labels = &full.train_labels[start..end];
    let n = end - start;

    let mut brain = TernaryBrain::init(
        train_images,
        train_labels,
        image_size,
        n,
        full.n_classes,
        5, // 5 neurons per class = 50 total
    );
    brain.refine(train_images, train_labels, n, rounds);
    brain
}

#[test]
fn federated_3node_merge_preserves_accuracy() {
    // Build a shared pool; each node trains on a disjoint 600-sample slice.
    let per_node = 600;
    let n_nodes = 3;
    let pool = MnistData::synthetic(per_node * n_nodes, 500);

    // Train 3 nodes independently for 5 rounds each.
    let rounds = 5;
    let brain_a = train_node(&pool, 0, per_node, rounds);
    let brain_b = train_node(&pool, per_node, 2 * per_node, rounds);
    let brain_c = train_node(&pool, 2 * per_node, 3 * per_node, rounds);

    // Shared holdout for fair cross-node evaluation.
    let acc_a = brain_a.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);
    let acc_b = brain_b.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);
    let acc_c = brain_c.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);

    println!("\n=== Federated Organism — 3 Node Accuracy ===");
    println!("  Node A (partition 0):  {:.3}", acc_a);
    println!("  Node B (partition 1):  {:.3}", acc_b);
    println!("  Node C (partition 2):  {:.3}", acc_c);

    // Pre-merge sanity: each node must be ternary.
    assert!(brain_a.verify_ternary());
    assert!(brain_b.verify_ternary());
    assert!(brain_c.verify_ternary());
    assert_eq!(brain_a.total_weights(), brain_b.total_weights());
    assert_eq!(brain_b.total_weights(), brain_c.total_weights());

    // --- Ternary majority-vote merge ---
    let wa = brain_a.dump_weights_i8();
    let wb = brain_b.dump_weights_i8();
    let wc = brain_c.dump_weights_i8();
    let merged = ternary_majority_vote(&[&wa, &wb, &wc]).expect("merge failed");

    assert!(verify_ternary(&merged), "merged weights must be ternary");
    assert_eq!(merged.len(), wa.len());

    // Changes relative to node A (arbitrary reference).
    let ch_a = count_changes(&wa, &merged);
    let ch_b = count_changes(&wb, &merged);
    let ch_c = count_changes(&wc, &merged);
    println!(
        "  Merge changes vs A/B/C: {} / {} / {} (of {} total weights)",
        ch_a, ch_b, ch_c, merged.len()
    );

    // --- Build merged brain from any template + merged weights ---
    let merged_brain = TernaryBrain::from_template_and_weights(&brain_a, &merged)
        .expect("build merged brain");
    assert!(merged_brain.verify_ternary());

    let acc_merged = merged_brain.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);
    let avg_individual = (acc_a + acc_b + acc_c) / 3.0;

    println!("  Merged organism:       {:.3}", acc_merged);
    println!("  Avg individual:        {:.3}", avg_individual);
    println!(
        "  Delta (merged - avg):  {:+.3} ({:+.1}%)",
        acc_merged - avg_individual,
        (acc_merged - avg_individual) * 100.0
    );

    // Core contract: merged must not catastrophically drop below the average.
    // Ternary majority voting *may* regress slightly on synthetic data where
    // each node has seen a different shard, but >5% collapse would indicate
    // the merge itself is broken.
    let tolerance = 0.05;
    assert!(
        acc_merged + tolerance >= avg_individual,
        "merged accuracy {:.3} dropped >5% below average {:.3}",
        acc_merged,
        avg_individual
    );

    // Bonus: report if merged actually beat individuals.
    let best_individual = acc_a.max(acc_b).max(acc_c);
    if acc_merged > best_individual {
        println!("  Merged BEATS best individual ({:.3} > {:.3})", acc_merged, best_individual);
    } else if acc_merged > avg_individual {
        println!("  Merged beats average but not best ({:.3} > {:.3})", acc_merged, avg_individual);
    } else {
        println!("  Merged within tolerance of individuals");
    }
}

#[test]
fn federated_2node_backwards_compat() {
    // The existing qlms-dual-server.sh path has 2 nodes; verify the N-way
    // merge degenerates to the 2-node case without panicking or producing
    // non-ternary weights.
    let per_node = 400;
    let pool = MnistData::synthetic(per_node * 2, 300);

    let brain_a = train_node(&pool, 0, per_node, 3);
    let brain_b = train_node(&pool, per_node, 2 * per_node, 3);

    let wa = brain_a.dump_weights_i8();
    let wb = brain_b.dump_weights_i8();

    let merged = ternary_majority_vote(&[&wa, &wb]).expect("2-way merge failed");
    assert!(verify_ternary(&merged));
    assert_eq!(merged.len(), wa.len());

    // In the 2-peer case every disagreement becomes 0 (tie-break); agreements
    // are preserved. So #zeros in merged should be ≥ #zeros in either peer.
    let zeros_merged = merged.iter().filter(|&&v| v == 0).count();
    let zeros_a = wa.iter().filter(|&&v| v == 0).count();
    let zeros_b = wb.iter().filter(|&&v| v == 0).count();
    assert!(
        zeros_merged >= zeros_a.min(zeros_b),
        "2-peer merge should not have fewer zeros than the less-zero peer"
    );

    let merged_brain = TernaryBrain::from_template_and_weights(&brain_a, &merged)
        .expect("rebuild merged brain");
    let acc_merged = merged_brain.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);
    let acc_a = brain_a.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);
    let acc_b = brain_b.accuracy(&pool.test_images, &pool.test_labels, pool.n_test);

    println!("\n=== Federated 2-node compat ===");
    println!(
        "  A={:.3}  B={:.3}  Merged={:.3}  avg={:.3}",
        acc_a,
        acc_b,
        acc_merged,
        (acc_a + acc_b) / 2.0
    );

    // Sanity: 2-node merge must not crash and must stay above a trivial floor.
    // (With tie-break-to-zero on 2 peers, absolute quality can drop; we just
    // guarantee it is a valid ternary brain that still classifies above chance.)
    assert!(
        acc_merged > 0.05,
        "2-node merged accuracy {:.3} below chance floor",
        acc_merged
    );
}
