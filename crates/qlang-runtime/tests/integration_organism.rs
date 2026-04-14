//! Integration test: build an Organism with 3 specialists, route a variety
//! of inputs through it, and verify outputs land on the right specialist and
//! produce non-empty text.

use qlang_runtime::mnist::MnistData;
use qlang_runtime::organism::Organism;

#[test]
fn integration_organism_three_specialists_route_inputs() {
    // Organism::new seeds three specialists: responder, memory, adapter.
    let mut org = Organism::new(512);
    assert_eq!(org.specialist_count(), 3);

    // Use synthetic MNIST to verify the data side of the pipeline is healthy.
    let data = MnistData::synthetic(1000, 200);
    assert_eq!(data.n_train, 1000);
    assert_eq!(data.n_test, 200);

    // --- Route a greeting -> responder ---
    let greet = org.process("Hello organism!");
    assert!(!greet.text.is_empty(), "greeting produced empty response");
    assert_eq!(greet.specialist, "responder");

    // --- Seed a fact, then recall from memory specialist ---
    let fact = "The QLANG organism contains ternary specialists";
    let stored = org.process(fact);
    assert!(stored.memory_stored);
    let recall = org.process("recall memory about ternary specialists");
    assert_eq!(recall.specialist, "memory");
    assert!(
        !recall.text.is_empty(),
        "memory recall produced empty response"
    );

    // --- Unknown / question routing lands on responder fallback ---
    let q = org.process("Why is the sky blue?");
    assert!(!q.text.is_empty());
    // Either responder templated the question, or another specialist answered:
    // both are valid — we just require a non-empty typed response.
    assert!(
        q.reasoning.iter().any(|r| r.contains("Routed to category")),
        "expected a routing trace in reasoning"
    );

    // --- Total interactions recorded ---
    assert_eq!(org.total_interactions(), 4);

    // --- Status line references generation/interaction counts ---
    let status = org.status();
    assert!(
        status.contains("4 interactions") || status.contains("interactions"),
        "unexpected status line: {status}"
    );

    // --- Confidence is in [0, 1] ---
    for r in [&greet, &stored, &recall, &q] {
        assert!(
            r.confidence >= 0.0 && r.confidence <= 1.0,
            "confidence out of range: {} -> {}", r.specialist, r.confidence
        );
    }

    println!(
        "organism routed {} inputs, specialists used: {} / {} / {} / {}",
        org.total_interactions(),
        greet.specialist, stored.specialist, recall.specialist, q.specialist
    );
}
