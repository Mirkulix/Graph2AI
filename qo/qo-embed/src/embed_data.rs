//! Utility to embed training data — run via `cargo test -p qo-embed embed_training_data`
#[cfg(test)]
mod tests {
    use crate::EmbeddingModel;
    use std::fs;
    
    #[test]
    #[ignore] // Run manually: cargo test -p qo-embed embed_training_data -- --ignored --nocapture
    fn embed_training_data() {
        let model = EmbeddingModel::load().expect("Model laden");
        println!("Modell geladen, {} Dimensionen", model.dimension());
        
        let data: serde_json::Value = serde_json::from_str(
            &fs::read_to_string("/tmp/qo_massive_train.json").expect("Daten laden")
        ).expect("JSON parse");
        
        let train = data["train"].as_array().unwrap();
        let test_data = data["test"].as_array().unwrap();
        
        // Limitiere auf 500 pro Kategorie für Speed
        let mut buckets: std::collections::HashMap<u8, Vec<&serde_json::Value>> = std::collections::HashMap::new();
        for item in train {
            let label = item[1].as_u64().unwrap() as u8;
            buckets.entry(label).or_default().push(item);
        }
        let mut train_subset: Vec<&serde_json::Value> = Vec::new();
        for (_, items) in &buckets {
            train_subset.extend(items.iter().take(500));
        }
        
        println!("Embedding {} train + {} test...", train_subset.len(), test_data.len().min(400));
        let start = std::time::Instant::now();
        
        let mut train_embedded: Vec<serde_json::Value> = Vec::new();
        for (i, item) in train_subset.iter().enumerate() {
            let text = item[0].as_str().unwrap();
            let label = item[1].as_u64().unwrap();
            let short: String = text.chars().take(200).collect();
            if let Ok(vec) = model.embed(&short) {
                train_embedded.push(serde_json::json!([vec, label]));
            }
            if (i+1) % 200 == 0 {
                println!("  Train: {}/{}", i+1, train_subset.len());
            }
        }
        
        let mut test_embedded: Vec<serde_json::Value> = Vec::new();
        let test_limit = test_data.len().min(400);
        for item in test_data.iter().take(test_limit) {
            let text = item[0].as_str().unwrap();
            let label = item[1].as_u64().unwrap();
            let short: String = text.chars().take(200).collect();
            if let Ok(vec) = model.embed(&short) {
                test_embedded.push(serde_json::json!([vec, label]));
            }
        }
        
        let elapsed = start.elapsed();
        println!("Embedded: {} train, {} test in {:.1}s ({:.1} sentences/s)", 
            train_embedded.len(), test_embedded.len(), elapsed.as_secs_f64(),
            (train_embedded.len() + test_embedded.len()) as f64 / elapsed.as_secs_f64());
        
        let output = serde_json::json!({
            "train": train_embedded,
            "test": test_embedded,
            "dim": 384,
        });
        fs::write("/tmp/qo_rust_embedded.json", output.to_string()).expect("Speichern");
        println!("Gespeichert: /tmp/qo_rust_embedded.json");
    }
}
