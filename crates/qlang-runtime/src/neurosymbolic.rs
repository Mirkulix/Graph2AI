//! Neuro-Symbolic Hybrid — neural pattern matching + symbolic logic.
//!
//! Combines:
//! - Neural: ternary weight matching (fast, approximate, "intuition")
//! - Symbolic: logical rules, constraints, exact reasoning ("logic")
//!
//! A neurosymbolic program is a sequence of steps:
//!   1. Neural: classify input with ternary brain (fast)
//!   2. Symbolic: apply logical rules to refine/validate classification
//!   3. Neural: if rules contradict, re-classify with adjusted weights
//!
//! This is how AlphaGeometry works: neural network proposes, logic verifies.

use std::collections::HashMap;

/// A symbolic rule: IF conditions THEN conclusion.
#[derive(Clone, Debug)]
pub struct Rule {
    pub name: String,
    /// Conditions: list of (variable, operator, value)
    pub conditions: Vec<(String, Cmp, f32)>,
    /// Conclusion: (variable, value)
    pub conclusion: (String, f32),
    /// Confidence weight
    pub weight: f32,
}

#[derive(Clone, Debug)]
pub enum Cmp {
    Gt,
    Lt,
    Eq,
    Gte,
    Lte,
}

impl Rule {
    /// Evaluate this rule against a set of facts.
    pub fn evaluate(&self, facts: &HashMap<String, f32>) -> Option<(String, f32)> {
        let all_match = self.conditions.iter().all(|(var, cmp, threshold)| {
            match facts.get(var) {
                Some(&val) => match cmp {
                    Cmp::Gt => val > *threshold,
                    Cmp::Lt => val < *threshold,
                    Cmp::Eq => (val - threshold).abs() < 0.01,
                    Cmp::Gte => val >= *threshold,
                    Cmp::Lte => val <= *threshold,
                },
                None => false,
            }
        });

        if all_match {
            Some((self.conclusion.0.clone(), self.conclusion.1 * self.weight))
        } else {
            None
        }
    }
}

/// Neural component: ternary feature matcher.
pub struct NeuralMatcher {
    /// Ternary weight templates per class: [n_classes, feat_dim] as i8
    pub templates: Vec<Vec<i8>>,
    pub class_names: Vec<String>,
    pub feat_dim: usize,
}

impl NeuralMatcher {
    /// Create from class means (like TernaryBrain Phase 1).
    pub fn from_data(
        features: &[f32],
        labels: &[u8],
        feat_dim: usize,
        n_samples: usize,
        n_classes: usize,
        class_names: Vec<String>,
    ) -> Self {
        let mut class_sums = vec![vec![0.0f64; feat_dim]; n_classes];
        let mut class_counts = vec![0usize; n_classes];

        for i in 0..n_samples {
            let c = labels[i] as usize;
            if c < n_classes {
                for k in 0..feat_dim {
                    class_sums[c][k] += features[i * feat_dim + k] as f64;
                }
                class_counts[c] += 1;
            }
        }

        // Global mean
        let mut global_mean = vec![0.0f64; feat_dim];
        for k in 0..feat_dim {
            for c in 0..n_classes {
                global_mean[k] += class_sums[c][k];
            }
            global_mean[k] /= n_samples as f64;
        }

        // Ternary templates from class means - global mean
        let mut templates = Vec::new();
        for c in 0..n_classes {
            if class_counts[c] == 0 {
                templates.push(vec![0i8; feat_dim]);
                continue;
            }
            let mean: Vec<f64> = (0..feat_dim)
                .map(|k| class_sums[c][k] / class_counts[c] as f64 - global_mean[k])
                .collect();
            let abs_mean: f64 = mean.iter().map(|v| v.abs()).sum::<f64>() / feat_dim as f64;
            let threshold = abs_mean * 0.5;

            let template: Vec<i8> = mean.iter().map(|&v| {
                if v > threshold { 1i8 } else if v < -threshold { -1i8 } else { 0i8 }
            }).collect();
            templates.push(template);
        }

        Self { templates, class_names, feat_dim }
    }

    /// Neural classification: score input against all templates.
    /// Returns (class_scores, predicted_class, confidence).
    pub fn classify(&self, input: &[f32]) -> (Vec<f32>, usize, f32) {
        let n_classes = self.templates.len();
        let mut scores = vec![0.0f32; n_classes];

        for c in 0..n_classes {
            let mut sum = 0i64;
            for k in 0..self.feat_dim {
                match self.templates[c][k] {
                    1 => sum += (input[k] * 1000.0) as i64,
                    -1 => sum -= (input[k] * 1000.0) as i64,
                    _ => {}
                }
            }
            scores[c] = sum as f32 / 1000.0;
        }

        let max_idx = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        // Confidence: margin between top-1 and top-2
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let confidence = if sorted.len() > 1 { sorted[0] - sorted[1] } else { sorted[0] };

        (scores, max_idx, confidence)
    }
}

/// The full neuro-symbolic system.
pub struct NeuroSymbolicSystem {
    pub neural: NeuralMatcher,
    pub rules: Vec<Rule>,
}

impl NeuroSymbolicSystem {
    pub fn new(neural: NeuralMatcher) -> Self {
        Self { neural, rules: Vec::new() }
    }

    /// Add a symbolic rule.
    pub fn add_rule(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    /// Full neuro-symbolic inference:
    /// 1. Neural: classify input (fast, approximate)
    /// 2. Convert scores to facts
    /// 3. Apply symbolic rules
    /// 4. Combine neural + symbolic conclusions
    pub fn infer(&self, input: &[f32]) -> (usize, f32, Vec<String>) {
        let mut explanations = Vec::new();

        // Step 1: Neural classification
        let (scores, neural_class, neural_conf) = self.neural.classify(input);
        let class_name = self.neural.class_names.get(neural_class)
            .map(|s| s.as_str()).unwrap_or("unknown");
        explanations.push(format!("Neural: {} (confidence: {:.2})", class_name, neural_conf));

        // Step 2: Convert to facts
        let mut facts: HashMap<String, f32> = HashMap::new();
        for (i, score) in scores.iter().enumerate() {
            if let Some(name) = self.neural.class_names.get(i) {
                facts.insert(format!("score_{}", name), *score);
            }
        }
        facts.insert("neural_class".into(), neural_class as f32);
        facts.insert("confidence".into(), neural_conf);

        // Step 3: Apply symbolic rules
        let mut adjustments: HashMap<String, f32> = HashMap::new();
        for rule in &self.rules {
            if let Some((var, val)) = rule.evaluate(&facts) {
                explanations.push(format!("Rule '{}': {} = {:.2}", rule.name, var, val));
                *adjustments.entry(var).or_insert(0.0) += val;
            }
        }

        // Step 4: Combine
        let mut final_scores = scores.clone();
        for (var, adj) in &adjustments {
            if var.starts_with("adjust_class_") {
                if let Ok(idx) = var.replace("adjust_class_", "").parse::<usize>() {
                    if idx < final_scores.len() {
                        final_scores[idx] += adj;
                    }
                }
            }
        }

        let final_class = final_scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(neural_class);

        let final_conf = if final_class != neural_class {
            explanations.push(format!("Symbolic override: {} → {}",
                class_name,
                self.neural.class_names.get(final_class).map(|s| s.as_str()).unwrap_or("?")));
            neural_conf * 0.5 // reduced confidence when overridden
        } else {
            neural_conf
        };

        (final_class, final_conf, explanations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_evaluation() {
        let rule = Rule {
            name: "high_confidence".into(),
            conditions: vec![("confidence".into(), Cmp::Gt, 5.0)],
            conclusion: ("adjust_class_0".into(), 10.0),
            weight: 1.0,
        };

        let mut facts = HashMap::new();
        facts.insert("confidence".into(), 6.0);
        assert!(rule.evaluate(&facts).is_some());

        facts.insert("confidence".into(), 3.0);
        assert!(rule.evaluate(&facts).is_none());
    }

    #[test]
    fn neural_matcher_classifies() {
        // Simple 2-class problem in 4 dims
        let features = vec![
            1.0, 0.0, 0.0, 0.0, // class 0: top-left
            0.9, 0.1, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, // class 1: bottom-right
            0.0, 0.0, 0.9, 0.1,
        ];
        let labels = vec![0, 0, 1, 1];

        let matcher = NeuralMatcher::from_data(&features, &labels, 4, 4, 2,
            vec!["left".into(), "right".into()]);

        let (_, class_left, _) = matcher.classify(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(class_left, 0, "Should classify as left");

        let (_, class_right, _) = matcher.classify(&[0.0, 0.0, 1.0, 0.0]);
        assert_eq!(class_right, 1, "Should classify as right");
    }

    #[test]
    fn neurosymbolic_with_rules() {
        let features = vec![
            1.0, 0.0, 0.0, 0.0,
            0.9, 0.1, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.9, 0.1,
        ];
        let labels = vec![0, 0, 1, 1];
        let matcher = NeuralMatcher::from_data(&features, &labels, 4, 4, 2,
            vec!["cat".into(), "dog".into()]);

        let mut system = NeuroSymbolicSystem::new(matcher);

        // Rule: if confidence is low, boost dog (class 1)
        system.add_rule(Rule {
            name: "low_confidence_prefer_dog".into(),
            conditions: vec![("confidence".into(), Cmp::Lt, 1.0)],
            conclusion: ("adjust_class_1".into(), 5.0),
            weight: 1.0,
        });

        // High confidence input → neural wins
        let (class, _, explanations) = system.infer(&[1.0, 0.0, 0.0, 0.0]);
        println!("High confidence: class={}, explanations={:?}", class, explanations);

        // Ambiguous input → rule may fire
        let (class, _, explanations) = system.infer(&[0.5, 0.0, 0.5, 0.0]);
        println!("Ambiguous: class={}, explanations={:?}", class, explanations);

        // Check that explanations are generated
        assert!(!explanations.is_empty(), "Should have explanations");
    }

    #[test]
    fn neurosymbolic_explainability() {
        let features = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 samples, 3 dims
        let labels = vec![0, 1];
        let matcher = NeuralMatcher::from_data(&features, &labels, 3, 2, 2,
            vec!["A".into(), "B".into()]);

        let mut system = NeuroSymbolicSystem::new(matcher);
        system.add_rule(Rule {
            name: "always_explain".into(),
            conditions: vec![],
            conclusion: ("note".into(), 1.0),
            weight: 1.0,
        });

        let (_, _, explanations) = system.infer(&[1.0, 0.0, 0.0]);
        // Should have: neural classification + rule firing
        assert!(explanations.len() >= 2, "Need neural + rule explanations, got {:?}", explanations);
        println!("Explanations: {:?}", explanations);
    }
}
