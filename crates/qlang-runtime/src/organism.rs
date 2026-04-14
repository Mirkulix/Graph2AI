//! QLANG Organism — a swarm of specialized ternary models that form one intelligence.
//!
//! Architecture:
//!   ┌─────────────────────────────────────────────────┐
//!   │                  Orchestrator                     │
//!   │  Decides which specialist handles each request    │
//!   └──────────┬──────────┬──────────┬────────────────┘
//!              │          │          │
//!   ┌──────────▼┐  ┌─────▼─────┐  ┌▼──────────┐
//!   │ Classifier │  │ Language  │  │  Memory   │
//!   │ (ternary)  │  │ (Mamba)   │  │  (HDC)    │
//!   └────────────┘  └───────────┘  └───────────┘
//!              │          │          │
//!   ═══════════╧══════════╧══════════╧═══════════
//!              Shared HDC Memory (10K-dim ternary)
//!
//! Each model is small, ternary, specialized.
//! Together they form an organism that can:
//! - Classify inputs (what kind of request?)
//! - Generate responses (language model)
//! - Remember context (HDC associative memory)
//! - Learn at runtime (TTT adaptation)
//! - Evolve (add/remove/retrain specialists)

use crate::hdc::{HdVector, HdMemory};
use crate::ttt::TttLayer;
use crate::ternary_brain::TernaryBrain;
use crate::neurosymbolic::{NeuralMatcher, NeuroSymbolicSystem, Rule};
use std::collections::HashMap;

/// A specialist in the organism — one small model with one job.
pub struct Specialist {
    pub name: String,
    pub role: SpecialistRole,
    /// How many times this specialist was called
    pub invocations: u64,
    /// Running success rate
    pub success_rate: f32,
    /// Optional pointer into `evolution::fitness::FitnessTracker` records.
    pub fitness_id: Option<String>,
}

pub enum SpecialistRole {
    /// Classifies input into categories
    Classifier {
        brain: TernaryBrain,
        class_names: Vec<String>,
    },
    /// Generates text/responses
    Responder {
        templates: HashMap<String, Vec<String>>,
    },
    /// Stores and retrieves memories
    Memory {
        hd_memory: HdMemory,
        dim: usize,
    },
    /// Adapts at runtime (TTT)
    Adapter {
        layer: TttLayer,
    },
    /// Applies logical rules
    Reasoner {
        system: NeuroSymbolicSystem,
    },
    /// Spiking neural network specialist (neuromorphic)
    Spiking {
        network: crate::spiking::SpikingNetwork,
        timesteps: usize,
        class_names: Vec<String>,
    },
    /// Trained Mamba language model for text generation
    LanguageModel {
        lm: crate::mamba_train::TrainableLM,
    },
}

/// The organism: orchestrator + specialists + shared memory.
pub struct Organism {
    pub specialists: Vec<Specialist>,
    /// Shared memory accessible by all specialists
    pub shared_memory: HdMemory,
    pub hd_dim: usize,
    /// Log of all interactions
    pub log: Vec<InteractionLog>,
    /// Generation counter (for evolution)
    pub generation: u32,
}

#[derive(Clone, Debug)]
pub struct InteractionLog {
    pub input: String,
    pub specialist_used: String,
    pub output: String,
    pub success: bool,
}

/// Response from the organism.
#[derive(Debug)]
pub struct OrganismResponse {
    pub text: String,
    pub specialist: String,
    pub confidence: f32,
    pub reasoning: Vec<String>,
    pub memory_stored: bool,
}

impl Organism {
    /// Create a new organism with default specialists.
    pub fn new(hd_dim: usize) -> Self {
        let shared_memory = HdMemory::new(hd_dim);

        // Create default responder with templates
        let mut templates: HashMap<String, Vec<String>> = HashMap::new();
        templates.insert("greeting".into(), vec![
            "Hello! I am the QLANG organism.".into(),
            "Hi there. How can I help?".into(),
            "Greetings. I consist of multiple specialized ternary models.".into(),
        ]);
        templates.insert("question".into(), vec![
            "That is an interesting question. Let me think...".into(),
            "I am processing your query through my specialist network.".into(),
        ]);
        templates.insert("unknown".into(), vec![
            "I don't have a specialist for that yet. I am still evolving.".into(),
            "My current specialists cannot handle this. I need to grow.".into(),
        ]);

        let responder = Specialist {
            name: "responder".into(),
            role: SpecialistRole::Responder { templates },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        };

        // Memory specialist
        let memory = Specialist {
            name: "memory".into(),
            role: SpecialistRole::Memory {
                hd_memory: HdMemory::new(hd_dim),
                dim: hd_dim,
            },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        };

        // Adapter specialist (TTT)
        let adapter = Specialist {
            name: "adapter".into(),
            role: SpecialistRole::Adapter {
                layer: TttLayer::new(hd_dim, hd_dim / 2, 0.37),
            },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        };

        Self {
            specialists: vec![responder, memory, adapter],
            shared_memory,
            hd_dim,
            log: Vec::new(),
            generation: 0,
        }
    }

    /// Add a classifier specialist trained on data.
    pub fn add_classifier(
        &mut self,
        name: &str,
        features: &[f32],
        labels: &[u8],
        feat_dim: usize,
        n_samples: usize,
        n_classes: usize,
        class_names: Vec<String>,
    ) {
        let brain = TernaryBrain::init(features, labels, feat_dim, n_samples, n_classes, 20);
        self.specialists.push(Specialist {
            name: name.to_string(),
            role: SpecialistRole::Classifier { brain, class_names },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        });
    }

    /// Add a spiking neural network specialist.
    pub fn add_spiking(
        &mut self,
        name: &str,
        layer_sizes: &[usize],
        timesteps: usize,
        class_names: Vec<String>,
    ) {
        let network = crate::spiking::SpikingNetwork::new(layer_sizes);
        self.specialists.push(Specialist {
            name: name.to_string(),
            role: SpecialistRole::Spiking { network, timesteps, class_names },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        });
    }

    /// Load a trained Mamba LM from a QLMB binary file and add as Language specialist.
    pub fn add_language_model(&mut self, path: &str, text_for_tokenizer: &str) -> Result<(), String> {
        let lm = crate::mamba_train::load_mamba_model(path, text_for_tokenizer)?;
        self.specialists.push(Specialist {
            name: "language".into(),
            role: SpecialistRole::LanguageModel { lm },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        });
        Ok(())
    }

    /// Add a rule-based reasoner.
    pub fn add_reasoner(&mut self, name: &str, matcher: NeuralMatcher, rules: Vec<Rule>) {
        let mut system = NeuroSymbolicSystem::new(matcher);
        for rule in rules { system.add_rule(rule); }
        self.specialists.push(Specialist {
            name: name.to_string(),
            role: SpecialistRole::Reasoner { system },
            invocations: 0,
            success_rate: 1.0,
            fitness_id: None,
        });
    }

    /// Process an input through the organism.
    pub fn process(&mut self, input: &str) -> OrganismResponse {
        let mut reasoning = Vec::new();

        // 1. Encode input as HD vector
        let input_words: Vec<&str> = input.split_whitespace().collect();
        let word_vectors: Vec<HdVector> = input_words.iter()
            .map(|w| HdVector::random(self.hd_dim, hash_str(w)))
            .collect();
        let input_vec = if word_vectors.is_empty() {
            HdVector::random(self.hd_dim, 0)
        } else {
            let refs: Vec<&HdVector> = word_vectors.iter().collect();
            HdVector::bundle(&refs)
        };
        reasoning.push(format!("Encoded input as {}-dim HD vector", self.hd_dim));
        let lower = input.to_lowercase();

        // 2. Check shared memory for similar past interactions
        let memory_match = self.shared_memory.query(&input_vec)
            .map(|(name, sim)| (name.to_string(), sim));
        if let Some((ref name, sim)) = memory_match {
            if sim > 0.3 {
                reasoning.push(format!("Memory match: '{}' (similarity: {:.2})", name, sim));
            }
        }

        // 3. Route to specialist
        let (specialist_name, response_text) = self.route(input, &input_vec, &mut reasoning);

        // 4. Store raw input in shared memory (clean, no response chains)
        self.shared_memory.store(input, input_vec.clone());

        // 5. Store meaningful inputs in memory specialist (skip commands/greetings)
        let is_command = lower.contains("recall") || lower.contains("memory") || lower.contains("hello") || lower.contains("hi ");
        if !is_command && input.split_whitespace().count() > 3 {
            for spec in &mut self.specialists {
                if let SpecialistRole::Memory { hd_memory, .. } = &mut spec.role {
                    hd_memory.store(input, input_vec.clone());
                    spec.invocations += 1;
                }
            }
            reasoning.push("Stored as knowledge in memory".into());
        }

        // 6. Log
        self.log.push(InteractionLog {
            input: input.to_string(),
            specialist_used: specialist_name.clone(),
            output: response_text.clone(),
            success: true,
        });

        OrganismResponse {
            text: response_text,
            specialist: specialist_name,
            confidence: memory_match.map(|(_, s)| s).unwrap_or(0.5),
            reasoning,
            memory_stored: true,
        }
    }

    /// Route input to the best specialist.
    fn route(&mut self, input: &str, input_vec: &HdVector, reasoning: &mut Vec<String>) -> (String, String) {
        let lower = input.to_lowercase();

        // Simple keyword routing (a real organism would use a classifier here)
        let category = if lower.contains("hello") || lower.contains("hi") || lower.contains("hey") {
            "greeting"
        } else if lower.contains("remember") || lower.contains("recall") || lower.contains("memory") {
            "memory"
        } else if lower.contains("classify") || lower.contains("predict") {
            "classify"
        } else if lower.contains("generate") || lower.contains("write") || lower.contains("tell me")
            || lower.contains("explain") || lower.contains("describe") || lower.contains("continue")
        {
            "language"
        } else if lower.contains('?') || lower.contains("what") || lower.contains("how") || lower.contains("why") {
            "question"
        } else {
            // Default: try classifier if available, otherwise unknown
            "classify_or_respond"
        };

        reasoning.push(format!("Routed to category: {}", category));

        // Find and invoke specialist
        match category {
            "language" => {
                for spec in &mut self.specialists {
                    if let SpecialistRole::LanguageModel { ref lm } = spec.role {
                        let generated = lm.generate_with_temp(input, 30, 0.7);
                        spec.invocations += 1;
                        reasoning.push("Language model generated response".into());
                        return (spec.name.clone(), generated);
                    }
                }
                // No language model loaded — fall through to responder
                reasoning.push("No language model loaded, using template".into());
                for spec in &mut self.specialists {
                    if let SpecialistRole::Responder { templates } = &spec.role {
                        let resps = templates.get("unknown").unwrap();
                        let idx = spec.invocations as usize % resps.len();
                        spec.invocations += 1;
                        return (spec.name.clone(), resps[idx].clone());
                    }
                }
            }
            "question" => {
                // Prefer language model for questions if available
                for spec in &mut self.specialists {
                    if let SpecialistRole::LanguageModel { ref lm } = spec.role {
                        let generated = lm.generate_with_temp(input, 30, 0.7);
                        spec.invocations += 1;
                        reasoning.push("Language model answered question".into());
                        return (spec.name.clone(), generated);
                    }
                }
                // Fall back to template responder
                for spec in &mut self.specialists {
                    if let SpecialistRole::Responder { templates } = &spec.role {
                        let resps = templates.get("question").or_else(|| templates.get("unknown"));
                        if let Some(resps) = resps {
                            let idx = (spec.invocations as usize) % resps.len();
                            spec.invocations += 1;
                            let response = resps[idx].clone();
                            reasoning.push(format!("Responder selected template #{}", idx));
                            return (spec.name.clone(), response);
                        }
                    }
                }
            }
            "greeting" | "unknown" => {
                for spec in &mut self.specialists {
                    if let SpecialistRole::Responder { templates } = &spec.role {
                        let responses = templates.get(category).or_else(|| templates.get("unknown"));
                        if let Some(resps) = responses {
                            let idx = (spec.invocations as usize) % resps.len();
                            spec.invocations += 1;
                            let response = resps[idx].clone();
                            reasoning.push(format!("Responder selected template #{}", idx));
                            return (spec.name.clone(), response);
                        }
                    }
                }
            }
            "memory" => {
                for spec in &mut self.specialists {
                    if let SpecialistRole::Memory { hd_memory, .. } = &spec.role {
                        if let Some((name, sim)) = hd_memory.query(input_vec) {
                            if sim > 0.2 {
                                spec.invocations += 1;
                                // Clean: only show the original fact, not chained responses
                                let clean = name.split('→').next().unwrap_or(name).trim();
                                reasoning.push(format!("Memory recalled: {} (sim={:.2})", clean, sim));
                                return (spec.name.clone(), format!("I remember: {}", clean));
                            }
                        }
                        return (spec.name.clone(), "No relevant memories found.".into());
                    }
                }
            }
            "classify" | "classify_or_respond" => {
                // Try spiking specialist first, then classifier
                for spec in &mut self.specialists {
                    if let SpecialistRole::Spiking { network, timesteps, class_names } = &mut spec.role {
                        let features: Vec<f32> = input_vec.data.iter()
                            .map(|&v| if v > 0 { 1.0 } else { 0.0 })
                            .collect();
                        let n_classes = class_names.len().max(1);
                        let class = network.classify(&features, *timesteps, n_classes);
                        let class_name = class_names.get(class).map(|s| s.as_str()).unwrap_or("?");
                        spec.invocations += 1;
                        reasoning.push(format!("Spiking classified as: {} (class {})", class_name, class));
                        let response = format!("Spiking network says: {}", class_name);
                        return (spec.name.clone(), response);
                    }
                }
                for spec in &mut self.specialists {
                    if let SpecialistRole::Classifier { brain, class_names } = &spec.role {
                        let features: Vec<f32> = input_vec.data.iter().map(|&v| v as f32).collect();
                        // Pad or truncate to match expected dim
                        let needed = brain.image_dim;
                        let mut feat = vec![0.0f32; needed];
                        for i in 0..needed.min(features.len()) { feat[i] = features[i]; }
                        let preds = brain.predict(&feat, 1);
                        let class = preds[0] as usize;
                        let class_name = class_names.get(class).map(|s| s.as_str()).unwrap_or("?");
                        spec.invocations += 1;
                        reasoning.push(format!("Classified as: {} (class {})", class_name, class));
                        let response = format!("I think this is about: {}. {}", class_name,
                            match class_name {
                                "World" => "This seems to be a world news topic.",
                                "Sports" => "This appears to be sports-related.",
                                "Business" => "This looks like a business/finance topic.",
                                "Tech" => "This seems to be about technology.",
                                _ => "I classified this input.",
                            });
                        return (spec.name.clone(), response);
                    }
                }
                // No classifier available — try language model, then responder
                for spec in &mut self.specialists {
                    if let SpecialistRole::LanguageModel { ref lm } = spec.role {
                        let generated = lm.generate_with_temp(input, 30, 0.7);
                        spec.invocations += 1;
                        reasoning.push("Language model generated response (fallback)".into());
                        return (spec.name.clone(), generated);
                    }
                }
                for spec in &mut self.specialists {
                    if let SpecialistRole::Responder { templates } = &spec.role {
                        let resps = templates.get("unknown").unwrap();
                        let idx = spec.invocations as usize % resps.len();
                        spec.invocations += 1;
                        return (spec.name.clone(), resps[idx].clone());
                    }
                }
            }
            _ => {}
        }

        ("organism".into(), "Processing...".into())
    }

    /// How many specialists does this organism have?
    pub fn specialist_count(&self) -> usize {
        self.specialists.len()
    }

    /// Total interactions processed.
    pub fn total_interactions(&self) -> usize {
        self.log.len()
    }

    /// Evolve: analyze logs, adjust specialist weights, potentially spawn new specialists.
    pub fn evolve(&mut self) {
        self.generation += 1;

        // Count which specialists are used most/least
        let mut usage: HashMap<String, u64> = HashMap::new();
        for entry in &self.log {
            *usage.entry(entry.specialist_used.clone()).or_insert(0) += 1;
        }

        // Log evolution
        let total = self.log.len() as f64;
        for (name, count) in &usage {
            let pct = (*count as f64 / total.max(1.0)) * 100.0;
            // Update specialist success rate based on usage
            for spec in &mut self.specialists {
                if spec.name == *name {
                    spec.success_rate = pct as f32 / 100.0;
                }
            }
        }
    }

    /// Status report.
    pub fn status(&self) -> String {
        let mut s = format!("Organism Gen {}: {} specialists, {} interactions\n",
            self.generation, self.specialists.len(), self.log.len());
        for spec in &self.specialists {
            s += &format!("  {} — {} invocations, {:.0}% success\n",
                spec.name, spec.invocations, spec.success_rate * 100.0);
        }
        s += &format!("  Shared memory: {} items\n", self.shared_memory.items.len());
        s
    }
}

/// Simple string hash for deterministic HD vector generation.
fn hash_str(s: &str) -> u64 {
    let mut hash = 5381u64;
    for b in s.bytes() { hash = hash.wrapping_mul(33).wrapping_add(b as u64); }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn organism_basic() {
        let mut org = Organism::new(1000);
        assert_eq!(org.specialist_count(), 3); // responder, memory, adapter

        let resp = org.process("Hello!");
        assert!(!resp.text.is_empty());
        assert_eq!(resp.specialist, "responder");
        println!("Greeting: \"{}\"", resp.text);
        println!("Reasoning: {:?}", resp.reasoning);
    }

    #[test]
    fn organism_memory() {
        let mut org = Organism::new(1000);

        // Store something
        org.process("The capital of France is Paris");
        org.process("Rust is a programming language");

        // Recall
        let resp = org.process("recall memory about France");
        println!("Memory: \"{}\"", resp.text);
        println!("Reasoning: {:?}", resp.reasoning);
        assert_eq!(resp.specialist, "memory");
    }

    #[test]
    fn organism_evolves() {
        let mut org = Organism::new(1000);

        // Process various inputs
        org.process("Hello");
        org.process("What is QLANG?");
        org.process("Remember: QLANG is a graph language");
        org.process("How does it work?");
        org.process("Hello again");

        // Evolve
        org.evolve();
        assert_eq!(org.generation, 1);

        let status = org.status();
        println!("{}", status);
        assert!(status.contains("Gen 1"));
        assert!(status.contains("5 interactions"));
    }

    #[test]
    fn organism_multi_specialist() {
        let mut org = Organism::new(500);

        // Add a classifier from MNIST data
        use crate::mnist::MnistData;
        let data = MnistData::synthetic(200, 50);
        let class_names: Vec<String> = (0..10).map(|i| format!("digit_{}", i)).collect();
        org.add_classifier("digit_classifier", &data.train_images, &data.train_labels,
            784, data.n_train, 10, class_names);

        assert_eq!(org.specialist_count(), 4); // responder + memory + adapter + classifier

        // Process
        let r1 = org.process("Hello!");
        let r2 = org.process("What is this?");
        let r3 = org.process("classify this image");

        println!("R1: {} → \"{}\"", r1.specialist, r1.text);
        println!("R2: {} → \"{}\"", r2.specialist, r2.text);
        println!("R3: {} → \"{}\"", r3.specialist, r3.text);

        assert!(org.total_interactions() == 3);
    }

    #[test]
    fn organism_shared_memory_across_specialists() {
        let mut org = Organism::new(500);

        // First interaction stores in shared memory
        org.process("QLANG uses ternary weights");

        // Second interaction can find it
        let resp = org.process("Do you remember ternary?");
        println!("Shared memory test: \"{}\"", resp.text);

        // Check shared memory has items
        assert!(org.shared_memory.items.len() >= 2);
    }
}
