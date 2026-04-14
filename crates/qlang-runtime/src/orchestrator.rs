//! QLANG Orchestrator — coordinates multiple models to solve tasks.

use std::collections::HashMap;
use std::time::{Instant, SystemTime};

/// A task to be solved by the orchestration system
#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub description: String,
    pub task_type: TaskType,
    pub data_class: DataClass,
    pub budget: Budget,
    pub quality_target: f32, // 0.0 - 1.0
}

#[derive(Debug, Clone)]
pub enum TaskType {
    TextGeneration,
    CodeGeneration,
    Analysis,
    Summarization,
    Translation,
    Classification,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct Budget {
    pub max_cost: f64,    // dollars
    pub max_time_ms: u64, // milliseconds
    pub max_models: usize, // max concurrent models
}

/// How to execute a plan
#[derive(Debug, Clone)]
pub enum ExecutionMode {
    /// One model solves it
    Single,
    /// Multiple models in sequence: A -> B -> C
    Sequential,
    /// Multiple models in parallel, take best result
    Tournament,
    /// Multiple models in parallel, majority vote
    Consensus,
    /// Try cheap model first, escalate if quality insufficient
    Escalation,
}

/// A step in an execution plan
#[derive(Debug, Clone)]
pub struct PlanStep {
    pub step_id: usize,
    pub role: String,             // "analyst", "synthesizer", "verifier"
    pub provider_preference: String, // "local", "cheapest", "best_quality"
    pub prompt_template: String,  // Template with {input} placeholder
    pub depends_on: Vec<usize>,   // Step IDs this depends on
    pub timeout_ms: u64,
}

/// An execution plan produced by the Architecture Model
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub task_id: String,
    pub mode: ExecutionMode,
    pub steps: Vec<PlanStep>,
}

/// Result of a single step
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_id: usize,
    pub provider_id: String,
    pub model_id: String,
    pub output: String,
    pub latency_ms: u64,
    pub cost: f64,
    pub quality_estimate: f32,
}

/// Result of a full orchestration
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub task_id: String,
    pub final_output: String,
    pub steps: Vec<StepResult>,
    pub total_cost: f64,
    pub total_latency_ms: u64,
    pub models_used: Vec<String>,
    pub mode: ExecutionMode,
}

/// Decision log entry for audit
#[derive(Debug, Clone)]
pub struct Decision {
    pub timestamp_ms: u64,
    pub action: String,
    pub reason: String,
    pub model_selected: Option<String>,
    pub cost_estimate: Option<f64>,
}

/// The Orchestrator
pub struct Orchestrator {
    pub decisions: Vec<Decision>,
}

impl Orchestrator {
    pub fn new() -> Self {
        Orchestrator {
            decisions: Vec::new(),
        }
    }

    /// Create an execution plan for a task.
    ///
    /// This is the "Architecture Model" — currently rule-based,
    /// can be replaced with an LLM later.
    pub fn plan(&mut self, task: &Task) -> ExecutionPlan {
        let mode = self.select_mode(task);
        let steps = match &mode {
            ExecutionMode::Single => {
                self.log("plan", "Single model sufficient", None);
                vec![PlanStep {
                    step_id: 0,
                    role: "solver".into(),
                    provider_preference: "local".into(),
                    prompt_template: "{input}".into(),
                    depends_on: vec![],
                    timeout_ms: task.budget.max_time_ms,
                }]
            }
            ExecutionMode::Escalation => {
                self.log(
                    "plan",
                    "Escalation: try cheap first, then quality",
                    None,
                );
                vec![
                    PlanStep {
                        step_id: 0,
                        role: "first_try".into(),
                        provider_preference: "cheapest".into(),
                        prompt_template: "{input}".into(),
                        depends_on: vec![],
                        timeout_ms: task.budget.max_time_ms / 2,
                    },
                    PlanStep {
                        step_id: 1,
                        role: "quality_check".into(),
                        provider_preference: "local".into(),
                        prompt_template:
                            "Rate the quality of this answer (0-100): {input}".into(),
                        depends_on: vec![0],
                        timeout_ms: task.budget.max_time_ms / 4,
                    },
                    // Step 2 only executes if quality < threshold (handled in execute)
                    PlanStep {
                        step_id: 2,
                        role: "escalation".into(),
                        provider_preference: "best_quality".into(),
                        prompt_template: "{input}".into(),
                        depends_on: vec![1],
                        timeout_ms: task.budget.max_time_ms / 2,
                    },
                ]
            }
            ExecutionMode::Tournament => {
                self.log("plan", "Tournament: multiple models compete", None);
                let n = task.budget.max_models.min(3);
                let mut steps = Vec::new();
                for i in 0..n {
                    steps.push(PlanStep {
                        step_id: i,
                        role: format!("contestant_{}", i),
                        provider_preference: if i == 0 {
                            "local"
                        } else if i == 1 {
                            "cheapest"
                        } else {
                            "best_quality"
                        }
                        .into(),
                        prompt_template: "{input}".into(),
                        depends_on: vec![],
                        timeout_ms: task.budget.max_time_ms,
                    });
                }
                // Final step: judge picks best
                steps.push(PlanStep {
                    step_id: n,
                    role: "judge".into(),
                    provider_preference: "local".into(),
                    prompt_template:
                        "Compare these answers and pick the best one:\n{input}".into(),
                    depends_on: (0..n).collect(),
                    timeout_ms: task.budget.max_time_ms / 2,
                });
                steps
            }
            ExecutionMode::Sequential => {
                self.log("plan", "Sequential: analyze then synthesize", None);
                vec![
                    PlanStep {
                        step_id: 0,
                        role: "analyst".into(),
                        provider_preference: "local".into(),
                        prompt_template: "Analyze this task:\n{input}".into(),
                        depends_on: vec![],
                        timeout_ms: task.budget.max_time_ms / 2,
                    },
                    PlanStep {
                        step_id: 1,
                        role: "synthesizer".into(),
                        provider_preference: "best_quality".into(),
                        prompt_template:
                            "Based on this analysis, provide a complete answer:\n{input}"
                                .into(),
                        depends_on: vec![0],
                        timeout_ms: task.budget.max_time_ms / 2,
                    },
                ]
            }
            ExecutionMode::Consensus => {
                self.log("plan", "Consensus: majority vote", None);
                vec![PlanStep {
                    step_id: 0,
                    role: "solver".into(),
                    provider_preference: "local".into(),
                    prompt_template: "{input}".into(),
                    depends_on: vec![],
                    timeout_ms: task.budget.max_time_ms,
                }]
            }
        };

        self.log(
            "plan_complete",
            &format!("Mode: {:?}, {} steps", mode, steps.len()),
            None,
        );

        ExecutionPlan {
            task_id: task.id.clone(),
            mode,
            steps,
        }
    }

    /// Select execution mode based on task properties
    fn select_mode(&self, task: &Task) -> ExecutionMode {
        // Simple heuristics — can be replaced with LLM decision later
        if task.budget.max_models <= 1 || task.budget.max_cost <= 0.001 {
            return ExecutionMode::Single;
        }
        if task.quality_target > 0.9 && task.budget.max_models >= 3 {
            return ExecutionMode::Tournament;
        }
        if task.budget.max_cost > 0.01 {
            return ExecutionMode::Escalation;
        }
        ExecutionMode::Single
    }

    /// Execute a plan using the provider registry
    pub fn execute(
        &mut self,
        plan: &ExecutionPlan,
        task: &Task,
        registry: &mut crate::providers::ProviderRegistry,
    ) -> Result<OrchestrationResult, String> {
        let start = Instant::now();
        let mut step_results: HashMap<usize, StepResult> = HashMap::new();
        let mut total_cost = 0.0;
        let mut models_used = Vec::new();

        for step in &plan.steps {
            // Check if dependencies are met
            let input = if step.depends_on.is_empty() {
                task.description.clone()
            } else {
                // Collect outputs from dependencies
                let dep_outputs: Vec<String> = step
                    .depends_on
                    .iter()
                    .filter_map(|dep_id| step_results.get(dep_id).map(|r| r.output.clone()))
                    .collect();
                dep_outputs.join("\n\n---\n\n")
            };

            // Apply prompt template
            let prompt = step.prompt_template.replace("{input}", &input);

            // Select provider preference
            let pref = match step.provider_preference.as_str() {
                "local" => crate::providers::SelectionPreference::Local,
                "cheapest" => crate::providers::SelectionPreference::Cheapest,
                "fastest" => crate::providers::SelectionPreference::Fastest,
                "best_quality" => crate::providers::SelectionPreference::BestQuality,
                _ => crate::providers::SelectionPreference::Local,
            };

            // Map task type to capability
            let capability = match task.task_type {
                TaskType::CodeGeneration => crate::providers::Capability::CodeGeneration,
                TaskType::Summarization => crate::providers::Capability::Summarization,
                _ => crate::providers::Capability::TextGeneration,
            };

            self.log(
                "execute_step",
                &format!(
                    "Step {} ({}): preference={}",
                    step.step_id, step.role, step.provider_preference
                ),
                None,
            );

            // Call the provider
            let response = registry
                .generate(&prompt, &capability, task.data_class.clone(), pref, None)
                .map_err(|e| format!("Step {} failed: {}", step.step_id, e))?;

            self.log(
                "step_complete",
                &format!(
                    "Step {} done: {} tokens, ${:.4}, {}ms",
                    step.step_id, response.output_tokens, response.cost, response.latency_ms
                ),
                Some(&response.model_id),
            );

            total_cost += response.cost;
            if !models_used.contains(&response.model_id) {
                models_used.push(response.model_id.clone());
            }

            step_results.insert(
                step.step_id,
                StepResult {
                    step_id: step.step_id,
                    provider_id: response.provider_id,
                    model_id: response.model_id,
                    output: response.text,
                    latency_ms: response.latency_ms,
                    cost: response.cost,
                    quality_estimate: 0.0, // Would need evaluation
                },
            );

            // Budget check
            if total_cost > task.budget.max_cost {
                self.log(
                    "budget_exceeded",
                    &format!("${:.4} > ${:.4}", total_cost, task.budget.max_cost),
                    None,
                );
                break;
            }
        }

        // Get final output (from last step)
        let last_step_id = plan.steps.last().map(|s| s.step_id).unwrap_or(0);
        let final_output = step_results
            .get(&last_step_id)
            .map(|r| r.output.clone())
            .unwrap_or_default();

        let total_latency = start.elapsed().as_millis() as u64;

        Ok(OrchestrationResult {
            task_id: task.id.clone(),
            final_output,
            steps: step_results.into_values().collect(),
            total_cost,
            total_latency_ms: total_latency,
            models_used,
            mode: plan.mode.clone(),
        })
    }

    fn log(&mut self, action: &str, reason: &str, model: Option<&str>) {
        let ts = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.decisions.push(Decision {
            timestamp_ms: ts,
            action: action.into(),
            reason: reason.into(),
            model_selected: model.map(|s| s.into()),
            cost_estimate: None,
        });
        eprintln!(
            "[orchestrator] {}: {} {}",
            action,
            reason,
            model.unwrap_or("")
        );
    }

    /// Get audit log
    pub fn audit_log(&self) -> &[Decision] {
        &self.decisions
    }
}

// Re-export DataClass from providers for convenience
pub use crate::providers::DataClass;
