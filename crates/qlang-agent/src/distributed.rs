//! Distributed Training — Split model training across multiple agents.
//!
//! Strategies:
//! 1. Data Parallelism: Each agent trains on different data, gradients are averaged
//! 2. Model Parallelism: Different agents own different layers
//! 3. Pipeline Parallelism: Agents form a pipeline, each processing one stage
//!
//! Communication uses the QLMS (QLANG Message Stream) protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::protocol::{AgentId, Capability};

/// A distributed training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub id: String,
    pub strategy: ParallelStrategy,
    pub workers: Vec<Worker>,
    pub hyperparams: Hyperparams,
    pub status: JobStatus,
}

/// How to distribute the training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelStrategy {
    /// Each worker trains on different data, gradients are averaged.
    DataParallel { n_workers: usize },
    /// Different workers own different layers.
    ModelParallel { layer_assignment: Vec<(usize, usize)> }, // (layer_start, layer_end) per worker
    /// Workers form a pipeline.
    Pipeline { n_stages: usize },
}

/// A worker in the distributed training job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Worker {
    pub id: String,
    pub agent: AgentId,
    pub role: WorkerRole,
    pub status: WorkerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerRole {
    /// Trains and produces gradients.
    Trainer { data_shard: usize },
    /// Aggregates gradients from trainers.
    Aggregator,
    /// Evaluates model on test data.
    Evaluator,
    /// Performs IGQK compression.
    Compressor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkerStatus {
    Idle,
    Training,
    WaitingForSync,
    Done,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Created,
    Running,
    Completed,
    Failed(String),
}

/// Training hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparams {
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub warmup_steps: usize,
}

impl Default for Hyperparams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 32,
            warmup_steps: 100,
        }
    }
}

/// Gradient aggregation methods.
#[derive(Debug, Clone)]
pub enum GradientAggregation {
    /// Simple average of all worker gradients.
    Average,
    /// Weighted average (workers with more data get more weight).
    WeightedAverage(Vec<f32>),
}

/// Aggregate gradients from multiple workers.
pub fn aggregate_gradients(
    gradients: &[Vec<f32>],
    method: &GradientAggregation,
) -> Vec<f32> {
    if gradients.is_empty() {
        return Vec::new();
    }

    let n = gradients[0].len();
    let mut result = vec![0.0f32; n];

    match method {
        GradientAggregation::Average => {
            for grad in gradients {
                for (i, &g) in grad.iter().enumerate() {
                    result[i] += g;
                }
            }
            let scale = 1.0 / gradients.len() as f32;
            for v in &mut result {
                *v *= scale;
            }
        }
        GradientAggregation::WeightedAverage(weights) => {
            let total_weight: f32 = weights.iter().sum();
            for (grad, &w) in gradients.iter().zip(weights.iter()) {
                let normalized_w = w / total_weight;
                for (i, &g) in grad.iter().enumerate() {
                    result[i] += g * normalized_w;
                }
            }
        }
    }

    result
}

/// Create a data-parallel training job.
pub fn create_data_parallel_job(
    job_id: &str,
    n_workers: usize,
    hyperparams: Hyperparams,
) -> TrainingJob {
    let mut workers = Vec::new();

    // Create trainer workers
    for i in 0..n_workers {
        workers.push(Worker {
            id: format!("trainer_{i}"),
            agent: AgentId {
                name: format!("worker_{i}"),
                capabilities: vec![Capability::Execute, Capability::Train],
            },
            role: WorkerRole::Trainer { data_shard: i },
            status: WorkerStatus::Idle,
        });
    }

    // Add aggregator
    workers.push(Worker {
        id: "aggregator".into(),
        agent: AgentId {
            name: "coordinator".into(),
            capabilities: vec![Capability::Execute, Capability::Optimize],
        },
        role: WorkerRole::Aggregator,
        status: WorkerStatus::Idle,
    });

    // Add compressor
    workers.push(Worker {
        id: "compressor".into(),
        agent: AgentId {
            name: "igqk_compressor".into(),
            capabilities: vec![Capability::Compress, Capability::Verify],
        },
        role: WorkerRole::Compressor,
        status: WorkerStatus::Idle,
    });

    TrainingJob {
        id: job_id.into(),
        strategy: ParallelStrategy::DataParallel { n_workers },
        workers,
        hyperparams,
        status: JobStatus::Created,
    }
}

/// Simulate a distributed training step.
pub fn simulate_distributed_step(
    job: &mut TrainingJob,
    worker_gradients: &HashMap<String, Vec<f32>>,
) -> Vec<f32> {
    // Mark trainers as training
    for worker in &mut job.workers {
        if matches!(worker.role, WorkerRole::Trainer { .. }) {
            worker.status = WorkerStatus::Training;
        }
    }

    // Collect gradients
    let gradients: Vec<Vec<f32>> = job.workers.iter()
        .filter(|w| matches!(w.role, WorkerRole::Trainer { .. }))
        .filter_map(|w| worker_gradients.get(&w.id).cloned())
        .collect();

    // Aggregate
    let aggregated = aggregate_gradients(&gradients, &GradientAggregation::Average);

    // Mark all as done
    for worker in &mut job.workers {
        worker.status = WorkerStatus::Done;
    }

    aggregated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_averaging() {
        let grads = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
        ];
        let avg = aggregate_gradients(&grads, &GradientAggregation::Average);
        assert_eq!(avg, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn weighted_gradient_averaging() {
        let grads = vec![
            vec![1.0, 0.0],
            vec![3.0, 4.0],
        ];
        let weights = vec![1.0, 3.0]; // 25% weight on first, 75% on second
        let avg = aggregate_gradients(&grads, &GradientAggregation::WeightedAverage(weights));
        assert!((avg[0] - 2.5).abs() < 1e-5);
        assert!((avg[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn create_job() {
        let job = create_data_parallel_job("test_job", 4, Hyperparams::default());
        assert_eq!(job.workers.len(), 6); // 4 trainers + 1 aggregator + 1 compressor
        assert!(matches!(job.status, JobStatus::Created));
    }

    #[test]
    fn simulate_step() {
        let mut job = create_data_parallel_job("sim_job", 2, Hyperparams::default());

        let mut worker_grads = HashMap::new();
        worker_grads.insert("trainer_0".into(), vec![1.0, 2.0]);
        worker_grads.insert("trainer_1".into(), vec![3.0, 4.0]);

        let result = simulate_distributed_step(&mut job, &worker_grads);
        assert_eq!(result, vec![2.0, 3.0]);

        // All workers should be done
        assert!(job.workers.iter().all(|w| w.status == WorkerStatus::Done));
    }
}
