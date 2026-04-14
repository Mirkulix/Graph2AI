//! Hybrid LLM routing primitives.
//!
//! This module defines a minimal typed state for:
//! - planner output from a large LLM
//! - small QLANG specialist routing/risk decisions
//! - final orchestration actions derived from those decisions

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Shape, TensorData, TensorType};

use crate::executor::{self, ExecutionError};
use crate::ollama::OllamaClient;
use crate::providers::{
    Capability as ProviderCapability, DataClass, LlmResponse, ProviderRegistry, SelectionPreference,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    Chat,
    Code,
    Search,
    Summarize,
    Unsafe,
}

impl Route {
    pub const ALL: [Route; 5] = [
        Route::Chat,
        Route::Code,
        Route::Search,
        Route::Summarize,
        Route::Unsafe,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Route::Chat => "chat",
            Route::Code => "code",
            Route::Search => "search",
            Route::Summarize => "summarize",
            Route::Unsafe => "unsafe",
        }
    }

    pub fn from_index(index: usize) -> Self {
        Self::ALL.get(index).copied().unwrap_or(Route::Unsafe)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    NeedsReview,
    High,
}

impl RiskLevel {
    pub const ALL: [RiskLevel; 3] = [
        RiskLevel::Low,
        RiskLevel::NeedsReview,
        RiskLevel::High,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RiskLevel::Low => "low",
            RiskLevel::NeedsReview => "needs_review",
            RiskLevel::High => "high",
        }
    }

    pub fn from_index(index: usize) -> Self {
        Self::ALL.get(index).copied().unwrap_or(RiskLevel::High)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerState {
    pub raw_text: String,
    pub summary: String,
    pub route_scores: Vec<f32>,
    pub risk_scores: Vec<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub route: Route,
    pub risk: RiskLevel,
    pub confidence: f32,
    pub next_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteTrainingExample {
    #[serde(default)]
    pub raw_text: String,
    pub route_scores: Vec<f32>,
    pub risk_scores: Vec<f32>,
    pub confidence: f32,
    pub label: Route,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskTrainingExample {
    #[serde(default)]
    pub raw_text: String,
    pub route_scores: Vec<f32>,
    pub risk_scores: Vec<f32>,
    pub confidence: f32,
    pub label: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSpecialistMetrics {
    pub epochs: usize,
    pub final_loss: f32,
    pub final_accuracy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSpecialistModel {
    pub input_dim: usize,
    pub output_dim: usize,
    pub prototypes: Vec<f32>,
    pub temperature: f32,
    pub metrics: RouteSpecialistMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSpecialistModel {
    pub input_dim: usize,
    pub output_dim: usize,
    pub prototypes: Vec<f32>,
    pub temperature: f32,
    pub metrics: RouteSpecialistMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRouterDataset {
    pub route_examples: Vec<RouteTrainingExample>,
    pub risk_examples: Vec<RiskTrainingExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridRouterSpecialists {
    pub route: RouteSpecialistModel,
    pub risk: RiskSpecialistModel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerRun {
    pub state: PlannerState,
    pub source: PlannerSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlannerSource {
    Heuristic,
    Llm {
        provider_id: String,
        model_id: String,
        latency_ms: u64,
    },
}

impl RoutingDecision {
    pub fn route_name(&self) -> &'static str {
        self.route.as_str()
    }

    pub fn risk_name(&self) -> &'static str {
        self.risk.as_str()
    }
}

impl RouteTrainingExample {
    pub fn from_state(state: &PlannerState, label: Route) -> Self {
        Self {
            raw_text: state.raw_text.clone(),
            route_scores: state.route_scores.clone(),
            risk_scores: state.risk_scores.clone(),
            confidence: state.confidence,
            label,
        }
    }
}

impl RiskTrainingExample {
    pub fn from_state(state: &PlannerState, label: RiskLevel) -> Self {
        Self {
            raw_text: state.raw_text.clone(),
            route_scores: state.route_scores.clone(),
            risk_scores: state.risk_scores.clone(),
            confidence: state.confidence,
            label,
        }
    }
}

impl HybridRouterDataset {
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref)
            .map_err(|e| format!("failed to read hybrid router dataset '{}': {e}", path_ref.display()))?;
        let dataset: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("failed to parse hybrid router dataset '{}': {e}", path_ref.display()))?;
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.route_examples.is_empty() {
            return Err("hybrid router dataset requires at least one route example".into());
        }
        if self.risk_examples.is_empty() {
            return Err("hybrid router dataset requires at least one risk example".into());
        }
        for example in &self.route_examples {
            if example.route_scores.len() != Route::ALL.len() {
                return Err("hybrid router route example has invalid route score width".into());
            }
            if example.risk_scores.len() != RiskLevel::ALL.len() {
                return Err("hybrid router route example has invalid risk score width".into());
            }
        }
        for example in &self.risk_examples {
            if example.route_scores.len() != Route::ALL.len() {
                return Err("hybrid router risk example has invalid route score width".into());
            }
            if example.risk_scores.len() != RiskLevel::ALL.len() {
                return Err("hybrid router risk example has invalid risk score width".into());
            }
        }
        Ok(())
    }
}

impl HybridRouterSpecialists {
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path_ref = path.as_ref();
        if let Some(parent) = path_ref.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create model directory '{}': {e}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize hybrid router specialists: {e}"))?;
        fs::write(path_ref, json)
            .map_err(|e| format!("failed to write hybrid router specialists '{}': {e}", path_ref.display()))?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref)
            .map_err(|e| format!("failed to read hybrid router specialists '{}': {e}", path_ref.display()))?;
        serde_json::from_str(&raw)
            .map_err(|e| format!("failed to parse hybrid router specialists '{}': {e}", path_ref.display()))
    }
}

impl RouteSpecialistModel {
    pub fn train(
        examples: &[RouteTrainingExample],
        epochs: usize,
    ) -> Result<Self, String> {
        if examples.is_empty() {
            return Err("route specialist training requires at least one example".into());
        }
        let input_dim = route_feature_dim();
        for example in examples {
            if example.route_scores.len() != Route::ALL.len() {
                return Err("route specialist example has invalid route score width".into());
            }
            if example.risk_scores.len() != RiskLevel::ALL.len() {
                return Err("route specialist example has invalid risk score width".into());
            }
        }

        let output_dim = Route::ALL.len();
        let mut prototypes = vec![0.0; output_dim * input_dim];
        let mut counts = vec![0usize; output_dim];
        for example in examples {
            let features = route_features_from_parts(
                &example.raw_text,
                &example.route_scores,
                &example.risk_scores,
                example.confidence,
            );
            let class_idx = route_index(example.label);
            counts[class_idx] += 1;
            for (feature_idx, value) in features.iter().enumerate() {
                prototypes[class_idx * input_dim + feature_idx] += *value;
            }
        }
        for class_idx in 0..output_dim {
            let count = counts[class_idx].max(1) as f32;
            let start = class_idx * input_dim;
            let end = start + input_dim;
            for value in &mut prototypes[start..end] {
                *value /= count;
            }
            l2_normalize_in_place(&mut prototypes[start..end]);
        }

        let model = Self {
            input_dim,
            output_dim,
            prototypes,
            temperature: 8.0,
            metrics: RouteSpecialistMetrics {
                epochs,
                final_loss: 0.0,
                final_accuracy: 0.0,
            },
        };
        let final_accuracy = model.evaluate_accuracy(examples);
        let final_loss = model.evaluate_loss(examples);

        Ok(Self {
            input_dim,
            output_dim,
            prototypes: model.prototypes,
            temperature: model.temperature,
            metrics: RouteSpecialistMetrics {
                epochs,
                final_loss,
                final_accuracy,
            },
        })
    }

    pub fn predict_scores(&self, state: &PlannerState) -> Vec<f32> {
        let mut features = route_features(state);
        l2_normalize_in_place(&mut features);
        let logits = (0..self.output_dim)
            .map(|class_idx| {
                let start = class_idx * self.input_dim;
                let end = start + self.input_dim;
                dot(&features, &self.prototypes[start..end]) * self.temperature
            })
            .collect::<Vec<_>>();
        softmax_vec(&logits)
    }

    pub fn predict_route(&self, state: &PlannerState) -> Route {
        let scores = self.predict_scores(state);
        Route::from_index(argmax(&scores))
    }

    pub fn evaluate_accuracy(&self, examples: &[RouteTrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        let correct = examples
            .iter()
            .filter(|example| {
                let state = PlannerState {
                    raw_text: example.raw_text.clone(),
                    summary: String::new(),
                    route_scores: example.route_scores.clone(),
                    risk_scores: example.risk_scores.clone(),
                    confidence: example.confidence,
                };
                self.predict_route(&state) == example.label
            })
            .count();
        correct as f32 / examples.len() as f32
    }

    pub fn evaluate_loss(&self, examples: &[RouteTrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for example in examples {
            let state = PlannerState {
                raw_text: example.raw_text.clone(),
                summary: String::new(),
                route_scores: example.route_scores.clone(),
                risk_scores: example.risk_scores.clone(),
                confidence: example.confidence,
            };
            let probs = self.predict_scores(&state);
            let target = route_index(example.label);
            total -= probs[target].max(1e-7).ln();
        }
        total / examples.len() as f32
    }
}

impl RiskSpecialistModel {
    pub fn train(
        examples: &[RiskTrainingExample],
        epochs: usize,
    ) -> Result<Self, String> {
        if examples.is_empty() {
            return Err("risk specialist training requires at least one example".into());
        }
        let input_dim = risk_feature_dim();
        for example in examples {
            if example.route_scores.len() != Route::ALL.len() {
                return Err("risk specialist example has invalid route score width".into());
            }
            if example.risk_scores.len() != RiskLevel::ALL.len() {
                return Err("risk specialist example has invalid risk score width".into());
            }
        }

        let output_dim = RiskLevel::ALL.len();
        let mut prototypes = vec![0.0; output_dim * input_dim];
        let mut counts = vec![0usize; output_dim];
        for example in examples {
            let features = risk_features_from_parts(
                &example.raw_text,
                &example.route_scores,
                &example.risk_scores,
                example.confidence,
            );
            let class_idx = risk_index(example.label);
            counts[class_idx] += 1;
            for (feature_idx, value) in features.iter().enumerate() {
                prototypes[class_idx * input_dim + feature_idx] += *value;
            }
        }
        for class_idx in 0..output_dim {
            let count = counts[class_idx].max(1) as f32;
            let start = class_idx * input_dim;
            let end = start + input_dim;
            for value in &mut prototypes[start..end] {
                *value /= count;
            }
            l2_normalize_in_place(&mut prototypes[start..end]);
        }

        let model = Self {
            input_dim,
            output_dim,
            prototypes,
            temperature: 8.0,
            metrics: RouteSpecialistMetrics {
                epochs,
                final_loss: 0.0,
                final_accuracy: 0.0,
            },
        };
        let final_accuracy = model.evaluate_accuracy(examples);
        let final_loss = model.evaluate_loss(examples);

        Ok(Self {
            input_dim,
            output_dim,
            prototypes: model.prototypes,
            temperature: model.temperature,
            metrics: RouteSpecialistMetrics {
                epochs,
                final_loss,
                final_accuracy,
            },
        })
    }

    pub fn predict_scores(&self, state: &PlannerState) -> Vec<f32> {
        let mut features = risk_features(state);
        l2_normalize_in_place(&mut features);
        let logits = (0..self.output_dim)
            .map(|class_idx| {
                let start = class_idx * self.input_dim;
                let end = start + self.input_dim;
                dot(&features, &self.prototypes[start..end]) * self.temperature
            })
            .collect::<Vec<_>>();
        softmax_vec(&logits)
    }

    pub fn predict_risk(&self, state: &PlannerState) -> RiskLevel {
        let scores = self.predict_scores(state);
        RiskLevel::from_index(argmax(&scores))
    }

    pub fn evaluate_accuracy(&self, examples: &[RiskTrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        let correct = examples
            .iter()
            .filter(|example| {
                let state = PlannerState {
                    raw_text: example.raw_text.clone(),
                    summary: String::new(),
                    route_scores: example.route_scores.clone(),
                    risk_scores: example.risk_scores.clone(),
                    confidence: example.confidence,
                };
                self.predict_risk(&state) == example.label
            })
            .count();
        correct as f32 / examples.len() as f32
    }

    pub fn evaluate_loss(&self, examples: &[RiskTrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for example in examples {
            let state = PlannerState {
                raw_text: example.raw_text.clone(),
                summary: String::new(),
                route_scores: example.route_scores.clone(),
                risk_scores: example.risk_scores.clone(),
                confidence: example.confidence,
            };
            let probs = self.predict_scores(&state);
            let target = risk_index(example.label);
            total -= probs[target].max(1e-7).ln();
        }
        total / examples.len() as f32
    }
}

pub fn default_route_training_examples() -> Vec<RouteTrainingExample> {
    vec![
        RouteTrainingExample { raw_text: "General chat request about project status".into(), route_scores: vec![0.82, 0.08, 0.03, 0.04, 0.03], risk_scores: vec![0.90, 0.08, 0.02], confidence: 0.91, label: Route::Chat },
        RouteTrainingExample { raw_text: "Please inspect this Rust panic and tell me which file to patch first".into(), route_scores: vec![0.18, 0.70, 0.05, 0.04, 0.03], risk_scores: vec![0.15, 0.75, 0.10], confidence: 0.86, label: Route::Code },
        RouteTrainingExample { raw_text: "Search the docs and find the API endpoint".into(), route_scores: vec![0.10, 0.14, 0.68, 0.05, 0.03], risk_scores: vec![0.70, 0.20, 0.10], confidence: 0.80, label: Route::Search },
        RouteTrainingExample { raw_text: "Summarize the meeting notes".into(), route_scores: vec![0.08, 0.08, 0.07, 0.73, 0.04], risk_scores: vec![0.85, 0.10, 0.05], confidence: 0.88, label: Route::Summarize },
        RouteTrainingExample { raw_text: "Financial advice request with compliance risk".into(), route_scores: vec![0.05, 0.05, 0.04, 0.06, 0.80], risk_scores: vec![0.05, 0.20, 0.75], confidence: 0.92, label: Route::Unsafe },
        RouteTrainingExample { raw_text: "Debug this production panic in a Rust service".into(), route_scores: vec![0.22, 0.58, 0.08, 0.08, 0.04], risk_scores: vec![0.10, 0.82, 0.08], confidence: 0.84, label: Route::Code },
        RouteTrainingExample { raw_text: "Answer a simple product question".into(), route_scores: vec![0.76, 0.10, 0.05, 0.06, 0.03], risk_scores: vec![0.88, 0.09, 0.03], confidence: 0.87, label: Route::Chat },
        RouteTrainingExample { raw_text: "Find the missing benchmark file in the repository".into(), route_scores: vec![0.09, 0.16, 0.64, 0.08, 0.03], risk_scores: vec![0.72, 0.18, 0.10], confidence: 0.81, label: Route::Search },
        RouteTrainingExample { raw_text: "Summarize the benchmark results".into(), route_scores: vec![0.07, 0.11, 0.08, 0.70, 0.04], risk_scores: vec![0.83, 0.12, 0.05], confidence: 0.85, label: Route::Summarize },
        RouteTrainingExample { raw_text: "Unsafe delete request in production".into(), route_scores: vec![0.04, 0.08, 0.03, 0.05, 0.80], risk_scores: vec![0.04, 0.26, 0.70], confidence: 0.90, label: Route::Unsafe },
        RouteTrainingExample { raw_text: "Patch this Rust module after a stacktrace".into(), route_scores: vec![0.15, 0.62, 0.11, 0.07, 0.05], risk_scores: vec![0.12, 0.78, 0.10], confidence: 0.82, label: Route::Code },
        RouteTrainingExample { raw_text: "Chat reply about release timing".into(), route_scores: vec![0.68, 0.14, 0.07, 0.07, 0.04], risk_scores: vec![0.86, 0.10, 0.04], confidence: 0.84, label: Route::Chat },
    ]
}

pub fn default_risk_training_examples() -> Vec<RiskTrainingExample> {
    vec![
        RiskTrainingExample { raw_text: "General chat request about project status".into(), route_scores: vec![0.82, 0.08, 0.03, 0.04, 0.03], risk_scores: vec![0.90, 0.08, 0.02], confidence: 0.91, label: RiskLevel::Low },
        RiskTrainingExample { raw_text: "Please inspect this Rust panic and tell me which file to patch first".into(), route_scores: vec![0.18, 0.70, 0.05, 0.04, 0.03], risk_scores: vec![0.15, 0.75, 0.10], confidence: 0.86, label: RiskLevel::NeedsReview },
        RiskTrainingExample { raw_text: "Financial advice request with compliance risk".into(), route_scores: vec![0.05, 0.05, 0.04, 0.06, 0.80], risk_scores: vec![0.05, 0.20, 0.75], confidence: 0.92, label: RiskLevel::High },
        RiskTrainingExample { raw_text: "Debug this production panic in a Rust service".into(), route_scores: vec![0.12, 0.64, 0.10, 0.08, 0.06], risk_scores: vec![0.10, 0.82, 0.08], confidence: 0.88, label: RiskLevel::NeedsReview },
        RiskTrainingExample { raw_text: "Answer a simple product question".into(), route_scores: vec![0.75, 0.09, 0.06, 0.06, 0.04], risk_scores: vec![0.86, 0.10, 0.04], confidence: 0.90, label: RiskLevel::Low },
        RiskTrainingExample { raw_text: "Unsafe delete request in production".into(), route_scores: vec![0.09, 0.08, 0.04, 0.07, 0.72], risk_scores: vec![0.04, 0.18, 0.78], confidence: 0.95, label: RiskLevel::High },
        RiskTrainingExample { raw_text: "Find the missing benchmark file in the repository".into(), route_scores: vec![0.08, 0.15, 0.62, 0.10, 0.05], risk_scores: vec![0.72, 0.20, 0.08], confidence: 0.82, label: RiskLevel::Low },
        RiskTrainingExample { raw_text: "Patch this Rust module after a stacktrace".into(), route_scores: vec![0.16, 0.58, 0.11, 0.08, 0.07], risk_scores: vec![0.18, 0.70, 0.12], confidence: 0.79, label: RiskLevel::NeedsReview },
        RiskTrainingExample { raw_text: "High-stakes unsafe production change".into(), route_scores: vec![0.05, 0.07, 0.06, 0.09, 0.73], risk_scores: vec![0.03, 0.19, 0.78], confidence: 0.93, label: RiskLevel::High },
    ]
}

pub fn default_hybrid_router_dataset() -> HybridRouterDataset {
    HybridRouterDataset {
        route_examples: default_route_training_examples(),
        risk_examples: default_risk_training_examples(),
    }
}

pub fn train_default_route_specialist() -> Result<RouteSpecialistModel, String> {
    RouteSpecialistModel::train(&default_route_training_examples(), 1)
}

pub fn train_default_risk_specialist() -> Result<RiskSpecialistModel, String> {
    RiskSpecialistModel::train(&default_risk_training_examples(), 1)
}

pub fn train_route_specialist_from_dataset(dataset: &HybridRouterDataset) -> Result<RouteSpecialistModel, String> {
    RouteSpecialistModel::train(&dataset.route_examples, 1)
}

pub fn train_risk_specialist_from_dataset(dataset: &HybridRouterDataset) -> Result<RiskSpecialistModel, String> {
    RiskSpecialistModel::train(&dataset.risk_examples, 1)
}

pub fn train_specialists_from_dataset(dataset: &HybridRouterDataset) -> Result<HybridRouterSpecialists, String> {
    Ok(HybridRouterSpecialists {
        route: train_route_specialist_from_dataset(dataset)?,
        risk: train_risk_specialist_from_dataset(dataset)?,
    })
}

/// Placeholder planner logic until a real large-model adapter is connected.
pub fn heuristic_planner_state(request: &str) -> PlannerState {
    let lower = request.to_ascii_lowercase();

    let mut route_scores = vec![0.10, 0.10, 0.10, 0.10, 0.05];
    if lower.contains("rust") || lower.contains("panic") || lower.contains("patch") || lower.contains("file") {
        route_scores = vec![0.05, 0.82, 0.04, 0.04, 0.05];
    } else if lower.contains("search") || lower.contains("find") {
        route_scores = vec![0.05, 0.08, 0.78, 0.04, 0.05];
    } else if lower.contains("summary") || lower.contains("summarize") {
        route_scores = vec![0.06, 0.06, 0.05, 0.78, 0.05];
    } else if lower.contains("medical") || lower.contains("legal") || lower.contains("financial") {
        route_scores = vec![0.04, 0.04, 0.05, 0.02, 0.85];
    }

    let mut risk_scores = vec![0.75, 0.20, 0.05];
    if lower.contains("panic") || lower.contains("production") || lower.contains("delete") {
        risk_scores = vec![0.15, 0.70, 0.15];
    }
    if lower.contains("medical") || lower.contains("legal") || lower.contains("financial") {
        risk_scores = vec![0.05, 0.20, 0.75];
    }

    PlannerState {
        raw_text: request.to_string(),
        summary: "Structured planner state: request should be routed before final response generation.".into(),
        route_scores,
        risk_scores,
        confidence: 0.78,
    }
}

pub fn planner_state_with_fallback(
    registry: &mut ProviderRegistry,
    request: &str,
) -> PlannerRun {
    match planner_state_with_llm(registry, request) {
        Ok((state, response)) => PlannerRun {
            state,
            source: PlannerSource::Llm {
                provider_id: response.provider_id,
                model_id: response.model_id,
                latency_ms: response.latency_ms,
            },
        },
        Err(err) => {
            eprintln!("[hybrid_router] planner llm fallback: {err}");
            PlannerRun {
                state: heuristic_planner_state(request),
                source: PlannerSource::Heuristic,
            }
        }
    }
}

pub fn planner_state_with_llm(
    registry: &mut ProviderRegistry,
    request: &str,
) -> Result<(PlannerState, LlmResponse), String> {
    let prompt = planner_prompt(request);
    if let Some(response) = try_local_ollama_planner(&prompt)? {
        let state = parse_planner_state(request, &response.text)?;
        return Ok((state, response));
    }
    let response = registry.generate(
        &prompt,
        &ProviderCapability::TextGeneration,
        DataClass::Internal,
        SelectionPreference::Local,
        None,
    )?;
    let state = parse_planner_state(request, &response.text)?;
    Ok((state, response))
}

pub fn build_route_graph() -> Graph {
    build_decision_graph("route-decision", "route_scores", "selected_route", Route::ALL.len())
}

pub fn build_risk_graph() -> Graph {
    build_decision_graph("risk-decision", "risk_scores", "selected_risk", RiskLevel::ALL.len())
}

pub fn execute_route_graph(scores: &[f32]) -> Result<TensorData, ExecutionError> {
    execute_decision_graph(&build_route_graph(), "route_scores", "selected_route", scores)
}

pub fn execute_risk_graph(scores: &[f32]) -> Result<TensorData, ExecutionError> {
    execute_decision_graph(&build_risk_graph(), "risk_scores", "selected_risk", scores)
}

pub fn resolve_decision(state: &PlannerState, route_output: &TensorData, risk_output: &TensorData) -> RoutingDecision {
    let route = Route::from_index(decode_one_hot_index(route_output));
    let risk = RiskLevel::from_index(decode_one_hot_index(risk_output));
    RoutingDecision {
        route,
        risk,
        confidence: state.confidence,
        next_action: final_action(&state.raw_text, &state.summary, route, risk),
    }
}

pub fn blend_scores(primary: &[f32], specialist: &[f32], primary_weight: f32) -> Vec<f32> {
    let weight = primary_weight.clamp(0.0, 1.0);
    let specialist_weight = 1.0 - weight;
    let len = primary.len().min(specialist.len());
    let mut mixed = Vec::with_capacity(len);
    for idx in 0..len {
        mixed.push(primary[idx] * weight + specialist[idx] * specialist_weight);
    }
    normalize_scores(mixed)
}

fn build_decision_graph(graph_id: &str, input_name: &str, output_name: &str, width: usize) -> Graph {
    let tensor = TensorType::f32_vector(width);
    let mut graph = Graph::new(graph_id);
    let input = graph.add_node(Op::Input { name: input_name.into() }, vec![], vec![tensor.clone()]);
    let choose = graph.add_node(Op::Measure, vec![tensor.clone()], vec![tensor.clone()]);
    let output = graph.add_node(Op::Output { name: output_name.into() }, vec![tensor.clone()], vec![]);
    graph.add_edge(input, 0, choose, 0, tensor.clone());
    graph.add_edge(choose, 0, output, 0, tensor);
    graph
}

fn route_feature_dim() -> usize {
    Route::ALL.len() + RiskLevel::ALL.len() + 1 + text_feature_dim()
}

fn risk_feature_dim() -> usize {
    Route::ALL.len() + RiskLevel::ALL.len() + 1 + text_feature_dim()
}

fn route_features(state: &PlannerState) -> Vec<f32> {
    route_features_from_parts(&state.raw_text, &state.route_scores, &state.risk_scores, state.confidence)
}

fn risk_features(state: &PlannerState) -> Vec<f32> {
    risk_features_from_parts(&state.raw_text, &state.route_scores, &state.risk_scores, state.confidence)
}

fn route_features_from_parts(raw_text: &str, route_scores: &[f32], risk_scores: &[f32], confidence: f32) -> Vec<f32> {
    let mut features = Vec::with_capacity(route_feature_dim());
    features.extend_from_slice(route_scores);
    features.extend_from_slice(risk_scores);
    features.push(confidence);
    features.extend(text_features(raw_text));
    features
}

fn risk_features_from_parts(raw_text: &str, route_scores: &[f32], risk_scores: &[f32], confidence: f32) -> Vec<f32> {
    let mut features = Vec::with_capacity(risk_feature_dim());
    features.extend_from_slice(risk_scores);
    features.extend_from_slice(route_scores);
    features.push(confidence);
    features.extend(text_features(raw_text));
    features
}

fn text_feature_dim() -> usize {
    5
}

fn text_features(raw_text: &str) -> Vec<f32> {
    let lower = raw_text.to_ascii_lowercase();
    vec![
        contains_any(&lower, &["panic", "crash", "stacktrace"]) as u8 as f32,
        contains_any(&lower, &["rust", "compile", "module", "file", "patch"]) as u8 as f32,
        contains_any(&lower, &["search", "find", "lookup", "retrieve"]) as u8 as f32,
        contains_any(&lower, &["summary", "summarize", "recap"]) as u8 as f32,
        contains_any(&lower, &["medical", "legal", "financial", "unsafe", "delete", "production"]) as u8 as f32,
    ]
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn route_index(route: Route) -> usize {
    Route::ALL
        .iter()
        .position(|candidate| *candidate == route)
        .unwrap_or(Route::ALL.len() - 1)
}

fn risk_index(risk: RiskLevel) -> usize {
    RiskLevel::ALL
        .iter()
        .position(|candidate| *candidate == risk)
        .unwrap_or(RiskLevel::ALL.len() - 1)
}

fn l2_normalize_in_place(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1e-6 || !norm.is_finite() {
        return;
    }
    for value in values {
        *value /= norm;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(lhs, rhs)| lhs * rhs).sum()
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    for value in &mut exps {
        *value /= sum;
    }
    exps
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn execute_decision_graph(
    graph: &Graph,
    input_name: &str,
    output_name: &str,
    scores: &[f32],
) -> Result<TensorData, ExecutionError> {
    let mut inputs = HashMap::new();
    inputs.insert(
        input_name.to_string(),
        TensorData::from_f32(Shape::vector(scores.len()), scores),
    );
    let result = executor::execute_unverified(graph, inputs)?;
    result
        .outputs
        .get(output_name)
        .cloned()
        .ok_or_else(|| ExecutionError::RuntimeError(format!("missing output '{output_name}'")))
}

fn planner_prompt(request: &str) -> String {
    format!(
        "You are a routing planner for a hybrid LLM + specialist system.\n\
 Return exactly one valid JSON object.\n\
 Do not include markdown fences.\n\
 Do not include comments.\n\
 Do not include explanations.\n\
 Do not include thinking text.\n\
  \n\
  Schema:\n\
  {{\n\
    \"summary\": string,\n\
    \"route_scores\": [number, number, number, number, number],\n\
  \"risk_scores\": [number, number, number],\n\
  \"confidence\": number\n\
}}\n\
\n\
Route order: [\"chat\", \"code\", \"search\", \"summarize\", \"unsafe\"]\n\
Risk order: [\"low\", \"needs_review\", \"high\"]\n\
\n\
Rules:\n\
- Scores must be non-negative.\n\
- Higher score means more likely.\n\
- Confidence must be between 0 and 1.\n\
- Keep the summary short and operational.\n\
\n\
Request:\n{request}"
    )
}

fn parse_planner_state(request: &str, raw: &str) -> Result<PlannerState, String> {
    let json_text = extract_json_object(raw).ok_or_else(|| "planner response did not contain JSON".to_string())?;
    let sanitized = strip_json_line_comments(&json_text);
    let value: Value =
        serde_json::from_str(&sanitized).map_err(|e| format!("planner JSON parse failed: {e}; raw={sanitized}"))?;

    let summary = value
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("Planner response omitted summary.")
        .to_string();

    let route_scores = normalize_scores(parse_score_array(
        value.get("route_scores"),
        Route::ALL.len(),
        "route_scores",
    )?);
    let risk_scores = normalize_scores(parse_score_array(
        value.get("risk_scores"),
        RiskLevel::ALL.len(),
        "risk_scores",
    )?);

    let confidence = value
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0) as f32;

    Ok(PlannerState {
        raw_text: request.to_string(),
        summary,
        route_scores,
        risk_scores,
        confidence,
    })
}

fn decode_one_hot_index(tensor: &TensorData) -> usize {
    let values = tensor.as_f32_slice().unwrap_or_default();
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn final_action(request: &str, summary: &str, route: Route, risk: RiskLevel) -> String {
    match (route, risk) {
        (Route::Code, RiskLevel::NeedsReview) => format!(
            "{summary} Next step: inspect the repository, locate the likely failing module, and prepare a patch plan before answering directly. Request='{request}'"
        ),
        (Route::Search, RiskLevel::Low) => format!(
            "{summary} Next step: gather context through search or retrieval, then synthesize a direct answer. Request='{request}'"
        ),
        (Route::Unsafe, _) | (_, RiskLevel::High) => format!(
            "{summary} Next step: escalate to a large verified LLM path and avoid autonomous execution. Request='{request}'"
        ),
        _ => format!(
            "{summary} Next step: answer through route='{}' with risk='{}'. Request='{request}'",
            route.as_str(),
            risk.as_str(),
        ),
    }
}

fn parse_score_array(value: Option<&Value>, expected_len: usize, field: &str) -> Result<Vec<f32>, String> {
    let arr = value
        .and_then(Value::as_array)
        .ok_or_else(|| format!("planner field '{field}' missing or not an array"))?;
    if arr.len() != expected_len {
        return Err(format!(
            "planner field '{field}' must contain {expected_len} scores, got {}",
            arr.len()
        ));
    }
    let mut out = Vec::with_capacity(expected_len);
    for item in arr {
        out.push(item.as_f64().unwrap_or(0.0).max(0.0) as f32);
    }
    Ok(out)
}

fn normalize_scores(mut scores: Vec<f32>) -> Vec<f32> {
    let sum: f32 = scores.iter().sum();
    if sum <= f32::EPSILON {
        let fill = 1.0 / scores.len().max(1) as f32;
        scores.fill(fill);
        return scores;
    }
    for score in &mut scores {
        *score /= sum;
    }
    scores
}

fn extract_json_object(raw: &str) -> Option<String> {
    if let Some(stripped) = raw
        .trim()
        .strip_prefix("```json")
        .and_then(|s| s.strip_suffix("```"))
    {
        return Some(stripped.trim().to_string());
    }
    let start = raw.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in raw[start..].char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset;
                    return Some(raw[start..=end].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_json_line_comments(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            result.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_string => {
                result.push(ch);
                escaped = true;
            }
            '"' => {
                in_string = !in_string;
                result.push(ch);
            }
            '/' if !in_string && chars.peek() == Some(&'/') => {
                chars.next();
                for next in chars.by_ref() {
                    if next == '\n' {
                        result.push('\n');
                        break;
                    }
                }
            }
            _ => result.push(ch),
        }
    }

    result
}

fn try_local_ollama_planner(prompt: &str) -> Result<Option<LlmResponse>, String> {
    let client = OllamaClient::from_env();
    let available = client
        .list_models()
        .map_err(|e| format!("Ollama model listing failed: {e}"))?;
    let model = preferred_planner_model(&available);
    let Some(model_id) = model else {
        return Ok(None);
    };

    let start = Instant::now();
    let text = client
        .generate(model_id, prompt, Some("Return only one valid JSON object with no comments, markdown, or reasoning text."))
        .map_err(|e| format!("Ollama planner call failed: {e}"))?;
    Ok(Some(LlmResponse {
        provider_id: "ollama".to_string(),
        model_id: model_id.to_string(),
        text,
        input_tokens: 0,
        output_tokens: 0,
        cost: 0.0,
        latency_ms: start.elapsed().as_millis() as u64,
    }))
}

fn preferred_planner_model<'a>(available: &'a [String]) -> Option<&'a str> {
    if let Ok(explicit) = std::env::var("QLANG_PLANNER_MODEL") {
        if let Some(found) = available.iter().find(|model| model.as_str() == explicit) {
            return Some(found.as_str());
        }
    }

    for preferred in ["qwen3:8b", "llama3:latest", "mistral-nemo:latest", "falcon3:1b"] {
        if let Some(found) = available.iter().find(|model| model.as_str() == preferred) {
            return Some(found.as_str());
        }
    }

    available
        .iter()
        .find(|model| !model.contains("embed") && !model.contains("cloud"))
        .map(|model| model.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_graph_selects_max_score() {
        let output = execute_route_graph(&[0.05, 0.80, 0.10, 0.03, 0.02]).unwrap();
        let decision = decode_one_hot_index(&output);
        assert_eq!(Route::from_index(decision), Route::Code);
    }

    #[test]
    fn risk_graph_selects_max_score() {
        let output = execute_risk_graph(&[0.10, 0.20, 0.70]).unwrap();
        let decision = decode_one_hot_index(&output);
        assert_eq!(RiskLevel::from_index(decision), RiskLevel::High);
    }

    #[test]
    fn decision_resolution_combines_outputs() {
        let state = heuristic_planner_state("Please inspect this Rust panic");
        let route_out = execute_route_graph(&state.route_scores).unwrap();
        let risk_out = execute_risk_graph(&state.risk_scores).unwrap();
        let decision = resolve_decision(&state, &route_out, &risk_out);
        assert_eq!(decision.route, Route::Code);
        assert_eq!(decision.risk, RiskLevel::NeedsReview);
        assert!(decision.next_action.contains("patch plan"));
    }

    #[test]
    fn parse_planner_json_normalizes_scores() {
        let raw = r#"{
            "summary": "Route this to code review.",
            "route_scores": [2, 6, 1, 1, 0],
            "risk_scores": [0, 3, 1],
            "confidence": 0.91
        }"#;
        let state = parse_planner_state("inspect panic", raw).unwrap();
        assert!((state.route_scores.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((state.risk_scores.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(state.summary, "Route this to code review.");
        assert!((state.confidence - 0.91).abs() < 1e-5);
    }

    #[test]
    fn extract_json_from_markdown_block() {
        let raw = "```json\n{\"summary\":\"ok\",\"route_scores\":[1,0,0,0,0],\"risk_scores\":[1,0,0],\"confidence\":0.5}\n```";
        let extracted = extract_json_object(raw).unwrap();
        assert!(extracted.starts_with('{'));
        assert!(extracted.ends_with('}'));
    }

    #[test]
    fn strip_json_comments_for_planner_output() {
        let raw = r#"{
            "summary": "ok",
            "route_scores": [
                0.2, // chat
                0.8,
                0.0,
                0.0,
                0.0
            ],
            "risk_scores": [0.1, 0.9, 0.0],
            "confidence": 0.7
        }"#;
        let state = parse_planner_state("inspect panic", raw).unwrap();
        assert_eq!(state.summary, "ok");
        assert_eq!(state.route_scores.len(), 5);
    }

    #[test]
    fn trained_route_specialist_prefers_code_for_panic_case() {
        let model = train_default_route_specialist().unwrap();
        let state = PlannerState {
            raw_text: "Please inspect this Rust panic".into(),
            summary: "Investigate crash".into(),
            route_scores: vec![0.12, 0.64, 0.10, 0.08, 0.06],
            risk_scores: vec![0.10, 0.82, 0.08],
            confidence: 0.88,
        };
        let predicted = model.predict_route(&state);
        assert_eq!(predicted, Route::Code);
        assert!(model.metrics.final_accuracy >= 0.80);
    }

    #[test]
    fn trained_risk_specialist_prefers_review_for_panic_case() {
        let model = train_default_risk_specialist().unwrap();
        let state = PlannerState {
            raw_text: "Please inspect this Rust panic".into(),
            summary: "Investigate crash".into(),
            route_scores: vec![0.12, 0.64, 0.10, 0.08, 0.06],
            risk_scores: vec![0.10, 0.82, 0.08],
            confidence: 0.88,
        };
        let predicted = model.predict_risk(&state);
        assert_eq!(predicted, RiskLevel::NeedsReview);
        assert!(model.metrics.final_accuracy >= 0.80);
    }

    #[test]
    fn default_dataset_validates() {
        default_hybrid_router_dataset().validate().unwrap();
    }

    #[test]
    fn load_dataset_from_json_file() {
        let path = std::env::temp_dir().join("hybrid-router-dataset-test.json");
        let json = r#"{
            "route_examples": [
                {
                    "route_scores": [0.1, 0.7, 0.1, 0.05, 0.05],
                    "risk_scores": [0.1, 0.8, 0.1],
                    "confidence": 0.9,
                    "label": "Code"
                }
            ],
            "risk_examples": [
                {
                    "route_scores": [0.1, 0.7, 0.1, 0.05, 0.05],
                    "risk_scores": [0.1, 0.8, 0.1],
                    "confidence": 0.9,
                    "label": "NeedsReview"
                }
            ]
        }"#;
        std::fs::write(&path, json).unwrap();
        let dataset = HybridRouterDataset::load_from_path(&path).unwrap();
        assert_eq!(dataset.route_examples.len(), 1);
        assert_eq!(dataset.risk_examples.len(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn persist_specialists_roundtrip() {
        let dataset = default_hybrid_router_dataset();
        let models = train_specialists_from_dataset(&dataset).unwrap();
        let path = std::env::temp_dir().join("hybrid-router-specialists-test.json");
        models.save_to_path(&path).unwrap();
        let loaded = HybridRouterSpecialists::load_from_path(&path).unwrap();
        assert_eq!(loaded.route.input_dim, models.route.input_dim);
        assert_eq!(loaded.risk.output_dim, models.risk.output_dim);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn blend_scores_preserves_strong_primary_signal() {
        let primary = vec![0.05, 0.82, 0.04, 0.04, 0.05];
        let specialist = vec![0.30, 0.14, 0.29, 0.25, 0.02];
        let blended = blend_scores(&primary, &specialist, 0.70);
        assert_eq!(argmax(&blended), route_index(Route::Code));
    }

    #[test]
    fn blended_risk_can_pull_high_down_to_review() {
        let primary = vec![0.0, 0.0, 1.0];
        let specialist = vec![0.05, 0.88, 0.07];
        let blended = blend_scores(&primary, &specialist, 0.40);
        assert_eq!(argmax(&blended), risk_index(RiskLevel::NeedsReview));
    }
}
