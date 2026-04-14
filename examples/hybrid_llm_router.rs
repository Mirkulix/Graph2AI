//! Hybrid LLM <-> QLANG specialist demo.
//!
//! This example shows the smallest useful version of the architecture:
//! - a "large LLM" planner interprets the raw user text
//! - two small QLANG specialists perform route/risk decisions
//! - results travel as typed graph messages instead of free-form text
//!
//! Modes:
//!   single (default):  process one hardcoded request
//!   --batch            read lines from stdin, output JSON Lines per request

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use qlang_agent::bus::{build_reply, MessageBus};
use qlang_agent::protocol::{next_msg_id, AgentId, Capability, GraphMessage, MessageIntent};
use qlang_core::binary;
use qlang_core::graph::Graph;
use qlang_core::tensor::{Shape, TensorData};
use qlang_runtime::executor;
use qlang_runtime::hybrid_router::{
    blend_scores, build_risk_graph, build_route_graph, default_hybrid_router_dataset, planner_state_with_fallback,
    train_specialists_from_dataset, HybridRouterDataset, HybridRouterSpecialists, PlannerSource, resolve_decision,
    Route, RiskLevel, RoutingDecision,
};
use qlang_runtime::providers::ProviderRegistry;

const VERSION: &str = "0.1.0";

#[derive(Debug, serde::Serialize)]
struct BatchResult {
    raw_text: String,
    route: String,
    risk: String,
    confidence: f32,
    next_action: String,
    planner_source: String,
    planner_latency_ms: Option<u64>,
}

fn routing_decision_to_batch_result(decision: &RoutingDecision, raw_text: &str, planner_source: &PlannerSource) -> BatchResult {
    let (source_str, latency_ms) = match planner_source {
        PlannerSource::Heuristic => ("heuristic".to_string(), None),
        PlannerSource::Llm { model_id, latency_ms, .. } => (model_id.clone(), Some(*latency_ms)),
    };
    BatchResult {
        raw_text: raw_text.to_string(),
        route: decision.route_name().to_string(),
        risk: decision.risk_name().to_string(),
        confidence: decision.confidence,
        next_action: decision.next_action.clone(),
        planner_source: source_str,
        planner_latency_ms: latency_ms,
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_help();
        return;
    }

    let batch_mode = args.iter().any(|a| a == "--batch" || a == "-b");

    if batch_mode {
        run_batch().await;
    } else {
        run_single().await;
    }
}

fn print_help() {
    println!(
        r#"hybrid_llm_router v{}

Usage:
  hybrid_llm_router              # single hardcoded request (demo mode)
  hybrid_llm_router --batch     # read lines from stdin, output JSON Lines
  hybrid_llm_router --help      # this help

Batch input format:
  one request per line, plain text
  empty lines are skipped

Batch output format (JSON Lines):
  {{"raw_text":"...","route":"...","risk":"...","confidence":0.85,"next_action":"...","planner_source":"...","planner_latency_ms":123}}

Examples:
  echo "Summarize this document" | hybrid_llm_router --batch
  hybrid_llm_router --batch < requests.txt > results.jsonl
  cat requests.txt | hybrid_llm_router --batch | jq '.route'
"#,
        VERSION
    );
}

async fn run_batch() {
    use std::io::{self, BufRead};

    let (dataset, specialists) = init_shared_batch().await;

    let mut providers = ProviderRegistry::new(1.0);
    providers.discover_all();

    let stdin = io::stdin();
    let reader = io::BufReader::new(stdin);

    let mut had_input = false;
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let request = line.trim();
        if request.is_empty() {
            continue;
        }
        had_input = true;

        let result = process_single_request(request, &dataset, &specialists, &mut providers).await;
        println!("{}", serde_json::to_string(&result).unwrap_or_default());
    }

    if !had_input {
        eprintln!(
            "[hybrid_llm_router] batch mode: no input read from stdin. \
             Pass lines via pipe or redirect. See --help."
        );
    }
}

async fn init_shared_batch() -> (HybridRouterDataset, HybridRouterSpecialists) {
    let dataset_path = hybrid_router_dataset_path();
    let model_path = hybrid_router_model_path();

    let dataset = load_dataset_with_fallback(&dataset_path);
    let specialists = load_or_train_specialists(&model_path, &dataset);

    (dataset, specialists)
}

async fn run_single() {
    let (dataset, specialists) = init_shared().await;

    println!("=== Hybrid LLM Routing Demo ===");
    println!("Use case: a large LLM prepares structured state, QLANG specialists make cheap routing decisions.\n");

    let request = "Please inspect this Rust panic, identify the likely failing module, and tell me which file to patch first.";
    println!("User request: {request}\n");

    let mut providers = ProviderRegistry::new(1.0);
    providers.discover_all();

    let planner_run = planner_state_with_fallback(&mut providers, request);
    let planner_state = planner_run.state;
    let route_specialist = &specialists.route;
    let risk_specialist = &specialists.risk;
    let specialist_route_scores = route_specialist.predict_scores(&planner_state);
    let specialist_risk_scores = risk_specialist.predict_scores(&planner_state);
    let blended_route_scores = blend_scores(&planner_state.route_scores, &specialist_route_scores, 0.70);
    let blended_risk_scores = blend_scores(&planner_state.risk_scores, &specialist_risk_scores, 0.30);

    println!("Training dataset: datasets/hybrid_router_training.json");
    println!("Specialist model: artifacts/hybrid_router_specialists.json");

    match planner_run.source {
        PlannerSource::Heuristic => println!("Planner source: heuristic fallback"),
        PlannerSource::Llm {
            provider_id,
            model_id,
            latency_ms,
        } => {
            println!("Planner source: llm ({provider_id}/{model_id}, {latency_ms} ms)")
        }
    }
    println!("Planner state:");
    println!("  summary: {}", planner_state.summary);
    println!("  route_scores: {:?}", planner_state.route_scores);
    println!("  risk_scores:  {:?}\n", planner_state.risk_scores);
    println!("Route specialist:");
    println!(
        "  trained: epochs={}, loss={:.4}, acc={:.2}%",
        route_specialist.metrics.epochs,
        route_specialist.metrics.final_loss,
        route_specialist.metrics.final_accuracy * 100.0
    );
    println!("  route_scores: {:?}\n", specialist_route_scores);
    println!("Risk specialist:");
    println!(
        "  trained: epochs={}, loss={:.4}, acc={:.2}%",
        risk_specialist.metrics.epochs,
        risk_specialist.metrics.final_loss,
        risk_specialist.metrics.final_accuracy * 100.0
    );
    println!("  risk_scores:  {:?}\n", specialist_risk_scores);
    println!("Blended specialist state:");
    println!("  route_scores: {:?}", blended_route_scores);
    println!("  risk_scores:  {:?}\n", blended_risk_scores);

    let route_graph = build_route_graph();
    let risk_graph = build_risk_graph();

    let route_bin = binary::to_binary(&route_graph);
    let route_json = serde_json::to_vec(&route_graph).expect("serialize route graph");
    println!(
        "Route graph exchange: binary={} bytes, json={} bytes",
        route_bin.len(),
        route_json.len()
    );

    let bus = MessageBus::new();

    let planner = agent("planner", vec![Capability::Optimize]);
    let router_agent = agent("route-specialist", vec![Capability::Execute]);
    let risk_agent = agent("risk-specialist", vec![Capability::Execute]);

    let mut planner_mailbox = bus.register(planner.clone()).await;
    let router_mailbox = bus.register(router_agent.clone()).await;
    let risk_mailbox = bus.register(risk_agent.clone()).await;

    spawn_specialist(bus.clone(), router_agent.clone(), router_mailbox);
    spawn_specialist(bus.clone(), risk_agent.clone(), risk_mailbox);

    let route_reply = bus
        .send_and_wait(
            GraphMessage {
                id: next_msg_id(),
                from: planner.clone(),
                to: router_agent.clone(),
                graph: route_graph,
                inputs: tensor_inputs("route_scores", &blended_route_scores),
                intent: MessageIntent::Execute,
                in_reply_to: None,
                signature: None,
                signer_pubkey: None,
                graph_hash: None,
            },
            &mut planner_mailbox,
            Duration::from_secs(2),
        )
        .await
        .expect("route decision");

    let risk_reply = bus
        .send_and_wait(
            GraphMessage {
                id: next_msg_id(),
                from: planner.clone(),
                to: risk_agent.clone(),
                graph: risk_graph,
                inputs: tensor_inputs("risk_scores", &blended_risk_scores),
                intent: MessageIntent::Execute,
                in_reply_to: None,
                signature: None,
                signer_pubkey: None,
                graph_hash: None,
            },
            &mut planner_mailbox,
            Duration::from_secs(2),
        )
        .await
        .expect("risk decision");

    let decision = resolve_decision(
        &planner_state,
        route_reply.inputs.get("selected_route").expect("route output"),
        risk_reply.inputs.get("selected_risk").expect("risk output"),
    );

    println!("\nSpecialist decisions:");
    println!("  route: {}", decision.route_name());
    println!("  risk:  {}", decision.risk_name());

    println!("\nPlanner decision:");
    println!("  {}", decision.next_action);

    let stats = bus.stats().await;
    println!("\nMessage bus stats:");
    println!("  messages: {}", stats.total_messages);
    println!("  delivered: {}", stats.delivered);
    println!("  conversations: {}", stats.active_conversations);

    drop(dataset);
    drop(specialists);
}

async fn init_shared() -> (
    HybridRouterDataset,
    HybridRouterSpecialists,
) {
    let dataset_path = hybrid_router_dataset_path();
    let model_path = hybrid_router_model_path();

    let dataset = load_dataset_with_fallback(&dataset_path);
    let specialists = load_or_train_specialists(&model_path, &dataset);

    (dataset, specialists)
}

async fn process_single_request(
    request: &str,
    _dataset: &HybridRouterDataset,
    specialists: &HybridRouterSpecialists,
    providers: &mut ProviderRegistry,
) -> BatchResult {
    let planner_run = planner_state_with_fallback(providers, request);
    let planner_state = planner_run.state;

    let specialist_route_scores = specialists.route.predict_scores(&planner_state);
    let specialist_risk_scores = specialists.risk.predict_scores(&planner_state);
    let blended_route_scores = blend_scores(&planner_state.route_scores, &specialist_route_scores, 0.70);
    let blended_risk_scores = blend_scores(&planner_state.risk_scores, &specialist_risk_scores, 0.30);

    let route_graph = build_route_graph();
    let risk_graph = build_risk_graph();

    let bus = MessageBus::new();

    let planner = agent("planner", vec![Capability::Optimize]);
    let router_agent = agent("route-specialist", vec![Capability::Execute]);
    let risk_agent = agent("risk-specialist", vec![Capability::Execute]);

    let mut planner_mailbox = bus.register(planner.clone()).await;
    let router_mailbox = bus.register(router_agent.clone()).await;
    let risk_mailbox = bus.register(risk_agent.clone()).await;

    spawn_specialist(bus.clone(), router_agent.clone(), router_mailbox);
    spawn_specialist(bus.clone(), risk_agent.clone(), risk_mailbox);

    let route_reply = match bus
        .send_and_wait(
            GraphMessage {
                id: next_msg_id(),
                from: planner.clone(),
                to: router_agent.clone(),
                graph: route_graph,
                inputs: tensor_inputs("route_scores", &blended_route_scores),
                intent: MessageIntent::Execute,
                in_reply_to: None,
                signature: None,
                signer_pubkey: None,
                graph_hash: None,
            },
            &mut planner_mailbox,
            Duration::from_secs(2),
        )
        .await
    {
        Ok(reply) => reply,
        Err(_) => {
            return BatchResult {
                raw_text: request.to_string(),
                route: Route::Unsafe.as_str().to_string(),
                risk: RiskLevel::High.as_str().to_string(),
                confidence: 0.0,
                next_action: "planner_timeout".to_string(),
                planner_source: "timeout".to_string(),
                planner_latency_ms: None,
            };
        }
    };

    let risk_reply = match bus
        .send_and_wait(
            GraphMessage {
                id: next_msg_id(),
                from: planner.clone(),
                to: risk_agent.clone(),
                graph: risk_graph,
                inputs: tensor_inputs("risk_scores", &blended_risk_scores),
                intent: MessageIntent::Execute,
                in_reply_to: None,
                signature: None,
                signer_pubkey: None,
                graph_hash: None,
            },
            &mut planner_mailbox,
            Duration::from_secs(2),
        )
        .await
    {
        Ok(reply) => reply,
        Err(_) => {
            return BatchResult {
                raw_text: request.to_string(),
                route: Route::Unsafe.as_str().to_string(),
                risk: RiskLevel::High.as_str().to_string(),
                confidence: 0.0,
                next_action: "risk_specialist_timeout".to_string(),
                planner_source: "timeout".to_string(),
                planner_latency_ms: None,
            };
        }
    };

    let route_tensor = route_reply.inputs.get("selected_route");
    let risk_tensor = risk_reply.inputs.get("selected_risk");

    let decision = if let (Some(route_t), Some(risk_t)) = (route_tensor, risk_tensor) {
        resolve_decision(&planner_state, route_t, risk_t)
    } else {
        RoutingDecision {
            route: Route::Unsafe,
            risk: RiskLevel::High,
            confidence: 0.0,
            next_action: "missing_specialist_output".to_string(),
        }
    };

    routing_decision_to_batch_result(&decision, request, &planner_run.source)
}

fn spawn_specialist(
    bus: std::sync::Arc<MessageBus>,
    me: AgentId,
    mut mailbox: qlang_agent::bus::Mailbox,
) {
    tokio::spawn(async move {
        while let Some(msg) = mailbox.recv().await {
            if !matches!(msg.intent, MessageIntent::Execute) {
                continue;
            }

            let result = executor::execute_unverified(&msg.graph, msg.inputs.clone())
                .expect("specialist graph execution");
            let reply = build_reply(&msg, me.clone(), Graph::new(format!("reply-{}", me.name)), result.outputs);
            let _ = bus.send(reply).await;
        }
    });
}

fn agent(name: &str, capabilities: Vec<Capability>) -> AgentId {
    AgentId {
        name: name.to_string(),
        capabilities,
    }
}

fn tensor_inputs(name: &str, values: &[f32]) -> HashMap<String, TensorData> {
    let mut inputs = HashMap::new();
    inputs.insert(name.to_string(), TensorData::from_f32(Shape::vector(values.len()), values));
    inputs
}

fn hybrid_router_dataset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("hybrid_router_training.json")
}

fn load_dataset_with_fallback(path: &PathBuf) -> HybridRouterDataset {
    match HybridRouterDataset::load_from_path(path) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("[hybrid_llm_router] dataset fallback: {err}");
            default_hybrid_router_dataset()
        }
    }
}

fn hybrid_router_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join("hybrid_router_specialists.json")
}

fn load_or_train_specialists(path: &PathBuf, dataset: &HybridRouterDataset) -> HybridRouterSpecialists {
    match HybridRouterSpecialists::load_from_path(path) {
        Ok(models) => models,
        Err(err) => {
            eprintln!("[hybrid_llm_router] model fallback: {err}");
            let models = train_specialists_from_dataset(dataset).expect("train hybrid router specialists");
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            models
                .save_to_path(path)
                .expect("persist hybrid router specialists");
            models
        }
    }
}
