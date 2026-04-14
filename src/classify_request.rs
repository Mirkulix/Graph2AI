use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;

use serde::Serialize;

use qlang_runtime::hybrid_router::{
    blend_scores, default_hybrid_router_dataset, execute_risk_graph, execute_route_graph,
    planner_state_with_fallback, train_specialists_from_dataset, HybridRouterDataset,
    HybridRouterSpecialists, PlannerSource, resolve_decision,
};
use qlang_runtime::providers::ProviderRegistry;

#[derive(Debug)]
struct CliOptions {
    request: Option<String>,
    dataset_path: PathBuf,
    model_path: PathBuf,
    pretty: bool,
    planner_model: Option<String>,
    batch: bool,
    text_file: Option<PathBuf>,
}

#[derive(Serialize)]
struct ClassificationOutput {
    request: String,
    planner_source: String,
    planner_summary: String,
    route: String,
    risk: String,
    confidence: f32,
    next_action: String,
    planner_route_scores: Vec<f32>,
    planner_risk_scores: Vec<f32>,
    specialist_route_scores: Vec<f32>,
    specialist_risk_scores: Vec<f32>,
    blended_route_scores: Vec<f32>,
    blended_risk_scores: Vec<f32>,
    dataset_path: String,
    model_path: String,
}

fn main() {
    let options = match parse_args(std::env::args().skip(1).collect()) {
        Ok(options) => options,
        Err(ParseOutcome::Help) => {
            print_usage();
            return;
        }
        Err(ParseOutcome::Message(message)) => {
            eprintln!("{message}");
            eprintln!();
            print_usage();
            std::process::exit(2);
        }
    };

    let previous_planner_model = options
        .planner_model
        .as_ref()
        .and_then(|value| std::env::var("QLANG_PLANNER_MODEL").ok().map(|current| (value.clone(), current)));
    if let Some(model) = &options.planner_model {
        unsafe {
            std::env::set_var("QLANG_PLANNER_MODEL", model);
        }
    }

    let dataset = load_dataset_with_fallback(&options.dataset_path);
    let specialists = load_or_train_specialists(&options.model_path, &dataset);
    let mut providers = ProviderRegistry::new(1.0);
    providers.discover_all();

    if options.batch {
        run_batch(&options, &specialists, &mut providers);
    } else {
        let request = options.request.clone().expect("request required in single mode");
        let output = classify_one(
            &request,
            &options.dataset_path,
            &options.model_path,
            &specialists,
            &mut providers,
        );
        let rendered = if options.pretty {
            serde_json::to_string_pretty(&output)
        } else {
            serde_json::to_string(&output)
        }
        .expect("serialize classification output");
        println!("{rendered}");
    }

    if let Some((_, previous)) = previous_planner_model {
        unsafe {
            std::env::set_var("QLANG_PLANNER_MODEL", previous);
        }
    } else if options.planner_model.is_some() {
        unsafe {
            std::env::remove_var("QLANG_PLANNER_MODEL");
        }
    }
}

enum ParseOutcome {
    Help,
    Message(String),
}

fn parse_args(args: Vec<String>) -> Result<CliOptions, ParseOutcome> {
    let mut dataset_path = hybrid_router_dataset_path();
    let mut model_path = hybrid_router_model_path();
    let mut pretty = true;
    let mut planner_model = None;
    let mut request_parts = Vec::new();
    let mut text_file: Option<PathBuf> = None;
    let mut batch = false;

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(ParseOutcome::Help),
            "--batch" | "-b" => batch = true,
            "--json" => pretty = false,
            "--pretty" => pretty = true,
            "--dataset-path" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err(ParseOutcome::Message("missing value for --dataset-path".into()));
                };
                dataset_path = PathBuf::from(value);
            }
            "--model-path" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err(ParseOutcome::Message("missing value for --model-path".into()));
                };
                model_path = PathBuf::from(value);
            }
            "--text-file" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err(ParseOutcome::Message("missing value for --text-file".into()));
                };
                text_file = Some(PathBuf::from(value));
            }
            "--planner-model" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err(ParseOutcome::Message("missing value for --planner-model".into()));
                };
                planner_model = Some(value.clone());
            }
            value if value.starts_with("--") => {
                return Err(ParseOutcome::Message(format!("unknown option: {value}")));
            }
            value => request_parts.push(value.to_string()),
        }
        idx += 1;
    }

    let request = if batch {
        let joined = request_parts.join(" ").trim().to_string();
        if joined.is_empty() { None } else { Some(joined) }
    } else if let Some(path) = &text_file {
        Some(
            fs::read_to_string(path)
                .map_err(|e| ParseOutcome::Message(format!("failed to read text file '{}': {e}", path.display())))?
                .trim()
                .to_string(),
        )
    } else {
        let joined = request_parts.join(" ").trim().to_string();
        if joined.is_empty() { None } else { Some(joined) }
    };

    if !batch && request.as_deref().unwrap_or_default().is_empty() {
        return Err(ParseOutcome::Message("missing request text".into()));
    }

    Ok(CliOptions {
        request,
        dataset_path,
        model_path,
        pretty,
        planner_model,
        batch,
        text_file,
    })
}

fn print_usage() {
    eprintln!("classify-request");
    eprintln!("  Classifies a request into route/risk using the hybrid LLM + QLANG specialists.");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cargo run --bin classify-request --no-default-features --offline -- [options] \"request text\"");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --batch, -b             Read one request per line and emit JSON Lines");
    eprintln!("  --text-file <path>       Read request text from a file");
    eprintln!("  --dataset-path <path>    Override the training dataset path");
    eprintln!("  --model-path <path>      Override the saved specialist model path");
    eprintln!("  --planner-model <name>   Force a specific local planner model, e.g. qwen3:8b");
    eprintln!("  --json                   Emit compact JSON");
    eprintln!("  --pretty                 Emit pretty JSON (default)");
    eprintln!("  -h, --help               Show this help");
}

fn run_batch(
    options: &CliOptions,
    specialists: &HybridRouterSpecialists,
    providers: &mut ProviderRegistry,
) {
    let mut had_input = false;
    if let Some(path) = &options.text_file {
        let file = fs::File::open(path)
            .unwrap_or_else(|e| panic!("failed to open batch text file '{}': {e}", path.display()));
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            let line = line.expect("failed to read batch line");
            let request = line.trim();
            if request.is_empty() {
                continue;
            }
            had_input = true;
            let output = classify_one(request, &options.dataset_path, &options.model_path, specialists, providers);
            println!("{}", serde_json::to_string(&output).expect("serialize batch output"));
        }
    } else {
        let stdin = io::stdin();
        let reader = io::BufReader::new(stdin.lock());
        for line in reader.lines() {
            let line = line.expect("failed to read stdin line");
            let request = line.trim();
            if request.is_empty() {
                continue;
            }
            had_input = true;
            let output = classify_one(request, &options.dataset_path, &options.model_path, specialists, providers);
            println!("{}", serde_json::to_string(&output).expect("serialize batch output"));
        }
    }

    if !had_input {
        eprintln!("[classify-request] batch mode: no input received");
    }
}

fn classify_one(
    request: &str,
    dataset_path: &PathBuf,
    model_path: &PathBuf,
    specialists: &HybridRouterSpecialists,
    providers: &mut ProviderRegistry,
) -> ClassificationOutput {
    let planner_run = planner_state_with_fallback(providers, request);
    let planner_state = planner_run.state;

    let specialist_route_scores = specialists.route.predict_scores(&planner_state);
    let specialist_risk_scores = specialists.risk.predict_scores(&planner_state);
    let blended_route_scores = blend_scores(&planner_state.route_scores, &specialist_route_scores, 0.70);
    let blended_risk_scores = blend_scores(&planner_state.risk_scores, &specialist_risk_scores, 0.30);

    let route_output = execute_route_graph(&blended_route_scores).expect("execute route graph");
    let risk_output = execute_risk_graph(&blended_risk_scores).expect("execute risk graph");
    let decision = resolve_decision(&planner_state, &route_output, &risk_output);

    let planner_source = match planner_run.source {
        PlannerSource::Heuristic => "heuristic".to_string(),
        PlannerSource::Llm {
            provider_id,
            model_id,
            latency_ms,
        } => format!("{provider_id}/{model_id} ({latency_ms} ms)"),
    };

    ClassificationOutput {
        request: request.to_string(),
        planner_source,
        planner_summary: planner_state.summary,
        route: decision.route_name().to_string(),
        risk: decision.risk_name().to_string(),
        confidence: decision.confidence,
        next_action: decision.next_action,
        planner_route_scores: planner_state.route_scores,
        planner_risk_scores: planner_state.risk_scores,
        specialist_route_scores,
        specialist_risk_scores,
        blended_route_scores,
        blended_risk_scores,
        dataset_path: dataset_path.display().to_string(),
        model_path: model_path.display().to_string(),
    }
}

fn hybrid_router_dataset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join("hybrid_router_training.json")
}

fn hybrid_router_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join("hybrid_router_specialists.json")
}

fn load_dataset_with_fallback(path: &PathBuf) -> HybridRouterDataset {
    match HybridRouterDataset::load_from_path(path) {
        Ok(dataset) => dataset,
        Err(err) => {
            eprintln!("[classify-request] dataset fallback: {err}");
            default_hybrid_router_dataset()
        }
    }
}

fn load_or_train_specialists(path: &PathBuf, dataset: &HybridRouterDataset) -> HybridRouterSpecialists {
    match HybridRouterSpecialists::load_from_path(path) {
        Ok(models) => models,
        Err(err) => {
            eprintln!("[classify-request] model fallback: {err}");
            let models = train_specialists_from_dataset(dataset).expect("train hybrid router specialists");
            models
                .save_to_path(path)
                .expect("persist hybrid router specialists");
            models
        }
    }
}
