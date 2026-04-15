use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use serde::Serialize;

use qlang::hybrid_router_cli::{
    hybrid_router_dataset_path, hybrid_router_model_path, load_dataset_with_fallback, load_specialists,
};
use qlang_runtime::hybrid_router::{
    blend_scores, execute_risk_graph, execute_route_graph, heuristic_planner_state, planner_state_with_fallback,
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
    planner_timeout_ms: Option<u64>,
    batch: bool,
    text_file: Option<PathBuf>,
    no_llm: bool,
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
        Err(ParseOutcome::Message(message)) => fail(2, &message),
    };

    let planner_model = options.planner_model.clone();
    let planner_timeout_ms = options.planner_timeout_ms;
    with_planner_env(planner_model.as_deref(), planner_timeout_ms, || run(options));
}

fn run(options: CliOptions) {
    let dataset = load_dataset_with_fallback(&options.dataset_path);
    let specialists = load_specialists(&options.model_path).unwrap_or_else(|err| {
        fail(
            3,
            &format!(
                "failed to load specialists from '{}': {err}\nRun `cargo run --bin train-hybrid-router --no-default-features --offline` first.",
                options.model_path.display()
            ),
        )
    });

    let mut providers = ProviderRegistry::new(1.0);
    if !options.no_llm {
        providers.discover_all();
    }

    if options.batch {
        run_batch(&options, &specialists, &mut providers);
        return;
    }

    let request = options
        .request
        .clone()
        .unwrap_or_else(|| fail(2, "missing request text"));
        let output = classify_one(
            &request,
            &options.dataset_path,
            &options.model_path,
            &specialists,
            &mut providers,
            options.no_llm,
        );
    print_output(&output, options.pretty);
    drop(dataset);
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
    let mut planner_timeout_ms = None;
    let mut request_parts = Vec::new();
    let mut text_file: Option<PathBuf> = None;
    let mut batch = false;
    let mut no_llm = false;

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(ParseOutcome::Help),
            "--batch" | "-b" => batch = true,
            "--json" => pretty = false,
            "--pretty" => pretty = true,
            "--no-llm" => no_llm = true,
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
            "--planner-timeout" => {
                idx += 1;
                let Some(value) = args.get(idx) else {
                    return Err(ParseOutcome::Message("missing value for --planner-timeout".into()));
                };
                planner_timeout_ms = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| ParseOutcome::Message("invalid value for --planner-timeout".into()))?,
                );
            }
            value if value.starts_with("--") => {
                return Err(ParseOutcome::Message(format!("unknown option: {value}")));
            }
            value => request_parts.push(value.to_string()),
        }
        idx += 1;
    }

    if no_llm && planner_model.is_some() {
        return Err(ParseOutcome::Message(
            "--no-llm cannot be combined with --planner-model".into(),
        ));
    }

    let request = if batch {
        let joined = request_parts.join(" ").trim().to_string();
        if joined.is_empty() { None } else { Some(joined) }
    } else if let Some(path) = &text_file {
        Some(read_text_file(path, "request text file")?)
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
        planner_timeout_ms,
        batch,
        text_file,
        no_llm,
    })
}

fn read_text_file(path: &Path, label: &str) -> Result<String, ParseOutcome> {
    fs::read_to_string(path)
        .map(|text| text.trim().to_string())
        .map_err(|e| ParseOutcome::Message(format!("failed to read {label} '{}': {e}", path.display())))
}

fn print_usage() {
    eprintln!("classify-request");
    eprintln!("  Classifies a request into route/risk using the hybrid LLM + QLANG specialists.");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cargo run --bin classify-request --no-default-features --offline -- [options] \"request text\"");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --batch, -b              Read one request per line and emit JSON Lines");
    eprintln!("  --text-file <path>       Read request text from a file");
    eprintln!("  --dataset-path <path>    Override the dataset path");
    eprintln!("  --model-path <path>      Override the saved specialist model path");
    eprintln!("  --planner-model <name>   Force a specific local planner model, e.g. qwen3:8b");
    eprintln!("  --planner-timeout <ms>   Set QLANG_PLANNER_TIMEOUT_MS for this run");
    eprintln!("  --no-llm                 Skip provider discovery and use heuristic planner only");
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
            .unwrap_or_else(|e| fail(2, &format!("failed to open batch text file '{}': {e}", path.display())));
        let reader = io::BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap_or_else(|e| fail(2, &format!("failed to read batch line: {e}")));
            let request = line.trim();
            if request.is_empty() {
                continue;
            }
            had_input = true;
            let output = classify_one(
                request,
                &options.dataset_path,
                &options.model_path,
                specialists,
                providers,
                options.no_llm,
            );
            println!("{}", serde_json::to_string(&output).expect("serialize batch output"));
        }
    } else {
        let stdin = io::stdin();
        let reader = io::BufReader::new(stdin.lock());
        for line in reader.lines() {
            let line = line.unwrap_or_else(|e| fail(2, &format!("failed to read stdin line: {e}")));
            let request = line.trim();
            if request.is_empty() {
                continue;
            }
            had_input = true;
            let output = classify_one(
                request,
                &options.dataset_path,
                &options.model_path,
                specialists,
                providers,
                options.no_llm,
            );
            println!("{}", serde_json::to_string(&output).expect("serialize batch output"));
        }
    }

    if !had_input {
        eprintln!("[classify-request] batch mode: no input received");
        std::process::exit(4);
    }
}

fn classify_one(
    request: &str,
    dataset_path: &Path,
    model_path: &Path,
    specialists: &HybridRouterSpecialists,
    providers: &mut ProviderRegistry,
    no_llm: bool,
) -> ClassificationOutput {
    let planner_run = if no_llm {
        qlang_runtime::hybrid_router::PlannerRun {
            state: heuristic_planner_state(request),
            source: PlannerSource::Heuristic,
        }
    } else {
        planner_state_with_fallback(providers, request)
    };
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

fn print_output(output: &ClassificationOutput, pretty: bool) {
    let rendered = if pretty {
        serde_json::to_string_pretty(output)
    } else {
        serde_json::to_string(output)
    }
    .expect("serialize classification output");
    println!("{rendered}");
}

fn with_planner_env<F>(planner_model: Option<&str>, planner_timeout_ms: Option<u64>, action: F)
where
    F: FnOnce(),
{
    let previous_model = std::env::var("QLANG_PLANNER_MODEL").ok();
    let previous_timeout = std::env::var("QLANG_PLANNER_TIMEOUT_MS").ok();

    if let Some(model) = planner_model {
        std::env::set_var("QLANG_PLANNER_MODEL", model);
    }
    if let Some(timeout_ms) = planner_timeout_ms {
        std::env::set_var("QLANG_PLANNER_TIMEOUT_MS", timeout_ms.to_string());
    }

    action();

    restore_env_var("QLANG_PLANNER_MODEL", previous_model.as_deref(), planner_model.is_some());
    restore_env_var(
        "QLANG_PLANNER_TIMEOUT_MS",
        previous_timeout.as_deref(),
        planner_timeout_ms.is_some(),
    );
}

fn restore_env_var(key: &str, previous: Option<&str>, touched: bool) {
    match previous {
        Some(value) => std::env::set_var(key, value),
        None if touched => std::env::remove_var(key),
        None => {}
    }
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}
