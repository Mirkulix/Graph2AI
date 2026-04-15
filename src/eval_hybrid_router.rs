use std::path::PathBuf;

use serde::Serialize;

use qlang::hybrid_router_cli::{
    hybrid_router_dataset_path, hybrid_router_model_path, load_dataset, load_specialists,
};
use qlang_runtime::hybrid_router::{
    PlannerState, RiskLevel, Route, HybridRouterSpecialists,
};

#[derive(Debug)]
struct CliOptions {
    dataset_path: PathBuf,
    model_path: PathBuf,
    pretty: bool,
}

#[derive(Serialize)]
struct EvalOutput {
    dataset_path: String,
    model_path: String,
    route: EvalSection,
    risk: EvalSection,
}

#[derive(Serialize)]
struct EvalSection {
    examples: usize,
    accuracy: f32,
    loss: f32,
    labels: Vec<String>,
    confusion_matrix: Vec<Vec<usize>>,
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

    let dataset = load_dataset(&options.dataset_path)
        .unwrap_or_else(|err| fail(2, &format!("failed to load dataset: {err}")));
    let specialists = load_specialists(&options.model_path)
        .unwrap_or_else(|err| fail(3, &format!("failed to load specialists: {err}")));

    let route_labels = Route::ALL
        .iter()
        .map(|label| label.as_str().to_string())
        .collect::<Vec<_>>();
    let risk_labels = RiskLevel::ALL
        .iter()
        .map(|label| label.as_str().to_string())
        .collect::<Vec<_>>();

    let output = EvalOutput {
        dataset_path: options.dataset_path.display().to_string(),
        model_path: options.model_path.display().to_string(),
        route: EvalSection {
            examples: dataset.route_examples.len(),
            accuracy: specialists.route.evaluate_accuracy(&dataset.route_examples),
            loss: specialists.route.evaluate_loss(&dataset.route_examples),
            labels: route_labels,
            confusion_matrix: route_confusion_matrix(&specialists, &dataset),
        },
        risk: EvalSection {
            examples: dataset.risk_examples.len(),
            accuracy: specialists.risk.evaluate_accuracy(&dataset.risk_examples),
            loss: specialists.risk.evaluate_loss(&dataset.risk_examples),
            labels: risk_labels,
            confusion_matrix: risk_confusion_matrix(&specialists, &dataset),
        },
    };

    let rendered = if options.pretty {
        serde_json::to_string_pretty(&output)
    } else {
        serde_json::to_string(&output)
    }
    .expect("serialize eval output");
    println!("{rendered}");
}

enum ParseOutcome {
    Help,
    Message(String),
}

fn parse_args(args: Vec<String>) -> Result<CliOptions, ParseOutcome> {
    let mut dataset_path = hybrid_router_dataset_path();
    let mut model_path = hybrid_router_model_path();
    let mut pretty = true;

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(ParseOutcome::Help),
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
            value => return Err(ParseOutcome::Message(format!("unknown argument: {value}"))),
        }
        idx += 1;
    }

    Ok(CliOptions {
        dataset_path,
        model_path,
        pretty,
    })
}

fn print_usage() {
    eprintln!("eval-hybrid-router");
    eprintln!("  Evaluates saved route/risk specialists against the hybrid router dataset.");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cargo run --bin eval-hybrid-router --no-default-features --offline -- [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --dataset-path <path>    Override the dataset path");
    eprintln!("  --model-path <path>      Override the saved specialist model path");
    eprintln!("  --json                   Emit compact JSON");
    eprintln!("  --pretty                 Emit pretty JSON (default)");
    eprintln!("  -h, --help               Show this help");
}

fn route_confusion_matrix(
    specialists: &HybridRouterSpecialists,
    dataset: &qlang_runtime::hybrid_router::HybridRouterDataset,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; Route::ALL.len()]; Route::ALL.len()];
    for example in &dataset.route_examples {
        let state = PlannerState {
            raw_text: example.raw_text.clone(),
            summary: String::new(),
            route_scores: example.route_scores.clone(),
            risk_scores: example.risk_scores.clone(),
            confidence: example.confidence,
        };
        let expected = route_index(example.label);
        let predicted = route_index(specialists.route.predict_route(&state));
        matrix[expected][predicted] += 1;
    }
    matrix
}

fn risk_confusion_matrix(
    specialists: &HybridRouterSpecialists,
    dataset: &qlang_runtime::hybrid_router::HybridRouterDataset,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; RiskLevel::ALL.len()]; RiskLevel::ALL.len()];
    for example in &dataset.risk_examples {
        let state = PlannerState {
            raw_text: example.raw_text.clone(),
            summary: String::new(),
            route_scores: example.route_scores.clone(),
            risk_scores: example.risk_scores.clone(),
            confidence: example.confidence,
        };
        let expected = risk_index(example.label);
        let predicted = risk_index(specialists.risk.predict_risk(&state));
        matrix[expected][predicted] += 1;
    }
    matrix
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

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}
