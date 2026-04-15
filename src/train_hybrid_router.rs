use std::path::{Path, PathBuf};

use serde::Serialize;

use qlang::hybrid_router_cli::{
    hybrid_router_dataset_path, hybrid_router_model_path, load_dataset,
};
use qlang_runtime::hybrid_router::{train_specialists_from_dataset, HybridRouterSpecialists};

#[derive(Debug)]
struct CliOptions {
    dataset_path: PathBuf,
    model_path: PathBuf,
    pretty: bool,
}

#[derive(Serialize)]
struct TrainingOutput {
    dataset_path: String,
    model_path: String,
    route_examples: usize,
    risk_examples: usize,
    route_metrics: SpecialistMetricsOutput,
    risk_metrics: SpecialistMetricsOutput,
}

#[derive(Serialize)]
struct SpecialistMetricsOutput {
    input_dim: usize,
    output_dim: usize,
    epochs: usize,
    final_loss: f32,
    final_accuracy: f32,
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
    let specialists = train_specialists_from_dataset(&dataset)
        .unwrap_or_else(|err| fail(3, &format!("failed to train specialists: {err}")));
    persist_specialists(&options.model_path, &specialists);

    let output = TrainingOutput {
        dataset_path: options.dataset_path.display().to_string(),
        model_path: options.model_path.display().to_string(),
        route_examples: dataset.route_examples.len(),
        risk_examples: dataset.risk_examples.len(),
        route_metrics: SpecialistMetricsOutput {
            input_dim: specialists.route.input_dim,
            output_dim: specialists.route.output_dim,
            epochs: specialists.route.metrics.epochs,
            final_loss: specialists.route.metrics.final_loss,
            final_accuracy: specialists.route.metrics.final_accuracy,
        },
        risk_metrics: SpecialistMetricsOutput {
            input_dim: specialists.risk.input_dim,
            output_dim: specialists.risk.output_dim,
            epochs: specialists.risk.metrics.epochs,
            final_loss: specialists.risk.metrics.final_loss,
            final_accuracy: specialists.risk.metrics.final_accuracy,
        },
    };

    let rendered = if options.pretty {
        serde_json::to_string_pretty(&output)
    } else {
        serde_json::to_string(&output)
    }
    .expect("serialize training output");
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
    eprintln!("train-hybrid-router");
    eprintln!("  Trains route/risk specialists from the hybrid router dataset and saves them.");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  cargo run --bin train-hybrid-router --no-default-features --offline -- [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --dataset-path <path>    Override the dataset path");
    eprintln!("  --model-path <path>      Override the saved specialist model path");
    eprintln!("  --json                   Emit compact JSON");
    eprintln!("  --pretty                 Emit pretty JSON (default)");
    eprintln!("  -h, --help               Show this help");
}

fn persist_specialists(path: &Path, specialists: &HybridRouterSpecialists) {
    specialists
        .save_to_path(path)
        .unwrap_or_else(|err| fail(4, &format!("failed to save specialists '{}': {err}", path.display())));
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}
