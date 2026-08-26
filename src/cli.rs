//! QLANG CLI — Train, Infer, Inspect ternary neural networks.
//!
//! Like BitNet.cpp, but with integrated training and agent communication.

mod graph_sync;
mod keys;
mod supervisor;





// ============================================================
// Main
// ============================================================

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "supervisor" => supervisor::handle_supervisor(&args[2..]),
        "run-agent" => supervisor::handle_run_agent(&args[2..]),
        "graph" => graph_sync::handle_graph(&args[2..]).await,
        "keys" => keys::handle_keys(&args[2..]),
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("QLANG — Agent-to-Agent Control Plane");
    println!();
    println!("A graph-native AI coordination layer for signed, executable graphs (QLMS).");
    println!();
    println!("USAGE:");
    println!("  qlang supervisor <subcommand> ...");
    println!("  qlang run-agent --state <supervisor.json> --agent <name> [--task <id>]");
    println!("  qlang graph <context|commit|deltas> ...   (worker sync against a running qo)");
    println!("  qlang keys <issue|list|revoke> ...          (manage per-seat access)");
    println!();
    println!("EXAMPLES:");
    println!("  qlang supervisor init --state .qlang/supervisor.json");
    println!("  qlang supervisor add-agent --state .qlang/supervisor.json --name researcher --kind claude-code");
    println!("  qlang supervisor tick --state .qlang/supervisor.json");
    println!("  qlang graph context --kind file --name src/auth.rs");
    println!("  qlang graph commit --file findings.qlang");
}
