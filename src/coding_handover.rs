use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use qlang_agent::protocol::{next_msg_id, AgentId, Capability, GraphMessage, MessageIntent};
use qlang_core::graph::Graph;
use qlang_core::ops::Op;
use qlang_core::tensor::{Dim, Shape, TensorData, TensorType};

const QLMS_MAGIC: &[u8; 4] = b"QLMS";
const PAYLOAD_KEY: &str = "coding_handover.payload_json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingHandoverPayload {
    pub schema_version: String,
    pub phase: String,
    pub status: String,
    pub summary: String,
    pub request: String,
    pub next_action: String,
    pub files: Vec<String>,
    pub evidence: Vec<String>,
    pub risks: Vec<String>,
    pub proposed_changes: Vec<String>,
    pub tests: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug)]
pub enum Command {
    Create(HandoverCommand),
    Reply(HandoverCommand),
    Show(ShowCommand),
}

#[derive(Debug)]
pub struct HandoverCommand {
    pub input: Option<PathBuf>,
    pub output: PathBuf,
    pub from: String,
    pub to: String,
    pub phase: String,
    pub status: String,
    pub summary: String,
    pub request: String,
    pub next_action: String,
    pub files: Vec<String>,
    pub evidence: Vec<String>,
    pub risks: Vec<String>,
    pub proposed_changes: Vec<String>,
    pub tests: Vec<String>,
    pub notes: Vec<String>,
    pub pretty: bool,
}

#[derive(Debug)]
pub struct ShowCommand {
    pub input: PathBuf,
    pub pretty: bool,
}

pub enum ParseOutcome {
    Help,
    Message(String),
}

pub fn run() {
    let command = match parse_args(std::env::args().skip(1).collect()) {
        Ok(command) => command,
        Err(ParseOutcome::Help) => {
            print_usage();
            return;
        }
        Err(ParseOutcome::Message(message)) => fail(2, &message),
    };

    match command {
        Command::Create(cmd) => {
            let payload = payload_from_command(&cmd);
            let message = build_message(
                agent_id(&cmd.from),
                agent_id(&cmd.to),
                payload,
                MessageIntent::Compose,
                None,
            );
            write_conversation(&cmd.output, &[message]);
            print_create_result(&cmd.output, cmd.pretty);
        }
        Command::Reply(cmd) => {
            let input = cmd
                .input
                .as_ref()
                .unwrap_or_else(|| fail(2, "--input is required for reply"));
            let mut conversation = read_conversation(input);
            let previous = conversation
                .last()
                .cloned()
                .unwrap_or_else(|| fail(3, "input conversation is empty"));
            let payload = payload_from_command(&cmd);
            let reply = build_message(
                agent_id(&cmd.from),
                agent_id(&cmd.to),
                payload,
                MessageIntent::Result {
                    original_message_id: previous.id,
                },
                Some(previous.id),
            );
            conversation.push(reply);
            write_conversation(&cmd.output, &conversation);
            print_create_result(&cmd.output, cmd.pretty);
        }
        Command::Show(cmd) => {
            let conversation = read_conversation(&cmd.input);
            let rendered = render_conversation(&conversation, cmd.pretty);
            println!("{rendered}");
        }
    }
}

fn main() {
    run();
}

fn parse_args(args: Vec<String>) -> Result<Command, ParseOutcome> {
    let Some(subcommand) = args.first().cloned() else {
        return Err(ParseOutcome::Help);
    };

    match subcommand.as_str() {
        "create" => parse_handover_command(&args[1..], false).map(Command::Create),
        "reply" => parse_handover_command(&args[1..], true).map(Command::Reply),
        "show" => parse_show_command(&args[1..]).map(Command::Show),
        "-h" | "--help" => Err(ParseOutcome::Help),
        other => Err(ParseOutcome::Message(format!("unknown subcommand: {other}"))),
    }
}

fn parse_handover_command(args: &[String], require_input: bool) -> Result<HandoverCommand, ParseOutcome> {
    let mut input = None;
    let mut output = None;
    let mut from = None;
    let mut to = None;
    let mut phase = if require_input { "reply".to_string() } else { "analyze".to_string() };
    let mut status = "open".to_string();
    let mut summary = None;
    let mut request = String::new();
    let mut next_action = String::new();
    let mut files = Vec::new();
    let mut evidence = Vec::new();
    let mut risks = Vec::new();
    let mut proposed_changes = Vec::new();
    let mut tests = Vec::new();
    let mut notes = Vec::new();
    let mut pretty = true;

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(ParseOutcome::Help),
            "--json" => pretty = false,
            "--pretty" => pretty = true,
            "--input" => input = Some(PathBuf::from(require_value(args, &mut idx, "--input")?)),
            "--output" => output = Some(PathBuf::from(require_value(args, &mut idx, "--output")?)),
            "--from" => from = Some(require_value(args, &mut idx, "--from")?.to_string()),
            "--to" => to = Some(require_value(args, &mut idx, "--to")?.to_string()),
            "--phase" => phase = require_value(args, &mut idx, "--phase")?.to_string(),
            "--status" => status = require_value(args, &mut idx, "--status")?.to_string(),
            "--summary" => summary = Some(require_value(args, &mut idx, "--summary")?.to_string()),
            "--request" => request = require_value(args, &mut idx, "--request")?.to_string(),
            "--next-action" => next_action = require_value(args, &mut idx, "--next-action")?.to_string(),
            "--file" => files.push(require_value(args, &mut idx, "--file")?.to_string()),
            "--evidence" => evidence.push(require_value(args, &mut idx, "--evidence")?.to_string()),
            "--risk" => risks.push(require_value(args, &mut idx, "--risk")?.to_string()),
            "--change" => proposed_changes.push(require_value(args, &mut idx, "--change")?.to_string()),
            "--test" => tests.push(require_value(args, &mut idx, "--test")?.to_string()),
            "--note" => notes.push(require_value(args, &mut idx, "--note")?.to_string()),
            value if value.starts_with("--") => {
                return Err(ParseOutcome::Message(format!("unknown option: {value}")));
            }
            value => {
                if request.is_empty() {
                    request = value.to_string();
                } else {
                    request.push(' ');
                    request.push_str(value);
                }
            }
        }
        idx += 1;
    }

    if require_input && input.is_none() {
        return Err(ParseOutcome::Message("--input is required for reply".into()));
    }

    Ok(HandoverCommand {
        input,
        output: output.unwrap_or_else(|| {
            if require_input {
                PathBuf::from("coding-handover-reply.qlms")
            } else {
                PathBuf::from("coding-handover.qlms")
            }
        }),
        from: from.unwrap_or_else(|| "agent-a".to_string()),
        to: to.unwrap_or_else(|| "agent-b".to_string()),
        phase,
        status,
        summary: summary.unwrap_or_else(|| fail(2, "--summary is required")),
        request,
        next_action,
        files,
        evidence,
        risks,
        proposed_changes,
        tests,
        notes,
        pretty,
    })
}

fn parse_show_command(args: &[String]) -> Result<ShowCommand, ParseOutcome> {
    let mut input = None;
    let mut pretty = true;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(ParseOutcome::Help),
            "--json" => pretty = false,
            "--pretty" => pretty = true,
            "--input" => input = Some(PathBuf::from(require_value(args, &mut idx, "--input")?)),
            value if value.starts_with("--") => {
                return Err(ParseOutcome::Message(format!("unknown option: {value}")));
            }
            value => input = Some(PathBuf::from(value)),
        }
        idx += 1;
    }
    Ok(ShowCommand {
        input: input.unwrap_or_else(|| fail(2, "--input is required for show")),
        pretty,
    })
}

fn require_value<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str, ParseOutcome> {
    *idx += 1;
    args.get(*idx)
        .map(|value| value.as_str())
        .ok_or_else(|| ParseOutcome::Message(format!("missing value for {flag}")))
}

fn payload_from_command(cmd: &HandoverCommand) -> CodingHandoverPayload {
    CodingHandoverPayload {
        schema_version: "coding-handover/v1".to_string(),
        phase: cmd.phase.clone(),
        status: cmd.status.clone(),
        summary: cmd.summary.clone(),
        request: cmd.request.clone(),
        next_action: cmd.next_action.clone(),
        files: cmd.files.clone(),
        evidence: cmd.evidence.clone(),
        risks: cmd.risks.clone(),
        proposed_changes: cmd.proposed_changes.clone(),
        tests: cmd.tests.clone(),
        notes: cmd.notes.clone(),
    }
}

fn build_message(
    from: AgentId,
    to: AgentId,
    payload: CodingHandoverPayload,
    intent: MessageIntent,
    in_reply_to: Option<u64>,
) -> GraphMessage {
    let payload_json = serde_json::to_string_pretty(&payload).expect("serialize coding payload");
    let mut inputs = HashMap::new();
    inputs.insert(PAYLOAD_KEY.to_string(), TensorData::from_string(&payload_json));
    inputs.insert("coding_handover.summary".to_string(), TensorData::from_string(&payload.summary));
    inputs.insert("coding_handover.request".to_string(), TensorData::from_string(&payload.request));
    inputs.insert("coding_handover.next_action".to_string(), TensorData::from_string(&payload.next_action));
    inputs.insert(
        "coding_handover.files_json".to_string(),
        TensorData::from_string(&serde_json::to_string(&payload.files).expect("serialize files")),
    );

    GraphMessage {
        id: next_msg_id(),
        from,
        to,
        graph: build_handover_graph(&payload),
        inputs,
        intent,
        in_reply_to,
        signature: None,
        signer_pubkey: None,
        graph_hash: None,
    }
}

fn build_handover_graph(payload: &CodingHandoverPayload) -> Graph {
    let utf8 = TensorType::new(qlang_core::tensor::Dtype::Utf8, Shape(vec![Dim::Dynamic]));
    let mut graph = Graph::new(format!("coding-handover-{}", payload.phase));
    graph.metadata.insert("handover_kind".into(), "coding_context".into());
    graph.metadata.insert("schema_version".into(), payload.schema_version.clone());
    graph.metadata.insert("phase".into(), payload.phase.clone());
    graph.metadata.insert("status".into(), payload.status.clone());
    graph.metadata.insert("summary".into(), payload.summary.clone());
    let input = graph.add_node(
        Op::Input {
            name: PAYLOAD_KEY.to_string(),
        },
        vec![],
        vec![utf8.clone()],
    );
    let output = graph.add_node(
        Op::Output {
            name: "coding_handover.payload_out".to_string(),
        },
        vec![utf8.clone()],
        vec![],
    );
    graph.add_edge(input, 0, output, 0, utf8);
    graph
}

fn agent_id(name: &str) -> AgentId {
    AgentId {
        name: name.to_string(),
        capabilities: vec![Capability::Execute, Capability::Optimize],
    }
}

fn write_conversation(path: &Path, messages: &[GraphMessage]) {
    let json = serde_json::to_vec(messages)
        .unwrap_or_else(|e| fail(3, &format!("failed to serialize conversation: {e}")));
    let mut bytes = Vec::with_capacity(8 + json.len());
    bytes.extend_from_slice(QLMS_MAGIC);
    bytes.extend_from_slice(&(messages.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&json);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                fail(3, &format!("failed to create output directory '{}': {e}", parent.display()))
            });
        }
    }
    fs::write(path, bytes)
        .unwrap_or_else(|e| fail(3, &format!("failed to write '{}': {e}", path.display())));
}

fn read_conversation(path: &Path) -> Vec<GraphMessage> {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| fail(3, &format!("failed to read '{}': {e}", path.display())));
    parse_qlms_messages(&bytes)
        .unwrap_or_else(|e| fail(3, &format!("failed to parse '{}': {e}", path.display())))
}

fn parse_qlms_messages(bytes: &[u8]) -> Result<Vec<GraphMessage>, String> {
    if bytes.len() < 8 {
        return Err("file too short for QLMS envelope".into());
    }
    if &bytes[0..4] != QLMS_MAGIC {
        return Err("invalid QLMS magic".into());
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let payload = &bytes[8..];
    let messages: Vec<GraphMessage> =
        serde_json::from_slice(payload).map_err(|e| format!("invalid QLMS JSON payload: {e}"))?;
    if messages.len() != count {
        return Err(format!(
            "QLMS message count mismatch: header says {count}, payload has {}",
            messages.len()
        ));
    }
    Ok(messages)
}

fn render_conversation(messages: &[GraphMessage], pretty: bool) -> String {
    let rendered = messages
        .iter()
        .enumerate()
        .map(|(idx, msg)| {
            let payload = extract_payload(msg).unwrap_or_else(|err| CodingHandoverPayload {
                schema_version: "coding-handover/v1".into(),
                phase: "unknown".into(),
                status: "invalid".into(),
                summary: format!("payload parse failed: {err}"),
                request: String::new(),
                next_action: String::new(),
                files: Vec::new(),
                evidence: Vec::new(),
                risks: Vec::new(),
                proposed_changes: Vec::new(),
                tests: Vec::new(),
                notes: Vec::new(),
            });
            MessageView {
                index: idx,
                id: msg.id,
                from: msg.from.name.clone(),
                to: msg.to.name.clone(),
                intent: intent_name(&msg.intent),
                in_reply_to: msg.in_reply_to,
                payload,
            }
        })
        .collect::<Vec<_>>();

    if pretty {
        serde_json::to_string_pretty(&rendered).expect("serialize rendered conversation")
    } else {
        serde_json::to_string(&rendered).expect("serialize rendered conversation")
    }
}

#[derive(Serialize)]
struct MessageView {
    index: usize,
    id: u64,
    from: String,
    to: String,
    intent: String,
    in_reply_to: Option<u64>,
    payload: CodingHandoverPayload,
}

fn extract_payload(msg: &GraphMessage) -> Result<CodingHandoverPayload, String> {
    let tensor = msg
        .inputs
        .get(PAYLOAD_KEY)
        .ok_or_else(|| format!("missing payload tensor '{PAYLOAD_KEY}'"))?;
    let json = tensor
        .as_string()
        .ok_or_else(|| "payload tensor is not valid UTF-8".to_string())?;
    serde_json::from_str(&json).map_err(|e| format!("invalid payload JSON: {e}"))
}

fn intent_name(intent: &MessageIntent) -> String {
    match intent {
        MessageIntent::Execute => "execute".into(),
        MessageIntent::Optimize => "optimize".into(),
        MessageIntent::Compress { method } => format!("compress:{method}"),
        MessageIntent::Verify => "verify".into(),
        MessageIntent::Result { original_message_id } => format!("result:{original_message_id}"),
        MessageIntent::Compose => "compose".into(),
        MessageIntent::Train { epochs } => format!("train:{epochs}"),
    }
}

fn print_create_result(path: &Path, pretty: bool) {
    let result = serde_json::json!({
        "written": path.display().to_string(),
        "format": "QLMS",
        "purpose": "coding_handover"
    });
    let rendered = if pretty {
        serde_json::to_string_pretty(&result)
    } else {
        serde_json::to_string(&result)
    }
    .expect("serialize result");
    println!("{rendered}");
}

fn print_usage() {
    eprintln!("coding-handover");
    eprintln!("  Exchange structured coding context between two CLI agents via QLMS files.");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  create   Create a new coding handover conversation");
    eprintln!("  reply    Append a structured reply to an existing QLMS handover");
    eprintln!("  show     Inspect a QLMS handover conversation");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  cargo run --bin coding-handover --no-default-features --offline -- create --from claude --to codex --phase analyze --summary \"Parser panic\" --request \"Inspect the parser crash\" --file crates/qlang-compile/src/parser.rs --evidence \"stacktrace points to parser.rs:118\" --next-action \"Codex should prepare the first patch\" --output handoff.qlms");
    eprintln!("  cargo run --bin coding-handover --no-default-features --offline -- reply --input handoff.qlms --from codex --to claude --phase patch --summary \"Prepared patch plan\" --request \"Parser crash narrowed down\" --change \"Guard empty token stream\" --test \"cargo test -p qlang-compile parser --offline\" --next-action \"Claude should review and message the user\" --output handoff-reply.qlms");
    eprintln!("  cargo run --bin coding-handover --no-default-features --offline -- show --input handoff-reply.qlms");
}

fn fail(code: i32, message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(code);
}
