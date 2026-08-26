//! `qlang graph` — transport-neutral adapter for the worker sync protocol.
//!
//! The MCP tools cover clients that speak MCP. This covers everything else:
//! a Gemini SDK worker, a shell script, a CI job. Same protocol, same
//! validation, over plain HTTP against a running `qo`.
//!
//! ```text
//! qlang graph context --kind file --name src/auth.rs
//! qlang graph commit --file findings.qlang
//! qlang graph deltas --conflicts-only
//! ```
//!
//! Two things this deliberately does not do, per the integration safeguards:
//! it never reads or forwards a session's prompt history, and it never sends
//! anything but the document it was handed.

use std::io::Read;

/// Where `qo` is listening. The token is read from the environment rather than
/// a flag so it cannot end up in shell history or a process listing.
struct Endpoint {
    base: String,
    token: Option<String>,
}

impl Endpoint {
    fn from_env() -> Self {
        let port = std::env::var("QO_PORT").unwrap_or_else(|_| "4646".to_string());
        Self {
            base: std::env::var("QO_URL")
                .unwrap_or_else(|_| format!("http://127.0.0.1:{port}")),
            token: std::env::var("QO_AUTH_TOKEN").ok().filter(|t| !t.is_empty()),
        }
    }

    /// Attach the bearer token, when one is configured.
    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => req.bearer_auth(token),
            None => req,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

pub async fn handle_graph(args: &[String]) {
    let Some(subcommand) = args.first() else {
        print_usage();
        std::process::exit(2);
    };

    let result = match subcommand.as_str() {
        "context" => context(&args[1..]).await,
        "keygen" => keygen(&args[1..]),
        "sign" => sign(&args[1..]),
        "commit" => commit(&args[1..]).await,
        "deltas" => deltas(&args[1..]).await,
        "export" => export(&args[1..]).await,
        "import" => import(&args[1..]).await,
        "health" => health(&args[1..]).await,
        "backup" => backup(&args[1..]).await,
        "backups" => backups(&args[1..]).await,
        "restore" => restore(&args[1..]).await,
        "events" => events(&args[1..]).await,
        "help" | "--help" | "-h" => {
            print_usage();
            return;
        }
        other => Err(format!("unknown subcommand: {other}")),
    };

    if let Err(message) = result {
        eprintln!("error: {message}");
        std::process::exit(1);
    }
}

/// Generate a producer keypair.
///
/// The seed is private key material: it is written to a file with a warning
/// rather than printed, so it does not end up in a terminal scrollback or a
/// CI log. The public half is printed, because that is what the operator
/// pastes into the trust store.
fn keygen(args: &[String]) -> Result<(), String> {
    let out = flag(args, "--out").ok_or("missing --out <path for the private seed>")?;
    let key_id = flag(args, "--key-id").unwrap_or_else(|| "k1".to_string());

    // Deriving from the OS RNG via the same path the protocol uses.
    let keypair = qlang_core::crypto::Keypair::generate();
    let seed_hex = hex(&keypair.secret_seed());

    // The seed is a private signing key. Write it owner-readable only, and
    // create it atomically so it never briefly exists with default (often
    // world-readable) permissions between creation and a chmod.
    write_private(&out, &format!("{seed_hex}\n"))?;

    println!("private seed written to {out} — keep it secret, never commit it");
    println!();
    println!("Add this to .qlang/trusted_delta_producers.json on the qo host:");
    println!();
    println!("  \"<your-producer-id>\": [");
    println!("    {{");
    println!("      \"key_id\": \"{key_id}\",");
    println!("      \"public_key_hex\": \"{}\",", hex(&keypair.public_key()));
    println!("      \"active_from\": 0");
    println!("    }}");
    println!("  ]");
    Ok(())
}

/// Sign an OrbitQLang document so `qo` will accept it.
fn sign(args: &[String]) -> Result<(), String> {
    let seed_path = flag(args, "--seed").ok_or("missing --seed <path to the private seed>")?;
    let key_id = flag(args, "--key-id").unwrap_or_else(|| "k1".to_string());
    let document = read_document(args)?;

    let seed_hex = std::fs::read_to_string(&seed_path)
        .map_err(|e| format!("cannot read {seed_path}: {e}"))?;
    let seed = parse_seed(seed_hex.trim())?;

    let mut delta = qo_knowledge::from_orbitql(&document)
        .map_err(|e| format!("document does not parse: {e}"))?;
    qo_knowledge::sign_delta(&mut delta, key_id, &seed)
        .map_err(|e| format!("cannot sign: {e}"))?;

    // Straight to stdout so it pipes into `qlang graph commit`.
    print!("{}", qo_knowledge::to_orbitql(&delta));
    Ok(())
}

fn parse_seed(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("seed must be 64 hex chars, found {}", hex.len()));
    }
    let mut out = [0u8; 32];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let value = u8::from_str_radix(
            std::str::from_utf8(pair).map_err(|_| "seed is not valid text")?,
            16,
        )
        .map_err(|_| "seed is not valid hex")?;
        out[i] = value;
    }
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Write private key material to a fresh file readable only by its owner.
///
/// Two properties matter here, both security-relevant:
///
/// - **Atomic create.** `create_new` fails if the path already exists, which
///   both refuses to clobber an existing key and closes the TOCTOU window a
///   separate `exists()` check would leave open.
/// - **Owner-only from creation.** On Unix the file is created `0o600`, so it
///   is never briefly world-readable between creation and a later chmod. On
///   Windows a new file under a user profile inherits that profile's ACL; we
///   additionally tighten it to the current user explicitly.
fn write_private(path: &str, contents: &str) -> Result<(), String> {
    use std::io::Write;

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }

    let mut file = opts.open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::AlreadyExists {
            format!("{path} already exists — refusing to overwrite a private key")
        } else {
            format!("cannot create {path}: {e}")
        }
    })?;

    file.write_all(contents.as_bytes())
        .map_err(|e| format!("cannot write {path}: {e}"))?;

    #[cfg(windows)]
    {
        // Reset inherited ACLs and grant only the current user full control.
        // Best-effort: the file is already outside the world-readable default
        // for a profile directory, so a failure here is a warning, not fatal.
        if let Ok(user) = std::env::var("USERNAME") {
            let _ = std::process::Command::new("icacls")
                .args([path, "/inheritance:r", "/grant:r", &format!("{user}:F")])
                .output();
        }
    }

    Ok(())
}

/// Read a document from `--file`, or stdin when the flag is absent.
fn read_document(args: &[String]) -> Result<String, String> {
    match flag(args, "--file") {
        Some(path) => {
            std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))
        }
        None => {
            let mut buffer = String::new();
            std::io::stdin()
                .read_to_string(&mut buffer)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(buffer)
        }
    }
}

/// Bounded context for a task, before the worker starts.
async fn context(args: &[String]) -> Result<(), String> {
    let kind = flag(args, "--kind").ok_or("missing --kind (e.g. file)")?;
    let name = flag(args, "--name").ok_or("missing --name (e.g. src/auth.rs)")?;
    let limit = flag(args, "--limit").unwrap_or_else(|| "10".to_string());

    let endpoint = Endpoint::from_env();
    let body = serde_json::json!({
        "name": "orbit_graph_context",
        "arguments": { "kind": kind, "name": name, "limit": limit.parse::<u64>().unwrap_or(10) }
    });

    let response = call_mcp(&endpoint, body).await?;
    println!("{response}");
    Ok(())
}

/// Submit an OrbitQLang document. Exits non-zero when the merge reported a
/// conflict, so a CI job can gate on it.
async fn commit(args: &[String]) -> Result<(), String> {
    let document = read_document(args)?;

    if document.trim().is_empty() {
        return Err("empty document (pass --file, or pipe one in)".to_string());
    }

    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.post(endpoint.url("/api/knowledge/delta")))
        .json(&serde_json::json!({ "document": document }))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;

    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());

    if !status.is_success() {
        return Err(format!("qo rejected the delta ({status})"));
    }

    let conflicts = value
        .get("conflicts")
        .and_then(|c| c.as_array())
        .map(|c| c.len())
        .unwrap_or(0);
    if conflicts > 0 {
        // A conflict is not a transport failure, but it does mean the worker's
        // findings were not fully accepted — say so in the exit code so a
        // script does not treat it as success.
        std::process::exit(3);
    }
    Ok(())
}

/// The recent delta feed, optionally only the ones that conflicted.
async fn deltas(args: &[String]) -> Result<(), String> {
    let conflicts_only = args.iter().any(|a| a == "--conflicts-only");
    let limit = flag(args, "--limit").unwrap_or_else(|| "20".to_string());

    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let path = format!(
        "/api/knowledge/deltas?limit={limit}{}",
        if conflicts_only { "&conflicts_only=true" } else { "" }
    );

    let response = endpoint
        .auth(client.get(endpoint.url(&path)))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;
    let value: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
    Ok(())
}

/// Snapshot the whole graph — every revision and counter-evidence — to a file,
/// or to stdout when `--out` is omitted.
async fn export(args: &[String]) -> Result<(), String> {
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.get(endpoint.url("/api/knowledge/export")))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("qo refused the export ({})", response.status()));
    }
    let body = response.text().await.map_err(|e| e.to_string())?;

    match flag(args, "--out") {
        Some(path) => {
            std::fs::write(&path, &body).map_err(|e| format!("cannot write {path}: {e}"))?;
            println!("graph exported to {path}");
        }
        None => print!("{body}"),
    }
    Ok(())
}

/// Restore an archive into the running graph, additively. Reads the archive
/// from `--file`, or from stdin when the flag is absent.
async fn import(args: &[String]) -> Result<(), String> {
    let archive = read_document(args)?;
    if archive.trim().is_empty() {
        return Err("empty archive (pass --file, or pipe one in)".to_string());
    }

    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.post(endpoint.url("/api/knowledge/import")))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(archive)
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());

    if !status.is_success() {
        return Err(format!("qo rejected the archive ({status})"));
    }
    Ok(())
}

/// The operator's one-block answer to "how is my graph doing?": load-bearing
/// count, open proposals, stale, refuted, divergences and entities.
async fn health(_args: &[String]) -> Result<(), String> {
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.get(endpoint.url("/api/knowledge/health")))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("qo refused the health check ({})", response.status()));
    }
    let health: qo_knowledge::GraphHealth = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;
    print!("{}", health.render());
    Ok(())
}

/// Write a timestamped snapshot of the whole graph to the server's backup
/// directory. The schedule is an operator decision (cron) — this is the
/// primitive.
async fn backup(_args: &[String]) -> Result<(), String> {
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.post(endpoint.url("/api/knowledge/backup")))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
    if !status.is_success() {
        return Err(format!("qo refused the backup ({status})"));
    }
    Ok(())
}

/// List the server's existing backups, newest first.
async fn backups(_args: &[String]) -> Result<(), String> {
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.get(endpoint.url("/api/knowledge/backups")))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
    Ok(())
}

/// Recover the graph from a backup: restores the newest backup, or a specific
/// one via `--exported-at <timestamp>`. Additive — never overwrites.
async fn restore(args: &[String]) -> Result<(), String> {
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let body = match flag(args, "--exported-at") {
        Some(ts) => serde_json::json!({
            "exported_at": ts
                .parse::<u64>()
                .map_err(|_| "invalid --exported-at (expected a unix timestamp)")?
        }),
        None => serde_json::json!({}),
    };
    let response = endpoint
        .auth(client.post(endpoint.url("/api/knowledge/restore")))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_default());
    if !status.is_success() {
        return Err(format!("qo refused the restore ({status})"));
    }
    Ok(())
}

/// The recent knowledge-lifecycle events (proposals, verifications, sweeps,
/// refreshes, heals, imports, backups) as a readable stream — the "what just
/// happened to the graph" counterpart to `health`'s "how is it doing".
async fn events(args: &[String]) -> Result<(), String> {
    let limit = flag(args, "--limit").unwrap_or_else(|| "30".to_string());
    let endpoint = Endpoint::from_env();
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.get(endpoint.url(&format!("/api/history?limit={limit}"))))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let events: Vec<serde_json::Value> = response
        .json()
        .await
        .map_err(|e| format!("qo returned a non-JSON response: {e}"))?;

    let mut shown = 0;
    for entry in events.iter().rev() {
        let action = entry.get("action_type").and_then(|a| a.as_str()).unwrap_or("");
        if !action.starts_with("knowledge_") && !action.starts_with("orbit_graph_") {
            continue;
        }
        let ts = entry.get("timestamp").and_then(|t| t.as_u64()).unwrap_or(0);
        let description = entry.get("description").and_then(|d| d.as_str()).unwrap_or("");
        let by = entry.get("details").and_then(|d| d.as_str()).unwrap_or("");
        println!("{ts}  {action:<34} {description}  [{by}]");
        shown += 1;
    }
    if shown == 0 {
        println!("No knowledge events recorded yet.");
    }
    Ok(())
}

async fn call_mcp(endpoint: &Endpoint, params: serde_json::Value) -> Result<String, String> {
    let client = reqwest::Client::new();
    let response = endpoint
        .auth(client.post(endpoint.url("/mcp/v1")))
        .json(&serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": params
        }))
        .send()
        .await
        .map_err(|e| format!("cannot reach qo: {e}"))?;

    let value: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    if let Some(error) = value.get("error") {
        return Err(error.to_string());
    }
    Ok(value
        .pointer("/result/content/0/text")
        .and_then(|t| t.as_str())
        .unwrap_or("(no content)")
        .to_string())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn print_usage() {
    println!("qlang graph — worker sync against a running qo");
    println!();
    println!("USAGE:");
    println!("  qlang graph keygen --out <seed-file> [--key-id <id>]");
    println!("  qlang graph sign --seed <seed-file> [--key-id <id>] [--file <path>]");
    println!("  qlang graph context --kind <kind> --name <name> [--limit N]");
    println!("  qlang graph commit [--file <path>]        (reads stdin if omitted)");
    println!("  qlang graph deltas [--conflicts-only] [--limit N]");
    println!("  qlang graph export [--out <path>]         (backup; stdout if omitted)");
    println!("  qlang graph import [--file <path>]        (restore; stdin if omitted)");
    println!("  qlang graph health                        (operator summary)");
    println!("  qlang graph backup                        (write a timestamped snapshot)");
    println!("  qlang graph backups                       (list backups, newest first)");
    println!("  qlang graph restore [--exported-at <ts>]  (recover from the newest/specific backup)");
    println!("  qlang graph events [--limit N]            (recent knowledge events)");
    println!();
    println!("ENVIRONMENT:");
    println!("  QO_URL          base URL (default http://127.0.0.1:$QO_PORT)");
    println!("  QO_PORT         port when QO_URL is unset (default 4646)");
    println!("  QO_AUTH_TOKEN   bearer token, when qo requires one");
    println!();
    println!("EXIT CODES:");
    println!("  0  success   1  transport or usage error   3  merged with conflicts");
    println!();
    println!("TYPICAL FLOW:");
    println!("  qlang graph sign --seed ~/.qlang/worker.key --file findings.qlang \\");
    println!("    | qlang graph commit");
}
