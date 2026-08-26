//! `qlang keys` — issue, list and revoke per-seat API keys.
//!
//! This is how a team admin hands out and takes back access without touching
//! JSON by hand or rotating one shared token for everyone. It edits
//! `.qlang/api_keys.json`, the same file the server loads at startup.
//!
//! The generated secret is printed once, here, and never again — the server
//! stores it and compares in constant time, but this is the only moment the
//! plaintext exists to be copied to the seat holder.

use qo_server::api_keys::{ApiKeyStore, Role};
use qlang_core::crypto::Keypair;

const STORE_PATH: &str = ".qlang/api_keys.json";

pub fn handle_keys(args: &[String]) {
    let Some(sub) = args.first() else {
        print_usage();
        std::process::exit(2);
    };

    let result = match sub.as_str() {
        "issue" => issue(&args[1..]),
        "list" => list(),
        "revoke" => revoke(&args[1..]),
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

fn load() -> Result<ApiKeyStore, String> {
    match std::fs::read_to_string(STORE_PATH) {
        Ok(contents) => ApiKeyStore::from_json(&contents)
            .map_err(|e| format!("{STORE_PATH} is malformed: {e}")),
        Err(_) => Ok(ApiKeyStore::new()),
    }
}

fn save(store: &ApiKeyStore) -> Result<(), String> {
    if let Some(parent) = std::path::Path::new(STORE_PATH).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cannot create .qlang/: {e}"))?;
    }
    let json = store.to_json().map_err(|e| e.to_string())?;

    // The store holds bearer secrets, so it must not be world-readable. Unlike
    // a fresh keypair this file is rewritten in place, so `create_new` does not
    // apply — instead the permissions are set explicitly, before the write on
    // Unix (so the bytes never land under a looser mode) and after on Windows.
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(STORE_PATH)
            .map_err(|e| format!("cannot write {STORE_PATH}: {e}"))?;
        // An existing file keeps its old mode on open; enforce 0o600 regardless.
        std::fs::set_permissions(STORE_PATH, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .map_err(|e| format!("cannot secure {STORE_PATH}: {e}"))?;
        return file
            .write_all(json.as_bytes())
            .map_err(|e| format!("cannot write {STORE_PATH}: {e}"));
    }

    #[cfg(not(unix))]
    {
        std::fs::write(STORE_PATH, json)
            .map_err(|e| format!("cannot write {STORE_PATH}: {e}"))?;
        #[cfg(windows)]
        if let Ok(user) = std::env::var("USERNAME") {
            let _ = std::process::Command::new("icacls")
                .args([STORE_PATH, "/inheritance:r", "/grant:r", &format!("{user}:F")])
                .output();
        }
        Ok(())
    }
}

fn issue(args: &[String]) -> Result<(), String> {
    let label = flag(args, "--label").ok_or("missing --label <name for this seat>")?;
    let role = match flag(args, "--role").as_deref() {
        None | Some("member") => Role::Member,
        Some("admin") => Role::Admin,
        Some("viewer") => Role::Viewer,
        Some(other) => return Err(format!("unknown role {other:?}; use member|admin|viewer")),
    };

    // A fresh 32-byte secret from the same RNG the protocol trusts, rendered
    // as hex so it copies cleanly into a header.
    let secret = hex(&Keypair::generate().secret_seed());

    let mut store = load()?;
    store.issue(&label, &secret, role);
    save(&store)?;

    println!("Issued a {} key for \"{label}\".", role.as_str());
    println!("  {}", role_blurb(role));
    println!();
    println!("Give this to the seat holder — it is shown once and not stored in the clear:");
    println!();
    println!("  {secret}");
    println!();
    println!("They authenticate with:  Authorization: Bearer {secret}");
    println!("Active seats: {}", store.active_seats());
    Ok(())
}

/// What a role is allowed to do, in one line — surfaced at issue time so an
/// admin sees exactly what they just granted, and in `--help`.
fn role_blurb(role: Role) -> &'static str {
    match role {
        Role::Member => "read + write the graph (deltas, verification, workspace tools).",
        Role::Admin => "everything a member can, plus providers, git, scheduler and supervisor config.",
        Role::Viewer => "read-only: inspect the graph and read endpoints; cannot change anything.",
    }
}

fn list() -> Result<(), String> {
    let store = load()?;
    if store.keys.is_empty() {
        println!("No API keys issued. `qlang keys issue --label <name>` to add one.");
        return Ok(());
    }
    println!("{:<24} {:<8} {}", "LABEL", "ROLE", "STATUS");
    for key in &store.keys {
        println!(
            "{:<24} {:<8} {}",
            key.label,
            key.role.as_str(),
            if key.revoked { "revoked" } else { "active" }
        );
    }
    println!("\nActive seats: {}", store.active_seats());
    Ok(())
}

fn revoke(args: &[String]) -> Result<(), String> {
    let label = flag(args, "--label").ok_or("missing --label <name to revoke>")?;
    let mut store = load()?;

    let mut hit = false;
    for key in &mut store.keys {
        if key.label == label && !key.revoked {
            key.revoked = true;
            hit = true;
        }
    }
    if !hit {
        return Err(format!("no active key labelled {label:?}"));
    }
    save(&store)?;
    println!("Revoked \"{label}\". Active seats: {}", store.active_seats());
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn print_usage() {
    println!("qlang keys — manage per-seat API keys (.qlang/api_keys.json)");
    println!();
    println!("USAGE:");
    println!("  qlang keys issue --label <name> [--role member|admin|viewer]");
    println!("  qlang keys list");
    println!("  qlang keys revoke --label <name>");
    println!();
    println!("ROLES:");
    println!("  member   {}", role_blurb(Role::Member));
    println!("  admin    {}", role_blurb(Role::Admin));
    println!("  viewer   {}", role_blurb(Role::Viewer));
    println!();
    println!("The secret is printed once at issue time and never again.");
}
