//! MCP-style tool-use for OrbitQLang server-side agents.
//!
//! Agents call tools via lightweight ReAct-style XML markers in their LLM
//! output, e.g.
//!
//! ```text
//!   <tool name="read_file" path="src/lib.rs"/>
//!   <tool name="write_file" path="output.txt">
//!   content here
//!   </tool>
//!   <tool name="web_fetch" url="https://example.com"/>
//!   <tool name="exec_shell" command="ls -la"/>
//! ```
//!
//! Each call is parsed, executed (via `execute_tool`) and the structured
//! `ToolResult` is fed back into the next LLM round. The agent loop in
//! `lib.rs` caps iterations at 3 to prevent runaway loops.
//!
//! Security model:
//!   * Filesystem: paths are sandboxed by [`workspace::sandbox_resolve`]
//!     (no `..`, no absolute, no drive letter); 64 KB read/write cap.
//!   * Network: HTTPS only; 10 s timeout; 8 KB body cap; HTML tags loosely
//!     stripped before being handed back to the model.
//!   * Shell: hard whitelist of read-only diagnostics (`ls`, `pwd`,
//!     `git status`, `git log -5`, `cargo --version`, `rustc --version`,
//!     `node --version`); 5 s timeout; 4 KB stdout cap.
//!
//! Tools always return a `ToolResult` — `execute_tool` never returns an
//! `Err`, so a failing tool produces `ok=false` plus an `error` string the
//! LLM can read and react to.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::routes::workspace::{sandbox_resolve, strip_workspace_prefix, WORKSPACE_DIRNAME};

/// Per-tool size guards.
const MAX_FILE_BYTES: usize = 64 * 1024;
const MAX_FETCH_BYTES: usize = 8 * 1024;
const MAX_SHELL_OUTPUT_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, String>,
    /// Body content for tools with multi-line payloads (e.g. write_file).
    pub body: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolResult {
    pub call: ToolCall,
    pub ok: bool,
    pub output: String,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Help text injected into agent system prompts
// ---------------------------------------------------------------------------

pub fn available_tools_help() -> String {
    r#"Available tools (call them with <tool name="X" arg="val" />):
  - read_file path="path"           — read a file from the workspace
  - write_file path="path"          — write a file. Body of the tool is the content.
  - web_fetch url="https://..."     — fetch a URL, returns text body (max 8 KB)
  - exec_shell command="ls -la"     — execute a shell command. Restricted to safe commands.

Use these tools when appropriate. Otherwise, just answer in plain text.
Keep tool calls minimal and only when truly needed."#
        .to_string()
}

// ---------------------------------------------------------------------------
// Parser — extracts <tool .../> and <tool ...>body</tool> markers.
// ---------------------------------------------------------------------------

/// Parse all `<tool name="..." key="val" .../>` and
/// `<tool name="..." ...>body</tool>` markers from the given text.
///
/// Implementation is a hand-rolled scanner — the grammar is small enough
/// that pulling in a full XML parser (or `regex`) is overkill, and the
/// project's "no new crate deps" rule applies.
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        // Find the next `<tool` token.
        let Some(start_rel) = find_subslice(&bytes[i..], b"<tool") else {
            break;
        };
        let tag_start = i + start_rel;
        // Must be followed by whitespace or `/` or `>` to be `<tool`,
        // not `<toolkit>` or similar.
        let after = tag_start + 5;
        if after >= bytes.len() {
            break;
        }
        let next = bytes[after];
        if !(next == b' ' || next == b'\t' || next == b'\n' || next == b'\r' || next == b'/' || next == b'>') {
            i = after;
            continue;
        }

        // Find the closing `>` of the opening tag.
        let Some(close_rel) = find_byte(&bytes[after..], b'>') else {
            break;
        };
        let tag_close = after + close_rel; // index of `>`
        let inner = &text[after..tag_close]; // attributes (and possible trailing `/`)
        let self_closing = inner.trim_end().ends_with('/');
        let attrs_str = if self_closing {
            inner.trim_end().trim_end_matches('/').trim()
        } else {
            inner.trim()
        };

        let attrs = parse_attrs(attrs_str);
        let Some(name) = attrs.get("name").cloned() else {
            // Skip a malformed tool tag without a name.
            i = tag_close + 1;
            continue;
        };

        let mut args = attrs;
        args.remove("name");

        if self_closing {
            out.push(ToolCall { name, args, body: None });
            i = tag_close + 1;
            continue;
        }

        // Body form — scan for the matching </tool>.
        let body_start = tag_close + 1;
        let Some(end_rel) = find_subslice(&bytes[body_start..], b"</tool>") else {
            // Unterminated tag — treat as self-closing with no body.
            out.push(ToolCall { name, args, body: None });
            i = tag_close + 1;
            continue;
        };
        let body_end = body_start + end_rel;
        let body = text[body_start..body_end].trim_matches(|c: char| c == '\n' || c == '\r').to_string();
        out.push(ToolCall {
            name,
            args,
            body: if body.is_empty() { None } else { Some(body) },
        });
        i = body_end + b"</tool>".len();
    }

    out
}

/// Parse a string of `key="value"` (or `key='value'`) pairs into a map.
/// Tolerates extra whitespace and ignores malformed fragments.
fn parse_attrs(s: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Skip whitespace.
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n' || bytes[i] == b'\r') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Read key — letters, digits, `_`, `-`.
        let key_start = i;
        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-') {
            i += 1;
        }
        if key_start == i {
            // No progress — bail to avoid an infinite loop.
            i += 1;
            continue;
        }
        let key = std::str::from_utf8(&bytes[key_start..i]).unwrap_or("").to_string();
        // Expect `=`.
        if i >= bytes.len() || bytes[i] != b'=' {
            continue;
        }
        i += 1;
        // Expect quote.
        if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
            continue;
        }
        let quote = bytes[i];
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            i += 1;
        }
        let val = std::str::from_utf8(&bytes[val_start..i]).unwrap_or("").to_string();
        if i < bytes.len() {
            i += 1; // skip closing quote
        }
        if !key.is_empty() {
            out.insert(key, val);
        }
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/// Execute a single tool call. The result is **always** a `ToolResult` —
/// errors are wrapped (`ok = false`, `error = Some(...)`) so the agent
/// loop never has to handle a Rust `Err` from this function.
pub async fn execute_tool(call: ToolCall) -> ToolResult {
    match call.name.as_str() {
        "read_file" => exec_read_file(call).await,
        "write_file" => exec_write_file(call).await,
        "web_fetch" => exec_web_fetch(call).await,
        "exec_shell" => exec_exec_shell(call).await,
        other => ToolResult {
            call: ToolCall {
                name: other.to_string(),
                args: HashMap::new(),
                body: None,
            },
            ok: false,
            output: String::new(),
            error: Some(format!("unknown tool: {other}")),
        },
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

fn workspace_root_for_tools() -> PathBuf {
    let data_dir = std::env::var("QO_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("data"));
    data_dir.join(WORKSPACE_DIRNAME)
}

async fn exec_read_file(call: ToolCall) -> ToolResult {
    let Some(path) = call.args.get("path").cloned() else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("missing required arg: path".into()),
        };
    };
    let root = workspace_root_for_tools();
    let normalised = strip_workspace_prefix(&path);
    let Some(target) = sandbox_resolve(&root, &normalised) else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("path {path:?} escapes the workspace sandbox")),
        };
    };
    if !target.exists() || !target.is_file() {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("file not found: {normalised}")),
        };
    }
    match std::fs::read(&target) {
        Ok(bytes) => {
            let truncated = bytes.len() > MAX_FILE_BYTES;
            let slice = if truncated { &bytes[..MAX_FILE_BYTES] } else { &bytes[..] };
            let mut text = String::from_utf8_lossy(slice).into_owned();
            if truncated {
                text.push_str(&format!(
                    "\n\n... [truncated at {MAX_FILE_BYTES} bytes — file is {} bytes]",
                    bytes.len()
                ));
            }
            ToolResult {
                call,
                ok: true,
                output: text,
                error: None,
            }
        }
        Err(e) => ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("read failed: {e}")),
        },
    }
}

async fn exec_write_file(call: ToolCall) -> ToolResult {
    let Some(path) = call.args.get("path").cloned() else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("missing required arg: path".into()),
        };
    };
    let Some(body) = call.body.clone() else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("write_file requires a body (content between <tool>...</tool>)".into()),
        };
    };
    if body.len() > MAX_FILE_BYTES {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!(
                "body too large: {} bytes (max {})",
                body.len(),
                MAX_FILE_BYTES
            )),
        };
    }
    let root = workspace_root_for_tools();
    let normalised = strip_workspace_prefix(&path);
    let Some(target) = sandbox_resolve(&root, &normalised) else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("path {path:?} escapes the workspace sandbox")),
        };
    };
    if let Some(parent) = target.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("mkdir parent failed: {e}")),
            };
        }
    }
    match std::fs::write(&target, body.as_bytes()) {
        Ok(_) => ToolResult {
            call,
            ok: true,
            output: format!("wrote {} bytes to {}", body.len(), normalised),
            error: None,
        },
        Err(e) => ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("write failed: {e}")),
        },
    }
}

async fn exec_web_fetch(call: ToolCall) -> ToolResult {
    let Some(url) = call.args.get("url").cloned() else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("missing required arg: url".into()),
        };
    };
    if !url.starts_with("https://") {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("only https:// URLs are allowed".into()),
        };
    }
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("client build failed: {e}")),
            };
        }
    };
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("fetch failed: {e}")),
            };
        }
    };
    let status = resp.status();
    let body = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("body read failed: {e}")),
            };
        }
    };
    if !status.is_success() {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!("HTTP {}", status.as_u16())),
        };
    }
    let stripped = strip_html(&body);
    let truncated = stripped.len() > MAX_FETCH_BYTES;
    let mut out = if truncated {
        stripped.chars().take(MAX_FETCH_BYTES).collect::<String>()
    } else {
        stripped
    };
    if truncated {
        out.push_str(&format!("\n\n... [truncated at {MAX_FETCH_BYTES} bytes]"));
    }
    ToolResult {
        call,
        ok: true,
        output: out,
        error: None,
    }
}

/// Loose HTML stripper — drops anything between `<` and `>`. Good enough
/// for feeding plaintext into an LLM; not a real sanitiser.
fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

async fn exec_exec_shell(call: ToolCall) -> ToolResult {
    let Some(command) = call.args.get("command").cloned() else {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some("missing required arg: command".into()),
        };
    };
    let trimmed = command.trim();
    if !is_whitelisted_command(trimmed) {
        return ToolResult {
            call,
            ok: false,
            output: String::new(),
            error: Some(format!(
                "command {trimmed:?} is not on the whitelist (allowed: ls, pwd, git status, git log -5, cargo --version, rustc --version, node --version)"
            )),
        };
    }
    // Split on whitespace — the whitelist is restrictive enough that
    // shell escaping isn't a concern.
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    let (prog, args) = match parts.split_first() {
        Some(s) => s,
        None => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some("empty command".into()),
            };
        }
    };

    let mut cmd = tokio::process::Command::new(prog);
    cmd.args(args);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("spawn failed: {e}")),
            };
        }
    };
    let output = match tokio::time::timeout(Duration::from_secs(5), child.wait_with_output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some(format!("wait failed: {e}")),
            };
        }
        Err(_) => {
            return ToolResult {
                call,
                ok: false,
                output: String::new(),
                error: Some("command timed out after 5s".into()),
            };
        }
    };
    let exit = output.status.code().unwrap_or(-1);
    let mut stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    if stdout.len() > MAX_SHELL_OUTPUT_BYTES {
        stdout.truncate(MAX_SHELL_OUTPUT_BYTES);
        stdout.push_str("\n... [truncated]");
    }
    if exit != 0 {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return ToolResult {
            call,
            ok: false,
            output: stdout,
            error: Some(format!("exit {exit}: {}", stderr.lines().next().unwrap_or("").trim())),
        };
    }
    ToolResult {
        call,
        ok: true,
        output: stdout,
        error: None,
    }
}

/// Hard whitelist of safe, read-only diagnostics. Anything else returns
/// an error from `exec_exec_shell`.
fn is_whitelisted_command(cmd: &str) -> bool {
    matches!(
        cmd,
        "ls"
            | "ls -la"
            | "ls -l"
            | "ls -a"
            | "pwd"
            | "git status"
            | "git log -5"
            | "cargo --version"
            | "rustc --version"
            | "node --version"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_self_closing_single() {
        let calls = parse_tool_calls("hello <tool name=\"read_file\" path=\"src/lib.rs\"/> world");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].args.get("path").map(String::as_str), Some("src/lib.rs"));
        assert!(calls[0].body.is_none());
    }

    #[test]
    fn parse_with_body() {
        let txt = "<tool name=\"write_file\" path=\"out.txt\">\nhello\nworld\n</tool>";
        let calls = parse_tool_calls(txt);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].body.as_deref(), Some("hello\nworld"));
    }

    #[test]
    fn parse_multiple() {
        let txt = "<tool name=\"read_file\" path=\"a\"/> and <tool name=\"web_fetch\" url=\"https://x\"/>";
        let calls = parse_tool_calls(txt);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[1].name, "web_fetch");
    }

    #[test]
    fn parse_ignores_text_without_tools() {
        assert!(parse_tool_calls("just a normal LLM reply with no tools").is_empty());
    }

    #[test]
    fn parse_ignores_lookalike_tags() {
        // `<toolkit>` should NOT be matched as a tool tag.
        assert!(parse_tool_calls("<toolkit name=\"x\"/>").is_empty());
    }

    #[test]
    fn web_fetch_rejects_http() {
        let call = ToolCall {
            name: "web_fetch".to_string(),
            args: [("url".to_string(), "http://example.com".to_string())].into(),
            body: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(exec_web_fetch(call));
        assert!(!result.ok);
        assert!(result.error.unwrap().contains("https"));
    }

    #[test]
    fn exec_shell_rejects_arbitrary() {
        let call = ToolCall {
            name: "exec_shell".to_string(),
            args: [("command".to_string(), "rm -rf /".to_string())].into(),
            body: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(exec_exec_shell(call));
        assert!(!result.ok);
    }

    #[test]
    fn unknown_tool_yields_error_result() {
        let call = ToolCall {
            name: "delete_database".to_string(),
            args: HashMap::new(),
            body: None,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(execute_tool(call));
        assert!(!result.ok);
        assert!(result.error.unwrap().contains("unknown tool"));
    }
}
