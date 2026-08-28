//! Harness registry — which coding systems are actually attached to QO.
//!
//! # What a "harness" is here
//!
//! A harness is an external coding system that drives QO's tools: Claude Code,
//! Codex, Gemini CLI, deepseek-harness, or anything else that speaks MCP. It is
//! *not* an LLM provider. The distinction matters and is the reason this module
//! exists separately from [`crate::routes::providers`]:
//!
//! - a **provider** is a model endpoint QO calls *outward* (DeepSeek, Ollama…),
//! - a **harness** is a client that calls *into* QO and executes its tools.
//!
//! The cockpit needs the second view — "who is driving this graph right now" —
//! and until now nothing recorded it. The MCP `initialize` handshake carries
//! `clientInfo.name`, and it was being discarded.
//!
//! # Why registration is passive
//!
//! A harness is recorded by *using* QO, not by being configured. Claude Code,
//! Codex and Gemini CLI have no notion of "register with your control plane";
//! they open an MCP session and start calling tools. So the registry is fed
//! from the handshake and from every tool call, which means the cockpit shows
//! what is true rather than what someone declared.
//!
//! # Why unknown clients are first-class
//!
//! [`HarnessKind::classify`] recognises the four systems above by name, but an
//! unrecognised client is recorded as [`HarnessKind::Other`] with its reported
//! name preserved — never dropped. That is the extension point: a new or
//! in-house harness (an extended deepseek-harness, a bespoke agent runner)
//! appears in the cockpit the first time it calls, with no code change here.
//! Adding a variant to `classify` only improves its label and icon.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// How long after its last call a harness is still considered attached.
///
/// Longer than the presence TTL (60s) on purpose: an MCP client is not
/// required to heartbeat, and a coding agent is legitimately idle between
/// tasks. Treating a two-minute pause as a disconnect would make the cockpit
/// flicker during normal work.
pub const HARNESS_TTL_SECS: u64 = 900;

/// Cap on distinct harness sessions retained. Bounds memory against a client
/// that reports a fresh name on every connection.
const MAX_SESSIONS: usize = 64;

/// A coding system that drives QO's tools.
///
/// `Other` is not a failure case — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarnessKind {
    ClaudeCode,
    Codex,
    Gemini,
    DeepseekHarness,
    Other,
}

impl HarnessKind {
    /// Classify a client by the name it reports in the MCP handshake.
    ///
    /// Matching is substring-based and case-insensitive because clients are
    /// inconsistent: "claude-code", "Claude Code", "claude_code_cli" all occur.
    /// An unrecognised name is `Other`, never rejected.
    pub fn classify(client_name: &str) -> Self {
        let name = client_name.to_ascii_lowercase().replace(['_', ' '], "-");
        // deepseek-harness is checked before the bare provider names so an
        // extended harness reporting "deepseek-harness-v2" classifies as a
        // harness rather than falling through to Other.
        if name.contains("deepseek") && name.contains("harness") {
            return Self::DeepseekHarness;
        }
        if name.contains("claude") {
            return Self::ClaudeCode;
        }
        if name.contains("codex") {
            return Self::Codex;
        }
        if name.contains("gemini") {
            return Self::Gemini;
        }
        if name.contains("deepseek") {
            return Self::DeepseekHarness;
        }
        Self::Other
    }

    /// Stable display label for the cockpit.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Codex => "Codex",
            Self::Gemini => "Gemini",
            Self::DeepseekHarness => "DeepSeek Harness",
            Self::Other => "Other MCP client",
        }
    }
}

/// One attached coding system, as the cockpit sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessSession {
    /// Stable key: the client name exactly as reported.
    pub id: String,
    pub kind: HarnessKind,
    /// Display label — the recognised product name, or the raw reported name
    /// for an unknown client so it is identifiable rather than "Other".
    pub label: String,
    /// Version string from `clientInfo.version`, when the client sends one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Unix seconds of the MCP handshake.
    pub connected_at: u64,
    /// Unix seconds of the most recent tool call (or the handshake).
    pub last_seen_at: u64,
    /// Total tool calls in this session — the honest activity signal.
    pub calls: u64,
    /// Distinct tools used, most recent first, capped for display.
    pub recent_tools: Vec<String>,
    /// False once `last_seen_at` is older than [`HARNESS_TTL_SECS`].
    pub online: bool,
}

/// Recent tool names kept per session.
const RECENT_TOOLS: usize = 8;

/// In-memory registry of attached harnesses.
///
/// Deliberately not persisted: an attachment is a live fact, and a restart of
/// `qo` must not resurrect a coding agent that is no longer there.
#[derive(Debug, Default)]
pub struct HarnessRegistry {
    sessions: Mutex<HashMap<String, HarnessSession>>,
}

impl HarnessRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an MCP handshake. Re-connecting an existing client refreshes it
    /// rather than resetting its call count, so the cockpit does not lose a
    /// session's history when a client reconnects mid-task.
    pub fn record_handshake(&self, client_name: &str, version: Option<String>) {
        let name = client_name.trim();
        if name.is_empty() {
            return;
        }
        let now = unix_now();
        let kind = HarnessKind::classify(name);
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(existing) = sessions.get_mut(name) {
            existing.last_seen_at = now;
            existing.online = true;
            if version.is_some() {
                existing.version = version;
            }
            return;
        }

        Self::evict_if_full(&mut sessions);
        sessions.insert(
            name.to_string(),
            HarnessSession {
                id: name.to_string(),
                kind,
                // An unknown client keeps its own name; a recognised one gets
                // the canonical product label.
                label: match kind {
                    HarnessKind::Other => name.to_string(),
                    known => known.label().to_string(),
                },
                version,
                connected_at: now,
                last_seen_at: now,
                calls: 0,
                recent_tools: Vec::new(),
                online: true,
            },
        );
    }

    /// Record a tool call against a client.
    ///
    /// A client that calls tools without a handshake (permitted by MCP over
    /// plain HTTP) is registered on the spot, so activity is never invisible.
    pub fn record_call(&self, client_name: &str, tool: &str) {
        let name = client_name.trim();
        if name.is_empty() {
            return;
        }
        let now = unix_now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());

        let entry = match sessions.get_mut(name) {
            Some(entry) => entry,
            None => {
                Self::evict_if_full(&mut sessions);
                let kind = HarnessKind::classify(name);
                sessions.entry(name.to_string()).or_insert(HarnessSession {
                    id: name.to_string(),
                    kind,
                    label: match kind {
                        HarnessKind::Other => name.to_string(),
                        known => known.label().to_string(),
                    },
                    version: None,
                    connected_at: now,
                    last_seen_at: now,
                    calls: 0,
                    recent_tools: Vec::new(),
                    online: true,
                })
            }
        };

        entry.last_seen_at = now;
        entry.online = true;
        entry.calls = entry.calls.saturating_add(1);
        // Most-recent-first, de-duplicated: the useful question is "what has
        // this harness been touching", not a raw call log.
        entry.recent_tools.retain(|t| t != tool);
        entry.recent_tools.insert(0, tool.to_string());
        entry.recent_tools.truncate(RECENT_TOOLS);
    }

    /// Every known session, online first, then by most recent activity.
    pub fn list(&self) -> Vec<HarnessSession> {
        let now = unix_now();
        let sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<HarnessSession> = sessions
            .values()
            .cloned()
            .map(|mut s| {
                s.online = now.saturating_sub(s.last_seen_at) < HARNESS_TTL_SECS;
                s
            })
            .collect();
        out.sort_by(|a, b| {
            b.online
                .cmp(&a.online)
                .then(b.last_seen_at.cmp(&a.last_seen_at))
                .then(a.id.cmp(&b.id))
        });
        out
    }

    /// Drop the least recently seen session when at capacity.
    fn evict_if_full(sessions: &mut HashMap<String, HarnessSession>) {
        if sessions.len() < MAX_SESSIONS {
            return;
        }
        if let Some(oldest) = sessions
            .values()
            .min_by_key(|s| s.last_seen_at)
            .map(|s| s.id.clone())
        {
            sessions.remove(&oldest);
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

use axum::{extract::State, Json};
use std::sync::Arc;

use crate::AppState;

/// What the cockpit's integration view renders.
#[derive(Debug, Serialize)]
pub struct HarnessOverview {
    /// Every harness seen since this server started.
    pub sessions: Vec<HarnessSession>,
    /// How many are currently within the TTL.
    pub online: usize,
    /// The MCP endpoint a harness must be pointed at.
    pub mcp_endpoint: String,
    /// Number of tools exposed over MCP — what a harness gains by attaching.
    pub tools: usize,
    /// Harness kinds QO recognises by name. The cockpit uses this to show
    /// systems that are supported but not currently attached, so the operator
    /// sees what *could* connect, not only what has.
    pub known_kinds: Vec<KnownKind>,
}

#[derive(Debug, Serialize)]
pub struct KnownKind {
    pub kind: HarnessKind,
    pub label: String,
    /// True when a session of this kind is currently online.
    pub attached: bool,
}

/// `GET /api/harness` — the integration overview.
pub async fn list_harnesses(State(state): State<Arc<AppState>>) -> Json<HarnessOverview> {
    let sessions = state.harnesses.list();
    let online = sessions.iter().filter(|s| s.online).count();

    let known_kinds = [
        HarnessKind::ClaudeCode,
        HarnessKind::Codex,
        HarnessKind::Gemini,
        HarnessKind::DeepseekHarness,
    ]
    .into_iter()
    .map(|kind| KnownKind {
        kind,
        label: kind.label().to_string(),
        attached: sessions.iter().any(|s| s.online && s.kind == kind),
    })
    .collect();

    Json(HarnessOverview {
        tools: crate::routes::mcp_server::tool_count(),
        sessions,
        online,
        mcp_endpoint: "/mcp/v1".to_string(),
        known_kinds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_the_four_target_systems() {
        assert_eq!(HarnessKind::classify("claude-code"), HarnessKind::ClaudeCode);
        assert_eq!(HarnessKind::classify("Claude Code"), HarnessKind::ClaudeCode);
        assert_eq!(HarnessKind::classify("codex-cli"), HarnessKind::Codex);
        assert_eq!(HarnessKind::classify("gemini-cli"), HarnessKind::Gemini);
        assert_eq!(
            HarnessKind::classify("deepseek-harness"),
            HarnessKind::DeepseekHarness
        );
    }

    #[test]
    fn an_extended_deepseek_harness_still_classifies_as_a_harness() {
        // The whole point of the extension story: a modified harness must not
        // fall out of its category because someone appended a suffix.
        assert_eq!(
            HarnessKind::classify("deepseek_harness_v2"),
            HarnessKind::DeepseekHarness
        );
        assert_eq!(
            HarnessKind::classify("my-deepseek-harness-fork"),
            HarnessKind::DeepseekHarness
        );
    }

    #[test]
    fn an_unknown_client_is_kept_under_its_own_name() {
        let registry = HarnessRegistry::new();
        registry.record_handshake("acme-agent-runner", None);
        let list = registry.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].kind, HarnessKind::Other);
        // Not "Other MCP client" — the operator must be able to tell which
        // unknown client this is.
        assert_eq!(list[0].label, "acme-agent-runner");
    }

    #[test]
    fn tool_calls_accumulate_and_deduplicate_recent_tools() {
        let registry = HarnessRegistry::new();
        registry.record_handshake("claude-code", Some("1.2.3".into()));
        registry.record_call("claude-code", "orbit_graph_context");
        registry.record_call("claude-code", "orbit_graph_propose");
        registry.record_call("claude-code", "orbit_graph_context");

        let list = registry.list();
        assert_eq!(list[0].calls, 3);
        // Re-using a tool moves it to the front rather than duplicating it.
        assert_eq!(
            list[0].recent_tools,
            vec!["orbit_graph_context", "orbit_graph_propose"]
        );
        assert_eq!(list[0].version.as_deref(), Some("1.2.3"));
    }

    #[test]
    fn a_call_without_a_handshake_still_registers() {
        let registry = HarnessRegistry::new();
        registry.record_call("codex-cli", "orbit_graph_health");
        let list = registry.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].kind, HarnessKind::Codex);
        assert_eq!(list[0].calls, 1);
    }

    #[test]
    fn reconnecting_preserves_the_call_count() {
        let registry = HarnessRegistry::new();
        registry.record_handshake("gemini-cli", None);
        registry.record_call("gemini-cli", "orbit_graph_search");
        registry.record_handshake("gemini-cli", Some("2.0".into()));

        let list = registry.list();
        assert_eq!(list[0].calls, 1, "a reconnect must not reset history");
        assert_eq!(list[0].version.as_deref(), Some("2.0"));
    }

    #[test]
    fn empty_client_names_are_ignored() {
        let registry = HarnessRegistry::new();
        registry.record_handshake("   ", None);
        registry.record_call("", "orbit_graph_health");
        assert!(registry.list().is_empty());
    }

    #[test]
    fn the_registry_is_bounded() {
        let registry = HarnessRegistry::new();
        for i in 0..(MAX_SESSIONS + 20) {
            registry.record_handshake(&format!("client-{i}"), None);
        }
        assert!(
            registry.list().len() <= MAX_SESSIONS,
            "registry must not grow without bound"
        );
    }

    #[test]
    fn online_sessions_sort_before_idle_ones() {
        let registry = HarnessRegistry::new();
        registry.record_handshake("claude-code", None);
        registry.record_handshake("codex-cli", None);
        {
            // Age one session past the TTL by hand.
            let mut sessions = registry.sessions.lock().unwrap();
            let entry = sessions.get_mut("claude-code").unwrap();
            entry.last_seen_at = unix_now() - HARNESS_TTL_SECS - 1;
        }
        let list = registry.list();
        assert_eq!(list[0].id, "codex-cli", "the live session comes first");
        assert!(!list.iter().find(|s| s.id == "claude-code").unwrap().online);
    }
}
