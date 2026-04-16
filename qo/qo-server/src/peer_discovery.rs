//! Plug-and-play peer discovery for a multi-node QO swarm (PRD Task 4.2).
//!
//! When the environment variable `PEER_DISCOVERY_SEEDS` is set, a
//! background tokio task invokes this node's own
//! `POST /api/qlms/federation/gossip` endpoint every
//! `PEER_DISCOVERY_INTERVAL_SECS` (default 10 s). The endpoint does the
//! actual legwork — fetching each peer's ternary weights, running the
//! majority-vote merge, and updating this node's specialist.
//!
//! Design choices:
//!
//! * We deliberately call the **loopback HTTP** endpoint rather than
//!   invoking `gossip()` directly. This keeps the gossip path identical
//!   to the one a human operator would use from `curl` and lets the
//!   endpoint's auth middleware (when enabled) run uniformly.
//! * Seeds are the full peer list, **including this node's own
//!   address**; the filter below strips it out so a node does not
//!   gossip with itself.
//! * Errors are logged, never panicked. A restart-loop-proof agent.

use std::env;
use std::time::Duration;

use serde::Serialize;
use tracing::{info, warn};

/// Env var holding a comma-separated peer list, e.g.:
/// `PEER_DISCOVERY_SEEDS=qo-a:4646,qo-b:4747,qo-c:4848`.
pub const ENV_SEEDS: &str = "PEER_DISCOVERY_SEEDS";
/// Optional self-identifier, e.g. `QO_NODE_ADDR=qo-a:4646`. When set,
/// this exact string is filtered from the peer list so a node never
/// gossips with itself.
pub const ENV_SELF: &str = "QO_NODE_ADDR";
/// Interval between gossip rounds, in seconds.
pub const ENV_INTERVAL: &str = "PEER_DISCOVERY_INTERVAL_SECS";
const DEFAULT_INTERVAL_SECS: u64 = 10;

/// Parse the seed list from `PEER_DISCOVERY_SEEDS`.
///
/// Returns `None` if the env var is unset or empty after trimming.
/// Otherwise returns all seeds **except** the one matching `QO_NODE_ADDR`
/// (if set), de-duplicated while preserving input order.
pub fn parse_seeds_from_env() -> Option<Vec<String>> {
    let raw = env::var(ENV_SEEDS).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let self_addr = env::var(ENV_SELF).ok();
    let peers = filter_self(split_seeds(trimmed), self_addr.as_deref());
    if peers.is_empty() {
        None
    } else {
        Some(peers)
    }
}

fn split_seeds(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

fn filter_self(peers: Vec<String>, self_addr: Option<&str>) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(peers.len());
    for p in peers {
        if let Some(me) = self_addr {
            if p == me {
                continue;
            }
        }
        if !out.iter().any(|existing| existing == &p) {
            out.push(p);
        }
    }
    out
}

fn interval_from_env() -> Duration {
    Duration::from_secs(
        env::var(ENV_INTERVAL)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS),
    )
}

#[derive(Serialize)]
struct GossipRequest<'a> {
    peers: &'a [String],
}

/// Spawn a background task that fires gossip rounds as long as the
/// process is alive. No-op if `PEER_DISCOVERY_SEEDS` is unset.
///
/// * `bind_port`: the port this qo-server is listening on. The task
///   will POST against `http://127.0.0.1:<port>/api/qlms/federation/gossip`.
/// * `auth_token`: optional; sent as `Authorization: Bearer …` so the
///   request passes the same auth middleware as any external client.
pub fn spawn_if_enabled(bind_port: u16, auth_token: Option<String>) -> bool {
    let Some(peers) = parse_seeds_from_env() else {
        info!(
            "peer discovery disabled (set {ENV_SEEDS}=host1:4646,host2:4747,... to enable)"
        );
        return false;
    };
    let interval = interval_from_env();
    info!(
        "peer discovery enabled: {} peer(s), gossip every {:?}",
        peers.len(),
        interval
    );

    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                warn!("peer discovery: failed to build HTTP client: {e}");
                return;
            }
        };

        let url = format!("http://127.0.0.1:{bind_port}/api/qlms/federation/gossip");
        let req_body = GossipRequest { peers: &peers };

        // Small initial delay so the server is actually listening before
        // the first round fires.
        tokio::time::sleep(Duration::from_secs(2)).await;

        loop {
            let mut request = client.post(&url).json(&req_body);
            if let Some(token) = auth_token.as_deref() {
                request = request.bearer_auth(token);
            }
            match request.send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("gossip round OK against {} peer(s)", peers.len());
                }
                Ok(resp) => {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    warn!("gossip round failed: HTTP {status}: {body}");
                }
                Err(e) => {
                    warn!("gossip round failed: {e}");
                }
            }
            tokio::time::sleep(interval).await;
        }
    });
    true
}

// ---------------------------------------------------------------------------
// Tests — pure env-parsing logic only. Background task is covered by the
// federation_soak integration test (Task 4.1, future work).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_seeds_handles_whitespace_and_blanks() {
        let got = split_seeds(" host1:4646 , host2:4747 , ,host3:4848");
        assert_eq!(got, vec!["host1:4646", "host2:4747", "host3:4848"]);
    }

    #[test]
    fn filter_self_strips_exact_match_only() {
        let input = vec![
            "host1:4646".to_string(),
            "me:9999".to_string(),
            "host2:4747".to_string(),
        ];
        let out = filter_self(input, Some("me:9999"));
        assert_eq!(out, vec!["host1:4646", "host2:4747"]);
    }

    #[test]
    fn filter_self_leaves_peers_alone_when_self_unset() {
        let input = vec!["host1:4646".to_string(), "host2:4747".to_string()];
        let out = filter_self(input.clone(), None);
        assert_eq!(out, input);
    }

    #[test]
    fn filter_self_dedupes_while_preserving_order() {
        let input = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "b".to_string(),
        ];
        let out = filter_self(input, None);
        assert_eq!(out, vec!["a", "b", "c"]);
    }
}
