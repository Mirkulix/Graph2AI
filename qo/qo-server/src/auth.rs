use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

use crate::api_keys::{Principal, Role};
use crate::AppState;
use qlang_core::crypto::ct_eq;

/// Extract the presented bearer token, from the `Authorization` header or, as a
/// fallback, a `?token=` query param (the WebSocket handshake cannot set
/// headers).
fn presented_token(headers: &HeaderMap, request: &Request) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .or_else(|| {
            request.uri().query().and_then(|q| {
                q.split('&')
                    .find(|p| p.starts_with("token="))
                    .map(|p| p.trim_start_matches("token=").to_string())
            })
        })
}

/// Is this request from the machine the server runs on?
///
/// Used only by the opt-in local mode below. A missing `ConnectInfo` (router
/// tests, and any transport that does not carry a peer address) is treated as
/// *not* local, so the check fails closed rather than opening every seat.
fn is_loopback(request: &Request) -> bool {
    request
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|info| info.0.ip().is_loopback())
        .unwrap_or(false)
}

/// Whether the single-machine convenience mode is on.
///
/// `QO_LOCAL_MODE=1` says "this instance serves only me, on this machine".
/// It exists because of a real gap: issuing even one seat previously turned
/// every route token-only, so a locally-installed MCP client (Claude Code,
/// Codex, …) got 401 until the operator hand-copied a secret into a config
/// file. That is a bad first-run experience for a single-user install, and
/// the workaround people reach for — deleting their seats — is worse.
///
/// It is safe *only* in combination with the bind rule in `main.rs`: an
/// instance in local mode binds to 127.0.0.1, so "came from loopback" cannot
/// be spoofed from the network. Turning it on for a network-bound instance is
/// refused there rather than honoured here.
fn local_mode_enabled() -> bool {
    std::env::var("QO_LOCAL_MODE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Authenticate a request and attach the resolved [`Principal`] to it.
///
/// Resolution order, all fail-closed:
///
/// 1. `QO_AUTH_TOKEN`, when set, authenticates as the implicit admin. This is
///    the original single-operator credential and keeps working untouched.
/// 2. Any non-revoked key in the API-key store authenticates as its own role.
/// 3. In local mode only, a request from loopback authenticates as admin
///    without a token — see [`local_mode_enabled`].
/// 4. Otherwise the request is refused.
///
/// The one open case is deliberate and unchanged from before: if no token is
/// set **and** the key store is empty, the server is unauthenticated — which
/// is why `main.rs` binds such an instance to loopback only.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let env_token = std::env::var("QO_AUTH_TOKEN").ok().filter(|t| !t.is_empty());
    let store_has_keys = !state.api_keys.read().await.keys.is_empty();

    // Fully open only when there is nothing configured to check against.
    if env_token.is_none() && !store_has_keys {
        request.extensions_mut().insert(Principal::root());
        return Ok(next.run(request).await);
    }

    // Local mode: a caller on this machine is the operator. Checked before the
    // token paths so a locally-installed client needs no credential at all,
    // which is the whole point — but still after the "nothing configured" case
    // so behaviour without seats is unchanged.
    if local_mode_enabled() && is_loopback(&request) {
        request.extensions_mut().insert(Principal::root());
        return Ok(next.run(request).await);
    }

    let Some(token) = presented_token(&headers, &request) else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    // The env token, when present, is the admin credential.
    if let Some(env_token) = &env_token {
        if ct_eq(token.as_bytes(), env_token.as_bytes()) {
            request.extensions_mut().insert(Principal::root());
            return Ok(next.run(request).await);
        }
    }

    // Then the per-seat store. The read guard must drop BEFORE `next.run`:
    // a guard bound inside an `if let` condition lives until the end of the
    // block, which would deadlock a handler that takes the write lock
    // (e.g. seat issue) against the same runtime.
    let principal = {
        let store = state.api_keys.read().await;
        store.authenticate(&token)
    };
    if let Some(principal) = principal {
        request.extensions_mut().insert(principal);
        return Ok(next.run(request).await);
    }

    Err(StatusCode::UNAUTHORIZED)
}

/// Read the role a request authenticated as, fail-closed: if no principal is
/// attached (which should not happen after [`auth_middleware`] ran), treat the
/// request as the least-privileged role rather than assuming admin.
fn role_of(request: &Request) -> Role {
    request
        .extensions()
        .get::<Principal>()
        .map(|p| p.role)
        .unwrap_or(Role::Viewer)
}

/// Allow the request only when the principal may write (`member` or `admin`).
/// A `viewer` seat is read-only and gets 403 here — this is what makes a
/// "$49/viewer" seat actually read-only rather than merely labelled.
pub async fn require_write(request: Request, next: Next) -> Result<Response, StatusCode> {
    if role_of(&request).can_write() {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

/// Allow the request only when the principal may administer (`admin` only).
pub async fn require_admin(request: Request, next: Next) -> Result<Response, StatusCode> {
    if role_of(&request).can_administer() {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::FORBIDDEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::ConnectInfo;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

    /// Serialises tests that mutate the process-wide `QO_LOCAL_MODE`.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn request_from(addr: Option<SocketAddr>) -> Request {
        let mut req = Request::new(axum::body::Body::empty());
        if let Some(addr) = addr {
            req.extensions_mut().insert(ConnectInfo(addr));
        }
        req
    }

    #[test]
    fn loopback_is_recognised_on_both_stacks() {
        let v4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4646);
        let v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), 4646);
        assert!(is_loopback(&request_from(Some(v4))));
        assert!(is_loopback(&request_from(Some(v6))));
    }

    #[test]
    fn a_remote_address_is_not_loopback() {
        let lan = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50)), 4646);
        let public = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)), 4646);
        assert!(!is_loopback(&request_from(Some(lan))));
        assert!(!is_loopback(&request_from(Some(public))));
    }

    /// The fail-closed half: with no peer address we must not guess "local".
    /// Router-level tests hit this path, and so would any transport that does
    /// not carry `ConnectInfo`.
    #[test]
    fn a_missing_peer_address_is_not_treated_as_local() {
        assert!(!is_loopback(&request_from(None)));
    }

    #[test]
    fn local_mode_is_off_unless_explicitly_enabled() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var("QO_LOCAL_MODE");
        assert!(!local_mode_enabled(), "must default to off");

        for off in ["0", "", "no", "false"] {
            std::env::set_var("QO_LOCAL_MODE", off);
            assert!(!local_mode_enabled(), "{off:?} must not enable local mode");
        }
        for on in ["1", "true", "TRUE"] {
            std::env::set_var("QO_LOCAL_MODE", on);
            assert!(local_mode_enabled(), "{on:?} must enable local mode");
        }
        std::env::remove_var("QO_LOCAL_MODE");
    }
}
