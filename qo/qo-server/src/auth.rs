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

/// Authenticate a request and attach the resolved [`Principal`] to it.
///
/// Resolution order, all fail-closed:
///
/// 1. `QO_AUTH_TOKEN`, when set, authenticates as the implicit admin. This is
///    the original single-operator credential and keeps working untouched.
/// 2. Any non-revoked key in the API-key store authenticates as its own role.
/// 3. Otherwise the request is refused.
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
    let store_has_keys = !state.api_keys.keys.is_empty();

    // Fully open only when there is nothing configured to check against.
    if env_token.is_none() && !store_has_keys {
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

    // Then the per-seat store.
    if let Some(principal) = state.api_keys.authenticate(&token) {
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
