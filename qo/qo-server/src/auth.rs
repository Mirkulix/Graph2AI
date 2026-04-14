use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};

pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Get token from env
    let required_token = std::env::var("QO_AUTH_TOKEN").ok();

    if let Some(token) = required_token {
        if token.is_empty() {
            return Ok(next.run(request).await);
        }
        // Check Authorization: Bearer <token> or ?token=<token>
        let provided = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|s| s.to_string())
            .or_else(|| {
                request.uri().query()
                    .and_then(|q| q.split('&').find(|p| p.starts_with("token=")))
                    .map(|p| p.trim_start_matches("token=").to_string())
            });

        match provided {
            Some(t) if t == token => Ok(next.run(request).await),
            _ => Err(StatusCode::UNAUTHORIZED),
        }
    } else {
        // No token configured — allow all
        Ok(next.run(request).await)
    }
}
