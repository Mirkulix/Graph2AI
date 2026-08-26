//! Rate limiting — a small in-tree token bucket, because a DoS bound should
//! not depend on an external crate. Keys requests by peer IP; direct in-process
//! calls (router tests, no `ConnectInfo`) share one "anonymous" bucket so a
//! generous limit never trips them.

use axum::extract::{ConnectInfo, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// A per-key token bucket: `rate` tokens refill per second, up to `capacity`
/// burst. One request spends one token; an empty bucket is refused.
pub struct RateLimiter {
    inner: Mutex<HashMap<String, (Instant, f64)>>,
    rate: f64,
    capacity: f64,
    idle_ttl: Duration,
}

impl RateLimiter {
    pub fn new(rate_per_second: u32, burst: u32) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            rate: f64::from(rate_per_second.max(1)),
            capacity: f64::from(burst.max(1)),
            idle_ttl: Duration::from_secs(60),
        }
    }

    /// True when the key may proceed: refill at `rate` tokens/s up to
    /// `capacity`, then spend one token.
    pub async fn allow(&self, key: &str) -> bool {
        let now = Instant::now();
        let mut map = self.inner.lock().await;

        // Occasional prune so a spoofed-IP flood cannot grow the map forever.
        if map.len() > 10_000 {
            map.retain(|_, (last, tokens)| {
                now.duration_since(*last) < self.idle_ttl || *tokens < self.capacity
            });
        }

        let entry = map
            .entry(key.to_string())
            .or_insert_with(|| (now, self.capacity));
        let (last, tokens) = entry;
        let elapsed = now.duration_since(*last).as_secs_f64();
        *tokens = (*tokens + elapsed * self.rate).min(self.capacity);
        *last = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Key a request by its peer IP; in-process calls fall back to "anonymous".
fn peer_ip(request: &Request<axum::body::Body>) -> String {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "anonymous".to_string())
}

/// Middleware: refuse with 429 when the peer's bucket is empty.
pub async fn middleware(
    State(limiter): State<Arc<RateLimiter>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let key = peer_ip(&request);
    if limiter.allow(&key).await {
        Ok(next.run(request).await)
    } else {
        Err(StatusCode::TOO_MANY_REQUESTS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bucket_exhausts_then_refills() {
        // rate 10/s, burst 3: three immediate requests pass, the fourth is
        // refused, and after ~120 ms a token has refilled.
        let limiter = RateLimiter::new(10, 3);
        assert!(limiter.allow("a").await);
        assert!(limiter.allow("a").await);
        assert!(limiter.allow("a").await);
        assert!(!limiter.allow("a").await);

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(limiter.allow("a").await);
    }

    #[tokio::test]
    async fn keys_are_independent() {
        let limiter = RateLimiter::new(1, 1);
        assert!(limiter.allow("a").await);
        assert!(!limiter.allow("a").await);
        assert!(limiter.allow("b").await, "a separate key has its own bucket");
    }

    #[tokio::test]
    async fn zero_config_clamps_to_at_least_one() {
        let limiter = RateLimiter::new(0, 0);
        assert!(limiter.allow("a").await);
    }
}
