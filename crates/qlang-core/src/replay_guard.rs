//! Replay protection for QLMS v1.1 — nonce + timestamp freshness.
//!
//! Spec: QLMS_PROTOCOL_v1_1.md §14 (replay protection).
//!
//! Messages include a 16-byte random nonce and a unix-seconds timestamp.
//! `ReplayGuard::check` rejects:
//!   - nonces that have already been seen (bounded memory — FIFO eviction)
//!   - timestamps older than `max_skew_secs` relative to local clock
//!   - timestamps further than `max_skew_secs` in the future
//!
//! This is intentionally simple: no external deps, pure Rust, suitable
//! for embedding in QLMS frame decoders.

use std::collections::{HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bounded-memory replay guard.
///
/// Tracks up to `max_nonces` recently-seen nonces. Beyond that cap the
/// oldest nonce is evicted (FIFO). Timestamps outside `±max_skew_secs`
/// of the local clock are rejected.
pub struct ReplayGuard {
    seen_nonces: HashSet<[u8; 16]>,
    order: VecDeque<[u8; 16]>,
    max_nonces: usize,
    max_skew_secs: u64,
}

impl ReplayGuard {
    /// Create a new guard.
    ///
    /// - `max_nonces`: maximum number of nonces to remember before FIFO eviction.
    /// - `max_skew_secs`: reject timestamps older (or more than this many seconds
    ///   in the future) than the local clock.
    pub fn new(max_nonces: usize, max_skew_secs: u64) -> Self {
        Self {
            seen_nonces: HashSet::with_capacity(max_nonces.max(1)),
            order: VecDeque::with_capacity(max_nonces.max(1)),
            max_nonces: max_nonces.max(1),
            max_skew_secs,
        }
    }

    /// Check a message for freshness.
    ///
    /// Returns `Ok(())` if the nonce has not been seen and the timestamp
    /// is within `±max_skew_secs` of the local clock.
    /// Returns `Err` with a static reason string otherwise.
    ///
    /// `timestamp` is expected to be seconds since unix epoch.
    pub fn check(&mut self, nonce: &[u8; 16], timestamp: u64) -> Result<(), &'static str> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Reject stale (too old)
        if now > timestamp && now - timestamp > self.max_skew_secs {
            return Err("stale timestamp");
        }
        // Reject future (too far ahead — tolerate small clock skew)
        if timestamp > now && timestamp - now > self.max_skew_secs {
            return Err("future timestamp");
        }

        // Reject replay
        if self.seen_nonces.contains(nonce) {
            return Err("replayed nonce");
        }

        // Record
        if self.order.len() >= self.max_nonces {
            if let Some(old) = self.order.pop_front() {
                self.seen_nonces.remove(&old);
            }
        }
        self.seen_nonces.insert(*nonce);
        self.order.push_back(*nonce);
        Ok(())
    }

    /// Number of nonces currently remembered.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Returns true if no nonces are remembered.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[test]
    fn test_replay_guard_detects_duplicate_nonce() {
        let mut g = ReplayGuard::new(128, 60);
        let nonce = [7u8; 16];
        let ts = now_secs();

        assert!(g.check(&nonce, ts).is_ok(), "first use must be fresh");
        let err = g.check(&nonce, ts).unwrap_err();
        assert_eq!(err, "replayed nonce");
    }

    #[test]
    fn test_replay_guard_rejects_stale_timestamp() {
        let mut g = ReplayGuard::new(128, 60);
        let nonce = [9u8; 16];
        let ts = now_secs().saturating_sub(3600); // 1 hour old
        let err = g.check(&nonce, ts).unwrap_err();
        assert_eq!(err, "stale timestamp");
    }

    #[test]
    fn test_replay_guard_rejects_future_timestamp() {
        let mut g = ReplayGuard::new(128, 60);
        let nonce = [10u8; 16];
        let ts = now_secs() + 3600; // 1 hour ahead
        let err = g.check(&nonce, ts).unwrap_err();
        assert_eq!(err, "future timestamp");
    }

    #[test]
    fn test_replay_guard_accepts_fresh_messages() {
        let mut g = ReplayGuard::new(128, 60);
        let ts = now_secs();
        for i in 0..50u8 {
            let mut nonce = [0u8; 16];
            nonce[0] = i;
            nonce[15] = i.wrapping_mul(13);
            assert!(
                g.check(&nonce, ts).is_ok(),
                "fresh message {} should be accepted",
                i
            );
        }
        assert_eq!(g.len(), 50);
    }

    #[test]
    fn test_replay_guard_fifo_eviction() {
        let mut g = ReplayGuard::new(4, 60);
        let ts = now_secs();
        let nonces: Vec<[u8; 16]> = (0..5u8)
            .map(|i| {
                let mut n = [0u8; 16];
                n[0] = i;
                n
            })
            .collect();

        for n in &nonces {
            assert!(g.check(n, ts).is_ok());
        }
        // Capacity 4, inserted 5 → first nonce should have been evicted
        // and must now be accepted as fresh again.
        assert!(g.check(&nonces[0], ts).is_ok());
    }

    #[test]
    fn test_replay_guard_within_skew_accepted() {
        let mut g = ReplayGuard::new(16, 60);
        let now = now_secs();
        let nonce1 = [1u8; 16];
        let nonce2 = [2u8; 16];
        // 30 seconds in the past: within 60s skew
        assert!(g.check(&nonce1, now.saturating_sub(30)).is_ok());
        // 30 seconds in the future: within 60s skew
        assert!(g.check(&nonce2, now + 30).is_ok());
    }
}
