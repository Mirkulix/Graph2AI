//! Per-seat API keys — the first thing that turns one shared token into
//! something a team can actually buy seats of.
//!
//! Until now the server had exactly one `QO_AUTH_TOKEN`: everyone who could
//! reach it shared one credential, and revoking one person meant rotating it
//! for everyone. That is fine for a single operator and a non-starter for a
//! hosted product. This adds a small store of named keys, each with a role,
//! that an admin can hand out and revoke one at a time.
//!
//! It does **not** replace the fail-closed model. The precedence is:
//!
//! 1. If `QO_AUTH_TOKEN` is set, it still works — as the admin key. Existing
//!    single-operator setups keep working unchanged.
//! 2. Any key in the store authenticates as its own role.
//! 3. If neither matches, the request is refused. A store that loads empty and
//!    no token means nobody gets in — the same direction of failure as the
//!    rest of the system.
//!
//! Keys are compared in constant time. The store is loaded from JSON an
//! operator manages, the same shape as the delta trust store, so the two are
//! administered the same way.

use qlang_core::crypto::ct_eq;
use serde::{Deserialize, Serialize};

/// What a key is allowed to do. Coarse on purpose — finer-grained policy is a
/// later refinement, and shipping three clear roles beats shipping a
/// permission matrix nobody configures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Read the graph and submit signed deltas. The default seat.
    Member,
    /// Everything a member can do, plus manage keys and the trust store.
    Admin,
    /// Read-only. For dashboards, auditors, CI that only inspects.
    Viewer,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::Member => "member",
            Role::Admin => "admin",
            Role::Viewer => "viewer",
        }
    }

    /// May this role change graph state (submit deltas, verify claims)?
    pub fn can_write(&self) -> bool {
        matches!(self, Role::Member | Role::Admin)
    }

    /// May this role manage keys, trust and other server configuration?
    pub fn can_administer(&self) -> bool {
        matches!(self, Role::Admin)
    }
}

/// One issued key. The `secret` is the bearer token the seat holder sends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// Human label — a person's name, a CI job, "audit-dashboard".
    pub label: String,
    /// The bearer token. Compared in constant time; never logged.
    pub secret: String,
    pub role: Role,
    /// When true the key is refused. Revoking is a flag, not a deletion, so
    /// the audit log keeps naming who a past action belonged to.
    #[serde(default)]
    pub revoked: bool,
}

/// The identity a request authenticated as, passed down to handlers that care
/// about who is asking.
#[derive(Debug, Clone)]
pub struct Principal {
    pub label: String,
    pub role: Role,
}

impl Principal {
    /// The implicit admin behind a bare `QO_AUTH_TOKEN`, so existing setups
    /// keep full access without an entry in the store.
    pub fn root() -> Self {
        Self {
            label: "root".into(),
            role: Role::Admin,
        }
    }
}

/// The set of keys the server will accept.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApiKeyStore {
    #[serde(default)]
    pub keys: Vec<ApiKey>,
}

impl ApiKeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn issue(&mut self, label: impl Into<String>, secret: impl Into<String>, role: Role) {
        self.keys.push(ApiKey {
            label: label.into(),
            secret: secret.into(),
            role,
            revoked: false,
        });
    }

    /// Revoke the first active key with this label. Returns whether one was
    /// revoked (false when the label is unknown or already revoked).
    pub fn revoke(&mut self, label: &str) -> bool {
        let mut hit = false;
        for key in &mut self.keys {
            if key.label == label && !key.revoked {
                key.revoked = true;
                hit = true;
            }
        }
        hit
    }

    /// Resolve a presented bearer token to a principal.
    ///
    /// Every non-revoked key is compared in constant time, and the loop does
    /// not stop at the first match — so the time taken does not reveal which
    /// key (or how early in the list) matched.
    pub fn authenticate(&self, presented: &str) -> Option<Principal> {
        let mut found: Option<Principal> = None;
        for key in &self.keys {
            if key.revoked {
                continue;
            }
            if ct_eq(presented.as_bytes(), key.secret.as_bytes()) {
                found = Some(Principal {
                    label: key.label.clone(),
                    role: key.role,
                });
            }
        }
        found
    }

    /// How many seats are currently active. What a per-seat invoice counts.
    pub fn active_seats(&self) -> usize {
        self.keys.iter().filter(|k| !k.revoked).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ApiKeyStore {
        let mut s = ApiKeyStore::new();
        s.issue("alice", "secret-a", Role::Admin);
        s.issue("bob", "secret-b", Role::Member);
        s.issue("dash", "secret-v", Role::Viewer);
        s
    }

    #[test]
    fn a_valid_key_resolves_to_its_role() {
        let s = store();
        assert_eq!(s.authenticate("secret-a").unwrap().role, Role::Admin);
        assert_eq!(s.authenticate("secret-b").unwrap().role, Role::Member);
        assert_eq!(s.authenticate("secret-v").unwrap().role, Role::Viewer);
    }

    #[test]
    fn an_unknown_key_resolves_to_nothing() {
        assert!(store().authenticate("nope").is_none());
    }

    #[test]
    fn a_revoked_key_is_refused() {
        let mut s = store();
        s.keys[1].revoked = true;
        assert!(s.authenticate("secret-b").is_none(), "revoked key still worked");
        // Others still work.
        assert!(s.authenticate("secret-a").is_some());
    }

    #[test]
    fn roles_gate_the_right_actions() {
        assert!(Role::Admin.can_write() && Role::Admin.can_administer());
        assert!(Role::Member.can_write() && !Role::Member.can_administer());
        assert!(!Role::Viewer.can_write() && !Role::Viewer.can_administer());
    }

    #[test]
    fn active_seats_counts_only_live_keys() {
        let mut s = store();
        assert_eq!(s.active_seats(), 3);
        s.keys[0].revoked = true;
        assert_eq!(s.active_seats(), 2);
    }

    #[test]
    fn the_store_round_trips_through_json() {
        let s = store();
        let parsed = ApiKeyStore::from_json(&s.to_json().unwrap()).unwrap();
        assert_eq!(parsed.active_seats(), s.active_seats());
        assert_eq!(parsed.authenticate("secret-a").unwrap().role, Role::Admin);
    }

    #[test]
    fn an_empty_store_authenticates_nobody() {
        assert!(ApiKeyStore::new().authenticate("anything").is_none());
    }
}
