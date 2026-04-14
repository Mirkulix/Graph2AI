//! Federated ternary merge — majority vote across N peers.
//!
//! Used by QLMS federation endpoints and integration tests to merge ternary
//! specialist weights from multiple QO servers. Pure add/compare, no multiply.
//!
//! # Algorithm
//!
//! For each weight position:
//! 1. Count how many peers have -1, 0, +1
//! 2. Pick the majority state
//! 3. Ties broken toward 0 (conservative: preserve "don't know")
//!
//! # Guarantees
//!
//! - Pure Rust, no external deps
//! - O(N * W) where N = peers, W = weight count
//! - Preserves ternary invariant: output ∈ {-1, 0, +1}
//! - Backwards compatible with 2-node merge (N=2 still works)

/// Majority-vote merge across `peers` ternary weight vectors.
///
/// All peer vectors MUST be the same length; returns an error if not.
///
/// # Tie-breaking
///
/// When two or more states have the same maximum count, we prefer `0` (the
/// "uncertain" / pruned state) over `+1`/`-1`. This prevents peers from being
/// forced into a sign they do not agree on.
///
/// # Edge cases
///
/// - Empty peers list → returns an empty vector.
/// - Single peer → returns that peer's weights unchanged.
pub fn ternary_majority_vote(peers: &[&[i8]]) -> Result<Vec<i8>, String> {
    if peers.is_empty() {
        return Ok(Vec::new());
    }
    let w = peers[0].len();
    for (i, p) in peers.iter().enumerate().skip(1) {
        if p.len() != w {
            return Err(format!(
                "ternary_majority_vote: peer {} length {} != expected {}",
                i,
                p.len(),
                w
            ));
        }
    }

    let mut out = vec![0i8; w];
    for k in 0..w {
        let mut neg: u32 = 0;
        let mut zero: u32 = 0;
        let mut pos: u32 = 0;
        for p in peers {
            match p[k] {
                v if v > 0 => pos += 1,
                v if v < 0 => neg += 1,
                _ => zero += 1,
            }
        }
        // Tie-break toward 0: require strict majority for a sign.
        out[k] = if pos > neg && pos > zero {
            1
        } else if neg > pos && neg > zero {
            -1
        } else {
            0
        };
    }
    Ok(out)
}

/// Count positions where the merged vector differs from `reference`.
///
/// Useful for diagnostics ("how many weights did gossip change for me?").
pub fn count_changes(reference: &[i8], merged: &[i8]) -> usize {
    reference
        .iter()
        .zip(merged.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Sanity check: all values are ternary.
pub fn verify_ternary(weights: &[i8]) -> bool {
    weights.iter().all(|&w| w == -1 || w == 0 || w == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_peers_empty_result() {
        let empty: Vec<&[i8]> = vec![];
        assert_eq!(ternary_majority_vote(&empty).unwrap(), Vec::<i8>::new());
    }

    #[test]
    fn single_peer_identity() {
        let a: &[i8] = &[-1, 0, 1, 1, -1];
        let merged = ternary_majority_vote(&[a]).unwrap();
        // Single peer: each value is its own majority → identity.
        assert_eq!(merged, vec![-1, 0, 1, 1, -1]);
    }

    #[test]
    fn three_peers_clear_majority() {
        let a: &[i8] = &[1, -1, 0, 1];
        let b: &[i8] = &[1, -1, 0, 0];
        let c: &[i8] = &[1, 1, 0, -1];
        let merged = ternary_majority_vote(&[a, b, c]).unwrap();
        // pos 0: pos=3 → 1
        // pos 1: neg=2, pos=1 → neg majority → -1
        // pos 2: zero=3 → 0
        // pos 3: pos=1, zero=1, neg=1 → tie → 0
        assert_eq!(merged, vec![1, -1, 0, 0]);
    }

    #[test]
    fn tie_breaks_to_zero() {
        // 2 pos vs 2 neg → tie → 0
        let a: &[i8] = &[1];
        let b: &[i8] = &[1];
        let c: &[i8] = &[-1];
        let d: &[i8] = &[-1];
        let merged = ternary_majority_vote(&[a, b, c, d]).unwrap();
        assert_eq!(merged, vec![0]);
    }

    #[test]
    fn two_node_backwards_compat() {
        // Both peers agree → keep value
        let a: &[i8] = &[1, -1, 0];
        let b: &[i8] = &[1, -1, 0];
        assert_eq!(ternary_majority_vote(&[a, b]).unwrap(), vec![1, -1, 0]);

        // Peers disagree → tie → 0 (conservative)
        let a: &[i8] = &[1, -1, 0];
        let b: &[i8] = &[-1, 1, 1];
        assert_eq!(ternary_majority_vote(&[a, b]).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn length_mismatch_errors() {
        let a: &[i8] = &[1, 0, -1];
        let b: &[i8] = &[1, 0];
        assert!(ternary_majority_vote(&[a, b]).is_err());
    }

    #[test]
    fn count_changes_works() {
        let a: &[i8] = &[1, -1, 0, 1];
        let b: &[i8] = &[1, 0, 0, -1];
        assert_eq!(count_changes(a, b), 2);
    }

    #[test]
    fn output_is_always_ternary() {
        let a: &[i8] = &[1, -1, 0, 1, -1];
        let b: &[i8] = &[-1, -1, 1, 0, 0];
        let c: &[i8] = &[0, 1, -1, 1, 1];
        let merged = ternary_majority_vote(&[a, b, c]).unwrap();
        assert!(verify_ternary(&merged));
    }
}
