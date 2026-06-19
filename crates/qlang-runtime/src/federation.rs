//! Reference helpers for ternary federation / gossip merges.

/// Returns `true` when every weight is one of `-1`, `0`, `1`.
pub fn verify_ternary(weights: &[i8]) -> bool {
    weights.iter().all(|&w| matches!(w, -1 | 0 | 1))
}

/// Count element-wise differences between two equal-length ternary vectors.
pub fn count_changes(before: &[i8], after: &[i8]) -> usize {
    before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Merge equal-length ternary vectors by majority vote.
///
/// Per position:
/// - more `+1` than `-1` -> `+1`
/// - more `-1` than `+1` -> `-1`
/// - tie or only zeroes     -> `0`
pub fn ternary_majority_vote(peers: &[&[i8]]) -> Result<Vec<i8>, String> {
    if peers.is_empty() {
        return Ok(Vec::new());
    }

    let len = peers[0].len();
    if peers.iter().any(|peer| peer.len() != len) {
        return Err("all peer vectors must have equal length".into());
    }
    if peers.iter().any(|peer| !verify_ternary(peer)) {
        return Err("all peer vectors must be ternary".into());
    }

    let mut merged = Vec::with_capacity(len);
    for idx in 0..len {
        let mut pos = 0usize;
        let mut neg = 0usize;
        for peer in peers {
            match peer[idx] {
                1 => pos += 1,
                -1 => neg += 1,
                0 => {}
                _ => unreachable!("verify_ternary guards this"),
            }
        }
        let out = if pos > neg {
            1
        } else if neg > pos {
            -1
        } else {
            0
        };
        merged.push(out);
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_vote_tie_breaks_to_zero() {
        let a: &[i8] = &[1, -1, 0];
        let b: &[i8] = &[-1, -1, 0];
        let c: &[i8] = &[1, 1, 0];
        assert_eq!(ternary_majority_vote(&[a, b, c]).unwrap(), vec![1, -1, 0]);
    }

    #[test]
    fn change_count_works() {
        assert_eq!(count_changes(&[-1, 0, 1], &[-1, 1, 0]), 2);
    }
}
