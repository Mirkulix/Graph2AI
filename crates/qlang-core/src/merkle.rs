//! Merkle Tree for Graph Verification.
//!
//! Every sub-computation in a QLANG graph gets its own hash.
//! These hashes form a Merkle tree -- you can verify any part of
//! a computation without re-running the whole thing.
//!
//! Use cases:
//! - Partial verification: prove a single node belongs to a signed graph
//! - Incremental updates: when a graph changes, only re-hash the changed path
//! - Distributed trust: share proofs without sharing the full graph

use crate::crypto::sha256;
use crate::graph::Graph;

/// A Merkle proof for a single node in a computation graph.
///
/// Contains the sibling hashes along the path from the leaf to the root,
/// allowing anyone to recompute the root hash from the node hash alone.
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// The node this proof is for.
    pub node_id: u32,
    /// SHA-256 hash of the node's content (op + input_types + output_types).
    pub node_hash: [u8; 32],
    /// Sibling hashes along the path to the root.
    /// Each entry is (sibling_hash, sibling_is_on_left).
    pub siblings: Vec<([u8; 32], bool)>,
    /// The expected root hash.
    pub root: [u8; 32],
}

/// Merkle tree over a computation graph.
///
/// Built bottom-up from per-node hashes. The root hash commits
/// to every node in the graph -- any change to any node changes the root.
#[derive(Debug, Clone)]
pub struct GraphMerkleTree {
    /// Root hash of the entire tree.
    pub root: [u8; 32],
    /// Per-node hashes: (node_id, hash).
    pub node_hashes: Vec<(u32, [u8; 32])>,
    /// Tree levels, bottom-up. Level 0 = leaf hashes, last level = root.
    pub tree_levels: Vec<Vec<[u8; 32]>>,
}

impl GraphMerkleTree {
    /// Build a Merkle tree from a graph.
    ///
    /// Each node is hashed as: SHA-256(op_display || input_type_displays || output_type_displays).
    /// An empty graph produces a tree with a zero root.
    pub fn build(graph: &Graph) -> Self {
        // Hash each node: hash(op || input_types || output_types)
        let mut node_hashes: Vec<(u32, [u8; 32])> = Vec::new();
        for node in &graph.nodes {
            let mut data = Vec::new();
            data.extend_from_slice(format!("{}", node.op).as_bytes());
            for it in &node.input_types {
                data.extend_from_slice(format!("{}", it).as_bytes());
            }
            for ot in &node.output_types {
                data.extend_from_slice(format!("{}", ot).as_bytes());
            }
            node_hashes.push((node.id, sha256(&data)));
        }

        // Handle empty graph
        if node_hashes.is_empty() {
            return GraphMerkleTree {
                root: [0u8; 32],
                node_hashes,
                tree_levels: vec![],
            };
        }

        // Build tree bottom-up
        let mut current_level: Vec<[u8; 32]> = node_hashes.iter().map(|(_, h)| *h).collect();
        let mut tree_levels = vec![current_level.clone()];

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let mut combined = Vec::with_capacity(64);
                combined.extend_from_slice(&chunk[0]);
                if chunk.len() > 1 {
                    combined.extend_from_slice(&chunk[1]);
                } else {
                    // Odd node: duplicate last hash
                    combined.extend_from_slice(&chunk[0]);
                }
                next_level.push(sha256(&combined));
            }
            tree_levels.push(next_level.clone());
            current_level = next_level;
        }

        let root = current_level[0];

        GraphMerkleTree {
            root,
            node_hashes,
            tree_levels,
        }
    }

    /// Generate a proof that a specific node is part of this graph.
    ///
    /// Returns `None` if the node_id is not in the tree.
    pub fn prove(&self, node_id: u32) -> Option<MerkleProof> {
        let idx = self.node_hashes.iter().position(|(id, _)| *id == node_id)?;
        let node_hash = self.node_hashes[idx].1;

        let mut siblings = Vec::new();
        let mut current_idx = idx;

        // Walk up the tree, collecting sibling hashes at each level.
        // We stop before the last level (which is the root itself).
        for level in &self.tree_levels[..self.tree_levels.len().saturating_sub(1)] {
            let sibling_idx = if current_idx % 2 == 0 {
                current_idx + 1
            } else {
                current_idx - 1
            };
            let sibling_hash = if sibling_idx < level.len() {
                level[sibling_idx]
            } else {
                // Odd number of nodes: the last node's sibling is itself
                level[current_idx]
            };
            // is_left means the sibling is on the left side of the concatenation
            let is_left = current_idx % 2 == 1;
            siblings.push((sibling_hash, is_left));
            current_idx /= 2;
        }

        Some(MerkleProof {
            node_id,
            node_hash,
            siblings,
            root: self.root,
        })
    }

    /// Verify a Merkle proof: recompute the root from the leaf and siblings.
    ///
    /// Returns `true` if the recomputed root matches the proof's root.
    pub fn verify_proof(proof: &MerkleProof) -> bool {
        let mut current = proof.node_hash;
        for (sibling, is_left) in &proof.siblings {
            let mut combined = Vec::with_capacity(64);
            if *is_left {
                combined.extend_from_slice(sibling);
                combined.extend_from_slice(&current);
            } else {
                combined.extend_from_slice(&current);
                combined.extend_from_slice(sibling);
            }
            current = sha256(&combined);
        }
        current == proof.root
    }

    /// Number of leaf nodes (graph nodes) in the tree.
    pub fn leaf_count(&self) -> usize {
        self.node_hashes.len()
    }

    /// Number of levels in the tree (including leaves and root).
    pub fn depth(&self) -> usize {
        self.tree_levels.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;
    use crate::tensor::TensorType;

    // ---- Helper ----

    fn make_graph(n_nodes: usize) -> Graph {
        let mut g = Graph::new("merkle_test");
        for i in 0..n_nodes {
            let name = format!("x{}", i);
            g.add_node(
                Op::Input { name },
                vec![],
                vec![TensorType::f32_vector(4)],
            );
        }
        // Wire up sequentially if more than one node
        if n_nodes > 1 {
            for i in 0..(n_nodes - 1) {
                g.add_edge(i as u32, 0, (i + 1) as u32, 0, TensorType::f32_vector(4));
            }
        }
        g
    }

    fn make_varied_graph() -> Graph {
        let mut g = Graph::new("varied");
        g.add_node(
            Op::Input { name: "a".into() },
            vec![],
            vec![TensorType::f32_vector(8)],
        );
        g.add_node(
            Op::Input { name: "b".into() },
            vec![],
            vec![TensorType::f32_vector(8)],
        );
        g.add_node(
            Op::Add,
            vec![TensorType::f32_vector(8), TensorType::f32_vector(8)],
            vec![TensorType::f32_vector(8)],
        );
        g.add_node(Op::Relu, vec![TensorType::f32_vector(8)], vec![TensorType::f32_vector(8)]);
        g.add_node(
            Op::Output { name: "y".into() },
            vec![TensorType::f32_vector(8)],
            vec![],
        );
        g.add_edge(0, 0, 2, 0, TensorType::f32_vector(8));
        g.add_edge(1, 0, 2, 1, TensorType::f32_vector(8));
        g.add_edge(2, 0, 3, 0, TensorType::f32_vector(8));
        g.add_edge(3, 0, 4, 0, TensorType::f32_vector(8));
        g
    }

    // ---- Empty graph ----

    #[test]
    fn empty_graph_has_zero_root() {
        let g = Graph::new("empty");
        let tree = GraphMerkleTree::build(&g);
        assert_eq!(tree.root, [0u8; 32]);
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn empty_graph_prove_returns_none() {
        let g = Graph::new("empty");
        let tree = GraphMerkleTree::build(&g);
        assert!(tree.prove(0).is_none());
    }

    // ---- Single-node graph ----

    #[test]
    fn single_node_tree() {
        let g = make_graph(1);
        let tree = GraphMerkleTree::build(&g);

        assert_eq!(tree.leaf_count(), 1);
        // Root should equal the single node hash
        assert_eq!(tree.root, tree.node_hashes[0].1);
        // Only one level (the leaf level, which is also the root)
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn single_node_proof_verifies() {
        let g = make_graph(1);
        let tree = GraphMerkleTree::build(&g);
        let proof = tree.prove(0).unwrap();
        assert!(GraphMerkleTree::verify_proof(&proof));
        // No siblings needed for a single-node tree
        assert_eq!(proof.siblings.len(), 0);
    }

    // ---- Two-node graph ----

    #[test]
    fn two_node_tree() {
        let g = make_graph(2);
        let tree = GraphMerkleTree::build(&g);

        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.depth(), 2); // leaf level + root level

        // Both proofs verify
        let p0 = tree.prove(0).unwrap();
        let p1 = tree.prove(1).unwrap();
        assert!(GraphMerkleTree::verify_proof(&p0));
        assert!(GraphMerkleTree::verify_proof(&p1));

        // Both have the same root
        assert_eq!(p0.root, p1.root);
        assert_eq!(p0.root, tree.root);
    }

    // ---- Various graph sizes ----

    #[test]
    fn power_of_two_nodes() {
        for &n in &[2, 4, 8, 16] {
            let g = make_graph(n);
            let tree = GraphMerkleTree::build(&g);
            assert_eq!(tree.leaf_count(), n);

            // Every node's proof verifies
            for i in 0..n {
                let proof = tree.prove(i as u32).unwrap();
                assert!(
                    GraphMerkleTree::verify_proof(&proof),
                    "proof failed for node {} in {}-node tree",
                    i,
                    n
                );
            }
        }
    }

    #[test]
    fn non_power_of_two_nodes() {
        // Odd-count trees require duplicate-last-hash handling
        for &n in &[3, 5, 7, 9, 13] {
            let g = make_graph(n);
            let tree = GraphMerkleTree::build(&g);
            assert_eq!(tree.leaf_count(), n);

            for i in 0..n {
                let proof = tree.prove(i as u32).unwrap();
                assert!(
                    GraphMerkleTree::verify_proof(&proof),
                    "proof failed for node {} in {}-node tree",
                    i,
                    n
                );
            }
        }
    }

    // ---- Varied ops graph ----

    #[test]
    fn varied_ops_graph_all_proofs_verify() {
        let g = make_varied_graph();
        let tree = GraphMerkleTree::build(&g);
        assert_eq!(tree.leaf_count(), 5);

        for i in 0..5u32 {
            let proof = tree.prove(i).unwrap();
            assert!(GraphMerkleTree::verify_proof(&proof));
        }
    }

    // ---- Determinism ----

    #[test]
    fn build_is_deterministic() {
        let g = make_varied_graph();
        let t1 = GraphMerkleTree::build(&g);
        let t2 = GraphMerkleTree::build(&g);
        assert_eq!(t1.root, t2.root);
        assert_eq!(t1.node_hashes.len(), t2.node_hashes.len());
        for (a, b) in t1.node_hashes.iter().zip(t2.node_hashes.iter()) {
            assert_eq!(a, b);
        }
    }

    // ---- Different graphs produce different roots ----

    #[test]
    fn different_graphs_different_roots() {
        let g1 = make_graph(3);
        let g2 = make_varied_graph();
        let t1 = GraphMerkleTree::build(&g1);
        let t2 = GraphMerkleTree::build(&g2);
        assert_ne!(t1.root, t2.root);
    }

    // ---- Tampered proof detection ----

    #[test]
    fn tampered_node_hash_fails() {
        let g = make_graph(4);
        let tree = GraphMerkleTree::build(&g);
        let mut proof = tree.prove(1).unwrap();
        // Flip a bit in the node hash
        proof.node_hash[0] ^= 0xff;
        assert!(!GraphMerkleTree::verify_proof(&proof));
    }

    #[test]
    fn tampered_sibling_hash_fails() {
        let g = make_graph(4);
        let tree = GraphMerkleTree::build(&g);
        let mut proof = tree.prove(2).unwrap();
        assert!(!proof.siblings.is_empty());
        // Flip a bit in the first sibling hash
        proof.siblings[0].0[0] ^= 0xff;
        assert!(!GraphMerkleTree::verify_proof(&proof));
    }

    #[test]
    fn tampered_root_fails() {
        let g = make_graph(4);
        let tree = GraphMerkleTree::build(&g);
        let mut proof = tree.prove(0).unwrap();
        // Flip a bit in the root
        proof.root[31] ^= 0x01;
        assert!(!GraphMerkleTree::verify_proof(&proof));
    }

    #[test]
    fn swapped_sibling_direction_fails() {
        let g = make_graph(4);
        let tree = GraphMerkleTree::build(&g);
        let mut proof = tree.prove(0).unwrap();
        if !proof.siblings.is_empty() {
            // Flip the is_left flag on the first sibling
            proof.siblings[0].1 = !proof.siblings[0].1;
            // This should change the computed root (hash(A||B) != hash(B||A) in general)
            // We can only assert it fails if it actually changes the result,
            // which it will for distinct sibling hashes.
            let result = GraphMerkleTree::verify_proof(&proof);
            // In a 4-node tree, node 0's sibling is node 1, which has a different hash,
            // so flipping the direction MUST change the result.
            assert!(!result);
        }
    }

    // ---- Prove for nonexistent node returns None ----

    #[test]
    fn prove_nonexistent_node_returns_none() {
        let g = make_graph(4);
        let tree = GraphMerkleTree::build(&g);
        assert!(tree.prove(99).is_none());
    }

    // ---- Large graph stress test ----

    #[test]
    fn large_graph_100_nodes() {
        let g = make_graph(100);
        let tree = GraphMerkleTree::build(&g);
        assert_eq!(tree.leaf_count(), 100);

        // Spot-check a few proofs
        for &i in &[0, 1, 49, 50, 98, 99] {
            let proof = tree.prove(i).unwrap();
            assert!(
                GraphMerkleTree::verify_proof(&proof),
                "proof failed for node {} in 100-node tree",
                i
            );
        }
    }

    // ---- Cross-proof: proof from one tree doesn't verify against another ----

    #[test]
    fn cross_tree_proof_fails() {
        let g1 = make_graph(4);
        let g2 = make_graph(8);
        let t1 = GraphMerkleTree::build(&g1);
        let t2 = GraphMerkleTree::build(&g2);

        let proof_from_t1 = t1.prove(0).unwrap();
        // The proof's root is from t1, so it shouldn't match t2's root
        assert_ne!(proof_from_t1.root, t2.root);
    }
}
