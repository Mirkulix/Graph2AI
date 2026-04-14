//! QLANG Package System — Reusable graph libraries.
//!
//! A package is a collection of named graphs that can be imported
//! and composed into larger programs.
//!
//! Example:
//! ```qlang
//! // Package: igqk_layers
//! graph dense_layer {
//!   input x: f32[?, input_dim]
//!   input W: f32[input_dim, output_dim]
//!   input b: f32[output_dim]
//!   node h = matmul(x, W)
//!   node biased = add(h, b)
//!   node activated = relu(biased)
//!   output y = activated
//! }
//!
//! graph ternary_compress {
//!   input weights: f32[?, ?]
//!   node compressed = to_ternary(weights) @proof theorem_5_2
//!   output compressed_weights = compressed
//! }
//! ```

use qlang_core::graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A QLANG package: a named collection of reusable graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Package {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub graphs: HashMap<String, Graph>,
    pub dependencies: Vec<PackageDep>,
}

/// A package dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageDep {
    pub name: String,
    pub version: String,
}

impl Package {
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: String::new(),
            author: String::new(),
            graphs: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    /// Add a graph to the package.
    pub fn add_graph(&mut self, graph: Graph) {
        self.graphs.insert(graph.id.clone(), graph);
    }

    /// Get a graph by name.
    pub fn get_graph(&self, name: &str) -> Option<&Graph> {
        self.graphs.get(name)
    }

    /// List all graph names.
    pub fn graph_names(&self) -> Vec<&str> {
        self.graphs.keys().map(|s| s.as_str()).collect()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Total number of nodes across all graphs.
    pub fn total_nodes(&self) -> usize {
        self.graphs.values().map(|g| g.nodes.len()).sum()
    }
}

/// A package registry: stores and resolves packages.
#[derive(Debug, Default)]
pub struct Registry {
    packages: HashMap<String, Package>,
}

impl Registry {
    pub fn new() -> Self {
        Self { packages: HashMap::new() }
    }

    /// Register a package.
    pub fn register(&mut self, package: Package) {
        self.packages.insert(package.name.clone(), package);
    }

    /// Get a package by name.
    pub fn get(&self, name: &str) -> Option<&Package> {
        self.packages.get(name)
    }

    /// List all packages.
    pub fn list(&self) -> Vec<(&str, &str)> {
        self.packages.values()
            .map(|p| (p.name.as_str(), p.version.as_str()))
            .collect()
    }

    /// Create the standard library package.
    pub fn load_stdlib(&mut self) {
        use qlang_core::ops::Op;
        use qlang_core::tensor::TensorType;

        let mut stdlib = Package::new("std", "0.1.0");
        stdlib.description = "QLANG Standard Library".into();
        stdlib.author = "QLANG Team".into();

        // Standard dense layer graph
        {
            let mut g = Graph::new("dense");
            let x = g.add_node(Op::Input { name: "x".into() }, vec![], vec![TensorType::f32_matrix(1, 128)]);
            let w = g.add_node(Op::Input { name: "W".into() }, vec![], vec![TensorType::f32_matrix(128, 64)]);
            let mm = g.add_node(Op::MatMul,
                vec![TensorType::f32_matrix(1, 128), TensorType::f32_matrix(128, 64)],
                vec![TensorType::f32_matrix(1, 64)]);
            let relu = g.add_node(Op::Relu,
                vec![TensorType::f32_matrix(1, 64)],
                vec![TensorType::f32_matrix(1, 64)]);
            let out = g.add_node(Op::Output { name: "y".into() },
                vec![TensorType::f32_matrix(1, 64)], vec![]);
            g.add_edge(x, 0, mm, 0, TensorType::f32_matrix(1, 128));
            g.add_edge(w, 0, mm, 1, TensorType::f32_matrix(128, 64));
            g.add_edge(mm, 0, relu, 0, TensorType::f32_matrix(1, 64));
            g.add_edge(relu, 0, out, 0, TensorType::f32_matrix(1, 64));
            stdlib.add_graph(g);
        }

        // IGQK compression graph
        {
            let mut g = Graph::new("igqk_ternary");
            let w = g.add_node(Op::Input { name: "weights".into() }, vec![], vec![TensorType::f32_matrix(128, 128)]);
            let t = g.add_node(Op::ToTernary,
                vec![TensorType::f32_matrix(128, 128)],
                vec![TensorType::ternary_matrix(128, 128)]);
            let out = g.add_node(Op::Output { name: "compressed".into() },
                vec![TensorType::ternary_matrix(128, 128)], vec![]);
            g.add_edge(w, 0, t, 0, TensorType::f32_matrix(128, 128));
            g.add_edge(t, 0, out, 0, TensorType::ternary_matrix(128, 128));
            stdlib.add_graph(g);
        }

        self.register(stdlib);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qlang_core::graph::Graph;

    #[test]
    fn package_basics() {
        let mut pkg = Package::new("test_pkg", "1.0.0");
        pkg.add_graph(Graph::new("graph_a"));
        pkg.add_graph(Graph::new("graph_b"));

        assert_eq!(pkg.graph_names().len(), 2);
        assert!(pkg.get_graph("graph_a").is_some());
    }

    #[test]
    fn package_json_roundtrip() {
        let mut pkg = Package::new("test", "0.1.0");
        pkg.description = "Test package".into();
        pkg.add_graph(Graph::new("my_graph"));

        let json = pkg.to_json().unwrap();
        let pkg2 = Package::from_json(&json).unwrap();

        assert_eq!(pkg2.name, "test");
        assert_eq!(pkg2.graphs.len(), 1);
    }

    #[test]
    fn registry_stdlib() {
        let mut registry = Registry::new();
        registry.load_stdlib();

        let stdlib = registry.get("std").unwrap();
        assert!(stdlib.get_graph("dense").is_some());
        assert!(stdlib.get_graph("igqk_ternary").is_some());
        assert!(stdlib.total_nodes() > 0);
    }
}
