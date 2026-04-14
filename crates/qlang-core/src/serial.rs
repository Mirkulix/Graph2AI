use crate::graph::Graph;

/// Magic bytes for QLANG binary format: "QLAN"
pub const MAGIC: [u8; 4] = [0x51, 0x4C, 0x41, 0x4E];

/// Current wire format version.
pub const WIRE_VERSION: u16 = 1;

/// Serialize a graph to JSON (human-readable format).
pub fn to_json(graph: &Graph) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(graph)
}

/// Deserialize a graph from JSON.
pub fn from_json(json: &str) -> Result<Graph, serde_json::Error> {
    serde_json::from_str(json)
}

/// Serialize a graph to compact binary format (.qlg).
///
/// Format:
/// - 4 bytes: magic "QLAN"
/// - 2 bytes: wire version (u16 LE)
/// - 2 bytes: flags (reserved)
/// - remaining: MessagePack-encoded graph (compact binary)
///
/// For Phase 1, we use JSON inside the binary envelope.
/// Phase 2 will use a true binary encoding for maximum efficiency.
pub fn to_binary(graph: &Graph) -> Result<Vec<u8>, serde_json::Error> {
    let json_bytes = serde_json::to_vec(graph)?;

    let mut buf = Vec::with_capacity(8 + json_bytes.len());
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // flags
    buf.extend_from_slice(&json_bytes);

    Ok(buf)
}

/// Deserialize a graph from binary format (.qlg).
pub fn from_binary(data: &[u8]) -> Result<Graph, SerialError> {
    if data.len() < 8 {
        return Err(SerialError::TooShort);
    }

    if data[0..4] != MAGIC {
        return Err(SerialError::InvalidMagic);
    }

    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != WIRE_VERSION {
        return Err(SerialError::UnsupportedVersion(version));
    }

    // Skip flags (bytes 6-7)
    let json_bytes = &data[8..];

    serde_json::from_slice(json_bytes).map_err(SerialError::Json)
}

#[derive(Debug, thiserror::Error)]
pub enum SerialError {
    #[error("data too short for QLANG binary format")]
    TooShort,
    #[error("invalid magic bytes (expected QLAN)")]
    InvalidMagic,
    #[error("unsupported wire version {0}")]
    UnsupportedVersion(u16),
    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::Op;
    use crate::tensor::TensorType;

    #[test]
    fn json_roundtrip() {
        let mut g = Graph::new("json_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_vector(10)],
        );

        let json = to_json(&g).unwrap();
        let g2 = from_json(&json).unwrap();

        assert_eq!(g.id, g2.id);
        assert_eq!(g.nodes.len(), g2.nodes.len());
    }

    #[test]
    fn binary_roundtrip() {
        let mut g = Graph::new("binary_test");
        g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            vec![TensorType::f32_matrix(28, 28)],
        );
        g.add_node(Op::ToTernary, vec![TensorType::f32_matrix(28, 28)], vec![TensorType::ternary_matrix(28, 28)]);
        g.add_edge(0, 0, 1, 0, TensorType::f32_matrix(28, 28));

        let binary = to_binary(&g).unwrap();

        // Check magic bytes
        assert_eq!(&binary[0..4], &MAGIC);

        let g2 = from_binary(&binary).unwrap();
        assert_eq!(g.id, g2.id);
        assert_eq!(g.nodes.len(), g2.nodes.len());
        assert_eq!(g.edges.len(), g2.edges.len());
    }

    #[test]
    fn invalid_magic_rejected() {
        let bad_data = vec![0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
        assert!(matches!(
            from_binary(&bad_data),
            Err(SerialError::InvalidMagic)
        ));
    }
}
