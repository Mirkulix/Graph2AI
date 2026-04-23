use qlang_core::graph::Graph;

const OPEN_TAG: &str = "<qo:qlms>";
const CLOSE_TAG: &str = "</qo:qlms>";

/// Scans the given LLM output text for `<qo:qlms>...</qo:qlms>` blocks.
/// For each block, attempts to parse the inner content as JSON into a `Graph`.
/// Malformed JSON or invalid structures are skipped.
pub fn extract_graphs(text: &str) -> Vec<Graph> {
    let mut out = Vec::new();
    let mut cursor = 0usize;

    while cursor < text.len() {
        let Some(start) = text[cursor..].find(OPEN_TAG) else {
            break;
        };
        let tag_start = cursor + start;
        let body_start = tag_start + OPEN_TAG.len();

        let Some(close_rel) = text[body_start..].find(CLOSE_TAG) else {
            break;
        };
        let body_end = body_start + close_rel;
        let body = &text[body_start..body_end];

        // Attempt to parse the body as a JSON serialized Graph
        if let Ok(graph) = serde_json::from_str::<Graph>(body.trim()) {
            out.push(graph);
        } else {
            tracing::warn!("Found <qo:qlms> block but failed to parse JSON as Graph");
        }

        cursor = body_end + CLOSE_TAG.len();
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_graph_json() {
        let json = r#"
        {
            "id": "test_graph",
            "version": "0.1",
            "nodes": [],
            "edges": [],
            "constraints": [],
            "metadata": {}
        }
        "#;
        
        let text = format!("Some text.\n<qo:qlms>\n{}\n</qo:qlms>\nEnd text.", json);
        
        let graphs = extract_graphs(&text);
        assert_eq!(graphs.len(), 1);
        assert_eq!(graphs[0].id, "test_graph");
    }
}
