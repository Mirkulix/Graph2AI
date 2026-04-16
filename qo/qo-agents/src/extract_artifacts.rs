//! Parses file-artifact tags out of LLM output.
//!
//! An agent writes a file by emitting a block like:
//!
//! ```text
//! <qo:file path="workspace/src/main.py">
//! def main():
//!     print("hi")
//! </qo:file>
//! ```
//!
//! The orchestrator scans every subtask result for these blocks and
//! POSTs each one to `/api/tools/write_file` so the file actually
//! lands on disk — no LLM function-calling required, works with every
//! model we speak to (Groq, DeepSeek, Ollama, …).
//!
//! Intentionally small and regex-free: a hand-rolled scan avoids a
//! dependency on `regex` (already in the graph but we keep this crate
//! lean) and behaves predictably on multi-line inputs with embedded
//! HTML-lookalike content.

use serde::{Deserialize, Serialize};

/// Token that opens a file artefact. Everything between the opening
/// `>` and the closing tag is treated as the file body verbatim.
const OPEN_TAG: &str = "<qo:file";
const CLOSE_TAG: &str = "</qo:file>";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// Workspace-relative path. Sanitisation happens on the server side.
    pub path: String,
    pub content: String,
}

/// Walk `text` and emit every well-formed `<qo:file path="...">…</qo:file>`
/// block in order of appearance. Malformed / unterminated blocks are
/// silently skipped — we would rather drop a bogus artefact than
/// surface a half-parsed one.
pub fn extract_artifacts(text: &str) -> Vec<Artifact> {
    let mut out: Vec<Artifact> = Vec::new();
    let mut cursor = 0usize;
    let bytes = text.as_bytes();

    while cursor < text.len() {
        let Some(start) = text[cursor..].find(OPEN_TAG) else {
            break;
        };
        let tag_start = cursor + start;
        let after_open = tag_start + OPEN_TAG.len();
        // Need a path="..." attribute. Accept single or double quotes,
        // tolerate extra whitespace.
        let Some(tag_end_rel) = text[after_open..].find('>') else {
            break;
        };
        let tag_end = after_open + tag_end_rel;
        let attrs = &text[after_open..tag_end];
        let Some(path) = extract_path_attr(attrs) else {
            // Malformed attribute — skip past this open tag and keep looking.
            cursor = tag_end + 1;
            continue;
        };

        let body_start = tag_end + 1; // past the '>'
        let Some(close_rel) = text[body_start..].find(CLOSE_TAG) else {
            break;
        };
        let body_end = body_start + close_rel;
        let mut body = text[body_start..body_end].to_string();
        // Strip a single leading newline so the rendered file does not
        // accidentally start with a blank line from the formatting.
        if body.starts_with('\n') {
            body.remove(0);
        } else if body.starts_with("\r\n") {
            body.drain(0..2);
        }
        // Validate path is non-empty + not absolute + no parent-dir.
        if is_safe_relative(&path) && !body.is_empty() {
            out.push(Artifact {
                path,
                content: body,
            });
        }

        cursor = body_end + CLOSE_TAG.len();
    }

    // Prevent returning a Vec that borrows `bytes` past the loop.
    let _ = bytes;
    out
}

fn extract_path_attr(attrs: &str) -> Option<String> {
    let lower = attrs.to_ascii_lowercase();
    let idx = lower.find("path")?;
    // Walk forward past optional whitespace + '=' + quote.
    let rest = &attrs[idx + 4..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=')?.trim_start();
    let (quote, rest) = match rest.as_bytes().first() {
        Some(b'"') => ('"', &rest[1..]),
        Some(b'\'') => ('\'', &rest[1..]),
        _ => return None,
    };
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn is_safe_relative(path: &str) -> bool {
    if path.is_empty() || path.contains('\0') {
        return false;
    }
    if path.starts_with('/') || path.starts_with('\\') {
        return false;
    }
    if path.len() >= 2 && path.as_bytes()[1] == b':' {
        return false; // drive letter on Windows
    }
    for seg in path.split(|c: char| c == '/' || c == '\\') {
        if seg == ".." {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_single_file() {
        let text = r#"
Vorab-Analyse.

<qo:file path="src/main.py">
def main():
    print("hi")
</qo:file>

Ende.
"#;
        let out = extract_artifacts(text);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "src/main.py");
        assert!(out[0].content.starts_with("def main():"));
        assert!(out[0].content.ends_with("print(\"hi\")\n"));
    }

    #[test]
    fn extracts_multiple_files_in_order() {
        let text = r#"
<qo:file path="a.txt">alpha</qo:file>
prose
<qo:file path="sub/b.txt">bravo</qo:file>
"#;
        let out = extract_artifacts(text);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].path, "a.txt");
        assert_eq!(out[0].content, "alpha");
        assert_eq!(out[1].path, "sub/b.txt");
        assert_eq!(out[1].content, "bravo");
    }

    #[test]
    fn ignores_unterminated_block() {
        let text = r#"
<qo:file path="broken.txt">
never closed ...
"#;
        let out = extract_artifacts(text);
        assert!(out.is_empty());
    }

    #[test]
    fn rejects_parent_dir() {
        let text = r#"<qo:file path="../evil.txt">nope</qo:file>"#;
        assert!(extract_artifacts(text).is_empty());
    }

    #[test]
    fn rejects_absolute_path() {
        let text = r#"<qo:file path="/etc/passwd">nope</qo:file>"#;
        assert!(extract_artifacts(text).is_empty());
        let text_win = r#"<qo:file path="C:\Windows\system32">nope</qo:file>"#;
        assert!(extract_artifacts(text_win).is_empty());
    }

    #[test]
    fn accepts_single_quotes() {
        let text = "<qo:file path='foo.md'>hello</qo:file>";
        let out = extract_artifacts(text);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "foo.md");
    }

    #[test]
    fn empty_body_is_skipped() {
        let text = r#"<qo:file path="zero.txt"></qo:file>"#;
        assert!(extract_artifacts(text).is_empty());
    }
}
