use std::process::Command;

/// Tool result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool: String,
    pub success: bool,
    pub output: String,
}

/// Read a local file
pub fn tool_read_file(path: &str) -> ToolResult {
    match std::fs::read_to_string(path) {
        Ok(content) => ToolResult {
            tool: "read_file".into(),
            success: true,
            output: if content.len() > 2000 {
                format!("{}...[truncated]", &content[..2000])
            } else {
                content
            },
        },
        Err(e) => ToolResult {
            tool: "read_file".into(),
            success: false,
            output: format!("Error: {e}"),
        },
    }
}

/// List files in a directory
pub fn tool_list_dir(path: &str) -> ToolResult {
    match std::fs::read_dir(path) {
        Ok(entries) => {
            let files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if is_dir {
                        format!("{name}/")
                    } else {
                        name
                    }
                })
                .collect();
            ToolResult {
                tool: "list_dir".into(),
                success: true,
                output: files.join("\n"),
            }
        }
        Err(e) => ToolResult {
            tool: "list_dir".into(),
            success: false,
            output: format!("Error: {e}"),
        },
    }
}

/// Execute a shell command (sandboxed — only allowed commands)
pub fn tool_shell(cmd: &str) -> ToolResult {
    // Allowlist of safe commands
    let allowed = [
        "ls", "cat", "head", "wc", "grep", "find", "date", "whoami", "uname", "df", "free",
        "uptime", "cargo", "git", "npm",
    ];
    let first_word = cmd.split_whitespace().next().unwrap_or("");

    if !allowed
        .iter()
        .any(|&a| first_word == a || first_word.ends_with(a))
    {
        return ToolResult {
            tool: "shell".into(),
            success: false,
            output: format!(
                "Nicht erlaubt: '{first_word}'. Erlaubte Befehle: {}",
                allowed.join(", ")
            ),
        };
    }

    match Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let out = if stdout.is_empty() {
                stderr.to_string()
            } else {
                stdout.to_string()
            };
            ToolResult {
                tool: "shell".into(),
                success: output.status.success(),
                output: if out.len() > 3000 {
                    format!("{}...[truncated]", &out[..3000])
                } else {
                    out
                },
            }
        }
        Err(e) => ToolResult {
            tool: "shell".into(),
            success: false,
            output: format!("Error: {e}"),
        },
    }
}

/// Search the web via DuckDuckGo Instant Answer API (no API key needed, JSON response)
pub async fn tool_web_search(query: &str) -> ToolResult {
    let client = reqwest::Client::new();
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding::encode(query)
    );

    match client
        .get(&url)
        .header("User-Agent", "QO/0.1")
        .send()
        .await
    {
        Ok(resp) => match resp.text().await {
            Ok(body) => {
                let mut parts = Vec::new();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    // Abstract text (main summary)
                    if let Some(text) = json.get("AbstractText").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            parts.push(format!("Zusammenfassung: {text}"));
                        }
                    }
                    // Abstract source
                    if let Some(src) = json.get("AbstractSource").and_then(|v| v.as_str()) {
                        if !src.is_empty() {
                            parts.push(format!("Quelle: {src}"));
                        }
                    }
                    // Related topics
                    if let Some(topics) = json.get("RelatedTopics").and_then(|v| v.as_array()) {
                        for topic in topics.iter().take(5) {
                            if let Some(text) = topic.get("Text").and_then(|v| v.as_str()) {
                                if !text.is_empty() {
                                    parts.push(format!("- {}", &text[..text.len().min(200)]));
                                }
                            }
                        }
                    }
                }
                ToolResult {
                    tool: "web_search".into(),
                    success: !parts.is_empty(),
                    output: if parts.is_empty() {
                        format!("Keine direkten Ergebnisse für '{}'. DuckDuckGo Instant Answer API liefert nur bei bekannten Themen.", query)
                    } else {
                        parts.join("\n\n")
                    },
                }
            }
            Err(e) => ToolResult {
                tool: "web_search".into(),
                success: false,
                output: format!("Error: {e}"),
            },
        },
        Err(e) => ToolResult {
            tool: "web_search".into(),
            success: false,
            output: format!("Error: {e}"),
        },
    }
}

#[allow(dead_code)] // Utility kept for HTML parsing in future tool outputs
fn strip_html_tags(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    result
}

/// Deterministic value check — NOT an LLM call
pub fn tool_values_check(
    action_description: &str,
    values: &qo_values::ValueScores,
) -> ToolResult {
    let mut issues = Vec::new();
    let mut score = 1.0f32;

    let desc_lower = action_description.to_lowercase();

    // Achtsamkeit: check for hasty/careless language
    if desc_lower.contains("schnell")
        || desc_lower.contains("sofort")
        || desc_lower.contains("ohne prüfung")
    {
        issues.push("Achtsamkeit: Aktion wirkt übereilt");
        score -= 0.2;
    }

    // Sinn: check if action has clear purpose
    if action_description.len() < 10 {
        issues.push("Sinn: Beschreibung zu kurz, Zweck unklar");
        score -= 0.1;
    }

    // Overall value alignment
    let avg = values.average();
    if avg < 0.3 {
        issues.push("Werte insgesamt niedrig — System braucht Reflexion");
        score -= 0.2;
    }

    ToolResult {
        tool: "values_check".into(),
        success: issues.is_empty(),
        output: if issues.is_empty() {
            format!("✓ Werte-Check bestanden (Score: {score:.1})")
        } else {
            format!(
                "⚠ Werte-Check: {}\nScore: {score:.1}",
                issues.join("; ")
            )
        },
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_check_is_deterministic() {
        let values = qo_values::ValueScores::default();
        let result1 = tool_values_check("Recherchiere Marktdaten sorgfältig", &values);
        let result2 = tool_values_check("Recherchiere Marktdaten sorgfältig", &values);
        // Same input = same output (no LLM randomness)
        assert_eq!(result1.output, result2.output);
        assert!(result1.success); // should pass
    }

    #[test]
    fn values_check_catches_hasty() {
        let values = qo_values::ValueScores::default();
        let result = tool_values_check("Mach das schnell ohne Prüfung", &values);
        assert!(!result.success); // should fail
        assert!(result.output.contains("Achtsamkeit"));
    }

    #[test]
    fn shell_blocks_dangerous_commands() {
        let result = tool_shell("rm -rf /");
        assert!(!result.success);
        assert!(result.output.contains("Nicht erlaubt"));
    }

    #[test]
    fn shell_allows_safe_commands() {
        let result = tool_shell("date");
        assert!(result.success);
    }

    #[test]
    fn read_file_works() {
        let result = tool_read_file("Cargo.toml");
        assert!(result.success);
        assert!(
            result.output.contains("[package]") || result.output.contains("[workspace]")
        );
    }

    #[test]
    fn list_dir_works() {
        let result = tool_list_dir(".");
        assert!(result.success);
        assert!(
            result.output.contains("Cargo.toml") || result.output.contains("qo/")
        );
    }
}
