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

/// Search the web across three sources, merging what's available:
///
///   1. **Tavily** (when `TAVILY_API_KEY` is set) — indexed web content,
///      current news, snippets with URLs. Best quality. Free tier: 1000
///      searches/month.
///   2. **Wikipedia** (always free, no key) — encyclopedic articles in
///      the user's language + English. Great for factual grounding;
///      complements Tavily's fresh-content strength.
///   3. **DuckDuckGo Instant Answer** — last-resort fallback that only
///      returns well-known Wikipedia-like topics. Kept because it
///      works offline-ish (no auth, minimal rate limit).
///
/// The implementation runs Tavily and Wikipedia **in parallel** when
/// Tavily is configured, then merges their outputs. If Tavily is
/// absent, Wikipedia alone fills the gap — no single-source blind
/// spot. DuckDuckGo only kicks in when both higher-quality sources
/// return empty.
pub async fn tool_web_search(query: &str) -> ToolResult {
    let tavily_key = std::env::var("TAVILY_API_KEY").ok().filter(|k| !k.trim().is_empty());

    let (tavily_res, wiki_res) = if let Some(key) = &tavily_key {
        let (a, b) = tokio::join!(tavily_search(key, query), wikipedia_search(query));
        (Some(a), Some(b))
    } else {
        (None, Some(wikipedia_search(query).await))
    };

    let mut merged = Vec::new();
    let mut sources = Vec::new();

    if let Some(Ok(r)) = &tavily_res {
        if r.success {
            sources.push("Tavily");
            merged.push(format!("### Websuche (Tavily)\n\n{}", r.output));
        }
    }
    if let Some(Ok(r)) = &wiki_res {
        if r.success {
            sources.push("Wikipedia");
            merged.push(format!("### Wikipedia\n\n{}", r.output));
        }
    }

    if !merged.is_empty() {
        return ToolResult {
            tool: format!("web_search[{}]", sources.join("+")),
            success: true,
            output: merged.join("\n\n---\n\n"),
        };
    }

    // Last resort — DuckDuckGo Instant Answer.
    duckduckgo_search(query).await
}

/// Wikipedia Search + Summary API. Queries the user's preferred
/// language (env `WIKI_LANG`, default `de`) with an English fallback.
/// No API key. Returns up to 3 articles with snippets.
async fn wikipedia_search(query: &str) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let primary_lang = std::env::var("WIKI_LANG").unwrap_or_else(|_| "de".to_string());
    let langs = if primary_lang == "en" {
        vec!["en"]
    } else {
        vec![primary_lang.as_str(), "en"]
    };

    let client = reqwest::Client::new();
    let mut all_hits: Vec<(String, String, String)> = Vec::new(); // (title, snippet, url)

    for lang in &langs {
        let url = format!(
            "https://{lang}.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}&srlimit=3&srprop=snippet",
            urlencoding::encode(query)
        );
        let resp = match client
            .get(&url)
            .header("User-Agent", "QO/0.1 (qlang research agent)")
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) => r,
            Err(_) => continue,
        };
        if !resp.status().is_success() {
            continue;
        }
        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(_) => continue,
        };
        if let Some(hits) = json
            .get("query")
            .and_then(|v| v.get("search"))
            .and_then(|v| v.as_array())
        {
            for hit in hits.iter().take(3) {
                let title = hit
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let snippet_html = hit
                    .get("snippet")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let snippet = strip_html_tags(&snippet_html).trim().to_string();
                if title.is_empty() {
                    continue;
                }
                let url_out = format!(
                    "https://{lang}.wikipedia.org/wiki/{}",
                    urlencoding::encode(&title.replace(' ', "_"))
                );
                all_hits.push((title, snippet, url_out));
            }
        }
        if !all_hits.is_empty() {
            break; // primary language already found results
        }
    }

    if all_hits.is_empty() {
        return Ok(ToolResult {
            tool: "web_search[wikipedia]".into(),
            success: false,
            output: format!("Wikipedia liefert keine Treffer für '{query}'."),
        });
    }

    let mut parts = Vec::new();
    for (title, snippet, url) in &all_hits {
        parts.push(format!(
            "**{title}**\n{snippet}\n({url})",
            title = title,
            snippet = if snippet.is_empty() { "_(kein Snippet)_" } else { snippet.as_str() },
            url = url
        ));
    }
    Ok(ToolResult {
        tool: "web_search[wikipedia]".into(),
        success: true,
        output: parts.join("\n\n"),
    })
}

async fn tavily_search(
    api_key: &str,
    query: &str,
) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let body = serde_json::json!({
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": true,
    });
    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.tavily.com/search")
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Tavily HTTP {status}: {text}").into());
    }
    let json: serde_json::Value = resp.json().await?;

    let mut parts = Vec::new();
    if let Some(answer) = json.get("answer").and_then(|v| v.as_str()) {
        if !answer.is_empty() {
            parts.push(format!("Zusammenfassung: {answer}"));
        }
    }
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        for r in results.iter().take(5) {
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("");
            let content = r.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let url = r.get("url").and_then(|v| v.as_str()).unwrap_or("");
            parts.push(format!(
                "- {}\n  {}\n  ({})",
                title,
                &content[..content.len().min(200)],
                url
            ));
        }
    }
    Ok(ToolResult {
        tool: "web_search[tavily]".into(),
        success: !parts.is_empty(),
        output: if parts.is_empty() {
            format!("Tavily lieferte keine Treffer für '{query}'.")
        } else {
            parts.join("\n\n")
        },
    })
}

async fn duckduckgo_search(query: &str) -> ToolResult {
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
