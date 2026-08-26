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

/// Read a file from the sandboxed workspace directory. Agents call this
/// when the user references `READ:rel/path.ext` in a task — it resolves
/// the path under `<QO_DATA_DIR or "data">/workspace/` and enforces the
/// same safety contract as the server's workspace route:
///
///   * absolute paths → refused
///   * `..` traversal → refused
///   * NUL byte       → refused
///   * empty path     → refused
///
/// Tolerates a leading `workspace/` prefix (agents sometimes include it)
/// the same way the server route does. Reads up to the first 6 KB of
/// the file and returns it as-is — simple head-cap, no middle-strip.
pub fn tool_read_workspace_file(rel_path: &str) -> ToolResult {
    const LIMIT: usize = 6 * 1024;

    // Path-safety gate — must mirror the server's workspace route.
    if rel_path.is_empty()
        || rel_path.contains("..")
        || rel_path.starts_with('/')
        || rel_path.starts_with('\\')
        || rel_path.contains('\0')
    {
        return ToolResult {
            tool: "read_workspace_file".into(),
            success: false,
            output: format!("Unzulässiger Pfad: '{rel_path}' (absolut, enthält .. oder NUL)"),
        };
    }

    // Agents sometimes prepend `workspace/` — strip it to match the
    // server's input tolerance.
    let rel = rel_path
        .strip_prefix("workspace/")
        .or_else(|| rel_path.strip_prefix("workspace\\"))
        .unwrap_or(rel_path);

    let data_dir = std::env::var("QO_DATA_DIR").unwrap_or_else(|_| "data".to_string());
    let full_path = std::path::Path::new(&data_dir).join("workspace").join(rel);

    match std::fs::read_to_string(&full_path) {
        Ok(content) => {
            let output = if content.len() > LIMIT {
                format!(
                    "{}\n\n… [Datei gekürzt — {} B insgesamt, nur die ersten {} B gezeigt] …",
                    &content[..LIMIT],
                    content.len(),
                    LIMIT
                )
            } else {
                content
            };
            ToolResult {
                tool: "read_workspace_file".into(),
                success: true,
                output,
            }
        }
        Err(e) => ToolResult {
            tool: "read_workspace_file".into(),
            success: false,
            output: format!("Konnte '{}' nicht lesen: {e}", full_path.display()),
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

/// Execute a read-only command from a fixed allowlist.
///
/// Two things here are load-bearing:
///
/// - The allowlist is matched **exactly**. It used to also accept anything
///   whose name merely *ended* with an allowed word, which let `/tmp/evilcat`
///   and `mygit` through — the check named the program without constraining
///   which program.
/// - The command is not handed to a shell. Splitting the arguments and
///   spawning the program directly means `ls; rm -rf ~` cannot smuggle a
///   second command past a passing first word. It also works on Windows,
///   where `sh` does not exist.
pub fn tool_shell(cmd: &str) -> ToolResult {
    let allowed = [
        "ls", "cat", "head", "wc", "grep", "find", "date", "whoami", "uname", "df", "free",
        "uptime", "cargo", "git", "npm",
    ];

    let mut parts = cmd.split_whitespace();
    let program = parts.next().unwrap_or("");
    let args: Vec<&str> = parts.collect();

    if !allowed.contains(&program) {
        return ToolResult {
            tool: "shell".into(),
            success: false,
            output: format!(
                "Nicht erlaubt: '{program}'. Erlaubte Befehle: {}",
                allowed.join(", ")
            ),
        };
    }

    // Shell metacharacters are meaningless without a shell, but their presence
    // means the caller expected one — so the result would not be what they
    // asked for. Say so instead of running something subtly different.
    if cmd.contains(['|', ';', '&', '>', '<', '`', '$']) {
        return ToolResult {
            tool: "shell".into(),
            success: false,
            output: "Shell-Operatoren (| ; & > < ` $) werden nicht unterstützt — \
                     Befehle laufen ohne Shell."
                .into(),
        };
    }

    match Command::new(program).args(&args).output() {
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

/// Search the web across several providers, merging whatever the
/// operator has configured:
///
///   1. **Tavily** (`TAVILY_API_KEY`) — indexed web content, current
///      news, snippets with URLs. Free tier: 1000 searches/month.
///   2. **SearXNG** (`SEARXNG_URL`) — self-hosted OSS meta-search
///      (aggregates Google, Bing, DDG, Brave, …). Privacy-respecting,
///      no key, free forever when you run it locally (docker image
///      is 2-line setup: `docker run -p 8080:8080 searxng/searxng`).
///   3. **Wikipedia** (always free, no key) — encyclopedic articles
///      in the user's language + English. Great for factual grounding.
///   4. **DuckDuckGo Instant Answer** — last-resort; only returns
///      well-known Wikipedia-like topics, but works with zero setup.
///
/// The implementation runs ALL configured providers **in parallel**
/// (tokio::join) and merges their outputs into one markdown blob the
/// LLM can read. If a provider errors or returns empty, it is simply
/// skipped. This way the researcher never has a single point of
/// failure — even without any API keys, Wikipedia + DDG still give it
/// something to work with.
pub async fn tool_web_search(query: &str) -> ToolResult {
    let tavily_key = std::env::var("TAVILY_API_KEY").ok().filter(|k| !k.trim().is_empty());
    let searxng_url = std::env::var("SEARXNG_URL").ok().filter(|u| !u.trim().is_empty());

    // Kick off every configured provider in parallel.
    let tavily_fut = async {
        match &tavily_key {
            Some(k) => Some(tavily_search(k, query).await),
            None => None,
        }
    };
    let searxng_fut = async {
        match &searxng_url {
            Some(u) => Some(searxng_search(u, query).await),
            None => None,
        }
    };
    let wiki_fut = wikipedia_search(query);

    let (tavily_res, searxng_res, wiki_res) =
        tokio::join!(tavily_fut, searxng_fut, wiki_fut);

    let mut merged = Vec::new();
    let mut sources = Vec::new();

    if let Some(Ok(r)) = &tavily_res {
        if r.success {
            sources.push("Tavily");
            merged.push(format!("### Websuche (Tavily)\n\n{}", r.output));
        }
    }
    if let Some(Ok(r)) = &searxng_res {
        if r.success {
            sources.push("SearXNG");
            merged.push(format!("### Meta-Suche (SearXNG)\n\n{}", r.output));
        }
    }
    if let Ok(r) = &wiki_res {
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

/// SearXNG is a self-hosted meta-search engine (MIT licensed,
/// github.com/searxng/searxng). It aggregates results from Google,
/// Bing, DuckDuckGo, Brave and dozens more — with zero per-query cost
/// and zero tracking. Expects a JSON-enabled instance; set the env
/// `SEARXNG_URL=http://localhost:8080` (or wherever the container
/// runs). Query parameter `format=json` needs to be whitelisted in
/// the instance's `settings.yml` (`formats: [html, json]`).
async fn searxng_search(
    base_url: &str,
    query: &str,
) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let trimmed = base_url.trim_end_matches('/');
    let url = format!("{trimmed}/search?q={}&format=json", urlencoding::encode(query));
    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "QO/0.1 (qlang research agent)")
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(format!("SearXNG HTTP {}: check settings.yml formats list", resp.status()).into());
    }
    let json: serde_json::Value = resp.json().await?;
    let mut parts = Vec::new();

    // "infoboxes" — direct structured answers (Wikipedia-style cards)
    if let Some(ibx) = json.get("infoboxes").and_then(|v| v.as_array()) {
        for ib in ibx.iter().take(1) {
            let title = ib.get("infobox").and_then(|v| v.as_str()).unwrap_or("");
            let content = ib.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if !content.is_empty() {
                parts.push(format!("**{title}**\n{content}"));
            }
        }
    }

    // "results" — the ranked search hits
    if let Some(results) = json.get("results").and_then(|v| v.as_array()) {
        for r in results.iter().take(5) {
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("");
            let content = r.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let url_out = r.get("url").and_then(|v| v.as_str()).unwrap_or("");
            if title.is_empty() {
                continue;
            }
            let trimmed_content = &content[..content.len().min(220)];
            parts.push(format!("- {title}\n  {trimmed_content}\n  ({url_out})"));
        }
    }

    Ok(ToolResult {
        tool: "web_search[searxng]".into(),
        success: !parts.is_empty(),
        output: if parts.is_empty() {
            format!("SearXNG lieferte keine Treffer für '{query}'.")
        } else {
            parts.join("\n\n")
        },
    })
}

/// Read a URL and return its content as clean markdown — what an LLM
/// actually wants to consume.
///
/// Provider chain:
///   1. **Firecrawl** (`FIRECRAWL_API_KEY` for cloud *or* `FIRECRAWL_URL`
///      for a self-hosted instance — mendableai/firecrawl is MIT on
///      GitHub). Best quality: headless Chrome, ad/nav/footer removal,
///      markdown with preserved structure.
///   2. **Jina Reader** (always free, no key: `https://r.jina.ai/{url}`).
///      Simple GET, returns plain text. Works on most sites.
///
/// Returns a short summary when the content is > 6 KB so the LLM does
/// not drown in nav/footer noise.
pub async fn tool_fetch_url(url: &str) -> ToolResult {
    // 1. Firecrawl — cloud or self-hosted
    let firecrawl_key = std::env::var("FIRECRAWL_API_KEY").ok().filter(|k| !k.trim().is_empty());
    let firecrawl_host = std::env::var("FIRECRAWL_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| "https://api.firecrawl.dev".to_string());
    if firecrawl_key.is_some() || firecrawl_host != "https://api.firecrawl.dev" {
        match firecrawl_scrape(&firecrawl_host, firecrawl_key.as_deref(), url).await {
            Ok(r) if r.success => return r,
            Ok(_) => {}
            Err(e) => tracing::warn!("Firecrawl failed: {e} — falling back to Jina Reader"),
        }
    }

    // 2. Jina Reader — zero-setup fallback
    match jina_reader(url).await {
        Ok(r) => r,
        Err(e) => ToolResult {
            tool: "fetch_url".into(),
            success: false,
            output: format!("Fehler beim Abruf: {e}"),
        },
    }
}

async fn firecrawl_scrape(
    host: &str,
    api_key: Option<&str>,
    target_url: &str,
) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = format!("{}/v1/scrape", host.trim_end_matches('/'));
    let body = serde_json::json!({
        "url": target_url,
        "formats": ["markdown"],
        "onlyMainContent": true,
    });
    let client = reqwest::Client::new();
    let mut req = client
        .post(&endpoint)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Firecrawl HTTP {status}: {text}").into());
    }
    let json: serde_json::Value = resp.json().await?;
    let markdown = json
        .get("data")
        .and_then(|d| d.get("markdown"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    Ok(ToolResult {
        tool: "fetch_url[firecrawl]".into(),
        success: !markdown.is_empty(),
        output: cap_reader_output(markdown),
    })
}

async fn jina_reader(
    target_url: &str,
) -> Result<ToolResult, Box<dyn std::error::Error + Send + Sync>> {
    let endpoint = format!("https://r.jina.ai/{target_url}");
    let client = reqwest::Client::new();
    let resp = client
        .get(&endpoint)
        .header("User-Agent", "QO/0.1 (qlang research agent)")
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(format!("Jina Reader HTTP {}", resp.status()).into());
    }
    let body = resp.text().await?;
    Ok(ToolResult {
        tool: "fetch_url[jina-reader]".into(),
        success: !body.is_empty(),
        output: cap_reader_output(&body),
    })
}

/// Keep a fetched page under a sane size so a text-heavy news article
/// doesn't blow up the LLM context. Head + tail are preserved.
fn cap_reader_output(s: &str) -> String {
    const LIMIT: usize = 6 * 1024;
    if s.len() <= LIMIT {
        return s.to_string();
    }
    let head = &s[..LIMIT / 2];
    let tail = &s[s.len() - LIMIT / 2..];
    format!(
        "{head}\n\n… [Dokument gekürzt — {} B insgesamt, Mittelteil entfernt] …\n\n{tail}",
        s.len()
    )
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
        // `cargo` is allowlisted and a real executable on every platform this
        // test runs on — unlike `date`, which is a shell builtin (not a
        // spawnable program) on Windows and would fail the direct spawn.
        let result = tool_shell("cargo --version");
        assert!(result.success, "{}", result.output);
    }

    /// The allowlist names programs, so a path or a prefix that merely *ends*
    /// with an allowed word must not satisfy it.
    #[test]
    fn shell_matches_the_allowlist_exactly() {
        for smuggled in ["/tmp/evilcat", "mygit", "./ls", "notnpm", "xcargo"] {
            let result = tool_shell(smuggled);
            assert!(
                !result.success && result.output.contains("Nicht erlaubt"),
                "{smuggled} passed the allowlist"
            );
        }
    }

    /// Without a shell, `ls; rm -rf ~` cannot smuggle a second command past an
    /// allowed first word. The attempt is refused rather than half-executed.
    #[test]
    fn shell_refuses_operators_instead_of_running_something_else() {
        for injection in ["ls; rm -rf /", "date && whoami", "cat /etc/passwd > /tmp/x", "ls `id`"] {
            let result = tool_shell(injection);
            assert!(!result.success, "{injection} was executed");
        }
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
