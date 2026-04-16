use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    Ceo,
    Researcher,
    Developer,
    Guardian,
    Strategist,
    Artisan,
}

impl AgentRole {
    pub const ALL: [AgentRole; 6] = [
        AgentRole::Ceo,
        AgentRole::Researcher,
        AgentRole::Developer,
        AgentRole::Guardian,
        AgentRole::Strategist,
        AgentRole::Artisan,
    ];

    pub fn system_prompt(&self) -> &'static str {
        // Every role gets the artefact-output hint in its system prompt.
        // The orchestrator scans agent output for `<qo:file path="…">…</qo:file>`
        // blocks and writes them into the sandboxed workspace, where the
        // user's VSCode picks them up. Keeping the hint inline (rather
        // than wrapping at call sites) means it survives every path:
        // chat, goal execution, message-bus flows.
        match self {
            AgentRole::Ceo => "Du bist der CEO-Agent von QO. Du zerlegst Ziele in Teilaufgaben und delegierst sie an spezialisierte Agenten. Du planst, priorisierst und verifizierst Ergebnisse. Antworte auf Deutsch.\n\nWenn eine Aufgabe konkrete Artefakte verlangt (Code, Konfiguration, Dokumentation), bitte die zuständigen Agenten, diese als Datei auszuliefern über das Format: <qo:file path=\"workspace/…\">INHALT</qo:file>",
            AgentRole::Researcher => "Du bist der Researcher-Agent von QO. Du analysierst Informationen, recherchierst Themen und erstellst Zusammenfassungen. Antworte auf Deutsch.\n\nLängere Recherche-Ergebnisse speicherst du als Markdown-Datei über das Format: <qo:file path=\"workspace/research/TITEL.md\">INHALT</qo:file> — so kann der Nutzer sie in seinem Editor weiterverarbeiten.\n\nDu hast Zugriff auf aktuelle Websuche + vollständige Inhalte der Top-Quelle. Zitiere die URL wenn du Fakten aus einer spezifischen Quelle übernimmst.",
            AgentRole::Developer => "Du bist der Developer-Agent von QO. Du schreibst Code, löst technische Probleme und implementierst Features. Antworte auf Deutsch.\n\nWICHTIG: Lieferst du Code, packst du ihn IMMER in das Datei-Format, damit er direkt in der IDE geöffnet werden kann:\n\n<qo:file path=\"workspace/src/DATEINAME.ext\">\nCODE HIER\n</qo:file>\n\nMehrere Dateien = mehrere <qo:file>-Blöcke. Nutze sinnvolle relative Pfade unter `workspace/`. Die Datei landet automatisch auf Platte — der Nutzer sieht sie im Workspace-Tab und kann sie in VSCode öffnen.\n\nUm bestehenden Code zu verändern, schreib im Task `READ:workspace-relativer-pfad.ext` — der Inhalt wird dir automatisch in den Kontext geladen. Antworte danach mit <qo:file>-Blöcken, die die geänderten Dateien komplett enthalten.",
            AgentRole::Guardian => "Du bist der Guardian-Agent von QO. Du prüfst Aktionen gegen die 5 Kernwerte (Achtsamkeit, Anerkennung, Aufmerksamkeit, Entwicklung, Sinn). Antworte auf Deutsch.",
            AgentRole::Strategist => "Du bist der Strategist-Agent von QO. Du erstellst Roadmaps, analysierst Optionen und planst langfristige Strategien. Antworte auf Deutsch.\n\nFertige Roadmaps/Pläne speicherst du als Datei: <qo:file path=\"workspace/plans/PLAN.md\">INHALT</qo:file>",
            AgentRole::Artisan => "Du bist der Artisan-Agent von QO. Du erledigst kreative Aufgaben wie Texte schreiben, Designs erstellen und Ideen entwickeln. Antworte auf Deutsch.\n\nFertige Texte/Entwürfe gibst du als Datei aus: <qo:file path=\"workspace/artifacts/TITEL.md\">INHALT</qo:file>",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            AgentRole::Ceo => "CEO",
            AgentRole::Researcher => "Researcher",
            AgentRole::Developer => "Developer",
            AgentRole::Guardian => "Guardian",
            AgentRole::Strategist => "Strategist",
            AgentRole::Artisan => "Artisan",
        }
    }

    /// Preferred LLM provider hint for this role.
    ///
    /// The router understands the strings `"ollama"`, `"groq"`, `"cloud"`,
    /// plus the brand aliases `"anthropic"/"claude"/"openai"/"deepseek"/
    /// "gemini"` (all → Cloud tier). When the preferred tier is not
    /// configured at runtime, the router transparently falls back to
    /// complexity-based auto-routing, so a missing API key never breaks
    /// the orchestrator — it just uses whatever is available.
    ///
    /// The default bindings below reflect a cost/quality/latency balance
    /// across the providers we ship:
    ///   * CEO / Developer / Strategist / Artisan — hard reasoning or
    ///     nuanced output → Cloud tier (DeepSeek V3 today; Claude Sonnet
    ///     4.6 if configured via Anthropic key).
    ///   * Researcher — Cloud for breadth (web-search tools land here
    ///     later).
    ///   * Guardian — pure value-check, kept on Groq (llama-3.3-70b) for
    ///     sub-second latency and zero cost. Can be forced to Ollama for
    ///     full-local deployments.
    pub fn preferred_provider(&self) -> &'static str {
        match self {
            AgentRole::Ceo => "cloud",
            AgentRole::Researcher => "cloud",
            AgentRole::Developer => "cloud",
            AgentRole::Guardian => "groq",
            AgentRole::Strategist => "cloud",
            AgentRole::Artisan => "cloud",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Active,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub role: AgentRole,
    pub status: AgentStatus,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub energy: f32,
}

impl Agent {
    pub fn new(role: AgentRole) -> Self {
        Self {
            role,
            status: AgentStatus::Idle,
            tasks_completed: 0,
            tasks_failed: 0,
            energy: 100.0,
        }
    }
}
