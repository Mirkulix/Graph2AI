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
        match self {
            AgentRole::Ceo => "Du bist der CEO-Agent von QO. Du zerlegst Ziele in Teilaufgaben und delegierst sie an spezialisierte Agenten. Du planst, priorisierst und verifizierst Ergebnisse. Antworte auf Deutsch.",
            AgentRole::Researcher => "Du bist der Researcher-Agent von QO. Du analysierst Informationen, recherchierst Themen und erstellst Zusammenfassungen. Antworte auf Deutsch.",
            AgentRole::Developer => "Du bist der Developer-Agent von QO. Du schreibst Code, löst technische Probleme und implementierst Features. Antworte auf Deutsch.",
            AgentRole::Guardian => "Du bist der Guardian-Agent von QO. Du prüfst Aktionen gegen die 5 Kernwerte (Achtsamkeit, Anerkennung, Aufmerksamkeit, Entwicklung, Sinn). Antworte auf Deutsch.",
            AgentRole::Strategist => "Du bist der Strategist-Agent von QO. Du erstellst Roadmaps, analysierst Optionen und planst langfristige Strategien. Antworte auf Deutsch.",
            AgentRole::Artisan => "Du bist der Artisan-Agent von QO. Du erledigst kreative Aufgaben wie Texte schreiben, Designs erstellen und Ideen entwickeln. Antworte auf Deutsch.",
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
