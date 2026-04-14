use serde::{Deserialize, Serialize};
use crate::pattern::{Pattern, PatternCategory};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    Implemented,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub based_on_pattern: String,
    pub priority: f32,
    pub status: ProposalStatus,
    pub created_at: u64,
}

pub struct ProposalEngine {
    proposals: Vec<Proposal>,
    next_id: u64,
}

impl ProposalEngine {
    pub fn new() -> Self {
        Self { proposals: Vec::new(), next_id: 1 }
    }

    /// Generate proposals from detected patterns
    pub fn generate_from_patterns(&mut self, patterns: &[&Pattern]) -> Vec<&Proposal> {
        let mut new_proposal_ids = Vec::new();

        for pattern in patterns {
            // Don't generate duplicate proposals
            if self.proposals.iter().any(|p| p.based_on_pattern == pattern.name && p.status == ProposalStatus::Pending) {
                continue;
            }

            let (title, description) = match pattern.category {
                PatternCategory::Error => (
                    format!("Fehlerrate reduzieren: {}", pattern.name),
                    format!("Das Muster '{}' wurde {} mal erkannt. Vorschlag: Fehlerbehandlung verbessern und Retry-Logik für betroffene Agenten einbauen.", pattern.description, pattern.frequency),
                ),
                PatternCategory::Stagnation => (
                    format!("Stagnation beheben: {}", pattern.name),
                    format!("Das Muster '{}' zeigt Stagnation. Vorschlag: Idle-Agenten aktiv mit Hintergrund-Tasks beschäftigen.", pattern.description),
                ),
                PatternCategory::Performance => (
                    format!("Performance verbessern: {}", pattern.name),
                    format!("Das Muster '{}' zeigt ein Performance-Problem. Vorschlag: Energie-Management optimieren.", pattern.description),
                ),
                PatternCategory::Growth => (
                    format!("Wachstum fortsetzen: {}", pattern.name),
                    format!("Positives Muster erkannt: '{}'. Vorschlag: Diese Strategie beibehalten und ausbauen.", pattern.description),
                ),
                PatternCategory::Behavior => (
                    format!("Verhalten optimieren: {}", pattern.name),
                    format!("Verhaltensmuster erkannt: '{}'. Vorschlag: Agent-Konfiguration anpassen.", pattern.description),
                ),
            };

            let id = self.next_id;
            self.next_id += 1;
            self.proposals.push(Proposal {
                id,
                title,
                description,
                based_on_pattern: pattern.name.clone(),
                priority: pattern.severity,
                status: ProposalStatus::Pending,
                created_at: now_secs(),
            });
            new_proposal_ids.push(id);
        }

        self.proposals.iter().filter(|p| new_proposal_ids.contains(&p.id)).collect()
    }

    pub fn approve(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            p.status = ProposalStatus::Approved;
            true
        } else { false }
    }

    pub fn reject(&mut self, id: u64) -> bool {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            p.status = ProposalStatus::Rejected;
            true
        } else { false }
    }

    pub fn pending(&self) -> Vec<&Proposal> {
        self.proposals.iter().filter(|p| p.status == ProposalStatus::Pending).collect()
    }

    pub fn all(&self) -> &[Proposal] {
        &self.proposals
    }

    /// Restore a previously persisted proposal. Keeps next_id consistent.
    pub fn restore_proposal(&mut self, proposal: Proposal) {
        if proposal.id >= self.next_id {
            self.next_id = proposal.id + 1;
        }
        if !self.proposals.iter().any(|p| p.id == proposal.id) {
            self.proposals.push(proposal);
        }
    }
}

impl Default for ProposalEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pattern(name: &str, category: PatternCategory) -> Pattern {
        Pattern {
            id: 1,
            name: name.to_string(),
            description: format!("Test pattern {}", name),
            frequency: 1,
            severity: 0.8,
            first_seen: 0,
            last_seen: 0,
            category,
        }
    }

    #[test]
    fn test_approve_changes_status() {
        let mut engine = ProposalEngine::new();
        let pattern = make_pattern("high_failure_rate", PatternCategory::Error);
        let patterns = vec![&pattern];
        let generated = engine.generate_from_patterns(&patterns);
        assert_eq!(generated.len(), 1);
        let id = generated[0].id;

        let ok = engine.approve(id);
        assert!(ok);
        let proposal = engine.all().iter().find(|p| p.id == id).unwrap();
        assert_eq!(proposal.status, ProposalStatus::Approved);
    }

    #[test]
    fn test_no_duplicate_pending() {
        let mut engine = ProposalEngine::new();
        let pattern = make_pattern("stagnation", PatternCategory::Stagnation);
        let patterns = vec![&pattern];

        let first = engine.generate_from_patterns(&patterns);
        assert_eq!(first.len(), 1);

        // Second call with same pattern → no new pending proposal
        let second = engine.generate_from_patterns(&patterns);
        assert_eq!(second.len(), 0);

        // Only one proposal total
        assert_eq!(engine.pending().len(), 1);
    }

    #[test]
    fn test_reject_proposal() {
        let mut engine = ProposalEngine::new();
        let pattern = make_pattern("low_energy", PatternCategory::Performance);
        let patterns = vec![&pattern];
        let generated = engine.generate_from_patterns(&patterns);
        let id = generated[0].id;

        let ok = engine.reject(id);
        assert!(ok);
        let proposal = engine.all().iter().find(|p| p.id == id).unwrap();
        assert_eq!(proposal.status, ProposalStatus::Rejected);
        // No longer in pending
        assert!(engine.pending().is_empty());
    }
}
