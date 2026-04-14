use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: u64,
    pub name: String,
    pub description: String,
    pub frequency: u32,        // how often detected
    pub severity: f32,         // 0.0 = info, 1.0 = critical
    pub first_seen: u64,
    pub last_seen: u64,
    pub category: PatternCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternCategory {
    Performance,    // system bottlenecks
    Behavior,       // agent behavior patterns
    Error,          // recurring errors
    Growth,         // positive development
    Stagnation,     // lack of progress
}

pub struct PatternDetector {
    patterns: Vec<Pattern>,
    next_id: u64,
}

impl PatternDetector {
    pub fn new() -> Self {
        Self { patterns: Vec::new(), next_id: 1 }
    }

    /// Analyze task completion stats and detect patterns
    pub fn analyze(&mut self, stats: &SystemStats) -> Vec<&Pattern> {
        let mut new_pattern_names = Vec::new();

        // Detect high failure rate
        if stats.total_tasks > 5 && stats.failure_rate() > 0.5 {
            self.upsert_pattern(
                "high_failure_rate",
                PatternCategory::Error,
                format!("Hohe Fehlerrate: {:.0}% der letzten {} Tasks fehlgeschlagen",
                    stats.failure_rate() * 100.0, stats.total_tasks),
                0.8,
            );
            new_pattern_names.push("high_failure_rate");
        }

        // Detect idle agents
        if stats.idle_ratio() > 0.8 && stats.total_tasks > 3 {
            self.upsert_pattern(
                "agent_idle_dominance",
                PatternCategory::Stagnation,
                format!("{:.0}% der Agenten sind idle", stats.idle_ratio() * 100.0),
                0.5,
            );
            new_pattern_names.push("agent_idle_dominance");
        }

        // Detect good completion streak
        if stats.completed_streak >= 5 {
            self.upsert_pattern(
                "completion_streak",
                PatternCategory::Growth,
                format!("{} Tasks in Folge erfolgreich abgeschlossen", stats.completed_streak),
                0.3,
            );
            new_pattern_names.push("completion_streak");
        }

        // Detect energy drain
        if stats.avg_energy < 30.0 {
            self.upsert_pattern(
                "low_energy",
                PatternCategory::Performance,
                format!("Durchschnittliche Energie bei {:.0}%", stats.avg_energy),
                0.6,
            );
            new_pattern_names.push("low_energy");
        }

        self.patterns.iter().filter(|p| new_pattern_names.contains(&p.name.as_str())).collect()
    }

    fn upsert_pattern(&mut self, name: &str, category: PatternCategory, description: String, severity: f32) {
        let now = now_secs();
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.name == name) {
            existing.frequency += 1;
            existing.last_seen = now;
            existing.description = description;
            return;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.patterns.push(Pattern {
            id,
            name: name.to_string(),
            description,
            frequency: 1,
            severity,
            first_seen: now,
            last_seen: now,
            category,
        });
    }

    pub fn all_patterns(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Restore a previously persisted pattern. Keeps next_id consistent.
    pub fn restore_pattern(&mut self, pattern: Pattern) {
        if pattern.id >= self.next_id {
            self.next_id = pattern.id + 1;
        }
        if !self.patterns.iter().any(|p| p.id == pattern.id) {
            self.patterns.push(pattern);
        }
    }

    pub fn active_patterns(&self) -> Vec<&Pattern> {
        let cutoff = now_secs().saturating_sub(3600); // last hour
        self.patterns.iter().filter(|p| p.last_seen >= cutoff).collect()
    }
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_tasks: u32,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub agents_active: u8,
    pub agents_idle: u8,
    pub avg_energy: f32,
    pub completed_streak: u32,
}

impl SystemStats {
    pub fn failure_rate(&self) -> f32 {
        if self.total_tasks == 0 { return 0.0; }
        self.tasks_failed as f32 / self.total_tasks as f32
    }

    pub fn idle_ratio(&self) -> f32 {
        let total = self.agents_active as f32 + self.agents_idle as f32;
        if total == 0.0 { return 0.0; }
        self.agents_idle as f32 / total
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_high_failure_rate() {
        let mut detector = PatternDetector::new();
        let stats = SystemStats {
            total_tasks: 10, tasks_completed: 3, tasks_failed: 7,
            agents_active: 1, agents_idle: 5, avg_energy: 50.0, completed_streak: 0,
        };
        let patterns = detector.analyze(&stats);
        assert!(patterns.iter().any(|p| p.name == "high_failure_rate"));
    }

    #[test]
    fn detect_completion_streak() {
        let mut detector = PatternDetector::new();
        let stats = SystemStats {
            total_tasks: 10, tasks_completed: 10, tasks_failed: 0,
            agents_active: 3, agents_idle: 3, avg_energy: 80.0, completed_streak: 7,
        };
        let patterns = detector.analyze(&stats);
        assert!(patterns.iter().any(|p| p.name == "completion_streak"));
    }

    #[test]
    fn test_upsert_increments_frequency() {
        let mut detector = PatternDetector::new();
        let stats = SystemStats {
            total_tasks: 10, tasks_completed: 3, tasks_failed: 7,
            agents_active: 1, agents_idle: 5, avg_energy: 50.0, completed_streak: 0,
        };
        detector.analyze(&stats);
        detector.analyze(&stats);
        // high_failure_rate pattern should have frequency 2 after two detections
        let pattern = detector.all_patterns().iter().find(|p| p.name == "high_failure_rate");
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().frequency, 2);
    }

    #[test]
    fn test_no_patterns_when_stats_are_good() {
        let mut detector = PatternDetector::new();
        let stats = SystemStats {
            // Low failure (below 50%), balanced agents (idle_ratio ~0.5), high energy
            total_tasks: 4, tasks_completed: 4, tasks_failed: 0,
            agents_active: 3, agents_idle: 3, avg_energy: 80.0, completed_streak: 0,
        };
        let patterns = detector.analyze(&stats);
        // Should not detect high_failure_rate (need >5 tasks) or agent_idle_dominance (ratio 0.5)
        assert!(!patterns.iter().any(|p| p.name == "high_failure_rate"));
        assert!(!patterns.iter().any(|p| p.name == "agent_idle_dominance"));
    }
}
