//! QO Intent Classifier — deterministic keyword-based implementation.
//! Replaces legacy MLP for radical project simplification.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent { Chat = 0, Goal = 1, Question = 2, Creative = 3 }

impl Intent {
    pub fn from_index(i: usize) -> Self {
        match i { 0 => Intent::Chat, 1 => Intent::Goal, 2 => Intent::Question, 3 => Intent::Creative, _ => Intent::Chat }
    }
    pub fn label(&self) -> &'static str {
        match self { Intent::Chat => "chat", Intent::Goal => "goal", Intent::Question => "question", Intent::Creative => "creative" }
    }
}

const KEYWORDS_GOAL: &[&str] = &["analysiere", "implementiere", "recherchiere", "plane", "erstelle", "liste", "vorteile", "auswirkungen"];
const KEYWORDS_QUESTION: &[&str] = &["was ist", "warum", "wie funktioniert", "unterschied", "wo finde ich", "kannst du", "frage"];
const KEYWORDS_CREATIVE: &[&str] = &["erfinde", "brainstorme", "schreibe", "gestalte", "design", "ideen", "entwirf", "roman", "geschichte"];
const KEYWORDS_CHAT: &[&str] = &["hallo", "guten tag", "guten morgen", "guten abend", "danke", "hilfe", "geht es", "schönen tag"];

pub fn classify_intent_cached(text: &str) -> (Intent, Vec<f32>) {
    let lower = text.to_lowercase();
    let mut scores = vec![0.0f32; 4];

    for k in KEYWORDS_CHAT { if lower.contains(k) { scores[0] += 1.0; } }
    for k in KEYWORDS_GOAL { if lower.contains(k) { scores[1] += 1.0; } }
    for k in KEYWORDS_QUESTION { if lower.contains(k) { scores[2] += 1.0; } }
    for k in KEYWORDS_CREATIVE { if lower.contains(k) { scores[3] += 1.0; } }

    let sum: f32 = scores.iter().sum();
    let probs = if sum > 0.0 {
        scores.iter().map(|s| s / sum).collect()
    } else {
        vec![0.25, 0.25, 0.25, 0.25]
    };

    let best = scores.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);

    (Intent::from_index(best), probs)
}

// Dummy for backward compatibility with other crates if needed
pub fn get_model_placeholder() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_goal() {
        let (intent, _) = classify_intent_cached("Analysiere die Performance");
        assert_eq!(intent, Intent::Goal);
    }

    #[test]
    fn classifies_question() {
        let (intent, _) = classify_intent_cached("Was ist der Unterschied?");
        assert_eq!(intent, Intent::Question);
    }

    #[test]
    fn classifies_chat() {
        let (intent, _) = classify_intent_cached("Hallo, danke für die Hilfe!");
        assert_eq!(intent, Intent::Chat);
    }

    #[test]
    fn classifies_creative() {
        let (intent, _) = classify_intent_cached("Schreibe einen Text");
        assert_eq!(intent, Intent::Creative);
    }
}
