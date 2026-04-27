//! Per-agent LLM (tier, model) mapping.
//!
//! Today the assignment is a hard-coded constant table. Keeping it as a
//! plain free function (rather than a `OnceLock`-backed `HashMap`) makes
//! the call site trivially const-foldable and avoids any locking on the
//! hot path of the agent mailbox loop.
//!
//! The LlmRouter respects the `model` only for tiers that support
//! per-call model overrides (today: DeepSeek). For other tiers (Ollama,
//! Groq, Cloud) the model name is informational and the tier's installed
//! default is used. That's intentional: `Tier::Ollama` clients are
//! constructed with a single model at install time, and rebuilding them
//! per call would add latency for no benefit at this stage.
//!
//! Defaults (per spec):
//!   developer   → DeepSeek / deepseek-coder
//!   strategist  → DeepSeek / deepseek-reasoner
//!   guardian    → Ollama   / llama3:latest
//!   artisan     → Ollama   / qwen3:8b
//!   researcher  → DeepSeek / deepseek-chat
//!   ceo         → DeepSeek / deepseek-chat
//!
//! Unknown agent names fall back to DeepSeek / deepseek-chat so a typo
//! in a future call site still produces a usable reply.

use qo_llm::Tier;

/// Returns the (preferred tier, optional model override) tuple for the
/// given agent name. The result is meant to be passed straight into
/// `LlmRouter::chat_with_model(Some(tier), model, msgs)`.
pub fn model_for_agent(agent: &str) -> (Tier, Option<String>) {
    match agent {
        "developer"  => (Tier::DeepSeek, Some("deepseek-coder".to_string())),
        "strategist" => (Tier::DeepSeek, Some("deepseek-reasoner".to_string())),
        "guardian"   => (Tier::Local,    Some("llama3:latest".to_string())),
        "artisan"    => (Tier::Local,    Some("qwen3:8b".to_string())),
        "researcher" => (Tier::DeepSeek, Some("deepseek-chat".to_string())),
        "ceo"        => (Tier::DeepSeek, Some("deepseek-chat".to_string())),
        _            => (Tier::DeepSeek, Some("deepseek-chat".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn developer_uses_coder() {
        let (tier, model) = model_for_agent("developer");
        assert_eq!(tier, Tier::DeepSeek);
        assert_eq!(model.as_deref(), Some("deepseek-coder"));
    }

    #[test]
    fn strategist_uses_reasoner() {
        let (tier, model) = model_for_agent("strategist");
        assert_eq!(tier, Tier::DeepSeek);
        assert_eq!(model.as_deref(), Some("deepseek-reasoner"));
    }

    #[test]
    fn guardian_uses_local_llama3() {
        let (tier, model) = model_for_agent("guardian");
        assert_eq!(tier, Tier::Local);
        assert_eq!(model.as_deref(), Some("llama3:latest"));
    }

    #[test]
    fn artisan_uses_local_qwen() {
        let (tier, model) = model_for_agent("artisan");
        assert_eq!(tier, Tier::Local);
        assert_eq!(model.as_deref(), Some("qwen3:8b"));
    }

    #[test]
    fn unknown_agent_falls_back_to_deepseek_chat() {
        let (tier, model) = model_for_agent("totally-not-an-agent");
        assert_eq!(tier, Tier::DeepSeek);
        assert_eq!(model.as_deref(), Some("deepseek-chat"));
    }
}
