use crate::agent::{Agent, AgentRole, AgentStatus};
use crate::goal::{Goal, GoalStatus};
use crate::executor;
use qlang_agent::bus::MessageBus;
use qo_evolution::QuantumState;
use qo_llm::LlmRouter;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct AgentSummary {
    pub role: AgentRole,
    pub status: AgentStatus,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
}

pub struct AgentRegistry {
    agents: HashMap<AgentRole, Agent>,
    goals: Vec<Goal>,
    next_goal_id: u64,
}

impl AgentRegistry {
    pub fn new() -> Self {
        let mut agents = HashMap::new();
        for role in AgentRole::ALL {
            agents.insert(role, Agent::new(role));
        }
        Self {
            agents,
            goals: Vec::new(),
            next_goal_id: 1,
        }
    }

    pub fn list_agents(&self) -> Vec<AgentSummary> {
        self.agents
            .values()
            .map(|a| AgentSummary {
                role: a.role,
                status: a.status,
                tasks_completed: a.tasks_completed,
                tasks_failed: a.tasks_failed,
            })
            .collect()
    }

    pub fn get_agent(&self, role: AgentRole) -> Option<&Agent> {
        self.agents.get(&role)
    }

    pub fn active_count(&self) -> u8 {
        self.agents
            .values()
            .filter(|a| a.status == AgentStatus::Active)
            .count() as u8
    }

    pub fn idle_count(&self) -> u8 {
        self.agents
            .values()
            .filter(|a| a.status == AgentStatus::Idle)
            .count() as u8
    }

    pub fn create_goal(&mut self, description: String) -> &Goal {
        let id = self.next_goal_id;
        self.next_goal_id += 1;
        self.goals.push(Goal::new(id, description));
        self.goals.last().unwrap()
    }

    pub fn list_goals(&self) -> &[Goal] {
        &self.goals
    }

    pub fn get_goal(&self, id: u64) -> Option<&Goal> {
        self.goals.iter().find(|g| g.id == id)
    }

    /// Execute a goal. Marks involved agents as active, then idle.
    /// Returns (chosen_strategy_name, goal_succeeded) on success.
    /// When a MessageBus is provided, all agent communication flows through it.
    pub async fn execute_goal(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
        quantum_state: Option<&QuantumState>,
        bus: Option<Arc<MessageBus>>,
    ) -> Result<(String, bool), Box<dyn std::error::Error + Send + Sync>> {
        // Mark CEO active
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let result = executor::execute_goal_qlang(llm, goal, quantum_state, bus).await;

        // Update agent stats based on subtask results
        if let Some(goal) = self.goals.iter().find(|g| g.id == goal_id) {
            for subtask in &goal.subtasks {
                if let Some(agent) = self.agents.get_mut(&subtask.assigned_to) {
                    match subtask.status {
                        GoalStatus::Completed => agent.tasks_completed += 1,
                        GoalStatus::Failed => agent.tasks_failed += 1,
                        _ => {}
                    }
                    agent.status = AgentStatus::Idle;
                }
            }
        }

        // Mark CEO idle
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Idle;
            match &result {
                Ok(_) => ceo.tasks_completed += 1,
                Err(_) => ceo.tasks_failed += 1,
            }
        }

        result.map(|(_, strategy, succeeded)| (strategy, succeeded))
    }

    /// Step 1: CEO decomposes the goal into subtasks.
    /// Returns (subtask_count, chosen_strategy).
    pub async fn execute_goal_decompose(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
        quantum_state: Option<&QuantumState>,
    ) -> Result<(usize, String), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let chosen_strategy = executor::decompose_goal(llm, goal, quantum_state).await?;
        let count = goal.subtasks.len();
        Ok((count, chosen_strategy))
    }

    /// Step 2: Execute a single subtask by index.
    pub async fn execute_goal_subtask(
        &mut self,
        goal_id: u64,
        subtask_index: usize,
        llm: &LlmRouter,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Mark the agent for this subtask as active
        let agent_role = {
            let goal = self
                .goals
                .iter()
                .find(|g| g.id == goal_id)
                .ok_or("Goal not found")?;
            goal.subtasks
                .get(subtask_index)
                .ok_or("Subtask not found")?
                .assigned_to
        };

        if let Some(agent) = self.agents.get_mut(&agent_role) {
            agent.status = AgentStatus::Active;
        }

        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        let result = executor::execute_subtask(llm, goal, subtask_index).await;

        if let Some(agent) = self.agents.get_mut(&agent_role) {
            agent.status = AgentStatus::Idle;
            match &result {
                Ok(_) => agent.tasks_completed += 1,
                Err(_) => agent.tasks_failed += 1,
            }
        }

        result
    }

    /// Step 3: CEO summarizes all subtask results. Returns summary string.
    pub async fn execute_goal_summarize(
        &mut self,
        goal_id: u64,
        llm: &LlmRouter,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let goal = self
            .goals
            .iter_mut()
            .find(|g| g.id == goal_id)
            .ok_or("Goal not found")?;

        executor::summarize_goal(llm, goal).await
    }

    /// Step 2 (parallel): Execute all subtasks of a goal concurrently.
    /// Returns a Vec of (index, agent_name, task_desc, result_ok, duration_ms).
    pub async fn execute_goal_subtasks_parallel(
        &mut self,
        goal_id: u64,
        llm: std::sync::Arc<qo_llm::LlmRouter>,
    ) -> Vec<(usize, String, String, bool, u64)> {
        // Collect subtask metadata without holding &mut self across await points
        let subtask_info: Vec<(usize, crate::agent::AgentRole, String, String)> = {
            match self.goals.iter().find(|g| g.id == goal_id) {
                None => return Vec::new(),
                Some(goal) => goal
                    .subtasks
                    .iter()
                    .enumerate()
                    .map(|(i, st)| (i, st.assigned_to, st.assigned_to.name().to_string(), st.description.clone()))
                    .collect(),
            }
        };
        let goal_desc: String = self
            .goals
            .iter()
            .find(|g| g.id == goal_id)
            .map(|g| g.description.clone())
            .unwrap_or_default();

        // Mark all involved agents as active
        for (_, role, _, _) in &subtask_info {
            if let Some(agent) = self.agents.get_mut(role) {
                agent.status = crate::agent::AgentStatus::Active;
            }
        }

        // Spawn all LLM calls in parallel
        let mut handles = Vec::new();
        for (idx, _role, agent_name, desc) in subtask_info.iter().cloned() {
            let llm_arc = llm.clone();
            let goal_desc_c = goal_desc.clone();
            let assigned_to = subtask_info[idx].1;
            handles.push(tokio::spawn(async move {
                let start = std::time::Instant::now();
                let result = crate::llm_node::llm_reason(&*llm_arc, assigned_to, &goal_desc_c, &desc).await;
                let duration_ms = start.elapsed().as_millis() as u64;
                (idx, agent_name, desc, result, duration_ms)
            }));
        }

        // Collect results and write back
        let mut outcomes = Vec::new();
        for handle in handles {
            match handle.await {
                Ok((idx, agent_name, task_desc, result, duration_ms)) => {
                    let succeeded = result.is_ok();
                    if let Some(goal) = self.goals.iter_mut().find(|g| g.id == goal_id) {
                        if let Some(subtask) = goal.subtasks.get_mut(idx) {
                            match result {
                                Ok(text) => {
                                    subtask.result = Some(text);
                                    subtask.status = crate::goal::GoalStatus::Completed;
                                }
                                Err(e) => {
                                    subtask.result = Some(format!("Fehler: {e}"));
                                    subtask.status = crate::goal::GoalStatus::Failed;
                                }
                            }
                        }
                    }
                    let role = subtask_info[idx].1;
                    if let Some(agent) = self.agents.get_mut(&role) {
                        agent.status = crate::agent::AgentStatus::Idle;
                        if succeeded {
                            agent.tasks_completed += 1;
                        } else {
                            agent.tasks_failed += 1;
                        }
                    }
                    outcomes.push((idx, agent_name, task_desc, succeeded, duration_ms));
                }
                Err(e) => {
                    tracing::warn!("subtask spawn join error: {e}");
                }
            }
        }
        outcomes
    }

    /// Load a previously persisted goal back into the registry.
    pub fn restore_goal(&mut self, goal: Goal) {
        // Keep next_goal_id ahead of any restored id
        if goal.id >= self.next_goal_id {
            self.next_goal_id = goal.id + 1;
        }
        // Avoid duplicates on double-load
        if !self.goals.iter().any(|g| g.id == goal.id) {
            self.goals.push(goal);
        }
    }

    /// Restore agent task counters (tasks_completed / tasks_failed) from persisted stats.
    /// Resets status to Idle regardless of stored status to avoid stale Active states after restart.
    pub fn restore_agent_stats(&mut self, role: AgentRole, tasks_completed: u32, tasks_failed: u32) {
        if let Some(agent) = self.agents.get_mut(&role) {
            agent.tasks_completed = tasks_completed;
            agent.tasks_failed = tasks_failed;
            agent.status = AgentStatus::Idle;
        }
    }

    /// Finalize: update CEO stats and mark goal complete.
    pub fn finalize_goal(&mut self, goal_id: u64) {
        let succeeded = self
            .goals
            .iter()
            .find(|g| g.id == goal_id)
            .map(|g| g.result.is_some())
            .unwrap_or(false);

        if let Some(ceo) = self.agents.get_mut(&AgentRole::Ceo) {
            ceo.status = AgentStatus::Idle;
            if succeeded {
                ceo.tasks_completed += 1;
            } else {
                ceo.tasks_failed += 1;
            }
        }
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_has_6_agents() {
        let registry = AgentRegistry::new();
        let agents = registry.list_agents();
        assert_eq!(agents.len(), 6);
        // Verify all roles are present
        for role in AgentRole::ALL {
            assert!(registry.get_agent(role).is_some(), "Missing role: {:?}", role);
        }
    }

    #[test]
    fn test_create_goal_increments_id() {
        let mut registry = AgentRegistry::new();
        let g1 = registry.create_goal("First goal".to_string());
        let id1 = g1.id;
        let g2 = registry.create_goal("Second goal".to_string());
        let id2 = g2.id;
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(registry.list_goals().len(), 2);
    }
}
