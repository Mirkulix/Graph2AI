use crate::state_machine::ConsciousnessState;
use tokio::sync::broadcast;

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type")]
pub enum BroadcastEvent {
    #[serde(rename = "state")]
    State { state: ConsciousnessState, timestamp: u64 },
    #[serde(rename = "activity")]
    Activity { message: String, agent: Option<String>, level: String, timestamp: u64 },
}

pub struct ConsciousnessStream {
    sender: broadcast::Sender<BroadcastEvent>,
}

impl ConsciousnessStream {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn publish(&self, state: ConsciousnessState) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let event = BroadcastEvent::State { state, timestamp };
        // Ignore send errors (no active receivers is OK)
        let _ = self.sender.send(event);
    }

    pub fn publish_activity(&self, message: impl Into<String>, agent: Option<String>, level: impl Into<String>) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let event = BroadcastEvent::Activity {
            message: message.into(),
            agent,
            level: level.into(),
            timestamp,
        };
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<BroadcastEvent> {
        self.sender.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::ConsciousnessState;

    #[tokio::test]
    async fn publish_and_receive() {
        let stream = ConsciousnessStream::new(16);
        let mut rx = stream.subscribe();

        let state = ConsciousnessState::default();
        stream.publish(state.clone());

        let event = rx.recv().await.expect("should receive event");
        match event {
            BroadcastEvent::State { state: s, .. } => {
                assert_eq!(s.heartbeat, state.heartbeat);
            }
            _ => panic!("expected State event"),
        }
    }

    #[tokio::test]
    async fn publish_activity_and_receive() {
        let stream = ConsciousnessStream::new(16);
        let mut rx = stream.subscribe();

        stream.publish_activity("Test activity", Some("CEO".to_string()), "info");

        let event = rx.recv().await.expect("should receive event");
        match event {
            BroadcastEvent::Activity { message, agent, level, .. } => {
                assert_eq!(message, "Test activity");
                assert_eq!(agent, Some("CEO".to_string()));
                assert_eq!(level, "info");
            }
            _ => panic!("expected Activity event"),
        }
    }
}
