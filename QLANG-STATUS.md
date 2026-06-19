# QLANG Status

Single source of truth for what currently works in this repository.

Last updated: 2026-06-18

## Current Product Scope

The repository is currently focused on the OrbitQLang control-plane path:

- `qo` server and supervisor flow
- agent orchestration in `qo-agents`
- QLMS / GraphMessage transport
- graph storage and message-bus streaming
- role-based LLM routing via `.qlang/llm_routing.toml`
- DeepSeek-first multi-agent orchestration via `Planner -> Worker -> Reviewer`
- simplified intent classification (deterministic keyword-based)

## Confirmed Working Surfaces

- `cargo check --workspace --all-targets` passes on the simplified architecture.
- `/api/chat`, `/api/chat/history`, `/api/goals`, `/api/agents`, `/api/messages/*`, `/api/neo/*` are the active server surfaces.
- `/api/multi-agent/run`, `/api/multi-agent/runs`, `/api/multi-agent/runs/{id}`, and `/api/multi-agent/stream` are active and wired into the cockpit.
- **QLMS v1.1 Bridge**: `/qlms/v1.1/deliver` and `/qlms/v1.1/reply` are active and integrated with the internal `MessageBus`.
- **IDE Integration**: VSCode / Trae extensions are functional, providing signed GraphMessage handover to the backend.
- Frontend production build currently passes via `cd frontend && npm run build`.
- All non-deterministic ML training and evolution loops have been purged.

## Intentionally Removed From Active Scope

These areas were removed from the project to ensure a lean, deterministic core:

- legacy ML/GPU logic (candle, mamba-tokenizer, training loops)
- deleted `qo-embed` and legacy `qlang-python` bindings
- spiking neural network (SNN) / STDP logic
- evolution/consciousness/organism subsystems
- legacy ML binaries and training examples

## Honesty Notes

- The project has been radically simplified to a pure AI-to-AI control plane.
- All heavy ML dependencies have been removed from `Cargo.toml`.
- The multi-agent product path is real but still intentionally narrow: planning, generation, review, run history, and cockpit visibility are implemented; general tool autonomy is not.
