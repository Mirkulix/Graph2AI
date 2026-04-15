# QLANG Supervisor

`qlang supervisor` is the first Rust control-plane MVP for orchestrating external coding CLIs.

It does not replace Claude Code, Codex, Gemini, or Kimi.

It tracks them.

## What it manages

- agent registry
- permission profile
- task queue
- sessions
- handover file references
- cockpit state
- recent events
- stdout/stderr session logs

The state is persisted as JSON, typically in:

`.\.qlang\supervisor.json`

## Core flow

Initialize:

```powershell
qlang supervisor init --state .qlang/supervisor.json
```

Register agents:

```powershell
qlang supervisor add-agent --state .qlang/supervisor.json --name claude --kind claude-code --command claude --arg code
qlang supervisor add-agent --state .qlang/supervisor.json --name codex --kind codex --command codex
```

Queue work:

```powershell
qlang supervisor enqueue --state .qlang/supervisor.json --title "Investigate parser panic" --goal "Find the root cause and propose the first patch" --agent claude
```

Schedule the next runnable task:

```powershell
qlang supervisor tick --state .qlang/supervisor.json
```

Schedule and spawn the agent process immediately:

```powershell
qlang supervisor tick --state .qlang/supervisor.json --spawn
```

Poll running sessions and mark finished tasks:

```powershell
qlang supervisor poll --state .qlang/supervisor.json
```

Inspect captured session logs:

```powershell
qlang supervisor logs --state .qlang/supervisor.json --session 1 --tail 50
```

Do both in one step:

```powershell
qlang supervisor cycle --state .qlang/supervisor.json --spawn
```

Run the supervisor as a continuous worker:

```powershell
qlang supervisor daemon --state .qlang/supervisor.json --spawn --interval-ms 2000
```

Attach a QLMS handover:

```powershell
qlang supervisor handover --state .qlang/supervisor.json --task 1 --path handoff.qlms
```

Mark work complete:

```powershell
qlang supervisor complete --state .qlang/supervisor.json --task 1 --note "Patch reviewed and merged"
```

Inspect the cockpit state:

```powershell
qlang supervisor show --state .qlang/supervisor.json
```

Open the first browser cockpit:

```powershell
cargo run --bin qo --offline
```

Then open:

`http://127.0.0.1:4646/supervisor`

The cockpit can now:

- register agents
- enqueue tasks
- trigger `poll`
- trigger `tick`
- trigger `tick --spawn`
- trigger `cycle --spawn`
- mark tasks as complete
- mark tasks as failed
- attach a handover path to a task
- create a QLMS coding handover
- append a QLMS handover reply
- inspect a QLMS handover conversation
- inspect session stdout/stderr logs
- receive live cockpit updates over SSE
- start and stop the supervisor daemon from the cockpit
- install agent presets for the built-in QLANG demo agent, Claude Code, Codex, Gemini, and Kimi
- see which agent presets are actually available on the current machine
- use role/capability-aware presets to prefill task routing in the cockpit
- use quick actions like Run GUI Smoke Test, Analyze with Claude, or Patch with Codex

## GUI smoke test

If you want to test the complete browser flow without depending on an external CLI, open the cockpit and use `Run GUI Smoke Test`.

That path uses the built-in `QLANG Demo Agent`, which:

- starts as a normal supervisor session
- writes deterministic stdout lines into the session log
- exits automatically after a short delay

This lets you verify:

- task creation
- session spawning
- live SSE updates
- session log capture
- daemon-driven completion

## Current scope

This is a persistence and orchestration layer.

It already gives you:

- one central task queue
- session tracking
- agent registration
- optional real process spawn with PID tracking
- polling of running agent sessions
- stdout/stderr capture into per-session log files
- handover tracking across tools
- a cockpit-style state snapshot
- a browser cockpit at `/supervisor` backed by the same supervisor state
- browser actions for registering agents and queueing/running work
- a daemon mode for continuous `poll + tick` task processing

It does not yet:

- launch real terminal tabs automatically
- enforce plugin permissions at runtime
- auto-resume blocked jobs
- run as a long-lived daemon

Those are the next steps after the persisted state model.
