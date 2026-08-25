---
description: Start an explicitly requested, tracked OrbitQLang background goal through QO.
disable-model-invocation: true
---

# OrbitQLang Goal

Use this skill only when the user explicitly requests multi-agent delegation,
background execution, or a tracked OrbitQLang goal.

1. Turn `$ARGUMENTS` into a concise, bounded goal. If it is empty, ask the
   user for the goal instead of starting work.
2. State that the execution is asynchronous.
3. Call `mcp__orbitqlang__qlang_run_goal` with `{ "goal": "..." }`.
4. Return the goal ID or status from the tool result without claiming that the
   background work has completed.

Do not send credentials, secrets, or paths outside the allowed workspace.
