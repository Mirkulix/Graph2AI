---
description: Before and after non-trivial work on a file or component, sync with the OrbitQLang knowledge graph — pull bounded context first, then submit validated proposals and evidence for what you actually established.
---

# OrbitQLang Sync

The knowledge graph is the durable memory between sessions. This skill is the
**closed loop** around a task: read bounded context before you start, hand back
a validated delta after you finish. It is what makes one session's findings
checkable — not just copied text — by the next.

## Non-negotiables (never violate these)

- **Request bounded context, never your full prompt history.** Ask the graph
  for the task's focus entity and its neighbours — nothing more.
- **Never submit unvalidated prose as knowledge.** The graph validates what you
  send; you only propose and, when you have evidence, verify.
- **Never put secrets, credentials, or absolute paths outside the workspace
  into a claim, evidence, or excerpt.**
- **Only `observed` and `verified` claims are facts.** Everything you write
  starts as `proposed`. Say so if you cite one.

## Before a non-trivial task

1. Call `mcp__orbitqlang__orbit_graph_context` with the task's focus entity
   (`kind` + `name`). This returns only load-bearing claims — build on them.
2. If the change may ripple, call `mcp__orbitqlang__orbit_graph_impact` for the
   bounded downstream picture.
3. Check `mcp__orbitqlang__orbit_graph_divergences` if you are changing an area
   where sessions have disagreed — the report shows both settled sides.

## After the task

For each finding worth the next session's time:

1. **Propose** it with `mcp__orbitqlang__orbit_graph_add_claim` (always lands
   as `proposed`).
2. **Verify** only what you actually checked:
   - If the code substantiates it, call `mcp__orbitqlang__orbit_graph_verify_source`
     (the graph reads the file itself and promotes only a literal match).
   - Otherwise call `mcp__orbitqlang__orbit_graph_verify_claim` with the exact
     evidence (locator, line range, excerpt) — never a bare assertion.
3. To hand back several findings at once, write an OrbitQLang document and call
   `mcp__orbitqlang__orbit_graph_commit_delta` (a `SIG` line from a trusted
   producer key is required).
4. If you disproved something, refute it (`verify_claim` with `supports: false`)
   rather than ignoring it — a recorded dead end stops the next agent repeating it.

Report what applied and what conflicted. The graph keeps every revision, so a
disagreement is data, not an error: surface it, do not silently pick a side.

## Lifecycle tools (the full loop)

- `mcp__orbitqlang__orbit_graph_verify_all` — sweep every open proposal against
  source (the harvest step).
- `mcp__orbitqlang__orbit_graph_refresh_sources` — mark settled claims stale
  when the code moved on.
- `mcp__orbitqlang__orbit_graph_heal_stale` — re-verify stale claims and heal
  the ones whose fact still holds.
- `mcp__orbitqlang__orbit_graph_receipt` — render one claim's full audit trail
  (revisions, evidence, disagreements) to prove why it is believed.
