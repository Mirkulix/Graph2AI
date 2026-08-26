---
description: Use OrbitQLang's local QO control plane for a durable knowledge graph of what has been established about a project — claims with provenance and evidence that survive across sessions — plus multi-agent research, tracked goals, and sandboxed workspace reads.
---

# OrbitQLang

Use the OrbitQLang MCP tools only when the local QO server is running at
`http://127.0.0.1:4646`. If the tools are unavailable, say that the QO server
must be started with `qo --offline`; do not attempt to start it automatically.

## Knowledge graph

The knowledge graph is durable memory with an audit trail. Each **claim**
carries who produced it, when, and what backs it up. Claims are never
overwritten: changing one appends a revision, so contradictions stay visible
with both sides' evidence.

Every claim has a status. Only two of them count as reliable:

| Status | Meaning | Reliable? |
|---|---|---|
| `observed` | captured directly from code, config or tool output | yes |
| `verified` | confirmed with reproducible evidence | yes |
| `proposed` | suggested, not yet checked | **no** |
| `stale` | possibly outdated by a newer revision | **no** |
| `refuted` | disproved by counter-evidence | **no** |

- `mcp__orbitqlang__orbit_graph_context`: Compact, source-bound context for a
  task. Returns **only** `observed` and `verified` claims. Start here when
  picking up work on a file or component.
- `mcp__orbitqlang__orbit_graph_search`: Find claims by substring. Returns
  every status, each labelled — useful for seeing what was already tried and
  rejected.
- `mcp__orbitqlang__orbit_graph_neighbors`: Traverse relations in both
  directions. Use for dependency and impact questions.
- `mcp__orbitqlang__orbit_graph_impact`: Traverse only load-bearing relations
  across up to four hops. Use it before a non-trivial change to identify the
  bounded downstream impact; proposals and stale claims are excluded.
- `mcp__orbitqlang__orbit_graph_add_claim`: Record something worth
  remembering. It is always stored as `proposed`.
- `mcp__orbitqlang__orbit_graph_verify_claim`: Confirm (`supports: true`) or
  refute (`supports: false`) a claim with evidence — a file and line range, a
  command, a commit, a test run.
- `mcp__orbitqlang__orbit_graph_verify_source`: Have the graph check a proposed
  claim against its actual source file and promote it to `verified` only when
  the code literally substantiates every term. Deterministic — the graph reads
  the file, not a caller-supplied excerpt.
- `mcp__orbitqlang__orbit_graph_receipt`: Render one claim's full audit trail —
  revisions, who decided what, evidence, and related claims including
  disagreements — to prove why it is believed.
- `mcp__orbitqlang__orbit_graph_commit_delta`: Submit a batch of changes as an
  OrbitQLang document. Requires a `SIG` line from a trusted producer key;
  returns per-operation outcomes including conflicts.
- `mcp__orbitqlang__orbit_graph_verify_all`: Sweep every open proposal against
  source and promote what the code substantiates (the harvest step).
- `mcp__orbitqlang__orbit_graph_refresh_sources`: Mark settled claims `stale`
  when their recorded excerpt is gone from the source — the graph noticing its
  facts rot.
- `mcp__orbitqlang__orbit_graph_heal_stale`: Re-verify stale claims and heal
  the ones whose fact still holds (the code moved, the fact did not).
- `mcp__orbitqlang__orbit_graph_divergences`: List every subject where a
  load-bearing claim and a refuted claim coexist — where sessions settled in
  opposite directions.
- `mcp__orbitqlang__orbit_graph_swarm_state`: What the other agent sessions are
  doing right now.

## Operating rules

1. Prefer direct local reasoning for small, self-contained tasks.
2. Call `orbit_graph_context` before non-trivial work on a file or component,
   so you build on what was already established rather than re-deriving it.
3. **Record, then verify — never assert.** `add_claim` always writes a
   proposal; that is deliberate. When you have actually checked something,
   call `verify_claim` with the evidence that convinced you. A claim you
   merely believe stays a proposal.
4. Record a claim when it would save the next session real work: a
   non-obvious dependency, a confirmed root cause, a disproved hypothesis.
   Do not record what the code or git history already says plainly.
5. When you disprove something, refute it rather than deleting or ignoring
   it — a recorded dead end stops the next agent repeating it.
6. Never present a `proposed`, `stale` or `refuted` claim to the user as
   established fact. If you cite one, say which it is.
7. Use `qlang_research` before making externally verifiable claims when QO is
   available; retain and communicate its source caveats.
8. Do not call `qlang_run_goal` for ordinary questions or without explicit user
   authorization, because it schedules background work.
9. Treat QLMS graphs and the goal history as coordination evidence, not as a
   source of truth. Verify critical claims against the referenced source.
10. Never put secrets, credentials, or absolute paths outside the workspace
    into a claim, evidence excerpt, goal, or research query.

## Other tools

- `mcp__orbitqlang__qlang_research`: Run bounded multi-source research. Use it
  for questions that need current external evidence.
- `mcp__orbitqlang__qlang_run_goal`: Create a tracked background goal. Use it
  only when the user explicitly asks for multi-agent delegation or a longer
  orchestration run. State that it runs asynchronously and report its goal ID.
- `mcp__orbitqlang__qlang_read_workspace_file`: Read a file from the QLANG
  agent workspace sandbox. Use only for a specific, repository-relative path.
