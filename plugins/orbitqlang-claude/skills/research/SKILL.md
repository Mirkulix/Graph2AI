---
description: Research a question through OrbitQLang's local QO control plane. Use when the user requests current, externally verifiable evidence and QO is available.
---

# OrbitQLang Research

Use `mcp__orbitqlang__qlang_research` with the user's question as `query`.

Before the call, ensure that the local QO MCP tools are available. If they are
not, state that QO must be started with `qo --offline` and do not start it
automatically.

Present research results as sourced, potentially fallible evidence. Separate
the QO result from Claude's own inference and point out missing, weak, or
conflicting evidence. Never include secrets, credentials, or private absolute
paths in the query.
