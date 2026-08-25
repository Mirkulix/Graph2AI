---
description: Read a specific repository-relative file through the OrbitQLang QO workspace sandbox when the QO tool is available.
---

# OrbitQLang Workspace Reader

Use `mcp__orbitqlang__qlang_read_workspace_file` only for a specific,
repository-relative path that is necessary for the task. Supply that path as
`{ "path": "..." }`.

Do not use this tool for broad discovery, arbitrary machine files, credentials,
or paths outside the workspace sandbox. If the requested path is not specific,
ask the user to name it or use ordinary repository navigation first.
