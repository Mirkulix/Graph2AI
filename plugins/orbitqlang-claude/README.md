# OrbitQLang für Claude Code

Dieses Plugin verbindet Claude Code mit dem lokalen OrbitQLang-QO-Control-Plane.
Es stellt drei QO-MCP-Werkzeuge bereit: Recherche, asynchrone Ziel-Orchestrierung
und einen eingeschränkten Workspace-Lesezugriff.

## Voraussetzungen

- Claude Code ist installiert und angemeldet.
- QO ist aus diesem Repository gebaut und kann lokal auf Port `4646` laufen.
- Für Tool-Aufrufe muss QO separat gestartet werden. Das Plugin startet keinen
  Server selbst:

  ```powershell
  cargo run --bin qo -- --offline
  ```

## Lokal installieren

Im Repository-Stamm ausführen:

```powershell
claude plugin marketplace add ./ --scope project
claude plugin install orbitqlang@orbitqlang-plugins --scope project
claude plugin enable orbitqlang@orbitqlang-plugins
```

Danach Claude Code neu starten oder `/reload-plugins` ausführen. Für Entwicklung
ohne Installation kann das Plugin direkt geladen werden:

```powershell
claude --plugin-dir .\plugins\orbitqlang-claude
```

## Nutzung

- `/orbitqlang:research <Frage>` recherchiert mit dem QO-Researcher.
- `/orbitqlang:goal <Ziel>` startet ausschließlich auf expliziten Aufruf ein
  nachverfolgbares Hintergrundziel.
- `/orbitqlang:workspace` leitet Claude bei einem gezielten,
  repository-relativen Workspace-Lesezugriff an.
- `/orbitqlang:sync` ist der geschlossene Wissens-Loop: vor nicht-trivialer
  Arbeit begrenzten Graph-Kontext ziehen, danach validierte Vorschläge und
  Belege zurückgeben.
- `/orbitqlang:orbit` enthält die allgemeinen Sicherheits- und Einsatzregeln.

Das Plugin lädt die MCP-Konfiguration aus `.mcp.json`. Solange QO nicht läuft,
sind die Skills sichtbar, die QO-Werkzeuge jedoch nicht verfügbar.

## Aktueller Umfang

Der QO-Server stellt über `POST /mcp/v1` die Agenten-Werkzeuge
`qlang_research`, `qlang_run_goal` und `qlang_read_workspace_file` bereit, plus
die **Knowledge-Graph-Werkzeuge** `orbit_graph_search`, `orbit_graph_neighbors`,
`orbit_graph_impact`, `orbit_graph_context`, `orbit_graph_add_claim`,
`orbit_graph_verify_claim`, `orbit_graph_commit_delta`,
`orbit_graph_swarm_state`, `orbit_graph_verify_source`,
`orbit_graph_receipt`, `orbit_graph_verify_all`,
`orbit_graph_refresh_sources`, `orbit_graph_divergences` und
`orbit_graph_heal_stale`. Damit ist der volle Lebenszyklus
`proposed → verified → stale → verified` über das Plugin ansprechbar — siehe
`skills/orbit/SKILL.md` und `skills/sync/SKILL.md`.
