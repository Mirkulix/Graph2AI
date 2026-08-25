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
- `/orbitqlang:orbit` enthält die allgemeinen Sicherheits- und Einsatzregeln.

Das Plugin lädt die MCP-Konfiguration aus `.mcp.json`. Solange QO nicht läuft,
sind die Skills sichtbar, die QO-Werkzeuge jedoch nicht verfügbar.

## Aktueller Umfang

Der aktuelle QO-Server implementiert `qlang_research`, `qlang_run_goal` und
`qlang_read_workspace_file`. Die in `qo-knowledge.md` beschriebene
Knowledge-Graph-API ist geplant und wird erst nach ihrer Implementierung als
zusätzliche MCP-Werkzeuge in dieses Plugin aufgenommen.
