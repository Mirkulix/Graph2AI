# OrbitQLang für Claude Code

Dieses Plugin verbindet Claude Code mit dem lokalen OrbitQLang-QO-Control-Plane
und stellt **20 MCP-Werkzeuge** bereit: den Wissensgraphen (17 `orbit_graph_*`
Tools: Kontext, Vorschlag, Verifikation gegen Quellcode, Belege, Divergenzen,
Health), Recherche, Ziel-Orchestrierung und einen eingeschränkten
Workspace-Lesezugriff.

## Schnellstart (lokal, eine Maschine)

1. **Server starten** — `start-cockpit.cmd` im Repository-Stamm. Baut beim
   ersten Mal selbst und öffnet das Cockpit als eigenes Fenster.
2. **Prüfen, dass es läuft:**

   ```powershell
   .\scripts\verify-install.ps1
   ```

   Sechs Checks vom Handshake bis zum echten Tool-Aufruf. Bei einem Fehler
   nennt die Ausgabe den Fix, nicht nur das Problem. Startet selbst einen
   Server, falls keiner läuft, und beendet nur den selbst gestarteten.
3. **Claude Code neu starten**, damit das Plugin greift.

## Warum kein Token nötig ist

`start-cockpit.cmd` setzt `QO_LOCAL_MODE=1`. Damit gilt: Ein Aufruf **von
dieser Maschine** ist der Operator und braucht kein Token — und der Server
bindet ausschließlich auf `127.0.0.1`, ist also aus dem Netz nicht erreichbar.

Das löst ein konkretes Problem: Sobald in `.qlang/api_keys.json` auch nur ein
Seat ausgestellt ist, verlangte vorher **jede** Route ein Token. Ein lokal
installierter MCP-Client bekam dann `401` — was Claude Code als
`JSON Parse error: Unrecognized token ' '` meldet, weil die leere Antwort kein
JSON ist. Die Fehlermeldung zeigt nicht auf die Ursache.

Für eine Instanz im Netz: `QO_LOCAL_MODE` weglassen und `QO_AUTH_TOKEN` oder
einen ausgestellten Seat verwenden. Beide Wege funktionieren unverändert.

## Voraussetzungen

- Claude Code ist installiert und angemeldet.
- QO ist aus diesem Repository gebaut (`start-cockpit.cmd` erledigt das).
- QO läuft auf Port `4646` — darauf zeigt die `.mcp.json` des Plugins.

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
