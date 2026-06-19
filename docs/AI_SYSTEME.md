# AI-Systeme im OrbitQO-Stack

> Status: aktiv, in Entwicklung

## Zweck

Diese Dokumentation beschreibt das AI-System, das in diesem Repo aktuell
entwickelt wird. Der Fokus liegt nicht mehr auf einer "neuen Sprache" als
Endprodukt, sondern auf einem lokalen und kontrollierbaren Multi-Agent-System
fuer technische Arbeit.

Kurzform:

- `QO` ist die Produkt- und Supervisor-Ebene
- `QLANG` ist die interne Graph- und Handover-Schicht
- `QLMS` ist das strukturierte Transportformat zwischen Agenten/Systemen
- `DeepSeek` ist aktuell der primaere LLM-Pfad fuer den neuen Multi-Agent-Flow

## Produktbild

Das angestrebte Produkt ist eine lokale Multi-Agent-Control-Plane fuer
technische Arbeit. Ein Nutzer gibt ein Ziel vor, das System plant die Arbeit,
bearbeitet sie, prueft das Ergebnis und speichert den Lauf sichtbar im Cockpit.

Der aktuelle Zielpfad ist:

1. `Planner` zerlegt das Ziel in einen umsetzbaren Plan.
2. `Worker` erzeugt die eigentliche Antwort oder das Artefakt.
3. `Reviewer` prueft das Ergebnis gegen Ziel und Abnahmekriterien.
4. `Supervisor/QO` speichert den Run, zeigt Status und Historie an und streamt
   Zwischenstaende an das Frontend.

## Aktueller Stand

Der derzeit verifizierte Produktpfad ist ein DeepSeek-first Multi-Agent-Flow:

- Backend-Route: `/api/multi-agent/run`
- Asynchroner Start: `/api/multi-agent/runs/start`
- Run-Historie: `/api/multi-agent/runs`
- Run-Details: `/api/multi-agent/runs/{id}`
- Live-Updates: `/api/multi-agent/stream`
- Cockpit-Ansicht: `Multi-Agent`

Aktuell implementiert:

- Planner/Worker/Reviewer-Workflow
- Run-Historie im Serverzustand
- Live-Snapshots per SSE
- Cockpit-Ansicht fuer Start, Beobachtung und Detailansicht
- optionale Artefakt-Schreibpfade ueber den Workspace-Sandbox-Bereich
- DeepSeek als aktiver Provider fuer alle drei Rollen

Aktuell nicht vorhanden oder bewusst nicht behauptet:

- vollautonome Tool-Nutzung pro Agent
- persistentes Langzeitgedaechtnis pro Multi-Agent-Run
- rollenbasierte Rechteverwaltung
- verteilte Produktionsorchestrierung

## Bausteine

### QO Server

Der `qo`-Server ist die zentrale Laufzeit:

- stellt HTTP- und SSE-Endpunkte bereit
- haelt den gemeinsamen `AppState`
- speichert Run-Snapshots und Historie
- liefert das Cockpit-Frontend aus

### Multi-Agent-Route

Die Route `multi_agent` bildet den aktuellen Produktkern:

- validiert Requests
- fuehrt die Rollen sequenziell aus
- verarbeitet Review-Schleifen
- extrahiert Artefakte
- aktualisiert den Run-Status ueber den gesamten Lebenszyklus

### LLM-Router

Die Provider-Auswahl liegt im LLM-Router. Fuer den aktuellen Produktpfad gilt:

- `Planner` -> DeepSeek Reasoning-Modell
- `Worker` -> DeepSeek Coding-/Chat-Modell
- `Reviewer` -> DeepSeek Review-/Chat-Modell

Andere Provider koennen im Repo existieren, sind aber fuer diesen
Produktpfad nicht der Standard.

### Frontend / Cockpit

Das Cockpit ist die operative Oberflaeche:

- startet neue Runs
- zeigt Run-Historie
- zeigt Plan, Worker-Runden und Reviewer-Runden
- beobachtet Live-Ereignisse ueber SSE

## Typischer Ablauf

Ein typischer Lauf sieht so aus:

1. Nutzer gibt im Cockpit ein Ziel ein.
2. Das Frontend startet einen Run ueber `/api/multi-agent/runs/start`.
3. Der Server setzt den Run auf `queued`, dann `planning`.
4. Der Planner erzeugt Plan und Abnahmekriterien.
5. Der Worker erzeugt die erste Arbeitsantwort.
6. Der Reviewer entscheidet `approved` oder `needs_revision`.
7. Bei Bedarf folgt eine begrenzte Ueberarbeitung.
8. Der finale Stand wird gespeichert und bleibt im Cockpit abrufbar.

## DeepSeek-Konfiguration

Fuer den aktuellen Produktpfad wird DeepSeek ueber `.env` konfiguriert.
Minimal notwendig:

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
```

Je nach Rollenmapping koennen weitere Modellnamen intern verwendet werden,
zum Beispiel `deepseek-reasoner` oder `deepseek-coder`.

## Linux-Zielbild

Das System soll zusaetzlich auf Linux stabil betrieben werden. Die Architektur
passt gut zu einem Linux-Zielsystem, weil der eigentliche Produktpfad aus:

- einem Rust-Server
- einem gebauten Vite/React-Frontend
- HTTP/SSE-Endpunkten
- externen LLM-Providern

besteht und damit keine Windows-spezifische GUI-Bindung hat.

### Empfohlener Linux-Betrieb

Fuer Linux ist der pragmatische Zielzustand:

1. Rust-Toolchain installieren
2. `frontend` bauen
3. `.env` mit DeepSeek-Key bereitstellen
4. `qo` als Dienst starten
5. Reverse Proxy vor `localhost:4646` setzen

Beispielhafter Ablauf:

```bash
cargo build --release
cd frontend && npm ci && npm run build
cd ..
DEEPSEEK_API_KEY=... cargo run --bin qo -- --offline
```

### Linux-Migrationspunkte

Beim Umzug auf Linux sind vor allem diese Punkte relevant:

- Dateipfade und Schreibrechte fuer den Workspace-Sandbox-Bereich
- Prozessstart als `systemd`-Service oder Container
- Reverse Proxy und TLS
- saubere Ablage der `.env`
- Trennung von Build- und Laufzeitumgebung

## Betriebsnotizen

Der aktuelle Produktpfad ist fuer transparente, kontrollierte Runs gedacht.
Deshalb sollte der Betrieb immer diese Eigenschaften behalten:

- klar sichtbare Run-Historie
- begrenzte Review-Schleifen
- nachvollziehbare Rollen
- ehrlich dokumentierter Provider-Pfad
- keine versteckten "magischen" Nebenlaeufe

## Nicht-Ziele

Dieses System ist aktuell nicht gedacht als:

- autonomer Forschungs-Schwarm ohne Kontrolle
- unsichtbarer Hintergrund-Agent mit Vollzugriff
- vollstaendige Ersatzplattform fuer CI/CD oder MLOps
- allgemeine AGI-Laufzeit

## Naechste Ausbaustufen

Realistische naechste Produktstufen sind:

1. konfigurierbare Modelle pro Rolle
2. bessere Artefakt-Preview im Cockpit
3. optionale Tool-Ausfuehrung unter strenger Sandbox
4. persistente Speicherung von Runs
5. Linux-Service-Setup fuer Dauerbetrieb
