# QO Knowledge Graph

> Status: **alle Etappen 1–6 umgesetzt.** `QLANG-STATUS.md` bleibt die maßgebliche
> Quelle für das, was wirklich läuft. Dieses Dokument beschreibt die Konzeption
> (Datenmodell, Vertrauensmodell, Schnittstellen); die Etappen unten sind
> abgehakt, und der tatsächliche Funktionsumfang ist weit darüber
> hinausgewachsen: Extraktion (Text→Graph), deterministische Quell-Verifikation,
> Proof Receipts, Sweep, Source-Refresh, Self-Healing, Divergenz-Report, Health,
> Backup/Restore und ein Claude-Code-Sync-Skill.
>
> Ursprünglich „offen" waren Etappe 2 (deterministischer Repository-Indexer) und
> Etappe 5 (LLM-Extraktion) — beide sind erledigt.

## Aufgabe

OrbitQLang erhält mit `qo-knowledge` eine dauerhafte, überprüfbare Wissensschicht
für Projekte, Agenten und ihre Ausführungen. Sie ergänzt den bestehenden
Ausführungs- und Kommunikationsgraphen; sie ersetzt ihn nicht.

Das System soll verlässliche Antworten auf Fragen wie diese ermöglichen:

- Welche Komponenten, Dateien, Symbole und APIs gehören zusammen?
- Welche Aussage wird durch welchen Code, Test, Dokument oder Run belegt?
- Welche Abhängigkeiten und Auswirkungen hat eine Änderung?
- Welcher Agent hat Wissen erzeugt, geprüft oder widerlegt?

## Grundprinzip

Ein LLM darf Wissen extrahieren und **Claims** vorschlagen. Es darf jedoch keinen
unbelegten Vorschlag als Wahrheit speichern. Deterministische Analyse und
Validierungsregeln erfassen überprüfbare Fakten aus dem Code und kontrollieren
alle Schreibvorgänge.

Jeder Claim muss eine Provenienz besitzen: Quelle, Erfassungszeit, Erzeuger,
Git-Revision oder Run-ID sowie einen Vertrauens- und Prüfstatus.

## Datenmodell

`qo-knowledge` führt mindestens diese typisierten Konzepte:

- **Entity**: z. B. Repository, Datei, Symbol, Service, Endpoint, Konzept oder
  Agent.
- **Relation**: gerichtete und typisierte Verbindung, z. B. `defines`,
  `calls`, `depends_on`, `implements` oder `contradicts`.
- **Claim**: eine überprüfbare Aussage über Entitäten und Relationen.
- **Evidence**: Beleg zu einem Claim, z. B. Datei und Zeilenbereich, Git-Commit,
  Testlauf, Tool-Run oder externe Quelle.
- **Revision**: unveränderliche Version eines Claims oder einer Relation.

Widersprüchliche Claims werden nicht überschrieben. Sie bleiben mit ihrem
Status und ihren Belegen sichtbar.

## Vertrauensmodell

Claims besitzen einen expliziten Status:

- `observed`: direkt aus Code, Konfiguration, Test oder Tool-Ausgabe erfasst.
- `proposed`: durch ein LLM oder einen Agenten vorgeschlagen.
- `verified`: durch reproduzierbaren Beleg oder autorisierte Prüfung bestätigt.
- `stale`: durch eine neuere Revision möglicherweise nicht mehr aktuell.
- `refuted`: durch Gegenbeleg widerlegt.

Nur `observed` und `verified` dürfen ohne Kennzeichnung als belastbarer Kontext
an Planungs- und Ausführungsagenten ausgegeben werden.

## Architektur

Der Dienst wird als Rust-Crate `qo-knowledge` im QO-Workspace implementiert.
Er bietet eine typisierte API für Schreiben, Versionierung und Traversals. Die
Speicherung benötigt Indizes für Entitäten, Relationen, Quellen, Revisionen und
Zeitpunkte; ein bloßes Ablegen ganzer JSON-Graphen genügt nicht.

Code-Analyse erzeugt zunächst deterministische Fakten über Dateien, Symbole,
Imports, Aufrufe, APIs und Tests. Ein LLM ergänzt anschließend semantische
Zusammenhänge als `proposed` Claims. Die bestehende QLMS-Schicht transportiert
signierte Wissensergänzungen zwischen Agenten.

## Schnittstellen

QO stellt dem Claude-Code-Plugin und anderen MCP-Clients mindestens diese
Fähigkeiten bereit:

- `orbit_graph_search`: Entitäten und belegte Claims suchen.
- `orbit_graph_neighbors`: Beziehungen und Auswirkungen traversieren.
- `orbit_graph_add_claim`: einen Claim samt Belegen als Vorschlag anlegen.
- `orbit_graph_verify_claim`: einen Claim anhand eines reproduzierbaren Belegs
  bestätigen oder widerlegen.
- `orbit_graph_context`: kompakten, quellengebundenen Kontext für eine Aufgabe
  liefern.

Schreibende Werkzeuge benötigen eine Agentenidentität, Audit-Eintrag und eine
Policy-Prüfung. Geheimnisse, Zugangsdaten und Inhalte außerhalb des erlaubten
Projektbereichs dürfen nicht als Evidenz oder Claim gespeichert werden.

## Umsetzungsetappen

Alle erledigt (`QLANG-STATUS.md` ist die Quelle für Details):

1. ~~Typen, Persistenz, Revisionen und Indizes in `qo-knowledge` implementieren.~~ ✓
2. ~~Deterministischen Repository-Indexer für Code-Entitäten und Abhängigkeiten
   anbinden.~~ ✓ (live-Scan, Revisionen)
3. ~~MCP-Read-Tools und nachvollziehbare Graph-Abfragen bereitstellen.~~ ✓
4. ~~Claim- und Evidenz-Write-API mit Policies, Audit und Verifikation ergänzen.~~ ✓
   (inkl. Ed25519-Delta-Signaturen, Trust-Store, Replay-Schutz)
5. ~~LLM-Extraktion als Vorschlags-Pipeline anschließen.~~ ✓
   (`qo-knowledge::extract` + deterministische Quell-Verifikation)
6. ~~Den Claude-Code-Plugin auf die Graph-Tools umstellen und End-to-End
   testen.~~ ✓ (`plugins/orbitqlang-claude`, inkl. Sync-Skill)

## Erfolgskriterium

Eine Antwort aus OrbitQLang kann zu jeder wichtigen Aussage die zugehörigen
Entitäten, Beziehungen, Belege und Revisionen ausgeben. Ein LLM beschleunigt
den Aufbau des Wissens, doch die Nachvollziehbarkeit und Gültigkeit des Graphen
beruhen auf Regeln, Quellen und reproduzierbarer Prüfung.
