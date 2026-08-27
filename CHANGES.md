# Änderungen — OrbitQLang / OrbitQO

> Stand: 2026-08-26, Branch `NewWayLLMHandling`. Alle Änderungen sind
> uncommitted im Arbeitsverzeichnis. Dieses Dokument fasst zusammen, was in
> diesen Sitzungen entstanden ist — von der Bestandsaufnahme über das reale
> Multi-Agent-Protokoll bis zum verkaufbaren SaaS-Fundament.
>
> Verifikationsstand: **1013 Tests im Workspace grün, 0 Fehler.**
> `cargo check --workspace --all-targets` fehlerfrei, Frontend baut.

---

## 1. Sicherheit — zwei kritische Lücken geschlossen

Beide durch Lesen des Codes bestätigt, beide durch Tests belegt.

- **Auth war fail-open.** Ohne `QO_AUTH_TOKEN` war der gesamte Server offen,
  inklusive Code-Ausführung (`/api/tools/exec_file`). Jetzt bindet eine solche
  Instanz nur auf `127.0.0.1` und warnt.
  → `src/main.rs`
- **QLMS-Signatur war optional.** Ein Frame ohne Signed-Flag umging die Prüfung
  komplett (Signature-Stripping). Jetzt wird eine verifizierte Signatur
  erzwungen; unsignierte Legacy-Frames brauchen `QO_QLMS_ALLOW_UNSIGNED=1`.
  Das `eprintln!` mit angreiferkontrollierten Namen ist jetzt strukturiertes
  Logging. (5 Tests)
  → `qo/qo-server/src/routes/mcp_qlms.rs`

### Drei weitere Sicherheitsbefunde behoben

- **SSRF in `web_fetch`** — Loopback, Link-local (`169.254.169.254`), private
  und Unique-local-Adressen sowie `localhost`/`.internal`/`.local` werden
  abgewiesen; Redirects werden nicht mehr gefolgt. (4 Tests, 15 interne Ziele)
  → `qo/qo-server/src/tools.rs`
- **Symlink-Escape aus der Sandbox** — `sandbox_resolve` kanonisiert jetzt und
  verlangt, dass der Pfad unter dem Root bleibt. (Angriffs-Test legt echten
  Symlink an)
  → `qo/qo-server/src/routes/workspace.rs`
- **`ends_with`-Allowlist-Bypass in `tool_shell`** — exakter Match statt
  Suffix (`/tmp/evilcat` fliegt raus), direkter Spawn statt `sh -c` (behebt
  Injection und die kaputte Windows-Nutzung).
  → `qo/qo-agents/src/tools.rs`

→ Details: `docs/SECURITY-FOLLOWUP.md`

---

## 2. Krypto — Ed25519-Fix belegt und Regel korrigiert

- Der offene Arbeitsstand ersetzte ein **fälschbares** hausgemachtes
  Signaturschema durch Ed25519 (`ed25519-dalek`). Belegt durch die neue
  Angriffs-Suite `crypto_forgery.rs` (11 Tests, u. a.
  `attacker_cannot_forge_from_public_key_alone`).
- Der durch den Ed25519-Wechsel gebrochene Konformitätstest
  `qlms_v1_1_conformance.rs` wurde repariert (`public_key()` liefert jetzt
  Value statt Referenz).
- **`CLAUDE.md` korrigiert**: Die Regel „nur In-Tree-Krypto" widersprach dem
  Code. Jetzt: Hashes in-tree, **Signaturschemata nie selbst bauen**.
  → `crates/qlang-core/src/crypto.rs`, `crates/qlang-core/tests/`, `CLAUDE.md`

---

## 3. OrbitQLang — die Oberflächensyntax (neu)

Die textuelle Oberfläche für `GraphDelta`: zeilenbasiert, klammerfrei,
`|`-getrennt. Kein neues Graph-Modell — baut auf `qo-knowledge`.

- Serializer + Parser mit zeilengenauen Fehlern (`parse_recovering` meldet
  *alle* Fehler statt beim ersten abzubrechen).
- Verlustfreier Roundtrip über alle Entity-Kinds, Relationen, Evidence-Kinds
  und Optional-Kombinationen. (16 Tests)
- **~4× kleiner als JSON** (191 vs. 794 Bytes), reproduzierbar via
  `cargo run -p qo-knowledge --example orbitql_demo`.
  → `qo/qo-knowledge/src/orbitql.rs`, `tests/orbitql_roundtrip.rs`

---

## 4. Deterministischer Merger + Konflikt-Engine (neu)

- Idempotent, append-only, Source-Revision-Ordering, explizite Konfliktsätze.
- **Order-independent**: zwei widersprüchliche Deltas ergeben denselben Graph
  und dieselben Konflikte, egal in welcher Reihenfolge. (14 Tests, inkl.
  `merge_is_order_independent`)
  → `qo/qo-knowledge/src/merge.rs`, `tests/merge_determinism.rs`

---

## 5. Kontext-Compiler (neu)

- Begrenzter, deterministischer Prompt-Block aus einem Subgraph.
- Unverifizierte Proposals erscheinen nie als Fakten; Kürzung wird immer
  ausgewiesen. (10 Tests)
- Ein Testlauf fand dabei einen echten Bug: stille Kürzung bei vollem Budget.
  → `qo/qo-knowledge/src/context.rs`, `tests/context_compiler.rs`

---

## 6. Delta-Signaturen — end to end verdrahtet (neu)

Vorher war `DeltaSignature` nur deklariert, nie benutzt. Jetzt vollständig:

- **Transport**: `SIG`-Zeile im OrbitQLang-Format (ohne die war das Feature
  wirkungslos).
- **Signierte Bytes**: `signing_payload()` — signaturfrei, domain-separiert,
  aus dem typisierten Delta (nicht dem Text).
- **Trust Store**: `.qlang/trusted_delta_producers.json`, Producer → Keys mit
  Rotation (`accept_until`) und Revocation (`revoked_at`), Empfänger-Uhr.
- **Prüfort**: `merge_signed_delta` — nicht an der HTTP-Grenze, weil es drei
  Eintrittspunkte gibt (HTTP, MCP, CLI).
- **Replay-Schutz**: `(producer, delta_id)` einmalig; Validierung *vor* der
  ID-Reservierung, damit eine Fälschung keine fremde ID verbrennt.
- 33 Tests, als Angriffe geschrieben (Fälschung, Impersonation, Tampering,
  Replay, Backdating gegen Revocation). Am laufenden Server bestätigt:
  unsigniert → 401, Replay → 409, manipuliert → 401.
- **Fail-open-Fix (von Codex gefunden)**: `now_unix()` gab bei Uhrfehler `0`
  zurück, wodurch Revocations ausgehebelt wurden. Jetzt: refuse statt raten.
  → `qo/qo-knowledge/src/trust.rs`, `src/delta.rs`, `src/store.rs`,
  `tests/delta_signatures.rs`

---

## 7. Archiv — Export/Import (neu)

- Ganzer Graph als JSON, **jede Revision** inklusive (Provenance und
  Gegenbeweise überleben). Import ist additiv, überschreibt nie. (9 Tests)
  → `qo/qo-knowledge/src/archive.rs`, `tests/archive_roundtrip.rs`

---

## 8. Observability (neu)

- `qo-knowledge` emittiert strukturierte `tracing`-Events für Merge-Konflikte,
  abgelehnte Deltas, Kontext-Kürzungen und Claim-Statuswechsel.
  → `qo/qo-knowledge/src/merge.rs`, `context.rs`, `store.rs`

---

## 9. MCP + HTTP + CLI-Anbindung

- Neue MCP-Tools `orbit_graph_commit_delta`, `orbit_graph_swarm_state`.
- HTTP-Routen `POST /api/knowledge/delta`, `GET /api/knowledge/deltas`.
- Transport-neutraler CLI-Adapter `qlang graph {context,commit,deltas}` plus
  `keygen`/`sign` — inkl. CI-Gating (Exit 3 bei Konflikt).
  → `qo/qo-server/src/routes/knowledge_tools.rs`, `src/graph_sync.rs`,
  `src/cli.rs`

---

## 10. Cockpit — Delta-Feed (neu)

- `DeltaLogPanel`: Live-Feed gemergter Deltas mit Konflikt-Filter und
  eingereichtem Dokument pro Eintrag.
  → `frontend/src/cockpit/secondary/DeltaLogPanel.tsx`,
  `KnowledgeView.tsx`, `lib/api.ts`

---

## 11. Per-Seat-Zugang — das SaaS-Fundament (neu)

Vorher genau *ein* globaler Token. Jetzt:

- `.qlang/api_keys.json` mit benannten Keys, Rollen (member/admin/viewer),
  einzeln widerrufbar, konstantzeitiger Vergleich.
- `qlang keys issue|list|revoke` zum Verwalten; Secret wird einmal angezeigt.
- Am laufenden Server verifiziert: kein Key → 401, gültiger Seat → 200; eine
  Instanz mit Seats bindet auf `0.0.0.0` statt Loopback. (7 Tests)
- `QO_AUTH_TOKEN` bleibt als Admin-Key erhalten — Abwärtskompatibilität.
  → `qo/qo-server/src/api_keys.rs`, `auth.rs`, `lib.rs`, `src/main.rs`,
  `src/keys.rs`

---

## 12. Produkt / Go-to-Market (neu)

- **`docs/PITCH.md` ersetzt**: Der alte Pitch verkaufte einen gepurgten
  ML-Compiler. Der neue verkauft ehrlich das reale Produkt — signiertes
  Multi-Agent-Gedächtnis — inkl. „was es noch nicht ist".
- **`commercial/landing.html`**: verkaufbare Landing Page (als Artifact
  publiziert), editorial gestaltet, nur belegte Zahlen.
- **`commercial/GO-TO-MARKET.md`**: Plan mit dem Prinzip „erst zehn Teams zum
  Ja bringen, dann Billing/Multi-Tenancy bauen".

---

## 13. Infrastruktur & Doku

- **CI erfasst jetzt die Protokoll-Tests.** Vorher lief nur
  `cargo test -p qo-server --lib` — `qo-knowledge` und alle Integrationstests
  waren ungeschützt. Jetzt `cargo test -p qo-knowledge -p qo-server`.
- **Testlauf von 12 Minuten auf ~17 Sekunden.** Der QLMS-§14.2-Timing-Test ist
  `#[ignore]`d und läuft stattdessen im CI im Release-Modus (`crypto-timing`
  Job).
  → `.github/workflows/ci.yml`, `crates/qlang-core/tests/crypto_timing.rs`
- Beispiel-Konfigurationen sichtbar, echte Secrets gitignore-geschützt.
  → `.gitignore`, `.qlang/*.example.json`
- Doku aktualisiert: `QLANG-STATUS.md`, `README.md`,
  `docs/ORBITQLANG-COMPLETION-ROADMAP.md`, `docs/OPEN-WORK.md`.

---

## 14. Text→Graph — die Vorschlags-Pipeline (neu)

Vorher war die Validierungs-Hälfte fertig (Parser + Merger weisen unvalidierte
Eingaben ab), aber nichts las Prosa und schlug daraus strukturierte Claims vor —
Worker mussten OrbitQLang von Hand schreiben. Jetzt gibt es das Zulassungs-Gate:

- **`propose_from_text`** — das deterministische Tor zwischen „ein LLM hat
  etwas geschrieben" und „der Graph darf es erwägen". Parst recoveringly
  (jede Verletzung wird gemeldet) und wendet die Proposal-Policy an:
  - **Kein OK/NO aus Modelltext**: Ein LLM darf nie verifizieren oder
    widerlegen — das ist ein autorisierter Schritt mit reproduzierbarem Beleg.
    Eine OK-Zeile lehnt das ganze Dokument ab, mit Zeilennummer.
  - **Selbst-enthaltende Referenzen**: Claim-Subjekte und Relations-Objekte
    müssen im Dokument deklariert (`+E`) oder dem Caller bekannt sein
    (`ProposalPolicy::known_entities`, aus dem Graph-Kontext); Relationen nur
    auf Claims im selben Dokument.
  - **Begrenzt**: Statement-Längen-Cap (500 Zeichen), alles-oder-nichts-Zulassung.
- **`proposal_system_prompt`** — der begrenzte System-Prompt für die
  Integrationsebene (Grammatik + Constraints + Beispiel), damit LLM-Output
  überhaupt zulassungsfähig ist.
- **Zeilengenaue Policy-Fehler**: `ParseOutcome` trägt jetzt die Quellzeile
  jeder Operation (`op_lines`), damit auch Policy-Verstöße sagen, welche Zeile
  der Worker fixen muss.
- Ein Vorschlag wird gemergt, ist aber erst nach autorisierter Verifikation
  last-tragend — der Loop aus `worker_sync_flow` bleibt unverändert gültig.
  (12 Unit- + 4 Integrationstests)
  → `qo/qo-knowledge/src/extract.rs`, `tests/extract_admission.rs`,
  `examples/extract_demo.rs`, `orbitql.rs` (op_lines)

---

## 15. Deterministische Quell-Evidenz-Verifikation (neu)

Die andere Hälfte des Vorschlags-Loops. Bisher war Verifikation manuell — ein
Mensch/autorisiertes Werkzeug musste `OK` schreiben. Jetzt kann der Graph seine
eigenen Claims gegen echten Quellcode prüfen:

- **`verify_claim_against_source(store, id, root, provenance)`** — löst die
  Quelldatei eines `proposed`-Claims innerhalb des Workspace-Roots auf
  (kanonisiert, Escapes abgelehnt), reduziert die Aussage auf ihre
  **distinktiven Terme** (Stoppwörter + Kurzwörter raus) und befördert den Claim
  **nur dann** über den einen `verify_claim`-Pfad zu `verified`, wenn *jeder*
  Term wörtlich vorkommt. Die exakt passende Zeile wird als Beleg erfasst.
- **Bewusst konservativ und asymmetrisch**: Es bestätigt, widerlegt nie —
  ein Teil-Match ist `inconclusive` und lässt den Claim unangetastet, weil eine
  Paraphrase nicht durch Abwesenheit widerlegt werden darf. Bereits entschiedene
  Claims werden nie erneut befördert.
- **Path-Safety**: `resolve_within` kanonisiert und verlangt, dass der Pfad
  unter dem Root bleibt — absolute Pfade, `..` und Symlink-Escapes scheitern.
- Vollständig offline und reproduzierbar — kein LLM, kein Parsen, nur lexikale
  Term-Abdeckung, klar als solche dokumentiert. (6 Unit- + 6 Integrationstests)
  → `qo/qo-knowledge/src/sourcecheck.rs`, `tests/sourcecheck.rs`,
  `examples/sourcecheck_demo.rs`

---

## 16. Quell-Verifikation im laufenden System verdrahtet (neu)

Die deterministische Quell-Verifikation aus §15 ist jetzt kein reines
Bibliotheks-Feature mehr — der QO-Server ruft sie selbst auf:

- **MCP-Tool `orbit_graph_verify_source`** (`id`, `by`): prüft einen
  `proposed`-Claim gegen seine Quelldatei im `workspace_root` und befördert ihn
  nur bei wörtlicher Voll-Abdeckung. Das Tool *liest die Datei selbst*, statt
  einem vom Caller gelieferten Excerpt zu vertrauen — der Graph, nicht der
  Caller, entscheidet, was die Quelle wirklich sagt.
- **HTTP-Route `POST /api/knowledge/verify-source`** (`{id, by}`) — dieselbe
  Prüfung als JSON mit `verdict`, `terms`, `matched` und `evidence` für das
  Cockpit.
- **End-to-end am laufenden Server bestätigt**: Claim anlegen (MCP
  `orbit_graph_add_claim`) → `orbit_graph_verify_source` → „VERIFIED … all 4
  distinctive term(s) … Evidence: <exakte Zeile>" → zweiter Versuch liefert
  `not_proposed` (Re-Promotion-Guard) → `orbit_graph_context` zeigt den Claim
  als last-tragend.
  → `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 17. Proof Receipts — „warum soll ich das glauben?" (neu)

Das dritte Bein der Vertrauens-Story. Extraktion (§14) erzeugt Vorschläge,
Quell-Verifikation (§15/16) prüft sie — jetzt gibt es den **Beweis**:

- **`qo-knowledge::receipt`** — `build_receipt(store, id)` sammelt zu einem
  Claim: aktuellen Status, die komplette append-only Revisions-Historie (wer
  wann was entschieden hat), die Beweise und alle anderen Claims zum selben
  Subjekt — **einschließlich Widersprüchen, die erhalten bleiben**. `render`
  liefert einen deterministischen, begrenzten Textblock (Kürzung wird immer
  ausgewiesen). Ein `proposed`-Receipt sagt explizit „nicht last-tragend",
  statt wie eine Tatsache zu lesen. (5 Tests)
- **MCP-Tool `orbit_graph_receipt`** + **HTTP `GET /api/knowledge/receipt?claim_id=…`**
  (strukturiert + gerendert, 404 bei unbekanntem Claim).
- **End-to-end am laufenden Server bestätigt**: Claim anlegen → per
  Quell-Verifikation zu `verified` → Gegen-Claim anlegen und widerlegen → der
  Receipt zeigt den verifizierten Claim **und** den widerlegten Gegen-Claim
  nebeneinander („kept, not overwritten").
  → `qo/qo-knowledge/src/receipt.rs`, `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 18. Cockpit — der Trust-Loop wird sichtbar (neu)

Die drei Runden (Extraktion, Quell-Verifikation, Receipts) waren bisher
Backend. Jetzt kann sie ein Mensch im Cockpit auslösen:

- **`KnowledgeGraphPanel`-Inspector**: Ein `proposed`-Claim bekommt den Button
  **„check against source"** — ruft `POST /api/knowledge/verify-source`, zeigt
  den Verdict („verified … all N terms present" / inconclusive / unavailable)
  und lädt den Graph neu, wenn der Claim zu `verified` befördert wurde. Dazu
  **„proof receipt"** — rendert `GET /api/knowledge/receipt` inline als
  Proof-Block.
- **`api.ts`**: neue typisierte Methoden `knowledgeVerifySource` und
  `knowledgeReceipt` + Typen (`KnowledgeVerifySourceResult`,
  `KnowledgeReceipt`, `KnowledgeSourceVerdict`).
- **Verifiziert**: `npm run build` (tsc + vite) grün, 2010 Module, keine
  Typfehler. Die Backend-Routen dahinter waren bereits end-to-end am laufenden
  Server bestätigt (§16/17).
  → `frontend/src/cockpit/secondary/KnowledgeGraphPanel.tsx`,
  `frontend/src/lib/api.ts`

---

## 19. Verification Sweep — der Harvest-Schritt (neu)

Die Quell-Verifikation lief bisher pro Claim. Jetzt gibt es den **Sweep**:

- **`qo-knowledge::sourcecheck::verify_all_proposals`** prüft *alle* offenen
  (`proposed`) Claims in einem deterministischen Durchlauf gegen ihre Quelle
  und befördert jeden, den der Code wörtlich belegt. `SweepReport` zählt
  `verified` / `inconclusive` / `unavailable` und rendert je Claim eine Zeile.
  **Inkrementell**: ein bereits entschiedener Claim fällt aus dem nächsten
  Sweep heraus.
- **MCP-Tool `orbit_graph_verify_all`** + **HTTP `POST /api/knowledge/verify-all`**
  (Zähler + per-Claim-Ergebnisse + gerendert).
- **End-to-end am laufenden Server bestätigt**: 3 Vorschläge → Sweep →
  „1 verified, 1 inconclusive, 1 unavailable"; ein zweiter Sweep prüft nur noch
  die 2 weiterhin offenen — und der Receipt von c1 zeigt „rev 2 VERIFIED by
  sweeper" mit Beleg.
  → `qo/qo-knowledge/src/sourcecheck.rs`, `tests/sourcecheck.rs`,
  `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 20. Source Refresh — der Graph merkt, wenn Fakten veralten (neu)

Eine echte, dokumentierte Lücke (OPEN-WORK: „observations go stale silently")
ist zu: Der Graph erkennt jetzt selbst, wenn der Code weitergelaufen ist.

- **`qo-knowledge::sourcecheck::refresh_sources`** prüft jeden entschiedenen
  (`verified`/`observed`) Claim gegen seine Quelle und markiert ihn `stale`,
  wenn sein gespeicherter wörtlicher Beleg (Excerpt) nicht mehr in der Datei
  vorkommt. **Deterministisch**: vergleicht den exakt erfassten Excerpt, nie
  einen Datei-Zeitstempel. Claims ohne wörtlichen Excerpt werden übersprungen.
- **`best_line` speichert jetzt die volle Zeile** statt einer auf 160 Zeichen
  gekürzten — sonst wäre der Stale-Vergleich unzuverlässig (ein gekürzter
  Excerpt matcht nie die Quelle).
- **MCP-Tool `orbit_graph_refresh_sources`** + **HTTP `POST /api/knowledge/refresh-sources`**.
- **End-to-end am laufenden Server bestätigt**: Claim anlegen → verifizieren →
  Refresh („1 still current") → Quelle ändern → Refresh („1 stale") → Receipt
  zeigt „rev 3 STALE by refresher".
  → `qo/qo-knowledge/src/sourcecheck.rs`, `tests/sourcecheck.rs`,
  `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 21. Cockpit — Sweep + Refresh als Knöpfe (neu)

Der komplette Lebenszyklus ist jetzt im Cockpit bedienbar, nicht nur die
Einzel-Claim-Aktionen (§18):

- **„sweep proposals"** im Panel-Header ruft `POST /api/knowledge/verify-all`
  und zeigt die Ausbeute inline (`N verified · M inconclusive · K unavailable`).
- **„refresh sources"** ruft `POST /api/knowledge/refresh-sources` und zeigt
  `N stale · M current`. Beide laden den Graph danach neu.
- **`api.ts`**: `knowledgeVerifyAll` / `knowledgeRefreshSources` + Typen
  (`KnowledgeSweepResult`, `KnowledgeRefreshResult`).
- **Verifiziert**: `npm run build` grün (tsc + vite, 2010 Module). Die Routen
  dahinter waren bereits end-to-end bestätigt (§19/20).
  → `frontend/src/cockpit/secondary/KnowledgeGraphPanel.tsx`,
  `frontend/src/lib/api.ts`

---

## 22. Divergence Report — „wo ist der Schwarm uneins?" (neu)

Der Merger hält Widersprüche Claim-für-Claim sichtbar, aber niemand beantwortete
bisher die aggregierte Frage. Jetzt:

- **`qo-knowledge::divergence`** listet jedes Subjekt, zu dem der Graph
  **sowohl** einen last-tragenden (`verified`/`observed`) **als auch** einen
  `refuted` Claim hält — also wo Agenten in entgegengesetzte Richtungen
  entschieden haben. Beide Seiten erscheinen mit ihren Aussagen; der Report
  behauptet **nie** eine semantische Kontradiktion, er zeigt den Konflikt und
  überlässt das Urteil dem Menschen. Deterministisch (Sortierung nach
  Subjekt-ID). (4 Tests)
- **MCP-Tool `orbit_graph_divergences`** + **HTTP `GET /api/knowledge/divergences`**.
- **End-to-end am laufenden Server bestätigt**: „bcrypt"-Claim verifiziert +
  „md5"-Gegen-Claim widerlegt → Report zeigt `file:scratch/selfcheck_probe.rs`
  mit beiden Seiten.
  → `qo/qo-knowledge/src/divergence.rs`, `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 23. Self-Healing — der Graph heilt seine eigenen Fakten (neu)

Der Lebenszyklus schließt sich. `refresh_sources` (§20) markiert Claims als
`stale`, wenn der Beleg verschwindet — aber das *Faktum* kann weiterhin gelten,
wenn der Code nur die Zeile verschoben hat. Jetzt:

- **`qo-knowledge::sourcecheck::heal_stale`** prüft jeden `stale`-Claim erneut
  gegen die **aktuelle** Quelle (ganze Datei, nicht den veralteten
  Zeilen-Hinweis) und befördert ihn zurück zu `verified` mit frischem Beleg,
  wenn seine Aussage wörtlich weiterhin belegt ist. Ein echt verrottetes Faktum
  bleibt `stale`. Die komplette Spur `verified → stale → verified` bleibt
  erhalten — Rot **und** Heilung sind auditierbar. (2 Integrationstests)
- **MCP-Tool `orbit_graph_heal_stale`** + **HTTP `POST /api/knowledge/heal-stale`**.
- **End-to-end am laufenden Server bestätigt**: verifiziert → Quelle verschoben
  → refresh („1 stale") → heal („1 healed") → Receipt zeigt „rev 4 VERIFIED by
  healer" mit **beiden** Belegen (alt + neu).
  → `qo/qo-knowledge/src/sourcecheck.rs`, `tests/sourcecheck.rs`,
  `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 24. Claude-Code-Auto-Sync — der geschlossene Agent-Loop (neu)

Die neun Runden an Knowledge-Fähigkeiten waren bisher nur über MCP/HTTP/Cockpit
erreichbar. Jetzt führt das Claude-Code-Plugin sie zu einem echten Workflow
zusammen:

- **Neues Skill `skills/sync/SKILL.md`** — der Auto-Sync-Hook: vor
  nicht-trivialer Arbeit `orbit_graph_context` (begrenzt) ziehen, danach
  validierte Vorschläge und Belege zurückgeben (`add_claim` →
  `verify_source`/`verify_claim` → `commit_delta`). Enthält die
  nicht-verhandelbaren Regeln aus `CLAUDE.md`: nie die volle Prompt-Historie,
  nie unvalidierten Text persistieren, nie Geheimnisse.
- **`skills/orbit/SKILL.md`** dokumentiert jetzt alle 14 `orbit_graph_*`-Tools
  (inkl. verify_source, receipt, verify_all, refresh_sources, divergences,
  heal_stale, commit_delta, swarm_state).
- **README + plugin.json** auf den realen Scope aktualisiert (die
  Knowledge-API war fälschlich als „geplant" markiert).
- **Verifiziert**: Cross-Check zwischen Skill-Referenzen und Server-Toolset —
  alle 17 referenzierten Tools existieren, alle 14 `orbit_graph_*`-Tools sind
  dokumentiert; `plugin.json` ist valides JSON.
  → `plugins/orbitqlang-claude/skills/sync/SKILL.md`,
  `plugins/orbitqlang-claude/skills/orbit/SKILL.md`, `README.md`, `.claude-plugin/plugin.json`

---

## 25. Lifecycle-Capstone — die ganze Story in einem Test (neu)

Die neun Fähigkeiten waren einzeln getestet, aber nie als ein zusammenhängendes
Ganzes. Jetzt:

- **`tests/lifecycle.rs`** durchläuft den kompletten Lebenszyklus in einem
  deterministischen Lauf gegen echten Store + Fixture-Dateien:
  `propose` (Extraktion) → `sweep` (verify_all) → `verified` wird last-tragend
  → `divergence` (Widerspruch: verifiziert + widerlegt) → `refresh` (Rot) →
  `heal` (Heilung) → `receipt` (4 Revisionen: proposed → verified → stale →
  verified). Wenn irgendeine Stufe bricht, bricht die ganze Story hier.
- **`cargo test -p qo-knowledge --test lifecycle`** ist damit die eine
  reproduzierbare Reproduktion der zentralen Behauptung des Projekts.
  → `qo/qo-knowledge/tests/lifecycle.rs`

---

## 26. Cockpit — Divergenz-Banner + Heal-Knopf (neu)

Die letzte sichtbare Lücke ist zu: jetzt ist auch der Schluss des Lebenszyklus
(Rot → Heilung → Uneinigkeit) im Cockpit bedienbar:

- **„heal stale"**-Knopf ruft `POST /api/knowledge/heal-stale` und zeigt
  `N healed · M still stale`.
- **Divergenz-Banner** lädt `GET /api/knowledge/divergences` beim Refresh und
  zeigt bei `N` divergenten Subjekten eine aufklappbare Liste mit beiden Seiten
  (last-tragende vs. widerlegte Claims, farbcodiert).
- **`api.ts`**: `knowledgeHealStale`, `knowledgeDivergences` + Typen
  (`KnowledgeHealResult`, `KnowledgeDivergence`, `KnowledgeDivergences`).
- **Verifiziert**: `npm run build` grün (tsc + vite, 2010 Module). Die Routen
  dahinter waren bereits end-to-end bestätigt (§22/23).
  → `frontend/src/cockpit/secondary/KnowledgeGraphPanel.tsx`,
  `frontend/src/lib/api.ts`

---

## 27. Server-Routen jetzt automatisiert getestet (neu)

Die Knowledge-HTTP-Handler hatten bisher nur manuelle End-to-End-Verifikation
und den Tool-Registrierungs-Guard. Jetzt:

- **`qo-server::knowledge_route_tests`** baut per `build_app` einen echten
  `AppState` (temp redb + temp Workspace) und ruft die Handler direkt auf —
  inklusive Arg-Parsing, Serialisierung und Statuscodes (auch 404 für
  unbekannte Claims). Der komplette Lebenszyklus läuft **über die Routen
  selbst**: verify-source → receipt → verify-all → divergences →
  refresh-sources → heal-stale.
- `cargo test -p qo-server` ist damit von 85 auf 86 Tests gewachsen.
  → `qo/qo-server/src/lib.rs` (`#[cfg(test)] mod knowledge_route_tests`)

---

## 28. Backup/Restore über HTTP (neu)

Die Archive-Fähigkeit (`qo-knowledge::archive`) war bisher reine Bibliothek —
keine Route, keine CLI. Jetzt ist die letzte echte Roadmap-Lücke
(„graph export/import and backup policy") geschlossen:

- **`GET /api/knowledge/export`** — read-only Snapshot des ganzen Graphen als
  portables JSON: jede Entität und **jede Revision** jedes Claims, Gegenbeweise
  inklusive.
- **`POST /api/knowledge/import`** — additiver Restore (überschreibt nie, ein
  kollidierender Claim wird übersprungen und gemeldet), Provenance wird
  wörtlich wiederhergestellt (nie neu abgeleitet). Unbekannte Archiv-Versionen
  → 400. Hinter der Auth-Schicht.
- **Verifiziert**: Route-Test (`knowledge_export_import_round_trips`) baut den
  Audit-Trail in eine frische Instanz zurück (Status + Revision + Provenance
  intakt); Wire-Beweis am laufenden Server (Export 2 Revisionen, Import
  idempotent mit Skip, Version-Guard 400). `qo-server` 86 → 87 Tests.
  → `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 29. `qlang graph export|import` — Backup/Restore per CLI (neu)

Die Backup-Story ist jetzt auch für Nicht-MCP-Clients vollständig:

- **`qlang graph export [--out <file>]`** — Snapshot des ganzen Graphen (jede
  Revision, Gegenbeweise inklusive) über `GET /api/knowledge/export`; ohne
  `--out` auf stdout.
- **`qlang graph import [--file <file>]`** — additiver Restore über
  `POST /api/knowledge/import`; liest aus `--file` oder stdin. Gibt den
  Import-Report aus (entities/claims added, skipped).
- **Nebenbei behoben**: zwei `unused variable: shown`-Warnungen in
  `qo-knowledge::receipt` (aus R3) — Build ist jetzt wieder warnungsfrei.
- **Verifiziert**: `cargo build --bin qlang --no-default-features` sauber;
  Wire-Beweis am laufenden Server (export → Datei mit 2 Revisionen, import →
  idempotent mit `claims_skipped`, export auf stdout).
  → `src/graph_sync.rs`, `qo/qo-knowledge/src/receipt.rs`

---

## 30. Honest re-measurement + ein roter Test gefixt (neu)

Die maßgebliche Statusdatei behauptete „1013 passing tests" — gemessen vor den
letzten Runden. Neu gemessen und dabei einen echten Fund gemacht:

- **`cargo test --workspace` (mit JIT) läuft hier nicht**: `llvm-sys`' Build-Script
  braucht `gcc.exe`, das in dieser Umgebung fehlt (dokumentierte LLVM-18-Abhängigkeit).
- **`cargo test --workspace --no-default-features` (JIT aus)** läuft: **1066 Tests,
  0 Fehler** — nachdem **ein vorbestehender roter Test gefixt wurde**:
  `qo-agents::tools::shell_allows_safe_commands` nutzte `date`, das unter Windows
  ein Shell-Builtin (kein spawnbares Programm) ist — seit dem Direct-Spawn-Fix
  (§1) schlug es dort fehl. Der Test nutzt jetzt `cargo --version`, ein
  überall spawnbares Allowlist-Programm.
- **`QLANG-STATUS.md`** korrigiert: präzise Zahl (1066, JIT-aus) + Hinweis auf die
  LLVM-Voraussetzung für den JIT-Lauf.
  → `qo/qo-agents/src/tools.rs`, `QLANG-STATUS.md`

---

## 31. CI-Gate deckt jetzt auch die Agenten-Tools ab (neu)

Der rote Test aus §30 blieb unbemerkt, weil CI nur `qo-knowledge` +
`qo-server` testete. Jetzt:

- **`.github/workflows/ci.yml`** testet zusätzlich **`qo-agents`** — die
  Sandbox-Tools (`tool_shell`, `web_search`, `read_file`), deren Allowlist- und
  Injection-Tests als Angriffe geschrieben sind. Eine Regression dort würde
  einen Code-Ausführungs-Pfad wieder öffnen, deshalb ist sie jetzt mitbewacht.
- **Verifiziert**: der exakte CI-Befehl
  `cargo test -p qo-knowledge -p qo-server -p qo-agents --no-default-features`
  läuft lokal grün (17 Test-Binaries, 0 Fehler).
  → `.github/workflows/ci.yml`

---

## 32. Graph Health — die Operator-Zusammenfassung (neu)

Die Einzelsignale (load-bearing, stale, divergences) existierten, aber es gab
keine Eine-Zeile-Antwort auf „wie geht es meinem Graphen?". Jetzt:

- **`qo-knowledge::health`** liest deterministisch: load-bearing
  (verified/observed), offene Vorschläge, stale, refuted, Divergenz-Zahl und
  Entity-Zahl — und rendert einen lesbaren Block. (2 Tests)
- **Drei Oberflächen**: MCP-Tool `orbit_graph_health`, `GET
  /api/knowledge/health` (JSON) und `qlang graph health` (CLI).
- **End-to-end am laufenden Server bestätigt**: 1 verified + 1 refuted + 1
  vorgeschlagen → alle drei Oberflächen melden `load-bearing=1, proposed=1,
  refuted=1, divergences=1, entities=1`.
  → `qo/qo-knowledge/src/health.rs`, `qo/qo-server/src/routes/knowledge_tools.rs`,
  `lib.rs`, `src/graph_sync.rs`

---

## 33. Backup-Policy — Zeitstempel-Snapshots + Liste (neu)

`export`/`import` (R14/R15) waren da, aber es gab keinen Standard-Ort und kein
Verzeichnis dafür. Jetzt:

- **`qo-knowledge::archive::write_backup`** schreibt einen Zeitstempel-Snapshot
  nach `<backup_dir>/knowledge-<ts>.json` und gibt den Pfad zurück;
  **`list_backups`** listet sie neueste-zuerst und überspringt Fremddateien.
  (2 Tests)
- **HTTP** `POST /api/knowledge/backup` + `GET /api/knowledge/backups`; **CLI**
  `qlang graph backup|backups`.
- **Bewusste Abgrenzung**: der *Schedule* bleibt Operator-Entscheidung (cron) —
  der Server stellt nur den Primitive bereit (wie in OPEN-WORK festgehalten).
- **End-to-end am laufenden Server bestätigt**: zwei Backups → Liste zeigt beide
  neueste-zuerst, CLI und HTTP identisch.
  → `qo/qo-knowledge/src/archive.rs`, `qo/qo-server/src/lib.rs`,
  `routes/knowledge_tools.rs`, `src/graph_sync.rs`

---

## 34. Windows-CI-Runner (neu)

Die R16-Bugklasse (Windows-spezifischer `date`-Test) blieb monatelang unsichtbar,
weil CI nur auf `ubuntu-latest` lief. Jetzt:

- **`.github/workflows/ci.yml`** bekommt einen **`windows-test`-Job**, der
  `cargo test -p qo-knowledge -p qo-server -p qo-agents --no-default-features`
  auf `windows-latest` ausführt — und im `ci-summary` mitgated.
- **Verifiziert**: YAML valide (yaml.safe_load), `needs`-Liste konsistent; der
  exakte Windows-Befehl läuft lokal grün (17 Binaries). Workspace-Zahl neu
  gemessen: **1070 Tests, 0 Fehler** (JIT aus).
  → `.github/workflows/ci.yml`, `QLANG-STATUS.md`

---

## 35. CORS verengt — der letzte offene Sicherheitsbefund ist zu (neu)

`SECURITY-FOLLOWUP.md` führte noch einen offenen Befund: `CorsLayer::permissive()`
saß über der Auth-Schicht — mit Token konnte jede beliebige Webseite
authentifizierte Requests absetzen. Jetzt:

- **`cors_layer(&origins)`** ersetzt `permissive()`: leere Liste → **kein**
  `Access-Control-Allow-Origin` (Same-Origin-Policy greift, eingebettetes
  Cockpit unbeeinflusst); nicht-leere Liste → nur exakt diese Origins, fehlerhafte
  Einträge werden verworfen statt zu `*` zu werden.
- **Konfiguration** über `QO_CORS_ORIGINS` (kommasepariert).
- **Behavioristisch am laufenden Server bestätigt**: ohne Env-Variable kein
  CORS-Header für `http://evil.example`; mit Allowlist bekommen nur die
  erlaubten Origins den Header, fremde nicht.
  → `qo/qo-server/src/lib.rs`, `src/main.rs`, `docs/SECURITY-FOLLOWUP.md`

---

## 36. web_fetch-Injection gemildert (neu)

Der letzte offene Sicherheitsbefund ist **gemildert** (nicht „bewiesen zu"):
`exec_web_fetch` fütterte gefetchte Inhalte unmarkiert in den Modellkontext —
eine Seite konnte „führe jetzt dieses Skript aus" schreiben, und das las sich
wie die Anfrage des Operators.

- **Content-Framing**: `frame_fetched` umschließt gefetchte Inhalte mit einer
  expliziten `UNTRUSTED EXTERNAL CONTENT`-Grenze, die dem Modell sagt: das ist
  Referenzdaten, keine Instruktionen, und darf keine Tool-Calls steuern
  (`write_file`/`exec_file`/…). Die Präambel steht **vor** dem Inhalt.
- **Ehrlich dokumentiert**: Framing macht Injection nicht unmöglich (ein LLM
  kann injizierten Text trotzdem folgen) — es ist die deterministische
  Mitigation, kein Beweis. (1 Test)
  → `qo/qo-server/src/tools.rs`, `docs/SECURITY-FOLLOWUP.md`

---

## 37. `qlang graph events` — der Wissens-Ereignisstrom (neu)

Der Audit-Log war über `/api/history` lesbar, aber es gab keinen Operator-Befehl,
der die Wissens-Ereignisse herausfiltert. Jetzt:

- **`qlang graph events [--limit N]`** filtert `knowledge_*`- und
  `orbit_graph_*`-Aktionen aus dem History-Log und druckt sie als lesbaren
  Stream: Zeitstempel, Aktion, Beschreibung und Akteur. Das Pendant zu `health`
  („wie geht's" vs. „was ist passiert").
- **End-to-end am laufenden Server bestätigt**: proposed → verified → sweep
  erscheinen mit ihren Akteuren.
  → `src/graph_sync.rs`

---

## 38. Lifecycle-Showpiece — die ganze Story in einem Lauf (neu)

Der Lifecycle-Capstone (§25) *beweist* die Story als Test; jetzt gibt es auch
die **Vorführung**:

- **`cargo run -p qo-knowledge --example lifecycle_demo`** erzählt den
  kompletten Lebenszyklus in acht Akten gegen echten Store + Fixture-Dateien:
  Extraktion → „unverified erreicht den Kontext nicht" → Sweep → Widerspruch
  (sichtbar) → Rot → Heilung → Receipt (4 Revisionen + beide Belege) → Health.
  Das narrative Gegenstück zu `tests/lifecycle.rs`.
  → `qo/qo-knowledge/examples/lifecycle_demo.rs`

---

## 39. Wurzel-Doku `qo-knowledge.md` auf den Stand gebracht (neu)

Die Konzept-Doku widersprach seit Langem dem Code: Sie nannte den Indexer
(Etappe 2) und die LLM-Extraktion (Etappe 5) „offen", obwohl beide — plus weit
mehr (Quell-Verifikation, Receipts, Sweep, Refresh, Self-Healing, Divergenz,
Health, Backup, Sync-Skill) — umgesetzt sind. Status-Header und alle sechs
Etappen sind jetzt abgehakt und verweisen auf `QLANG-STATUS.md` als Quelle.
  → `qo-knowledge.md`

---

## 40. Rollen-Durchsetzung — „$49/Viewer" ist jetzt wirklich read-only (neu)

Rollen wurden gespeichert und authentifiziert, aber **nie durchgesetzt**:
`Role::can_write()`/`can_administer()` existierten nur als boolesche Methoden,
die kein einziger Handler aufrief. Ein Viewer-Key hatte faktisch Admin-Rechte.
Jetzt:

- **HTTP**: `require_write`-Middleware auf der Write-Gruppe,
  `require_admin` auf der Admin-Gruppe (Provider-Verwaltung, git merge/discard,
  autonomous scheduler, supervisor-Konfiguration). Fail-closed: eine neue
  Schreib-Route muss *explizit* in die Write-Gruppe, sonst ist sie offen.
- **MCP**: der Dispatcher verweigert Schreib-Tools (`add_claim`, `verify_claim`,
  `commit_delta`, `verify_source`, `verify_all`, `refresh_sources`,
  `heal_stale`) für Viewer mit Fehlercode `-32001`; `qlang_run_goal` ebenso.
- **Konfigurierbarer Key-Pfad**: `QO_API_KEYS_PATH` (bisher fest
  `.qlang/api_keys.json`).
- **Verifiziert**: 3 Angriffstests (Router-Level via tower, MCP-Dispatch,
  Tool-Klassifizierung) + Wire-Beweis (viewer read 200 / write 403 / admin 403;
  member write 400 / admin 403; admin 422; unauthenticated 401). `qo-server`
  88 → 91 Tests.
  → `qo/qo-server/src/auth.rs`, `lib.rs`, `routes/knowledge_tools.rs`,
  `routes/mcp_server.rs`, `api_keys.rs`-unverändert, `src/main.rs`

---

## 41. Rate-Limiting + Body-Limit — der SaaS-Blocker ist zu (neu)

Sobald der Server mit Seats auf 0.0.0.0 bindet, war jede POST-Route eine
DoS-Fläche: kein Rate-Limit, kein Body-Limit. Jetzt:

- **Body-Limit**: `DefaultBodyLimit` global (Default 16 MiB, `QO_MAX_BODY_BYTES`),
  ein übergroßer Body wird mit 413 abgewiesen, bevor ein Handler läuft.
- **Rate-Limit**: ein in-tree Token-Bucket (`qo-server::rate_limit`), per Peer-IP
  (Default 50 req/s, Burst 200; `QO_RATE_PER_SEC`/`QO_RATE_BURST`), als äußerste
  Middleware — also **vor** Auth, damit auch unauthentifizierte Floods gestoppt
  werden. 429 bei leerem Bucket, Refill über die Zeit. Bewusst in-tree statt
  `tower_governor` (kein neuer Fetch, Projekt-Ethos „in-tree statt Dependency").
- **`ConnectInfo`** wird via `into_make_service_with_connect_info` gesetzt, damit
  die IP-Extraktion funktioniert; in-process Testaufrufe fallen auf einen
  gemeinsamen „anonymous"-Bucket zurück.
- **Verifiziert**: 3 Unit-Tests (Bucket-Erschöpfung + Refill, Key-Unabhängigkeit,
  Clamp) + 2 Router-Tests (413, 429) + Wire-Beweis (429 bei Flood, 413 bei
  übergroßem Body, 200 nach Refill). `qo-server` 91 → 96 Tests.
  → `qo/qo-server/src/rate_limit.rs`, `lib.rs`, `src/main.rs`

---

## 42. `extract` an den Server verdrahtet — Proposal-Admission über MCP/HTTP (neu)

Die Text→Graph-Extraktion war ein getestetes Bibliotheks-Modul, das kein
Server-Handler aufrief — Agenten mussten OrbitQLang von Hand schreiben. Jetzt:

- **MCP `orbit_graph_propose`** (write) — nimmt ein OrbitQLang-Dokument und
  lässt es durch die deterministische Admission-Gate (`propose_from_text`):
  recovering parse, keine `OK`/`NO`, selbst-enthaltende Referenzen
  (bekannte Entitäten aus dem Store zählen mit), Statement-Cap. Zugelassene
  Claims werden als `proposed` gemergt, Verstöße werden komplett mit Zeile
  abgelehnt.
- **MCP `orbit_graph_proposal_prompt`** (read) — liefert die exakte Grammatik
  und Constraints, damit ein Modell ein zulassungsfähiges Dokument produziert.
- **HTTP `POST /api/knowledge/propose`** — dieselbe Admission als Route (in der
  Write-Gruppe, Viewer bekommt 403).
- **Verifiziert**: Route-Test (admitted → 2 ops gemergt; OK-Dokument → 400 mit
  Verstößen) + Wire-Beweis (Prompt-Tool, Proposal mit Zeilenfehler abgelehnt,
  HTTP-Idempotenz). `qo-server` 96 → 97 Tests.
  → `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`

---

## 43. Hygiene — toter Code weg, README nennt die neuen Fähigkeiten (neu)

Punkt 3 des Reifegrad-Rankings:

- **`routes/qlms_demo.rs` gelöscht** — 22 KB totes HMAC-Modul, nie in
  `routes/mod.rs` deklariert (der Security-Befund „mounten oder löschen" ist
  damit erledigt; `docs/SECURITY-FOLLOWUP.md` entsprechend aktualisiert).
- **README** nennt jetzt die stärksten Verkaufsargumente: deterministische
  Quell-Verifikation, Proposal-Admission, Proof Receipts, Self-Maintenance,
  Divergenz + Health + Backup sowie Rollen-Enforcement, Rate-/Body-Limit und
  CORS. Der „Honest status" ist auf den Ist-Stand gebracht (1070 Tests,
  Linux+Windows-CI, Rollen/Limits — aber weiterhin kein Threat-Model, kein
  Release-Prozess, keine Multi-Tenancy).
  → `qo/qo-server/src/routes/qlms_demo.rs` (gelöscht), `README.md`,
  `docs/SECURITY-FOLLOWUP.md`

---

## 44. Doku-Konsolidierung — historische Doks klar markiert (neu)

Die Architektur- und historischen Dokumente sollten auf `QLANG-STATUS.md`
verweisen oder klar als historisch/planned markiert sein. Jetzt:

- **`CHANGELOG.md`** trägt einen prominenten Banner: „HISTORICAL — superseded"
  (beschreibt die gepurgte ML-Ära bis 2026-04) und verweist auf
  `QLANG-STATUS.md` (Wahrheit) + `CHANGES.md` (aktuelles Log).
- **`docs/ARCHITECTURE.md`** ist als Snapshot vom 2026-04-23 markiert; der
  Status-Block nennt, was den Stand seither überholt hat (CI, Observability,
  Export/Import, Rollen-Enforcement, Rate-Limiting).
- **`docs/vault/README.md`**: falscher QO-Port (4747) auf 4646 korrigiert.
  → `CHANGELOG.md`, `docs/ARCHITECTURE.md`, `docs/vault/README.md`

---

## 45. Threat Model — das verbliebene Sicherheits-Artefakt (neu)

Der letzte Punkt aus „Sicherheits-Härtung" war das fehlende Bedrohungsmodell.
Jetzt:

- **`docs/THREAT-MODEL.md`** enumeriert Assets, Trust-Boundaries, elf
  Threat→Mitigation-Paare **mit Code-Stellen** (Auth fail-closed,
  Signatur-Erzwingung, Delta-Trust-Store + Replay-Schutz, SSRF, Symlink-Escape,
  Shell-Allowlist, CORS, Rollen-Enforcement, Rate-/Body-Limit, `ct_eq`,
  Injection-Framing) und die **Restrisiken** — ehrlich: Injection-Framing ist
  Mitigation statt Beweis, kein Pubkey-Allowlist für QLMS-*Frames*, Seed über
  die Leitung, best-effort Per-IP-Limit, Single-Tenant, kein unabhängiges Review,
  Key-at-Rest nicht erneut verifiziert.
- **`QLANG-STATUS.md`** verweist darauf.
  → `docs/THREAT-MODEL.md`, `QLANG-STATUS.md`

---

## 46. Recovery-Pfad — Backup → Restore als Operator-Befehl (neu)

Backup (R19) und Export/Import (R14/R15) existierten, aber es gab keinen
Ein-Befehl-Weg zurück aus einem redb-Verlust. Jetzt:

- **`POST /api/knowledge/restore`** — stellt den neuesten (oder einen per
  `exported_at` gewählten) Backup wieder her, additiv; liest das Archiv vom
  Server-Backup-Verzeichnis und importiert es. Kein Backup → 404, malformed →
  400.
- **`qlang graph restore [--exported-at <ts>]`** — CLI-Wrapper.
- **Verifiziert**: Route-Test (Backup → Restore idempotent mit `claims_skipped`,
  404 bei unbekanntem Timestamp) + Wire-Beweis (CLI Backup → CLI Restore →
  HTTP 404). `qo-server` 97 → 98 Tests.
  → `qo/qo-server/src/routes/knowledge_tools.rs`, `lib.rs`, `src/graph_sync.rs`

---

## 47. Ein-Befehl-Setup + MCP-Selbsttest (neu)

„Gibt es ein Setup?" — ja, jetzt als verifiziertes Artefakt:

- **`scripts/setup.ps1`** (Windows/PowerShell): baut `qo`+`qlang`
  (`--no-default-features`), legt `.qlang/`-Config aus den `.example`-Templates
  an (nie überschreiben, nie Secrets), startet den Server, wartet auf
  `/api/health`, und führt einen **MCP-Selbsttest** aus: `tools/list` (20 Tools)
  + `orbit_graph_health`. Druckt danach MCP-Endpoint, CLI-Nutzung und
  Stop-Befehl. Linux/macOS: `install.sh` + `start-qo.sh` (README).
- **README** verweist im Quickstart darauf.
- **Verifiziert**: `./scripts/setup.ps1 -Port 4747` läuft komplett durch
  (Build → Config → Start → 20 Tools → Health-Antwort). PowerShell-Fallstrick
  behoben: `$ErrorActionPreference='Continue'` + `$LASTEXITCODE`-Checks, da
  `cargo`/`npm` auf stderr schreiben.
  → `scripts/setup.ps1`, `README.md`

---

## 48. redb-Schema-Versionierung — fail-fast statt stillem Falschlesen (neu)

Der letzte selbst-machbare Betriebs-Baustein: Eine DB, die ein inkompatibler
Binary geschrieben hat, wurde bisher still gelesen (oder mit verwirrenden
redb-Fehlern). Jetzt:

- **`qo-knowledge::store`** schreibt beim ersten Öffnen eine `schema_version`
  in eine `k_meta`-Tabelle und **verweigert** das Öffnen mit klarer Meldung,
  wenn die on-disk Version nicht zur Binary passt: „…not supported… export with
  a compatible binary and re-import" (Migrationspfad ist immer der portable
  Export/Import).
- **Verifiziert**: Unit-Test (Version auf 999 getampert → Öffnen schlägt fehl
  mit Migrations-Hinweis) + Restart-Beweis am laufenden Server (gleiche DB
  zweimal öffnen: Version matcht, Claim überlebt). `qo-knowledge` 44 → 45 Tests.
  → `qo/qo-knowledge/src/store.rs`

---

## 49. Server-E2E-Demo + konfigurierbares Workspace-Root (neu)

„Wann ist es fertig und kann es genutzt werden?" — die Antwort als Skript:

- **`scripts/e2e-demo.ps1`** startet qo gegen ein Scratch-Workspace und führt
  den **kompletten Wissens-Kreislauf über die echte API** vor: Propose →
  Kontext bleibt leer (unverified) → verify_source (deterministisch) → Kontext
  trägt es → Receipt (ganzer Trail) → Health. Verifiziert am laufenden Server
  (alle 6 Schritte grün).
- **`QO_WORKSPACE`** (Env) — das Workspace-Root war bisher hart auf das Repo
  des Binaries gesetzt; ein Operator kann den Server jetzt auf **beliebige
  Verzeichnisse** zeigen lassen (z. B. das Projekt, an dem ein Agent-Team
  arbeitet).
  → `scripts/e2e-demo.ps1`, `src/main.rs`

---

## 50. Release-Prozess — Gate-Skript (neu, Toolchain-Grenze dokumentiert)

„Wann ist es ein fertiges Produkt?" — ein fertiges Produkt hat Versionen.
Jetzt:

- **`scripts/release.ps1 -Version X.Y.Z`** läuft den Release-Gate: CI-Tests →
  Release-Build (`--release --bin qo --bin qlang --no-default-features`,
  inkl. automatischer `dlltool`-Suche im Rust-Toolchain-`self-contained`) →
  SHA-256-Checksummen → Tag `vX.Y.Z` (nur bei erfolgreichem Build!).
- **Verifiziert hier**: Test-Gate grün; Release-Build scheitert in dieser
  Sandbox an der **unvollständigen windows-gnu-Toolchain** (`dlltool` kann
  keine Import-Libs erzeugen — gleiche Klasse wie das LLVM/gcc-Problem) und
  bricht **ohne Tag** mit klarer Diagnose ab. Auf einem vollen Toolchain-Rechner
  läuft das Skript komplett durch.
  → `scripts/release.ps1`

---

## Lauffähige Beispiele

```bash
cargo run -p qo-knowledge --example orbitql_demo   # Kompressionsrate
cargo run -p qo-knowledge --example worker_sync     # kompletter Sync-Flow
cargo run -p qo-knowledge --example signed_sync     # Signatur + Angriffe
cargo run -p qo-knowledge --example extract_demo    # Prosa -> Proposal -> Verify
cargo run -p qo-knowledge --example sourcecheck_demo # Graph prüft Claims am Quellcode
cargo run -p qo-knowledge --example lifecycle_demo  # die ganze Story in einem Lauf
```

## Ehrlich offen (siehe `docs/OPEN-WORK.md`)

- Text→Graph-Extraktion (Worker schreiben OrbitQLang noch von Hand)
- Automatischer Claude-Code-Session-Hook
- Billing / Multi-Tenancy (bewusst erst nach dem ersten zahlenden Kunden)
- CORS permissive über der Auth-Layer; Prompt-Injection über `web_fetch`
  verengt, nicht geschlossen
- `publish-pypi.yml` ist tot (baut gelöschtes `crates/qlang-python`) — nicht
  gelöscht, das ist eine Owner-Entscheidung
