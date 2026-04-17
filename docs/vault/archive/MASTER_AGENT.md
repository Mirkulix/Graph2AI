# QLANG Master Agent — Autonomer Projekt-Push

## Was es ist

Ein persistenter Orchestrator-Agent der kontinuierlich am QLANG-Projekt
arbeitet: Status checken, Priorität wählen, Sub-Agenten spawnen, tracken.

## Die drei Laufmodi

### Modus 1: Interaktiv
```
Du im Chat:  @qlang-master check status
             @qlang-master mach den nächsten Schritt
             @qlang-master commit pending work und push P1.1
```

### Modus 2: Recurring (dieser ist aktiv)
```
Cron:  17 * * * *   (jede Stunde um :17)
Job-ID: 4d90aef1
Duration: 7 Tage Auto-Expire
```
Claude Code triggert stündlich den Master. Er macht eine kleine Arbeitseinheit
(10 Min oder 1 Sub-Agent), berichtet kurz, und wartet auf die nächste Stunde.

Pausieren: `CronDelete` mit der Job-ID.

### Modus 3: Shell Launcher
```bash
./scripts/qlang-master.sh                    # Status-Check
./scripts/qlang-master.sh "push P1.1"        # Explizite Aufgabe
```

## Prioritäten-Leiter (was der Master in Reihenfolge angeht)

### P0 Foundation — muss grün sein
- [x] Build mit `--features cuda` kompiliert
- [x] GPU QAT >80% ternary (erreicht: 84.6% in 24s)
- [x] `/api/demo/mnist-igqk` stabil
- [ ] Alle Tests green

### P1 Killer-Demos — bevor irgendwas anderes
- [ ] **Demo A: QLMS AI-to-AI Round-Trip** (2 QO-Server, signierter Model-Transfer)
- [ ] **Demo B: Federated Organism** (3 Knoten, Gossip-Merge)

### P2 Standards
- [ ] QLMS v1.1 Spec für Linux Foundation AAIF
- [ ] MCP ↔ QLMS Bridge
- [ ] GitHub README + 90s Demo-Video

### P3 Tiefe
- [ ] Tokenizer in QLMB embedden
- [ ] Spiking MNIST 85%+
- [ ] Backprop CNN 95%+
- [ ] Loihi Compile-Target
- [ ] Security Audit

## Was der Master NICHT tut

- Keine neuen Dashboards bevor AI-to-AI Demo läuft
- Keine weiteren STATUS.md / SUMMARY.md Files
- Kein Feature-Creep — 2 Demos zu Ende bringen, nicht 10 anfangen
- Kein Fake — 84.6% sind 84.6%, keine "99.6% MNIST" Marketing-Zahlen
