#!/bin/bash
VAULT="$HOME/Dokumente/Obsidian Vault/QO"
mkdir -p "$VAULT"/{Bewusstsein,Chat,Ziele,Agenten,Patterns,Evolution,Reflexionen}

# Create index
cat > "$VAULT/Index.md" << 'EOF'
---
type: index
tags: [qo, index]
---

# QO — Map of Content

## Bewusstsein
_Automatisch generiert durch QO Consciousness Stream_

## Chat
_Chat-Protokolle_

## Ziele
_Goal-Tracking und Ergebnisse_

## Agenten
_Agent-Aktivitäten_

## Patterns
_Erkannte Muster_

## Evolution
_Quantum Evolution und Proposals_

## Reflexionen
_System-Reflexionen_
EOF

echo "QO Obsidian Vault erstellt: $VAULT"
