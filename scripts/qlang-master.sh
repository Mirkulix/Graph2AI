#!/bin/bash
# QLANG Master Agent Launcher
# Usage:
#   ./scripts/qlang-master.sh                    — Interactive status check
#   ./scripts/qlang-master.sh "mach alles"       — Autonomous mode
#   ./scripts/qlang-master.sh --push             — Explicit push mode (commit + agents)
#   ./scripts/qlang-master.sh --loop 30m         — Repeat every 30 minutes

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"

# Default prompt: invoke the master agent with empty input (triggers status + proposal)
PROMPT="${1:-Check status und schlage nächsten Schritt vor.}"
MODE="$2"

echo "══════════════════════════════════════════"
echo "  QLANG MASTER AGENT"
echo "══════════════════════════════════════════"
echo "  Working dir: $PROJECT_DIR"
echo "  Prompt:      $PROMPT"
echo "  Mode:        ${MODE:-interactive}"
echo "══════════════════════════════════════════"
echo ""

# Invoke the master agent via Claude Code CLI (if available)
# or log the intent so an interactive Claude session can pick it up
if command -v claude &>/dev/null; then
  # Use Claude Code with the qlang-master agent
  claude --agent qlang-master "$PROMPT"
else
  # Fallback: write a trigger file the user can pick up in Claude Code
  mkdir -p .claude/triggers
  TRIGGER_FILE=".claude/triggers/master-$(date +%Y%m%d-%H%M%S).md"
  cat > "$TRIGGER_FILE" <<EOF
# QLANG Master Trigger — $(date)

Prompt: $PROMPT
Mode: ${MODE:-interactive}

Next time you open Claude Code for this project, tell it:
"Invoke the qlang-master agent: $PROMPT"

Or paste this directly:
> @qlang-master $PROMPT
EOF
  echo "Claude CLI nicht gefunden. Trigger geschrieben: $TRIGGER_FILE"
  echo ""
  echo "In Claude Code: '@qlang-master $PROMPT' eingeben"
fi
