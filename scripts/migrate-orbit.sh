#!/bin/bash
set -e

ORBIT_DB="/home/mirkulix/Orbit/backend/orbit.db"
EXPORT_DIR="data/orbit-import"

echo "=== QO Orbit Migration ==="

if [ ! -f "$ORBIT_DB" ]; then
    echo "Orbit DB not found at $ORBIT_DB"
    echo "Skipping migration."
    exit 0
fi

mkdir -p "$EXPORT_DIR"

echo "Exporting messages..."
sqlite3 "$ORBIT_DB" "SELECT json_object('id', id, 'role', role, 'content', content, 'timestamp', timestamp) FROM messages ORDER BY id;" > "$EXPORT_DIR/messages.jsonl" 2>/dev/null || echo "[]" > "$EXPORT_DIR/messages.jsonl"
MSG_COUNT=$(wc -l < "$EXPORT_DIR/messages.jsonl")

echo "Exporting goals..."
sqlite3 "$ORBIT_DB" "SELECT json_object('id', id, 'description', description, 'status', status, 'created_at', created_at) FROM goals ORDER BY id;" > "$EXPORT_DIR/goals.jsonl" 2>/dev/null || echo "[]" > "$EXPORT_DIR/goals.jsonl"
GOAL_COUNT=$(wc -l < "$EXPORT_DIR/goals.jsonl")

echo "Exporting patterns..."
sqlite3 "$ORBIT_DB" "SELECT json_object('id', id, 'pattern', pattern, 'frequency', frequency, 'created_at', created_at) FROM learned_patterns ORDER BY id;" > "$EXPORT_DIR/patterns.jsonl" 2>/dev/null || echo "[]" > "$EXPORT_DIR/patterns.jsonl"
PATTERN_COUNT=$(wc -l < "$EXPORT_DIR/patterns.jsonl")

echo "Exporting proposals..."
sqlite3 "$ORBIT_DB" "SELECT json_object('id', id, 'title', title, 'description', description, 'status', status, 'created_at', created_at) FROM purposes ORDER BY id;" > "$EXPORT_DIR/proposals.jsonl" 2>/dev/null || echo "[]" > "$EXPORT_DIR/proposals.jsonl"
PROPOSAL_COUNT=$(wc -l < "$EXPORT_DIR/proposals.jsonl")

echo ""
echo "=== Migration Complete ==="
echo "Messages:  $MSG_COUNT"
echo "Goals:     $GOAL_COUNT"
echo "Patterns:  $PATTERN_COUNT"
echo "Proposals: $PROPOSAL_COUNT"
echo ""
echo "Exported to: $EXPORT_DIR/"
echo "Obsidian Vault: No migration needed (shared location)"
