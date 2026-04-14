#!/bin/bash
echo "=== Stopping Orbit Services ==="

# Stop PM2 processes
if command -v pm2 &>/dev/null; then
    pm2 stop orbit 2>/dev/null && echo "Stopped: orbit backend" || echo "orbit backend not running"
    pm2 stop acios-sidecar 2>/dev/null && echo "Stopped: acios sidecar" || echo "acios sidecar not running"
else
    echo "PM2 not found, skipping"
fi

echo ""
echo "Orbit stopped. Start QO with: ./scripts/start-qo.sh"
