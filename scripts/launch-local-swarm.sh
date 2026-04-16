#!/usr/bin/env bash
# launch-local-swarm.sh
# -----------------------------------------------------------------------------
# Spawn 2 additional qo-server processes (ports 4647 + 4648) alongside a
# primary node on 4646, giving a 3-node federation that the Swarm Map
# will render with real peers. Requires a debug or release build of `qo`.
#
# Usage:
#   ./scripts/launch-local-swarm.sh              # launches nodes 2 + 3 in background
#   ./scripts/launch-local-swarm.sh --stop       # stops nodes 2 + 3
#
# Notes:
# - Assumes node 1 is already running at :4646 (cargo run -p qo-server).
# - Each node uses its own QO_DATA_DIR so they don't stomp each other.
# - Peer seeds chain all three so gossip converges quickly.
# - PID files land in .swarm-pids/ (gitignored via .gitignore's data/ rule
#   — add explicitly if needed).
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${QO_BIN:-target/debug/qo.exe}"
if [[ ! -f "$BIN" ]]; then
  BIN="${QO_BIN:-target/debug/qo}"
fi
if [[ ! -f "$BIN" ]]; then
  echo "[swarm] qo binary not found at target/debug/qo(.exe) — run 'cargo build -p qo-server' first" >&2
  exit 1
fi

PID_DIR="$ROOT/.swarm-pids"
mkdir -p "$PID_DIR"

SEEDS="localhost:4646,localhost:4647,localhost:4648"

stop_node() {
  local port="$1"
  local pid_file="$PID_DIR/node-$port.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill "$pid" 2>/dev/null; then
      echo "[swarm] stopped node on :$port (pid $pid)"
    fi
    rm -f "$pid_file"
  fi
}

start_node() {
  local port="$1"
  local pid_file="$PID_DIR/node-$port.pid"
  local log_file="$PID_DIR/node-$port.log"
  local data_dir="$ROOT/data/swarm-node-$port"
  mkdir -p "$data_dir"

  stop_node "$port"

  echo "[swarm] starting node on :$port (data: $data_dir)"
  QO_PORT="$port" \
  QO_NODE_ADDR="localhost:$port" \
  QO_DATA_DIR="$data_dir" \
  PEER_DISCOVERY_SEEDS="$SEEDS" \
    nohup "$BIN" >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  sleep 1
  if ! curl -s -o /dev/null "http://localhost:$port/api/federation/peers"; then
    echo "[swarm] node :$port not responding yet — check $log_file"
  fi
}

if [[ "${1:-}" == "--stop" ]]; then
  stop_node 4647
  stop_node 4648
  echo "[swarm] stopped"
  exit 0
fi

if ! curl -s -o /dev/null "http://localhost:4646/api/federation/peers"; then
  echo "[swarm] primary node on :4646 is not responding — start it first with 'cargo run -p qo-server'" >&2
  exit 1
fi

start_node 4647
start_node 4648

echo
echo "[swarm] all nodes up. Verify with:"
echo "        curl http://localhost:4646/api/federation/peers"
echo
echo "[swarm] logs: $PID_DIR/node-{4647,4648}.log"
