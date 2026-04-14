#!/bin/bash
# QLMS federated launcher — starts 3 QO servers on ports 4646, 4747, 4848
# with isolated data dirs data/server-{a,b,c} and distinct QO_NODE_IDs so
# each node trains its specialist on a different MNIST partition.
#
# Usage:
#   ./scripts/qlms-federated-launcher.sh
#
#   # On node A, gossip with B + C:
#   curl -X POST http://localhost:4646/api/qlms/federation/gossip \
#        -H 'content-type: application/json' \
#        -d '{"peers":["localhost:4747","localhost:4848"]}'
#
#   # Measure a single node's accuracy on its holdout:
#   curl http://localhost:4646/api/qlms/federation/eval
#
# Stops old instances on these ports before starting.

set -u

cd /home/mirkulix/AI/neoqlang/qlang

TORCH_LIB=$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>/dev/null || echo "")
CUDA_LIBS="/home/mirkulix/.local/lib/python3.14/site-packages/nvidia/cu13/lib"

PORTS=(4646 4747 4848)
NODES=(a b c)

# Kill any old instances on these ports
for port in "${PORTS[@]}"; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Killing existing process(es) on :$port  ($pids)"
    kill $pids 2>/dev/null || true
  fi
done
sleep 1

PIDS=()
for i in 0 1 2; do
  PORT=${PORTS[$i]}
  NODE=${NODES[$i]}
  DATA_DIR="data/server-$NODE"
  LOG="/tmp/qo-$NODE.log"
  mkdir -p "$DATA_DIR"

  LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so" \
  LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_LIBS}" \
  QO_PORT="$PORT" QO_DATA_DIR="$DATA_DIR" QO_NODE_ID="node-$NODE" \
    nohup ./target/release/qo > "$LOG" 2>&1 &
  PID=$!
  PIDS+=($PID)
  echo "Server $NODE (port $PORT, node-id node-$NODE) PID: $PID  log: $LOG"
done

# Wait for health endpoints
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  ALL_OK=1
  for port in "${PORTS[@]}"; do
    H=$(curl -s -m 1 "http://localhost:$port/api/health" || true)
    if [ -z "$H" ]; then
      ALL_OK=0
    fi
  done
  if [ "$ALL_OK" = "1" ]; then
    break
  fi
done

echo
echo "=== Health ==="
for i in 0 1 2; do
  PORT=${PORTS[$i]}
  NODE=${NODES[$i]}
  H=$(curl -s -m 1 "http://localhost:$port/api/health" || echo "DOWN")
  echo "node-$NODE (:$PORT) → $H"
done

echo
echo "PIDS: ${PIDS[*]}"
echo "To stop: kill ${PIDS[*]}"
