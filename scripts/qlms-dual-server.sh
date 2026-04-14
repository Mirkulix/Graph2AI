#!/bin/bash
# QLMS dual-server launcher — starts 2 QO servers on ports 4646 + 4747
# for AI-to-AI round-trip demos.
#
# Usage:  ./scripts/qlms-dual-server.sh
#         curl -X POST http://localhost:4646/api/qlms/send-model \
#              -H 'content-type: application/json' \
#              -d '{"target_host":"localhost:4747"}'

set -u

cd /home/mirkulix/AI/neoqlang/qlang

TORCH_LIB=$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>/dev/null || echo "")
CUDA_LIBS="/home/mirkulix/.local/lib/python3.14/site-packages/nvidia/cu13/lib"

# Kill any old instances on these ports
for port in 4646 4747; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "Killing existing process(es) on :$port  ($pids)"
    kill $pids 2>/dev/null || true
  fi
done
sleep 1

mkdir -p data/server-a data/server-b

# Start server A on 4646
LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so" \
LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_LIBS}" \
QO_PORT=4646 QO_DATA_DIR=data/server-a \
  nohup ./target/release/qo > /tmp/qo-a.log 2>&1 &
PID_A=$!
echo "Server A (4646) PID: $PID_A"

# Start server B on 4747
LD_PRELOAD="${TORCH_LIB}/libtorch_cuda.so" \
LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_LIBS}" \
QO_PORT=4747 QO_DATA_DIR=data/server-b \
  nohup ./target/release/qo > /tmp/qo-b.log 2>&1 &
PID_B=$!
echo "Server B (4747) PID: $PID_B"

# Wait for health endpoints
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  HA=$(curl -s -m 1 http://localhost:4646/api/health || true)
  HB=$(curl -s -m 1 http://localhost:4747/api/health || true)
  if [ -n "$HA" ] && [ -n "$HB" ]; then
    break
  fi
done

echo "A: ${HA:-DOWN}"
echo "B: ${HB:-DOWN}"
