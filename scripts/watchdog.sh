#!/bin/bash
# QLANG Watchdog — Keeps all services alive, protects GPU 0, auto-restart on crash.

cd /home/mirkulix/AI/neoqlang/qlang
export LIBTORCH_USE_PYTORCH=1
TORCH_LIB=$(python3 -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>/dev/null)
CUDA_LIBS="/home/mirkulix/.local/lib/python3.14/site-packages/nvidia/cu13/lib"
LOG_DIR=/tmp/qlang-watchdog
mkdir -p "$LOG_DIR"

LOG() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/watchdog.log"; }

check_gpu0_safe() {
  # GPU 0 has display — abort training if > 7GB used
  GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  if [ "$GPU0_USED" -gt 7000 ] 2>/dev/null; then
    LOG "DANGER: GPU 0 at ${GPU0_USED} MiB - killing training!"
    pkill -f "cuda_if_available.*0" 2>/dev/null
    return 1
  fi
  return 0
}

ensure_qo_server() {
  if ! curl -s http://localhost:4646/api/health >/dev/null 2>&1; then
    LOG "QO server offline — restarting..."
    lsof -ti:4646 2>/dev/null | xargs -r kill 2>/dev/null
    sleep 1
    LD_PRELOAD="$TORCH_LIB/libtorch_cuda.so" \
      LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_LIBS" \
      QO_PORT=4646 \
      nohup ./target/release/qo > "$LOG_DIR/qo-server.log" 2>&1 &
    LOG "QO server restarted PID $!"
    sleep 3
  fi
}

ensure_demo_server() {
  if ! curl -s http://localhost:4747/ >/dev/null 2>&1; then
    lsof -ti:4747 2>/dev/null | xargs -r kill 2>/dev/null
    sleep 1
    nohup node scripts/serve-demo.mjs > "$LOG_DIR/demo-server.log" 2>&1 &
    LOG "Demo server restarted PID $!"
  fi
}

ensure_dashboards() {
  pgrep -f "agent-dashboard.mjs" >/dev/null || {
    nohup node scripts/agent-dashboard.mjs > "$LOG_DIR/agent-dashboard.log" 2>&1 &
    LOG "Agent dashboard started PID $!"
  }
  pgrep -f "claude-agents-dashboard.mjs" >/dev/null || {
    nohup node scripts/claude-agents-dashboard.mjs > "$LOG_DIR/claude-dashboard.log" 2>&1 &
    LOG "Claude dashboard started PID $!"
  }
}

LOG "Watchdog started — checking every 30s"
while true; do
  check_gpu0_safe
  ensure_qo_server
  ensure_demo_server
  ensure_dashboards
  sleep 30
done
