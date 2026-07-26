#!/usr/bin/env bash
# Peak VRAM per config. PeakMemoryRecorder never emitted into the run logs, so poll nvidia-smi
# on the pod at 1 Hz and take the max.
#
# The poller must be setsid'd: a plain `nohup ... &` inside an ssh command dies when the ssh
# session closes, which silently produced an empty log on the first attempt.
set -u
cd "$(dirname "$0")/.."
POD_SSH="-o StrictHostKeyChecking=no -o BatchMode=yes -p 42178 root@80.15.7.37"
RUNS=${*:-"D_qwt_freeze14 G_freeze7_came H_freeze7_muon"}

for name in $RUNS; do
  cfg="perf_tests/configs2_cloud/${name}.json"
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  echo "=== $name ($(date +%H:%M:%S)) ==="
  ssh $POD_SSH "pkill -f 'nvidia-smi.*-l 1' 2>/dev/null; rm -f /workspace/vram.log; \
      setsid nohup nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
      > /workspace/vram.log 2>/dev/null < /dev/null &" > /dev/null 2>&1
  sleep 3
  ssh $POD_SSH "test -s /workspace/vram.log && echo '  poller ok' || echo '  POLLER DEAD'" 2>/dev/null

  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/train.py \
      --config-path="$cfg" --secrets-path=perf_tests/_cloud_secrets.json \
      > "perf_tests/logs2/${name}.vram.log" 2>&1

  peak=$(ssh $POD_SSH "pkill -f 'nvidia-smi.*-l 1' 2>/dev/null; sort -n /workspace/vram.log | tail -1" 2>/dev/null | tr -d '\r')
  echo "  peak VRAM: ${peak:-?} MiB"
  ssh $POD_SSH "rm -rf /workspace/remote/Auto/OneTrainer/perf_tests/out2/*.safetensors \
      /workspace/remote/Auto/OneTrainer/perf_tests/workspace2/*/backup" > /dev/null 2>&1
done
