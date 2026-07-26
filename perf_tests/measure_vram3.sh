#!/usr/bin/env bash
# Peak VRAM per config, measured reliably.
#
# Three things bit the earlier attempts and are all handled here:
#  1. a plain `nohup ... &` poller inside ssh dies with the ssh session -> setsid it
#  2. killing a local client leaves the REMOTE trainer alive, holding 17+ GiB; the next run then
#     "OOMs" for a reason that has nothing to do with its config -> assert a clean GPU first
#  3. nvidia-smi memory.used is whole-GPU, so any orphan silently inflates the number
# Peak is read in its own ssh call BEFORE the poller is killed; combining them lost the output.
set -u
cd "$(dirname "$0")/.."
POD_SSH="-o StrictHostKeyChecking=no -o BatchMode=yes -p 42178 root@80.15.7.37"
RUNS=${*:-"B_qwt_fast D_qwt_freeze14 G_freeze7_came"}

for name in $RUNS; do
  cfg="perf_tests/configs2_cloud/${name}.json"
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  echo "=== $name ($(date +%H:%M:%S)) ==="

  ssh $POD_SSH "pkill -9 -f train_remote 2>/dev/null; pkill -9 -f 'nvidia-smi.*-l 1' 2>/dev/null; \
      rm -rf /workspace/remote/Auto/OneTrainer/perf_tests/out2/*.safetensors \
             /workspace/remote/Auto/OneTrainer/perf_tests/workspace2/*/backup; \
      rm -f /workspace/vram.log" > /dev/null 2>&1
  sleep 6
  base=$(ssh $POD_SSH "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d '\r')
  if [ "${base:-9999}" -gt 500 ]; then
    echo "  ABORT: GPU not clean (${base} MiB in use) - measurement would be contaminated"
    continue
  fi
  echo "  GPU clean (${base} MiB)"

  ssh $POD_SSH "setsid nohup nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
      > /workspace/vram.log 2>/dev/null < /dev/null &" > /dev/null 2>&1
  sleep 3

  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/train.py \
      --config-path="$cfg" --secrets-path=perf_tests/_cloud_secrets.json \
      > "perf_tests/logs2/${name}.vram.log" 2>&1

  peak=$(ssh $POD_SSH "sort -n /workspace/vram.log | tail -1" 2>/dev/null | tr -d '\r')
  ssh $POD_SSH "pkill -9 -f 'nvidia-smi.*-l 1' 2>/dev/null" > /dev/null 2>&1
  if [ -n "$peak" ]; then
    echo "  peak VRAM: ${peak} MiB ($(awk "BEGIN{printf \"%.1f\", ${peak}/1024}") GiB)"
  else
    echo "  peak VRAM: UNREAD"
  fi
done
