#!/usr/bin/env bash
# Round-3 runner: peak VRAM + OOM verdict per config, with the traps from round 2 handled.
#
#   bash perf_tests/run_probes3.sh P08_came_bs8 P16_came_bs16 P32_came_bs32 P04_came_bs4_nockpt
#
# Per run: kill orphans, assert a clean GPU, setsid a 1 Hz poller, run, read the peak in its own
# ssh call, then clear the disk. Every one of those steps is here because skipping it produced a
# wrong number last time.
set -u
cd "$(dirname "$0")/.."
mkdir -p perf_tests/logs3
POD=$(bash perf_tests/pod_ssh.sh) || { echo "no pod"; exit 1; }
echo "pod: $POD"
RUNS=${*:-"P08_came_bs8 P16_came_bs16 P32_came_bs32 P04_came_bs4_nockpt"}

for name in $RUNS; do
  cfg="perf_tests/configs3_cloud/${name}.json"
  log="perf_tests/logs3/${name}.log"
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  echo "=== $name ($(date +%H:%M:%S)) ==="

  # orphaned remote trainers survive a killed local client and hold 17+ GiB, which turns the next
  # run into a meaningless OOM
  ssh $POD "pkill -9 -f train_remote 2>/dev/null; pkill -9 -f 'nvidia-smi.*-l 1' 2>/dev/null; \
      rm -rf /workspace/remote/Auto/OneTrainer/perf_tests/out3/*.safetensors \
             /workspace/remote/Auto/OneTrainer/perf_tests/workspace3/*/backup; \
      rm -f /workspace/vram.log" > /dev/null 2>&1
  sleep 6
  base=$(ssh $POD "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null | tr -d '\r')
  if [ "${base:-9999}" -gt 500 ]; then
    echo "  ABORT: GPU dirty (${base} MiB) - number would be contaminated"; continue
  fi

  ssh $POD "setsid nohup nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 \
      > /workspace/vram.log 2>/dev/null < /dev/null &" > /dev/null 2>&1
  sleep 3

  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/train.py \
      --config-path="$cfg" --secrets-path=perf_tests/_cloud_secrets.json > "$log" 2>&1
  status=$?

  peak=$(ssh $POD "sort -n /workspace/vram.log | tail -1" 2>/dev/null | tr -d '\r')
  ssh $POD "pkill -9 -f 'nvidia-smi.*-l 1' 2>/dev/null" > /dev/null 2>&1

  if grep -qiE "out of memory|CUDA out of memory" "$log"; then
    verdict="OOM"
  elif grep -qi "No space left" "$log"; then
    verdict="ok (save hit disk-full, post-training)"
  elif [ $status -ne 0 ]; then
    verdict="FAILED: $(grep -oiE '^[A-Za-z.]*(Error|Exception): .*' "$log" | head -1 | cut -c1-70)"
  else
    verdict="ok"
  fi
  gib=$([ -n "$peak" ] && awk "BEGIN{printf \"%.1f\", ${peak}/1024}" || echo "?")
  printf "  peak %s MiB (%s GiB)   %s\n" "${peak:-?}" "$gib" "$verdict"
done
echo
echo "REMEMBER: terminate the pod and confirm 0 running"
