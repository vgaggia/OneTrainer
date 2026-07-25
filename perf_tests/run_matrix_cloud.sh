#!/usr/bin/env bash
# Run the cloud perf matrix sequentially on one already-created pod.
# The pod id lives in perf_tests/_cloud_secrets.json; every config has create=false, so
# nothing here creates or deletes a pod - termination is deliberate and separate.
#
#   bash perf_tests/run_matrix_cloud.sh                    # B..I
#   bash perf_tests/run_matrix_cloud.sh C_qwt_nockpt       # just one
set -u
cd "$(dirname "$0")/.."
mkdir -p perf_tests/logs2

RUNS=${*:-"B_qwt_fast C_qwt_nockpt D_qwt_freeze14 E_qwt_muon F_freeze14_came G_freeze7_came H_freeze7_muon I_freeze7_adafactor"}

for name in $RUNS; do
  cfg="perf_tests/configs2_cloud/${name}.json"
  log="perf_tests/logs2/${name}.log"
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  echo "=== $name  ($(date +%H:%M:%S)) ==="
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/train.py \
      --config-path="$cfg" --secrets-path=perf_tests/_cloud_secrets.json > "$log" 2>&1
  status=$?
  if grep -qi "out of memory\|CUDA out of memory" "$log"; then
    echo "  OOM"
  elif [ $status -ne 0 ]; then
    echo "  FAILED (exit $status):"
    grep -iE "^[A-Za-z.]*(Error|Exception):" "$log" | tail -2 | sed 's/^/    /'
  else
    echo "  ok"
  fi
  # steady-state s/it: drop the first 30% (compile + cudnn warmup dominate early steps)
  python - "$log" <<'PY'
import re,sys,statistics as st
rates=[]
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    for m in re.finditer(r'(\d+\.?\d*)s/it', line): rates.append(float(m.group(1)))
    for m in re.finditer(r'(\d+\.?\d*)it/s', line):
        v=float(m.group(1))
        if v>0: rates.append(1/v)
if rates:
    tail=rates[int(len(rates)*0.3):]
    print(f"    median {st.median(tail):.3f} s/it  (n={len(tail)}, min {min(tail):.3f})")
else:
    print("    no rate samples")
PY
done
echo
echo "logs: perf_tests/logs2/   REMEMBER: terminate the pod when done"
