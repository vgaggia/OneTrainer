#!/usr/bin/env bash
# Full fine-tune perf matrix at 512 on the Akko subset (459 imgs).
# First run builds the shared cache; the rest reuse it.
#   bash perf_tests/run_matrix2.sh            # all
#   bash perf_tests/run_matrix2.sh B_qwt_fast # one
set -u
cd "$(dirname "$0")/.."
mkdir -p perf_tests/logs2 perf_tests/out2

RUNS=${*:-"A_qwt_baseline B_qwt_fast C_qwt_nockpt D_qwt_freeze14 E_qwt_muon"}

for name in $RUNS; do
  cfg="perf_tests/configs2/${name}.json"
  log="perf_tests/logs2/${name}.log"
  [ -f "$cfg" ] || { echo "missing $cfg"; continue; }
  echo "=== $name ==="
  # PYTHONIOENCODING: tqdm block chars vs the cp1252 console kill the run otherwise
  PYTHONIOENCODING=utf-8 venv/Scripts/python.exe scripts/train.py --config-path="$cfg" > "$log" 2>&1
  status=$?
  if grep -qi "out of memory" "$log"; then
    echo "  OOM"
  elif [ $status -ne 0 ]; then
    echo "  FAILED (exit $status) - last error:"
    grep -iE "error|Traceback" "$log" | tail -2 | sed 's/^/    /'
  else
    echo "  ok"
  fi
  # steady-state s/it: drop the first 30% of samples (compile warmup)
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
    print(f"    median {st.median(tail):.3f} s/it   (n={len(tail)})")
else:
    print("    no rate samples")
PY
done
echo
echo "logs: perf_tests/logs2/  samples: perf_tests/workspace2/<run>/samples/"
