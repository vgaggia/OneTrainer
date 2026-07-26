#!/usr/bin/env python
"""Parse the cloud matrix logs into one table.

The naive parser in run_matrix_cloud.sh matches every 's/it' in the log, which also catches the
31-step sampling loop and drags the median around. Training lines are the ones carrying 'loss=',
so key on that. Steady state drops the first 30% (compile/cuDNN/allocator warmup).
"""
import glob
import os
import re
import statistics as st

# tqdm flips to 'it/s' once a step drops under a second, so both units must be caught or fast
# runs get silently biased toward their slow samples
STEP = re.compile(r"(\d+)/(\d+)\s*\[[^\]]*?(\d+\.?\d*)(s/it|it/s)[^\]]*loss=")
VRAM = re.compile(r"(?:peak|max).{0,24}?(\d+\.?\d*)\s*(GiB|GB|MiB|MB)", re.I)
ORDER = ["A_qwt_baseline", "B_qwt_fast", "C_qwt_nockpt", "D_qwt_freeze14", "E_qwt_muon",
         "F_freeze14_came", "G_freeze7_came", "H_freeze7_muon", "I_freeze7_adafactor"]


def parse(path):
    rates, last_step, total, err, vram = [], 0, 0, "", None
    for line in open(path, encoding="utf-8", errors="replace"):
        for m in STEP.finditer(line):
            last_step, total = int(m.group(1)), int(m.group(2))
            v = float(m.group(3))
            if m.group(4) == "it/s":
                if v <= 0:
                    continue
                v = 1.0 / v
            rates.append(v)
        if m := VRAM.search(line):
            v = float(m.group(1))
            vram = v / 1024 if m.group(2).lower() in ("mib", "mb") else v
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            err = "OOM"
        elif not err and (m2 := re.match(r"^([A-Za-z_.]*(?:Error|Exception)): (.+)", line.strip())):
            if "No space left" in line:
                err = "save: disk full"
            else:
                err = f"{m2.group(1)}: {m2.group(2)[:44]}"
    tail = rates[int(len(rates) * 0.3):] if rates else []
    return {
        "steps": f"{last_step}/{total}" if total else "-",
        "median": st.median(tail) if tail else None,
        "min": min(tail) if tail else None,
        "n": len(tail),
        "vram": vram,
        "err": err,
    }


rows = []
for name in ORDER:
    p = f"perf_tests/logs2/{name}.log"
    if not os.path.exists(p):
        continue
    r = parse(p)
    r["name"] = name
    r["samples"] = len(glob.glob(f"perf_tests/workspace2/{name}/samples/**/*.jpg", recursive=True))
    rows.append(r)

hdr = f"{'run':<20} {'steps':>9} {'median s/it':>12} {'min':>7} {'n':>5} {'img/s':>7} {'samples':>8}  note"
print(hdr)
print("-" * len(hdr))
BATCH = {"G_freeze7_came": 4, "I_freeze7_adafactor": 4}
for r in rows:
    med = f"{r['median']:.3f}" if r["median"] else "-"
    mn = f"{r['min']:.3f}" if r["min"] else "-"
    ips = f"{BATCH.get(r['name'], 2) / r['median']:.2f}" if r["median"] else "-"
    print(f"{r['name']:<20} {r['steps']:>9} {med:>12} {mn:>7} {r['n']:>5} {ips:>7} {r['samples']:>8}  {r['err']}")
