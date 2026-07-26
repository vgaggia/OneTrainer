#!/usr/bin/env python
"""Pull tensorboard event files off the pod and print validation loss per run.

Validation loss is written ONLY via tensorboard.add_scalar (GenericTrainer.__validate) - it is
never printed to the log - and cloud.download_tensorboard is False, so without this the numbers
die with the pod. The writer runs even when `tensorboard: false`, which is why the data exists.

    python perf_tests/fetch_validation.py --fetch     # scp the event files down first
    python perf_tests/fetch_validation.py             # parse whatever is local
"""
import glob
import os
import subprocess
import sys

LOCAL = "perf_tests/tb_events"
REMOTE = "/workspace/remote/Auto/OneTrainer/perf_tests/workspace3"

if "--fetch" in sys.argv:
    ssh_args = subprocess.run(["bash", "perf_tests/pod_ssh.sh"], capture_output=True, text=True).stdout.strip()
    if not ssh_args:
        sys.exit("no pod")
    # ssh_args looks like: -o ... -p PORT root@IP
    parts = ssh_args.split()
    port = parts[parts.index("-p") + 1]
    host = parts[-1]
    os.makedirs(LOCAL, exist_ok=True)
    subprocess.run([
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P", port, "-r",
        f"{host}:{REMOTE}", LOCAL,
    ], check=False)
    print(f"fetched into {LOCAL}")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not installed in this venv")

files = sorted(glob.glob(f"{LOCAL}/**/events.out.tfevents.*", recursive=True))
if not files:
    sys.exit(f"no event files under {LOCAL} - run with --fetch first")

runs = {}
for f in files:
    # .../workspace3/<run>/tensorboard/<timestamp>/events...
    parts = f.replace("\\", "/").split("/")
    run = parts[parts.index("workspace3") + 1] if "workspace3" in parts else os.path.basename(os.path.dirname(f))
    acc = EventAccumulator(f, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    val = [t for t in tags if "validation" in t]
    if not val:
        continue
    # keep the event file with the most validation points for a run (later re-runs overwrite)
    best = runs.get(run)
    pts = {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in val}
    n = max((len(v) for v in pts.values()), default=0)
    if not best or n > best[0]:
        runs[run] = (n, pts)

if not runs:
    print("no validation scalars found (runs with validation:false have none)")
    print("available tags in the newest file:")
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0}); acc.Reload()
    for t in acc.Tags().get("scalars", [])[:12]:
        print("   ", t)
    sys.exit(0)

for run, (_, pts) in sorted(runs.items()):
    print(f"\n=== {run} ===")
    for tag, series in sorted(pts.items()):
        vals = "  ".join(f"{s}:{v:.5f}" for s, v in series)
        print(f"  {tag}")
        print(f"    {vals}")
        if len(series) >= 2:
            first, last = series[0][1], series[-1][1]
            print(f"    first {first:.5f} -> last {last:.5f}   delta {last - first:+.5f}")
