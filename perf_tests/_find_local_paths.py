"""Report config fields holding local filesystem paths.

CloudTrainer uploads any path that exists locally (adjust(..., if_exists=True)), so a stale
transformer override pointing at a 12 GB local checkpoint gets pushed over the home uplink
instead of being downloaded on the pod.
"""
import json
import os
import re
import sys

WINPATH = re.compile(r"^[A-Za-z]:[\\/]")

path = sys.argv[1] if len(sys.argv) > 1 else "perf_tests/configs2_cloud/A_qwt_baseline.json"


def walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]")
    elif isinstance(obj, str) and obj and (WINPATH.match(obj) or obj.startswith("/")):
        exists = os.path.exists(obj)
        size = f"{os.path.getsize(obj) / 1e9:.1f} GB" if exists and os.path.isfile(obj) else ""
        flag = "  <-- WILL BE UPLOADED" if exists else ""
        print(f"  {prefix} = {obj}   exists={exists} {size}{flag}")


walk(json.load(open(path)))
