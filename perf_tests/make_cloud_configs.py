#!/usr/bin/env python
"""Derive cloud variants of perf_tests/configs2/*.json into perf_tests/configs2_cloud/.

CloudTrainer already remaps debug_dir/workspace_dir/cache_dir/output_model_destination and the
concept path onto /workspace/remote/..., so those are left alone here. Only the cloud block, the
quantization cache (which CloudTrainer does *not* remap) and the pod id need touching.

    python perf_tests/make_cloud_configs.py <git-branch> [pod-id] [src-dir]
"""
import json
import sys
from pathlib import Path

BRANCH = sys.argv[1] if len(sys.argv) > 1 else "pod/krea2-matrix"
POD_ID = sys.argv[2] if len(sys.argv) > 2 else ""  # empty on the first run: creates the pod
SRC_NAME = sys.argv[3] if len(sys.argv) > 3 else "configs2"

SRC = Path(__file__).parent / SRC_NAME
DST = Path(__file__).parent / f"{SRC_NAME}_cloud"

CLOUD = {
    "__version": 0,
    "enabled": True,
    "type": "RUNPOD",
    "file_sync": "NATIVE_SCP",
    "create": not POD_ID,
    "name": "OneTrainer-perf2",
    "tensorboard_tunnel": False,
    "sub_type": "COMMUNITY",
    "gpu_type": "NVIDIA GeForce RTX 5090",
    "gpu_count": 1,
    "cuda_version": "13.0",
    "volume_size": 100,
    "network_volume_id": "",   # deliberately none: 459 imgs cache in minutes
    "data_center_id": "",      # so it can land wherever there is 5090 stock
    # the 58 GB model pull dominates setup; floor host download speed so we do not land on
    # a slow server. Do not lower this to widen the pool: slow hosts cost more than they save.
    "min_download": 1000,
    "remote_dir": "/workspace",
    "huggingface_cache_dir": "/workspace/huggingface_cache",
    "onetrainer_dir": "/workspace/OneTrainer",
    "install_cmd": f"git clone --branch {BRANCH} https://github.com/vgaggia/OneTrainer.git",
    "install_onetrainer": True,
    "update_onetrainer": False,
    "detach_trainer": True,
    "download_samples": True,
    "download_output_model": False,
    "download_saves": False,
    "download_backups": False,
    "download_tensorboard": False,
    "delete_workspace": False,
    "on_finish": "NONE",   # keep the pod for the next run in the matrix
    "on_error": "NONE",
    "on_detached_finish": "NONE",
    "on_detached_error": "NONE",
}

DST.mkdir(exist_ok=True)
for src in sorted(SRC.glob("*.json")):
    cfg = json.loads(src.read_text(encoding="utf-8"))
    cfg["cloud"] = dict(CLOUD, run_id=src.stem)
    # not in CloudTrainer's remap list, so a Windows path would land verbatim on the pod
    cfg["quantization"]["cache_dir"] = f"/workspace/perftest-quant/{src.stem}"
    cfg["tensorboard"] = False
    (DST / src.name).write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    print(f"wrote {DST / src.name}")

if __name__ == "__main__" and "--check" in sys.argv:
    c = json.loads((DST / "A_qwt_baseline.json").read_text(encoding="utf-8"))
    assert c["cloud"]["enabled"] and c["cloud"]["gpu_count"] == 1
    assert not c["cloud"]["network_volume_id"] and not c["cloud"]["data_center_id"]
    assert BRANCH in c["cloud"]["install_cmd"]
    assert not c["quantization"]["cache_dir"].startswith("F:")
    print("check ok")
