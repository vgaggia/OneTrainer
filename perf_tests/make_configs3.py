#!/usr/bin/env python
"""Generate the round-3 matrix: batch ladder, checkpointing-OFF probe, and the quality pair.

Derived from configs2/I_freeze7_adafactor.json (freeze blocks 0-20, QWT, compile off, TREAD off).

Two kinds of run:
  probe   - memory/OOM only. Sampling and validation OFF so it is fast and measures nothing else.
  quality - sampling + validation ON, epochs chosen so the run gets ~100 optimizer steps.

Why epochs is computed rather than fixed at 1: at batch 16 one epoch over 395 images is only 25
steps, which is far too few to separate two optimizers. 'Quality per GPU-hour' still wants a
comparable number of updates on each side.

    python perf_tests/make_configs3.py
"""
import copy
import json
from pathlib import Path

SRC = Path("perf_tests/configs2/I_freeze7_adafactor.json")
DST = Path("perf_tests/configs3")
CONCEPTS = "training_concepts/_perftest_akko_split.json"

TRAIN_IMAGES = 395
TARGET_STEPS = 100          # for quality runs
VAL_IMAGES = 64

FREEZE7 = ",".join(f"transformer_blocks.{i}." for i in range(21, 28))


def steps_per_epoch(batch: int) -> int:
    return max(1, TRAIN_IMAGES // batch)


def build(name, batch, optimizer, *, kind, checkpointing="ON"):
    d = json.loads(SRC.read_text(encoding="utf-8"))
    d["__version"] = 10                      # let upstream migration_10 fan out the offload keys
    d["concept_file_name"] = CONCEPTS
    d["concepts"] = None
    d["layer_filter"] = FREEZE7
    d["batch_size"] = batch
    d["optimizer"]["optimizer"] = optimizer
    d["optimizer"]["fused_back_pass"] = True
    d["quantized_weight_training"] = True
    d["compile"] = False                     # hard guard: QWT forbids compile
    d["tread_enabled"] = False
    d["gradient_checkpointing"] = checkpointing
    d["layer_offload_fraction"] = 0.0
    d["enable_activation_offloading"] = False
    d["backup_before_save"] = False
    d["rolling_backup"] = False
    d["workspace_dir"] = f"G:/Auto/OneTrainer/perf_tests/workspace3/{name}"
    d["cache_dir"] = "F:/Krea2 Model/perftest-cache/shared3"
    d["output_model_destination"] = f"G:/Auto/OneTrainer/perf_tests/out3/{name}.safetensors"
    d["tensorboard"] = False

    spe = steps_per_epoch(batch)
    if kind == "probe":
        # pure memory probe: peak VRAM is reached in the first handful of steps
        d["epochs"] = 1
        d["validation"] = False
        d["sample_after"] = 10 ** 6          # effectively never
        d["sample_after_unit"] = "STEP"
    else:
        d["epochs"] = max(1, -(-TARGET_STEPS // spe))     # ceil
        total = d["epochs"] * spe
        d["validation"] = True
        d["validate_after_unit"] = "STEP"
        # ~5 validation points; EPOCH unit with few epochs repeats the one-data-point bug
        d["validate_after"] = max(1, total // 5)
        d["sample_after_unit"] = "STEP"
        d["sample_after"] = max(1, total // 2)            # step 0 + midpoint + end-ish
    return d, spe


PROBES = [
    ("P08_came_bs8", 8, "CAME", "ON"),
    ("P16_came_bs16", 16, "CAME", "ON"),
    ("P32_came_bs32", 32, "CAME", "ON"),
    ("P04_came_bs4_nockpt", 4, "CAME", "OFF"),
]

DST.mkdir(exist_ok=True)
rows = []
for name, batch, opt, ckpt in PROBES:
    d, spe = build(name, batch, opt, kind="probe", checkpointing=ckpt)
    (DST / f"{name}.json").write_text(json.dumps(d, indent=4), encoding="utf-8")
    rows.append((name, batch, opt, ckpt, d["epochs"], spe, d["epochs"] * spe, "-", "-"))

# quality pair batch is filled in after the ladder; default 8 so the files always exist
for name, opt in (("Q_came", "CAME"), ("Q_adafactor", "ADAFACTOR")):
    d, spe = build(f"{name}_bs8", 8, opt, kind="quality")
    (DST / f"{name}_bs8.json").write_text(json.dumps(d, indent=4), encoding="utf-8")
    rows.append((f"{name}_bs8", 8, opt, "ON", d["epochs"], spe, d["epochs"] * spe,
                 d["validate_after"], d["sample_after"]))

hdr = f"{'run':<22}{'bs':>4}{'opt':>11}{'ckpt':>6}{'ep':>4}{'spe':>5}{'steps':>7}{'val@':>6}{'smp@':>6}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:<22}{r[1]:>4}{r[2]:>11}{r[3]:>6}{r[4]:>4}{r[5]:>5}{r[6]:>7}{str(r[7]):>6}{str(r[8]):>6}")
print(f"\nvalidation passes cost {VAL_IMAGES}/batch steps each")
