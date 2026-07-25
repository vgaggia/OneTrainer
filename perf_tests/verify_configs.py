#!/usr/bin/env python
"""Load every perf_tests/configs2/*.json the way train.py does and print what actually takes effect.

Exists because upstream #1476 moved gradient_checkpointing / layer_offload_fraction /
enable_activation_offloading into per-component config via __migration_10. Our branch had its own
unrelated __migration_10, so both stamp __version 11 — a config stamped 11 skips upstream's fan-out
and silently falls back to defaults, collapsing A/B/C into the same run. Restamping to 10 lets
upstream's migration translate them. This script proves the translation happened.

    python perf_tests/verify_configs.py           # report
    python perf_tests/verify_configs.py --fix     # restamp __version 11 -> 10, then report
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from util.import_util import script_imports  # noqa: E402

script_imports()

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

CONFIGS = sorted((Path(__file__).parent / "configs2").glob("*.json"))


def restamp():
    for p in CONFIGS:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("__version") == 11 and "gradient_checkpointing" in d:
            d["__version"] = 10
            p.write_text(json.dumps(d, indent=4), encoding="utf-8")
            print(f"restamped {p.name} -> __version 10")


def load(path: Path) -> TrainConfig:
    c = TrainConfig.default_values()
    with open(path) as f:
        c.from_dict(json.load(f), migrate=True)
    return c


def report():
    rows = []
    for p in CONFIGS:
        c = load(p)
        t = c.transformer
        trainable = [x for x in c.layer_filter.split(",") if x.strip()]
        blocks = sorted({int(x.split(".")[1]) for x in trainable}) if trainable else []
        rows.append((
            p.stem,
            f"{t.gradient_checkpointing}/{t.offload_fraction}/{t.activation_offloading}",
            c.optimizer.optimizer.name if hasattr(c.optimizer.optimizer, "name") else str(c.optimizer.optimizer),
            f"{c.learning_rate:g}",
            str(c.batch_size),
            str(c.optimizer.fused_back_pass),
            f"{blocks[0]}-{blocks[-1]}" if blocks else "all 28",
            str(c.quantized_weight_training),
            str(c.compile),
        ))

    hdr = ("run", "ckpt/frac/actoff", "optimizer", "lr", "bs", "fused", "trained", "qwt", "compile")
    w = [max(len(str(r[i])) for r in (*rows, hdr)) for i in range(len(hdr))]
    line = "  ".join(h.ljust(w[i]) for i, h in enumerate(hdr))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))

    # the whole point: A/B/C must differ in the per-component offload triple
    triples = {r[0]: r[1] for r in rows}
    a, b, c_ = triples.get("A_qwt_baseline"), triples.get("B_qwt_fast"), triples.get("C_qwt_nockpt")
    print()
    if a == b == c_:
        print(f"FAIL: A/B/C all resolve to {a} - the migration did not run, runs are not distinct")
        return 1
    print(f"ok: A={a}  B={b}  C={c_} are distinct")
    if any(r[7] != "True" for r in rows):
        print("FAIL: some run has quantized_weight_training=False (would train norms/biases only)")
        return 1
    print("ok: quantized_weight_training=True on every run")

    # TREAD inflates s/it by ~1.68x and damages fine-tuning; every run must have it off or the
    # whole matrix measures the wrong thing
    tread_on = [p.stem for p in CONFIGS if load(p).tread_enabled]
    if tread_on:
        print(f"FAIL: tread_enabled=True on {tread_on} - speed numbers would be inflated and quality damaged")
        return 1
    print("ok: tread_enabled=False on every run")
    return 0


if __name__ == "__main__":
    if "--fix" in sys.argv:
        restamp()
        print()
    sys.exit(report())
