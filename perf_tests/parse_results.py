"""Summarize perf-test runs: steady-state step time, peak VRAM, loss curve.

Run from repo root:  venv\\Scripts\\python.exe perf_tests\\parse_results.py
Reads perf_tests/logs/*.log, *_vram.csv and tensorboard events in perf_tests/workspace/<run>.
Writes perf_tests/results/summary.md and per-run .json files.
"""

import json
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PT = REPO / "perf_tests"
RESULTS = PT / "results"

# matches tqdm rate stamps like "1.35s/it" or "2.10it/s"
RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(s/it|it/s)")
# matches "9/21" style progress counters
PROG_RE = re.compile(r"(\d+)/(\d+)")
LOSS_RE = re.compile(r"loss[=:]\s*([0-9.]+(?:e[+-]?\d+)?)", re.IGNORECASE)


def read_text_any(p: Path) -> str:
    raw = p.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_log(log_path: Path) -> dict:
    text = read_text_any(log_path)
    # tqdm uses \r; split on both
    lines = re.split(r"[\r\n]+", text)
    sec_per_it = []
    losses = []
    for line in lines:
        # only training-step progress bars; ignore "caching:", "sampling:" etc.
        if not line.lstrip().startswith("step"):
            continue
        m = RATE_RE.search(line)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            if val > 0:
                sec_per_it.append(val if unit == "s/it" else 1.0 / val)
        ml = LOSS_RE.search(line)
        if ml:
            try:
                losses.append(float(ml.group(1)))
            except ValueError:
                pass
    out = {"n_rate_samples": len(sec_per_it), "n_loss_samples": len(losses)}
    if sec_per_it:
        # steady state: drop first 30% (cache, compile warmup, cudnn autotune)
        tail = sec_per_it[int(len(sec_per_it) * 0.3):]
        out["median_s_per_it"] = round(statistics.median(tail), 4)
        out["p10_s_per_it"] = round(sorted(tail)[int(len(tail) * 0.1)], 4)
        out["p90_s_per_it"] = round(sorted(tail)[int(len(tail) * 0.9)], 4)
    if losses:
        k = max(1, len(losses) // 5)
        out["mean_loss_last_20pct"] = round(sum(losses[-k:]) / k, 5)
    # surface crashes
    if "Traceback" in text:
        tb = text[text.rfind("Traceback"):]
        out["error_tail"] = tb[:1500]
    return out


def parse_vram(csv_path: Path) -> dict:
    peak = 0
    for line in read_text_any(csv_path).splitlines():
        m = re.search(r"(\d+)\s*MiB", line)
        if m:
            peak = max(peak, int(m.group(1)))
    return {"peak_vram_mib": peak}


def parse_tensorboard(run_dir: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    events = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not events:
        return {}
    acc = EventAccumulator(str(events[-1].parent))
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        if "loss" in tag.lower() or "smooth" in tag.lower():
            vals = [(e.step, e.value) for e in acc.Scalars(tag)]
            if vals:
                out[f"tb::{tag}"] = {
                    "n": len(vals),
                    "first": round(vals[0][1], 5),
                    "last": round(vals[-1][1], 5),
                    "mean_last_20pct": round(
                        sum(v for _, v in vals[-max(1, len(vals) // 5):])
                        / max(1, len(vals) // 5), 5),
                }
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for log in sorted((PT / "logs").glob("*.log")):
        name = log.stem
        rec = {"run": name}
        rec.update(parse_log(log))
        vram = log.with_name(f"{name}_vram.csv")
        if vram.exists():
            rec.update(parse_vram(vram))
        ws = PT / "workspace" / name
        if ws.exists():
            rec.update(parse_tensorboard(ws))
        rows.append(rec)
        (RESULTS / f"{name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    cols = ["run", "median_s_per_it", "p10_s_per_it", "p90_s_per_it",
            "peak_vram_mib", "mean_loss_last_20pct", "n_rate_samples"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |")
        if "error_tail" in r:
            md.append(f"\n**{r['run']} CRASHED:**\n```\n{r['error_tail'][:600]}\n```\n")
    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
