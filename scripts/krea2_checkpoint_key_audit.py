"""Inspect Krea 2 Raw/Turbo/full-FT/LoRA safetensors without loading tensor data.

Examples:
    python scripts/krea2_checkpoint_key_audit.py Krea-2-Raw.safetensors Krea-2-Turbo.safetensors
    python scripts/krea2_checkpoint_key_audit.py Krea-2-Raw.safetensors --lora my_lora.safetensors
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from safetensors import safe_open


def inventory(path: Path):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        shapes = {key: tuple(f.get_slice(key).get_shape()) for key in keys}
        meta = f.metadata() or {}
    return keys, shapes, meta


def top_prefix(key: str) -> str:
    return key.split(".", 1)[0]


def print_inv(label: str, path: Path):
    keys, shapes, meta = inventory(path)
    print(f"\n[{label}] {path}\nkeys={len(keys)} metadata={meta}")
    print("top prefixes:", dict(collections.Counter(top_prefix(key) for key in keys)))
    for key in keys[:20]:
        print(" ", key, shapes[key])
    return keys, shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("turbo", type=Path, nargs="?")
    parser.add_argument("--full-ft", type=Path)
    parser.add_argument("--lora", type=Path)
    args = parser.parse_args()

    raw_keys, raw_shapes = print_inv("raw", args.raw)
    raw_set = set(raw_keys)
    expected = {"blocks", "first", "last", "tmlp", "tproj", "txtfusion", "txtmlp"}
    actual = set(map(top_prefix, raw_keys))
    print("raw bare-prefix check:", actual == expected, "expected", sorted(expected), "actual", sorted(actual))

    if args.turbo:
        turbo_keys, turbo_shapes = print_inv("turbo", args.turbo)
        print("raw/turbo key sets equal:", set(turbo_keys) == raw_set)
        shape_bad = [key for key in raw_set & set(turbo_keys) if raw_shapes[key] != turbo_shapes[key]]
        print("raw/turbo shape mismatches:", len(shape_bad), shape_bad[:20])

    if args.full_ft:
        ft_keys, ft_shapes = print_inv("full-ft", args.full_ft)
        print("full-ft/raw key sets equal:", set(ft_keys) == raw_set)
        print("missing:", sorted(raw_set - set(ft_keys))[:30])
        print("unexpected:", sorted(set(ft_keys) - raw_set)[:30])
        shape_bad = [key for key in raw_set & set(ft_keys) if raw_shapes[key] != ft_shapes[key]]
        print("shape mismatches:", len(shape_bad), shape_bad[:20])

    if args.lora:
        lora_keys, _ = print_inv("lora", args.lora)
        counts = collections.Counter()
        for key in lora_keys:
            if key.startswith("diffusion_model."):
                counts["diffusion_model"] += 1
            if key.startswith("transformer."):
                counts["transformer"] += 1
            if "lora_up" in key or "lora_B" in key:
                counts["up/B"] += 1
            if "lora_down" in key or "lora_A" in key:
                counts["down/A"] += 1
            if key.endswith("alpha") or ".alpha" in key:
                counts["alpha"] += 1
        print("LoRA namespace/features:", dict(counts))
        if counts["transformer"] and not counts["diffusion_model"]:
            print("WARNING: this adapter is in OneTrainer-native transformer.* namespace, not external Krea 2 namespace.")


if __name__ == "__main__":
    main()
