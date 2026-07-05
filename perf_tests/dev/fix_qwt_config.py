"""Normalize the QWT test config: enforce every field quantized-weight training
requires, regardless of what UI state was saved over it."""

import json

p = "training_configs/Krea2 5090 QWT test.json"
c = json.load(open(p, encoding="utf-8"))
fixes = []


def need(key, val, node=None, label=None):
    node = node if node is not None else c
    label = label or key
    if node.get(key) != val:
        fixes.append(f"{label}: {node.get(key)} -> {val}")
        node[key] = val


need("compile", False)
need("quantized_weight_training", True)
need("gradient_accumulation_steps", 1)
need("gradient_checkpointing", "CPU_OFFLOADED")
need("fused_back_pass", True, c["optimizer"], "optimizer.fused_back_pass")
if "ADAFACTOR" in c.get("optimizer_defaults", {}):
    need("fused_back_pass", True, c["optimizer_defaults"]["ADAFACTOR"],
         "optimizer_defaults.ADAFACTOR.fused_back_pass")
need("weight_dtype", "INT_W8A8", c["transformer"], "transformer.weight_dtype")
need("attention_mechanism", "CUDNN")

json.dump(c, open(p, "w", encoding="utf-8"), indent=4)
print("fixed:" if fixes else "file was already consistent")
for f in fixes:
    print("  " + f)
print("tread:", c.get("tread_enabled"), c.get("tread_selection_ratio"),
      "| offload:", c.get("layer_offload_fraction"))
