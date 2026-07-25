"""Fix Krea2 configs whose base_model_name points at a single .safetensors file:
move the file to the transformer override and set the diffusers repo as base."""

import glob
import json

for f in glob.glob("training_configs/*.json") + glob.glob("training_presets/Krea 2/*.json"):
    try:
        c = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if c.get("model_type") != "KREA_2":
        continue
    base = c.get("base_model_name", "")
    if base.lower().endswith(".safetensors"):
        if not c.get("transformer", {}).get("model_name"):
            c["transformer"]["model_name"] = base
        c["base_model_name"] = "krea/Krea-2-Raw"
        json.dump(c, open(f, "w", encoding="utf-8"), indent=4)
        print(f"fixed: {f}  (transformer override -> {c['transformer']['model_name']})")
