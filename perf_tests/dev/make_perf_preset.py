"""Create 'krea2 FT RTX 5090 Long PERF.json' from the user's LONG preset.
Applies only the validated improvements; never touches the original preset."""

import json

SRC = "training_presets/krea2 FT RTX 5090 Long.json"
DST = "training_presets/krea2 FT RTX 5090 Long PERF.json"

c = json.load(open(SRC, encoding="utf-8"))

# validated: int8 W8A8 compute, same trainable set as FLOAT_8, ~1.3x step time (run 10)
c["transformer"]["weight_dtype"] = "INT_W8A8"

# TREAD token routing: big speed win (run 09: 2.635 -> 1.57 s/it at ratio 0.5).
# Shipped CONSERVATIVELY at 0.25 because long-run quality is not yet validated;
# set tread_enabled false if samples degrade, or raise ratio for more speed.
c["tread_enabled"] = True
c["tread_selection_ratio"] = 0.25
c["tread_start_layer"] = 2
c["tread_end_layer"] = -3

# stale-format fix is already applied to the source preset, but be explicit:
c["output_model_format"] = "ORIGINAL_TRANSFORMER"

json.dump(c, open(DST, "w", encoding="utf-8"), indent=4)
print("wrote", DST)
