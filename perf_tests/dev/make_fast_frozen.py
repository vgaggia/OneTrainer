"""Build 'Krea2 5090 FAST frozen.json': the fastest validated NON-QWT stack.
Same data/concepts/schedule as the QWT config, but frozen-weight semantics
(identical training behavior to all pre-QWT fine-tunes, ~1.2 s/it class)."""

import json

SRC = "training_configs/Krea2 5090 QWT test.json"
DST = "training_configs/Krea2 5090 FAST frozen.json"

c = json.load(open(SRC, encoding="utf-8"))

c["quantized_weight_training"] = False   # frozen weights: norms/tables/biases train
c["compile"] = True                      # compiled blocks: fine without QWT
c["gradient_accumulation_steps"] = 4     # original schedule semantics
c["gradient_checkpointing"] = "ON"       # no offload needed: ~22GB peak
c["layer_offload_fraction"] = 0.0
c["tread_enabled"] = True
c["tread_selection_ratio"] = 0.25        # conservative ratio; set false if samples degrade
c["tread_start_layer"] = 2
c["tread_end_layer"] = -3
c["transformer"]["weight_dtype"] = "INT_W8A8"
c["attention_mechanism"] = "CUDNN"
c["tensorboard"] = True

# fused BP has no benefit at accum>1; keep the optimizer as their original frozen runs
c["optimizer"]["fused_back_pass"] = False
if "ADAFACTOR" in c.get("optimizer_defaults", {}):
    c["optimizer_defaults"]["ADAFACTOR"]["fused_back_pass"] = False

json.dump(c, open(DST, "w", encoding="utf-8"), indent=4)
print("wrote", DST)
