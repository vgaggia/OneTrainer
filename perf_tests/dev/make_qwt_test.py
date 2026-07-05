"""Build the QWT smoke-test config from the user's real config (uploaded copy
placed at perf_tests/dev/user_config.json). Real everything, except:
true quantized-weight training on, accum 1, compile off, TREAD off (isolate),
small offload fraction for sampling headroom. Cache is REUSED (their real cache);
run is killed at ~110 steps by the operator, so no final save happens."""

import json

c = json.load(open("perf_tests/dev/user_config.json", encoding="utf-8"))

# old config predates several fixes: single-file base -> transformer override,
# int8 compute, cuDNN attention
if c["base_model_name"].lower().endswith(".safetensors"):
    c["transformer"]["model_name"] = c["base_model_name"]
    c["base_model_name"] = "krea/Krea-2-Raw"
c["transformer"]["weight_dtype"] = "INT_W8A8"
c["attention_mechanism"] = "CUDNN"
c["output_model_format"] = "ORIGINAL_TRANSFORMER"

c["quantized_weight_training"] = True
c["gradient_accumulation_steps"] = 1
c["compile"] = False
c["tread_enabled"] = False
c["layer_offload_fraction"] = 0.2          # headroom: qwt peaked 31.9GB without sampling
c["clear_cache_before_training"] = False   # reuse the real cache
c["tensorboard_always_on"] = False
c["tensorboard"] = False  # smoke tests are killed mid-run; orphaned tensorboards squat port 6006

# keep their real cache, but use a separate workspace so tensorboard runs don't mix
c["workspace_dir"] = "G:\\Auto\\OneTrainer\\perf_tests\\workspace\\16_qwt_real_smoke"

out = "training_configs/Krea2 5090 QWT test.json"
json.dump(c, open(out, "w", encoding="utf-8"), indent=4)
print("wrote", out)
print("transformer:", c["transformer"]["model_name"])
print("cache:", c["cache_dir"])
