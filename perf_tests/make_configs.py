"""Generate perf-test configs derived from the user's 'krea2 FT RTX 5090 Long' preset.

Run from repo root:  venv\\Scripts\\python.exe perf_tests\\make_configs.py
Regenerates everything under perf_tests/{concepts,samples,configs}.
Never touches the user's own presets/concepts/samples.
"""

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PT = REPO / "perf_tests"
PRESET = REPO / "training_presets" / "krea2 FT RTX 5090 Long.json"
DATASET = "G:\\Datasets\\Smallgia"
EPOCHS = 6  # ~42 imgs, bs2 -> 21 iters/epoch -> ~126 iterations per run


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    print(f"wrote {p.relative_to(REPO)}")


def make_concept():
    # Schema mirrors ConceptConfig v2; content is our own (perf dataset).
    template = load(REPO / "training_concepts" / "Gelly.json")[0]
    c = copy.deepcopy(template)
    c["name"] = "perf_smallgia"
    c["path"] = DATASET
    c["seed"] = 42
    c["enabled"] = True
    c["include_subdirectories"] = False
    c["image_variations"] = 1
    c["text_variations"] = 1
    c["balancing"] = 1.0
    c["loss_weight"] = 1.0
    c["concept_stats"] = {}
    # deterministic-ish: keep crop jitter (their default), no random flips/rotates
    for k in list(c["image"].keys()):
        if k.startswith("enable_random_") or k.startswith("enable_fixed_"):
            c["image"][k] = False
    c["image"]["enable_crop_jitter"] = True
    c["text"]["prompt_source"] = "sample"  # per-image .txt captions
    return [c]


def make_samples():
    base = {
        "__version": 0,
        "enabled": True,
        "prompt": "",
        "negative_prompt": "",
        "height": 768,
        "width": 768,
        "frames": 1,
        "length": 10.0,
        "seed": 42,
        "random_seed": False,
        "diffusion_steps": 20,
        "cfg_scale": 4.5,
        "noise_scheduler": "EULER_A",
        "text_encoder_1_layer_skip": 0,
        "text_encoder_2_layer_skip": 0,
        "text_encoder_3_layer_skip": 0,
        "text_encoder_4_layer_skip": 0,
        "prior_attention_mask": False,
        "force_last_timestep": False,
        "sample_inpainting": False,
        "base_image_path": "",
        "mask_image_path": "",
    }
    prompts = [
        "a detailed portrait photograph of a woman with red hair, soft studio lighting",
        "a small wooden cabin beside a mountain lake at sunrise, photorealistic",
    ]
    out = []
    for p in prompts:
        s = copy.deepcopy(base)
        s["prompt"] = p
        out.append(s)
    return out


def base_config():
    cfg = load(PRESET)
    # The preset points base_model_name at a single-file safetensors, which the
    # Krea2 loader rejects. Mirror the working arrangement from the live config:
    # diffusers repo as base + transformer-only single-file override.
    cfg["transformer"]["model_name"] = cfg["base_model_name"]
    cfg["base_model_name"] = "krea/Krea-2-Raw"
    # preset carries stale "SAFETENSORS", which Krea2ModelSaver rejects for FT
    if cfg.get("output_model_format") == "SAFETENSORS" and cfg.get("training_method") == "FINE_TUNE":
        cfg["output_model_format"] = "ORIGINAL_TRANSFORMER"
    cfg["concepts"] = make_concept()
    cfg["concept_file_name"] = str(PT / "concepts" / "smallgia.json")
    cfg["epochs"] = EPOCHS
    cfg["validation"] = False
    cfg["debug_mode"] = False
    cfg["dataloader_threads"] = 1
    cfg["tensorboard"] = True
    cfg["tensorboard_always_on"] = False
    cfg["latent_caching"] = True
    cfg["clear_cache_before_training"] = False
    # disable sampling / backups / intermediate saves for clean timing
    for key, val in [
        ("sample_after_unit", "NEVER"),
        ("backup_after_unit", "NEVER"),
        ("save_every_unit", "NEVER"),
        ("rolling_backup", False),
        ("continue_last_backup", False),
    ]:
        if key in cfg:
            cfg[key] = val
    if "sample_definition_file_name" in cfg:
        cfg["sample_definition_file_name"] = str(PT / "samples" / "perf_samples.json")
    return cfg


def variant(name, **overrides):
    cfg = base_config()
    cfg["workspace_dir"] = str(PT / "workspace" / name)
    cfg["cache_dir"] = str(PT / "cache" / "res768")  # shared: identical preprocessing
    cfg["output_model_destination"] = str(PT / "out" / f"{name}.safetensors")
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        assert parts[-1] in node, f"unknown config key: {dotted}"
        node[parts[-1]] = value
    save(PT / "configs" / f"{name}.json", cfg)


def main():
    save(PT / "concepts" / "smallgia.json", make_concept())
    save(PT / "samples" / "perf_samples.json", make_samples())

    # 00: their full-FT preset, unchanged behavior (FLOAT_8 transformer, compile on)
    variant("00_ft_baseline")

    # 01-03: LoRA family (transformer frozen). Same everything except base weight dtype.
    lora = {
        "training_method": "LORA",
        "peft_type": "LORA",
        "lora_rank": 16,
        "lora_alpha": 1.0,
        "lora_weight_dtype": "FLOAT_32",
        # preset's stale SAFETENSORS maps to LEGACY_LORA, which Krea2 rejects
        "output_model_format": "COMFY_LORA",
    }
    variant("01_lora_f8_storage", **lora, **{"transformer.weight_dtype": "FLOAT_8"})
    variant("02_lora_int_w8a8", **lora, **{"transformer.weight_dtype": "INT_W8A8"})
    variant("03_lora_fp8_w8a8", **lora, **{"transformer.weight_dtype": "FLOAT_W8A8"})

    # 04/05: fused back-pass A/B. Fused BP only reduces VRAM at accum=1
    # (GenericTrainer warns otherwise), so both runs use accum=1.
    variant("04_ft_accum1_control", **{"gradient_accumulation_steps": 1})
    variant("05_ft_accum1_fused_bp",
            **{"gradient_accumulation_steps": 1, "optimizer.fused_back_pass": True})


if __name__ == "__main__":
    main()
