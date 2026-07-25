# perf_tests

Performance experiment harness for the `perf/efficiency-experiments` branch.
Everything here is generated/disposable; nothing touches the user's own presets,
concepts, samples, or workspace.

## Layout

- `make_configs.py` - regenerates concepts/samples/configs from the
  `krea2 FT RTX 5090 Long` preset (base: `krea/Krea-2-Raw` diffusers repo +
  transformer-only single-file override, HF cache at `E:\vllm\hf-cache`).
- `parse_results.py` - summarizes logs/VRAM/tensorboard into `results/summary.md`.
- `configs/` - one JSON per run, numbered in execution order.
- `logs/` - `<run>.log` (cmd redirect, UTF-8) + `<run>_vram.csv` (nvidia-smi poller).
- `workspace/<run>/` - OneTrainer workspace incl. tensorboard events.
- `cache/res768/` - shared MGDS latent/text-embed cache (identical preprocessing
  across runs; W8A8 etc. only changes the transformer, not preprocessing).

## Run matrix

| run | method | transformer dtype | purpose |
|---|---|---|---|
| 00_ft_baseline | FINE_TUNE | FLOAT_8 | user's daily-driver behavior, reference |
| 01_lora_f8_storage | LORA | FLOAT_8 | LoRA reference (8-bit storage, bf16 compute) |
| 02_lora_int_w8a8 | LORA | INT_W8A8 | true int8 matmul compute |
| 03_lora_fp8_w8a8 | LORA | FLOAT_W8A8 | true fp8 matmul compute |

Later phases append runs (fused back-pass, per-channel W8A8 scales, H2D-only
offload, fp8 activation offload, TREAD) - see the session task list.

## Protocol

~42 images (G:\Datasets\Smallgia), bs2 res768, 6 epochs = ~126 iterations.
Timing = median s/it over the last 70% of step-bar samples (skips compile
warmup). Loss comparability: same concept seed(42), same data order; treat
<130-step loss deltas as sanity checks, not quality verdicts.

Launch pattern (cmd, not powershell - keeps logs UTF-8):

```
cd /d G:\Auto\OneTrainer && set HF_HOME=E:\vllm\hf-cache&& set HF_HUB_DISABLE_XET=1&& venv\Scripts\python.exe scripts\train.py --config-path perf_tests\configs\<run>.json > perf_tests\logs\<run>.log 2>&1
```

Known gotchas hit so far:
- PowerShell `*>` writes UTF-16 logs; use cmd redirects.
- Crashed runs leave orphaned `tensorboard` processes holding log handles - kill them.
- Preset-style single-file `base_model_name` is rejected by the Krea2 loader;
  must use diffusers repo + transformer override.
