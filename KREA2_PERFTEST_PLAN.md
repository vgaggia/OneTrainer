# Krea2 perf-test plan — cloud, small dataset, 2026-07-25

Goal: find the fastest **full fine-tune** config for Krea2 at 512 **without using the local 5090**
(that machine is for gaming). Small dataset, cheap GPU, all variants in one pod session.

## Why this shape

- The dataset is ~459 images, so the latent cache is ~4 GB and builds in minutes. **No network
  volume needed** — which also removes the EUR-IS-1 datacenter lock and its "Low" GPU stock.
- Renting an **RTX 5090** (~$0.69/hr, seen in EUR-IS-1) gives results that transfer directly: the
  5090 and the RTX PRO 6000 are the same GB202 silicon, and the 5090 has the same 32 GB the real
  local run would have. Fidelity to the eventual local run is the point.
- Model download (25 GB Krea2 + 8.3 GB Qwen3-VL) is the slow part, so **run every config in one
  pod session** — download once, cache once, then iterate.
- Rough budget: ~15 min setup + ~15 min per config x 5 = **~90 min, ~$1**. Balance is $36.51.

## Step 1 — private HF dataset of just Akko

Source (already extracted locally):
```
E:\vllm\hf-cache\onetainer_archives\vgaggia--vgaggia_3_2026\9aa45c1f2a6fdea8\Akko
459 images + 501 .txt captions
```
Follow the existing convention (see memory `hf-dataset-upload-workflow`): zip it, push to a
**private** repo, reference it as `hf-archive:vgaggia/<repo>/<file>.zip`. `HF_TOKEN` is in env.
Suggested repo: `vgaggia/akko_perftest`.

The concept file `training_concepts/_perftest_akko.json` currently points at the **local** path —
switch `path` to the `hf-archive:` form once uploaded. Keep `seed: 12345` fixed so the cache is
reused across every run.

## Step 2 — pod

One pod, no network volume. `cloud.gpu_count: 1`. Reuse the working cloud settings from
`training_configs/Latest krea FT Cloud.json` (cuda_version 13.0, template, detach etc.), but
**drop `network_volume_id` and `data_center_id`** so it can land anywhere with stock.

The client-side fixes in `perf/efficiency-experiments` are required and already committed —
notably the requirements re-sync (`85119283`), without which the pod boots a stale venv and dies
on an ImportError.

## Step 3 — the matrix

Configs already written to `perf_tests/configs2/` (derived from `Krea2 5090 QWT tuned.json`,
FINE_TUNE + INT_W8A8 + QWT, 512, batch 2, 1 epoch). Runner: `perf_tests/run_matrix2.sh`.
They need cloud paths (`cache_dir`, `workspace_dir`, concept path) before use.

| Run | Change | Question |
|---|---|---|
| `A_qwt_baseline` | as-tuned: `CPU_OFFLOADED`, offload 0.2, `compile: False` | reproduce run 14's stack at 512 |
| `B_qwt_fast` | checkpointing `ON`, offload `0.0`, `compile: True` | how much were the 768-era memory crutches costing? |
| `C_qwt_nockpt` | checkpointing `OFF` | **zero quality change**, ~25-33% faster, costs VRAM |
| `D_qwt_freeze14` | `layer_filter` = blocks 14-27 | ~33% compute off, mild quality cost |
| `E_qwt_muon` | `MUON` instead of Adafactor | does the momentum buffer even fit? (expect OOM, see below) |

Each run ends with 4 sample images from `training_samples/_perftest_prompts.json`: the trained
subject, an **unrelated** subject (catastrophic-forgetting check), a macro detail shot, and a
colour/composition test. Eyeball these, not just s/it — the whole point is speed *without* quality
loss.

Log the median steady-state s/it (drop the first 30% for compile warmup) and peak VRAM per run.

## Findings that shape this (do not re-derive)

### The frozen-config trap
`quantization_util.py` warns: with `quantized_weight_training: False`, FINE_TUNE trains **only
norms, biases and embeddings** — quantized Linears cannot receive gradients. Commit `3e206d68` is
literally titled "FAST **frozen** config". So the fast local numbers (run 10 at 1.75 s/it, run 15
at 1.125) are **not real fine-tunes**. Every config in this matrix sets
`quantized_weight_training: True`.

### Run 14 was crippled, so 67 h/epoch is probably wrong
`Krea2 5090 QWT tuned.json` has `gradient_checkpointing: CPU_OFFLOADED`, `layer_offload_fraction:
0.2` and **`compile: False`** — crutches that 768/batch-4 forced. At 512 they should mostly come
off. Runs A vs B vs C measure exactly this.

### Layer filters DO work for full fine-tune
`Krea2FineTuneSetup.py:41` passes `freeze=ModuleFilter.create(config)`; `BaseModelSetup.py:199`
shows **match = trained, non-match = frozen**. Freezing the *first* k blocks lets autograd stop
backprop below the first trainable block, so those blocks cost forward only — **2/3 of their cost
disappears**. Freezing 14 of 28 ~= 33% off total compute.

### Muon: right idea, wrong VRAM
Muon is genuinely the "designed around tensor cores" optimizer (Newton-Schulz orthogonalisation =
matmuls on tensor cores, and better-conditioned updates — its real selling point is convergence,
not memory). But it keeps a **persistent momentum buffer**: 12.8B x 2 B = **25.6 GB**, plus ~12.8
GB of INT8 weights = 38.4 GB against 32 GB. Adafactor's factored state is ~0.05 GB, which is why
it is the only thing that fits. Muon would fit on a 96 GB PRO 6000. Run E to confirm empirically.

### LR scaler is sqrt, and the docs are wrong
`NamedParameterGroup.py:41`: `lr = lr * (scale ** 0.5)`. `KREA2_TUNING_REFERENCE.md:167` omits the
`** 0.5` and would lead you to set an LR **2x too high**. Scaling also applies to per-part LR
overrides. **This doc line should be fixed.**

### TREAD: not a misconfiguration, a method/task mismatch
Implementation is correct (text tokens excluded, rope gathered, mask gathered, inference clean).
The problem is that `[2, -3]` on 28 blocks means dropped tokens traverse only **4 of 28 blocks**,
and `route_info` is a local in `forward` so the loss **cannot** be masked — bypassed tokens'
errors backprop into blocks 26/27 and `final_layer`, which every token uses. That distils a
4-block shortcut into pretrained weights. Fine for pretraining, harmful for fine-tuning. Measured
1.68x speedup matches 24/28 x 0.5 = 42.9% exactly, i.e. **the speedup is the damage**.
`perf_tests/results/REPORT.md` still calls the loss spike "mechanically expected" — **that is
wrong and should be corrected.**

### Economics that killed the cloud full run
1x PRO 6000, batch 8, res 1024, bf16 full FT: **17.36 s/it, 100% util, 599.5 W of a 600 W cap** ->
94 h/epoch, **~$187/epoch**, 8 epochs ~= 31 days / ~$1,500. More GPUs don't reduce cost (compute is
compute); they only cut wall-clock. Real levers are dataset size, epoch count, resolution.

## Local state

- **Orphaned cache deleted.** `F:\Krea2 Model\workspace-cache\run` (1.2 TB, 320k files) is gone;
  F: went from 90 GB free to **1,325.6 GB**. It was orphaned because the cache key includes
  `concept.seed` and OneTrainer assigns a random seed when a concept is created — regenerating a
  concept file orphans its entire cache. That directory accumulated 18 groups from past runs.
- Text embeddings were **fp32**: 65 tokens x 12 tapped layers x 2560 dim x 4 B ~= 7.6 MB/item, i.e.
  1.2 TB of the 1.24 TB. bf16 would halve it. Worth checking if that is configurable.
- **7 zero-filled (corrupt) images** found in `datasets--vgaggia--vgaggia_2026/.../08_eeveelutionsmix`
  (Eeveelution, Espeon). Right file size, all-zero contents — a bad download, not a format issue.
  List in `bad_images.txt`. The scan only covered `hub/`, **not** `onetainer_archives/`, so the
  count is partial. These crash caching with `PIL.UnidentifiedImageError`.
- Stack is current and needs no work: torch 2.12.0+cu130, cuDNN 9.20, Triton 3.7.0, sm_120,
  driver 610.74, 31.8 GB usable.

## Cost / cleanup

- Balance **$36.51**, no pods running.
- Network volume `1heen9fpx7` (2000 GB, EUR-IS-1) still bills **$100/mo = $0.137/hr**, which
  drains the balance in ~11 days. **User is deleting this themselves.** Nothing on it is
  irreplaceable: model and datasets are on HF, and its 1024 latents are useless for a 512 run.

## Config artifacts

| Path | What |
|---|---|
| `training_configs/Krea2 LoRA 512 Local 3ep.json` | the LoRA run if you ever want it: 3 ep, 512, batch 3 x accum 2, COSINE, lr 1.5e-4 x BOTH scaler (= 3.674e-4 effective), backup 4h x3 |
| `perf_tests/configs2/*.json` | the 5 full-FT variants above |
| `perf_tests/run_matrix2.sh` | runner; parses median s/it, flags OOM |
| `training_concepts/_perftest_akko.json` | Akko concept, seed 12345 — **repoint at hf-archive** |
| `training_samples/_perftest_prompts.json` | 4 eval prompts @ 512 |

Committed on `perf/efficiency-experiments`: `64870439` (network volume + 3 cloud bugs),
`85703245` (parallel zip extract 18->105 files/s), `fb8d58b0` (krea2 saver dequantises before
writing), `6ebc49fd` (int8 gemm padding), `3402ad47` (cache guard + tests), `85119283`
(requirements re-sync), `7faf3dff` (expandable_segments). 17 tests:
`python -m unittest discover -s tests -t .`
