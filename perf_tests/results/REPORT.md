# OneTrainer perf experiments - session report (2026-07-04)

Branch: `perf/efficiency-experiments` (off `krea2-upstream`). All runs: Krea 2
12.9B, 42-image dataset, bs2 res768, ~126 iterations, RTX 5090. Timing = median
steady-state s/it (first 30% of samples dropped for compile warmup).

## Measured results

| run | s/it | speedup | loss (last 20%) | notes |
|---|---|---|---|---|
| 00 FT FLOAT_8 (daily driver) | 2.635 | FT ref | 0.1295 | ~27.5GB peak |
| 01 LoRA FLOAT_8 storage | 3.055 | LoRA ref | 0.1295 | dequant-to-bf16 compute |
| 03 LoRA FLOAT_W8A8 | 2.24 | 1.36x | 0.1311 | fp8 half-rate FP32-accum on GeForce |
| 02 LoRA INT_W8A8 | 1.92 | 1.59x | 0.1307 | int8 tensor cores, full rate |
| 06 LoRA INT_W8A8 per-channel | 1.99 | 1.54x | 0.1305 | new code; see verdict below |
| 08 LoRA INT_W8A8 + TREAD 0.5 | 1.00 | 3.05x | 0.425* | *see TREAD caveat |
| 09 FT + TREAD 0.5 | 1.57 | 1.68x (vs 00) | 0.41* | works on full FT unchanged |
| 04 FT accum=1 control | 2.98 | - | 0.1327 | optimizer step every iter |
| 05 FT accum=1 fused BP | cancelled | - | - | low value for accum-4 workflow |

## Code changes committed (the real improvements)

1. **LinearW8A8 alignment fix** - `torch._int_mm`/`_scaled_mm` require K/N
   alignment; Krea 2 has an in_features=12 Linear that hard-crashed W8A8.
   Misaligned layers now use the dequant fallback. W8A8 was unusable on Krea 2
   before this.
2. **Per-channel W8A8 weight scales** (default; `OT_W8A8_TENSORWISE=1` reverts).
   Kernel test: int8 forward rel-err 0.010 vs 0.033 tensorwise at equal kernel
   speed; backward err rises (0.033 -> 0.055) because outlier range moves into
   grad quantization; fp8 unaffected either way. End-to-end: loss 0.1305 vs
   0.1307, step time +3.6% (1.99 vs 1.92). Verdict: quality-per-setting slightly
   better, small speed cost; keep default, revisit with scale-spread clamping.
3. **TREAD token routing for Krea 2** - new `modules/util/TreadRouter.py`,
   routing hooks + batched-rope apply in `transformer_krea2.py` (pinned diffusers
   checkout, committed there), `tread_*` fields in TrainConfig, wiring in
   BaseKrea2Setup. Training-only, inference untouched, unit-tested (roundtrip,
   gradient flow, rope equivalence).

## TREAD caveat (important)

Loss during these 126-iteration runs sits ~3x higher. Mechanically expected:
dropped tokens' predictions ride an identity skip past blocks 2..25 and the
model needs hundreds of optimizer steps to adapt (SimpleTuner documents the same
spike; more pronounced for LoRA). These short runs prove SPEED (3.05x LoRA,
1.68x FT) and mechanical correctness, not final quality. Before adopting:
run a real-length FT with tread_selection_ratio 0.25-0.5 and compare samples.
Lower ratio = smaller distortion, smaller speedup.

## Incidental findings

- Preset "krea2 FT RTX 5090 Long" has stale `output_model_format: SAFETENSORS`:
  crashes Krea2 FT saver at end of training. Live config's ORIGINAL_TRANSFORMER
  is correct; fix the preset.
- ~45s dataloader restart overhead per epoch boundary (visible on small sets).
- Fused back-pass DOES run with grad accumulation but gives no VRAM benefit then
  (GenericTrainer.py:561 warning). Feature idea: offloaded gradient accumulation
  (pinned CPU accumulators) to make it pay off at accum>1.
- inductor suggests `torch.set_float32_matmul_precision('high')` (TF32) - minor.
- No torch.compile graph breaks or recompiles in any run log, including W8A8's
  autograd.Function paths.

## Not done this session (designs ready, see perf_tests/dev/ + session notes)

- H2D-only offloading for frozen weights (musubi `LoRAStreamOffloader` mapped:
  flat pinned CPU master per block, reference swap, ring slots; requires
  gradient checkpointing; frozen weights only).
- FP8-compressed activation offload (opt-in).
- W8A8 full fine-tune (stretch; master/compute split).
- Offloaded gradient accumulation (new idea).
