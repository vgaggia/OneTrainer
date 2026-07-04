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

## Session 2: the "not done" items, now done

### Key discovery first

`LinearFp8.forward` (and all quantized Linears) DETACH their weights: quantized
Linear weights never receive gradients, even with training_method FINE_TUNE.
A quantized-weights "full fine-tune" trains only norms, per-block modulation
tables, and the few biases. This is why FLOAT_8 FT fits in 32GB and why it was
faster than LoRA. Consequence: W8A8 compute could be enabled for FT by fixing
the actual blockers (trainable biases on quantized layers; blanket
requires_grad_ on int8 params is illegal).

### New measured results

| run | s/it | vs | loss |
|---|---|---|---|
| 10 FT INT_W8A8 | 1.75 | 1.51x vs FT baseline 2.635 | 0.1299 (baseline 0.1295) |
| 11 LoRA INT_W8A8, offload 0.5, copy-back control | 2.04 | - | 0.13053 |
| 12 same, H2D-only masters | 1.95 | 4.4% faster than 11; only 1.5% over no-offload (1.92) | 0.13053 |
| 13 same + fp8 activation compression | 1.94 | activation PCIe/pinned cache halved | 0.13053 |

### Code shipped (session 2)

1. **W8A8 for fine-tuning**: generalized W8A8 backward (bias gradients, partial
   needs_input_grad); per-parameter requires_grad handling that keeps quantized /
   integer params frozen. FT+INT_W8A8 = 1.51x with a matched loss curve.
2. **H2D-only offload**: first eviction of a frozen layer builds a pinned CPU
   master; every later eviction repoints tensor.data (zero D2H). Auto-detected
   per layer; trainable layers keep copy-back. `OT_DISABLE_H2D_ONLY=1` reverts.
3. **FP8 activation-offload compression** (`activation_offload_compression`,
   opt-in): per-tensor-scaled e4m3 for the PCIe trip. Debugging this exposed a
   subtle cross-stream use-after-free (caching allocator reuses freed storage on
   the origin stream while another stream still reads it); fixed with
   record_stream guards, same pattern the stock code uses.
4. **Preset fixes**: stale `output_model_format: SAFETENSORS` corrected in all
   Krea2 FT presets (crashed the saver at end of training).
5. **New preset**: `krea2 FT RTX 5090 Long PERF.json` = LONG + INT_W8A8 +
   TREAD at conservative 0.25. Expected step time vs LONG: ~1.5x from int8
   alone, more from TREAD. Set `tread_enabled: false` if long-run samples degrade.

### Consciously not built: offloaded gradient accumulation

The idea assumed large persistent grad buffers in FT. With quantized weights
frozen (see discovery), Krea2 FT grads are tiny (norms/tables/biases), so
there is nothing worth offloading. Only pays for true-bf16 full FTs, which
don't fit on 32GB regardless. Documented instead of built.

## Session 3: true quantized-weight training (SHIPPED, experimental)

`quantized_weight_training: true` makes the 12.9B int8 Linear weights actually
train, for the first time on this hardware. Design: no master weights; the
weight gradient is computed in int8 inside autograd backward (dY and X
quantized per-column so scales factor out of the token contraction; wgrad
rel-err 1.25% vs bf16 reference), then a fused per-layer factored Adafactor
step updates the dequantized weight and re-quantizes with STOCHASTIC ROUNDING
(int8 per-channel ULP ~0.8% of channel max, ~2x coarser than bf16's 0.39%,
same principle as OneTrainer's bf16-SR training). Optimizer state: one fp32
row+col vector per matrix (~50KB/layer).

Constraints (enforced): FINE_TUNE, INT_W8A8, gradient_accumulation_steps=1
(update fused into backward; effective batch drops accordingly), compile=off.
Constant-lr scheduler assumed (v1 reads lr from config).

Validation:
- unit: toy 512x512 fit converges 0.331 -> 0.0013 in 600 SR steps; sub-ULP
  drift confirmed at lr 2e-5.
- run 14 (126 iterations, bs2, 264 layers training): stable, no NaN, VRAM
  peak ~31.9GB (tight), 6.97 s/it median. Smooth loss declined 0.123 -> 0.118
  while every frozen-weight run stayed flat at ~0.124: the weights are learning.

Speed reality: 6.97 s/it vs 1.75 frozen. Costs: no compile, wgrad GEMMs, and
above all the eager fp32 dequant->update->SR->requant pass over 12.9B weights
per step plus allocator pressure at the VRAM ceiling. Future optimizations
(likely 2-3x): a fused Triton update kernel (single pass, bf16 transients,
chunked rows), compile compatibility by moving the update out of the traced
graph, dropping per-step scale refresh to every N steps.

Recommendation: try a real run with `14_ft_qwt_int8_sr`-style config and judge
samples. This changes WHAT trains, not just how fast: results should differ
qualitatively from the norm/bias-only FT you have been running.

## Session 3 addendum: attention backend A/B (upstream #1227 exposed the knob)

Run 15 = run 10 with `attention_mechanism: CUDNN` instead of SDP. Result on the
RTX 5090: **1.125 s/it vs 1.75 (1.56x), identical loss to 4 decimals.** cuDNN's
SDPA kernels are simply much faster than the default backend for Krea 2's GQA
attention on Blackwell. Baked into the PERF preset. Combined frozen-FT stack:
2.635 (original) -> 1.125 s/it = 2.34x, before TREAD.
