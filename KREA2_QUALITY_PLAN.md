# Krea2 — optimising for learning quality, not speed (2026-07-25)

Companion to `KREA2_PERFTEST_PLAN.md`. That doc is about seconds-per-step; this one is about
**quality per GPU-hour**. Stated goal: *"one epoch that learns extremely well"* beats three fast ones.

## The arithmetic that governs everything

Krea2 is 12.8B params (~10.92B of that in the 28 transformer blocks, so **~390M per block**).
Weights live as INT8: **~12.8 GB resident**, always, because the full model must run forward.
Usable VRAM on a 5090: **31.8 GB**. So the optimizer + activations must fit in ~19 GB.

Optimizer state for **all 12.8B** params:

| Optimizer | Persistent state | Fits in 19 GB? |
|---|---|---|
| AdamW fp32 (2 buffers) | 95.4 GB | no |
| AdamW 8-bit | 23.8 GB | no |
| Lion / Muon (1 momentum buffer, bf16) | 25.6 GB | no |
| **CAME** (`exp_avg` is full-size, `CAME.py:101`, + factored rows/cols) | **25.6 GB** | **no** |
| **Adafactor, beta1=None** (factored 2nd moment only, no momentum) | **~0.05 GB** | **yes** |

**Conclusion: if you train all 28 blocks on a 32 GB card, Adafactor-without-momentum is not a
preference, it is the only option.** Every optimizer with a per-parameter momentum buffer needs
>=25.6 GB. This is why the current config uses it, and it is a real quality compromise — factoring
the second moment is an approximation of Adam, and dropping momentum entirely costs more.

## The key idea: freezing layers is what buys you a better optimizer

Optimizer state is only allocated for **trainable** parameters (`BaseModelSetup.py:193-203` — only
params matching `layer_filter` enter the parameter groups). So freezing blocks does two things at
once:

1. cuts compute (autograd stops backprop below the first trainable block — those blocks cost
   forward only, so **2/3 of their cost disappears**)
2. cuts optimizer state **proportionally**, which is what unlocks CAME or Muon

This reframes `D_qwt_freeze14` completely. It was filed as a speed run with a quality cost. It is
actually **the enabler for a better optimizer** — which is a quality *gain*.

### The design curve

Trainable blocks -> trainable params -> momentum cost (bf16) -> free VRAM after 12.8 GB weights:

| Trainable blocks | Params | CAME/Muon momentum | Optimizer+weights | Left for activations |
|---|---|---|---|---|
| 28 (all) | 10.9B | 21.8 GB | 34.6 GB | **negative — blocked** |
| 21 (7-27) | 8.2B | 16.4 GB | 29.2 GB | ~2.6 GB (too tight) |
| **14 (14-27)** | **5.5B** | **10.9 GB** | **23.7 GB** | **~8 GB — plausible** |
| **7 (21-27)** | **2.7B** | **5.5 GB** | **18.3 GB** | **~13 GB — comfortable, bigger batch** |

So there is a genuine trade to explore: **fewer trainable blocks -> affords a real optimizer AND a
larger batch -> better gradient quality on the layers you actually train.** For a style/subject
fine-tune, late blocks matter most, so this may cost far less than it sounds.

Note the weights column never shrinks — all 28 blocks must stay resident for the forward pass.
Only optimizer state and backward compute scale with the trainable count.

## CAME

- `modules/util/optimizer/CAME.py`. Confidence-guided Adaptive Memory Efficient optimization.
  Designed explicitly to fix Adafactor's instability at similar memory — closer to Adam quality.
- State: `exp_avg` (**full size**) + factored `exp_avg_sq_row/col` + factored `exp_avg_res_row/col`.
  The factored parts are negligible; the momentum buffer is the cost.
- **Has `step_parameter` (`CAME.py:84`)**, so fused back pass works — which is mandatory for full FT
  on this card.
- `CAME_8BIT` also exists and would roughly halve the momentum buffer — worth testing if 14
  trainable blocks is still too tight.

**Verdict: the best available quality upgrade over Adafactor, but only affordable with layer
freezing.** Test at 14 and 7 trainable blocks.

## Muon

You keep asking, so: **the idea is right and it does not fit at full scale.**

- Muon's selling point is *convergence*, not memory — it orthogonalises the momentum via
  Newton-Schulz, which is a short series of matmuls that run on tensor cores nearly free, giving
  better-conditioned updates. Proven at frontier scale (Kimi K2). This is genuinely the
  "designed around the hardware" optimizer.
- But it needs **one persistent momentum buffer** and the orthogonalisation needs the full matrix,
  so it cannot be factored the way Adafactor factors its second moment. 12.8B -> 25.6 GB. Blocked.
- **Same escape hatch as CAME, but tighter.** Muon has **no `step_parameter`**, so it cannot use
  fused back pass — which means gradients for every trainable param coexist, on top of the momentum
  buffer. At 14 trainable blocks that is 10.9 GB grads + 10.9 GB momentum + 12.8 GB weights =
  **34.6 GB, over budget**. At **7 blocks** it is 5.5 + 5.5 + 12.8 = 23.8 GB, leaving ~8 GB.
  **Muon is only viable at <=7 trainable blocks.** CAME does not have this problem because
  `step_parameter` keeps gradients transient.
- OneTrainer already has `muon_util.build_muon_adam_key_fn` to route 2D weight matrices to Muon and
  everything else (norms, biases, embeddings) to Adam — exactly how Muon is meant to be used.
  `MuonWithAuxAdam: True` is the default; aux-Adam LR defaults to `3e-4`.
- `MUON`, `MUON_ADV`, `ADAMUON_ADV` are all available.

**Caveat: Muon needs its own LR.** Its matrix-group LR typically lives near 1e-2, not the 2e-5 used
for full FT here. Swapping the optimizer while keeping the tuned LR tells you nothing. Muon runs
must retune LR or they are wasted.

**Verdict: worth testing at 7-14 trainable blocks, as a convergence experiment, with its own LR.**
Not a drop-in.

## Effective batch size — the other quality ceiling

Currently **effective batch 4** for a 12.8B model. That is very noisy.

Gradient accumulation is **blocked**: full FT on 32 GB requires `fused_back_pass: True` (the
optimizer step runs per-parameter as each gradient arrives, so gradients never all coexist), and
accumulation requires gradients to persist across micro-batches — a full ~25 GB gradient buffer.
Confirmed in `KREA2_TUNING_REFERENCE.md:515`.

So the only lever is **raising true `batch_size`**, and 512 gives far more headroom than the 768
config assumed. **Run a batch ladder** (4 -> 6 -> 8 -> 10 -> OOM) on the winning config. Going 4 -> 10
is a 2.5x reduction in gradient noise, and it composes with the freezing above (fewer trainable
blocks -> more free VRAM -> bigger batch).

## Already correct — do not touch

- `timestep_distribution: LOGIT_NORMAL` with `dynamic_timestep_shifting: True` — the right setup for
  a flow-matching model, and the dynamic shift auto-adapts for 512 vs 1024.
- `clip_grad_norm: 1.0`.
- `stochastic_rounding: True` — required to update low-precision weights without an fp32 master copy.

## Blocked, for the record

- **EMA** — normally a solid final-quality win, but needs a full weight copy (12.8 GB+). Only viable
  on LoRA (~0.2 GB there).
- **Gradient accumulation** — see above.
- **Larger effective batch via more GPUs** — a 96 GB card removes both the optimizer and batch
  ceilings entirely. That is the one thing the 5090 fundamentally cannot do (~$187/epoch at 1024).

## Suggested runs (add to the matrix)

Order matters — each needs the previous run's VRAM number.

All written to `perf_tests/configs2/`. Order matters — each needs the previous run's VRAM number.

| Run | Trainable | Optimizer | Batch | LR | Purpose |
|---|---|---|---|---|---|
| `C_qwt_nockpt` | all 28 | Adafactor | 2 | 2e-5 | read **peak VRAM** — the budget everything else spends |
| `F_freeze14_came` | 14-27 | CAME (fused bp) | 2 | 2e-5 | headline experiment |
| `I_freeze7_adafactor` | 21-27 | Adafactor | 4 | 2e-5 | **control** — same blocks, current optimizer |
| `G_freeze7_came` | 21-27 | CAME (fused bp) | 4 | 2e-5 | vs I, isolates the optimizer |
| `H_freeze7_muon` | 21-27 | Muon + auxAdam | 2 | **1e-2** | only viable Muon config |

Then a **batch ladder** (4 -> 6 -> 8 -> 10 -> OOM) on whichever survives with the most headroom.

**Do not compare H's loss to the others** — different optimizer *and* different LR. Judge it on
samples only. `I` exists so that `G` and `H` have a like-for-like baseline at the same trainable
block count; without it, any difference confounds "fewer blocks" with "different optimizer".

Judge these on the **sample images** (`training_samples/_perftest_prompts.json` — trained subject,
unrelated subject for catastrophic-forgetting, macro detail, colour/composition), not on loss
alone. Loss across different optimizers and different trainable-parameter counts is not comparable.

## Correction log

- I recommended CAME before checking its state allocation. It keeps a full momentum buffer and does
  **not** fit at 28 trainable blocks. The recommendation only survives with layer freezing.
- `D_qwt_freeze14` was originally filed as a speed-with-quality-cost run. It is better understood as
  the enabler for a momentum-based optimizer.
