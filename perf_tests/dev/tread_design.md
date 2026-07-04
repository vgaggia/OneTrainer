# TREAD port for Krea 2 - implementation plan (apply after run 03)

Mechanism follows TREAD (arXiv 2501.04765) as shipped in SimpleTuner, written
fresh for OneTrainer. Token routing is training-only: a random subset of IMAGE
tokens skips blocks [start..end]; dropped positions are restored from the
pre-route tensor (identity skip, gradients flow). Text tokens are never routed.

## Pieces

1. `modules/util/TreadRouter.py` (new)
   - `TreadRouter(seed)` with dedicated `torch.Generator` on train device.
   - `get_mask(img_tokens, dropout_ratio) -> RouteInfo(ids_shuffle, ids_restore, keep_len)`
     scores = uniform noise; ids_shuffle = argsort(scores); keep K = S - round(S*ratio).
   - `start_route(x, info)`, `end_route(x_routed, info, original_x)` via take_along_dim.
   - `route_rope(cos_or_sin (S,D), info, text_len) -> (B, text_len+K, D)`:
     expand to batch, gather image rows by ids_keep, cat text rows in front.

2. `transformer_krea2.py` (pinned diffusers checkout)
   - `set_tread_router(router, routes)` setter; `_tread_router=None` default.
   - In `forward` block loop: route start/end as in the design above; routing sits
     OUTSIDE OneTrainer's checkpoint/offload wrappers (loop level).
   - Gate: `self.training and torch.is_grad_enabled() and self._tread_router`.
   - attention_mask (B,1,1,S): gather image columns by ids_keep during route.
   - `Krea2AttnProcessor`: if `image_rotary_emb[0].ndim == 3` (batched, routed),
     apply rotary inline with `cos[:, :, None, :]` unsqueeze; else existing path.

3. `modules/util/config/TrainConfig.py`: add fields
   - `tread_enabled: bool = False`
   - `tread_selection_ratio: float = 0.5`   (fraction of image tokens DROPPED)
   - `tread_start_layer: int = 2`
   - `tread_end_layer: int = -2`  (negative = from end, resolved against num blocks)

4. `modules/modelSetup/BaseKrea2Setup.py`: after checkpointing setup, if
   `config.tread_enabled`, construct router (seed 42) and call
   `model.transformer.set_tread_router(...)`.

## Test runs

- 08_lora_tread: winner-W8A8 dtype + tread 0.5, layers 2..-2. Expect ~20-40%
  step-time cut at 768px; loss curve will spike early then converge (TREAD docs).
- Later optional: FT + TREAD.

## Caveats to inherit from SimpleTuner docs

- ratio > 0.35 impacts convergence; 0.5 is their "balanced" for Flux.
- never route first 2 / last block.
- keep-budget K is per-batch scalar.
- initial loss spike more pronounced for LoRA.
