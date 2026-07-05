"""True quantized-weight training: gradients into int8 weights with stochastic rounding.

Quantized Linear weights are normally frozen (their forward detaches them). This module
attaches a fused per-layer Adafactor updater to LinearW8A8 layers: the weight gradient is
computed in int8 inside the autograd backward, the Adafactor update is applied to the
dequantized weight, and the result is re-quantized with STOCHASTIC ROUNDING so that
sub-ULP updates survive in expectation (same principle as OneTrainer's bf16 SR training,
int8 per-channel ULP is ~0.8% of the channel max vs bf16's fixed 0.39%).

Design constraints (validated at setup):
- FINE_TUNE only, gradient_accumulation_steps == 1 (updates are fused into backward;
  there is no gradient buffer to accumulate into, and int8 tensors cannot hold grads).
- INT_W8A8 only (fp8 e4m3's 6-12% relative ULP is too coarse for SR training).
- v1 reads a constant learning rate from the config (CONSTANT scheduler).

Optimizer state is factored Adafactor (one fp32 row + col vector per matrix, ~50KB per
layer), so the whole 12.9B-weight update machinery adds no meaningful VRAM.
"""

import os

import torch
from torch import nn

_EAGER = os.environ.get("OT_QWT_EAGER", "0") == "1"


def _update_math(
        grad: torch.Tensor,        # fp32 (n, k), consumed
        weight: torch.Tensor,      # int8 (n, k), mutated
        scale: torch.Tensor,       # fp32 (n, 1), mutated
        row: torch.Tensor,         # fp32 (n,), mutated
        col: torch.Tensor,         # fp32 (k,), mutated
        beta2t: torch.Tensor,      # 0-dim fp32 (tensor so shapes stay static under compile)
        clip_norm: torch.Tensor,   # 0-dim fp32
        clip_threshold: torch.Tensor,  # 0-dim fp32
        lr: torch.Tensor,          # 0-dim fp32
        eps1: float,
):
    norm = grad.norm()
    grad = grad * torch.clamp(clip_norm / (norm + 1e-6), max=1.0)

    sq = grad * grad
    row.mul_(beta2t).add_(sq.mean(dim=1) * (1.0 - beta2t))
    col.mul_(beta2t).add_(sq.mean(dim=0) * (1.0 - beta2t))

    r = (row / row.mean().clamp(min=eps1)).clamp(min=eps1).rsqrt()
    c = col.clamp(min=eps1).rsqrt()
    u = grad * r[:, None] * c[None, :]
    rms_u = (u * u).mean().sqrt()
    u = u / torch.clamp(rms_u / clip_threshold, min=1.0)

    w = weight.float() * scale - lr * u
    new_scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    v = w / new_scale
    low = v.floor()
    q = low + (torch.rand_like(v) < (v - low))
    weight.copy_(q.clamp_(-127.0, 127.0).to(torch.int8))
    scale.copy_(new_scale)


_update_math_compiled = _update_math if _EAGER else torch.compile(_update_math, dynamic=False)


class FusedQuantizedAdafactor:
    """Per-layer Adafactor (factored second moment, no momentum) with SR re-quantization.

    Update rule follows the standard Adafactor formulation: beta2_t = 1 - step^-0.8,
    factored V ~ outer(R, C) / mean(R), update RMS-clipped at clip_threshold.
    """

    def __init__(
            self,
            module: nn.Module,  # LinearW8A8, weight int8 [out, in], scale fp32 [out, 1]
            lr: float,
            clip_grad_norm: float | None,
            eps1: float = 1e-30,
            eps2: float = 1e-3,
            clip_threshold: float = 1.0,
            decay_rate: float = -0.8,
    ):
        self.module = module
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.eps1 = eps1
        self.eps2 = eps2
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.step_count = 0
        self.row: torch.Tensor | None = None  # fp32 [out]
        self.col: torch.Tensor | None = None  # fp32 [in]

    @torch.no_grad()
    def step(self, grad: torch.Tensor):
        """grad: fp32 [out, in], consumed (may be mutated)."""
        self.step_count += 1

        if self.row is None:
            dev = grad.device
            self.row = torch.zeros(grad.shape[0], dtype=torch.float32, device=dev)
            self.col = torch.zeros(grad.shape[1], dtype=torch.float32, device=dev)
            # scalar staging tensors: passing python floats would recompile every step
            self._beta2t = torch.zeros((), dtype=torch.float32, device=dev)
            self._clip_norm = torch.tensor(
                self.clip_grad_norm if self.clip_grad_norm is not None else float("inf"),
                dtype=torch.float32, device=dev)
            self._clip_threshold = torch.tensor(self.clip_threshold, dtype=torch.float32, device=dev)
            self._lr = torch.tensor(self.lr, dtype=torch.float32, device=dev)

        self._beta2t.fill_(1.0 - self.step_count ** self.decay_rate)

        _update_math_compiled(
            grad, self.module.weight.data, self.module.scale,
            self.row, self.col,
            self._beta2t, self._clip_norm, self._clip_threshold, self._lr,
            self.eps1,
        )


def enable_quantized_weight_training(root: nn.Module, config) -> int:
    """Attach fused updaters to every trainable-eligible LinearW8A8 under root.
    Returns the number of layers enabled."""
    from modules.module.quantized.LinearW8A8 import LinearW8A8
    from modules.util.enum.TrainingMethod import TrainingMethod

    if config.training_method != TrainingMethod.FINE_TUNE:
        raise ValueError("quantized_weight_training requires the FINE_TUNE training method")
    if config.gradient_accumulation_steps != 1:
        raise ValueError(
            "quantized_weight_training fuses updates into the backward pass and requires "
            "gradient_accumulation_steps == 1 (there is no gradient buffer to accumulate into)")
    if config.compile:
        raise ValueError(
            "quantized_weight_training is currently incompatible with compile=True: the fused "
            "update runs Python side effects inside autograd backward, which fullgraph tracing "
            "cannot handle. Disable compile for quantized-weight training runs.")

    count = 0
    for module in root.modules():
        if isinstance(module, LinearW8A8):
            if module._dtype != torch.int8:
                raise ValueError(
                    "quantized_weight_training supports INT_W8A8 only; fp8 e4m3's relative "
                    "ULP is too coarse for stochastic-rounding weight updates")
            module.qwt_updater = FusedQuantizedAdafactor(
                module,
                lr=config.learning_rate,
                clip_grad_norm=config.clip_grad_norm,
            )
            count += 1
    print(f"quantized_weight_training: {count} int8 Linear layers now receive fused SR updates")
    return count
