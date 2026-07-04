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

import torch
from torch import nn


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
        weight = self.module.weight
        scale = self.module.scale
        self.step_count += 1

        if self.row is None:
            self.row = torch.zeros(grad.shape[0], dtype=torch.float32, device=grad.device)
            self.col = torch.zeros(grad.shape[1], dtype=torch.float32, device=grad.device)

        if self.clip_grad_norm is not None:
            norm = grad.norm()
            grad.mul_(torch.clamp(self.clip_grad_norm / (norm + 1e-6), max=1.0))

        beta2t = 1.0 - self.step_count ** self.decay_rate
        sq_row = grad.pow(2).mean(dim=1)  # [out]
        sq_col = grad.pow(2).mean(dim=0)  # [in]
        self.row.mul_(beta2t).add_(sq_row, alpha=1.0 - beta2t)
        self.col.mul_(beta2t).add_(sq_col, alpha=1.0 - beta2t)

        # u = grad / sqrt(outer(R, C) / mean(R)); computed via two broadcasts
        r = (self.row / self.row.mean().clamp(min=self.eps1)).clamp(min=self.eps1).rsqrt()  # [out]
        c = self.col.clamp(min=self.eps1).rsqrt()  # [in]
        grad.mul_(r[:, None]).mul_(c[None, :])  # grad is now u

        rms_u = grad.pow(2).mean().sqrt()
        grad.div_(torch.clamp(rms_u / self.clip_threshold, min=1.0))

        # dequantize, update, re-quantize with stochastic rounding, refresh scales
        w = weight.data.float().mul_(scale)  # fp32 [out, in]
        w.add_(grad, alpha=-self.lr)

        new_scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
        w.div_(new_scale)  # w is now the real-valued quantization target in [-127, 127]
        low = w.floor()
        w.sub_(low)  # w is now the fractional part
        low.add_(torch.rand_like(w) < w)  # stochastic rounding
        weight.data.copy_(low.clamp_(-127.0, 127.0).to(torch.int8))
        scale.copy_(new_scale)


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
