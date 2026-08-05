
import os

from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin
from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin
from modules.util.mm_8bit import mm_8bit as mm_8bit
from modules.util.quantization_util import (
    dequantize,
    quantize_fp8_axiswise,
    quantize_fp8_tensorwise,
    quantize_int8_axiswise,
    quantize_int8_tensorwise,
)

import torch
from torch import Tensor, nn

# Weight scales are per-output-channel (axiswise, DeepSeek-style fine-grained scaling)
# by default. Set OT_W8A8_TENSORWISE=1 to restore the old single-scale-per-tensor
# behavior. The scale buffer always has shape (out_features, 1); tensorwise mode just
# fills it with one repeated value, so all compute paths below are shared.
_TENSORWISE_WEIGHTS = os.environ.get("OT_W8A8_TENSORWISE", "0") == "1"


@torch.no_grad()
def int8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    # x_scale: (m, 1); weight_scale: (out, 1) applied per output column.
    res_scaled = res.float().mul_(x_scale).mul_(weight_scale.view(1, -1)).to(compute_dtype)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def fp8_forward_tokenwise(x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
    x_8, x_scale = quantize_fp8_axiswise(x, dim=-1)
    one = torch.tensor(1.0, device=x.device)
    res = torch._scaled_mm(x_8, weight.T, scale_a=one, scale_b=one, out_dtype=torch.float)
    #scaling in the epilogue is much faster than scaled by _scaled_mm:
    res_scaled = res.mul_(x_scale).mul_(weight_scale.view(1, -1)).to(compute_dtype)
    if bias is not None:
        res_scaled.add_(bias)
    return res_scaled

@torch.no_grad()
def int8_weight_grad(output: Tensor, x: Tensor) -> Tensor:
    """grad_W = dY^T @ X in int8. dY and X are quantized per-COLUMN (dim=0) so both
    scales are constant along the token contraction and factor out post-GEMM."""
    dy_8, dy_scale = quantize_int8_axiswise(output, dim=0)   # scales (1, n)
    x_8, x_scale = quantize_int8_axiswise(x, dim=0)          # scales (1, k)
    # the token count becomes the GEMM contraction dim; cuBLAS int8 requires it to be
    # a multiple of 16 (bucketed datasets produce arbitrary counts). Zero token-rows
    # contribute nothing to dY^T @ X, so padding is exact.
    pad = (-dy_8.shape[0]) % 16
    if pad:
        dy_8 = torch.nn.functional.pad(dy_8, (0, 0, 0, pad))
        x_8 = torch.nn.functional.pad(x_8, (0, 0, 0, pad))
    res = mm_8bit(dy_8.t().contiguous(), x_8)                # (n, k) int32
    return res.float().mul_(dy_scale.t()).mul_(x_scale)      # fp32 (n, k)


@torch.no_grad()
def int8_backward_axiswise(output: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
    out_dtype = output.dtype
    # fold the per-output-channel weight scale into the grad before quantization;
    # it multiplies the contraction dim of (dY @ W_q), so it cannot be applied post-GEMM
    output = output.float().mul(weight_scale.view(1, -1))
    output_8, output_scale = quantize_int8_axiswise(output, dim=-1)
    #almost always, grad outputs are already contiguous and this is a no-op. But there are some grad outputs from SDXL that are non-contiguous:
    output_8 = output_8.contiguous()
    # cuBLAS int8 NN gemm (torch._int_mm) rejects token counts that aren't a multiple of
    # 32 when the contraction dim (out_features) is < 128 (measured empirically on
    # sm_120 / cu130; TN and K >= 128 are unaffected). Only tiny layers like the final
    # projection hit this. Zero rows are exact: they contribute zero output rows, which
    # are sliced off below. Scales are per-real-row and computed before padding.
    n_tokens = output_8.shape[0]
    pad = (-n_tokens) % 32 if weight.shape[0] < 128 else 0
    if pad:
        output_8 = torch.nn.functional.pad(output_8, (0, 0, 0, pad))
    mm_res = mm_8bit(output_8, weight)
    if pad:
        mm_res = mm_res[:n_tokens]
    return mm_res.float().mul_(output_scale).to(out_dtype)

@torch.no_grad()
def fp8_backward_axiswise(output: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
    out_dtype = output.dtype
    output = output.float().mul(weight_scale.view(1, -1))
    output_8, output_scale = quantize_fp8_axiswise(output, dim=-1)
    mm_res = mm_8bit(output_8.contiguous(), weight)
    return mm_res.float().mul_(output_scale).to(out_dtype)


# Compiled variants for quantized-weight training: the model graph cannot compile in
# that mode (fused updates run inside autograd backward), but these helpers are pure
# tensor functions and fuse well in isolation. dynamic=True keeps one graph across
# bucket token counts.
_int8_forward_tokenwise_c = torch.compile(int8_forward_tokenwise, dynamic=True)
_int8_backward_axiswise_c = torch.compile(int8_backward_axiswise, dynamic=True)
_int8_weight_grad_c = torch.compile(int8_weight_grad, dynamic=True)


class LinearInt8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype, updater=None) -> Tensor:
        if updater is not None:
            ctx.save_for_backward(weight, weight_scale, x)
        else:
            ctx.save_for_backward(weight, weight_scale)
        ctx.updater = updater
        ctx.bias_dtype = None if bias is None else bias.dtype
        forward_fn = _int8_forward_tokenwise_c if updater is not None else int8_forward_tokenwise
        return forward_fn(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise NotImplementedError(
                "Int W8A8 weights are frozen and cannot receive gradients. "
                "Use a non-quantized weight dtype to train the quantized layers themselves.")

        if ctx.updater is not None:
            weight, weight_scale, x = ctx.saved_tensors
            # grad_x BEFORE the weight update: the update must use the same weights
            # that produced the forward, and grad_x must match the forward weights too
            grad_x = _int8_backward_axiswise_c(output, weight, weight_scale) if ctx.needs_input_grad[0] else None
            ctx.updater.step(_int8_weight_grad_c(output, x))
        else:
            weight, weight_scale = ctx.saved_tensors
            grad_x = int8_backward_axiswise(output, weight, weight_scale) if ctx.needs_input_grad[0] else None
        grad_bias = output.float().sum(dim=0).to(ctx.bias_dtype) if ctx.needs_input_grad[3] else None
        return grad_x, None, None, grad_bias, None, None

class LinearFp8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, weight_scale: Tensor, bias: Tensor | None, compute_dtype: torch.dtype) -> Tensor:
        ctx.save_for_backward(weight, weight_scale)
        ctx.bias_dtype = None if bias is None else bias.dtype
        return fp8_forward_tokenwise(x, weight, weight_scale, bias, compute_dtype)

    @staticmethod
    def backward(ctx, output: Tensor):
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise NotImplementedError(
                "Float W8A8 weights are frozen and cannot receive gradients. "
                "Use a non-quantized weight dtype to train the quantized layers themselves.")

        weight, weight_scale = ctx.saved_tensors
        grad_x = fp8_backward_axiswise(output, weight, weight_scale) if ctx.needs_input_grad[0] else None
        grad_bias = output.float().sum(dim=0).to(ctx.bias_dtype) if ctx.needs_input_grad[3] else None
        return grad_x, None, None, grad_bias, None

class LinearW8A8(
    nn.Linear,
    QuantizedModuleMixin,
    QuantizedLinearMixin,
):
    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert dtype in [torch.int8, torch.float8_e4m3fn]
        self._dtype = dtype

        self.__is_quantized = False
        self.compute_dtype = None
        self.qwt_updater = None  # set by enable_quantized_weight_training for true int8 training
        self.register_buffer("scale", torch.ones(self.out_features, 1, dtype=torch.float32))

    def original_weight_shape(self) -> tuple[int, ...]:
        return self.weight.shape

    def unquantized_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return dequantize(self.weight.detach(), self.scale).to(dtype)

    @torch.no_grad()
    def quantize(self, device: torch.device | None = None):
        if self.__is_quantized:
            return
        self.__is_quantized = True

        weight = self.weight.detach()

        # Internal backups store LinearW8A8 exactly as an already-quantized weight plus
        # its scale sidecar. The loader reconstructs this class before assigning those
        # tensors, so dtype is the reliable indication that no quantization is needed.
        # Re-quantizing the int8 codes would replace the saved (typically ~1e-3) scales
        # with scales near 1.0 and catastrophically inflate the restored weights.
        if weight.dtype == self._dtype:
            self.requires_grad_(False)
            # Scale is optimizer state as well as quantization metadata during QWT.
            # Keep it in fp32 even when a legacy backup serialized it in the model's
            # bf16 train dtype.
            self.scale.data = self.scale.data.to(dtype=torch.float32)
            return

        orig_device = weight.device
        if device is not None:
            weight = weight.to(device=device)
        if self._dtype == torch.int8:
            if _TENSORWISE_WEIGHTS:
                weight, scale = quantize_int8_tensorwise(weight)
            else:
                weight, scale = quantize_int8_axiswise(weight, dim=1)
        else:
            if _TENSORWISE_WEIGHTS:
                weight, scale = quantize_fp8_tensorwise(weight)
            else:
                weight, scale = quantize_fp8_axiswise(weight, dim=1)

        if device is not None:
            weight = weight.to(device=orig_device)

        self.requires_grad_(False)
        self.weight.data = weight

        self.scale.data = scale.to(device=self.scale.device, dtype=torch.float32)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        assert not self.weight.requires_grad
        assert self.__is_quantized
        x = x_orig.reshape(-1, x_orig.shape[-1])

        # torch._int_mm requires K/N to be multiples of 8, torch._scaled_mm multiples of 16.
        # Some models (e.g. Krea 2) contain tiny Linear layers (in_features=12) that violate
        # this; route them through the dequantize fallback below.
        aligned = x.shape[1] % 16 == 0 and self.weight.shape[0] % 16 == 0

        if x.shape[0] > 16 and aligned:
            if self._dtype == torch.int8:
                updater = self.qwt_updater if (self.training and torch.is_grad_enabled()) else None
                y = LinearInt8Function.apply(x, self.weight, self.scale, self.bias, self.compute_dtype, updater)
            else:
                y = LinearFp8Function.apply(x, self.weight, self.scale, self.bias, self.compute_dtype)
        else:
            w = dequantize(self.weight.detach(), self.scale)
            y = torch.nn.functional.linear(x, w, self.bias)

        return y.reshape(x_orig.shape[:-1] + (y.shape[-1], ))

def run_benchmark(fn, desc, steps=10000, warmup=500, compile=False):
    if compile:
        fn = torch.compile(fn, fullgraph=True)
    from tqdm import tqdm
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in tqdm(range(steps), desc=desc):
        fn()
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_int8(m, k, n, device = 'cuda'):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.int8)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.int8)
    w_8 = torch.ones (n,k, device=device, dtype=torch.int8)
    w_scale = torch.ones(1, device=device)


    run_benchmark(lambda: torch._int_mm(x_8, w_8.T), "torch mm int")
    run_benchmark(lambda: mm_8bit(x_8, w_8.T), "triton mm int")
    def torch_backward(a, b):
        torch._int_mm(a, b.T.contiguous().T)
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward int8")
    run_benchmark(lambda: mm_8bit(y_8, w_8), "triton mm backward int8")

    run_benchmark(lambda: int8_forward_tokenwise(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "torch forward int", compile=True)
    run_benchmark(lambda: int8_backward_axiswise(y, w_8, w_scale), "triton backward int", compile=True)


@torch.no_grad()
def benchmark_fp8(m, k, n, device = 'cuda'):
    x   = torch.randn(m,k, device=device, dtype=torch.bfloat16)
    x_8 = torch.ones (m,k, device=device, dtype=torch.float8_e4m3fn)
    y   = torch.randn(m,n, device=device, dtype=torch.bfloat16)
    y_8 = torch.ones (m,n, device=device, dtype=torch.float8_e4m3fn)
    w_8 = torch.ones (n,k, device=device, dtype=torch.float8_e4m3fn)
    w_scale = torch.ones(1, device=device, dtype=torch.bfloat16)
    one_scale = torch.ones(1, device=device)

    run_benchmark(lambda: torch._scaled_mm(x_8, w_8.T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float()), "torch mm fp8")
    run_benchmark(lambda: mm_8bit(x_8, w_8.T), "triton mm fp8")
    def torch_backward(a, b):
        torch._scaled_mm(a, b.T.contiguous().T, out_dtype=torch.bfloat16, scale_a=one_scale.float(), scale_b=w_scale.float())
    run_benchmark(lambda: torch_backward(y_8, w_8), "torch mm backward fp8")
    run_benchmark(lambda: mm_8bit(y_8, w_8), "triton mm backward fp8")
    run_benchmark(lambda: fp8_forward_tokenwise(x, w_8, w_scale, bias=None, compute_dtype=torch.bfloat16), "torch forward fp8", compile=True)
    run_benchmark(lambda: fp8_backward_axiswise(y, w_8, w_scale), "triton backward fp8", compile=True)


if __name__ == "__main__":
    benchmark_int8(2 * 1024 + 50, 3072, 3072 + 16)
    benchmark_fp8(2 * 1024 + 50, 3072, 3072 + 16)
