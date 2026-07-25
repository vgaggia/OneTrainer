import os

import torch


def _mm_8bit_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert a.dtype in [torch.int8, torch.float8_e4m3fn]
    if a.dtype == torch.int8:
        return torch._int_mm(a, b)
    else:
        one = torch.ones(1, device=a.device)
        return torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one)


# The triton kernel autotunes per (M, N, K, stride) key. On heavily bucketed
# datasets running in eager mode (e.g. quantized-weight training, which cannot
# compile), every new bucket shape triggers a full autotune sweep mid-training.
# OT_MM8_NO_TRITON=1 selects the cuBLAS path, which uses shape heuristics
# instead of benchmarking and has stable per-step cost.
if os.environ.get("OT_MM8_NO_TRITON", "0") == "1":
    mm_8bit = _mm_8bit_torch
else:
    try:
        from modules.util.triton_mm_8bit import mm_8bit
    except ImportError as e:
        print(str(e) + ", continuing without triton")
        mm_8bit = _mm_8bit_torch
