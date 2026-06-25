"""Krea 2 SDPA backend / padding-mask probe.

Run inside the exact OneTrainer environment on the target NVIDIA GPU:

    python scripts/krea2_sdpa_gpu_probe.py --dtype bfloat16

The pairwise mask matches the older risky query-valid AND key-valid construction.
The key_only mask matches the safer Krea 2 implementation in modules/model/krea2/mmdit.py.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F


def make_mask(kind: str, real: int, total: int, device: torch.device) -> torch.Tensor:
    valid = torch.arange(total, device=device) < real
    if kind == "pairwise":
        return (valid[:, None] & valid[None, :])[None, None]
    if kind == "key_only":
        return valid[None, None, None, :]
    raise ValueError(kind)


def run_one(backend, kind: str, args) -> None:
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    total = ((args.real_tokens + args.pad_multiple - 1) // args.pad_multiple) * args.pad_multiple
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q = torch.randn(args.batch, args.q_heads, total, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(args.batch, args.kv_heads, total, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = make_mask(kind, args.real_tokens, total, device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.nn.attention.sdpa_kernel([backend]):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            loss = y.float().square().mean()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        finite_out = bool(torch.isfinite(y).all())
        finite_grad = all(bool(torch.isfinite(x.grad).all()) for x in (q, k, v))
        padded_finite = bool(torch.isfinite(y[:, :, args.real_tokens:]).all())
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{backend.name:24s} {kind:9s} OK  out={finite_out} grad={finite_grad} "
              f"pad={padded_finite} peak={peak:.2f} GiB time={elapsed:.3f}s shape={tuple(y.shape)}")
    except Exception as exc:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{backend.name:24s} {kind:9s} FAIL {type(exc).__name__}: {exc} peak={peak:.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-tokens", type=int, default=513)
    parser.add_argument("--pad-multiple", type=int, default=256)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=48)
    parser.add_argument("--kv-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment")
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name())
    for backend in [
        torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        torch.nn.attention.SDPBackend.MATH,
    ]:
        for kind in ("pairwise", "key_only"):
            run_one(backend, kind, args)


if __name__ == "__main__":
    main()
