"""Sanity test for per-channel W8A8 scales: forward/backward error vs bf16 reference,
plus a micro-benchmark. Run: venv\\Scripts\\python.exe perf_tests\\dev\\test_w8a8_perchannel.py"""

import sys
import time

sys.path.insert(0, ".")

import torch

import modules.util.quantization_util  # noqa: F401  (break circular import)
from modules.module.quantized.LinearW8A8 import LinearW8A8


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def run(dtype, name):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m, k, n = 4608, 6144, 16384

    ref = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device=dev)
    # inject channel-scale outliers like real transformer weights
    with torch.no_grad():
        ref.weight[: n // 64] *= 50.0

    q = LinearW8A8(dtype, k, n, bias=False).to(dev)
    with torch.no_grad():
        q.weight.copy_(ref.weight)
    q.compute_dtype = torch.bfloat16
    q.quantize(device=dev)

    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    y_ref = ref(x)
    y_q = q(x2)
    fwd = rel_err(y_q.float(), y_ref.float())

    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_q.backward(g)
    bwd = rel_err(x2.grad.float(), x.grad.float())

    # micro-benchmark forward
    for _ in range(10):
        q(x2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        q(x2)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 50 * 1000
    print(f"{name}: fwd rel_err={fwd:.5f} bwd rel_err={bwd:.5f} fwd_time={dt:.3f}ms")
    return fwd, bwd


if __name__ == "__main__":
    import os
    print(f"tensorwise mode: {os.environ.get('OT_W8A8_TENSORWISE', '0')}")
    run(torch.int8, "int8")
    run(torch.float8_e4m3fn, "fp8 ")
