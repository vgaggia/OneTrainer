"""Unit tests for true quantized-weight training (int8 + SR fused Adafactor).
Run: venv\\Scripts\\python.exe perf_tests\\dev\\test_qwt.py"""

import sys

sys.path.insert(0, ".")

import torch

import modules.util.quantization_util  # noqa: F401 (break circular import)
from modules.module.quantized.LinearW8A8 import LinearW8A8, int8_weight_grad
from modules.util.quantized_weight_training import FusedQuantizedAdafactor

dev = torch.device("cuda")
torch.manual_seed(0)

# 1. int8 weight-grad vs bf16 autograd reference
m, k, n = 4601, 6144, 4096  # deliberately non-multiple-of-8 token count
ref = torch.nn.Linear(k, n, bias=False, dtype=torch.bfloat16, device=dev)
x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
g = torch.randn(m, n, dtype=torch.bfloat16, device=dev)
ref.weight.requires_grad_(True)
ref(x).backward(g)
ref_grad = ref.weight.grad.float()
got = int8_weight_grad(g, x)
rel = ((got - ref_grad).norm() / ref_grad.norm()).item()
print(f"wgrad rel_err={rel:.5f}")
assert rel < 0.05, f"weight grad too inaccurate: {rel}"

# 2. toy convergence: can SR-int8 training actually fit a target?
torch.manual_seed(1)
kk, nn_ = 512, 512
target = torch.randn(nn_, kk, device=dev) * 0.02
q = LinearW8A8(torch.int8, kk, nn_, bias=False).to(dev)
with torch.no_grad():
    q.weight.copy_(torch.randn(nn_, kk) * 0.02)
q.compute_dtype = torch.bfloat16
q.quantize(device=dev)
q.qwt_updater = FusedQuantizedAdafactor(q, lr=1e-3, clip_grad_norm=1.0)
q.train()

losses = []
for step in range(600):
    xb = torch.randn(64, kk, dtype=torch.bfloat16, device=dev, requires_grad=True)
    y_t = xb.float() @ target.T
    y = q(xb)
    loss = torch.nn.functional.mse_loss(y.float(), y_t)
    loss.backward()  # fused update happens inside
    losses.append(loss.item())
first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
print(f"toy convergence: loss {first:.5f} -> {last:.5f}")
assert last < first * 0.5, "SR-int8 training failed to reduce loss"

# 3. sub-ULP drift: with tiny lr, weights should still move in expectation
torch.manual_seed(2)
q2 = LinearW8A8(torch.int8, 256, 256, bias=False).to(dev)
with torch.no_grad():
    q2.weight.copy_(torch.randn(256, 256) * 0.02)
q2.compute_dtype = torch.bfloat16
q2.quantize(device=dev)
w_before = (q2.weight.float() * q2.scale).clone()
q2.qwt_updater = FusedQuantizedAdafactor(q2, lr=2e-5, clip_grad_norm=1.0)
q2.train()
tgt = torch.randn(256, 256, device=dev) * 0.02
for step in range(200):
    xb = torch.randn(64, 256, dtype=torch.bfloat16, device=dev, requires_grad=True)
    y = q2(xb)
    torch.nn.functional.mse_loss(y.float(), xb.float() @ tgt.T).backward()
w_after = q2.weight.float() * q2.scale
moved = (w_after - w_before).abs().mean().item()
print(f"sub-ULP drift after 200 steps @ lr 2e-5: mean |dW| = {moved:.7f}")
assert moved > 0, "weights never moved"

print("all QWT unit tests passed")
