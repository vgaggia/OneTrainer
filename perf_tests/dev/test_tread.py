"""Unit tests for TREAD routing pieces. Run: venv\\Scripts\\python.exe perf_tests\\dev\\test_tread.py"""

import sys

sys.path.insert(0, ".")

import torch

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import _apply_rotary_emb_batched

from modules.util.TreadRouter import TreadRouter

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r = TreadRouter(seed=7, device=dev)

# 1. ratio=0 roundtrip is identity
x = torch.randn(2, 40, 8, device=dev)
info = r.get_mask(x, 0.0)
assert info.keep_len == 40
y = r.end_route(r.start_route(x, info), info, x)
assert torch.equal(y, x), "ratio=0 roundtrip failed"

# 2. ratio=0.5: kept tokens preserved, dropped tokens = original (skip)
info = r.get_mask(x, 0.5)
assert info.keep_len == 20
routed = r.start_route(x, info)
processed = routed * 3.0  # fake block work
y = r.end_route(processed, info, x)
kept_ids = info.ids_shuffle[:, : info.keep_len]
for b in range(2):
    for s in range(40):
        if (kept_ids[b] == s).any():
            assert torch.allclose(y[b, s], x[b, s] * 3.0), "kept token wrong"
        else:
            assert torch.equal(y[b, s], x[b, s]), "dropped token not identity"

# 3. gradient flows to dropped positions
x2 = torch.randn(2, 40, 8, device=dev, requires_grad=True)
info = r.get_mask(x2, 0.5)
out = r.end_route(r.start_route(x2, info) * 2.0, info, x2)
out.sum().backward()
assert (x2.grad.abs() > 0).all(), "gradients missing somewhere"

# 4. batched rope apply == diffusers apply when cos/sin equal across batch
q = torch.randn(2, 40, 4, 16, device=dev, dtype=torch.bfloat16)
cos = torch.randn(40, 16, device=dev)
sin = torch.randn(40, 16, device=dev)
ref = apply_rotary_emb(q, (cos, sin), sequence_dim=1)
got = _apply_rotary_emb_batched(q, cos[None].expand(2, -1, -1), sin[None].expand(2, -1, -1))
assert torch.allclose(ref.float(), got.float(), atol=1e-3), f"rope mismatch {(ref.float()-got.float()).abs().max()}"

# 5. route_rope picks matching rows
rope = torch.arange(40.0, device=dev)[:, None].expand(40, 16).contiguous()
rr = r.route_rope(rope, info, batch_size=2)
assert rr.shape == (2, info.keep_len, 16)
assert torch.equal(rr[0, :, 0], info.ids_shuffle[0, : info.keep_len].float())

print("all TREAD unit tests passed")
