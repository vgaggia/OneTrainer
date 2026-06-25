"""CPU smoke test for the Krea 2 predict/sample assembly.

Exercises the real pack/unpack, build_pos_mask and a tiny SingleStreamDiT forward
(the pieces most prone to silent shape bugs) without the 26 GB checkpoint or a GPU.

Run: ``python -m modules.model.krea2.smoke_test``

This is only a CPU/shape gate. The full Krea 2 stack still needs a CUDA gate for
backend selection, quantized LoRA gradients and external LoRA load/apply parity.
"""

from modules.model.krea2.mmdit import SingleMMDiTConfig, SingleStreamDiT
from modules.model.Krea2Model import Krea2Model

import torch

# Tiny config: headdim = 32/2 = 16 -> rope axes [16-12, 6, 6] = [4,6,6] (sum 16, all even).
TINY = SingleMMDiTConfig(
    features=32, tdim=16, txtdim=16, heads=2, kvheads=1, multiplier=2,
    layers=2, patch=2, channels=4, txtheads=2, txtkvheads=1, txtlayers=3,
)


def _assemble(net, latent, context_flat, text_mask):
    """Replicate the predict/sample assembly against the real helpers."""
    b = latent.shape[0]
    img = Krea2Model.pack_latents(latent)  # (B, L_img, channels*patch^2)
    context = context_flat.reshape(b, context_flat.shape[1], net.config.txtlayers, -1)
    pos, mask = Krea2Model.build_pos_mask(
        b, text_mask, latent.shape[-2], latent.shape[-1], net.config.patch, latent.device,
    )
    t = torch.full((b,), 0.5)
    out = net(img=img, context=context, t=t, pos=pos, mask=mask)  # (B, L_img, channels*patch^2)
    return out, img


def demo():
    torch.manual_seed(0)
    net = SingleStreamDiT(TINY).float()
    b, c, h, w = 2, TINY.channels, 8, 8  # latent spatial 8x8
    l_txt = 5

    # --- pack/unpack are exact inverses ---
    latent = torch.randn(b, c, 1, h, w)
    packed = Krea2Model.pack_latents(latent)
    assert packed.shape == (b, (h // 2) * (w // 2), c * 4), packed.shape
    round_trip = Krea2Model.unpack_latents(packed, height=h, width=w)
    assert torch.allclose(round_trip, latent, atol=1e-5), "pack/unpack not invertible"

    # --- build_pos_mask shapes ---
    full_mask = torch.ones(b, l_txt, dtype=torch.bool)
    pos, mask = Krea2Model.build_pos_mask(b, full_mask, h, w, TINY.patch, latent.device)
    l_img = (h // 2) * (w // 2)
    assert pos.shape == (b, l_txt + l_img, 3), pos.shape
    assert mask.shape == (b, l_txt + l_img) and mask.dtype == torch.bool

    # --- all-valid mask: forward + backward must be finite ---
    context_flat = torch.randn(b, l_txt, TINY.txtlayers * TINY.txtdim, requires_grad=False)
    latent_a = torch.randn(b, c, 1, h, w)
    out, _ = _assemble(net, latent_a, context_flat, full_mask)
    assert out.shape == (b, l_img, c * TINY.patch**2), out.shape
    pred = Krea2Model.unpack_latents(out, height=h, width=w)
    noise = torch.randn_like(latent_a)
    loss = torch.nn.functional.mse_loss(pred, noise - latent_a)
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed to the transformer"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"

    # --- padded text mask: key-only masking keeps image-token output finite ---
    net.zero_grad()
    pad_mask = full_mask.clone()
    pad_mask[:, 3:] = False  # only the first 3 text tokens are real
    out_b, _ = _assemble(net, torch.randn(b, c, 1, h, w), torch.randn(b, l_txt, TINY.txtlayers * TINY.txtdim), pad_mask)
    assert torch.isfinite(out_b).all(), "image-token output not finite with padded text mask"

    print("krea2 smoke test OK: pack/unpack invertible, assembly shapes correct, "
          "finite loss+grads, finite image output with padded text mask")


if __name__ == "__main__":
    demo()
