"""Token routing for efficient diffusion training (TREAD, arXiv 2501.04765).

During training only, a random subset of image tokens bypasses the transformer
blocks between a start and end layer. Dropped positions are restored from the
pre-route hidden states (an identity skip connection, so gradients still flow to
earlier layers). Inference is completely unaffected.

The router is model-agnostic; the model's forward decides where routes start and
end and what counts as an image token. Implementation written for OneTrainer,
mechanism after the TREAD paper and SimpleTuner's production integration.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RouteInfo:
    ids_shuffle: Tensor  # (B, S) kept-first permutation of token indices
    ids_restore: Tensor  # (B, S) inverse permutation
    keep_len: int        # K = number of tokens that continue through the blocks


class TreadRouter:
    def __init__(self, seed: int, device: torch.device):
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def get_mask(self, x: Tensor, dropout_ratio: float) -> RouteInfo:
        b, s, _ = x.shape
        keep_len = max(1, s - int(round(s * dropout_ratio)))
        scores = torch.rand(b, s, device=x.device, generator=self.generator)
        ids_shuffle = torch.argsort(scores, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        return RouteInfo(ids_shuffle=ids_shuffle, ids_restore=ids_restore, keep_len=keep_len)

    @staticmethod
    def _gather(x: Tensor, ids: Tensor) -> Tensor:
        return torch.take_along_dim(x, ids[..., None].expand(-1, -1, x.shape[-1]), dim=1)

    def start_route(self, x: Tensor, info: RouteInfo) -> Tensor:
        """(B, S, D) -> (B, K, D): kept tokens only, shuffled order."""
        return self._gather(x, info.ids_shuffle)[:, :info.keep_len]

    def end_route(self, x_routed: Tensor, info: RouteInfo, original_x: Tensor) -> Tensor:
        """Merge routed tokens back; dropped positions come from original_x (skip)."""
        shuffled_orig = self._gather(original_x, info.ids_shuffle)
        merged = torch.cat([x_routed, shuffled_orig[:, info.keep_len:]], dim=1)
        return self._gather(merged, info.ids_restore)

    def route_rope(self, rope_part: Tensor, info: RouteInfo, batch_size: int) -> Tensor:
        """Gather per-sample rotary rows for kept tokens: (S, D) -> (B, K, D)."""
        expanded = rope_part[None].expand(batch_size, -1, -1)
        return self._gather(expanded, info.ids_shuffle)[:, :info.keep_len]
