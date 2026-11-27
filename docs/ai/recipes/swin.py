"""Swin-like hierarchical transformer with HealPixWindowAttention.

Stages of pre-norm transformer blocks, x + attn(norm(x)) then
x + mlp(norm(x)), where attn is HealPixWindowAttention (attention inside
NESTED windows of 4**window_level cells; with the default
shift="coarse_centred" the keys of a window reach across its border, so
the receptive field grows at every layer without shifted windows). Between
stages, HealPixDown(mode="mean") halves the resolution and a pointwise
linear layer widens the channels (Swin's patch merging). The norms are
masked HealPixLayerNorm; the readout is a masked global_pool.

Runs on CPU in about a second at level 5; set the level higher on a GPU,
where the attention runs fused Triton kernels.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import (
    HealPixConv,
    HealPixDown,
    HealPixLayerNorm,
    HealPixWindowAttention,
    global_pool,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    """Every NESTED descendant at ``level`` of ``coarse_ids`` (sorted)."""
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


class WindowBlock(nn.Module):
    """Pre-norm transformer block with window attention, on [B, C, N]."""

    def __init__(self, level, cell_ids, dim, heads, window_level, mlp_ratio=2):
        super().__init__()
        self.norm1 = HealPixLayerNorm(dim)            # over the channels of each cell
        self.attn = HealPixWindowAttention(level, dim, heads, window_level,
                                           cell_ids=cell_ids, device=DEVICE)
        self.norm2 = HealPixLayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv1d(dim, mlp_ratio * dim, 1), nn.GELU(),
                                 nn.Conv1d(mlp_ratio * dim, dim, 1))   # per-cell MLP

    def forward(self, x, mask):                       # x [B, C, N], mask [B, N]
        x = x + self.attn(self.norm1(x, mask))
        return x + self.mlp(self.norm2(x, mask))


class SwinLike(nn.Module):
    def __init__(self, level, cell_ids, in_channels, n_out,
                 dims=(32, 64), depths=(2, 2), heads=(2, 4), window_level=2):
        super().__init__()
        ids = np.asarray(cell_ids)
        # embedding: a 3x3 HealPixConv (the "patch embedding" of a cell)
        self.embed = HealPixConv(level, in_channels + 1, dims[0], kernel_sz=3,
                                 cell_ids=ids, device=DEVICE)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        for s, (dim, depth, nh) in enumerate(zip(dims, depths, heads)):
            self.stages.append(nn.ModuleList(
                WindowBlock(level - s, ids, dim, nh, window_level) for _ in range(depth)))
            if s + 1 < len(dims):
                down = HealPixDown(level - s, mode="mean", cell_ids=ids, device=DEVICE)
                self.merges.append(nn.ModuleList([down, nn.Conv1d(dim, dims[s + 1], 1)]))
                ids = np.unique(ids // 4)             # cells of the next stage
        self.norm = HealPixLayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], n_out)

    def forward(self, x):                             # x [B, C, N], NaN = missing
        mask = torch.isfinite(x).all(dim=1)
        h = self.embed(torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], 1))
        for s, blocks in enumerate(self.stages):
            for block in blocks:
                h = block(h, mask)
            if s < len(self.merges):
                down, widen = self.merges[s]
                h = widen(down(h)[0])                 # one level coarser, wider
                mask = down(mask.to(h.dtype))[0] > 0
        return self.head(global_pool(self.norm(h, mask), mask))


def main(level=5):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[16, 17, 18, 19], coarse_level=1)
    model = SwinLike(level, cell_ids, in_channels=3, n_out=4).to(DEVICE)
    x = torch.randn(2, 3, len(cell_ids), device=DEVICE)
    x[:, :, :200] = float("nan")
    out = model(x)
    out.sum().backward()
    assert out.shape == (2, 4) and torch.isfinite(out).all()
    print(f"N = {len(cell_ids)} cells, windows of {4 ** 2} cells, output {tuple(out.shape)}")


if __name__ == "__main__":
    main()
