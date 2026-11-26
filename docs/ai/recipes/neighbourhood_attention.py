"""Neighbourhood-attention transformer for per-cell prediction (segmentation).

Each block is x + attn(norm(x)) then x + mlp(norm(x)), where attn is
HealPixNeighbourhoodAttention: every cell attends to the 3x3 taps of its
gauge-oriented HealPixConv stencil, with a relative position bias shared by
the whole domain. All blocks stay at the input level, so the output has
one value per cell. Masked HealPixLayerNorm, NaN-aware input.

Runs on CPU in about a second at level 5; set the level higher on a GPU,
where the attention runs fused Triton kernels.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixConv, HealPixLayerNorm, HealPixNeighbourhoodAttention

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


class NABlock(nn.Module):
    def __init__(self, level, cell_ids, dim, heads, kernel_sz=3):
        super().__init__()
        self.norm1 = HealPixLayerNorm(dim)
        # n_gauges=1: the output has dim channels (n_gauges=G gives G*dim)
        self.attn = HealPixNeighbourhoodAttention(level, dim, heads, kernel_sz=kernel_sz,
                                                  cell_ids=cell_ids, device=DEVICE)
        self.norm2 = HealPixLayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv1d(dim, 2 * dim, 1), nn.GELU(), nn.Conv1d(2 * dim, dim, 1))

    def forward(self, x, mask):                       # x [B, C, N], mask [B, N]
        x = x + self.attn(self.norm1(x, mask))
        return x + self.mlp(self.norm2(x, mask))


class NASegmenter(nn.Module):
    def __init__(self, level, cell_ids, in_channels, n_classes, dim=32, depth=2, heads=4):
        super().__init__()
        self.embed = HealPixConv(level, in_channels + 1, dim, kernel_sz=3,
                                 cell_ids=cell_ids, device=DEVICE)
        self.blocks = nn.ModuleList(NABlock(level, cell_ids, dim, heads) for _ in range(depth))
        self.norm = HealPixLayerNorm(dim)
        self.head = nn.Conv1d(dim, n_classes, 1)

    def forward(self, x):                             # x [B, C, N], NaN = missing
        mask = torch.isfinite(x).all(dim=1)
        h = self.embed(torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], 1))
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.norm(h, mask))          # [B, n_classes, N] logits


def main(level=5):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[24, 25, 26, 27], coarse_level=1)
    model = NASegmenter(level, cell_ids, in_channels=2, n_classes=3).to(DEVICE)
    x = torch.randn(2, 2, len(cell_ids), device=DEVICE)
    x[:, :, 500:700] = float("nan")
    labels = torch.randint(0, 3, (2, len(cell_ids)), device=DEVICE)
    labels[torch.isnan(x).any(dim=1)] = -100          # no label under the gap
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, labels, ignore_index=-100)
    loss.backward()
    assert logits.shape == (2, 3, len(cell_ids)) and torch.isfinite(loss)
    print(f"N = {len(cell_ids)} cells, logits {tuple(logits.shape)}, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
