"""ResNet block and a small classifier on a regional HEALPix domain.

Blocks: HealPixConv (learned 3x3 stencil), HealPixGroupNorm (masked),
HealPixDown (one level coarser), global_pool (masked readout).
Runs on CPU in about a second at level 4; set LEVEL higher on a GPU.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixConv, HealPixDown, HealPixGroupNorm, global_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    """Every NESTED descendant at ``level`` of ``coarse_ids`` (sorted).

    A domain built this way has complete parents at every coarser level,
    which keeps HealPixDown(mode="mean") exact down to ``coarse_level``.
    """
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


class ResBlock(nn.Module):
    """x + conv(act(norm(conv(act(norm(x)))))): a pre-activation ResNet block."""

    def __init__(self, level, cell_ids, channels, groups=4):
        super().__init__()
        self.norm1 = HealPixGroupNorm(groups, channels)
        self.conv1 = HealPixConv(level, channels, channels, kernel_sz=3,
                                 cell_ids=cell_ids, device=DEVICE)
        self.norm2 = HealPixGroupNorm(groups, channels)
        self.conv2 = HealPixConv(level, channels, channels, kernel_sz=3,
                                 cell_ids=cell_ids, device=DEVICE)
        self.act = nn.GELU()

    def forward(self, x, mask):           # x [B, C, N], mask [B, N] bool
        h = self.conv1(self.act(self.norm1(x, mask)))
        h = self.conv2(self.act(self.norm2(h, mask)))
        return x + h


class Pool(nn.Module):
    """HealPixDown(mode="mean") on [B, C, N] values and on the mask."""

    def __init__(self, level, cell_ids):
        super().__init__()
        self.down = HealPixDown(level, mode="mean", cell_ids=cell_ids, device=DEVICE)
        self.cell_ids_out = np.unique(np.asarray(cell_ids) // 4)  # the output cells

    def forward(self, x, mask):
        B, C, N = x.shape
        # HealPixDown takes [N] or [B, N]: fold the channels into the batch
        y, _ = self.down(x.reshape(B * C, N))
        m, _ = self.down(mask.to(x.dtype))     # fraction of valid children
        return y.reshape(B, C, -1), m > 0


class Classifier(nn.Module):
    def __init__(self, level, cell_ids, in_channels, n_classes, width=16):
        super().__init__()
        ids1 = np.asarray(cell_ids)
        self.stem = HealPixConv(level, in_channels + 1, width, kernel_sz=3,
                                cell_ids=ids1, device=DEVICE)
        self.block1 = ResBlock(level, ids1, width)
        self.pool = Pool(level, ids1)
        self.block2 = ResBlock(level - 1, self.pool.cell_ids_out, width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):                  # x [B, C, N], NaN = missing
        mask = torch.isfinite(x).all(dim=1)                     # [B, N]
        x = torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], dim=1)
        h = self.block1(self.stem(x), mask)
        h, mask = self.pool(h, mask)
        h = self.block2(h, mask)
        return self.head(global_pool(h, mask))                  # [B, n_classes]


def main(level=4):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[20, 21, 22, 23], coarse_level=1)
    model = Classifier(level, cell_ids, in_channels=3, n_classes=5).to(DEVICE)

    x = torch.randn(8, 3, len(cell_ids), device=DEVICE)
    x[:, :, :50] = float("nan")                                 # a gap
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.randint(0, 5, (8,), device=DEVICE))
    loss.backward()

    assert logits.shape == (8, 5) and torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"N = {len(cell_ids)} cells, logits {tuple(logits.shape)}, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
