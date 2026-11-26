"""U-Net on a regional HEALPix domain with gaps (NaN).

Encoder: HealPixConv + HealPixGroupNorm (masked) at each level, then one
level coarser with a learned strided convolution, HealPixConv(stride=2).
Decoder: HealPixConvTranspose (the adjoint of the strided layer) back to
the finer level, concatenation with the skip connection, HealPixConv.
``sampling="fixed"`` replaces the learned resampling by HealPixDown(mode=
"mean") and HealPixUp (paired with that Down). The validity mask is pooled
with HealPixDown in both variants.
Output: one value per cell of the input domain, [B, C_out, N].
Runs on CPU in about a second at level 5; set the level higher on a GPU.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import (
    HealPixConv,
    HealPixConvTranspose,
    HealPixDown,
    HealPixGroupNorm,
    HealPixUp,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    """Every NESTED descendant at ``level`` of ``coarse_ids`` (sorted)."""
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


class ConvNormAct(nn.Module):
    def __init__(self, level, cell_ids, c_in, c_out, groups=4):
        super().__init__()
        self.conv = HealPixConv(level, c_in, c_out, kernel_sz=3,
                                cell_ids=cell_ids, device=DEVICE)
        self.norm = HealPixGroupNorm(groups, c_out)   # fill="zero": 0 at masked cells
        self.act = nn.GELU()

    def forward(self, x, mask):
        return self.act(self.norm(self.conv(x), mask))


class Resample(nn.Module):
    """One level coarser ("down") or finer ("up"), learned or fixed; channels kept."""

    def __init__(self, level, cell_ids, channels, sampling, pool):
        super().__init__()
        self.learned = sampling == "strided"
        if self.learned:
            # the stencil of level `level`, evaluated at the parent centres
            self.conv = HealPixConv(level, channels, channels, kernel_sz=3, stride=2,
                                    cell_ids=cell_ids, device=DEVICE)
            # its adjoint: parents of `cell_ids` -> `cell_ids`
            self.convT = HealPixConvTranspose(level, channels, channels, kernel_sz=3,
                                              cell_ids=cell_ids, device=DEVICE)
        else:
            self.pool = pool                                            # mean of 4 children
            self.unpool = HealPixUp(level - 1, paired_down=pool, device=DEVICE)  # parent value

    def down(self, x):                          # [B, C, N] -> [B, C, N / 4]
        return self.conv(x) if self.learned else self.pool(x)[0]

    def up(self, x):                            # [B, C, N / 4] -> [B, C, N]
        return self.convT(x) if self.learned else self.unpool(x)[0]


class UNet(nn.Module):
    def __init__(self, level, cell_ids, c_in, c_out, widths=(16, 32, 64), sampling="strided"):
        super().__init__()
        ids = [np.asarray(cell_ids)]
        for _ in widths[1:]:
            ids.append(np.unique(ids[-1] // 4))        # cells one level coarser
        lv = [level - i for i in range(len(widths))]

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()                     # mean Down: the mask (and "fixed")
        self.res = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = c_in + 1                                     # + validity channel
        for i, w in enumerate(widths):
            self.enc.append(ConvNormAct(lv[i], ids[i], c, w))
            c = w
            if i + 1 < len(widths):
                pool = HealPixDown(lv[i], mode="mean", cell_ids=ids[i], device=DEVICE)
                self.pool.append(pool)
                self.res.append(Resample(lv[i], ids[i], w, sampling, pool))
        for i in reversed(range(len(widths) - 1)):
            # after the up step: widths[i] channels (resampled) + widths[i] (skip)
            self.dec.append(ConvNormAct(lv[i], ids[i], 2 * widths[i], widths[i]))
        self.widen = nn.ModuleList(nn.Conv1d(widths[i + 1], widths[i], 1)
                                   for i in reversed(range(len(widths) - 1)))
        self.head = HealPixConv(level, widths[0], c_out, kernel_sz=3,
                                cell_ids=ids[0], device=DEVICE)

    def forward(self, x):                       # x [B, C_in, N], NaN = missing
        mask = torch.isfinite(x).all(dim=1)     # [B, N]
        h = torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], dim=1)
        skips, masks = [], []
        for i, enc in enumerate(self.enc):
            h = enc(h, mask)
            if i < len(self.res):
                skips.append(h)
                masks.append(mask)
                h = self.res[i].down(h)                          # [B, C, N] -> [B, C, N / 4]
                mask = self.pool[i](mask.to(h.dtype))[0] > 0     # any valid child
        for j, (dec, widen) in enumerate(zip(self.dec, self.widen)):
            i = len(self.res) - 1 - j
            h = self.res[i].up(widen(h))                         # fewer channels, then finer
            mask = masks.pop()
            h = dec(torch.cat([h, skips.pop()], dim=1), mask)
        return self.head(h)                     # [B, C_out, N]


def main(level=5):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[4 * 9 + j for j in range(4)], coarse_level=1)
    x = torch.randn(4, 2, len(cell_ids), device=DEVICE)
    x[:, :, 100:300] = float("nan")             # a cloud
    target = torch.randn(4, 1, len(cell_ids), device=DEVICE)
    for sampling in ("strided", "fixed"):
        model = UNet(level, cell_ids, c_in=2, c_out=1, sampling=sampling).to(DEVICE)
        y = model(x)
        loss = nn.functional.mse_loss(y, target)
        loss.backward()
        assert y.shape == (4, 1, len(cell_ids)) and torch.isfinite(y).all()
        print(f"{sampling}: N = {len(cell_ids)} cells, output {tuple(y.shape)}, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
