"""U-Net on a regional HEALPix domain with gaps (NaN).

Encoder: HealPixConv + HealPixGroupNorm (masked), HealPixDown(mode="mean").
Decoder: HealPixUp (paired with the Down), concatenation with the skip connection, HealPixConv.
Output: one value per cell of the input domain, [B, C_out, N].
Runs on CPU in a few seconds at level 5; set LEVEL higher on a GPU.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixConv, HealPixDown, HealPixGroupNorm, HealPixUp

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


class UNet(nn.Module):
    def __init__(self, level, cell_ids, c_in, c_out, widths=(16, 32, 64)):
        super().__init__()
        ids = [np.asarray(cell_ids)]
        for _ in widths[1:]:
            ids.append(np.unique(ids[-1] // 4))        # cells one level coarser
        lv = [level - i for i in range(len(widths))]

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = c_in + 1                                     # + validity channel
        for i, w in enumerate(widths):
            self.enc.append(ConvNormAct(lv[i], ids[i], c, w))
            c = w
            if i + 1 < len(widths):
                self.down.append(HealPixDown(lv[i], mode="mean", cell_ids=ids[i],
                                             device=DEVICE))
        for i in reversed(range(len(widths) - 1)):
            # paired with the mean Down of the same level: every child gets the
            # value of its parent (HealPixUp(lv[i + 1], cell_ids=ids[i + 1])
            # would interpolate with the Gaussian transpose instead)
            self.up.append(HealPixUp(lv[i + 1], paired_down=self.down[i], device=DEVICE))
            self.dec.append(ConvNormAct(lv[i], ids[i], widths[i + 1] + widths[i], widths[i]))
        self.head = HealPixConv(level, widths[0], c_out, kernel_sz=3,
                                cell_ids=ids[0], device=DEVICE)

    def forward(self, x):                       # x [B, C_in, N], NaN = missing
        mask = torch.isfinite(x).all(dim=1)     # [B, N]
        h = torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], dim=1)
        skips, masks = [], []
        for i, enc in enumerate(self.enc):
            h = enc(h, mask)
            if i < len(self.down):
                skips.append(h)
                masks.append(mask)
                h, _ = self.down[i](h)                 # [B, C, N] -> [B, C, N / 4]
                frac, _ = self.down[i](mask.to(h.dtype))   # valid fraction of the 4 children
                mask = frac > 0
        for up, dec in zip(self.up, self.dec):
            h, _ = up(h)
            mask = masks.pop()
            h = dec(torch.cat([h, skips.pop()], dim=1), mask)
        return self.head(h)                     # [B, C_out, N]


def main(level=5):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[4 * 9 + j for j in range(4)], coarse_level=1)
    model = UNet(level, cell_ids, c_in=2, c_out=1).to(DEVICE)

    x = torch.randn(4, 2, len(cell_ids), device=DEVICE)
    x[:, :, 100:300] = float("nan")             # a cloud
    target = torch.randn(4, 1, len(cell_ids), device=DEVICE)
    y = model(x)
    loss = nn.functional.mse_loss(y, target)
    loss.backward()

    assert y.shape == (4, 1, len(cell_ids)) and torch.isfinite(y).all()
    print(f"N = {len(cell_ids)} cells, output {tuple(y.shape)}, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
