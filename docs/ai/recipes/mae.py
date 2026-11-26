"""Masked autoencoder (ViT / MAE) with HealPixPatches tokens.

HealPixPatches cuts a field [B, C, K] into M square N x N patches centred on
the cells of level ``level - log2(N)``, oriented by the gauge (the same
orientation everywhere on the sphere). Each token gets the
HealPixPositionalEncoding of its centre cell. The encoder (torch.nn
transformer layers) sees the kept tokens only, gathered with
``patches(x, index=keep)``; the decoder predicts every patch, and
``patches.fold`` puts the prediction back onto the cells. The loss counts
the masked tokens and the valid pixels only.

Runs on CPU in about a second at level 4; raise the level and the widths
on a GPU, where extraction and fold run Triton kernels.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixPatches, HealPixPositionalEncoding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HealPixMAE(nn.Module):
    def __init__(self, level, channels, cell_ids=None, patch_size=4, dim=64, depth=2,
                 heads=4, dec_dim=32, dec_depth=1, mask_ratio=0.75):
        super().__init__()
        # patches [B, C, K] -> [B, M, G, C, N, N]; G = 1 gauge here
        self.patches = HealPixPatches(level, patch_size=patch_size, cell_ids=cell_ids,
                                      device=DEVICE)
        self.C, self.N, self.mask_ratio = channels, patch_size, mask_ratio
        token = 2 * channels * patch_size ** 2               # values + validity channel

        # positional encoding of the patch centres: [F, M], a constant buffer
        self.pe = HealPixPositionalEncoding(self.patches.centre_level,
                                            self.patches.centre_ids, device=DEVICE)
        F_ = self.pe.num_features
        self.embed = nn.Linear(token, dim)
        self.pos = nn.Linear(F_, dim)
        layer = lambda d: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                                     batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(dim), depth)
        self.to_dec = nn.Linear(dim, dec_dim)
        self.dec_pos = nn.Linear(F_, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        self.decoder = nn.TransformerEncoder(layer(dec_dim), dec_depth)
        self.out = nn.Linear(dec_dim, channels * patch_size ** 2)

    def forward(self, x):                      # x [B, C, K], NaN = missing
        B, M = x.shape[0], self.patches.M
        valid = torch.isfinite(x)
        x0 = torch.nan_to_num(x)
        pe = self.pe().T.to(x.dtype)           # [M, F]

        # masked token sampling: one draw per sample, only kept patches are gathered
        n_keep = max(1, int(M * (1 - self.mask_ratio)))
        keep = torch.rand(B, M, device=x.device).argsort(dim=1)[:, :n_keep]   # [B, n_keep]
        vis = self.patches(torch.cat([x0, valid.to(x.dtype)], 1), index=keep)  # [B, n_keep, 1, 2C, N, N]
        h = self.embed(vis.flatten(2)) + self.pos(pe[keep])                    # [B, n_keep, dim]
        h = self.encoder(h)

        # decoder: mask tokens everywhere, encoded tokens scattered back to their place
        d = self.mask_token.expand(B, M, -1).clone()
        d = d.scatter(1, keep[..., None].expand(-1, -1, d.shape[-1]), self.to_dec(h))
        d = self.decoder(d + self.dec_pos(pe))
        pred = self.out(d).view(B, M, 1, self.C, self.N, self.N)              # all patches

        # loss on the masked tokens, valid pixels only, in patch space
        target = self.patches(x0)                                             # [B, M, 1, C, N, N]
        w = self.patches(valid.to(x.dtype)) > 0.999                           # pixel fully valid
        masked = torch.ones(B, M, dtype=torch.bool, device=x.device)
        masked[torch.arange(B, device=x.device)[:, None], keep] = False
        w = w & masked[:, :, None, None, None, None]
        loss = ((pred - target) ** 2 * w).sum() / w.sum().clamp_min(1)

        # reconstruction on the cells: fold the predicted patches ([B, C, K])
        return loss, self.patches.fold(pred)


def main(level=4):
    torch.manual_seed(0)
    K = 12 * 4 ** level                         # full sphere, NESTED order
    model = HealPixMAE(level, channels=2).to(DEVICE)
    x = torch.randn(4, 2, K, device=DEVICE)
    x[:, :, 1000:1300] = float("nan")
    loss, rec = model(x)
    loss.backward()
    assert rec.shape == x.shape and torch.isfinite(loss)
    print(f"K = {K} cells, {model.patches.M} tokens of 4x4, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
