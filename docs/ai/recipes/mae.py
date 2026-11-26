"""Masked autoencoder (ViT / MAE) with NESTED-block tokens.

In NESTED order the 4**k cells of one parent at level ``level - k`` are
consecutive, so a plain reshape turns a field [B, C, N] into tokens
[B, T, C * 4**k], one per parent cell (the "patch"). Each token gets the
HealPixPositionalEncoding of its parent cell. Encoder and decoder are
torch.nn.TransformerEncoder layers (scaled dot-product attention). The
loss is computed on the masked tokens only, and only on valid cells.

The domain must be sorted and made of complete parents (the full sphere,
or every descendant of a set of coarse cells). Runs on CPU in about a
second at level 4.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixPositionalEncoding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tokens(x, k):
    """[B, C, N] -> [B, N // 4**k, C * 4**k] (one token per parent cell)."""
    B, C, N = x.shape
    return x.reshape(B, C, N // 4 ** k, 4 ** k).permute(0, 2, 1, 3).reshape(B, N // 4 ** k, -1)


def from_tokens(t, C, k):
    """Inverse of to_tokens: [B, T, C * 4**k] -> [B, C, T * 4**k]."""
    B, T, _ = t.shape
    return t.reshape(B, T, C, 4 ** k).permute(0, 2, 1, 3).reshape(B, C, T * 4 ** k)


class HealPixMAE(nn.Module):
    def __init__(self, level, cell_ids, channels, k=2, dim=64, depth=2, heads=4,
                 dec_dim=32, dec_depth=1, mask_ratio=0.75):
        super().__init__()
        cell_ids = np.asarray(cell_ids)
        assert np.all(np.diff(cell_ids) > 0) and len(cell_ids) % 4 ** k == 0
        token_ids = cell_ids[:: 4 ** k] // 4 ** k          # parent of each token
        self.k, self.C, self.mask_ratio = k, channels, mask_ratio
        patch = channels * 4 ** k

        # positional encoding of the token cells: [F, T], a constant buffer
        self.pe = HealPixPositionalEncoding(level - k, token_ids, device=DEVICE)
        F_ = self.pe.num_features
        self.embed = nn.Linear(2 * patch, dim)              # values + validity
        self.pos = nn.Linear(F_, dim)
        layer = lambda d: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                                     batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(dim), depth)
        self.to_dec = nn.Linear(dim, dec_dim)
        self.dec_pos = nn.Linear(F_, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(dec_dim))
        self.decoder = nn.TransformerEncoder(layer(dec_dim), dec_depth)
        self.out = nn.Linear(dec_dim, patch)

    def forward(self, x):                      # x [B, C, N], NaN = missing
        B = x.shape[0]
        valid = torch.isfinite(x)
        tok = to_tokens(torch.cat([torch.nan_to_num(x), valid.to(x.dtype)], 1), self.k)
        T = tok.shape[1]
        pe = self.pe().T.to(x.dtype)           # [T, F]

        # random masking: keep n_keep tokens per sample
        n_keep = max(1, int(T * (1 - self.mask_ratio)))
        order = torch.rand(B, T, device=x.device).argsort(dim=1)
        keep = order[:, :n_keep]                                  # [B, n_keep]
        h = self.embed(tok) + self.pos(pe)                        # [B, T, dim]
        h = torch.gather(h, 1, keep[..., None].expand(-1, -1, h.shape[-1]))
        h = self.encoder(h)                                       # visible tokens only

        # decoder: mask tokens everywhere, encoded tokens scattered back
        d = self.mask_token.expand(B, T, -1).clone()
        d = d.scatter(1, keep[..., None].expand(-1, -1, d.shape[-1]), self.to_dec(h))
        d = self.decoder(d + self.dec_pos(pe))
        pred = from_tokens(self.out(d), self.C, self.k)           # [B, C, N]

        masked = torch.ones(B, T, dtype=torch.bool, device=x.device)
        masked[torch.arange(B, device=x.device)[:, None], keep] = False
        masked_cells = masked.repeat_interleave(4 ** self.k, dim=1)[:, None]   # [B, 1, N]
        w = (valid & masked_cells).to(x.dtype)
        loss = ((pred - torch.nan_to_num(x)) ** 2 * w).sum() / w.sum().clamp_min(1)
        return loss, pred


def main(level=4):
    torch.manual_seed(0)
    N = 12 * 4 ** level                         # full sphere, NESTED order
    model = HealPixMAE(level, np.arange(N), channels=2).to(DEVICE)
    x = torch.randn(4, 2, N, device=DEVICE)
    x[:, :, 1000:1300] = float("nan")
    loss, pred = model(x)
    loss.backward()
    assert pred.shape == x.shape and torch.isfinite(loss)
    print(f"N = {N} cells, {N // 16} tokens, loss {loss.item():.3f}")


if __name__ == "__main__":
    main()
