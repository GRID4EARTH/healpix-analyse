"""Training loop: batched GPU augmentation, NaN masks, masked loss, bf16 autocast.

Task: denoise a smooth field observed with noise and clouds (NaN) on a
regional domain. Every sample of a batch lives on the same cell_ids, so a
batch is a plain stacked tensor [B, C, N].

Augmentation (HealPixAugment, own random parameters per sample, one fused
gather per batch on CUDA): a shift on the sphere, a rotation about the
domain centre, a zoom and a mirror, drawn once as a plan and applied to the
input and to the target alike; then a random NaN masking of NESTED blocks
on the input only. Cells read outside the domain come back as NaN
(fill) with valid=False, so they drop out of the input mask and of the loss.

Mixed precision: inside torch.autocast, torch.nn.Linear runs in bfloat16;
HealPixConv casts its input to its own dtype (float32 here), and the masked
norms keep the dtype of their input. bfloat16 needs no GradScaler.
Runs on CPU in a few seconds at level 4.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from healpix_analyse import (
    HealPixAugment,
    HealPixConv,
    HealPixGroupNorm,
    HealPixUp,
    RandomFlip,
    RandomMask,
    RandomRotation,
    RandomShift,
    RandomZoom,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


def make_data(level, cell_ids, n, cloud_fraction=0.2):
    """Smooth fields (HealPixUp of coarse noise), noisy copies with NaN clouds."""
    coarse = np.unique(np.asarray(cell_ids) // 16)
    up1 = HealPixUp(level - 2, cell_ids=coarse, device="cpu")
    up2 = HealPixUp(level - 1, cell_ids=np.unique(np.asarray(cell_ids) // 4), device="cpu")
    clean = up2(up1(torch.randn(n, len(coarse)))[0])[0][:, None]     # [n, 1, N]
    noisy = clean + 0.3 * torch.randn_like(clean)
    clouds = torch.rand(n, 1, len(cell_ids)) < cloud_fraction
    noisy[clouds] = float("nan")
    return noisy, clean


class Denoiser(nn.Module):
    def __init__(self, level, cell_ids, width=16):
        super().__init__()
        self.conv1 = HealPixConv(level, 2, width, kernel_sz=3, cell_ids=cell_ids, device=DEVICE)
        self.norm1 = HealPixGroupNorm(4, width)
        self.mix = nn.Linear(width, width)          # per-cell channel mixing (bf16 under autocast)
        self.conv2 = HealPixConv(level, width, 1, kernel_sz=3, cell_ids=cell_ids, device=DEVICE)

    def forward(self, x):                           # x [B, 1, N], NaN = missing
        mask = torch.isfinite(x[:, 0])              # [B, N]
        h = torch.cat([torch.nan_to_num(x), mask[:, None].to(x.dtype)], dim=1)
        h = torch.relu(self.norm1(self.conv1(h), mask))
        h = self.mix(h.transpose(1, 2)).transpose(1, 2)
        return self.conv2(h)                        # cast to float32 by the layer


def masked_mse(pred, target, valid):
    w = valid.to(torch.float32)
    return ((pred.float() - torch.nan_to_num(target)) ** 2 * w).sum() / w.sum().clamp_min(1)


def main(level=4, steps=40):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[40, 41, 42, 43], coarse_level=1)
    noisy, clean = make_data(level, cell_ids, n=32)
    loader = DataLoader(TensorDataset(noisy, clean), batch_size=8, shuffle=True)

    # geometric augmentation: one plan per batch, applied to input and target
    geom = HealPixAugment([RandomShift(0.05),            # up to 0.05 rad on the sphere
                           RandomRotation(math.pi),      # any angle about the domain centre
                           RandomZoom((0.8, 1.25)),      # log-uniform per sample
                           RandomFlip()],                # east-west mirror, p = 0.5
                          level=level, cell_ids=cell_ids).to(DEVICE)
    # synthetic gaps on the input only: 20 % of the 16-cell NESTED blocks set to NaN
    gaps = HealPixAugment([RandomMask(0.2, block=16, fill=float("nan"))],
                          level=level, cell_ids=cell_ids).to(DEVICE)
    gen = torch.Generator().manual_seed(0)              # reproducible draws

    model = Denoiser(level, cell_ids).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses, step = [], 0
    while step < steps:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            plan = geom.sample(x.shape[0], generator=gen)
            xa, ya = plan.apply(x), plan.apply(y)       # same draw, bilinear for both
            x_in = gaps(xa.x, generator=gen).x          # NaN outside the domain and in the gaps
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                pred = model(x_in)
            # score every cell where the augmented target exists (valid=False outside)
            loss = masked_mse(pred, ya.x, ya.valid[:, None] & torch.isfinite(ya.x))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step == steps:
                break
    assert all(np.isfinite(losses)) and np.mean(losses[-5:]) < np.mean(losses[:5])
    print(f"loss {np.mean(losses[:5]):.3f} -> {np.mean(losses[-5:]):.3f} in {steps} steps")


if __name__ == "__main__":
    main()
