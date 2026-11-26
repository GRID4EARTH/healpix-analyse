"""GAN critic with a gradient penalty (WGAN-GP): a double backward.

The penalty (||d critic / d x|| - 1)**2 is computed with
torch.autograd.grad(..., create_graph=True), then the total loss is
differentiated again with respect to the weights: a second-order gradient
through every layer of the critic. HealPixConv (stride 1 and 2) runs it on
its own kernels (Triton on CUDA); the critic here uses HealPixConv,
LeakyReLU and a global_pool readout. A small generator made of HealPixConv
layers is trained against it for a few steps.

Runs on CPU in a few seconds at level 4.
"""

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import HealPixConv, global_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def regional_domain(level, coarse_ids, coarse_level):
    k = 4 ** (level - coarse_level)
    return (np.asarray(coarse_ids)[:, None] * k + np.arange(k)).ravel()


class Critic(nn.Module):
    def __init__(self, level, cell_ids, c_in, width=16):
        super().__init__()
        self.c1 = HealPixConv(level, c_in, width, kernel_sz=3, cell_ids=cell_ids, device=DEVICE)
        self.c2 = HealPixConv(level, width, 2 * width, kernel_sz=3, stride=2,
                              cell_ids=cell_ids, device=DEVICE)          # one level coarser
        self.c3 = HealPixConv(level - 1, 2 * width, 2 * width, kernel_sz=3,
                              cell_ids=self.c2.cell_ids_out, device=DEVICE)
        self.act = nn.LeakyReLU(0.2)
        self.head = nn.Linear(2 * width, 1)

    def forward(self, x):                        # [B, C, N] -> [B]
        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        h = self.act(self.c3(h))
        return self.head(global_pool(h)).squeeze(-1)


class Generator(nn.Module):
    def __init__(self, level, cell_ids, c_noise, c_out, width=16):
        super().__init__()
        self.net = nn.ModuleList([
            HealPixConv(level, c_noise, width, kernel_sz=3, cell_ids=cell_ids, device=DEVICE),
            HealPixConv(level, width, c_out, kernel_sz=3, cell_ids=cell_ids, device=DEVICE)])

    def forward(self, z):
        return self.net[1](torch.relu(self.net[0](z)))


def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.shape[0], 1, 1, device=real.device)
    x = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(x).sum(), x, create_graph=True)   # keeps the graph
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def main(level=4, steps=5, lam=10.0):
    torch.manual_seed(0)
    cell_ids = regional_domain(level, coarse_ids=[8, 9, 10, 11], coarse_level=1)
    N = len(cell_ids)
    real = torch.sin(torch.linspace(0, 12, N, device=DEVICE))[None, None].repeat(16, 1, 1)
    real = real + 0.1 * torch.randn_like(real)

    critic, gen = Critic(level, cell_ids, 1).to(DEVICE), Generator(level, cell_ids, 4, 1).to(DEVICE)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.0, 0.9))
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.9))
    for _ in range(steps):
        fake = gen(torch.randn(16, 4, N, device=DEVICE)).detach()
        gp = gradient_penalty(critic, real, fake)
        loss_c = critic(fake).mean() - critic(real).mean() + lam * gp
        opt_c.zero_grad(set_to_none=True)
        loss_c.backward()                        # second-order through the HealPixConv layers
        opt_c.step()

        loss_g = -critic(gen(torch.randn(16, 4, N, device=DEVICE))).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()

    w = critic.c2.weight.grad
    assert w is not None and torch.isfinite(w).all() and torch.isfinite(gp)
    print(f"N = {N} cells, critic loss {loss_c.item():.3f}, gradient penalty {gp.item():.3f}")


if __name__ == "__main__":
    main()
