"""Runnable quickstart for pyramidal HEALPix convolution.

Run directly: ``python examples/pyramid_conv_quickstart.py``

Builds a synthetic masked map, filters it with a NaN-aware pyramidal
Gaussian-shaped smoother, and reports basic sanity numbers (no plotting
dependency required). See docs/pyramid_convolution.md for the full
guide/math/API/validation write-up.
"""

from __future__ import annotations

import numpy as np
import torch

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.kernel_pyramid import HealPixKernelPyramid, kernel_gaussian
from healpix_analyse.pyramid_conv import HealPixPyramidConv


def main():
    level = 6
    npix = 12 * 4 ** level

    # 1) Build the exact-reconstruction pyramid on the true sphere.
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", Jmax=3, dtype=torch.float32)

    # 2) Build one fixed, non-learnable 5x5 Gaussian-shaped kernel per band.
    kernel_pyramid = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(sigma_pix=1.2), compact_kernel_sz=5, gauge_type="phi",
    )

    # 3) Wrap both in the NaN/weight-aware pyramidal convolution.
    pconv = HealPixPyramidConv(decomp, kernel_pyramid, mode="normalized")

    # 4) A synthetic field with a "coastline"-like missing-data region plus
    #    scattered random dropouts, similar to an ocean/land mask.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(npix).astype(np.float32)
    mask = np.ones(npix, dtype=bool)
    mask[: npix // 6] = False            # one large contiguous "land" block
    mask &= rng.random(npix) > 0.05      # scattered dropouts
    x[~mask] = np.nan

    print(f"level={level}, npix={npix}, missing fraction={1 - mask.mean():.3f}")

    y, support = pconv(x, return_support=True)

    finite = np.isfinite(y)
    print(f"output finite fraction: {finite.mean():.3f}")
    print(f"support range where finite: [{support[finite].min():.3f}, {support[finite].max():.3f}]")

    # Sanity check independent of any test file: a constant field behind the
    # same mask should reconstruct back to that same constant everywhere it
    # has support (see docs/pyramid_convolution.md section A.2).
    c = 3.0
    xc = np.full(npix, c, dtype=np.float32)
    xc[~mask] = np.nan
    yc = pconv(xc)
    finite_c = np.isfinite(yc)
    max_err = np.max(np.abs(yc[finite_c] - c))
    print(f"constant-field reconstruction max abs error where supported: {max_err:.2e}")


if __name__ == "__main__":
    main()
