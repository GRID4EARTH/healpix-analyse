"""CPU benchmark for HealPixDecomp.compute_weighted / HealPixKernelPyramid /
HealPixPyramidConv.

Run directly: ``python scripts/benchmark_pyramid_conv.py``

Honesty note: this machine has no GPU, so only CPU numbers are reported
here. Do not extrapolate GPU throughput from these figures -- report them
as CPU-only until measured on real GPU hardware.
"""

from __future__ import annotations

import platform
import time

import numpy as np
import torch

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.kernel_pyramid import HealPixKernelPyramid, kernel_gaussian
from healpix_analyse.pyramid_conv import HealPixPyramidConv


def _time(fn, *, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    torch.set_num_threads(torch.get_num_threads())
    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}, torch: {torch.__version__}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()} -- benchmark below is CPU only")
    print()
    header = (
        f"{'level':>5} {'npix':>9} {'Jmax':>4} {'build_s':>10} "
        f"{'fwd_ms':>8} {'fwd_ms/px(1e-6)':>16}"
    )
    print(header)
    print("-" * len(header))

    for level, jmax in [(5, 2), (6, 3), (7, 3)]:
        npix = 12 * 4 ** level
        decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float32, Jmax=jmax)
        kernel = kernel_gaussian(sigma_pix=1.2)

        def build():
            return HealPixKernelPyramid.from_kernel(
                decomp, kernel, compact_kernel_sz=5, gauge_type="phi", dtype=torch.float32
            )

        build_time = _time(build, repeats=1)
        kp = build()
        pconv = HealPixPyramidConv(decomp, kp, mode="normalized")

        rng = np.random.default_rng(0)
        x = rng.standard_normal(npix).astype(np.float32)
        mask = rng.random(npix) > 0.1
        x[~mask] = np.nan

        def forward():
            with torch.no_grad():
                pconv(x)

        fwd_time = _time(forward, repeats=5)
        print(
            f"{level:5d} {npix:9d} {jmax:4d} {build_time:10.3f} "
            f"{fwd_time * 1e3:8.2f} {fwd_time / npix * 1e6:16.4f}"
        )

    print()
    print(
        "Notes: build_s is one-time geometry construction (cached to disk by "
        "HealPixConv after the first run at a given configuration); fwd_ms is "
        "a single compute_weighted+apply+invert forward pass; fwd_ms/px is that "
        "time divided by pixel count (x1e-6 for readability). No GPU was "
        "available in this environment; these are CPU-only figures."
    )


if __name__ == "__main__":
    main()
