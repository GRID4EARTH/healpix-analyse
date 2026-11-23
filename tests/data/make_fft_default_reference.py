"""Make tests/data/fft_default_reference.npz.

Run it against the code of commit 272df21 (before the ``projection`` option
existed), from a copy of that commit::

    PYTHONPATH=<copy of 272df21> python tests/data/make_fft_default_reference.py <out.npz>

The test ``test_default_path_matches_reference_made_with_main`` compares the
current default path with this file.
"""

import math
import sys

import healpix_geo
import numpy as np
import torch

from healpix_analyse import HealPixFFTConv
from healpix_analyse.fft_local import LocalFFT

LEVEL = 8


def domain_ids():
    all_ids = np.arange(12 * 4**LEVEL, dtype=np.int64)
    lon, lat = healpix_geo.nested.healpix_to_lonlat(all_ids, LEVEL)
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    near = np.hypot((lon + 4.0) * math.cos(math.radians(47.0)), np.asarray(lat) - 47.0)
    return all_ids[near < 6.0]


def gaussian_weight(sigma=1.5, size=9):
    half = size // 2
    axis = np.arange(-half, half + 1, dtype=np.float64)
    profile = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel = np.outer(profile, profile)
    return (kernel / kernel.sum())[None, None]


if __name__ == "__main__":
    ids = domain_ids()
    rng = np.random.default_rng(12345)
    x = rng.random((3, ids.size))
    layer = HealPixFFTConv(
        LEVEL, 1, 1, 9, cell_ids=ids, dtype=torch.float64, device="cpu"
    ).eval()
    layer.set_kernel(gaussian_weight())
    transform = LocalFFT(ids, LEVEL, dtype=torch.float64, device="cpu")
    spectrum = transform.fft(x)
    np.savez_compressed(
        sys.argv[1],
        ids=ids,
        x=x,
        conv=layer(torch.as_tensor(x)).detach().numpy(),
        projected=transform.project(x),
        round_trip=transform.ifft(spectrum),
    )
