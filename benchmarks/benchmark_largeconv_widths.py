"""Effective width of ``LargeConv`` against the requested width.

A caller that wants a Gaussian of one-axis width ``sigma`` (in cells of the
fine level) builds the compact kernel as a Gaussian of width
``sqrt(max(sigma**2 - blur**2, 0)) / 2**n`` coarse cells, where ``blur`` is
the Down/Up blur of the ``n`` levels in use (``down_up_blur``).  Two ways of
choosing ``n`` are compared:

* ``default``: ``n`` follows from the kernel size (``kernel_sz = 2 ceil(3 sigma) + 1``,
  ``max_compact_kernel_sz = 7``), the existing rule;
* ``levels_for_width``: ``n`` is the largest number of levels whose blur does
  not exceed ``sigma`` and is passed as ``n_levels``.

The effective width is ``sqrt(sum(w d^2) / (2 sum(w)))`` of the response ``w``
to one cell, ``d`` the angular distance to that cell, in cells.

Run from the repository root::

    python benchmarks/benchmark_largeconv_widths.py --json benchmarks/results/largeconv-widths.json
"""

from __future__ import annotations

import argparse
import json
import math

import healpix_geo.nested as hgn
import numpy as np
import torch

from healpix_analyse import LargeConv, down_up_blur, levels_for_width
from healpix_analyse.large_conv import _kernel_plan


def patch(level: int, depth: int):
    parent_level = level - depth
    parent = 5 if parent_level == 0 else 5 * 4**parent_level + 3 * 4 ** (parent_level - 1)
    ids = parent * 4**depth + np.arange(4**depth, dtype=np.int64)
    lon, lat = hgn.healpix_to_lonlat(ids, level, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    centre = int(np.argmax(xyz @ xyz.mean(axis=0)))
    d = np.arccos(np.clip(xyz @ xyz[centre], -1.0, 1.0))
    return ids, centre, d


def gaussian_compact(kernel_sz: int, sigma_coarse: float) -> np.ndarray:
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    w = np.exp(-(xx**2 + yy**2) / (2 * max(sigma_coarse, 1e-6) ** 2)).ravel()
    return (w / w.sum()).reshape(1, 1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8, help="patch of 4**depth cells")
    ap.add_argument("--sigma-min", type=float, default=0.4)
    ap.add_argument("--sigma-max", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cell = math.sqrt(4 * math.pi / 12) / 2**args.level  # rad
    ids, centre, d = patch(args.level, args.depth)
    impulse = np.zeros(ids.size)
    impulse[centre] = 1.0

    blur_cells = {
        n: down_up_blur(args.level, n, patch_depth=args.depth) / cell for n in range(0, 5)
    }
    layers = {}

    def effective(n, kernel_sz, sigma_c):
        key = (n, kernel_sz)
        if key not in layers:
            layers[key] = LargeConv(
                args.level, 1, 1, kernel_sz, n_levels=n, cell_ids=ids,
                ellipsoid="WGS84", dtype=torch.float64, device="cpu", cache_dir=None,
            )
        layer = layers[key]
        layer.set_compact_kernel(gaussian_compact(layer.compact_kernel_sz, sigma_c))
        w = np.asarray(layer(impulse), dtype=np.float64).ravel()
        return float(np.sqrt(np.sum(w * d**2) / (2 * w.sum())) / cell), float(w.sum())

    rows = []
    for sigma in np.round(np.geomspace(args.sigma_min, args.sigma_max, args.n), 4):
        kernel_sz = 2 * math.ceil(3 * sigma) + 1
        n_def, _ = _kernel_plan(kernel_sz, 7)
        n_new = levels_for_width(
            args.level, sigma * cell, patch_depth=args.depth
        )
        row = {"sigma_cells": float(sigma), "kernel_sz": kernel_sz}
        for tag, n, ksz in (
            ("default", n_def, kernel_sz),
            ("levels_for_width", n_new, kernel_sz),
        ):
            sc = math.sqrt(max(sigma**2 - blur_cells[n] ** 2, 0.0)) / 2**n
            eff, mass = effective(n, ksz, sc)
            row[tag] = {"n_levels": n, "compact_sigma_coarse": sc,
                        "effective_cells": eff, "mass": mass}
        rows.append(row)
        print(
            f"sigma {sigma:7.3f}  default n={n_def} eff={row['default']['effective_cells']:7.3f}"
            f"   levels_for_width n={n_new} eff={row['levels_for_width']['effective_cells']:7.3f}"
        )

    out = {
        "level": args.level,
        "patch_cells": int(ids.size),
        "cell_rad": cell,
        "down_up_blur_cells": blur_cells,
        "rows": rows,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
