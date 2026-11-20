"""Total, constants and effective width of the local FFT convolution.

For a Gaussian kernel of requested width ``sigma`` (grid pixels) this measures,
on a real 8125-cell domain at level 10 (``data/fft_domain_level10.npz``):

* column sums: the sum over the output of the response to one cell (1 means the
  total is kept);
* row sums: the response to a constant field (1 means a constant is kept);
* effective width of the response to one cell versus the requested width.

Interior cells are those at least ``ceil(5 sigma) + 2`` grid pixels away from
the closest grid point not covered by the domain; edge cells are within 2
pixels of it.  Run with the Python of the project environment::

    python benchmarks/benchmark_fft_total.py --json fft-total.json

The script runs on releases without the ``projection`` option (only the
default convention is then measured).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time

import numpy as np
import torch
from scipy import ndimage

from healpix_analyse import HealPixFFTConv

HERE = os.path.dirname(os.path.abspath(__file__))


def gaussian_weight(sigma_px: float, kernel_sz: int) -> torch.Tensor:
    half = kernel_sz // 2
    axis = torch.arange(-half, half + 1, dtype=torch.float64)
    profile = torch.exp(-0.5 * (axis / sigma_px) ** 2)
    kernel = profile[:, None] * profile[None, :]
    return (kernel / kernel.sum())[None, None]


def stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", default=os.path.join(HERE, "data", "fft_domain_level10.npz"))
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--n-width-cells", type=int, default=150, help="interior cells used for the effective width")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    domain = np.load(args.domain)
    ids, level = domain["ids"], int(domain["level"])
    modes = ["average"]
    if "projection" in inspect.signature(HealPixFFTConv.__init__).parameters:
        modes.append("conservative")

    result = {
        "domain": os.path.basename(args.domain),
        "level": level,
        "n_cells": int(ids.size),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "modes": {},
    }
    rng = np.random.default_rng(0)

    for mode in modes:
        extra = {"projection": mode} if mode != "average" or len(modes) > 1 else {}
        rows = []
        for sigma in args.sigmas:
            kernel_sz = max(5, 2 * math.ceil(5 * sigma) + 1)
            t0 = time.time()
            layer = HealPixFFTConv(
                level, 1, 1, kernel_sz, cell_ids=ids, dtype=torch.float64,
                device="cpu", **extra,
            ).eval()
            layer.set_kernel(gaussian_weight(sigma, kernel_sz))
            tr = layer.transform
            px = tr.pixel_size_rad
            centre = 0.5 * (tr.grid_size - 1)
            gx = np.rint(tr.projected_x.numpy() / px + centre).astype(int)
            gy = np.rint(tr.projected_y.numpy() / px + centre).astype(int)
            edt = ndimage.distance_transform_edt(tr.coverage_mask.numpy())[gy, gx]
            interior = edt >= math.ceil(5 * sigma) + 2
            edge = edt <= 2
            x = tr.projected_x.numpy() / px
            y = tr.projected_y.numpy() / px

            n = ids.size
            col = np.empty(n)
            width_cells = np.flatnonzero(interior)
            if width_cells.size > args.n_width_cells:
                width_cells = np.sort(rng.choice(width_cells, args.n_width_cells, replace=False))
            width_set = set(int(c) for c in width_cells)
            widths = []
            with torch.no_grad():
                for start in range(0, n, args.batch):
                    sel = np.arange(start, min(start + args.batch, n))
                    eye = torch.zeros(sel.size, n, dtype=torch.float64)
                    eye[torch.arange(sel.size), torch.as_tensor(sel)] = 1.0
                    out = layer(eye).reshape(sel.size, n)
                    col[sel] = out.sum(-1).numpy()
                    for k, j in enumerate(sel):
                        if j in width_set:
                            r = out[k].numpy()
                            d2 = (x - x[j]) ** 2 + (y - y[j]) ** 2
                            widths.append(math.sqrt(0.5 * (r * d2).sum() / r.sum()))
                row = layer(torch.ones(n, dtype=torch.float64)).reshape(n).numpy()

            floor = (
                float(np.median(layer.blur_floor_px()))
                if hasattr(layer, "blur_floor_px")
                else None
            )
            rows.append(
                {
                    "sigma_px": sigma,
                    "kernel_sz": kernel_sz,
                    "column_sum_interior": stats(col[interior]),
                    "column_sum_edge": stats(col[edge]),
                    "column_sum_all": stats(col),
                    "row_sum_interior": stats(row[interior]),
                    "row_sum_edge": stats(row[edge]),
                    "effective_width_px": stats(np.array(widths)),
                    "expected_width_px_sqrt_sigma2_plus_floor2": math.hypot(sigma, 0.586),
                    "blur_floor_px_median_from_geometry": floor,
                    "seconds": round(time.time() - t0, 1),
                }
            )
            c = rows[-1]["column_sum_interior"]
            print(
                f"{mode:12s} sigma={sigma:4.2f}px interior n={c.get('n')} "
                f"col [{c.get('min', float('nan')):.3f},{c.get('max', float('nan')):.3f}] "
                f"edge max {rows[-1]['column_sum_edge'].get('max', float('nan')):.3f} "
                f"row [{rows[-1]['row_sum_interior'].get('min', float('nan')):.3f},"
                f"{rows[-1]['row_sum_interior'].get('max', float('nan')):.3f}] "
                f"width {rows[-1]['effective_width_px'].get('mean', float('nan')):.3f}",
                flush=True,
            )
        result["modes"][mode] = rows

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
