"""``HealPixDown(mode="smooth_conservative")`` against ``smooth`` and ``mean``.

Chain from ``--max-level`` to level 0 on a synthetic temperature-like field:
drift of the global mean for the three modes, the offset added at each step,
the time to build and to apply the operators, and the number of coarse cells
reached by a fine cell.

    python benchmarks/benchmark_smooth_conservative.py --max-level 8 \
        --json benchmarks/results/smooth_conservative.json

float64, CPU, one thread recommended (OMP_NUM_THREADS=1).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings

import numpy as np
import torch
from healpix_geo import nested

from healpix_analyse.down import HealPixDown

MODES = ("smooth", "smooth_conservative", "mean")


def temperature_like(level: int) -> np.ndarray:
    n = 12 * 4**level
    lon, lat = nested.healpix_to_lonlat(np.arange(n), level, ellipsoid="sphere")
    lo, la = np.deg2rad(np.asarray(lon)), np.deg2rad(np.asarray(lat))
    return (
        288.0 - 45.0 * np.sin(la) ** 2
        + 3.0 * np.sin(3 * lo) * np.cos(la) ** 2
        + 2.0 * np.cos(2 * lo + 1) * np.sin(2 * la)
        + 1.5 * np.sin(5 * lo) * np.cos(3 * la)
    )


def make(level, mode, **kw):
    return HealPixDown(level, mode=mode, ellipsoid="sphere", device="cpu",
                       dtype=torch.float64, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-level", type=int, default=8)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore", UserWarning)

    ops = {m: {} for m in MODES}
    per_level = []
    for level in range(a.max_level, 0, -1):
        row = {"level_in": level, "n_in": 12 * 4**level}
        for m in MODES:
            t0 = time.perf_counter()
            ops[m][level] = make(level, m)
            row[f"build_{m}_s"] = time.perf_counter() - t0
        x = temperature_like(level)
        for m in MODES:
            ops[m][level](x)                       # warm-up
            t0 = time.perf_counter()
            for _ in range(5):
                ops[m][level](x)
            row[f"apply_{m}_s"] = (time.perf_counter() - t0) / 5
        ys = ops["smooth"][level](x)[0]
        yc = ops["smooth_conservative"][level](x)[0]
        row["offset_K"] = float((yc - ys).mean())
        per_level.append(row)
        print(f"L{level}->{level-1}: build smooth {row['build_smooth_s']:.2f}s, "
              f"conservative {row['build_smooth_conservative_s']:.2f}s; apply "
              f"smooth {row['apply_smooth_s']*1e3:.1f} ms, conservative "
              f"{row['apply_smooth_conservative_s']*1e3:.1f} ms; offset {row['offset_K']:+.4f} K",
              flush=True)

    x0 = temperature_like(a.max_level)
    m0 = float(x0.mean())
    chain = {}
    for m in MODES:
        x, drift = x0.copy(), {}
        for level in range(a.max_level, 0, -1):
            x = ops[m][level](x)[0]
            drift[level - 1] = float(x.mean()) - m0
        chain[m] = drift
    print("drift of the global mean (K):")
    for m, d in chain.items():
        print(f"  {m:20s}", " ".join(f"L{k}:{v:+.1e}" for k, v in d.items()))

    L = min(5, a.max_level)
    reach = {}
    for m in MODES:
        M = ops[m][L].sparse_matrix().coalesce()
        i, v = M.indices().numpy(), M.values().numpy()
        n = np.bincount(i[1][v > 0], minlength=12 * 4**L)
        reach[m] = dict(level_in=L, min=int(n.min()), median=float(np.median(n)),
                        max=int(n.max()))
    print(f"coarse cells reached by a fine cell (level {L}):", reach)

    out = dict(max_level=a.max_level, per_level=per_level, drift=chain, reach=reach,
               env=dict(python=platform.python_version(), torch=torch.__version__,
                        machine=platform.machine(), system=platform.platform()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
