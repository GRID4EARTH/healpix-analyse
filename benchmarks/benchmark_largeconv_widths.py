"""Effective width of ``LargeConv`` against the requested width.

Compared on a dense sweep of requested one-axis widths ``sigma`` (cells of the
fine level, nominal size ``sqrt(4 pi / 12) / 2**level``):

* ``default``: the automatic level rule (``kernel_sz = 2 ceil(3 sigma) + 1``,
  ``max_compact_kernel_sz = 7``) with a compact Gaussian of width
  ``sqrt(max(sigma**2 - floor**2, 0)) / 2**n`` coarse cells, ``floor`` being
  ``down_up_blur(level, n)``;
* ``gaussian_plan``: levels and compact width from ``gaussian_plan``, calibrated
  on the equatorial patch, and (second location) with ``reference_cell`` there.

The effective width is ``sqrt(sum(w d^2) / (2 sum(w)))`` of the response ``w``
to one cell, ``d`` the angular distance to that cell, in cells.  It is measured
at the location used for the calibration (equatorial patch) and at a second
location (centre of base face 0, latitude about 55 degrees) to show how far
the calibration carries.

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

from healpix_analyse import LargeConv, down_up_blur, gaussian_plan
from healpix_analyse.large_conv import _kernel_plan, _nominal_cell


def make_patch(level: int, depth: int, face_centre: int):
    """Patch of 4**depth cells in the middle of a base face (5: equator, 0: 55 N)."""
    pl = level - depth
    parent = face_centre if pl == 0 else face_centre * 4**pl + 3 * 4 ** (pl - 1)
    ids = parent * 4**depth + np.arange(4**depth, dtype=np.int64)
    lon, lat = hgn.healpix_to_lonlat(ids, level, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    centre = int(np.argmax(xyz @ xyz.mean(axis=0)))
    d = np.arccos(np.clip(xyz @ xyz[centre], -1.0, 1.0))
    return ids, centre, d, float(np.rad2deg(lat[centre]))


def _lonlat_xyz(ids, level):
    lon, lat = hgn.healpix_to_lonlat(ids, level, ellipsoid="WGS84")
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def _widths_totals(layer, xyz, impulses, cell):
    batch = np.zeros((len(impulses), len(xyz)))
    batch[np.arange(len(impulses)), impulses] = 1.0
    y = np.asarray(layer(batch))[:, 0]
    widths = []
    for k, c in enumerate(impulses):
        d = np.arccos(np.clip(xyz @ xyz[c], -1.0, 1.0))
        widths.append(math.sqrt(np.sum(y[k] * d**2) / (2 * y[k].sum())) / cell)
    return np.array(widths), y.sum(axis=1)


def position_study(level, depth, cell):
    """Width and total of the response to 64 random cells near the patch centre.

    ``gaussian_plan`` (ratio width / sigma) and the automatic rule on the same
    chain (ratio to its own mean width; it does not aim at sigma).
    """
    ids, centre, d0, _ = make_patch(level, depth, 5)
    xyz = _lonlat_xyz(ids, level)
    out = {}
    for window in (16, 60, 110):
        cand = np.where(d0 < window * cell)[0]
        imp = np.random.default_rng(0).choice(cand, 64, replace=False)
        rows = {}
        for sigma in (3, 6, 9, 18, 36):
            layer = LargeConv.from_gaussian(
                level, sigma * cell, ids, patch_depth=depth, dtype=torch.float64,
                device="cpu", cache_dir=None,
            )
            w, t = _widths_totals(layer, xyz, imp, cell)
            ksz = 2 * math.ceil(3 * sigma) + 1
            auto = LargeConv(level, 1, 1, ksz, cell_ids=ids, dtype=torch.float64,
                             device="cpu", cache_dir=None)
            # truncated Gaussian on the (small) automatic compact kernel
            rad = (auto.compact_kernel_sz - 1) // 2
            gx, gy = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1))
            wgt = np.exp(-(gx**2 + gy**2).ravel() / (2 * max(sigma / 2**auto.n_levels, 0.3) ** 2))
            auto.set_compact_kernel((wgt / wgt.sum()).reshape(1, 1, -1))
            wa, ta = _widths_totals(auto, xyz, imp, cell)
            rows[str(sigma)] = {
                "plan_n_levels": layer.n_levels,
                "plan_width_over_sigma": [float(w.min() / sigma), float(w.mean() / sigma), float(w.max() / sigma)],
                "plan_total": [float(t.min()), float(t.max())],
                "auto_n_levels": auto.n_levels,
                "auto_width_over_own_mean": [float(wa.min() / wa.mean()), float(wa.max() / wa.mean())],
                "auto_total": [float(ta.min()), float(ta.max())],
            }
        out[f"window_{window}_cells"] = rows
        print("position window", window, json.dumps(rows))
    return out


def edge_study(level=8, radius_deg=15.0):
    """Response near the edge of a partial domain (disc), zero padding."""
    cell = _nominal_cell(level)
    allids = np.arange(12 * 4**level)
    xyz_all = _lonlat_xyz(allids, level)
    c0 = np.array([math.cos(math.radians(40)) * math.cos(0.5),
                   math.cos(math.radians(40)) * math.sin(0.5), math.sin(math.radians(40))])
    ids = allids[np.arccos(np.clip(xyz_all @ c0, -1, 1)) < math.radians(radius_deg)]
    pos = xyz_all[ids]
    r = np.arccos(np.clip(pos @ c0, -1, 1))
    out = {"level": level, "radius_deg": radius_deg, "cells": int(ids.size)}
    for sigma in (3, 6):
        layer = LargeConv.from_gaussian(
            level, sigma * cell, ids, dtype=torch.float64, device="cpu", cache_dir=None
        )
        rows = {}
        for frac in (0.0, 0.5, 0.8, 0.92):
            k = int(np.argmin(np.abs(r - frac * math.radians(radius_deg))))
            w, t = _widths_totals(layer, pos, [k], cell)
            rows[str(frac)] = {"width_over_sigma": float(w[0] / sigma), "total": float(t[0])}
        const = np.asarray(layer(np.ones(ids.size))).ravel()
        rows["constant"] = {
            "centre": float(const[np.argmin(r)]),
            "at_0.92R": float(const[np.argmin(np.abs(r - 0.92 * math.radians(radius_deg)))]),
            "min": float(const.min()),
        }
        out[f"sigma_{sigma}"] = rows
    print("edge", json.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8, help="patch of 4**depth cells")
    ap.add_argument("--sigma-min", type=float, default=2.25)
    ap.add_argument("--sigma-max", type=float, default=40.0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    level = args.level
    cell = _nominal_cell(level)
    sigmas = np.arange(args.sigma_min, args.sigma_max + 1e-9, args.step)
    floors = {n: down_up_blur(level, n, patch_depth=args.depth) / cell for n in range(0, 6)}

    plans = [gaussian_plan(level, s * cell, patch_depth=args.depth) for s in sigmas]
    defaults = []
    for s in sigmas:
        kernel_sz = 2 * math.ceil(3 * s) + 1
        n, _ = _kernel_plan(kernel_sz, 7)
        if n not in floors:  # more levels than the calibration patch allows
            defaults.append(None)
            continue
        sc = math.sqrt(max(s**2 - floors[n] ** 2, 0.0)) / 2**n if n else s
        defaults.append((n, kernel_sz, max(sc, 1e-3)))

    locations = {"equator": 5, "face0_55N": 0}
    result_loc = {}
    for name, face in locations.items():
        ids, centre, d, lat = make_patch(level, args.depth, face)
        impulse = np.zeros(ids.size)
        impulse[centre] = 1.0
        layers = {}

        def width(n, kernel_sz, setter):
            key = (n, kernel_sz)
            if key not in layers:
                layers[key] = LargeConv(
                    level, 1, 1, kernel_sz, n_levels=n, cell_ids=ids,
                    dtype=torch.float64, device="cpu", cache_dir=None,
                )
            layer = layers[key]
            setter(layer)
            w = np.asarray(layer(impulse), dtype=np.float64).ravel()
            return float(np.sqrt(np.sum(w * d**2) / (2 * w.sum())) / cell)

        # one layer per level, the largest kernel needed
        biggest = {}
        for p in plans:
            biggest[p.n_levels] = max(biggest.get(p.n_levels, 1), p.fine_kernel_sz)
        new = [
            width(p.n_levels, biggest[p.n_levels], lambda L, p=p: L.set_gaussian_kernel(p.compact_sigma))
            for p in plans
        ]
        # calibrated on the block that contains a cell of this location
        local_plans = []
        for s in sigmas:
            try:
                local_plans.append(
                    gaussian_plan(level, s * cell, patch_depth=args.depth,
                                  reference_cell=int(ids[centre]))
                )
            except ValueError:  # below the narrowest width at this location
                local_plans.append(None)
        biggest_local = {}
        for p in local_plans:
            if p is not None:
                biggest_local[p.n_levels] = max(biggest_local.get(p.n_levels, 1), p.fine_kernel_sz)
        local = [
            float("nan") if p is None else
            width(p.n_levels, biggest_local[p.n_levels], lambda L, p=p: L.set_gaussian_kernel(p.compact_sigma))
            for p in local_plans
        ]
        biggest_default = {}
        for dflt in defaults:
            if dflt is not None:
                biggest_default[dflt[0]] = max(biggest_default.get(dflt[0], 1), dflt[1])
        old = [
            float("nan") if dflt is None else
            width(dflt[0], biggest_default[dflt[0]], lambda L, sc=dflt[2]: L.set_gaussian_kernel(sc))
            for dflt in defaults
        ]
        result_loc[name] = {
            "latitude_deg": lat, "gaussian_plan": new,
            "gaussian_plan_reference_cell": local, "default": old,
        }

    rows = []
    for k, s in enumerate(sigmas):
        rows.append({
            "sigma_cells": float(round(s, 4)),
            "n_levels_plan": plans[k].n_levels,
            "compact_sigma": plans[k].compact_sigma,
            "n_levels_default": None if defaults[k] is None else defaults[k][0],
        })
    summary = {}
    for name, r in result_loc.items():
        new = np.array(r["gaussian_plan"])
        old = np.array(r["default"])
        loc = np.array(r["gaussian_plan_reference_cell"])
        rel = np.abs(new - sigmas) / sigmas
        summary[name] = {
            "latitude_deg": r["latitude_deg"],
            "plan_min_diff": float(np.diff(new).min()),
            "plan_strictly_increasing": bool(np.all(np.diff(new) > 0)),
            "plan_max_rel_error": float(rel.max()),
            "plan_median_rel_error": float(np.median(rel)),
            "reference_cell_sigma_from_cells": float(sigmas[~np.isnan(loc)][0]),
            "reference_cell_min_diff": float(np.diff(loc[~np.isnan(loc)]).min()),
            "reference_cell_max_rel_error": float(np.nanmax(np.abs(loc - sigmas) / sigmas)),
            "default_sigma_up_to_cells": float(sigmas[~np.isnan(old)][-1]),
            "default_flat_steps": int((np.diff(old[~np.isnan(old)]) == 0).sum()),
            "default_steps": int(np.diff(old[~np.isnan(old)]).size),
            "default_strictly_increasing": bool(np.all(np.diff(old[~np.isnan(old)]) > 0)),
            "default_max_rel_error": float(np.nanmax(np.abs(old - sigmas) / sigmas)),
        }
        print(name, json.dumps(summary[name], indent=1))
    switches = [
        round(float(sigmas[k]), 2)
        for k in range(1, len(plans))
        if plans[k].n_levels != plans[k - 1].n_levels
    ]
    print("switches (cells)", switches)
    out = {
        "level": level,
        "patch_cells": 4**args.depth,
        "cell_rad": cell,
        "blur_floor_cells": floors,
        "switches_cells": switches,
        "summary": summary,
        "sigma_cells": [float(round(s, 4)) for s in sigmas],
        "position": position_study(level, args.depth, cell),
        "domain_edge": edge_study(),
        "rows": rows,
        "effective_cells": {
            name: {
                "gaussian_plan": r["gaussian_plan"],
                "gaussian_plan_reference_cell": r["gaussian_plan_reference_cell"],
                "default": r["default"],
            }
            for name, r in result_loc.items()
        },
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
