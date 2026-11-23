"""Land barrier experiment: PR 81 operator against the barrier operators.

Scenes on the 8125-cell level-10 domain of ``--cells`` (synthetic coast: a
two-cell strip across the domain with and without a four-cell strait, and an
island of radius four cells) and, if readable, the real land mask of the
pangeo-fish demo (``--demo``, level 10, 39 388 cells, 30 % land = NaN in ``mask``).

Variants (all ``width_rule="matched"``, ``truncate=4``):

``pr81``      full-disc operator without barrier, land masked after every
              step (what pangeo-fish does today: mass that falls on land is
              lost, straight-line distance)
``lost``      barrier, ``normalization="full_disc"``  (a)
``kept``      barrier, ``normalization="source"``      (b)

Writes ``land_barrier_experiment.json`` and ``land_barrier_experiment.png``
next to this file.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from healpix_geo import nested

from healpix_analyse import apply_filter_operator, gaussian_filter_operator
from healpix_analyse.radial_filter import _cell_size_m, _clear_filter_caches

HERE = Path(__file__).parent
LEVEL = 10
CELL = _cell_size_m(LEVEL)
EARTH = 6_371_007.1809
SIGMAS = (0.7, 1.0, 2.0, 4.0)
STEPS = (1, 40)


def plane(ids, centre):
    lon, lat = (np.asarray(a) for a in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    lon = (lon + 180.0) % 360.0 - 180.0
    x = np.radians(lon - centre[0]) * np.cos(np.radians(centre[1])) * EARTH
    y = np.radians(lat - centre[1]) * EARTH
    return x / CELL, y / CELL


def build(ids, blocked, normalization, sigma):
    return gaussian_filter_operator(
        ids,
        LEVEL,
        sigma_m=sigma * CELL,
        normalization=normalization,
        width_rule="matched",
        blocked_cells=blocked,
    )


def matrix(op, n):
    return sp.csr_matrix(
        (op.weights, op.column_indices, op.row_offsets), shape=(len(op.row_offsets) - 1, n)
    )


def run_steps(matrices, mask_land, state, steps):
    """Mass after each requested number of steps (no renormalisation)."""
    out = {}
    for step in range(1, max(steps) + 1):
        state = matrices @ state
        if mask_land is not None:
            state = np.where(mask_land, 0.0, state)
        if step in steps:
            out[step] = state.copy()
    return out


def scene_metrics(ids, land_ids, u, v, source, far, label):
    """Mass after 1 and 40 steps for the three variants, per sigma."""
    n = ids.size
    land = np.isin(ids, land_ids)
    result = {}
    for sigma in SIGMAS:
        entry = {}
        for name, normalization, blocked in (
            ("pr81", "full_disc", None),
            ("lost", "full_disc", land_ids),
            ("kept", "source", land_ids),
        ):
            op = build(ids, blocked, normalization, sigma)
            state = np.zeros(n)
            state[source] = 1.0
            after = run_steps(matrix(op, n), land if name == "pr81" else None, state, STEPS)
            entry[name] = {
                f"step{s}": {
                    "total": float(after[s].sum()),
                    "far": float(after[s][far].sum()),
                }
                for s in STEPS
            }
        result[f"sigma{sigma}"] = entry
        print(label, sigma, {k: {s: round(v2["far"], 5) for s, v2 in e.items()} for k, e in entry.items()}, flush=True)
    return result


def coast_loss(ids, land_ids, u, v, sigma=2.0):
    """One-step mass lost by a delta at 1, 3 and 6 cells from the strip."""
    n = ids.size
    land = np.isin(ids, land_ids)
    ops = {
        "pr81": matrix(build(ids, None, "full_disc", sigma), n),
        "lost": matrix(build(ids, land_ids, "full_disc", sigma), n),
        "kept": matrix(build(ids, land_ids, "source", sigma), n),
    }
    out = {}
    for distance in (1.0, 3.0, 6.0):
        source = int(np.argmin((u + 1.0 + distance) ** 2 + v**2))
        state = np.zeros(n)
        state[source] = 1.0
        row = {}
        for name, m in ops.items():
            after = m @ state
            if name == "pr81":
                after = np.where(land, 0.0, after)
            row[name] = float(1.0 - after.sum())
        out[f"{distance:g}_cells"] = row
    return out


def timing(ids, land_ids, sigma):
    """Build time and peak memory of the operators (caches cleared)."""
    out = {}
    for name, normalization, blocked in (
        ("pr81", "full_disc", None),
        ("lost", "full_disc", land_ids),
        ("kept", "source", land_ids),
    ):
        _clear_filter_caches()
        t0 = time.perf_counter()
        op = build(ids, blocked, normalization, sigma)
        seconds = time.perf_counter() - t0
        _clear_filter_caches()
        tracemalloc.start()
        build(ids, blocked, normalization, sigma)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        out[name] = {
            "seconds": round(seconds, 2),
            "peak_mib": round(peak / 2**20, 1),
            "nnz": int(op.weights.size),
        }
    return out


def synthetic(cells_path):
    ids = np.load(cells_path)["ids"].astype(np.uint64)
    lon, lat = (np.asarray(a) for a in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    lon = (lon + 180.0) % 360.0 - 180.0
    centre = (float(np.median(lon)), float(np.median(lat)))
    u, v = plane(ids, centre)
    print("domain", ids.size, "centre", centre, "u", u.min(), u.max(), "v", v.min(), v.max())
    scenes = {}
    strip = np.abs(u) <= 1.0
    strait = strip & (np.abs(v) >= 2.0)
    island = np.hypot(u, v) <= 4.0
    source = int(np.argmin((u + 3.0) ** 2 + v**2))
    island_source = int(np.argmin((u + 6.0) ** 2 + v**2))
    shadow = (u >= 3.0) & (u <= 8.0) & (np.abs(v) <= 2.0)
    scenes["strip"] = (ids[strip], source, u > 1.5)
    scenes["strip_with_strait"] = (ids[strait], source, u > 1.5)
    scenes["island"] = (ids[island], island_source, shadow)
    metrics = {}
    for name, (land_ids, src, far) in scenes.items():
        print("scene", name, "land cells", land_ids.size)
        metrics[name] = {
            "land_cells": int(land_ids.size),
            "mass": scene_metrics(ids, land_ids, u, v, src, far, name),
        }
    strip_ids = scenes["strip"][0]
    metrics["strip"]["one_step_loss_sigma2"] = coast_loss(ids, strip_ids, u, v)
    metrics["strip"]["timing_sigma2"] = timing(ids, strip_ids, 2.0)
    metrics["strip"]["timing_sigma1"] = timing(ids, strip_ids, 1.0)
    metrics["island"]["timing_sigma2"] = timing(ids, scenes["island"][0], 2.0)
    figure(ids, u, v, scenes)
    return metrics


def figure(ids, u, v, scenes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = ids.size
    land_ids, source, _ = scenes["strip_with_strait"]
    land = np.isin(ids, land_ids)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharex=True, sharey=True)
    state = np.zeros(n)
    state[source] = 1.0
    panels = (
        ("PR 81 (straight line, land masked)", None, "full_disc", True),
        ("barrier, mass lost (full_disc)", land_ids, "full_disc", False),
        ("barrier, mass kept (source)", land_ids, "source", False),
    )
    for ax, (title, blocked, norm, mask) in zip(axes[1:], panels):
        m = matrix(build(ids, blocked, norm, 2.0), n)
        after = m @ state
        if mask:
            after = np.where(land, 0.0, after)
        shown = np.where(land, np.nan, np.log10(np.maximum(after, 1e-8)))
        sc = ax.scatter(u, v, c=shown, s=2, vmin=-6, vmax=-1.5, cmap="viridis")
        ax.set_title(title, fontsize=9)
    axes[0].scatter(u[~land], v[~land], s=1, c="lightblue")
    axes[0].scatter(u[land], v[land], s=1, c="saddlebrown")
    axes[0].scatter(u[source], v[source], s=25, c="red")
    axes[0].set_title("scene: strip with strait, source (red)", fontsize=9)
    fig.colorbar(sc, ax=axes, label="log10 weight after one step (sigma 2 cells)", shrink=0.8)
    for ax in axes:
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_aspect("equal")
    fig.savefig(HERE / "land_barrier_experiment.png", dpi=110, bbox_inches="tight")


def demo(path, sigmas=(0.7, 1.0, 2.0)):
    import zarr

    group = zarr.open_group(str(path), mode="r")
    ids = np.asarray(group["cell_ids"][:]).astype(np.uint64)
    mask = np.asarray(group["mask"][:])
    land = np.isnan(mask)  # sea is 1 (up to rounding), land is NaN
    land_ids = ids[land]
    n = ids.size
    print("demo cells", n, "land fraction", land.mean(), flush=True)
    # distance of every water cell to the nearest land cell, in cells
    from scipy.spatial import cKDTree

    lon, lat = (np.asarray(a) for a in nested.healpix_to_lonlat(ids, LEVEL, ellipsoid="WGS84"))
    lon = (lon + 180.0) % 360.0 - 180.0
    u, v = plane(ids, (float(np.median(lon)), float(np.median(lat))))
    tree = cKDTree(np.column_stack((u[land], v[land])))
    distance = np.full(n, np.inf)
    distance[~land] = tree.query(np.column_stack((u[~land], v[~land])))[0]
    out = {"cells": int(n), "land_fraction": float(land.mean())}
    for sigma in sigmas:
        _clear_filter_caches()
        t0 = time.perf_counter()
        plain = build(ids, None, "full_disc", sigma)
        t_plain = time.perf_counter() - t0
        _clear_filter_caches()
        t0 = time.perf_counter()
        lost = build(ids, land_ids, "full_disc", sigma)
        t_lost = time.perf_counter() - t0
        kept = build(ids, land_ids, "source", sigma)
        W = matrix(plain, n)
        L = matrix(lost, n)
        K = matrix(kept, n)
        water = ~land
        # one-step weight of the PR 81 operator between water cells that the
        # barrier operator does not connect: mass that crosses land
        Wc = W.tocoo()
        keep = water[Wc.row] & water[Wc.col]
        Wwater = sp.csr_matrix((Wc.data[keep], (Wc.row[keep], Wc.col[keep])), shape=(n, n))
        Lb = L.copy()
        Lb.data = np.ones_like(Lb.data)
        connected = Wwater.multiply(Lb)
        crossing = np.asarray((Wwater - connected).sum(axis=0)).ravel()  # per source
        loss_pr81 = 1.0 - np.asarray((sp.diags(water.astype(float)) @ W).sum(axis=0)).ravel()
        loss_lost = 1.0 - np.asarray(L.sum(axis=0)).ravel()
        loss_kept = 1.0 - np.asarray(K.sum(axis=0)).ravel()
        bands = {}
        for label, low, high in (("0-1 cells", 0, 1.5), ("1-3 cells", 1.5, 3.5), ("3-6 cells", 3.5, 6.5)):
            sel = water & (distance > low) & (distance <= high)
            if sel.sum() == 0:
                continue
            bands[label] = {
                "sources": int(sel.sum()),
                "crossing_land_mean": float(crossing[sel].mean()),
                "crossing_land_max": float(crossing[sel].max()),
                "loss_pr81_mean": float(loss_pr81[sel].mean()),
                "loss_lost_mean": float(loss_lost[sel].mean()),
                "loss_kept_max_abs": float(np.abs(loss_kept[sel]).max()),
            }
        out[f"sigma{sigma}"] = {
            "nnz_pr81": int(plain.weights.size),
            "nnz_barrier": int(lost.weights.size),
            "build_seconds_pr81": round(t_plain, 1),
            "build_seconds_barrier_full_disc": round(t_lost, 1),
            "by_distance_to_land": bands,
        }
        print("demo", sigma, json.dumps(out[f"sigma{sigma}"])[:600], flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        default="/Users/todaka/python/git/FISH/regrid-diff-check-2026-11-17/small-sigma/data/cells.npz",
    )
    parser.add_argument(
        "--demo",
        default="/Users/todaka/python/git/FISH/pangeo-fish/notebooks/test/A19124/emission_diff.zarr",
    )
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--only-demo", action="store_true", help="keep the stored synthetic results")
    parser.add_argument("--demo-sigmas", default="0.7,1,2")
    args = parser.parse_args()
    path = HERE / "land_barrier_experiment.json"
    if args.only_demo:
        results = json.loads(path.read_text())
    else:
        results = {"synthetic": synthetic(args.cells)}
    path.write_text(json.dumps(results, indent=1))
    if not args.skip_demo:
        try:
            results["demo"] = demo(args.demo, tuple(float(s) for s in args.demo_sigmas.split(",")))
        except Exception as exc:  # noqa: BLE001
            results["demo"] = {"error": repr(exc)}
    path.write_text(json.dumps(results, indent=1))
