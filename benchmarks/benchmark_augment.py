"""Benchmark the batched augmentation of healpix_analyse.augment (issue #109).

Throughput in samples per second for a batch of regional domains (default: 64
domains of 4**8 = 65 536 cells at level 10, one NESTED block of a level-2
cell), on CPU or CUDA.  One JSON line per (pipeline, backend, channels),
then a Markdown table::

    python benchmarks/benchmark_augment.py --device cuda --out aug_cuda.jsonl
    python benchmarks/benchmark_augment.py --device cpu --threads 16 --out aug_cpu.jsonl
    python benchmarks/benchmark_augment.py --table aug_cuda.jsonl aug_cpu.jsonl

Timed stages (median of ``--reps`` runs after one warm-up)
----------------------------------------------------------
plan       ``aug.sample(B, generator)`` alone ("draw ms"): the per-sample
           draw (the geometry is computed lazily, at the first apply).
total      ``aug.sample(B, generator).apply(x)`` without gradient: one
           augmented batch, draw and geometry included ("batch ms" in the
           table), the number behind samples/s.
fwd_bwd    a fresh draw, apply and the backward pass with respect to the data.
geometry   (Triton) the fused kernel without the gather: the cost of
           computing the bilinear weights alone.
fwd GB/s   (Triton) minimal traffic of the fused forward (read the input
           once, write the output and the per-cell scale) / batch time, and
           its fraction of the GPU's peak DRAM bandwidth.

The ``naive`` pipeline is the per-sample way of earlier libraries, for
comparison: a rotation about the domain centre whose source positions are
computed and looked up per sample on the CPU with NumPy
(``get_interp_weights``), then one gather per sample on the device.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time

import numpy as np
import torch

from healpix_analyse import _augment_backend as ab
from healpix_analyse.augment import (
    AdditiveNoise,
    HealPixAugment,
    RandomFlip,
    RandomGaugeRotation,
    RandomMask,
    RandomRotation,
    RandomShift,
    RandomSO3,
    RandomZoom,
)
from healpix_analyse.healpix_interp import get_interp_weights
from healpix_analyse._augment_backend import fused_augment_triton


def _peak_bw(dev):
    """Peak DRAM bandwidth (GB/s) of known GPUs, None otherwise."""
    name = torch.cuda.get_device_name(dev)
    for key, bw in (("A100-SXM4-80GB", 2039), ("A100 80GB PCIe", 1935), ("A100", 1555),
                    ("H100", 3350), ("H200", 4800)):
        if key in name:
            return bw
    return None

GEOM = [RandomShift(0.05), RandomRotation(), RandomZoom((0.8, 1.25)), RandomFlip()]

PIPELINES = {
    # name: (transforms, kwargs, channels-multiplier note)
    "rotate": ([RandomRotation()], {}),
    "rot_zoom_in": ([RandomRotation(), RandomZoom((1.0, 1.5))], {}),
    "shift_rot_zoom_flip": (GEOM, {}),
    "shift_rot_zoom_flip_pyramid": (GEOM, {"antialias": "pyramid"}),
    "noise_mask": ([AdditiveNoise(0.1), RandomMask(0.75, block=64)], {}),
    "gauge": ([RandomGaugeRotation(4, flip=0.5)], {}),
    "mae_full": (GEOM + [AdditiveNoise(0.05), RandomMask(0.75, block=64)], {}),
    # full sphere at --full-level, batch --full-batch (cell_ids=None)
    "so3_full_sphere": ([RandomSO3()], {}),
}
FULL_SPHERE = {"so3_full_sphere"}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn, dev, reps):
    fn()  # two warm-up calls: the first one may autotune / compile
    fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _naive(level, ids, B, C, dev, reps, dtype):
    """Per-sample CPU geometry (NumPy get_interp_weights), per-sample gather."""
    import healpix_geo

    lon, lat = healpix_geo.nested.healpix_to_lonlat(ids.astype(np.uint64), level)
    lon, lat = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    c = xyz.mean(0)
    c /= np.linalg.norm(c)
    order = np.argsort(ids)
    ids_sorted = ids[order]
    x = torch.randn(B, C, ids.size, device=dev, dtype=dtype)
    rng = np.random.default_rng(0)

    def run():
        outs = []
        for b in range(B):
            a = rng.uniform(-math.pi, math.pi)
            K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
            R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
            q = xyz @ R.T
            lo = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
            la = np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1)))
            pix, w = get_interp_weights(lo, la, level)
            p = np.searchsorted(ids_sorted, pix.astype(np.int64)).clip(0, ids.size - 1)
            ok = ids_sorted[p] == pix.astype(np.int64)
            pos = torch.as_tensor(order[p], device=dev)
            wt = torch.as_tensor(np.where(ok, w, 0.0), device=dev, dtype=dtype)
            outs.append((x[b][:, pos] * wt).sum(-1))
        return torch.stack(outs)

    t = _time(run, dev, max(1, reps // 2))
    return {"plan_s": None, "apply_s": None, "fwd_bwd_s": None, "total_s": t}


def run(args):
    dev = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    k = int(round(math.log(args.cells, 4)))
    assert 4**k == args.cells, "--cells must be a power of 4"
    ids = np.arange(args.cells, dtype=np.int64) + 7 * args.cells  # children of cell 7 of level L-k
    B = args.batch
    rows = []
    backends = args.backends or (["triton", "torch"] if dev.type == "cuda" and ab.triton_available() else ["torch"])
    for name in args.pipelines:
        for C in args.channels:
            if name == "naive":
                r = _naive(args.level, ids, args.batch, C, dev, args.reps, dtype)
                rows.append(dict(pipeline=name, backend="numpy+torch", C=C, **r))
                _emit(rows[-1], args, args.batch)
                continue
            tr, kw = PIPELINES[name]
            Ce = C * 4 if name == "gauge" else C
            full = name in FULL_SPHERE
            lev = args.full_level if full else args.level
            ncell = 12 * 4**lev if full else args.cells
            B = args.full_batch if full else args.batch
            for be in backends:
                aug = HealPixAugment(tr, lev, None if full else ids, backend=be, **kw).to(dev)
                gen = torch.Generator().manual_seed(0)
                x = torch.randn(B, Ce, ncell, device=dev, dtype=dtype)
                gy = torch.randn_like(x)
                plan = aug.sample(B, gen)
                t_plan = _time(lambda: aug.sample(B, gen), dev, args.reps)
                # A fresh draw every time: the maps are built lazily and cached
                # on a plan, so re-applying one plan would skip the geometry.
                with torch.no_grad():
                    t_total = _time(lambda: aug.sample(B, gen).apply(x), dev, args.reps)
                t_apply = t_total - t_plan
                xg = x.clone().requires_grad_(True)

                def fb():
                    y = aug.sample(B, gen).apply(xg).x
                    torch.autograd.grad((y.nan_to_num() * gy).sum(), xg)

                t_fb = _time(fb, dev, args.reps) if not args.no_backward else None
                n_taps = max([g.n_taps for g in plan.groups] or [0])
                row = dict(pipeline=name, backend=be, C=Ce, taps=n_taps,
                           plan_s=t_plan, apply_s=t_apply, fwd_bwd_s=t_fb,
                           total_s=t_total)
                if plan.groups and be == "triton" and len(plan.groups) == 1 and len(tr) == len(
                        [t for t in tr if t.kind_for(aug.grid) in ("R", "A")]):
                    # Geometry alone (weights, no gather) and the bandwidth of
                    # the fused forward against the minimal traffic: read the
                    # input once, write the output and the per-cell scale.
                    grp = plan.groups[0]
                    grp.geo()
                    row["geom_s"] = _time(lambda: fused_augment_triton(grp.geo(), None), dev, args.reps)
                    es = x.element_size()
                    nbytes = B * Ce * ncell * es * 2 + B * ncell * 8
                    row["fwd_GBps"] = nbytes / t_total / 1e9
                    row["peak_GBps"] = _peak_bw(dev)
                    if t_fb is not None:
                        row["bwd_s"] = t_fb - t_total
                row.update(level=lev, cells=ncell)
                rows.append(row)
                _emit(rows[-1], args, B)
    return rows


def _emit(r, args, B):
    r.setdefault("cells", args.cells)
    r.setdefault("level", args.level)
    r.update(device=args.device, B=B, dtype=args.dtype,
             threads=torch.get_num_threads(), label=args.label,
             samples_per_s=B / r["total_s"] if r.get("total_s") else None)
    line = json.dumps(r)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


def table(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]

    def ms(v):
        return "—" if v is None else f"{1e3 * v:.1f}"

    print("| device | dtype | pipeline | B x cells | backend | C | taps | draw ms | batch ms | geometry ms "
          "| fwd+bwd ms | samples/s | fwd GB/s (% peak) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bw = "—"
        if r.get("fwd_GBps"):
            pk = r.get("peak_GBps")
            bw = f"{r['fwd_GBps']:.0f}" + (f" ({100 * r['fwd_GBps'] / pk:.0f} %)" if pk else "")
        print(f"| {r['device']} | {r.get('dtype', '')} | {r['pipeline']} | {r['B']} x {r['cells']} | "
              f"{r['backend']} | {r['C']} | "
              f"{r.get('taps', '—')} | {ms(r.get('plan_s'))} | {ms(r.get('total_s'))} | "
              f"{ms(r.get('geom_s'))} | {ms(r.get('fwd_bwd_s'))} | {r['samples_per_s']:.0f} | {bw} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--cells", type=int, default=4**8)
    ap.add_argument("--channels", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--full-level", type=int, default=8, help="level of the full-sphere case")
    ap.add_argument("--full-batch", type=int, default=16, help="batch of the full-sphere case")
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES) + ["naive"],
                    choices=list(PIPELINES) + ["naive"])
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--table", nargs="+", default=None, help="print the Markdown table of JSON-line files")
    args = ap.parse_args(argv)
    if args.table:
        table(args.table)
        return
    run(args)


if __name__ == "__main__":
    main()
