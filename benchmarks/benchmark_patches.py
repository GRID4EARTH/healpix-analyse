"""Benchmark HealPixPatches: extraction, extraction + backward, fold, peak memory.

One JSON line per configuration and backend, on CPU or CUDA::

    python benchmarks/benchmark_patches.py run --device cuda --configs gpu > gpu.jsonl
    python benchmarks/benchmark_patches.py run --device cpu --configs cpu > cpu.jsonl
    python benchmarks/benchmark_patches.py table gpu.jsonl          # Markdown table

For each configuration it measures, on every backend asked for
(``--backends``, default ``auto torch``):

* ``build_s``: construction of the layer (geometry);
* ``fwd_ms``: ``y = patches(x)`` (no grad);
* ``fwdbwd_ms``: ``patches(x).backward(gy)``, gradient with respect to x;
* ``fold_ms``: ``patches.fold(y)`` with the default normalisation;
* ``peak_GiB`` (CUDA): maximum memory allocated during the training step
  above what was allocated before it (input, output gradient, geometry).

The reference is the bandwidth of a plain gather of the same size:
``gather_ms`` is ``torch.index_select`` producing as many values as the
patches (one index per value, the cheapest possible extraction) and
``gather_bwd_ms`` its forward + backward (``index_add_``).  ``ratio`` is
``fwdbwd_ms / gather_bwd_ms``: about 1 means the extraction and its backward
run at the bandwidth of the gather.  ``GBps`` is the effective bandwidth of
the forward: bytes of x read + bytes of y written + bytes of geometry read,
over ``fwd_ms``.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time

import numpy as np
import torch

CONFIGS = {
    "cpu": [
        dict(name="L6 full N4 coarse gauge G1 B4 C8", level=6, N=4, B=4, C=8),
        dict(name="L6 full N4 coarse gauge G4 B4 C8", level=6, N=4, B=4, C=8, G=4),
        dict(name="L6 full N4 all gauge G1 B2 C4", level=6, N=4, B=2, C=4, centres="all"),
        dict(name="L8 full N16 coarse gauge G1 B2 C3", level=8, N=16, B=2, C=3),
        dict(name="L8 full N16 coarse nested B2 C3", level=8, N=16, B=2, C=3, mode="nested"),
        dict(name="L8 regional 65536 N8 coarse gauge B4 C8", level=8, N=8, B=4, C=8, m=8),
        dict(name="L8 full N16 gauge B2 C3 keep 25%", level=8, N=16, B=2, C=3, keep=0.25),
    ],
    "gpu": [
        dict(name="L8 full N4 coarse gauge G1 B8 C16", level=8, N=4, B=8, C=16),
        dict(name="L8 full N4 coarse gauge G4 B8 C16", level=8, N=4, B=8, C=16, G=4),
        dict(name="L8 full N4 all gauge G1 B2 C8", level=8, N=4, B=2, C=8, centres="all"),
        dict(name="L10 full N16 coarse gauge G1 B4 C3", level=10, N=16, B=4, C=3),
        dict(name="L10 full N16 coarse nested B4 C3", level=10, N=16, B=4, C=3, mode="nested"),
        dict(name="L10 full N16 gauge B4 C3 keep 25%", level=10, N=16, B=4, C=3, keep=0.25),
        dict(name="L10 full N16 nested B4 C3 keep 25%", level=10, N=16, B=4, C=3,
             mode="nested", keep=0.25),
        dict(name="L10 regional 262144 N8 coarse gauge B16 C16", level=10, N=8, B=16, C=16, m=9),
    ],
    "gpu_short": [  # reduced A100 run: levels 8 and 10, full sphere and one regional domain
        dict(name="L8 full N4 coarse gauge G1 B8 C16", level=8, N=4, B=8, C=16),
        dict(name="L10 full N16 coarse gauge G1 B4 C3", level=10, N=16, B=4, C=3),
        dict(name="L10 full N16 coarse nested B4 C3", level=10, N=16, B=4, C=3, mode="nested"),
        dict(name="L10 full N16 gauge B4 C3 keep 25%", level=10, N=16, B=4, C=3, keep=0.25),
        dict(name="L10 regional 262144 N8 coarse gauge B16 C16", level=10, N=8, B=16, C=16, m=9),
    ],
    "gpu_masked": [  # MAE masking: the masked fold and the training step at 25 % / 50 % kept
        dict(name="L10 full N16 gauge B4 C3 keep 25%", level=10, N=16, B=4, C=3, keep=0.25),
        dict(name="L10 full N16 gauge B4 C3 keep 50%", level=10, N=16, B=4, C=3, keep=0.5),
        dict(name="L8 full N4 gauge B8 C16 keep 25%", level=8, N=4, B=8, C=16, keep=0.25),
        dict(name="L8 full N4 gauge B8 C16 keep 50%", level=8, N=4, B=8, C=16, keep=0.5),
        dict(name="L10 full N16 nested B4 C3 keep 25%", level=10, N=16, B=4, C=3,
             mode="nested", keep=0.25),
        dict(name="L10 full N16 nested B4 C3 keep 50%", level=10, N=16, B=4, C=3,
             mode="nested", keep=0.5),
    ],
    "tiny": [
        dict(name="L3 full N2 coarse gauge B2 C2", level=3, N=2, B=2, C=2),
        dict(name="L4 regional 256 N4 nested keep 25%", level=4, N=4, B=2, C=2, m=4,
             mode="nested", keep=0.25),
    ],
}


def regional_ids(level, m, face=4):
    """A complete NESTED block of 4**m cells at ``level`` inside base face ``face``,
    in a random order."""
    start = face * 4**level
    ids = np.arange(start, start + 4**m)
    return np.random.default_rng(0).permutation(ids)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeit(fn, device, reps):
    for _ in range(2):          # warm-up: autotune, caches, autograd threads
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return 1e3 * statistics.median(ts)


def cmd_run(args):
    from healpix_analyse.patches import HealPixPatches

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)
    import healpix_analyse._patches_backend as pb
    default_cl = (pb.GATHER_CL_MIN_C, pb.FOLD_CL_MIN_C)
    for cfg in CONFIGS[args.configs]:
        if args.only and not any(s in cfg["name"] for s in args.only):
            continue
        ids = regional_ids(cfg["level"], cfg["m"]) if cfg.get("m") else None
        for variant in args.backends:
            backend, _, opt = variant.partition(":")

            if opt in ("cl0", "cl1"):
                pb.GATHER_CL_MIN_C = pb.FOLD_CL_MIN_C = (1 << 30) if opt == "cl0" else 1
            else:
                # the module defaults (set from the A100 measurements)
                pb.GATHER_CL_MIN_C, pb.FOLD_CL_MIN_C = default_cl
            # "noskip" / "skip": the masked fold never / always skips the cells
            # that no selected patch reads (default: the module rule)
            if hasattr(pb, "MASKED_FOLD_SKIP"):
                pb.MASKED_FOLD_SKIP = {"noskip": False, "skip": True}.get(opt, "auto")
            rec = dict(config=cfg["name"], backend=variant, device=str(device),
                       dtype=args.dtype, torch=torch.__version__)
            try:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()
                hp = HealPixPatches(cfg["level"], cfg["N"], cell_ids=ids,
                                    centres=cfg.get("centres", "coarse"),
                                    mode=cfg.get("mode", "gauge"), n_gauges=cfg.get("G", 1),
                                    backend=backend, device=device)
                rec["build_s"] = time.perf_counter() - t0
                B, C = cfg["B"], cfg["C"]
                x = torch.randn(B, C, hp.K, device=device, dtype=dtype)
                index = None
                if cfg.get("keep"):
                    n = max(1, int(cfg["keep"] * hp.M))
                    index = torch.argsort(torch.rand(B, hp.M, device=device), 1)[:, :n]
                with torch.no_grad():
                    y = hp(x, index=index)
                gy = torch.randn_like(y)
                rec["M"], rec["out_shape"] = hp.M, list(y.shape)
                rec["fwd_ms"] = timeit(lambda: hp(x, index=index), device, args.reps)
                xg = x.clone().requires_grad_()

                def step():
                    xg.grad = None
                    hp(xg, index=index).backward(gy)

                if device.type == "cuda":
                    step()
                    sync(device)
                    base = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    step()
                    sync(device)
                    rec["peak_GiB"] = (torch.cuda.max_memory_allocated(device) - base) / 2**30
                rec["fwdbwd_ms"] = timeit(step, device, args.reps)
                rec["fold_ms"] = timeit(lambda: hp.fold(gy, index=index), device, args.reps)
                geom = hp._geometry()
                geom_bytes = (geom.idx.element_size() + geom.w.element_size()) * geom.idx.numel()
                if index is not None:
                    geom_bytes = geom_bytes * index.shape[1] // hp.M
                if hp.mode == "nested" and getattr(hp, "contiguous", False) and index is None:
                    geom_bytes = 0
                xy_bytes = x.numel() * x.element_size() + y.numel() * y.element_size()
                rec["GBps"] = (xy_bytes + geom_bytes) / (rec["fwd_ms"] * 1e-3) / 1e9
                # fold / backward: y read, x written, transposed map read
                # (ptr int64, rows int32, weights)
                if not (hp.mode == "nested" and getattr(hp, "contiguous", False)
                        and index is None):
                    ptr, rows, tw = geom.transposed()
                    map_bytes = (ptr.numel() * 8 + rows.numel() * 4
                                 + tw.numel() * tw.element_size())
                else:
                    map_bytes = 0
                rec["fold_GBps"] = (xy_bytes + map_bytes) / (rec["fold_ms"] * 1e-3) / 1e9
                rec["fwdbwd_GBps"] = (2 * xy_bytes + geom_bytes + map_bytes) / (
                    rec["fwdbwd_ms"] * 1e-3) / 1e9
                # reference: a plain gather producing y.numel() values
                flat = torch.randint(0, hp.K, (y.numel() // (B * C),), device=device)
                xv = x.view(B * C, hp.K)
                rec["gather_ms"] = timeit(lambda: xv.index_select(1, flat), device, args.reps)
                xr = xv.clone().requires_grad_()
                gr = torch.randn(B * C, flat.numel(), device=device, dtype=dtype)

                def gstep():
                    xr.grad = None
                    xr.index_select(1, flat).backward(gr)

                rec["gather_bwd_ms"] = timeit(gstep, device, args.reps)
                rec["ratio"] = rec["fwdbwd_ms"] / rec["gather_bwd_ms"]
                del hp, x, xg, y, gy, xr, gr, flat
            except Exception as exc:  # noqa: BLE001 - report and go on
                rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            print(json.dumps(rec), flush=True)


def cmd_table(args):
    rows = []
    for path in args.files:
        with open(path) as fh:
            rows += [json.loads(line) for line in fh if line.strip().startswith("{")]

    def f(v, nd=1):
        return "—" if v is None else f"{v:.{nd}f}"

    print("| configuration | backend | dtype | fwd ms | fwd+bwd ms | fold ms "
          "| gather fwd / fwd+bwd ms | fwd+bwd / gather | GB/s fwd / fwd+bwd / fold "
          "| peak GiB |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            print(f"| {r['config']} | {r['backend']} | {r['dtype']} | {r['error'][:60]} ||||||||")
            continue
        print(f"| {r['config']} | {r['backend']} | {r['dtype']} | {f(r['fwd_ms'], 2)} "
              f"| {f(r['fwdbwd_ms'], 2)} | {f(r['fold_ms'], 2)} "
              f"| {f(r['gather_ms'], 2)} / {f(r['gather_bwd_ms'], 2)} | {f(r['ratio'], 2)} "
              f"| {f(r['GBps'], 0)} / {f(r.get('fwdbwd_GBps'), 0)} / {f(r.get('fold_GBps'), 0)} "
              f"| {f(r.get('peak_GiB'), 2)} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cpu")
    r.add_argument("--configs", default="cpu", choices=sorted(CONFIGS))
    r.add_argument("--only", nargs="*")
    r.add_argument("--backends", nargs="+", default=["auto", "torch"])
    r.add_argument("--dtype", default="float32")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--threads", type=int, default=0)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
