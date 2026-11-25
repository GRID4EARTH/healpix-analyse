"""Benchmark the masked normalisation, pooling and positional-encoding layers.

Issue #109, sections 5 and 6.  One JSON line per (configuration, variant),
on CPU or CUDA, then a Markdown table::

    python benchmarks/benchmark_layers.py norm --device cuda > norm_gpu.jsonl
    python benchmarks/benchmark_layers.py pool --device cuda > pool_gpu.jsonl
    python benchmarks/benchmark_layers.py posenc --device cuda > posenc.jsonl
    python benchmarks/benchmark_layers.py profile --config "L8 full B4 C128"
    python benchmarks/benchmark_layers.py table norm_gpu.jsonl pool_gpu.jsonl

Subcommands
-----------
norm
    GroupNorm (32 groups), LayerNorm (over the channels of each cell) and
    BatchNorm: forward time (no grad), forward + backward time (loss
    ``(y * g).sum()`` with a fixed random ``g``, gradients for x and the
    affine parameters) and peak memory of the step (CUDA: maximum allocated
    above what is allocated before the step).  Variants:

    ``nn``      ``torch.nn`` on the same tensor, without a mask (the floor:
                it cannot skip masked cells);
    ``torch``   the masked layer, PyTorch backend;
    ``triton``  the masked layer, fused Triton kernels (CUDA only).

    The masked variants get a ``[B, N]`` bool mask with 30 % of the cells
    invalid, and NaN at those cells.  ``GBps_*`` is the useful traffic
    (forward: read x, write y; backward: read x and dy, write dx) over the
    time, to compare with the peak HBM bandwidth of the GPU.
pool
    ``global_pool`` and ``HealPixRegionPool`` (regions = parent cells 2
    levels up) in mean mode, masked, on each backend, against the unmasked
    ``x.mean(-1)``.  Useful traffic: forward read x; backward read x, write dx.
posenc
    ``healpix_positional_encoding`` on the full sphere and a regional
    domain: first call (geometry from healpix-geo) and cached call per
    backend; GB/s = bytes of the [F, N] result over the cached time.
profile
    Per-kernel CUDA times (torch.profiler) of five forward + backward steps
    of each layer on one GPU configuration.
table
    Markdown table of the JSON lines given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

CONFIGS = {
    "cpu": [
        dict(name="L6 full B4 C32", level=6, m=None, B=4, C=32),
        dict(name="L7 full B2 C64", level=7, m=None, B=2, C=64),
        dict(name="L8 regional(1/12) B4 C64", level=8, m=1, B=4, C=64),
    ],
    "gpu": [
        dict(name="L7 full B8 C64", level=7, m=None, B=8, C=64),
        dict(name="L8 full B4 C128", level=8, m=None, B=4, C=128),
        dict(name="L9 full B2 C64", level=9, m=None, B=2, C=64),
        dict(name="L10 regional(1/12) B4 C64", level=10, m=1, B=4, C=64),
        dict(name="L8 full B16 C256", level=8, m=None, B=16, C=256),
    ],
}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device, reps):
    fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _peak(fn, device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 30


def _domain(level, m):
    n = 12 * 4 ** level
    if m is None:
        return np.arange(n)
    rng = np.random.default_rng(0)
    return rng.permutation(np.arange(n // 12 * m, n // 12 * (m + 1)))


def _data(cfg, device, dtype):
    ids = _domain(cfg["level"], cfg["m"])
    N = ids.size
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(cfg["B"], cfg["C"], N, generator=g).to(device, dtype)
    mask = (torch.rand(cfg["B"], N, generator=g) > 0.3).to(device)
    xn = torch.where(mask[:, None], x, torch.full_like(x, float("nan")))
    return ids, x, xn, mask


def _emit(rec):
    print(json.dumps(rec), flush=True)


def _measure(rec, f, xin, params, device, reps, t_fwd, t_step):
    """Forward / step times, peak memory and useful bandwidth."""
    gout = torch.randn_like(f(xin).detach())
    xg = xin.detach().clone().requires_grad_()

    def fwd():
        with torch.no_grad():
            f(xin)

    def step():
        (f(xg) * gout).sum().backward()
        xg.grad = None
        for p in params:
            p.grad = None

    rec["fwd_ms"] = 1e3 * _time(fwd, device, reps)
    rec["fwdbwd_ms"] = 1e3 * _time(step, device, reps)
    rec["peak_GiB"] = _peak(step, device)
    rec["GBps_fwd"] = t_fwd / rec["fwd_ms"] * 1e-6
    rec["GBps_step"] = t_step / rec["fwdbwd_ms"] * 1e-6


def bench_norm(args):
    from healpix_analyse import HealPixBatchNorm, HealPixGroupNorm, HealPixLayerNorm
    from healpix_analyse.masked_norm import triton_available

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    variants = args.variants or (["nn", "torch", "triton"] if device.type == "cuda"
                                 and triton_available() else ["nn", "torch"])
    for cfg in CONFIGS[args.configs]:
        ids, x, xn, mask = _data(cfg, device, dtype)
        B, C, N = x.shape
        T = x.numel() * x.element_size()
        layers = {
            "GroupNorm": (lambda: nn.GroupNorm(32, C),
                          lambda be: HealPixGroupNorm(32, C, backend=be), lambda t: t),
            "LayerNorm": (lambda: nn.LayerNorm(C),
                          lambda be: HealPixLayerNorm(C, backend=be),
                          lambda t: t.transpose(1, 2)),
            "BatchNorm": (lambda: nn.BatchNorm1d(C),
                          lambda be: HealPixBatchNorm(C, backend=be), lambda t: t),
        }
        for lname, (mk_nn, mk_hp, tr) in layers.items():
            for var in variants:
                rec = dict(op=lname, config=cfg["name"], variant=var, device=args.device,
                           dtype=args.dtype, B=B, C=C, N=N)
                try:
                    if var == "nn":
                        layer = mk_nn().to(device, dtype)
                        xin = tr(x)

                        def call(xx, layer=layer):
                            return layer(xx)
                    else:
                        layer = mk_hp(var).to(device, dtype)
                        xin = xn

                        def call(xx, layer=layer):
                            return layer(xx, mask)
                    # useful traffic: fwd reads x, writes y; bwd reads x, dy, writes dx
                    _measure(rec, call, xin, list(layer.parameters()), device, args.reps,
                             2 * T, 5 * T)
                except Exception as e:  # OOM or missing backend
                    rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                _emit(rec)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del x, xn, mask


def bench_pool(args):
    from healpix_analyse import HealPixRegionPool, global_pool
    from healpix_analyse.masked_norm import triton_available

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    backends = ["torch", "triton"] if device.type == "cuda" and triton_available() else ["torch"]
    for cfg in CONFIGS[args.configs]:
        ids, x, xn, mask = _data(cfg, device, dtype)
        B, C, N = x.shape
        T = x.numel() * x.element_size()
        cases = {"mean (no mask, x.mean)": (x, lambda xx: xx.mean(-1))}
        for be in backends:
            pool = HealPixRegionPool.from_cell_ids(ids, cfg["level"], cfg["level"] - 2,
                                                   backend=be).to(device)
            cases[f"global_pool masked [{be}]"] = (
                xn, lambda xx, be=be: global_pool(xx, mask, backend=be))
            cases[f"region_pool masked [{be}]"] = (xn, lambda xx, pool=pool: pool(xx, mask))
        for name, (xin, f) in cases.items():
            rec = dict(op="pool", config=cfg["name"], variant=name, device=args.device,
                       dtype=args.dtype, B=B, C=C, N=N)
            try:
                _measure(rec, f, xin, [], device, args.reps, T, 3 * T)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            _emit(rec)
        del x, xn, mask


def bench_posenc(args):
    from healpix_analyse import clear_positional_encoding_cache, healpix_positional_encoding
    from healpix_analyse.masked_norm import triton_available

    device = torch.device(args.device)
    backends = ["torch", "triton"] if device.type == "cuda" and triton_available() else ["torch"]
    for level in args.levels:
        for m in (None, 1):
            ids = _domain(level, m)
            for gauge in (None, "projected_ref"):
                clear_positional_encoding_cache()
                t0 = time.perf_counter()
                enc = healpix_positional_encoding(level, ids, gauge_type=gauge, device=device)
                _sync(device)
                first = time.perf_counter() - t0
                for be in backends:
                    def call(be=be):
                        return healpix_positional_encoding(level, ids, gauge_type=gauge,
                                                           device=device, backend=be)
                    t = _time(call, device, 5)
                    nbytes = enc.numel() * enc.element_size()
                    _emit(dict(op="posenc", level=level,
                               domain="full" if m is None else "regional(1/12)",
                               gauge=gauge, backend=be, device=args.device, N=ids.size,
                               F=enc.shape[0], first_s=first, cached_s=t,
                               GiB=nbytes / 2 ** 30, GBps=nbytes / t * 1e-9))
                del enc
    clear_positional_encoding_cache()


def bench_profile(args):
    """Per-kernel CUDA times of five forward + backward steps of each layer."""
    from torch.profiler import ProfilerActivity, profile

    from healpix_analyse import (HealPixBatchNorm, HealPixGroupNorm, HealPixLayerNorm,
                                 HealPixRegionPool, global_pool)

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    cfg = [c for c in CONFIGS["gpu"] if c["name"] == args.config][0]
    ids, x, xn, mask = _data(cfg, device, dtype)
    B, C, N = x.shape
    T = x.numel() * x.element_size()
    print(f"config {cfg['name']}  dtype {args.dtype}  T = {T / 2**30:.3f} GiB per pass over x")
    pool = HealPixRegionPool.from_cell_ids(ids, cfg["level"], cfg["level"] - 2).to(device)
    layers = {
        "GroupNorm": HealPixGroupNorm(32, C).to(device, dtype),
        "LayerNorm": HealPixLayerNorm(C).to(device, dtype),
        "BatchNorm": HealPixBatchNorm(C).to(device, dtype),
    }
    fns = {k: (lambda xx, l=l: l(xx, mask)) for k, l in layers.items()}
    fns["global_pool"] = lambda xx: global_pool(xx, mask)
    fns["region_pool"] = lambda xx: pool(xx, mask)
    for name, f in fns.items():
        xg = xn.detach().clone().requires_grad_()
        gout = torch.randn_like(f(xn))
        for _ in range(3):
            (f(xg) * gout).sum().backward()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                (f(xg) * gout).sum().backward()
            torch.cuda.synchronize()
        print(f"== {name} (5 steps)")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10,
                                        max_name_column_width=50))


def table(args):
    rows = []
    for path in args.files:
        with open(path) as fh:
            rows += [json.loads(l) for l in fh if l.strip().startswith("{")]
    norm = [r for r in rows if r["op"] != "posenc"]
    if norm:
        print("| op | config | variant | fwd ms | fwd+bwd ms | peak GiB | fwd GB/s | step GB/s |")
        print("|---|---|---|---:|---:|---:|---:|---:|")
        for r in norm:
            if "error" in r:
                print(f"| {r['op']} | {r['config']} | {r['variant']} | {r['error']} | | | | |")
                continue
            pk = "" if r.get("peak_GiB") is None else f"{r['peak_GiB']:.2f}"
            print(f"| {r['op']} | {r['config']} | {r['variant']} | {r['fwd_ms']:.2f} "
                  f"| {r['fwdbwd_ms']:.2f} | {pk} | {r['GBps_fwd']:.0f} | {r['GBps_step']:.0f} |")
    pe = [r for r in rows if r["op"] == "posenc"]
    if pe:
        print()
        print("| level | domain | N | gauge | backend | F | first call s | cached ms | GiB | GB/s |")
        print("|---:|---|---:|---|---|---:|---:|---:|---:|---:|")
        for r in pe:
            print(f"| {r['level']} | {r['domain']} | {r['N']} | {r['gauge']} "
                  f"| {r.get('backend', 'torch')} | {r['F']} | {r['first_s']:.3f} "
                  f"| {1e3 * r['cached_s']:.2f} | {r['GiB']:.3f} | {r.get('GBps', 0):.0f} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("norm", "pool"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cpu")
        p.add_argument("--configs", default=None, choices=list(CONFIGS))
        p.add_argument("--dtype", default="float32")
        p.add_argument("--reps", type=int, default=10)
        if name == "norm":
            p.add_argument("--variants", nargs="*", default=None)
    p = sub.add_parser("posenc")
    p.add_argument("--device", default="cpu")
    p.add_argument("--levels", type=int, nargs="*", default=[6, 8, 10])
    p = sub.add_parser("profile")
    p.add_argument("--config", default="L8 full B4 C128")
    p.add_argument("--dtype", default="float32")
    p = sub.add_parser("table")
    p.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    if getattr(args, "configs", "x") is None:
        args.configs = "gpu" if args.device.startswith("cuda") else "cpu"
    {"norm": bench_norm, "pool": bench_pool, "posenc": bench_posenc,
     "profile": bench_profile, "table": table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
