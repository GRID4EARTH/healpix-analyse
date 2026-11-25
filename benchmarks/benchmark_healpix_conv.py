"""Benchmark HealPixConv: forward, forward + backward, peak memory, geometry build.

One JSON line per configuration, on CPU or CUDA.  The script only uses the
public layer, so the same file measures any version of the package: put the
checkout to measure first on ``PYTHONPATH`` and label the run, e.g.::

    PYTHONPATH=/path/to/main   python benchmarks/benchmark_healpix_conv.py conv --device cuda --configs gpu --label main
    PYTHONPATH=/path/to/branch python benchmarks/benchmark_healpix_conv.py conv --device cuda --configs gpu --label branch \\
        --variants layer layer_det
    python benchmarks/benchmark_healpix_conv.py table main.jsonl branch.jsonl     # Markdown gain table

Subcommands
-----------
conv
    For each configuration: build time, forward time (no grad), forward +
    backward time (loss ``y.square().mean()``, gradients for x, weight and
    bias) and peak memory.  On CUDA the peak is the maximum allocated memory
    during the step above what is allocated before it (the layer, its
    geometry and the input), as in issue #90; ``*_total_GiB`` is the absolute
    peak, layer and geometry included.  On Linux CPUs the peak resident set
    size above the one before the step is sampled every 2 ms; run with
    ``MALLOC_MMAP_THRESHOLD_=65536`` so that glibc returns freed tensors to
    the system and the resident set follows the allocations.  That setting
    makes every allocation fault its pages in, so take the CPU times from a
    run without it.
geometry
    Build time of the geometry, stage by stage, for regional domains at
    several levels (no disk cache).
table
    Reads JSON lines of a "before" and an "after" run and prints the gain table.

Variants (``--variants``)
-------------------------
layer        ``conv(x)`` with the defaults of the installed version
layer_det    ``deterministic=True`` (versions with a Triton backend)
torch        ``backend="torch"``: the PyTorch tap loop (versions with backends)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

OOM = getattr(torch, "OutOfMemoryError", getattr(torch.cuda, "OutOfMemoryError", RuntimeError))

CONFIGS = {
    "cpu": [
        dict(name="L6 full k3 G4 B2 8->8", level=6, m=None, k=3, G=4, B=2, C=8, Co=8),
        dict(name="L8 regional 16384 k5 G4 B2 8->8", level=8, m=7, k=5, G=4, B=2, C=8, Co=8),
    ],
    "gpu": [
        dict(name="L7 full k3 G4 B8 16->16", level=7, m=None, k=3, G=4, B=8, C=16, Co=16),
        dict(name="L7 full k5 G4 B8 16->16", level=7, m=None, k=5, G=4, B=8, C=16, Co=16),
        dict(name="L8 full k5 G4 B4 16->16", level=8, m=None, k=5, G=4, B=4, C=16, Co=16),
        dict(name="L8 full k3 G8 B4 32->32", level=8, m=None, k=3, G=8, B=4, C=32, Co=32),
        dict(name="L10 regional 262144 k5 G4 B8 32->32", level=10, m=9, k=5, G=4, B=8, C=32, Co=32),
        dict(name="L7 full k5 G4 B8 64->64", level=7, m=None, k=5, G=4, B=8, C=64, Co=64),
    ],
    "tiny": [  # smoke test of the script itself
        dict(name="L3 full k3 G2 B2 4->4", level=3, m=None, k=3, G=2, B=2, C=4, Co=4),
        dict(name="L5 regional 256 k5 G2 B2 4->4", level=5, m=4, k=5, G=2, B=2, C=4, Co=4),
    ],
}


def nested_patch(level, m, face=4):
    """A square NESTED regional domain of 4**m cells at `level` inside base face `face`."""
    n = 4 ** m
    start = face * 4 ** level + (4 ** level // 2 // n) * n
    return np.arange(start, start + n, dtype=np.int64)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeit(fn, device, reps):
    fn()  # warm-up (and Triton autotuning)
    sync(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


class _RssPeak:
    """Peak resident set size above the starting one (Linux), sampled in a thread."""

    def __init__(self, period=0.002):
        self.period = period
        self.ok = os.path.exists("/proc/self/statm")
        self.page = os.sysconf("SC_PAGE_SIZE") if self.ok else 0

    def _rss(self):
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * self.page

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(self.period)

    def __enter__(self):
        if self.ok:
            gc.collect()
            self.base = self.peak = self._rss()
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        if self.ok:
            self.peak = max(self.peak, self._rss())
            self._stop.set()
            self._t.join()

    @property
    def gib(self):
        return (self.peak - self.base) / 2**30 if self.ok else None


def peak_mem(fn, device):
    """(peak above baseline, absolute peak) in GiB."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        return (peak - base) / 2**30, peak / 2**30
    with _RssPeak() as r:
        fn()
    return r.gib, None


def _buffers_gib(conv):
    return sum(b.numel() * b.element_size() for b in conv.buffers()) / 2**30


def _package_info():
    import healpix_analyse
    path = os.path.dirname(os.path.dirname(os.path.abspath(healpix_analyse.__file__)))
    try:
        commit = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return path, commit


def _set_variant(conv, variant):
    """Configure the layer for a variant; False if this version does not have it."""
    if not hasattr(conv, "backend"):
        return variant == "layer"
    if variant == "layer":
        conv.backend, conv.deterministic = "auto", False
    elif variant == "layer_det":
        conv.backend, conv.deterministic = "auto", True
    elif variant == "torch":
        conv.backend, conv.deterministic = "torch", False
    else:
        raise ValueError(variant)
    return True


def cmd_conv(args):
    from healpix_analyse.convol import HealPixConv

    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    path, commit = _package_info()
    commit = args.commit or commit
    torch.manual_seed(0)
    configs = CONFIGS[args.configs]
    if args.only:
        configs = [c for c in configs if any(s in c["name"] for s in args.only)]
    for cfg in configs:
        ids = None if cfg["m"] is None else nested_patch(cfg["level"], cfg["m"])
        rec = dict(label=args.label, commit=commit, package=path, config=cfg["name"],
                   device=str(device), dtype=args.dtype, torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   threads=torch.get_num_threads(), results={})
        try:
            t0 = time.perf_counter()
            conv = HealPixConv(cfg["level"], cfg["C"], cfg["Co"], kernel_sz=cfg["k"],
                               n_gauges=cfg["G"], cell_ids=ids, device=device,
                               dtype=getattr(torch, args.dtype), cache_dir=args.cache_dir)
            sync(device)
            rec["build_s"] = time.perf_counter() - t0
        except OOM:
            rec["build_error"] = "OOM"
            print(json.dumps(rec), flush=True)
            continue
        rec["K"] = conv.K
        rec["E_elems"] = cfg["B"] * cfg["C"] * cfg["G"] * conv.K * cfg["k"] ** 2
        rec["buffers_GiB"] = _buffers_gib(conv)
        conv.weight.requires_grad_(True)
        conv.bias.requires_grad_(True)
        x = torch.randn(cfg["B"], cfg["C"], conv.K, device=device, dtype=getattr(torch, args.dtype))

        for variant in args.variants:
            if not _set_variant(conv, variant):
                continue

            def fwd():
                with torch.no_grad():
                    conv(x)

            def fwdbwd():
                xr = x.detach().requires_grad_(True)
                conv.zero_grad(set_to_none=True)
                conv(xr).square().mean().backward()

            r = {}
            try:
                r["fwd_ms"] = 1e3 * timeit(fwd, device, args.reps)
                r["fwd_peak_GiB"], r["fwd_total_GiB"] = peak_mem(fwd, device)
                r["fwdbwd_ms"] = 1e3 * timeit(fwdbwd, device, args.reps)
                r["fwdbwd_peak_GiB"], r["fwdbwd_total_GiB"] = peak_mem(fwdbwd, device)
                r["bwd_over_fwd"] = (r["fwdbwd_ms"] - r["fwd_ms"]) / r["fwd_ms"]
            except OOM:
                r["error"] = "OOM"
            except Exception as exc:  # noqa: BLE001
                r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            conv.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            rec["results"][variant] = r
        print(json.dumps(rec), flush=True)
        del conv, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cmd_geometry(args):
    import healpix_analyse.convol as hc
    from healpix_analyse.convol import HealPixConv

    device = torch.device(args.device)
    path, commit = _package_info()
    timers = {}

    def wrap(mod, name, label):
        if not hasattr(mod, name):
            return
        orig = getattr(mod, name)

        def f(*a, **kw):
            sync(device)
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            sync(device)
            timers[label] = timers.get(label, 0.0) + time.perf_counter() - t0
            return out
        setattr(mod, name, f)

    wrap(hc.healpix_geo.nested, "healpix_to_lonlat", "1_healpix_to_lonlat")
    wrap(hc, "_build_rotation_matrices", "2_rotations")
    wrap(hc, "_get_interp_weights", "3_interp_weights")
    wrap(hc, "_bind_support_batched", "4_bind_support")
    wrap(hc, "tap_geometry", "5_tap_geometry")
    for level in args.levels:
        for m in args.patch_m:
            if m > level:
                continue
            for k, G in ((3, 4), (5, 4)):
                ids = nested_patch(level, m)
                timers.clear()
                t0 = time.perf_counter()
                conv = HealPixConv(level, 1, 1, kernel_sz=k, n_gauges=G, cell_ids=ids,
                                   device=device, cache_dir=None)
                sync(device)
                total = time.perf_counter() - t0
                rec = dict(label=args.label, commit=commit, level=level, K=len(ids), k=k, G=G,
                           device=str(device), total_s=total,
                           stages_s={kk: round(v, 4) for kk, v in sorted(timers.items())},
                           us_per_cell=1e6 * total / len(ids))
                print(json.dumps(rec), flush=True)
                del conv


def _load(path):
    with open(path) as f:
        return {r["config"]: r for r in (json.loads(line) for line in f if line.startswith("{"))
                if "config" in r}


def _fmt_ms(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}"


def _fmt_gib(v):
    return "n/a" if v is None else f"{v:.2f}"


def _ratio(a, b):
    return f"{a / b:.1f}x" if a and b else "n/a"


def cmd_table(args):
    before, after = _load(args.before), _load(args.after)
    va, vb = args.variant_before, args.variant_after
    print("| Configuration | before fwd / fwd+bwd (ms) | before peak (GiB) "
          "| after fwd / fwd+bwd (ms) | after peak (GiB) | after bwd / fwd "
          "| speed-up fwd / fwd+bwd | memory ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for name, ra in before.items():
        rb = after.get(name)
        if rb is None:
            continue
        a, b = ra["results"].get(va, {}), rb["results"].get(vb, {})
        if "error" in b or not b:
            print(f"| {name} | | | {b.get('error', 'not run')} | | | | |")
            continue
        after_t = f"{_fmt_ms(b['fwd_ms'])} / {_fmt_ms(b['fwdbwd_ms'])}"
        after_m = _fmt_gib(b["fwdbwd_peak_GiB"])
        ratio_b = f"{b['bwd_over_fwd']:.1f}"
        if "error" in a or not a:
            err = a.get("error", "not run")
            print(f"| {name} | {err} | {err} | {after_t} | {after_m} | {ratio_b} | runs | runs |")
            continue
        sp = f"{_ratio(a['fwd_ms'], b['fwd_ms'])} / {_ratio(a['fwdbwd_ms'], b['fwdbwd_ms'])}"
        mr = _ratio(a["fwdbwd_peak_GiB"], b["fwdbwd_peak_GiB"])
        print(f"| {name} | {_fmt_ms(a['fwd_ms'])} / {_fmt_ms(a['fwdbwd_ms'])} "
              f"| {_fmt_gib(a['fwdbwd_peak_GiB'])} | {after_t} | {after_m} | {ratio_b} | {sp} | {mr} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conv")
    c.add_argument("--device", default="cpu")
    c.add_argument("--configs", default="cpu", choices=sorted(CONFIGS))
    c.add_argument("--only", nargs="*", help="substrings of the configuration names to run")
    c.add_argument("--variants", nargs="+", default=["layer"], choices=["layer", "layer_det", "torch"])
    c.add_argument("--dtype", default="float32")
    c.add_argument("--reps", type=int, default=5)
    c.add_argument("--threads", type=int, default=0)
    c.add_argument("--cache-dir", default=None)
    c.add_argument("--label", default="")
    c.add_argument("--commit", default="", help="recorded in the output (default: git HEAD)")
    g = sub.add_parser("geometry")
    g.add_argument("--device", default="cpu")
    g.add_argument("--levels", type=int, nargs="+", default=[8, 10, 12])
    g.add_argument("--patch_m", type=int, nargs="+", default=[6, 7, 8])
    g.add_argument("--label", default="")
    t = sub.add_parser("table")
    t.add_argument("before")
    t.add_argument("after")
    t.add_argument("--variant-before", default="layer")
    t.add_argument("--variant-after", default="layer")
    args = ap.parse_args()
    torch.set_grad_enabled(True)
    {"conv": cmd_conv, "geometry": cmd_geometry, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
