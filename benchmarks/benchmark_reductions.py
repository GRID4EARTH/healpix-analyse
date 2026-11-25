"""Benchmark the neighbourhood reductions, the local gradient and HealPixDivCurl.

One JSON line per (configuration, operator), on CPU or CUDA.  The script only
uses public classes (plus the private helpers of ``gradient.py`` on versions
without ``HealPixGradient``), so the same file measures any version of the
package: put the checkout to measure first on ``PYTHONPATH`` and label the
run, e.g.::

    PYTHONPATH=/path/to/main   python benchmarks/benchmark_reductions.py run --device cuda --label main
    PYTHONPATH=/path/to/branch python benchmarks/benchmark_reductions.py run --device cuda --label branch
    python benchmarks/benchmark_reductions.py table main.jsonl branch.jsonl     # Markdown gain table

Subcommands
-----------
run
    For each level, domain and operator: build time of the operator (its
    geometry), forward time (no grad), forward + backward time (loss
    ``nan_to_num(y).square().mean()``, gradient of the input) and peak memory
    (on CUDA, the maximum allocated memory during the step above what is
    allocated before it; on Linux CPUs, the sampled resident set size above
    the one before the step).  Operators:

    ``mean`` ``median`` ``max`` ``min`` ``count``
        ``HealPixNeighbourReducer`` with a radius of 1.6 cell spacings (7 to
        9 cells per neighbourhood), ``values [B, N]``.
    ``gradient``
        Least-squares gradient of ``values [B, N]``: ``HealPixGradient`` when
        the version has it, otherwise the geometry of ``gradient()`` built
        once and its torch path applied row by row (``gradient()`` only takes
        1-D values on those versions).
    ``divcurl``
        ``HealPixDivCurl`` (kernel 3, one gauge) on ``velocity [B, 2, N]``.

    A reference implementation that is too slow is skipped: once one call
    exceeds ``--budget`` seconds, larger levels of the same operator are not
    run (recorded as ``"skipped"``).
table
    Reads JSON lines of a "before" and an "after" run and prints the gain table.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import threading
import time

import numpy as np
import torch

OOM = getattr(torch, "OutOfMemoryError", getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
REDUCTIONS = ("mean", "median", "max", "min", "count")
OPS = REDUCTIONS + ("gradient", "divcurl")


def regional_ids(level):
    """A NESTED regional domain: a quarter of base face 4 (4**(level-1) cells)."""
    n = 4 ** (level - 1)
    start = 4 * 4 ** level + n
    return np.arange(start, start + n, dtype=np.int64)


def spacing_m(level):
    return float(np.sqrt(4 * np.pi / (12 * 4 ** level)) * 6_371_008.8)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timeit(fn, device, reps, budget):
    """Median time; one call only when the first one exceeds ``budget`` s."""
    t0 = time.perf_counter()
    fn()                                   # warm-up (and Triton compilation)
    sync(device)
    first = time.perf_counter() - t0
    if first > budget:
        return first, 1
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), reps


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
        return (self.peak - self.base) / 2 ** 30 if self.ok else None


def peak_mem(fn, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        return (torch.cuda.max_memory_allocated(device) - base) / 2 ** 30
    with _RssPeak() as r:
        fn()
    return r.gib


def _package_info():
    import healpix_analyse
    path = os.path.dirname(healpix_analyse.__file__)
    try:
        commit = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    return path, commit


# ---------------------------------------------------------------------------
# Operators: build() -> apply(x) callable, input shape
# ---------------------------------------------------------------------------

_REDUCER_CACHE = {}


def _build(op, level, ids, device, dtype):
    if op in REDUCTIONS:
        from healpix_analyse import HealPixNeighbourReducer
        key = (level, len(ids))
        if key not in _REDUCER_CACHE:     # one geometry for all the reductions
            _REDUCER_CACHE.clear()
            _REDUCER_CACHE[key] = HealPixNeighbourReducer(
                ids, level, radius_m=1.6 * spacing_m(level))
        red = _REDUCER_CACHE[key]
        return (lambda x: red(x, reduction=op)), (len(ids),)
    if op == "gradient":
        import importlib
        gmod = importlib.import_module("healpix_analyse.gradient")
        if hasattr(gmod, "HealPixGradient"):
            g = gmod.HealPixGradient(ids, level)
            return (lambda x: torch.stack(g(x), 1)), (len(ids),)
        cells = gmod._normalise_ids(ids, name="cell_ids", refinement_level=level)
        geom = gmod._prepare_gradient_geometry(cells, cells.copy(), level)

        def apply(x):
            rows = [torch.stack(gmod._torch_gradient_from_geometry(row, *geom)) for row in x]
            return torch.stack(rows)

        return apply, (len(ids),)
    if op == "divcurl":
        from healpix_analyse import HealPixDivCurl
        full = len(ids) == 12 * 4 ** level
        m = HealPixDivCurl(level, None if full else ids, kernel_sz=3, device=device,
                           dtype=dtype if dtype in (torch.float32, torch.float64) else torch.float32,
                           cache_dir=None)
        return m, (2, len(ids))
    raise ValueError(op)


def cmd_run(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)
    path, commit = _package_info()
    print(json.dumps(dict(meta=True, label=args.label, package=path, commit=args.commit or commit,
                          torch=torch.__version__, device=str(device),
                          gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          threads=torch.get_num_threads())), flush=True)
    too_slow = set()
    for level in args.levels:
        for domain in args.domains:
            ids = (np.arange(12 * 4 ** level, dtype=np.int64) if domain == "full"
                   else regional_ids(level))
            for op in args.ops:
                name = f"L{level} {domain} {len(ids)} {op}"
                rec = dict(config=name, op=op, level=level, domain=domain, cells=len(ids),
                           B=args.batch, dtype=args.dtype, label=args.label)
                if (op, domain) in too_slow:
                    rec["skipped"] = f"reference over {args.budget:g} s at a lower level"
                    print(json.dumps(rec), flush=True)
                    continue
                try:
                    gc.collect()
                    t0 = time.perf_counter()
                    fn, tail = _build(op, level, ids, device, dtype)
                    rec["build_s"] = round(time.perf_counter() - t0, 3)
                    gen = torch.Generator().manual_seed(0)
                    x = torch.randn((args.batch, *tail), generator=gen).to(device=device, dtype=dtype)

                    def fwd():
                        with torch.no_grad():
                            fn(x)

                    t, reps = timeit(fwd, device, args.reps, args.budget)
                    rec["fwd_ms"], rec["fwd_reps"] = 1e3 * t, reps
                    if t > args.budget:
                        too_slow.add((op, domain))
                    if op != "count" and t <= args.budget:
                        xg = x.clone().requires_grad_(True)

                        def fwdbwd():
                            y = fn(xg)
                            torch.nan_to_num(y).square().mean().backward()
                            xg.grad = None

                        t2, reps2 = timeit(fwdbwd, device, args.reps, args.budget)
                        rec["fwdbwd_ms"], rec["fwdbwd_reps"] = 1e3 * t2, reps2
                        rec["peak_GiB"] = peak_mem(fwdbwd, device)
                        if t2 > args.budget:
                            too_slow.add((op, domain))
                    else:
                        rec["peak_GiB"] = peak_mem(fwd, device)
                    del fn, x
                except OOM as e:  # pragma: no cover - depends on the GPU
                    rec["error"] = "OOM"
                    rec["detail"] = str(e).splitlines()[0][:200]
                except Exception as e:  # noqa: BLE001 - recorded in the output
                    rec["error"] = type(e).__name__
                    rec["detail"] = str(e).splitlines()[0][:200] if str(e) else ""
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _load(path):
    with open(path) as f:
        return {r["config"]: r for r in (json.loads(line) for line in f if line.startswith("{"))
                if "config" in r}


def _ms(v):
    if v is None:
        return "–"
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 1 else f"{v:.2f}")


def _gib(v):
    return "–" if v is None else f"{v:.2f}"


def _ratio(a, b):
    return f"{a / b:,.0f}x" if a and b and a / b >= 100 else (f"{a / b:.1f}x" if a and b else "–")


def _cell(r, key):
    if r is None:
        return "not run"
    if "error" in r:
        return r["error"]
    if "skipped" in r:
        return "skipped"
    v = r.get(key)
    s = _ms(v)
    if v is not None and r.get(key.replace("_ms", "_reps")) == 1 and v > 1000:
        s += " (1 call)"
    return s


def cmd_table(args):
    before, after = _load(args.before), _load(args.after)
    print("| Configuration | before fwd / fwd+bwd (ms) | after fwd / fwd+bwd (ms) "
          "| speed-up fwd / fwd+bwd | before / after peak (GiB) | build before / after (s) |")
    print("|---|---|---|---|---|---|")
    for name, rb in after.items():
        ra = before.get(name)
        sp_f = _ratio((ra or {}).get("fwd_ms"), rb.get("fwd_ms"))
        sp_b = _ratio((ra or {}).get("fwdbwd_ms"), rb.get("fwdbwd_ms"))
        print(f"| {name} | {_cell(ra, 'fwd_ms')} / {_cell(ra, 'fwdbwd_ms')} "
              f"| {_cell(rb, 'fwd_ms')} / {_cell(rb, 'fwdbwd_ms')} | {sp_f} / {sp_b} "
              f"| {_gib((ra or {}).get('peak_GiB'))} / {_gib(rb.get('peak_GiB'))} "
              f"| {(ra or {}).get('build_s', '–')} / {rb.get('build_s', '–')} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cpu")
    r.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    r.add_argument("--domains", nargs="+", default=["full", "regional"], choices=["full", "regional"])
    r.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--dtype", default="float32")
    r.add_argument("--reps", type=int, default=5)
    r.add_argument("--budget", type=float, default=60.0,
                   help="seconds; slower calls are timed once and larger levels skipped")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--label", default="")
    r.add_argument("--commit", default="")
    t = sub.add_parser("table")
    t.add_argument("before")
    t.add_argument("after")
    args = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
