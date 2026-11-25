"""Benchmark HealPixDown / HealPixUp: forward, forward + backward, peak memory.

One JSON line per (operator, level, dtype) on CPU or CUDA.  The script only
uses the public layers, so the same file measures any version of the package:
put the checkout to measure first on ``PYTHONPATH`` and label the run, e.g.::

    PYTHONPATH=/path/to/main   python benchmarks/benchmark_updown.py run --device cuda --label main
    PYTHONPATH=/path/to/branch python benchmarks/benchmark_updown.py run --device cuda --label branch \\
        --variants default csr
    python benchmarks/benchmark_updown.py table main.jsonl branch.jsonl     # Markdown gain table

Operators (``--ops``)
---------------------
down_mean, down_maxpool         ``HealPixDown(mode=...)``
down_smooth, down_smooth_cons   ``mode="smooth"`` / ``"smooth_conservative"``
up_smooth                       ``HealPixUp()`` (smooth transpose, ``col_l1``)
up_mean                         ``HealPixUp(paired_down=HealPixDown(mode="mean"))``
                                (each child takes its parent's value)

``--domain full`` is the full sphere; ``--domain regional`` is a square NESTED
patch of a quarter of base face 4 (sorted, complete parents).  ``level`` is
the fine level (the input of Down, the output of Up).  The input has
``rows = B * C`` maps ``[rows, N]``.

Measures, as in ``benchmark_healpix_conv.py``: forward time (no grad),
forward + backward time (loss ``y.square().mean()``, gradient for x), and the
peak memory of each above what is allocated before it (on CUDA: allocator
peak; on Linux CPUs: sampled resident set, run with
``MALLOC_MMAP_THRESHOLD_=65536`` for it and take times from a run without).
``buffers_MiB`` is the memory the operator keeps (index arrays, matrix).

Variants (``--variants``)
-------------------------
default   the constructor defaults of the installed version
csr       ``backend="csr"``: the CSR reference (versions with backends)
torch     ``backend="torch"``: ELL with the PyTorch gather (versions with backends)
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

OPS = ("down_mean", "down_maxpool", "down_smooth", "down_smooth_cons", "up_smooth", "up_mean")

# rows (= B * C) per fine level
ROWS = {
    "gpu": {6: 64, 7: 64, 8: 64, 9: 64, 10: 16},
    "cpu": {6: 16, 7: 16, 8: 8, 9: 8, 10: 2},
    "tiny": {3: 4, 4: 4},
}


def regional_patch(level, face=4):
    """A quarter of base face ``face`` at ``level``: sorted, complete parents."""
    n = 4 ** (level - 1)
    start = face * 4 ** level
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
    def mib(self):
        return (self.peak - self.base) / 2**20 if self.ok else None


def peak_mem(fn, device):
    """Peak above baseline, in MiB."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        fn()
        torch.cuda.synchronize(device)
        return (torch.cuda.max_memory_allocated(device) - base) / 2**20
    with _RssPeak() as r:
        fn()
    return r.mib


def _package_info():
    import healpix_analyse
    path = os.path.dirname(os.path.dirname(os.path.abspath(healpix_analyse.__file__)))
    try:
        commit = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return path, commit


def _has_backend():
    import inspect
    from healpix_analyse.down import HealPixDown
    return "backend" in inspect.signature(HealPixDown).parameters


def build(op, level, ids, device, dtype, variant):
    """The operator ``op`` at fine level ``level``; None if the variant does not exist."""
    from healpix_analyse.down import HealPixDown
    from healpix_analyse.up import HealPixUp

    kw = dict(device=device, dtype=dtype)
    if variant != "default":
        if not _has_backend():
            return None
        kw["backend"] = variant
    if op.startswith("down_"):
        mode = {"down_smooth_cons": "smooth_conservative"}.get(op, op[5:])
        return HealPixDown(level, mode=mode, cell_ids=ids, **kw)
    coarse = None if ids is None else np.unique(ids // 4)
    if op == "up_smooth":
        return HealPixUp(level - 1, cell_ids=coarse, **kw)
    down_kw = dict(kw)
    down_kw.pop("backend", None)
    down = HealPixDown(level, mode="mean", cell_ids=ids, **down_kw)
    return HealPixUp(level - 1, paired_down=down, **kw)


def cmd_run(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)
    path, commit = _package_info()
    commit = args.commit or commit
    torch.manual_seed(0)
    rows = ROWS[args.rows]
    for level in args.levels:
        ids = None if args.domain == "full" else regional_patch(level)
        for op in args.ops:
            name = f"{op} L{level} {args.domain} rows{rows[level]} {args.dtype}"
            rec = dict(label=args.label, commit=commit, package=path, config=name, op=op,
                       level=level, domain=args.domain, rows=rows[level], device=str(device),
                       dtype=args.dtype, torch=torch.__version__,
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       threads=torch.get_num_threads(), results={})
            for variant in args.variants:
                r = {}
                try:
                    t0 = time.perf_counter()
                    layer = build(op, level, ids, device, dtype, variant)
                    if layer is None:
                        continue
                    sync(device)
                    r["build_s"] = time.perf_counter() - t0
                    r["buffers_MiB"] = sum(b.numel() * b.element_size()
                                           for b in layer.buffers()) / 2**20
                    x = torch.randn(rows[level], layer.N_in, device=device).to(dtype)

                    def fwd():
                        with torch.no_grad():
                            layer(x)

                    def fwdbwd():
                        xr = x.detach().requires_grad_(True)
                        layer(xr)[0].float().square().mean().backward()

                    r["fwd_ms"] = 1e3 * timeit(fwd, device, args.reps)
                    r["fwd_peak_MiB"] = peak_mem(fwd, device)
                    r["fwdbwd_ms"] = 1e3 * timeit(fwdbwd, device, args.reps)
                    r["fwdbwd_peak_MiB"] = peak_mem(fwdbwd, device)
                    del layer, x
                except OOM:
                    r["error"] = "OOM"
                except Exception as exc:  # noqa: BLE001
                    r["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rec["results"][variant] = r
            print(json.dumps(rec), flush=True)


def _load(path):
    with open(path) as f:
        return {r["config"]: r for r in (json.loads(line) for line in f if line.startswith("{"))
                if "config" in r}


def _fmt_ms(v):
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 1 else f"{v:.3f}")


def _fmt_mib(v):
    return "n/a" if v is None else f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def _ratio(a, b):
    return f"{a / b:.1f}x" if a and b else "n/a"


def cmd_table(args):
    before, after = _load(args.before), _load(args.after)
    va, vb = args.variant_before, args.variant_after
    print("| Operator | before fwd / fwd+bwd (ms) | before peak / buffers (MiB) "
          "| after fwd / fwd+bwd (ms) | after peak / buffers (MiB) "
          "| speed-up fwd / fwd+bwd | peak ratio |")
    print("|---|---|---|---|---|---|---|")
    for name, ra in before.items():
        rb = after.get(name)
        if rb is None:
            continue
        a, b = ra["results"].get(va, {}), rb["results"].get(vb, {})
        if "error" in b or not b:
            print(f"| {name} | | | {b.get('error', 'not run')} | | | |")
            continue
        after_t = f"{_fmt_ms(b['fwd_ms'])} / {_fmt_ms(b['fwdbwd_ms'])}"
        after_m = f"{_fmt_mib(b['fwdbwd_peak_MiB'])} / {_fmt_mib(b['buffers_MiB'])}"
        if "error" in a or not a:
            err = a.get("error", "not run")
            print(f"| {name} | {err} | {err} | {after_t} | {after_m} | runs | runs |")
            continue
        sp = f"{_ratio(a['fwd_ms'], b['fwd_ms'])} / {_ratio(a['fwdbwd_ms'], b['fwdbwd_ms'])}"
        mr = _ratio(a["fwdbwd_peak_MiB"], b["fwdbwd_peak_MiB"])
        print(f"| {name} | {_fmt_ms(a['fwd_ms'])} / {_fmt_ms(a['fwdbwd_ms'])} "
              f"| {_fmt_mib(a['fwdbwd_peak_MiB'])} / {_fmt_mib(a['buffers_MiB'])} "
              f"| {after_t} | {after_m} | {sp} | {mr} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("run")
    c.add_argument("--device", default="cpu")
    c.add_argument("--levels", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    c.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    c.add_argument("--domain", default="full", choices=["full", "regional"])
    c.add_argument("--rows", default=None, choices=sorted(ROWS),
                   help="rows per level (default: gpu on CUDA, cpu otherwise)")
    c.add_argument("--variants", nargs="+", default=["default"],
                   choices=["default", "csr", "torch"])
    c.add_argument("--dtype", default="float32")
    c.add_argument("--reps", type=int, default=10)
    c.add_argument("--threads", type=int, default=0)
    c.add_argument("--label", default="")
    c.add_argument("--commit", default="", help="recorded in the output (default: git HEAD)")
    t = sub.add_parser("table")
    t.add_argument("before")
    t.add_argument("after")
    t.add_argument("--variant-before", default="default")
    t.add_argument("--variant-after", default="default")
    args = ap.parse_args()
    if args.cmd == "run" and args.rows is None:
        args.rows = "gpu" if args.device.startswith("cuda") else "cpu"
    torch.set_grad_enabled(True)
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
