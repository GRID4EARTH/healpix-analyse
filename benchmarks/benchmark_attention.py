"""Benchmark the attention layers of healpix_analyse.attention (issue #109, section 3).

One JSON line per (configuration, variant), on CPU or CUDA::

    python benchmarks/benchmark_attention.py window --device cuda > window.jsonl
    python benchmarks/benchmark_attention.py neighbourhood --device cuda > na.jsonl
    python benchmarks/benchmark_attention.py receptive                 # Markdown table
    python benchmarks/benchmark_attention.py table window.jsonl na.jsonl

Subcommands
-----------
window
    ``HealPixWindowAttention`` (attention core: q, k, v given) against dense
    attention over the N cells: ``dense`` is global attention (what the
    window replaces), ``dense_masked`` the same block-diagonal attention as
    the window written with a dense ``[N, N]`` mask.  Variants
    ``window_none`` (reshape + SDPA) and ``window_centred``
    (``shift="coarse_centred"``: gathered keys + mask).
neighbourhood
    ``HealPixNeighbourhoodAttention.attend`` (fused kernel on CUDA, tap loop
    on CPU) against ``naive``: the gathered ``[B, H, G, N, P, D]`` keys and
    values materialised, softmax over the taps, autograd backward.
receptive
    Number of cells, and angular radius, of the receptive field of one cell
    after 1 to 6 stacked window layers, for each value of ``shift``.
table
    Markdown tables from the JSON lines.

Times are medians over ``--reps`` runs after warm-up: ``fwd`` without
gradients, ``step`` forward + backward of ``out.square().mean()`` (gradients
for q, k, v and the bias).  ``peak_GiB`` is the CUDA memory allocated during
the step above what was allocated before it (inputs included in the latter).
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
import torch.nn.functional as F

OOM = getattr(torch, "OutOfMemoryError", RuntimeError)


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time(fn, dev, reps, warmup=2):
    for _ in range(warmup):
        fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _measure(make_inputs, run, dev, reps):
    """fwd time, step time, peak of the step (CUDA)."""
    inputs = make_inputs()
    res = {}

    def fwd():
        with torch.no_grad():
            run(*inputs)

    def step():
        for t in inputs:
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t.grad = None
        run(*inputs).float().square().mean().backward()

    res["fwd_ms"] = 1e3 * _time(fwd, dev, reps)
    res["step_ms"] = 1e3 * _time(step, dev, reps)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        for t in inputs:
            if isinstance(t, torch.Tensor):
                t.grad = None
        base = torch.cuda.memory_allocated(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        step()
        torch.cuda.synchronize(dev)
        res["peak_GiB"] = (torch.cuda.max_memory_allocated(dev) - base) / 2**30
    return res


def _emit(rec):
    print(json.dumps(rec), flush=True)


def _qkv(B, H, N, D, dev, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(B, H, N, D, generator=g).to(dev, dtype).requires_grad_() for _ in range(3)]


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------

WINDOW_CONFIGS = {
    "cpu": [dict(level=5, w=2, B=2, H=4, D=32, region=None),
            dict(level=5, w=3, B=2, H=4, D=32, region=None),
            dict(level=6, w=3, B=1, H=4, D=32, region=None),
            dict(level=7, w=3, B=1, H=4, D=32, region=20.0)],
    "gpu": [dict(level=6, w=3, B=4, H=4, D=32, region=None),
            dict(level=7, w=3, B=4, H=4, D=32, region=None),
            dict(level=8, w=4, B=2, H=8, D=32, region=None),
            dict(level=9, w=4, B=1, H=8, D=32, region=None),
            dict(level=9, w=3, B=2, H=4, D=64, region=None),
            dict(level=10, w=4, B=1, H=4, D=32, region=25.0)],
}


def bench_window(args):
    from healpix_analyse.attention import HealPixWindowAttention
    from healpix_analyse._attention_backend import window_attention
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    for cfg in WINDOW_CONFIGS[args.configs]:
        L, w, B, H, D = cfg["level"], cfg["w"], cfg["B"], cfg["H"], cfg["D"]
        ids = None if cfg["region"] is None else _region_ids(L, cfg["region"])
        mods = {sh: HealPixWindowAttention(L, H * D, H, w, cell_ids=ids, shift=sh, device=dev)
                for sh in ("none", "coarse_centred")}
        N = mods["none"].N
        name = f"L{L} {'full' if ids is None else 'regional'} N={N} w{w} B{B} H{H} D{D}"

        def make():
            g = torch.Generator(device="cpu").manual_seed(0)
            return (torch.randn(B, N, 3, H, D, generator=g).to(dev, dtype).requires_grad_(),)

        variants = {}
        for sh, m in mods.items():
            tag = "none" if sh == "none" else "centred"
            plan = m.plan()
            for be in (("triton", "torch") if dev.type == "cuda" else ("torch",)):
                variants[f"{tag}_{'sdpa' if be == 'torch' else be}"] = (
                    (lambda plan, be: lambda qkv: window_attention(qkv, plan, backend=be))(plan, be),
                    m)

        def split(qkv):
            return [qkv[:, :, i].transpose(1, 2) for i in range(3)]

        T = 4 ** w
        if ids is None:
            variants["dense"] = (lambda qkv: F.scaled_dot_product_attention(*split(qkv)), None)

            def dense_masked(qkv):
                blk = torch.arange(N, device=dev) // T
                return F.scaled_dot_product_attention(*split(qkv),
                                                      attn_mask=blk[:, None] == blk[None, :])
            variants["dense_masked"] = (dense_masked, None)
        for vname, (run, m) in variants.items():
            if args.variants and vname not in args.variants:
                continue
            rec = dict(bench="window", config=name, variant=vname, device=dev.type,
                       dtype=args.dtype, N=N, label=args.label)
            if vname.startswith("dense") and N > args.dense_max_cells:
                rec["skipped"] = f"N > {args.dense_max_cells}"
                _emit(rec)
                continue
            try:
                rec.update(_measure(make, run, dev, args.reps))
            except OOM:
                rec["oom"] = True
            if m is not None and "fwd_ms" in rec:
                # FLOPs of q k^T and p v on the (padded) windows
                flops = 4.0 * B * H * m.n_windows * m.window_len * m.key_len * D
                rec["keys_per_window"] = m.key_len
                rec["fwd_TFLOPs"] = flops / (rec["fwd_ms"] * 1e-3) / 1e12
            _emit(rec)
            if dev.type == "cuda":
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# neighbourhood
# ---------------------------------------------------------------------------

NA_CONFIGS = {
    "cpu": [dict(level=5, k=3, G=1, B=2, H=4, D=32, region=None),
            dict(level=6, k=3, G=1, B=1, H=4, D=32, region=None),
            dict(level=6, k=5, G=1, B=1, H=4, D=32, region=None),
            dict(level=7, k=3, G=1, B=1, H=4, D=32, region=20.0)],
    "gpu": [dict(level=7, k=3, G=1, B=4, H=4, D=32, region=None),
            dict(level=8, k=3, G=1, B=2, H=8, D=32, region=None),
            dict(level=8, k=5, G=1, B=2, H=8, D=32, region=None),
            dict(level=8, k=3, G=4, B=1, H=4, D=32, region=None),
            dict(level=9, k=3, G=1, B=1, H=8, D=32, region=None),
            dict(level=10, k=3, G=1, B=1, H=4, D=32, region=25.0)],
}


def _region_ids(level, radius_deg, seed=0):
    import healpix_geo
    ids, _, _ = healpix_geo.nested.cone_coverage((10.0, 45.0), radius_deg, level)
    ids = np.asarray(ids, dtype=np.int64)
    return np.random.default_rng(seed).permutation(ids)     # unsorted on purpose


def bench_neighbourhood(args):
    from healpix_analyse.attention import HealPixNeighbourhoodAttention
    from healpix_analyse._attention_backend import neighbourhood_attention_naive
    dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    for cfg in NA_CONFIGS[args.configs]:
        L, ks, G, B, H, D = (cfg[k] for k in ("level", "k", "G", "B", "H", "D"))
        ids = None if cfg["region"] is None else _region_ids(L, cfg["region"])
        t0 = time.perf_counter()
        na = HealPixNeighbourhoodAttention(L, H * D, H, kernel_sz=ks, n_gauges=G, cell_ids=ids,
                                           device=dev, dtype=torch.float32)
        build_s = time.perf_counter() - t0
        N = na.N
        name = (f"L{L} {'full' if ids is None else 'regional'} N={N} k{ks} G{G} B{B} H{H} D{D}")
        w = na._tap_w

        def make():
            g = torch.Generator(device="cpu").manual_seed(0)
            qkv = torch.randn(B, N, 3, H, D, generator=g).to(dev, dtype).requires_grad_()
            b = na.pos_bias.detach().clone().to(dtype).requires_grad_()
            return qkv, b

        def fused(qkv, b):
            from healpix_analyse._attention_backend import neighbourhood_attention_qkv
            return neighbourhood_attention_qkv(qkv, b, na._tap_idx, w, na._valid,
                                               backend=args.backend,
                                               transposed=na._transposed_map)

        def naive(qkv, b):
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            return neighbourhood_attention_naive(q, k, v, b, na._tap_idx, w, na._valid)

        for vname, run in (("fused", fused), ("naive", naive)):
            if args.variants and vname not in args.variants:
                continue
            rec = dict(bench="neighbourhood", config=name, variant=vname, device=dev.type,
                       dtype=args.dtype, N=N, P=na.P, build_s=build_s, label=args.label)
            try:
                rec.update(_measure(make, run, dev, args.reps))
            except OOM:
                rec["oom"] = True
            if "fwd_ms" in rec:
                # compulsory traffic of the forward: read q, k, v, write the output
                esz = torch.finfo(dtype).bits // 8
                nbytes = (3 + G) * B * N * H * D * esz
                rec["fwd_GBs"] = nbytes / (rec["fwd_ms"] * 1e-3) / 1e9
            _emit(rec)
            if dev.type == "cuda":
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# receptive field
# ---------------------------------------------------------------------------

def _keys_of(m):
    """For each cell position, the positions of its keys (list of arrays)."""
    N = m.N
    if m._kv_index is None:
        T = m.window_len
        return lambda pos: np.unique((pos // T)[:, None] * T + np.arange(T))
    kv = m._kv_index.cpu().numpy()
    q_win = (np.arange(N) // m.window_len) if m.fast else \
        (m._q_inv.cpu().numpy() // m._q_index.shape[1])

    def f(pos):
        k = kv[np.unique(q_win[pos])].ravel()
        return np.unique(k[k < N])
    return f


def bench_receptive(args):
    from healpix_analyse.attention import HealPixWindowAttention, _unit_vectors
    L, w, layers = args.level, args.window_level, 6
    N = 12 * 4 ** L
    vec = _unit_vectors(np.arange(N), L, "WGS84")
    # one cell at the centre of a block (the last descendant of its child 0,
    # next to the block centre) and one on a block corner (its first cell)
    T = 4 ** w
    base = 5 * N // 12 + 7 * T                              # a block of base cell 5
    cells = {"block centre": base + T // 4 - 1, "block corner": base}
    rows = []
    for shift in ("none", "alternate_level", "coarse_centred"):
        mods = [HealPixWindowAttention(L, 8, 1, w, shift=shift, shifted=bool(i % 2), device="cpu")
                for i in range(2)]
        keys = [_keys_of(m) for m in mods]
        for cname, c in cells.items():
            sizes, radii = [], []
            for n in range(1, layers + 1):
                # a stack of n layers, shifted=False first: walk it from the output
                S = np.array([c])
                for i in reversed(range(n)):
                    S = keys[i % 2](S)
                sizes.append(S.size)
                ang = np.degrees(np.arccos(np.clip(vec[S] @ vec[c], -1, 1))).max()
                radii.append(ang)
            rows.append((shift, cname, sizes, radii))
    pix = math.degrees(math.sqrt(4 * math.pi / N))
    print(f"Receptive field of one cell, level {L} (cell size {pix:.2f} deg), windows of "
          f"4^{w} = {T} cells; cells / radius in cell sizes after n layers "
          f"(layers alternate shifted=False/True)\n")
    print("| shift | cell | " + " | ".join(f"n={n + 1}" for n in range(layers)) + " |")
    print("|---|---|" + "---|" * layers)
    for shift, cname, sizes, radii in rows:
        print(f"| {shift} | {cname} | " + " | ".join(
            f"{s} / {r / pix:.1f}" for s, r in zip(sizes, radii)) + " |")


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------

def table(args):
    recs = []
    for path in args.files:
        with open(path) as f:
            recs += [json.loads(line) for line in f if line.startswith("{")]
    for bench, ref in (("window", "centred_sdpa"), ("neighbourhood", "naive")):
        rs = [r for r in recs if r["bench"] == bench]
        if not rs:
            continue
        configs = list(dict.fromkeys((r["device"], r["dtype"], r["config"]) for r in rs))
        variants = list(dict.fromkeys(r["variant"] for r in rs))
        print(f"\n### {bench}\n")
        print("| device | config | variant | fwd ms | step ms | step/fwd | peak GiB | fwd rate "
              "| step vs " + ref + " |")
        print("|---|---|---|---|---|---|---|---|---|")
        for devc, dt, cfg in configs:
            by = {r["variant"]: r for r in rs if (r["device"], r["dtype"], r["config"]) == (devc, dt, cfg)}
            devc = devc if dt == "float32" else f"{devc} {dt}"
            refstep = by.get(ref, {}).get("step_ms")
            for v in variants:
                r = by.get(v)
                if r is None:
                    continue
                if "skipped" in r or r.get("oom"):
                    note = "OOM" if r.get("oom") else "skipped"
                    print(f"| {devc} | {cfg} | {v} | {note} | | | | | |")
                    continue
                gain = f"{refstep / r['step_ms']:.1f}x" if refstep else ""
                peak = f"{r['peak_GiB']:.2f}" if "peak_GiB" in r else ""
                rate = (f"{r['fwd_TFLOPs']:.1f} TFLOP/s" if "fwd_TFLOPs" in r else
                        f"{r['fwd_GBs']:.0f} GB/s" if "fwd_GBs" in r else "")
                print(f"| {devc} | {cfg} | {v} | {r['fwd_ms']:.2f} | {r['step_ms']:.2f} | "
                      f"{r['step_ms'] / r['fwd_ms']:.1f} | {peak} | {rate} | {gain} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("window", "neighbourhood"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--configs", choices=("cpu", "gpu"), default=None)
        p.add_argument("--dtype", default="float32")
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--variants", nargs="*", default=None)
        p.add_argument("--label", default="")
        if name == "window":
            p.add_argument("--dense-max-cells", type=int, default=200_000)
        else:
            p.add_argument("--backend", default="auto")
    p = sub.add_parser("receptive")
    p.add_argument("--level", type=int, default=8)
    p.add_argument("--window-level", type=int, default=3)
    p = sub.add_parser("table")
    p.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    if getattr(args, "configs", "x") is None:
        args.configs = "gpu" if args.device.startswith("cuda") else "cpu"
    {"window": bench_window, "neighbourhood": bench_neighbourhood,
     "receptive": bench_receptive, "table": table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
