"""Profile the masked fold of HealPixPatches on CUDA, piece by piece.

For each configuration and each variant of the mark (none = walk over the
whole map; shift k = blocks of 2**k cells), prints the CUDA time of every
kernel of ``patches.fold(y, index=sel)`` and of the full fold, from
``torch.profiler``.  Used to choose ``MARK_SHIFT`` (#109, follow-up of #113)::

    python benchmarks/profile_masked_fold.py
"""
from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

import healpix_analyse._patches_backend as pb
from healpix_analyse.patches import HealPixPatches

CONFIGS = [
    dict(level=10, N=16, B=4, C=3),
    dict(level=8, N=4, B=8, C=16),
]


def kernel_table(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None:
            t = e.cuda_time_total
        if t > 0:
            rows.append((t / reps / 1e3, e.key[:70]))
    rows.sort(reverse=True)
    return rows


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    for cfg in CONFIGS:
        hp = HealPixPatches(cfg["level"], cfg["N"], device=dev)
        B, C = cfg["B"], cfg["C"]
        geom = hp._geometry()
        geom.transposed()
        for keep in (0.25, 0.5):
            n = int(keep * hp.M)
            sel = torch.argsort(torch.rand(B, hp.M, device=dev), 1)[:, :n]
            y = torch.randn(B, n, hp.G, C, cfg["N"], cfg["N"], device=dev)
            for shift in (None, 0, 2, 4):
                if shift is None:
                    pb.MASKED_FOLD_SKIP = False
                    label = "noskip"
                else:
                    pb.MASKED_FOLD_SKIP = True
                    pb.MARK_SHIFT = shift
                    geom._footprint = None
                    label = f"shift {shift} (F = {geom.footprint().shape[1]})"
                rows = kernel_table(lambda: hp.fold(y, index=sel))
                total = sum(r[0] for r in rows)
                frac = ""
                if shift is not None:
                    mk = pb.selection_mark(geom, sel.int())
                    frac = f", marked {float(mk[:, :-1].float().mean()):.2f}"
                print(f"\n== L{cfg['level']} N{cfg['N']} B{B} C{C} keep {keep:.0%} {label}"
                      f": CUDA total {total:.3f} ms{frac}")
                for t, k in rows[:8]:
                    print(f"   {t:8.3f} ms  {k}")
        pb.MASKED_FOLD_SKIP = True
        yf = torch.randn(B, hp.M, hp.G, C, cfg["N"], cfg["N"], device=dev)
        rows = kernel_table(lambda: hp.fold(yf))
        print(f"\n== L{cfg['level']} N{cfg['N']} full fold: CUDA total "
              f"{sum(r[0] for r in rows):.3f} ms")
        for t, k in rows[:6]:
            print(f"   {t:8.3f} ms  {k}")


if __name__ == "__main__":
    main()
