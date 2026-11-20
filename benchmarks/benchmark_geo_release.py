"""Compare released scalar coverage with the integration batch path.

Run in each GEO environment, with the same ANALYSE checkout. The optional
scalar mode hides only cone_coverage_many, leaving all other optimizations
intact. Timings exclude fixture creation, imports and output serialization.
"""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from healpix_geo import nested

from benchmark_gaussian_filter import (
    healpix_local_coordinates, patch, synthetic_scene,
)
import healpix_analyse._neighbourhood as neighbourhood
from healpix_analyse.radial_filter import (
    _clear_filter_caches, configure_radial_filter_cache, gaussian_filter,
    radial_filter_cache_info,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-mib", type=float, help="Set each cache limit for an explicit cache-capacity experiment")
    parser.add_argument("--cases", nargs="+", default=["19:600:4", "20:600:5", "19:3600:4", "20:1200:5"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.cache_mib is not None:
        configure_radial_filter_cache(geometry_max_mib=args.cache_mib, weight_max_mib=args.cache_mib)
    batch = getattr(nested, "cone_coverage_many", None)
    original = neighbourhood._cone_candidate_csr
    candidate_times = []

    def measured(*a, **kw):
        start = time.perf_counter()
        result = original(*a, **kw)
        candidate_times.append(time.perf_counter() - start)
        return result

    neighbourhood._cone_candidate_csr = measured
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "repeats": args.repeats,
        "packages": {n: importlib.metadata.version(n) for n in ["healpix-geo", "numpy", "pyproj", "healpy"]},
        "geo_direct_url": importlib.metadata.distribution("healpix-geo").read_text("direct_url.json"),
        "cases": [],
    }
    rng = np.random.default_rng(20260908)
    try:
        # Exercise lazy initialization before measuring cold *filter-cache* runs.
        ids = patch(50, 19)
        gaussian_filter(np.ones(ids.size), ids, 19, sigma_m=20)
        for spec in args.cases:
            level, size, truncate = map(int, spec.split(":"))
            ids = patch(size, level)
            x, y = healpix_local_coordinates(ids, level)
            values = synthetic_scene(x, y)
            key = spec.replace(":", "-")
            np.savez(args.output / f"{key}-input.npz", ids=ids, values=values)
            case = {"level": level, "size_m": size, "sigma_m": 20, "truncate": truncate, "cells": ids.size, "samples": []}
            modes = ["batch", "scalar"] if batch else ["scalar"]
            for repetition in range(args.repeats):
                for mode in rng.permutation(modes):
                    if mode == "batch":
                        nested.cone_coverage_many = batch
                    elif hasattr(nested, "cone_coverage_many"):
                        del nested.cone_coverage_many
                    _clear_filter_caches()
                    gc.collect()
                    candidate_times.clear()
                    start = time.perf_counter()
                    result = gaussian_filter(values, ids, level, sigma_m=20, truncate=truncate)
                    cold = time.perf_counter() - start
                    coverage = sum(candidate_times)
                    cache = radial_filter_cache_info()
                    candidate_times.clear()
                    start = time.perf_counter()
                    repeated = gaussian_filter(values + 1, ids, level, sigma_m=20, truncate=truncate)
                    repeat = time.perf_counter() - start
                    np.testing.assert_allclose(repeated, result + 1, rtol=1e-12, atol=1e-12)
                    np.save(args.output / f"{key}-{mode}.npy", result)
                    sample = {"mode": str(mode), "repetition": repetition, "cold_s": cold, "coverage_s": coverage, "repeat_s": repeat, "repeat_coverage_calls": len(candidate_times), "cache": cache}
                    case["samples"].append(sample)
                    print(json.dumps({"case": key, **sample}), flush=True)
            report["cases"].append(case)
            (args.output / "timings.json").write_text(json.dumps(report, indent=2))
    finally:
        neighbourhood._cone_candidate_csr = original
        if batch is not None:
            nested.cone_coverage_many = batch
        _clear_filter_caches()


if __name__ == "__main__":
    main()
