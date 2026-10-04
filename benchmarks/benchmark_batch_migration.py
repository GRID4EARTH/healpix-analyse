"""Time a batch implementation using a saved, identical input fixture.

Run in old and new GEO/ANALYSE environments, alternating process order.
"""

import argparse
import importlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

radial = importlib.import_module("healpix_analyse.radial_filter")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
level, size, truncate = map(
    int, args.fixture.name.removesuffix("-input.npz").split("-")
)
data = np.load(args.fixture)
ids, values = data["ids"], data["values"]
# Initialize library/thread machinery independently of timed filter caches.
radial.gaussian_filter(values[:10], ids[:10], level, sigma_m=20, truncate=truncate)
radial._clear_filter_caches()
start = time.perf_counter()
result = radial.gaussian_filter(values, ids, level, sigma_m=20, truncate=truncate)
cold = time.perf_counter() - start
start = time.perf_counter()
repeat = radial.gaussian_filter(values + 1, ids, level, sigma_m=20, truncate=truncate)
warm = time.perf_counter() - start
np.testing.assert_allclose(repeat, result + 1, rtol=1e-12, atol=1e-12)
args.output.parent.mkdir(parents=True, exist_ok=True)
np.save(args.output, result)
print(
    json.dumps(
        {
            "level": level,
            "size_m": size,
            "cells": ids.size,
            "cold_s": cold,
            "repeat_s": warm,
            "geo_version": importlib.metadata.version("healpix-geo"),
            "analyse_module": radial.__file__,
            "cache": radial.radial_filter_cache_info(),
        }
    )
)
