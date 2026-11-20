"""Validate the adapted checkout against healpy, including high levels."""
import json
from pathlib import Path
import numpy as np
import healpy as hp
from healpix_analyse._topology import nested_neighbours

rng = np.random.default_rng(20260908)
rows = []
for level in [0, 1, 3, 6, 16, 20, 29]:
    count = 12 * 4**level
    ids = np.arange(count, dtype=np.int64) if level <= 6 else rng.integers(0, count, size=10000, dtype=np.int64)
    expected = hp.get_all_neighbours(2**level, ids, nest=True).T
    for connectivity in ["edge", "edge_or_vertex"]:
        result = nested_neighbours(ids, level, connectivity=connectivity)
        ref = expected[:, [0, 2, 4, 6]] if connectivity == "edge" else expected
        np.testing.assert_array_equal(result, ref)
        assert result.dtype == np.int64 and result.flags.c_contiguous
        rows.append({"level": level, "cells": ids.size, "connectivity": connectivity, "exact_match": True, "missing_positions": int(np.sum(result == -1))})
path = Path(__file__).with_name("topology-validation.json")
path.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
