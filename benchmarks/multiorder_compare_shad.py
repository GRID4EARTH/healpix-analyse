"""Reproduce the real-data numbers of the multiorder_filter_operator PR.

Loads the Shad leaf set (5577 leaves, levels 11 to 15) and the prototype graph
(its leaf contacts and centre-to-centre lengths), builds
``multiorder_filter_operator`` and prints:

* the contacts: leaf contacts found by the library against the prototype's;
* distances, with the water contacts and with every geometric contact;
* the kernel at sigma = 20 days (sigma_h = 20 / sqrt(24) km, truncate 6)
  against the prototype kernel (Gaussian of a scipy Dijkstra over the
  prototype graph, leaf area times wet fraction, rows normalised).

Needs numpy, scipy and healpix-analyse (with healpix-geo and torch).

    python compare_shad.py                       # downloads from the HTTPS folder
    python compare_shad.py --data /path/to/dir   # or HPA_MULTIORDER_DATA=/path/to/dir
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import tempfile
from pathlib import Path

import numpy as np

URL = "https://data.grid4earth.eu/tmp/healpix-analyse-multiorder-2026-11-22/"
FILES = ("shad_leafset.npz", "shad_prototype_graph.npz")
SIGMA_DAY = 20.0
TRUNCATE = 6.0


def locate(data=None):
    """Folder holding the two .npz files: a local one, else a download."""
    data = data or os.environ.get("HPA_MULTIORDER_DATA")
    if data:
        return Path(data)
    folder = Path(tempfile.gettempdir()) / "healpix-analyse-multiorder-2026-11-22"
    folder.mkdir(exist_ok=True)
    for name in FILES:
        if not (folder / name).exists():
            urllib.request.urlretrieve(URL + name, folder / name)
    return folder


def load(folder):
    leaf = np.load(Path(folder) / "shad_leafset.npz")
    graph = np.load(Path(folder) / "shad_prototype_graph.npz")
    return leaf, graph


def compare(folder, sigma_day=SIGMA_DAY, verbose=True):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    from healpix_analyse import multiorder_filter_operator
    from healpix_analyse.multiorder_filter import _water_pairs, leaf_adjacency

    leaf, graph = load(folder)
    levels = leaf["levels"].astype(np.int64)
    ids = leaf["cell_ids"]
    wet = leaf["wet_fraction"]
    n = levels.size
    first, second = graph["first"].astype(np.int64), graph["second"].astype(np.int64)
    length = graph["length_km"]
    result = {"n": n}

    ours_first, ours_second = leaf_adjacency(levels, ids)[:2]
    ours = {(int(a), int(b)) for a, b in zip(ours_first, ours_second)}
    proto = {(int(a), int(b)) for a, b in zip(first, second)}
    result["contacts_library"] = len(ours)
    result["contacts_prototype"] = len(proto)
    result["contacts_only_library"] = len(ours - proto)
    result["contacts_only_prototype"] = len(proto - ours)

    sigma_km = sigma_day / np.sqrt(24.0)
    graph_matrix = coo_matrix(
        (np.r_[length, length], (np.r_[first, second], np.r_[second, first])),
        shape=(n, n),
    ).tocsr()
    dist = dijkstra(graph_matrix, directed=False)
    area = 4 * np.pi * 6371.0072**2 / (12 * 4.0**levels)
    kernel = np.exp(-(dist**2) / (2 * sigma_km**2)) * (area * wet)[None, :]
    kernel /= kernel.sum(1, keepdims=True)  # rows = source, sum 1

    for label, contacts in (
        ("every geometric contact", None),
        ("prototype (water) contacts", (first, second)),
    ):
        rows, cols, d = _water_pairs(
            levels, ids, 8 * sigma_km * 1000, "none",
            np.zeros(n, bool), contacts, "WGS84",
        )
        ref = dist[rows, cols]
        use = ref > 0
        ratio = d[use] / 1000 / ref[use]
        result[label] = {
            "pairs": int(rows.size),
            "ratio_median": float(np.median(ratio)),
            "ratio_p1": float(np.percentile(ratio, 1)),
            "ratio_p99": float(np.percentile(ratio, 99)),
            "share_more_than_0.1pct_shorter": float((d / 1000 < ref * 0.999).mean()),
        }
        op = multiorder_filter_operator(
            levels, ids, sigma_m=sigma_km * 1000, truncate=TRUNCATE,
            normalization="source", wet_fraction=wet, water_contacts=contacts,
        )
        m = np.zeros((n, n))
        r = np.repeat(np.arange(n), np.diff(op.row_offsets))
        np.add.at(m, (r, op.column_indices), op.weights)
        diff = np.abs(m.T - kernel)  # operator[target, source] -> rows = source
        result[label].update(
            kernel_max_abs_diff=float(diff.max()),
            kernel_mean_row_l1=float(diff.sum(1).mean()),
            kernel_max_row_l1=float(diff.sum(1).max()),
        )
    if verbose:
        print(f"leaves {n}, levels {dict(zip(*np.unique(levels, return_counts=True)))}")
        print(
            f"leaf contacts: library {result['contacts_library']}, prototype "
            f"{result['contacts_prototype']}; geometric contacts without joined "
            f"water {result['contacts_only_library']} of {result['contacts_library']}"
        )
        for label in ("every geometric contact", "prototype (water) contacts"):
            x = result[label]
            print(
                f"{label}: {x['pairs']} pairs; distance ratio library/prototype "
                f"median {x['ratio_median']:.5f}, 1 % {x['ratio_p1']:.4f}, "
                f"99 % {x['ratio_p99']:.4f}; pairs more than 0.1 % shorter: "
                f"{x['share_more_than_0.1pct_shorter']:.1%}"
            )
            print(
                f"   sigma {sigma_day:g} days kernel: max |diff| "
                f"{x['kernel_max_abs_diff']:.3e}, mean row L1 "
                f"{x['kernel_mean_row_l1']:.3e}, max row L1 {x['kernel_max_row_l1']:.3e}"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", help="folder with the .npz files (default: download)")
    compare(locate(parser.parse_args().data))
