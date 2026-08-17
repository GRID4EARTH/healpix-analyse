"""Reproducible benchmark for the connected-component topology adapter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Callable

import healpy as hp
import numpy as np
from healpix_geo import nested as healpix_geo_nested

from healpix_analyse import _topology as topology

EDGE_INDICES = np.asarray([0, 2, 4, 6], dtype=np.intp)


def previous_healpy_adapter(
    cell_ids,
    refinement_level: int,
    *,
    connectivity: str,
) -> np.ndarray:
    """Reproduce the adapter removed by PR #42, including its conversions."""

    refinement_level = topology._validate_refinement_level(refinement_level)
    cells = topology._as_cell_ids(cell_ids)

    if connectivity not in ("edge", "edge_or_vertex"):
        raise ValueError("connectivity must be 'edge' or 'edge_or_vertex'")

    width = 4 if connectivity == "edge" else 8
    if cells.size == 0:
        return np.empty((0, width), dtype=np.int64)

    npix = topology._npix(refinement_level)
    if np.any(cells >= npix):
        raise ValueError("cell_ids contains an identifier outside the valid range")

    neighbours = np.asarray(
        hp.get_all_neighbours(
            1 << refinement_level,
            cells.astype(np.int64, copy=False),
            nest=True,
        ),
        dtype=np.int64,
    )

    if neighbours.ndim == 1:
        neighbours = neighbours.reshape(8, 1)

    expected_shape = (8, cells.size)
    if neighbours.shape != expected_shape:
        raise RuntimeError(
            "Unexpected shape returned by healpy.get_all_neighbours: "
            f"{neighbours.shape}; expected {expected_shape}"
        )

    if connectivity == "edge":
        neighbours = neighbours[EDGE_INDICES]

    return neighbours.T.copy()


def elapsed_ms(function: Callable[[], np.ndarray], calls: int) -> float:
    """Return elapsed milliseconds per call over one sample."""

    started = time.perf_counter_ns()
    for _ in range(calls):
        function()
    return (time.perf_counter_ns() - started) / calls / 1e6


def measure(
    functions: dict[str, Callable[[], np.ndarray]],
    *,
    calls_per_sample: int,
    repetitions: int,
    warmups: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Measure methods in shuffled order to reduce ordering and thermal bias."""

    for function in functions.values():
        for _ in range(warmups):
            function()

    samples = {name: [] for name in functions}
    order = list(functions)
    random_generator = random.Random(seed)

    for _ in range(repetitions):
        random_generator.shuffle(order)
        for name in order:
            samples[name].append(elapsed_ms(functions[name], calls_per_sample))

    return {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }


def environment() -> dict[str, str]:
    """Return enough environment metadata to interpret the timings."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "healpy": hp.__version__,
        "healpix_geo": importlib.metadata.version("healpix-geo"),
    }


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run correctness checks and collect benchmark samples."""

    random_generator = np.random.default_rng(args.seed)
    maximum_cell = 12 * (1 << args.depth) ** 2
    rows = []

    for size in args.sizes:
        cells = random_generator.integers(
            0,
            maximum_cell,
            size=size,
            dtype=np.int64,
        ).astype(args.dtype, copy=False)

        calls_per_sample = max(
            args.minimum_calls,
            args.sample_cells // size,
        )

        for connectivity in ("edge", "edge_or_vertex"):
            functions = {
                "previous_healpy_adapter": (
                    lambda cells=cells, connectivity=connectivity: previous_healpy_adapter(
                        cells,
                        args.depth,
                        connectivity=connectivity,
                    )
                ),
                "new_adapter": (
                    lambda cells=cells, connectivity=connectivity: topology.nested_neighbours(
                        cells,
                        args.depth,
                        connectivity=connectivity,
                    )
                ),
                "backend_auto": (
                    lambda cells=cells, connectivity=connectivity: healpix_geo_nested.neighbours(
                        cells,
                        args.depth,
                        connectivity=connectivity,
                        num_threads=0,
                    )
                ),
                "backend_one_thread": (
                    lambda cells=cells, connectivity=connectivity: healpix_geo_nested.neighbours(
                        cells,
                        args.depth,
                        connectivity=connectivity,
                        num_threads=1,
                    )
                ),
            }

            expected = functions["previous_healpy_adapter"]()
            np.testing.assert_array_equal(functions["new_adapter"](), expected)
            np.testing.assert_array_equal(functions["backend_auto"](), expected)
            np.testing.assert_array_equal(functions["backend_one_thread"](), expected)

            timings = measure(
                functions,
                calls_per_sample=calls_per_sample,
                repetitions=args.repetitions,
                warmups=args.warmups,
                seed=args.seed + size,
            )
            previous_ms = timings["previous_healpy_adapter"]["median_ms"]
            new_ms = timings["new_adapter"]["median_ms"]

            rows.append(
                {
                    "cells": size,
                    "connectivity": connectivity,
                    "calls_per_sample": calls_per_sample,
                    "timings": timings,
                    "speedup": previous_ms / new_ms,
                }
            )

    return {
        "environment": environment(),
        "configuration": {
            "depth": args.depth,
            "sizes": args.sizes,
            "dtype": args.dtype,
            "seed": args.seed,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "sample_cells": args.sample_cells,
            "minimum_calls": args.minimum_calls,
        },
        "results": rows,
    }


def print_report(report: dict[str, object]) -> None:
    """Print environment metadata and a Markdown-compatible result table."""

    environment_values = report["environment"]
    configuration = report["configuration"]

    print(json.dumps(environment_values, indent=2))
    print(json.dumps(configuration, indent=2))
    print()
    print(
        "| cells | connectivity | previous (ms) | new (ms) | speedup | "
        "backend auto (ms) | backend 1 thread (ms) |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")

    for row in report["results"]:
        timings = row["timings"]
        print(
            f"| {row['cells']:,} | {row['connectivity']} | "
            f"{timings['previous_healpy_adapter']['median_ms']:.3f} | "
            f"{timings['new_adapter']['median_ms']:.3f} | "
            f"{row['speedup']:.2f}x | "
            f"{timings['backend_auto']['median_ms']:.3f} | "
            f"{timings['backend_one_thread']['median_ms']:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000, 10_000, 100_000, 1_000_000],
    )
    parser.add_argument("--dtype", choices=("uint64", "int64"), default="uint64")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--sample-cells",
        type=int,
        default=1_000_000,
        help="Minimum processed cells per timing sample for small calls.",
    )
    parser.add_argument(
        "--minimum-calls",
        type=int,
        default=10,
        help="Minimum calls per timing sample, including large inputs.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        parser.error("--sizes values must be positive")
    if args.repetitions <= 0 or args.warmups < 0:
        parser.error("--repetitions must be positive and --warmups non-negative")
    if args.sample_cells <= 0 or args.minimum_calls <= 0:
        parser.error("--sample-cells and --minimum-calls must be positive")

    report = benchmark(args)
    print_report(report)

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
