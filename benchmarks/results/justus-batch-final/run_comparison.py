"""Run benchmark_batch_migration.py in two environments, alternating order.

Example:

    python run_comparison.py \
        --old-root ../healpix-analyse-old --old-python ../old-env/bin/python \
        --new-root . --new-python .pixi/envs/default/bin/python \
        --fixtures path/to/fixtures --output results
"""

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--old-root", type=Path, required=True)
parser.add_argument("--old-python", type=Path, required=True)
parser.add_argument("--new-root", type=Path, required=True)
parser.add_argument("--new-python", type=Path, required=True)
parser.add_argument(
    "--fixtures",
    type=Path,
    required=True,
    help="directory with the {case}-input.npz files",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=5)
args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)
script = args.new_root / "benchmarks/benchmark_batch_migration.py"
envs = {
    "old": (args.old_root, args.old_python),
    "new": (args.new_root, args.new_python),
}
rng = random.Random(20260908)
rows = []
for case in ["19-600-4", "20-600-5", "19-3600-4", "20-1200-5"]:
    for repetition in range(args.repetitions):
        labels = ["old", "new"]
        rng.shuffle(labels)
        for label in labels:
            root, python = envs[label]
            result = subprocess.run(
                [
                    str(python),
                    str(script),
                    "--fixture",
                    str(args.fixtures / f"{case}-input.npz"),
                    "--output",
                    str(args.output / f"{case}-{label}-{repetition}.npy"),
                ],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(root)},
                text=True,
                capture_output=True,
                check=True,
            )
            row = dict(
                json.loads(result.stdout), label=label, repetition=repetition, case=case
            )
            rows.append(row)
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
