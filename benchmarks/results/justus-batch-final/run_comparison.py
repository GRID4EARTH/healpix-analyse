import json
import os
from pathlib import Path
import random
import subprocess

old = Path('/Users/todaka/python/git/G4E/healpix-analyse')
new = Path('/tmp/healpix-analyse-batched')
out = new / 'benchmarks/results/justus-batch-final'
out.mkdir(parents=True, exist_ok=True)
script = new / 'benchmarks/benchmark_batch_migration.py'
envs = {'old': (old, old/'.pixi/envs/default/bin/python'), 'new': (new, Path('/tmp/healpix-followup-check/bin/python'))}
rng = random.Random(20260908)
rows = []
for case in ['19-600-4', '20-600-5', '19-3600-4', '20-1200-5']:
    for repetition in range(5):
        labels = ['old', 'new']
        rng.shuffle(labels)
        for label in labels:
            root, python = envs[label]
            env = dict(os.environ, PYTHONPATH=str(root), MPLCONFIGDIR='/tmp/healpix-mpl', PROJ_DATA=str(old/'.pixi/envs/default/share/proj'))
            result = subprocess.run([str(python), str(script), '--fixture', str(old/f'benchmarks/results/geo040-performance/integration/{case}-input.npz'), '--output', str(out/f'{case}-{label}-{repetition}.npy')], cwd=root, env=env, text=True, capture_output=True, check=True)
            row = dict(json.loads(result.stdout), label=label, repetition=repetition, case=case)
            rows.append(row)
            (out/'timings.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
