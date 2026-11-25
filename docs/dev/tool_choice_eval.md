# Documentation for humans and for AI assistants

Users increasingly ask an AI assistant which function to use. The
documentation is organised so that a person and a model reach the same
answer from the same source.

## One source: `docs/tools.yaml`

Every public tool has one entry in `docs/tools.yaml`: what it does, the tasks
it answers ("I want to …"), keywords and synonyms, domain, ordering, default
ellipsoid, shapes, differentiability, GPU, NaN behaviour, cost, pros, cons,
what to use instead, optional dependencies. The local Sphinx extension
`docs/_ext/tool_catalogue.py` generates from it, at every build:

- the "I want to …" tables of the home page and of the four family pages;
- the "At a glance" box at the top of every function page;
- the API reference grouped by family;
- `llms.txt` (an index) and `llms-full.txt` (the decision guide, the
  conventions and the full catalogue as plain text), at the root of the site.

The build fails if a name of `healpix_analyse.__all__` is in no entry (or in
two), if a page does not exist, or if an answer of the evaluation set below is
not a catalogued name. When you add or change a public function, update its
entry; keep every statement traceable to the code or to a measurement, and
write "to be measured" rather than a guess.

## Evaluation: does a model pick the right tool?

`docs/dev/tool_choice_eval.yaml` holds realistic requests, phrased as users
describe their problem ("smooth SST with a 20 km Gaussian", "sample a map at
500 buoys"), each with the acceptable answers.
`scripts/eval_tool_choice.py` gives a model `llms-full.txt` as its only source
and asks it to name the tool it would propose, then prints the hit rate and
the misses:

```bash
pixi run -e docs build-docs
pip install anthropic pyyaml
python scripts/eval_tool_choice.py --catalogue docs/_build/llms-full.txt
```

It needs Anthropic API credentials and costs one request per case (the
catalogue is a cached system prompt), so it is not run in CI. Run it after
changing the catalogue; a miss usually means a task, a keyword or a "use
something else when" line is missing from an entry. Results are to be
measured: none is recorded yet.
