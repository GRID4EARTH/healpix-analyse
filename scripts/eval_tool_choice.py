"""Measure whether an LLM picks the right healpix-analyse tool from the docs.

For every case of docs/dev/tool_choice_eval.yaml, the model receives the tool
catalogue published with the documentation (llms-full.txt) as its only source
and the user's question, and must name the tool it would propose. The script
prints the hit rate and the misses.

Usage (from the repository root, after building the docs)::

    pip install anthropic pyyaml
    python scripts/eval_tool_choice.py \
        --catalogue docs/_build/llms-full.txt --cases docs/dev/tool_choice_eval.yaml

Credentials are read by the Anthropic SDK from the environment
(ANTHROPIC_API_KEY or an `ant auth login` profile). Each run sends one request
per case and costs real money; the catalogue is sent as a cached system
prompt so that only the first request pays for it in full. This script is not
run in CI.
"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic
import yaml

INSTRUCTIONS = """\
You help users of the Python package healpix-analyse choose a tool. The
documentation below is your only source. Answer with the one public name
(function, class, or Class.method) you would propose first, then up to two
alternatives, and one sentence of justification that mentions the main limit
to watch. Use names exactly as written in the documentation.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "alternatives": {"type": "array", "items": {"type": "string"}},
        "justification": {"type": "string"},
    },
    "required": ["tool", "alternatives", "justification"],
    "additionalProperties": False,
}


def ask(client, model, catalogue, question):
    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        # opt-in server-side fallback if the model declines a request
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=[
            {"type": "text", "text": INSTRUCTIONS},
            {"type": "text", "text": catalogue, "cache_control": {"type": "ephemeral"}},
        ],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": question}],
    )
    if response.stop_reason == "refusal":
        return None
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalogue", default="docs/_build/llms-full.txt")
    parser.add_argument("--cases", default="docs/dev/tool_choice_eval.yaml")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--only", nargs="*", help="run these case ids only")
    args = parser.parse_args(argv)

    with open(args.catalogue, encoding="utf-8") as fh:
        catalogue = fh.read()
    with open(args.cases, encoding="utf-8") as fh:
        cases = yaml.safe_load(fh)["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    client = anthropic.Anthropic()
    hits, misses = 0, []
    for case in cases:
        answer = ask(client, args.model, catalogue, case["question"])
        chosen = (answer or {}).get("tool", "<refused>")
        ok = chosen in case["expected"]
        hits += ok
        print(f"{'OK  ' if ok else 'MISS'} {case['id']:<24} {chosen}")
        if not ok:
            misses.append((case, answer))

    print(f"\n{hits}/{len(cases)} first choices are right ({100 * hits / max(len(cases), 1):.0f} %)")
    for case, answer in misses:
        print(f"\n- {case['id']}: expected one of {case['expected']}")
        if answer:
            print(f"  got {answer['tool']} (alternatives {answer['alternatives']})")
            print(f"  {answer['justification']}")
    return 0 if not misses else 1


if __name__ == "__main__":
    sys.exit(main())
