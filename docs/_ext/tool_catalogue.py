"""Sphinx extension: the tool catalogue (``docs/tools.yaml``).

The catalogue is the single source of truth for "which tool for which task".
At ``config-inited`` this extension

1. checks it against the public API: every name in
   ``healpix_analyse.__all__`` must be covered by exactly one catalogue entry,
   every entry must point to an existing page, and every answer of the
   tool-choice evaluation set (``docs/dev/tool_choice_eval.yaml``) must be a
   catalogued symbol;
2. writes the generated Markdown fragments into ``docs/_generated/``:

   - ``decisions_<family>.md``: the "I want to ... -> use ..." table of each
     family landing page;
   - ``decisions_all.md``: the same rows for every family, for the index;
   - ``glance/<id>.md``: the "At a glance" box of each tool (what it does,
     inputs and outputs, pros, cons, what to use instead, cost,
     differentiability, optional dependencies), included at the top of its
     page;
   - ``api_families.md``: the API reference, grouped by family, as
     ``autoapisummary`` tables built from the qualified names of the public
     symbols;
   - ``ai_blocks.md``, ``ai_architectures.md`` and ``ai_cards.md``: the
     "Basic tools for AI" tables and API cards (``docs/ai/``), from the
     ``ai:`` fields of the tools and the ``ai:`` section; the signatures of
     the cards are read from the source code with :mod:`ast`;
3. writes redirect stubs for the pages moved by the docs overhaul (the
   ``redirects`` mapping of the catalogue), and ``llms.txt`` and
   ``llms-full.txt`` (https://llmstxt.org) into
   ``docs/_generated/extra/``, which ``html_extra_path`` copies to the root of
   the built site.

Any inconsistency raises an error, so the build fails (with or without
``-W``). The package is never imported: ``__all__`` and the import aliases are
read from ``healpix_analyse/__init__.py`` with :mod:`ast`.
"""

from __future__ import annotations

import ast
import copy
import os
import textwrap
from collections import Counter

import yaml
from sphinx.errors import ExtensionError

FAMILY_ORDER = ("A", "B", "C", "D")
STATUSES = ("stable", "experimental", "unused", "legacy")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _public_api(init_path):
    """Return ``(__all__, {public name: qualified name})`` from ``__init__.py``."""
    with open(init_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names, qual = None, {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level:  # "from .x import y"
                module = "healpix_analyse." + (node.module or "")
            else:
                module = node.module
            for alias in node.names:
                qual[alias.asname or alias.name] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = [ast.literal_eval(e) for e in node.value.elts]
    if names is None:
        raise ExtensionError("tool catalogue: no __all__ in healpix_analyse/__init__.py")
    return names, qual


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check(cat, public, qual, srcdir, eval_set, require_complete=True):
    errors = []
    families = cat["families"]
    seen = Counter()
    for tool in cat["tools"]:
        tid = tool.get("id", "?")
        for key in ("id", "family", "page", "summary", "tasks", "pros", "cons",
                    "use_instead", "domain", "differentiable", "cost"):
            if key not in tool:
                errors.append(f"{tid}: missing field {key!r}")
        if tool.get("status", "stable") not in STATUSES:
            errors.append(f"{tid}: status {tool.get('status')!r} is not one of {STATUSES}")
        if tool.get("family") not in families:
            errors.append(f"{tid}: unknown family {tool.get('family')!r}")
        page = tool.get("page", "")
        if not os.path.exists(os.path.join(srcdir, page.split("#")[0] + ".md")):
            errors.append(f"{tid}: page {page!r} does not exist")
        for name in tool.get("symbols", []):
            seen[name] += 1
            if name not in public:
                errors.append(f"{tid}: {name!r} is not in healpix_analyse.__all__ "
                              "(list it under 'not_exported' with its module path)")
        for item in tool.get("use_instead", []):
            if "page" in item and not os.path.exists(
                os.path.join(srcdir, item["page"].split("#")[0] + ".md")
            ):
                errors.append(f"{tid}: use_instead page {item['page']!r} does not exist")
    for name in public:
        if seen[name] == 0:
            if require_complete:
                errors.append(f"public symbol {name!r} is in no catalogue entry")
        elif seen[name] > 1:
            errors.append(f"public symbol {name!r} is in {seen[name]} catalogue entries")
    for old, new in (cat.get("redirects") or {}).items():
        if os.path.exists(os.path.join(srcdir, old + ".md")):
            errors.append(f"redirect {old!r}: a page with that name exists")
        if not os.path.exists(os.path.join(srcdir, new + ".md")):
            errors.append(f"redirect {old!r}: target page {new!r} does not exist")
    for fam in families.values():
        if not os.path.exists(os.path.join(srcdir, fam["page"] + ".md")):
            errors.append(f"family page {fam['page']!r} does not exist")

    known = set(public)
    for tool in cat["tools"]:
        known.update(q.rsplit(".", 1)[-1] for q in tool.get("not_exported", []))
        known.update(tool.get("also", []))
    for case in (eval_set or {}).get("cases", []):
        for ans in case.get("expected", []):
            if ans.split(".")[0] not in known:
                errors.append(f"eval case {case.get('id')!r}: expected answer {ans!r} "
                              "is not a catalogued symbol")
    errors += _check_ai(cat, known, srcdir)
    if errors:
        raise ExtensionError("tool catalogue (docs/tools.yaml) is inconsistent:\n  - "
                             + "\n  - ".join(errors))


AI_FIELDS = ("role", "does", "api", "shapes", "grad", "gpu_kernel", "deterministic")


def _check_ai(cat, known, srcdir):
    """Checks of the AI section and of the ``ai:`` field of the tools."""
    errors = []
    ai = cat.get("ai")
    if not ai:
        return errors
    for key in ("page", "data_page"):
        if not os.path.exists(os.path.join(srcdir, ai[key] + ".md")):
            errors.append(f"ai.{key}: page {ai[key]!r} does not exist")
    roles = ai.get("roles", {})
    for tool in _ai_tools(cat):
        tid, spec = tool["id"], tool["ai"]
        for key in AI_FIELDS:
            if key not in spec:
                errors.append(f"{tid}: missing field ai.{key}")
        if spec.get("role") not in roles:
            errors.append(f"{tid}: ai.role {spec.get('role')!r} is not one of {sorted(roles)}")
        for qual in spec.get("api", []):
            try:
                _signatures(srcdir, qual)
            except (OSError, LookupError) as exc:
                errors.append(f"{tid}: ai.api {qual!r}: {exc}")
    ids = set()
    for rec in ai.get("recipes", []):
        ids.add(rec["id"])
        for key in ("page", "script"):
            path = rec[key] + (".md" if key == "page" else "")
            if not os.path.exists(os.path.join(srcdir, path)):
                errors.append(f"ai recipe {rec['id']!r}: {key} {rec[key]!r} does not exist")
        for name in rec.get("uses", []):
            if name.split(".")[0] not in known:
                errors.append(f"ai recipe {rec['id']!r}: {name!r} is not a catalogued symbol")
    for arch in ai.get("architectures", []):
        if arch.get("recipe") and arch["recipe"] not in ids:
            errors.append(f"ai architecture {arch['name']!r}: unknown recipe {arch['recipe']!r}")
    return errors


# ---------------------------------------------------------------------------
# AI building blocks: signatures read from the source code
# ---------------------------------------------------------------------------


def _ai_tools(cat):
    return [t for t in cat["tools"] if t.get("ai")]


def _fmt_args(fn, skip_first):
    """The argument list of ``fn``, without ``self`` / ``cls`` when ``skip_first``."""
    args = copy.deepcopy(fn.args)
    if skip_first and args.args:
        args.args = args.args[1:]
        # defaults belong to the last arguments: drop the one of self, if any
        if len(args.defaults) > len(args.args):
            args.defaults = args.defaults[len(args.defaults) - len(args.args):]
    return ast.unparse(args)


def _signatures(srcdir, qual):
    """Signature lines of ``healpix_analyse.module.Name``, read with :mod:`ast`.

    A class gives its constructor and its ``forward`` (when defined), a
    function its own signature. Raises LookupError when the name is absent.
    """
    parts = qual.split(".")
    if parts[0] != "healpix_analyse" or len(parts) != 3:
        raise LookupError("expected healpix_analyse.<module>.<name>")
    path = os.path.join(srcdir, "..", *parts[:2]) + ".py"
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    name = parts[2]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [f"{name}({_fmt_args(node, False)})"]
        if isinstance(node, ast.ClassDef) and node.name == name:
            methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            # forward inherited from a base class of the same module
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            for base in (n for n in tree.body
                         if isinstance(n, ast.ClassDef) and n.name in bases):
                for m in base.body:
                    if isinstance(m, ast.FunctionDef) and m.name == "forward":
                        methods.setdefault("forward", m)
            out = []
            if "__init__" in methods:
                out.append(f"{name}({_fmt_args(methods['__init__'], True)})")
            if "forward" in methods:
                out.append(f"{name}.forward({_fmt_args(methods['forward'], True)})")
            elif "__call__" in methods:  # plain callable operators (not nn.Module)
                out.append(f"{name}.__call__({_fmt_args(methods['__call__'], True)})")
            return out or [f"{name}(...)"]
    raise LookupError(f"{name} not defined in {'/'.join(parts[:2])}.py")


def _ai_table(cat):
    """The building-block table of docs/ai/index.md."""
    roles = cat["ai"]["roles"]
    out = ["| Block | Role | What it does | Shapes | Differentiable | GPU kernel "
           "| Deterministic backward | Page |\n|---|---|---|---|---|---|---|---|\n"]
    for role in roles:
        for tool in (t for t in _ai_tools(cat) if t["ai"]["role"] == role):
            a = tool["ai"]
            names = ", ".join(f"`{q.rsplit('.', 1)[-1]}`" for q in a["api"])
            out.append(f"| {names} | {role} | {_cell(a['does'])} | {_cell(a['shapes'])} "
                       f"| {_cell(a['grad'])} | {_cell(a['gpu_kernel'])} "
                       f"| {_cell(a['deterministic'])} | {_doc_link(tool['page'])} |\n")
    return "".join(out)


def _ai_architectures(cat, link):
    recipes = {r["id"]: r for r in cat["ai"].get("recipes", [])}
    out = ["| Architecture | Blocks | Recipe |\n|---|---|---|\n"]
    for arch in cat["ai"].get("architectures", []):
        rec = recipes.get(arch.get("recipe"))
        out.append(f"| {_cell(arch['name'])} | {_cell(arch['blocks'])} | "
                   + (link(rec) if rec else "—") + " |\n")
    return "".join(out)


def _ai_card(tool, srcdir, md=True):
    """API card of one AI building block (Markdown for the docs, text for llms)."""
    a = tool["ai"]
    sigs = [s for q in a["api"] for s in _signatures(srcdir, q)]
    imports = ", ".join(q.rsplit(".", 1)[-1] for q in a["api"])
    lines = [f"Role: {a['role']} — {_cell(a['does'])}",
             f"Import: from healpix_analyse import {imports}",
             f"Shapes: {_cell(a['shapes'])}",
             f"GPU and device: {_cell(tool.get('gpu', ''))}",
             f"Gradient: {_cell(a['grad'])}",
             f"GPU kernel: {_cell(a['gpu_kernel'])}",
             f"Deterministic backward: {_cell(a['deterministic'])}"]
    if tool.get("nan"):
        lines.append(f"Missing values (NaN): {_cell(tool['nan'])}")
    notes = a.get("notes", [])
    if md:
        out = [f"### {tool['title']}\n\n```python\n" + "\n".join(sigs) + "\n```\n\n"]
        out += [f"- {line}\n" for line in lines]
        out += [f"- Note: {_cell(n)}\n" for n in notes]
        out.append(f"- Page: {_doc_link(tool['page'])}\n\n")
        return "".join(out)
    out = [f"\n### {tool['title']}\n\n"]
    out += ["Signature: " + s + "\n" for s in sigs]
    out += [line + "\n" for line in lines]
    out += [f"Note: {_cell(n)}\n" for n in notes]
    return "".join(out)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _names(tool):
    names = list(tool.get("symbols", []))
    names += [q.rsplit(".", 1)[-1] for q in tool.get("not_exported", [])]
    names += tool.get("also", [])
    return names


def _cell(text):
    if isinstance(text, bool):  # YAML reads a bare yes / no as a boolean
        text = "yes" if text else "no"
    return " ".join(str(text).split()).replace("|", "\\|")


def _doc_link(page, label=None):
    page = page.split("#")[0]
    return f"{{doc}}`{label} </{page}>`" if label else f"{{doc}}`/{page}`"


def _task_text(task):
    """A task is a string, or {do: ..., use: [...]} when it needs other names."""
    return task["do"] if isinstance(task, dict) else task


def _decision_rows(tools):
    rows = []
    for tool in tools:
        default = tool.get("use") or _names(tool)[:1]
        for task in tool["tasks"]:
            names = task.get("use", default) if isinstance(task, dict) else default
            use = ", ".join(f"`{n}`" for n in names)
            rows.append((_task_text(task), use, tool["page"], tool.get("status", "stable")))
    return rows


def _decision_table(tools, with_family=None):
    head = "| I want to … | Use | Page |\n|---|---|---|\n"
    if with_family:
        head = "| I want to … | Use | Family | Page |\n|---|---|---|---|\n"
    out = [head]
    for tool in tools:
        for task, use, page, status in _decision_rows([tool]):
            tag = "" if status == "stable" else f" *({status})*"
            if with_family:
                out.append(f"| {_cell(task)} | {use}{tag} | {with_family[tool['family']]} "
                           f"| {_doc_link(page)} |\n")
            else:
                out.append(f"| {_cell(task)} | {use}{tag} | {_doc_link(page)} |\n")
    return "".join(out)


def _glance(tool):
    """The "At a glance" box of one tool, following the page template."""
    nl = "\n"

    def items(lst):
        return "".join(f"- {i}{nl}" for i in lst)

    def section(title, lst):
        return f"**{title}**{nl}{nl}" + items(lst) + nl

    names = ", ".join(f"`{n}`" for n in tool.get("symbols", [])
                      + [q.rsplit(".", 1)[-1] for q in tool.get("not_exported", [])])
    out = [f":::{{admonition}} At a glance: {names}{nl}:class: tip{nl}{nl}"]
    if tool.get("status", "stable") != "stable":
        out.append(f"**Status: {tool['status']}.**{nl}{nl}")
    if tool.get("not_exported"):
        out.append("**Import from its module** (not exported from `healpix_analyse`): "
                   + ", ".join(f"`{q}`" for q in tool["not_exported"]) + f".{nl}{nl}")
    out.append(f"**What it does.** {tool['summary'].strip()}{nl}{nl}")
    out.append(section("Use it to", [_task_text(t) for t in tool["tasks"]]))
    io = [f"{label}: {tool[field]}"
          for field, label in (("domain", "Domain"), ("ordering", "Ordering"),
                               ("ellipsoid", "Default ellipsoid"), ("io", "Shapes and types"))
          if tool.get(field)]
    out.append(section("Inputs and outputs", io))
    out.append(section("Pros", tool["pros"]))
    out.append(section("Cons and limits", tool["cons"]))
    alt = [f"{i['when']}: use {i['use']}"
           + (f" ({_doc_link(i['page'])})" if i.get("page") else "")
           for i in tool["use_instead"]]
    out.append(section("Use something else when", alt))
    misc = [f"Cost: {tool['cost']}", f"Differentiable: {tool['differentiable']}"]
    for field, label in (("gpu", "GPU"), ("nan", "Missing values (NaN)"),
                         ("optional_deps", "Optional dependencies")):
        if tool.get(field):
            misc.append(f"{label}: {tool[field]}")
    out.append(section("Cost and properties", misc) + f":::{nl}")
    return "".join(out)


def _redirect_html(new):
    """A static stub that sends an old top-level URL to its new page."""
    target = f"{new}.html"
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta http-equiv=\"refresh\" content=\"0; url={target}\">\n"
            f"<link rel=\"canonical\" href=\"{target}\">\n"
            "<title>Page moved</title>\n</head>\n<body>\n"
            f"<p>This page has moved to <a href=\"{target}\">{target}</a>.</p>\n"
            "</body>\n</html>\n")


def _api_page(cat, qual):
    out = []
    for key in FAMILY_ORDER:
        fam = cat["families"][key]
        out.append(f"## {key}. {fam['title']}\n\n")
        out.append(f"Guide: {_doc_link(fam['page'])}.\n\n")
        for tool in (t for t in cat["tools"] if t["family"] == key):
            names = [qual[n] for n in tool.get("symbols", [])]
            names += tool.get("not_exported", [])
            if not names:
                continue
            out.append(f"### {tool['title']}\n\n")
            note = (f"Guide: {_doc_link(tool['page'])}."
                    + (f" Status: **{tool['status']}**." if tool.get("status", "stable") != "stable" else ""))
            if tool.get("not_exported"):
                note += (" Not exported from `healpix_analyse`: import "
                         + ", ".join(f"`{q}`" for q in tool["not_exported"])
                         + " from its module.")
            out.append(note + "\n\n")
            # eval-rst: the table cells autoapisummary emits are reST; the
            # current module lets short references in the summaries resolve
            module = names[0].rsplit(".", 1)[0]
            out.append(f"```{{eval-rst}}\n.. py:currentmodule:: {module}\n\n.. autoapisummary::\n\n"
                       + "".join(f"   {n}\n" for n in names) + "```\n\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# llms.txt
# ---------------------------------------------------------------------------


def _url(base, page):
    page, _, anchor = page.partition("#")
    return f"{base}{page}.html" + (f"#{anchor}" if anchor else "")


def _llms_txt(cat, base):
    out = [f"# healpix-analyse\n\n> {_cell(cat['intro'])}\n\n"]
    out.append(textwrap.dedent("""\
        Read llms-full.txt for the full tool catalogue (pros, cons, when to use
        something else for every public function). Before proposing a tool, ask
        the user the questions of the decision guide: whole sphere or regional
        domain, learned or fixed weights, gaps (NaN) in the data, scale in metres
        or in cells, size of the domain.

        """))
    out.append(f"- [Full catalogue for LLMs]({base}llms-full.txt): every tool with "
               "pros, cons and alternatives\n")
    for page, label in cat["key_pages"]:
        out.append(f"- [{label}]({_url(base, page)})\n")
    for key in FAMILY_ORDER:
        fam = cat["families"][key]
        out.append(f"\n## {key}. {fam['title']}\n\n")
        out.append(f"- [{fam['title']} — landing page]({_url(base, fam['page'])}): "
                   f"{_cell(fam['summary'])}\n")
        for tool in (t for t in cat["tools"] if t["family"] == key):
            names = ", ".join(_names(tool))
            status = "" if tool.get("status", "stable") == "stable" else f" [{tool['status']}]"
            out.append(f"- [{names}]({_url(base, tool['page'])}){status}: "
                       f"{_cell(tool['summary'])}\n")
    ai = cat.get("ai")
    if ai:
        out.append("\n## AI building blocks\n\n")
        out.append(f"- [Basic tools for AI — landing page]({_url(base, ai['page'])}): "
                   "the layers, norms, pooling, encodings and resampling operators to "
                   "build neural networks on HEALPix data, with runnable recipes; "
                   "llms-full.txt has an API card (signature, shapes, gradient, "
                   "determinism) for each and the full code of the recipes\n")
        for tool in _ai_tools(cat):
            names = ", ".join(q.rsplit(".", 1)[-1] for q in tool["ai"]["api"])
            out.append(f"- [{names}]({_url(base, tool['page'])}) [{tool['ai']['role']}]: "
                       f"{_cell(tool['ai']['does'])}\n")
        for rec in ai.get("recipes", []):
            out.append(f"- [Recipe: {rec['title']}]({_url(base, rec['page'])})\n")
    return "".join(out)


def _bullets(label, items):
    if not items:
        return ""
    return f"{label}:\n" + "".join(f"  - {_cell(i)}\n" for i in items)


def _llms_full(cat, base, srcdir):
    out = [_llms_txt(cat, base), "\n\n# Decision guide and conventions\n\n"]
    for page in cat["full_text_pages"]:
        with open(os.path.join(srcdir, page + ".md"), encoding="utf-8") as fh:
            out.append(f"\n\n<!-- source: {_url(base, page)} -->\n\n" + fh.read())
    out.append("\n\n# Tool catalogue\n")
    for key in FAMILY_ORDER:
        fam = cat["families"][key]
        out.append(f"\n## {key}. {fam['title']}\n\n{_cell(fam['summary'])}\n")
        for tool in (t for t in cat["tools"] if t["family"] == key):
            out.append(f"\n### {tool['title']}\n\n")
            out.append(f"Symbols: {', '.join(_names(tool))}\n")
            if tool.get("not_exported"):
                out.append("Not exported from healpix_analyse; import from: "
                           + ", ".join(tool["not_exported"]) + "\n")
            out.append(f"Status: {tool.get('status', 'stable')}\n")
            out.append(f"Page: {_url(base, tool['page'])}\n")
            out.append(f"What it does: {_cell(tool['summary'])}\n")
            out.append(_bullets("Use it to", [_task_text(t) for t in tool["tasks"]]))
            if tool.get("keywords"):
                out.append(f"Keywords: {', '.join(tool['keywords'])}\n")
            for field, label in (("domain", "Domain"), ("ordering", "Ordering"),
                                 ("ellipsoid", "Default ellipsoid"), ("io", "Inputs and outputs"),
                                 ("differentiable", "Differentiable"), ("gpu", "GPU"),
                                 ("nan", "Missing values (NaN)"), ("cost", "Cost")):
                if tool.get(field):
                    out.append(f"{label}: {_cell(tool[field])}\n")
            out.append(_bullets("Pros", tool["pros"]))
            out.append(_bullets("Cons and limits", tool["cons"]))
            out.append(_bullets("Do not use it for", [
                f"{i['when']} -> use {i['use']}" for i in tool["use_instead"]]))
            if tool.get("optional_deps"):
                out.append(f"Optional dependencies: {_cell(tool['optional_deps'])}\n")
    if cat.get("ai"):
        out.append(_llms_full_ai(cat, base, srcdir))
    return "".join(out)


def _llms_full_ai(cat, base, srcdir):
    """The "AI building blocks" part of llms-full.txt."""
    ai = cat["ai"]
    out = ["\n\n# AI building blocks\n\n",
           f"Landing page: {_url(base, ai['page'])}\n\n",
           "To build a neural network on HEALPix data, combine these blocks with "
           "torch.nn. Signatures below are read from the source code. The layers "
           "take [B, C, N] tensors on a domain fixed at construction (cell_ids, "
           "NESTED); HealPixDown / HealPixUp return a tuple (values, cell_ids_out); "
           "HealPixGradient and HealPixNeighbourReducer are plain callables "
           "without parameters.\n\n"
           "Roles:\n"]
    out += [f"  - {k}: {v}\n" for k, v in ai["roles"].items()]
    out.append("\n## Which block for which architecture\n\n")
    out.append(_ai_architectures(cat, lambda r: f"recipe {r['id']} ({_url(base, r['page'])})"))
    out.append("\n## API cards\n")
    for tool in _ai_tools(cat):
        out.append(_ai_card(tool, srcdir, md=False))
    with open(os.path.join(srcdir, ai["data_page"] + ".md"), encoding="utf-8") as fh:
        out.append(f"\n\n<!-- source: {_url(base, ai['data_page'])} -->\n\n" + fh.read())
    out.append("\n\n## Recipes (complete, runnable code)\n")
    for rec in ai.get("recipes", []):
        with open(os.path.join(srcdir, rec["script"]), encoding="utf-8") as fh:
            code = fh.read()
        out.append(f"\n### Recipe: {rec['title']}\n\nPage: {_url(base, rec['page'])}\n"
                   f"Uses: {', '.join(rec.get('uses', []))}\n\n```python\n{code}```\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(app, config):
    srcdir = str(app.srcdir)
    cat = _load(os.path.join(srcdir, "tools.yaml"))
    eval_path = os.path.join(srcdir, "dev", "tool_choice_eval.yaml")
    eval_set = _load(eval_path) if os.path.exists(eval_path) else None
    public, qual = _public_api(os.path.join(srcdir, "..", "healpix_analyse", "__init__.py"))
    _check(cat, public, qual, srcdir, eval_set, config.tool_catalogue_complete)

    gen = os.path.join(srcdir, "_generated")
    extra = os.path.join(gen, "extra")
    os.makedirs(extra, exist_ok=True)

    def write(path, text):
        # only rewrite on change, so that Sphinx does not re-read every page
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                if fh.read() == text:
                    return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    titles = {k: f"{k}. {v['title']}" for k, v in cat["families"].items()}
    for key in FAMILY_ORDER:
        tools = [t for t in cat["tools"] if t["family"] == key]
        write(os.path.join(gen, f"decisions_{key}.md"), _decision_table(tools))
    write(os.path.join(gen, "decisions_all.md"),
          _decision_table(cat["tools"], with_family=titles))
    write(os.path.join(gen, "api_families.md"), _api_page(cat, qual))
    os.makedirs(os.path.join(gen, "glance"), exist_ok=True)
    for tool in cat["tools"]:
        write(os.path.join(gen, "glance", f"{tool['id']}.md"), _glance(tool))
    if cat.get("ai"):
        write(os.path.join(gen, "ai_blocks.md"), _ai_table(cat))
        write(os.path.join(gen, "ai_architectures.md"), _ai_architectures(
            cat, lambda r: _doc_link(r["page"], r["title"])))
        write(os.path.join(gen, "ai_cards.md"),
              "".join(_ai_card(t, srcdir) for t in _ai_tools(cat)))
    base = config.html_baseurl or ""
    if base and not base.endswith("/"):
        base += "/"
    write(os.path.join(extra, "llms.txt"), _llms_txt(cat, base))
    for old, new in (cat.get("redirects") or {}).items():
        write(os.path.join(extra, f"{old}.html"), _redirect_html(new))
    write(os.path.join(extra, "llms-full.txt"), _llms_full(cat, base, srcdir))


def setup(app):
    # False while the catalogue is being filled in; True requires every name
    # of healpix_analyse.__all__ to be catalogued.
    app.add_config_value("tool_catalogue_complete", True, "env")
    # config-inited: before html_extra_path is checked and before any page is read
    app.connect("config-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
