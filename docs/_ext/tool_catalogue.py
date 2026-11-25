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
3. writes ``llms.txt`` and ``llms-full.txt`` (https://llmstxt.org) into
   ``docs/_generated/extra/``, which ``html_extra_path`` copies to the root of
   the built site.

Any inconsistency raises an error, so the build fails (with or without
``-W``). The package is never imported: ``__all__`` and the import aliases are
read from ``healpix_analyse/__init__.py`` with :mod:`ast`.
"""

from __future__ import annotations

import ast
import os
import textwrap
from collections import Counter

import yaml
from sphinx.errors import ExtensionError

FAMILY_ORDER = ("A", "B", "C", "D")


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
    if errors:
        raise ExtensionError("tool catalogue (docs/tools.yaml) is inconsistent:\n  - "
                             + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _names(tool):
    names = list(tool.get("symbols", []))
    names += [q.rsplit(".", 1)[-1] for q in tool.get("not_exported", [])]
    names += tool.get("also", [])
    return names


def _cell(text):
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
    base = config.html_baseurl or ""
    if base and not base.endswith("/"):
        base += "/"
    write(os.path.join(extra, "llms.txt"), _llms_txt(cat, base))
    write(os.path.join(extra, "llms-full.txt"), _llms_full(cat, base, srcdir))


def setup(app):
    # False while the catalogue is being filled in; True requires every name
    # of healpix_analyse.__all__ to be catalogued.
    app.add_config_value("tool_catalogue_complete", True, "env")
    # config-inited: before html_extra_path is checked and before any page is read
    app.connect("config-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
