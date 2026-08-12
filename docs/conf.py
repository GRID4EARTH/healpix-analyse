# Configuration file for the Sphinx documentation builder.
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Notebooks live in ``<repo root>/Notebooks`` so they can be run/edited
# outside of docs/, but Sphinx can only pull toctree entries from within its
# source directory (docs/). Mirror the folder into docs/external_notebooks
# (symlink when possible, plain copy otherwise) before each build so that
# the entries in docs/notebooks/index.rst resolve. (Named differently from
# "Notebooks" — not just a case change — so this can't collide with
# docs/notebooks/ on case-insensitive filesystems such as macOS/Windows.)
# ---------------------------------------------------------------------------
_NOTEBOOKS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Notebooks"))
_NOTEBOOKS_DST = os.path.join(os.path.dirname(__file__), "external_notebooks")


def _sync_notebooks(app):
    if not os.path.isdir(_NOTEBOOKS_SRC):
        return
    if os.path.islink(_NOTEBOOKS_DST):
        os.unlink(_NOTEBOOKS_DST)
    elif os.path.isdir(_NOTEBOOKS_DST):
        shutil.rmtree(_NOTEBOOKS_DST)
    elif os.path.exists(_NOTEBOOKS_DST):
        os.remove(_NOTEBOOKS_DST)
    try:
        os.symlink(_NOTEBOOKS_SRC, _NOTEBOOKS_DST)
    except OSError:
        shutil.copytree(_NOTEBOOKS_SRC, _NOTEBOOKS_DST)


def setup(app):
    app.connect("builder-inited", _sync_notebooks)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "healpix-analyse"
year = "2025"
author = "Jean-Marc Delouis, Tina Odaka"
copyright = f"{year}, {author}"
release = "0.1.0"

# root toctree document (root_doc for Sphinx >=4, master_doc for older)
root_doc = "index"
master_doc = "index"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_nb",
]

# ---------------------------------------------------------------------------
# MyST
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "deflist",
]

# Generate GitHub-style slug anchors for headings (up to this depth) so that
# in-page TOC links like [4 Algorithm](#4-algorithm) inside the same MyST
# document resolve.
myst_heading_anchors = 4

nb_execution_mode = "off"

# ---------------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------------
autoapi_dirs = ["../healpix_analyse"]
autoapi_type = "python"
autoapi_output_dir = "autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_keep_files = True
autoapi_python_use_implicit_namespaces = True

# ---------------------------------------------------------------------------
# Suppress warnings
# ---------------------------------------------------------------------------
suppress_warnings = [
    "autoapi.python_import_resolution",
    "autoapi",
    "myst.header",
    "ref.python",
    "intersphinx.external",
    # napoleon/numpydoc renders free-text type descriptions (e.g. "array-like",
    # "shape [N]", "torch.float32") as py:class/py:func cross-references.
    # Under nitpicky mode (-n) those are reported as unresolved, but they are
    # not real Python objects to link to — nothing to fix on our side.
    "ref.class",
    "ref.func",
    "ref.obj",
    # Supplementary/dev notebooks in Notebooks/ that are intentionally not
    # linked from any toctree.
    "toc.not_included",
    # Notebooks (.ipynb) don't always start with a Markdown H1 cell, so
    # myst-nb can't derive a page title from them; we always give them an
    # explicit title in the toctree entry itself.
    "toc.no_title",
]

# ---------------------------------------------------------------------------
# Napoleon
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "healpix_geo": ("https://healpix-geo.readthedocs.io/en/latest/", None),
}

# ---------------------------------------------------------------------------
# HTML — PyData theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "healpix-analyse"
html_theme_options = {
    "navigation_depth": 4,
    "show_toc_level": 2,
    "github_url": "https://github.com/GRID4EARTH/healpix-analyse",
    "icon_links_label": "Quick Links",
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
}

# ---------------------------------------------------------------------------
# Source suffixes
# myst_nb (in extensions) registers .md and .ipynb automatically.
# Listing .md first avoids Sphinx resolving root_doc to index.rst.
# ---------------------------------------------------------------------------
source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}
