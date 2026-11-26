"""Run the AI recipes of docs/ai/recipes/ (the code shown in the docs).

Each recipe is a standalone script whose ``main()`` builds a small model,
runs a forward and backward pass (or a few training steps) and checks the
shapes and values with ``assert``. Keeping them in CI keeps the docs code
runnable.
"""

import pathlib
import runpy

import pytest

RECIPES = pathlib.Path(__file__).resolve().parents[1] / "docs" / "ai" / "recipes"
SCRIPTS = sorted(p.name for p in RECIPES.glob("*.py"))


def test_recipes_found():
    assert {"resnet.py", "unet.py", "mae.py", "training.py"} <= set(SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS)
def test_recipe_runs(script):
    runpy.run_path(str(RECIPES / script), run_name="__main__")
