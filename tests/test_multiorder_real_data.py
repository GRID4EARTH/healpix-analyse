"""Real-data check of ``multiorder_filter_operator`` (opt-in, needs data).

The leaf set of the Scheldt (5577 leaves, levels 11 to 15) and the prototype
graph live in a public folder, https://data.grid4earth.eu/tmp/healpix-analyse-multiorder-2026-11-22/ ;
``benchmarks/multiorder_compare_shad.py`` prints the numbers of the pull
request.  This test is skipped unless one of these is set:

* ``HPA_MULTIORDER_DATA``: a local folder with ``shad_leafset.npz`` and
  ``shad_prototype_graph.npz``;
* ``HPA_ONLINE_TESTS=1``: download them from the folder above.

The normal test run never touches the network.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("scipy")

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "multiorder_compare_shad.py"
_LOCAL = os.environ.get("HPA_MULTIORDER_DATA")
_ONLINE = os.environ.get("HPA_ONLINE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_LOCAL or _ONLINE) or not _SCRIPT.exists(),
    reason="set HPA_MULTIORDER_DATA (local folder) or HPA_ONLINE_TESTS=1 to run",
)


@pytest.fixture(scope="module")
def result():
    spec = importlib.util.spec_from_file_location("multiorder_compare_shad", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare(module.locate(), verbose=False)


def test_contacts_without_joined_water(result):
    assert result["n"] == 5577
    assert result["contacts_library"] == 18561
    assert result["contacts_prototype"] == 16511
    assert result["contacts_only_library"] == 2050
    assert result["contacts_only_prototype"] == 0


def test_geometric_contacts_shorten_the_paths(result):
    x = result["every geometric contact"]
    assert x["share_more_than_0.1pct_shorter"] == pytest.approx(0.685, abs=0.005)
    assert x["kernel_mean_row_l1"] > 1e-2


def test_water_contacts_agree_with_the_prototype(result):
    x = result["prototype (water) contacts"]
    assert x["ratio_p1"] > 0.9990 and x["ratio_p99"] < 1.0010
    assert x["share_more_than_0.1pct_shorter"] == 0.0
    assert x["kernel_max_abs_diff"] < 2e-4
    assert x["kernel_mean_row_l1"] < 4e-4
