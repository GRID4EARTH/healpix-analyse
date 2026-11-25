"""Legacy flat spectrum: deprecation, quiet constructor, ``method="alm"``."""

import numpy as np
import pytest
import torch

import healpix_analyse
from healpix_analyse.alm import AlmCoeffs, AlmTransform
from healpix_analyse.powerspectra import powerspectra


def _equatorial_block(level, ny=8, nx=16):
    """A 2-D block of RING cell ids: ``ny`` equatorial rings, ``nx`` cells each."""
    nside = 2**level
    rows = []
    for k in range(ny):
        ring = nside + k  # 1-based ring index, inside the equatorial belt
        first = 2 * nside * (nside - 1) + (ring - nside) * 4 * nside
        rows.append(first + np.arange(nx))
    return np.stack(rows).astype(np.int64)


def test_powerspectra_is_deprecated():
    level = 4
    cell_ids = _equatorial_block(level)
    data = np.random.default_rng(0).standard_normal(cell_ids.shape)
    with pytest.warns(DeprecationWarning, match="anafast"):
        f, pk = powerspectra(cell_ids, level, data)
    assert f.shape == pk.shape == (min(cell_ids.shape) // 2,)
    assert np.isfinite(pk).any()


def test_alm_transform_constructor_does_not_print(capsys):
    AlmTransform(_equatorial_block(4), 4, device="cpu")
    assert capsys.readouterr().out == ""


def test_alm_transform_method_alm_not_implemented():
    with pytest.raises(NotImplementedError, match="method='alm'"):
        AlmTransform(_equatorial_block(4), 4, method="alm", device="cpu")


def test_alm_coeffs_not_exported_but_importable():
    assert "AlmCoeffs" not in healpix_analyse.__all__
    assert not hasattr(healpix_analyse, "AlmCoeffs")
    c = AlmCoeffs(
        alm=torch.zeros(3, dtype=torch.complex64),
        l=torch.tensor([0, 1, 1]),
        m=torch.tensor([0, 0, 1]),
        meta={},
    )
    assert c.n_modes == 3


@pytest.mark.parametrize(
    "module",
    [
        "healpix_analyse.powerspectra_lonlat",
        "healpix_analyse.ps",
        "healpix_analyse.demo_ps",
        "healpix_analyse.LocalizedFlatSkyAlm",
        "healpix_analyse.LocalizedAlm_optimized_v13_fixed",
        "healpix_analyse.make_rectangle",
    ],
)
def test_legacy_modules_removed(module):
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "name",
    [
        "HealPixConv",
        "HEALPixSHT",
        "binary_dilation",
        "binary_erosion",
        "get_interp_val",
        "get_interp_weights",
        "resample_to_latlon_grid",
    ],
)
def test_entry_points_exported(name):
    assert name in healpix_analyse.__all__
    assert callable(getattr(healpix_analyse, name))


def test_import_does_not_load_optional_dependencies():
    """``import healpix_analyse`` loads none of the optional packages."""
    import subprocess
    import sys

    optional = ["scipy", "healpy", "quaternionic", "spherical", "matplotlib",
                "transformers", "umap", "termcolor"]
    code = (
        "import sys, healpix_analyse; "
        f"print(','.join(m for m in {optional!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == ""
