"""Ellipsoid names are accepted in any casing by the public entry points.

``healpix_geo`` only knows ``"WGS84"`` and ``"sphere"`` spelt exactly that
way; the package re-cases user input with
:func:`healpix_analyse._ellipsoid.canonicalize_ellipsoid` (issue #96, item 1).
"""
import numpy as np
import pytest

from healpix_analyse._ellipsoid import canonicalize_ellipsoid
from healpix_analyse.healpix_interp import get_interp_val, get_interp_weights


@pytest.mark.parametrize(
    "name, expected",
    [
        ("WGS84", "WGS84"),
        ("wgs84", "WGS84"),
        ("Wgs84", "WGS84"),
        ("sphere", "sphere"),
        ("Sphere", "sphere"),
        ("SPHERE", "sphere"),
        ("GRS80", "GRS80"),
        ("grs80", "GRS80"),
    ],
)
def test_canonicalize_recases_known_names(name, expected):
    assert canonicalize_ellipsoid(name) == expected
    # idempotent
    assert canonicalize_ellipsoid(expected) == expected


def test_canonicalize_leaves_unknown_names_alone():
    assert canonicalize_ellipsoid("not-an-ellipsoid") == "not-an-ellipsoid"


LON = np.array([2.3522, -70.0, 120.5])
LAT = np.array([48.8566, -33.4, 10.0])
DEPTH = 6


@pytest.mark.parametrize(
    "variant, reference", [("wgs84", "WGS84"), ("Wgs84", "WGS84"),
                           ("Sphere", "sphere"), ("SPHERE", "sphere")]
)
def test_interp_weights_accept_any_casing(variant, reference):
    pix_v, w_v = get_interp_weights(LON, LAT, DEPTH, ellipsoid=variant)
    pix_r, w_r = get_interp_weights(LON, LAT, DEPTH, ellipsoid=reference)
    np.testing.assert_array_equal(pix_v, pix_r)
    np.testing.assert_array_equal(w_v, w_r)


@pytest.mark.parametrize("variant, reference", [("wgs84", "WGS84"), ("Sphere", "sphere")])
def test_interp_val_accepts_any_casing(variant, reference):
    hpx_map = np.arange(12 * 4**DEPTH, dtype=np.float64)
    v = get_interp_val(hpx_map, LON, LAT, depth=DEPTH, ellipsoid=variant)
    r = get_interp_val(hpx_map, LON, LAT, depth=DEPTH, ellipsoid=reference)
    np.testing.assert_array_equal(v, r)


def test_interp_unknown_ellipsoid_still_fails():
    with pytest.raises(Exception):
        get_interp_weights(LON, LAT, DEPTH, ellipsoid="not-an-ellipsoid")


def test_wgs84_only_filters_accept_lower_case():
    """The WGS84-only code paths compared the raw string to ``"WGS84"``."""
    from healpix_analyse import mean_filter

    level = 8
    cells = np.arange(4 * 64, 4 * 64 + 64, dtype=np.uint64)
    values = np.random.default_rng(0).standard_normal(cells.size)
    ref = mean_filter(values, cells, level, radius_m=60_000.0)
    low = mean_filter(values, cells, level, radius_m=60_000.0, ellipsoid="wgs84")
    np.testing.assert_allclose(np.asarray(low), np.asarray(ref))

    from healpix_analyse import gaussian_filter

    ref = gaussian_filter(values, cells, level, sigma_m=30_000.0)
    low = gaussian_filter(values, cells, level, sigma_m=30_000.0, ellipsoid="Wgs84")
    np.testing.assert_allclose(np.asarray(low), np.asarray(ref))


def test_set_ellipsoid_accepts_title_case():
    from healpix_analyse import dino

    previous = dino.set_ellipsoid("Sphere")
    try:
        assert dino.HEALPIX_ELLIPSOID == "sphere"
        with pytest.raises(ValueError):
            dino.set_ellipsoid("not-an-ellipsoid")
    finally:
        dino.set_ellipsoid(previous)
