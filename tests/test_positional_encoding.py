"""Tests of the HEALPix positional encodings (issue #109, section 6)."""

import math

import numpy as np
import pytest
import torch

import healpix_geo

from healpix_analyse import (
    HealPixPositionalEncoding,
    clear_positional_encoding_cache,
    healpix_positional_encoding,
    positional_encoding_cache_info,
)


def _xyz(ids, level, ellipsoid):
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        np.asarray(ids, dtype=np.uint64), level, ellipsoid=ellipsoid)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def test_full_sphere_features():
    level = 2
    enc = healpix_positional_encoding(level, dtype=torch.float64)
    N = 12 * 4 ** level
    nf = level + 1
    assert enc.shape == (3 + 6 * nf, N)
    xyz = _xyz(np.arange(N), level, "WGS84")
    np.testing.assert_allclose(enc[:3].numpy(), xyz, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(enc[:3].numpy(), axis=0), 1.0, atol=1e-14)
    for k in range(nf):
        f = 2.0 ** k * math.pi
        blk = enc[3 + 6 * k: 9 + 6 * k].numpy()
        np.testing.assert_allclose(blk[:3], np.sin(f * xyz), atol=1e-13)
        np.testing.assert_allclose(blk[3:], np.cos(f * xyz), atol=1e-13)


def test_unsorted_regional_domain_follows_cell_order():
    level = 4
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.arange(256 * 5, 256 * 5 + 300))
    full = healpix_positional_encoding(level, dtype=torch.float64, gauge_type="projected_ref",
                                       include_level=True)
    reg = healpix_positional_encoding(level, ids, dtype=torch.float64,
                                      gauge_type="projected_ref", include_level=True)
    torch.testing.assert_close(reg, full[:, torch.as_tensor(ids)], atol=0, rtol=0)
    torch.testing.assert_close(reg[-1], torch.full((300,), level / 29, dtype=torch.float64))


def test_ellipsoid_changes_latitude_only():
    ids = np.arange(100)
    a = healpix_positional_encoding(3, ids, n_frequencies=0, ellipsoid="WGS84", dtype=torch.float64)
    b = healpix_positional_encoding(3, ids, n_frequencies=0, ellipsoid="sphere", dtype=torch.float64)
    np.testing.assert_allclose(a.numpy(), _xyz(ids, 3, "WGS84"), atol=1e-14)
    np.testing.assert_allclose(b.numpy(), _xyz(ids, 3, "sphere"), atol=1e-14)
    assert not torch.allclose(a[2], b[2])
    # lower-case names are accepted, as elsewhere in the package
    c = healpix_positional_encoding(3, ids, n_frequencies=0, ellipsoid="wgs84", dtype=torch.float64)
    torch.testing.assert_close(a, c)


@pytest.mark.parametrize("gauge_type", ["phi", "cosmo", "projected_ref"])
def test_gauge_angle_matches_healpix_conv_frame(gauge_type):
    """alpha rotates e_theta onto the first axis of the HealPixConv frame."""
    from healpix_analyse.convol import _build_rotation_matrices

    level = 2
    ids = np.arange(0, 12 * 16, 7)
    enc = healpix_positional_encoding(level, ids, n_frequencies=0, include_xyz=False,
                                      gauge_type=gauge_type, dtype=torch.float64)
    ca, sa = enc[0].numpy(), enc[1].numpy()
    np.testing.assert_allclose(ca ** 2 + sa ** 2, 1.0, atol=1e-12)
    lon, lat = healpix_geo.nested.healpix_to_lonlat(ids.astype(np.uint64), level, ellipsoid="WGS84")
    th, ph = np.deg2rad(90 - lat), np.deg2rad(lon)
    R = _build_rotation_matrices(th, ph, 1, gauge_type, "cpu", torch.float64)[:, 0].numpy()
    e_th = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], 1)
    e_ph = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], 1)
    expected = ca[:, None] * e_th + sa[:, None] * e_ph
    np.testing.assert_allclose(R[:, :, 0], expected, atol=1e-12)
    if gauge_type == "phi":
        np.testing.assert_allclose(ca, 1.0, atol=1e-12)


def test_explicit_frequencies_and_names():
    pe = HealPixPositionalEncoding(1, frequencies=[0.5, 3.0], include_xyz=False,
                                   gauge_type="phi", include_level=True)
    assert pe.num_features == 12 + 2 + 1 == len(pe.feature_names)
    assert pe.feature_names[0] == "sin(0.5*pi*x)" and pe.feature_names[-1] == "level/29"
    assert HealPixPositionalEncoding(1, n_frequencies=0).num_features == 3


def test_cache_is_keyed_by_cell_ids():
    clear_positional_encoding_cache()
    ids = np.array([5, 3, 9, 1])
    healpix_positional_encoding(2, ids)
    healpix_positional_encoding(2, ids, n_frequencies=1)       # same geometry
    info = positional_encoding_cache_info()
    assert info["misses"] == 1 and info["hits"] == 1 and info["size"] == 1
    healpix_positional_encoding(2, ids[::-1].copy())          # other order = other domain
    assert positional_encoding_cache_info()["misses"] == 2
    healpix_positional_encoding(2, torch.as_tensor(ids))      # tensor ids: same key
    assert positional_encoding_cache_info()["hits"] == 2
    for k in range(10):
        healpix_positional_encoding(2, np.array([k]))
    assert positional_encoding_cache_info()["size"] == 8
    clear_positional_encoding_cache()
    assert positional_encoding_cache_info()["size"] == 0


def test_module_buffer_follows_to_and_concat():
    pe = HealPixPositionalEncoding(1)
    assert "encoding" not in pe.state_dict()
    pe = pe.to(torch.float64)
    assert pe.encoding.dtype == torch.float64
    x = torch.randn(2, 5, 48, dtype=torch.float64)
    y = pe(x)
    assert y.shape == (2, 5 + pe.num_features, 48)
    torch.testing.assert_close(y[:, 5:], pe.encoding.expand(2, -1, -1))
    with pytest.raises(ValueError):
        pe(torch.randn(2, 5, 47))
    if torch.cuda.is_available():
        pe = pe.cuda()
        assert pe.encoding.is_cuda


def test_input_checks():
    with pytest.raises(ValueError):
        healpix_positional_encoding(1, [48])
    with pytest.raises(ValueError):
        healpix_positional_encoding(1, gauge_type="bad")
    with pytest.raises(ValueError):
        healpix_positional_encoding(30)
