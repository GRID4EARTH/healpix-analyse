"""Tests for healpix_analyse.alm_latlon (issue #92)."""

import math

import numpy as np
import pytest
import torch

from healpix_analyse.alm_latlon import (
    _rfft2fft,
    alm2map_latlon,
    anafast_latlon,
    build_rings_from_latlon,
    comp_tf_latlon,
    compute_weights,
    grid_summary,
    map2alm_latlon,
)

LMAX = 8


def gauss_grid(n_rings, n_lon, phi0=0.0):
    """Gauss-Legendre colatitudes and uniform longitudes, flattened."""
    x, _ = np.polynomial.legendre.leggauss(n_rings)
    theta = np.arccos(x)
    phi = phi0 + 2.0 * np.pi * np.arange(n_lon) / n_lon
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    return build_rings_from_latlon(th.ravel(), ph.ravel())


def fields(theta, phi):
    """Band-limited real fields with m = 0, 1, 2 and a mixture."""
    st, ct = np.sin(theta), np.cos(theta)
    out = {
        "m0": ct,
        "m1": st * np.cos(phi),
        "m2": st**2 * np.cos(2 * phi),
        "m1_sin": st * ct * np.sin(phi),
    }
    out["mix"] = (
        0.3 + out["m0"] - 2.0 * out["m1"] + 0.7 * out["m2"]
        + 1.5 * out["m1_sin"] + st**3 * np.sin(3 * phi + 0.4)
    )
    return out


def flat_coords(ring_theta, ring_phi_list, ring_counts):
    theta = np.repeat(ring_theta, ring_counts)
    phi = np.concatenate(ring_phi_list)
    return theta, phi


def round_trip(f, ring_theta, ring_phi_list, ring_counts, quadrature):
    alm = map2alm_latlon(
        f, ring_theta, ring_phi_list, ring_counts, lmax=LMAX, quadrature=quadrature
    )
    return alm2map_latlon(
        alm, ring_theta, ring_phi_list, ring_counts, lmax=LMAX
    ).numpy()


# ---------------------------------------------------------------------------
# _rfft2fft and comp_tf_latlon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 8, 17, 64])
def test_rfft2fft_matches_fft(n):
    v = torch.randn(3, n, dtype=torch.float64)
    out = _rfft2fft(v)
    assert out.shape == (3, n)
    torch.testing.assert_close(out, torch.fft.fft(v), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n", [5, 7, 8])
def test_comp_tf_uniform_matches_direct_dft_with_aliasing(n):
    # mmax well above N exercises the tiling of the full spectrum.
    mmax = 3 * n
    phi0 = 0.37
    phi = phi0 + 2 * np.pi * np.arange(n) / n
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)                     # pixels need not be sorted
    phi = phi[perm]
    v = rng.standard_normal(n)
    w = np.full(n, 0.25)
    ft = comp_tf_latlon(
        torch.tensor(v)[None], [phi], np.array([n]), w, mmax=mmax
    )[0, 0].numpy()
    m = np.arange(mmax + 1)
    ref = (v[:, None] * w[:, None] * np.exp(-1j * np.outer(phi, m))).sum(0)
    np.testing.assert_allclose(ft, ref, rtol=0, atol=1e-12)


def test_comp_tf_irregular_matches_direct_dft():
    rng = np.random.default_rng(1)
    n, mmax = 11, 9
    phi = np.sort(rng.uniform(0, 2 * np.pi, n))
    v = rng.standard_normal(n)
    w = rng.uniform(0.5, 1.5, n)
    ft = comp_tf_latlon(
        torch.tensor(v), [phi], np.array([n]), w, mmax=mmax
    )[0].numpy()
    ref = (v[:, None] * w[:, None] * np.exp(-1j * np.outer(phi, np.arange(mmax + 1)))).sum(0)
    np.testing.assert_allclose(ft, ref, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_lon", [2 * LMAX + 2, 2 * LMAX + 1, 33])
@pytest.mark.parametrize("name", ["m0", "m1", "m2", "m1_sin", "mix"])
def test_round_trip_uniform_rings(n_lon, name):
    """Gauss-Legendre rings + uniform longitudes: exact for ℓ ≤ lmax (odd N too)."""
    ring_theta, ring_phi_list, ring_counts, order = gauss_grid(LMAX + 1, n_lon, phi0=0.1)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)[name]
    rec = round_trip(f, ring_theta, ring_phi_list, ring_counts, "gauss_legendre")
    np.testing.assert_allclose(rec, f, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", ["m0", "m1", "m2"])
def test_round_trip_amplitude_ratio(name):
    """The ratio <x, y> / <x, x> of issue #92 is 1 for m = 0, 1, 2."""
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    x = fields(theta, phi)[name]
    y = round_trip(x, ring_theta, ring_phi_list, ring_counts, "gauss_legendre")
    assert abs(np.dot(x, y) / np.dot(x, x) - 1.0) < 1e-12


def test_round_trip_regular_latlon_grid():
    """Regular lat/lon grid (trapeze): accurate, not exact."""
    n_lat, n_lon = 64, 128
    theta = (np.arange(n_lat) + 0.5) * np.pi / n_lat
    phi = 2 * np.pi * np.arange(n_lon) / n_lon
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    ring_theta, ring_phi_list, ring_counts, order = build_rings_from_latlon(
        th.ravel(), ph.ravel()
    )
    t, p = flat_coords(ring_theta, ring_phi_list, ring_counts)
    for name in ["m0", "m1", "m2"]:
        x = fields(t, p)[name]
        y = round_trip(x, ring_theta, ring_phi_list, ring_counts, "trapeze")
        assert abs(np.dot(x, y) / np.dot(x, x) - 1.0) < 1e-2, name


def test_round_trip_irregular_rings():
    """Irregular longitudes use the direct DFT path in both directions."""
    rng = np.random.default_rng(2)
    n_rings, n_lon = 32, 96
    x_gl, _ = np.polynomial.legendre.leggauss(n_rings)
    ring_theta = np.sort(np.arccos(x_gl))
    base = 2 * np.pi * np.arange(n_lon) / n_lon
    ring_phi_list = [
        np.sort((base + rng.uniform(-0.2, 0.2, n_lon) * 2 * np.pi / n_lon) % (2 * np.pi))
        for _ in range(n_rings)
    ]
    ring_counts = np.full(n_rings, n_lon)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)["mix"]

    # Synthesis at arbitrary points is exact.
    ring_theta_u, ring_phi_u, counts_u, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    tu, pu = flat_coords(ring_theta_u, ring_phi_u, counts_u)
    alm_exact = map2alm_latlon(
        fields(tu, pu)["mix"], ring_theta_u, ring_phi_u, counts_u,
        lmax=LMAX, quadrature="gauss_legendre",
    )
    syn = alm2map_latlon(alm_exact, ring_theta, ring_phi_list, ring_counts).numpy()
    np.testing.assert_allclose(syn, f, rtol=0, atol=1e-10)

    # Analysis on irregular rings: wrapped trapezoid in φ, close to exact.
    alm_irr = map2alm_latlon(
        f, ring_theta, ring_phi_list, ring_counts,
        lmax=LMAX, quadrature="gauss_legendre",
    )
    np.testing.assert_allclose(alm_irr.numpy(), alm_exact.numpy(), rtol=0, atol=2e-3)
    rec = alm2map_latlon(alm_irr, ring_theta, ring_phi_list, ring_counts).numpy()
    np.testing.assert_allclose(rec, f, rtol=0, atol=2e-2)


def test_round_trip_batched_and_flat_phi():
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 1)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    fs = fields(theta, phi)
    batch = torch.tensor(np.stack([fs["m1"], fs["mix"]]))
    phi_flat = np.concatenate(ring_phi_list)
    alm = map2alm_latlon(batch, ring_theta, phi_flat, ring_counts,
                         lmax=LMAX, quadrature="gauss_legendre")
    assert alm.shape == (2, (LMAX + 1) * (LMAX + 2) // 2)
    rec = alm2map_latlon(alm, ring_theta, phi_flat, ring_counts)
    np.testing.assert_allclose(rec.numpy(), batch.numpy(), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Known coefficients and power spectrum
# ---------------------------------------------------------------------------

def _alm_index(ell, m, lmax):
    return m * (2 * lmax + 3 - m) // 2 + ell - m


def test_known_alm():
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    fs = fields(theta, phi)
    a0 = map2alm_latlon(fs["m0"], ring_theta, ring_phi_list, ring_counts,
                        lmax=LMAX, quadrature="gauss_legendre").numpy()
    a1 = map2alm_latlon(fs["m1"], ring_theta, ring_phi_list, ring_counts,
                        lmax=LMAX, quadrature="gauss_legendre").numpy()
    expected0 = np.zeros_like(a0)
    expected0[_alm_index(1, 0, LMAX)] = math.sqrt(4 * math.pi / 3)
    expected1 = np.zeros_like(a1)
    # Condon-Shortley phase: Y_11 = -sqrt(3/8π) sinθ e^{iφ}
    expected1[_alm_index(1, 1, LMAX)] = -math.sqrt(2 * math.pi / 3)
    np.testing.assert_allclose(a0, expected0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a1, expected1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["m0", "m1", "m1_sin"])
def test_anafast_known_field(name):
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)[name]
    cl = anafast_latlon(f, ring_theta, ring_phi_list, ring_counts,
                        lmax=LMAX, quadrature="gauss_legendre").numpy()
    expected = np.zeros(LMAX + 1)
    if name == "m1_sin":
        # sinθ cosθ sinφ is a pure ℓ = 2 field: ∫ f² dΩ = 4π/15 = 5 C_2
        expected[2] = 4 * math.pi / 75
    else:
        # ∫ f² dΩ = 4π/3 = 3 C_1
        expected[1] = 4 * math.pi / 9
    np.testing.assert_allclose(cl, expected, rtol=0, atol=1e-12)


def test_anafast_parseval():
    """Σ (2ℓ+1) C_ℓ = ∫ f² dΩ for a band-limited field (exact quadrature)."""
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(2 * LMAX + 1, 4 * LMAX + 2)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)["mix"]
    cl = anafast_latlon(f, ring_theta, ring_phi_list, ring_counts,
                        lmax=LMAX, quadrature="gauss_legendre").numpy()
    w = compute_weights(ring_theta, ring_phi_list, ring_counts, "gauss_legendre")
    total = np.sum(w * f**2)
    np.testing.assert_allclose(np.sum((2 * np.arange(LMAX + 1) + 1) * cl), total,
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# lmax defaults and validation
# ---------------------------------------------------------------------------

def test_lmax_defaults_aligned(capsys):
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(70, 140)
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)["m1"]
    alm = map2alm_latlon(f, ring_theta, ring_phi_list, ring_counts)
    assert alm.shape[-1] == 65 * 66 // 2
    cl = anafast_latlon(f, ring_theta, ring_phi_list, ring_counts)
    assert cl.shape[-1] == 65
    # alm2map_latlon deduces lmax from the length of alm.
    rec = alm2map_latlon(alm, ring_theta, ring_phi_list, ring_counts)
    assert rec.shape == f.shape
    assert grid_summary(ring_theta, ring_phi_list, ring_counts)["n_alm"] == 65 * 66 // 2


def test_alm2map_lmax_mismatch_raises():
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    alm = torch.zeros((LMAX + 1) * (LMAX + 2) // 2, dtype=torch.complex128)
    with pytest.raises(ValueError):
        alm2map_latlon(alm, ring_theta, ring_phi_list, ring_counts, lmax=LMAX + 1)
    with pytest.raises(ValueError):
        alm2map_latlon(alm[:-1], ring_theta, ring_phi_list, ring_counts)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_uniform_and_irregular():
    rng = np.random.default_rng(3)
    ring_theta, ring_phi_list, ring_counts, _ = gauss_grid(LMAX + 1, 2 * LMAX + 2)
    # Make one ring irregular to exercise the direct DFT path.
    ring_phi_list = list(ring_phi_list)
    ring_phi_list[3] = np.sort(rng.uniform(0, 2 * np.pi, ring_counts[3]))
    theta, phi = flat_coords(ring_theta, ring_phi_list, ring_counts)
    f = fields(theta, phi)["mix"]
    alm_cpu = map2alm_latlon(torch.tensor(f), ring_theta, ring_phi_list, ring_counts,
                             lmax=LMAX, quadrature="gauss_legendre")
    alm_gpu = map2alm_latlon(torch.tensor(f, device="cuda"), ring_theta,
                             ring_phi_list, ring_counts,
                             lmax=LMAX, quadrature="gauss_legendre")
    assert alm_gpu.device.type == "cuda"
    np.testing.assert_allclose(alm_gpu.cpu().numpy(), alm_cpu.numpy(), atol=1e-12)
    rec = alm2map_latlon(alm_gpu, ring_theta, ring_phi_list, ring_counts)
    assert rec.device.type == "cuda"
    cl = anafast_latlon(torch.tensor(f, device="cuda"), ring_theta, ring_phi_list,
                        ring_counts, lmax=LMAX, quadrature="gauss_legendre")
    assert cl.device.type == "cuda"
