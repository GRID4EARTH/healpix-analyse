"""HEALPixSHT.uv_to_curl_div / uv_to_eb against an analytic flow (#94)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("healpix_geo")

from healpix_analyse.healpix_sht import _HAS_SPHERICAL, HEALPixSHT

requires_spin = pytest.mark.skipif(
    not _HAS_SPHERICAL,
    reason="spin transforms need the optional quaternionic and spherical packages",
)

R_EARTH = 6371008.8
U0 = 10.0  # m/s, scale of the flow


def _ring_angles(sht):
    """Colatitude and longitude of every pixel, RING order."""
    theta = np.concatenate(
        [np.full(int(n), t) for t, n in zip(sht._ring_theta, sht._ring_n)]
    )
    phi = np.concatenate(
        [p0 + 2.0 * np.pi * np.arange(int(n)) / int(n)
         for p0, n in zip(sht._ring_phi0, sht._ring_n)]
    )
    return theta, phi


def _analytic_flow(theta, phi, radius):
    """u = ∇χ + k × ∇ψ with χ ∝ Re Y_32 (ℓ=3) and ψ ∝ Im Y_21 (ℓ=2).

    Returns u (east), v (north), χ, ψ.  e_θ points south, so v = -u_θ.
    """
    s, c = np.sin(theta), np.cos(theta)
    a = radius * U0
    chi = a * s**2 * c * np.cos(2 * phi)
    dchi_t = a * (2 * s * c**2 - s**3) * np.cos(2 * phi)
    dchi_p = -2.0 * a * s**2 * c * np.sin(2 * phi)
    psi = a * s * c * np.sin(phi)
    dpsi_t = a * (c**2 - s**2) * np.sin(phi)
    dpsi_p = a * s * c * np.cos(phi)
    # ∇f = (∂θ f e_θ + ∂φ f / sinθ e_φ) / R ;  k × e_θ = e_φ, k × e_φ = -e_θ
    u_theta = (dchi_t - dpsi_p / s) / radius
    u_phi = (dchi_p / s + dpsi_t) / radius
    return u_phi, -u_theta, chi, psi


def _rel_err(x, ref):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    return np.abs(x - ref).max() / np.abs(ref).max()


@requires_spin
@pytest.mark.parametrize("nest", [False, True])
def test_uv_to_curl_div_matches_laplacian_of_potentials(nest):
    # Band limit well below 3·nside: the derivative factor √(ℓ(ℓ+1)) would
    # amplify the HEALPix quadrature error near lmax.
    sht = HEALPixSHT(level=5, lmax=10, dtype=torch.float64, device="cpu")
    theta, phi = _ring_angles(sht)
    u, v, chi, psi = _analytic_flow(theta, phi, R_EARTH)
    div_ref = -3 * 4 * chi / R_EARTH**2   # ∇²χ, ℓ = 3
    vor_ref = -2 * 3 * psi / R_EARTH**2   # ∇²ψ, ℓ = 2

    if nest:
        perm = sht._r2n()
        u_n, v_n = np.empty_like(u), np.empty_like(v)
        u_n[perm], v_n[perm] = u, v
        d_n, w_n = np.empty_like(div_ref), np.empty_like(vor_ref)
        d_n[perm], w_n[perm] = div_ref, vor_ref
        u, v, div_ref, vor_ref = u_n, v_n, d_n, w_n

    div, curl = sht.uv_to_curl_div(u, v, nest=nest)

    # HEALPix quadrature at nside=32, lmax=10: about 0.2 % (0.9 % at nside=16).
    assert _rel_err(div, div_ref) < 5e-3
    assert _rel_err(curl, vor_ref) < 5e-3
    # Order of magnitude in s⁻¹: U0 ℓ(ℓ+1) / R ~ 1e-5.
    assert 1e-6 < np.abs(div_ref).max() < 1e-4


@requires_spin
def test_uv_to_curl_div_radius_scales_inverse():
    sht = HEALPixSHT(level=3, lmax=6, dtype=torch.float64, device="cpu")
    theta, phi = _ring_angles(sht)
    u, v, _, _ = _analytic_flow(theta, phi, R_EARTH)
    d1, c1 = sht.uv_to_curl_div(u, v)
    d2, c2 = sht.uv_to_curl_div(u, v, radius=2 * R_EARTH)
    torch.testing.assert_close(d2, d1 / 2)
    torch.testing.assert_close(c2, c1 / 2)


@requires_spin
def test_uv_to_eb_relation_to_curl_div():
    """uv_to_eb keeps the E/B maps; div/curl are -√(ℓ(ℓ+1))/R times them."""
    sht = HEALPixSHT(level=4, lmax=10, dtype=torch.float64, device="cpu")
    theta, phi = _ring_angles(sht)
    u, v, chi, psi = _analytic_flow(theta, phi, R_EARTH)

    e_map, b_map = sht.uv_to_eb(u, v)
    div, curl = sht.uv_to_curl_div(u, v)

    # Single-ℓ potentials: E = √(ℓ(ℓ+1))/R χ (ℓ=3), B = √(ℓ(ℓ+1))/R ψ (ℓ=2).
    assert _rel_err(e_map, np.sqrt(12.0) / R_EARTH * chi) < 2e-2
    assert _rel_err(b_map, np.sqrt(6.0) / R_EARTH * psi) < 2e-2
    # uv_to_eb is the former uv_to_curl_div: spin-1 analysis, spin-0 synthesis.
    alm_e, alm_b = sht.map2alm_spin(-v, u, spin=1)
    torch.testing.assert_close(e_map, sht.alm2map(alm_e))
    torch.testing.assert_close(b_map, sht.alm2map(alm_b))
    # Same sign and scale as the divergence / vorticity (up to quadrature).
    assert _rel_err(div, (-np.sqrt(12.0) / R_EARTH * e_map).numpy()) < 2e-2
    assert _rel_err(curl, (-np.sqrt(6.0) / R_EARTH * b_map).numpy()) < 2e-2


def test_map2alm_spin_spin0_is_minus_map2alm():
    """Documented HEALPix/healpy convention: spin=0 returns -map2alm."""
    sht = HEALPixSHT(level=2, lmax=5, dtype=torch.float64, device="cpu")
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(sht.n_pix, generator=gen, dtype=torch.float64)
    u = torch.randn(sht.n_pix, generator=gen, dtype=torch.float64)
    e, b = sht.map2alm_spin(q, u, spin=0)
    torch.testing.assert_close(e, -sht.map2alm(q))
    torch.testing.assert_close(b, -sht.map2alm(u))
    q2, u2 = sht.alm2map_spin(e, b, spin=0)
    torch.testing.assert_close(q2, sht.alm2map(sht.map2alm(q)))
    torch.testing.assert_close(u2, sht.alm2map(sht.map2alm(u)))
