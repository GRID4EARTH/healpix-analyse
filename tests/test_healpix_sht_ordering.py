"""Regression tests for HEALPixSHT RING/NESTED ordering."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("healpix_geo")

from healpix_analyse.healpix_sht import HEALPixSHT


def _ring_to_nested_by_assignment(sht, ring_map):
    """Reference permutation without using a gather-direction assumption."""
    nested_map = torch.empty_like(ring_map)
    nested_map[..., sht._r2n()] = ring_map
    return nested_map


def test_alm2map_nest_matches_ring_to_nested_assignment():
    sht = HEALPixSHT(level=1, lmax=3, dtype=torch.float64, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(7)
    alm = torch.complex(
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
    )

    ring_map = sht.alm2map(alm, nest=False)
    nested_map = sht.alm2map(alm, nest=True)
    expected = _ring_to_nested_by_assignment(sht, ring_map)

    torch.testing.assert_close(nested_map, expected, rtol=0.0, atol=0.0)


def test_map2alm_nest_reads_the_same_physical_map_as_ring():
    sht = HEALPixSHT(level=1, lmax=3, dtype=torch.float64, device="cpu")
    ring_map = torch.arange(sht.n_pix, dtype=sht.dtype)
    nested_map = _ring_to_nested_by_assignment(sht, ring_map)

    ring_alm = sht.map2alm(ring_map, nest=False)
    nested_alm = sht.map2alm(nested_map, nest=True)

    torch.testing.assert_close(nested_alm, ring_alm, rtol=0.0, atol=0.0)


def test_spin_synthesis_nest_matches_ring_to_nested_assignment():
    sht = HEALPixSHT(level=1, lmax=3, dtype=torch.float64, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(11)
    alm_e = torch.complex(
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
    )
    alm_b = torch.complex(
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
        torch.randn(sht.n_alm, generator=generator, dtype=sht.dtype),
    )

    q_ring, u_ring = sht.alm2map_spin(alm_e, alm_b, spin=1, nest=False)
    q_nested, u_nested = sht.alm2map_spin(alm_e, alm_b, spin=1, nest=True)

    torch.testing.assert_close(
        q_nested,
        _ring_to_nested_by_assignment(sht, q_ring),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        u_nested,
        _ring_to_nested_by_assignment(sht, u_ring),
        rtol=0.0,
        atol=0.0,
    )


def test_spin_analysis_nest_reads_the_same_physical_maps_as_ring():
    sht = HEALPixSHT(level=1, lmax=3, dtype=torch.float64, device="cpu")
    q_ring = torch.arange(sht.n_pix, dtype=sht.dtype)
    u_ring = torch.flip(q_ring, dims=(-1,))
    q_nested = _ring_to_nested_by_assignment(sht, q_ring)
    u_nested = _ring_to_nested_by_assignment(sht, u_ring)

    e_ring, b_ring = sht.map2alm_spin(q_ring, u_ring, spin=1, nest=False)
    e_nested, b_nested = sht.map2alm_spin(
        q_nested, u_nested, spin=1, nest=True
    )

    torch.testing.assert_close(e_nested, e_ring, rtol=0.0, atol=0.0)
    torch.testing.assert_close(b_nested, b_ring, rtol=0.0, atol=0.0)
