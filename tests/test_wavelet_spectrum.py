"""Wavelet power spectrum: constants, NaN, batches, gradients, anafast."""

import math

import numpy as np
import pytest
import torch

from healpix_analyse import (
    HealPixIsotropicWaveletSpectrum,
    HealPixWaveletSpectrum,
    wavelet_power_spectrum,
)
from healpix_analyse.healpix_sht import HEALPixSHT

F64 = dict(device="cpu", dtype=torch.float64, cache_dir=None)
REGION = np.arange(3 * 64, dtype=np.int64)                  # 3 base pixels at level 3


def _make(cls, level=3, **kw):
    return cls(level, ellipsoid="sphere", **F64, **kw)


def _nan_mask(n, frac, seed):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.random(n) < frac)


@pytest.mark.parametrize("cls", [HealPixWaveletSpectrum, HealPixIsotropicWaveletSpectrum])
@pytest.mark.parametrize("edge", ["mean_corrected", "complete", "none"])
@pytest.mark.parametrize("domain", ["full", "full_nan", "region_nan"])
def test_constant_map_gives_zero_power_and_its_monopole(cls, edge, domain):
    cell_ids = REGION if domain == "region_nan" else None
    ws = _make(cls, cell_ids=cell_ids, edge=edge, s1=True)
    n = ws.n_cells[0]
    x = torch.full((n,), 287.15, dtype=torch.float64)            # SST in kelvin
    if domain != "full":
        x[_nan_mask(n, 0.2, 1)] = float("nan")
    out = ws(x)
    assert torch.allclose(out.monopole, torch.tensor(287.15, dtype=torch.float64))
    ok = out.n_eff > 0
    if edge != "complete":                                       # 20 % NaN: no complete 5x5 kernel
        assert ok[0]
    assert torch.all(out.power_total[ok].abs() < 1e-18)
    assert torch.all(out.s1[ok].abs() < 1e-8)
    assert abs(float(out.lowpass_var)) < 1e-18


@pytest.mark.parametrize("cls", [HealPixWaveletSpectrum, HealPixIsotropicWaveletSpectrum])
def test_nan_gives_finite_outputs_and_gradients(cls):
    ws = _make(cls, cell_ids=REGION, s1=True)
    x = torch.randn(2, len(REGION), dtype=torch.float64) + 3.0
    nan = _nan_mask(len(REGION), 0.25, 2)
    x[:, nan] = float("nan")
    x.requires_grad_(True)
    out = ws(x)
    for t in (out.power, out.power_total, out.s1, out.monopole, out.lowpass_var):
        assert torch.isfinite(t).all()
    loss = out.power_total.sum() + out.s1.sum() + out.monopole.sum() + out.lowpass_var.sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[:, nan] == 0)


def test_batch_equals_loop_and_numpy_input():
    ws = _make(HealPixWaveletSpectrum, s1=True)
    x = torch.randn(3, 2, 12 * 4 ** 3, dtype=torch.float64)
    x[0, 1, :50] = float("nan")
    out = ws(x)
    assert out.power.shape == (3, 2, ws.n_scale, 4)
    for b in range(3):
        for c in range(2):
            one = ws(x[b, c].numpy())
            torch.testing.assert_close(out.power[b, c], one.power)
            torch.testing.assert_close(out.s1[b, c], one.s1)
            torch.testing.assert_close(out.monopole[b, c], one.monopole)


def test_gradcheck_float64():
    ws = _make(HealPixWaveletSpectrum, level=2, s1=True)
    n = 12 * 4 ** 2
    nan = _nan_mask(n, 0.1, 4)
    base = torch.randn(n, dtype=torch.float64)

    def f(v):
        x = torch.full((n,), float("nan"), dtype=torch.float64)
        x = x.masked_scatter(~nan, v)
        out = ws(x)
        return out.power_total, out.s1.sum(-1), out.lowpass_var.reshape(1)

    v = base[~nan].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (v,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_against_band_averaged_anafast():
    """Full sphere, level 3: P_s against sum_ell C_hat_ell * response[s, ell].

    The response is measured once with ``windows()`` (exact for a
    rotation-invariant operator).  The spectrum is invariant only up to the
    pixelisation and the finite number of orientations, so a single map
    deviates from the band-averaged anafast by a few per cent at the levels
    where the stencil is local (measured: at most 4.2 % at level 4 and
    5 % at level 3 over 60 maps; 6 % is the tolerance).  At level 1 the
    5x5 stencil reaches 83 degrees (tolerance 20 %) and at level 0 it covers
    the sphere, which is not tested.
    """
    level = 3
    ws = _make(HealPixWaveletSpectrum, level=level)
    win = ws.windows()
    sht = HEALPixSHT(level, dtype=torch.float64, device="cpu")
    L = sht.lmax
    ls = np.concatenate([np.arange(m, L + 1) for m in range(L + 1)])
    ms = np.concatenate([np.full(L + 1 - m, m) for m in range(L + 1)])
    rng = np.random.default_rng(5)
    for alpha in (0.0, 3.0):
        cl = (np.arange(L + 1) + 1.0) ** (-alpha)
        a = (rng.standard_normal((8, ls.size)) + 1j * rng.standard_normal((8, ls.size)))
        a *= np.sqrt(cl[ls] / 2)
        a[:, ms == 0] = rng.standard_normal((8, int((ms == 0).sum()))) * np.sqrt(cl[ls[ms == 0]])
        maps = sht.alm2map(torch.as_tensor(a), nest=True).real
        chat = sht.anafast(maps, nest=True).double()
        rel = (ws(maps).power_total / (chat @ win.response.T) - 1).abs()
        lev = ws._levels
        assert rel[:, lev >= 2].max() < 0.06, rel.max(0).values
        assert rel[:, lev == 1].max() < 0.20, rel.max(0).values


def test_littlewood_paley_closes_the_variance():
    ws = _make(HealPixIsotropicWaveletSpectrum, level=3)
    win = ws.windows()
    lp = ws.littlewood_paley(win)
    sht = HEALPixSHT(3, dtype=torch.float64, device="cpu")
    L = sht.lmax
    n_alm = (L + 1) * (L + 2) // 2
    rng = np.random.default_rng(6)
    a = (rng.standard_normal((6, n_alm)) + 1j * rng.standard_normal((6, n_alm))) / math.sqrt(2)
    a[:, : L + 1] = rng.standard_normal((6, L + 1))            # m = 0 block is real
    maps = sht.alm2map(torch.as_tensor(a), nest=True).real
    out = ws(maps)
    est = (out.power_total * lp.constants).sum(-1) + lp.lowpass * out.lowpass_var
    var = maps.var(-1, unbiased=False)
    assert torch.all((est / var - 1).abs() < max(0.1, float(lp.ripple)))


def test_cross_spectra():
    ws = _make(HealPixWaveletSpectrum, cell_ids=REGION)
    n = len(REGION)
    x = torch.randn(2, n, dtype=torch.float64)
    y = 0.5 * x + torch.randn(2, n, dtype=torch.float64)
    x[:, _nan_mask(n, 0.15, 7)] = float("nan")
    y[:, _nan_mask(n, 0.15, 8)] = float("nan")
    auto = ws(x)
    same = ws.cross(x, x)
    torch.testing.assert_close(same.co_total, auto.power_total)
    torch.testing.assert_close(same.coherence, torch.ones_like(same.coherence))
    scaled = ws.cross(x, 3.0 * x)
    torch.testing.assert_close(scaled.coherence, torch.ones_like(scaled.coherence))
    cr = ws.cross(x, y)
    ok = cr.n_eff > 0
    assert torch.all(cr.coherence[ok].abs() <= 1 + 1e-12)
    assert cr.quad.shape == cr.co.shape == (2, ws.n_scale, 4)
    allp = ws.cross(torch.stack([x, y], dim=1))                # [B, C=2, N]
    assert allp.co_total.shape == (2, 2, 2, ws.n_scale)
    torch.testing.assert_close(allp.co_total[:, 0, 1], cr.co_total, equal_nan=True)
    torch.testing.assert_close(allp.co_total[:, 0, 1], allp.co_total[:, 1, 0], equal_nan=True)


def test_isotropic_cross_is_real():
    ws = _make(HealPixIsotropicWaveletSpectrum)
    x = torch.randn(12 * 4 ** 3, dtype=torch.float64)
    cr = ws.cross(x, -x)
    assert cr.quad is None
    torch.testing.assert_close(cr.coherence, -torch.ones_like(cr.coherence))


def test_scales_per_octave_constraints():
    ws = _make(HealPixWaveletSpectrum, scales_per_octave=2)
    assert ws.kernel_sz == 11 and ws.G == 8
    # two voices where the 11x11 stencil fits (corners <= 180 deg), else 5x5
    assert ws._levels.tolist() == [3, 3, 2, 2, 1, 0]
    assert ws._voices.tolist() == [0, 1, 0, 1, 0, 0]
    with pytest.raises(ValueError, match="cuts"):
        _make(HealPixWaveletSpectrum, scales_per_octave=2, kernel_sz=9)
    with pytest.raises(ValueError):
        _make(HealPixWaveletSpectrum, scales_per_octave=3)
    iso = _make(HealPixIsotropicWaveletSpectrum)
    assert iso.kernel_sz == 7 and iso.G == 1
    assert iso._levels.tolist() == [3, 2, 1, 0]


def test_edge_modes_keep_or_drop_pixels():
    x = torch.randn(len(REGION), dtype=torch.float64)
    x[_nan_mask(len(REGION), 0.2, 9)] = float("nan")
    mc = _make(HealPixWaveletSpectrum, cell_ids=REGION)(x)
    cp = _make(HealPixWaveletSpectrum, cell_ids=REGION, edge="complete")(x)
    cut = _make(HealPixWaveletSpectrum, cell_ids=REGION, coverage_weight=False,
                min_coverage=0.9)(x)
    # complete keeps only full kernels (e = 1), a subset of e >= 0.9
    assert torch.all(cp.n_eff <= cut.n_eff + 1e-9)
    assert torch.all(cut.n_eff <= mc.n_eff / 0.9 + 1e-9)
    assert torch.all(mc.n_eff > 0)


def test_function_form():
    x = torch.randn(12 * 4 ** 3, dtype=torch.float64)
    a = wavelet_power_spectrum(x, 3, isotropic=True, ellipsoid="sphere", **F64)
    b = _make(HealPixIsotropicWaveletSpectrum)(x)
    torch.testing.assert_close(a.power_total, b.power_total)


def test_down_mode_and_n_scales_validation():
    for bad in ("maxpool", "smooth_conservative"):
        with pytest.raises(ValueError, match="down_mode"):
            _make(HealPixWaveletSpectrum, level=2, down_mode=bad)
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="n_scales"):
            _make(HealPixIsotropicWaveletSpectrum, level=2, n_scales=bad)
    assert _make(HealPixIsotropicWaveletSpectrum, n_scales=1)._levels.tolist() == [3]
    # "mean" keeps q/m a weighted mean of the valid pixels too
    ws = _make(HealPixWaveletSpectrum, cell_ids=REGION, down_mode="mean")
    x = torch.full((len(REGION),), 287.15, dtype=torch.float64)
    x[_nan_mask(len(REGION), 0.2, 10)] = float("nan")
    out = ws(x)
    assert torch.allclose(out.monopole, torch.tensor(287.15, dtype=torch.float64))
    assert torch.all(out.power_total.abs() < 1e-18)
    assert abs(float(out.lowpass_var)) < 1e-18


def test_sample_without_valid_pixel_is_nan():
    ws = _make(HealPixIsotropicWaveletSpectrum, level=2)
    x = torch.randn(3, 12 * 4 ** 2, dtype=torch.float64)
    x[1] = float("nan")
    wt = torch.ones_like(x)
    wt[2] = 0.0
    out = ws(x, weights=wt)
    assert torch.isfinite(out.power_total[0]).all()
    assert torch.isfinite(out.monopole[0]) and torch.isfinite(out.lowpass_var[0])
    for b in (1, 2):
        assert torch.isnan(out.power_total[b]).all()
        assert torch.isnan(out.monopole[b]) and torch.isnan(out.lowpass_var[b])
        assert torch.all(out.n_eff[b] == 0)


def test_building_leaves_the_global_rng_alone():
    torch.manual_seed(123)
    ref = torch.rand(4)
    torch.manual_seed(123)
    ws = _make(HealPixWaveletSpectrum, level=2)
    assert torch.equal(torch.rand(4), ref)
    for conv in ws.convs:
        assert torch.all(conv.weight == 0) and torch.all(conv.bias == 0)


def test_monopole_is_the_weighted_mean_of_the_valid_pixels():
    ws = _make(HealPixWaveletSpectrum, cell_ids=REGION)
    n = len(REGION)
    x = 2.0 * torch.randn(3, n, dtype=torch.float64) + 5.0
    x[:, _nan_mask(n, 0.3, 12)] = float("nan")
    x[1, _nan_mask(n, 0.5, 13)] = float("nan")
    torch.testing.assert_close(ws(x).monopole, torch.nanmean(x, -1))
    wt = torch.as_tensor(np.random.default_rng(14).random(n))
    ok = torch.isfinite(x)
    expected = torch.where(ok, wt * x, 0).sum(-1) / torch.where(ok, wt, 0).sum(-1)
    torch.testing.assert_close(ws(x, weights=wt).monopole, expected)


def test_values_under_nan_or_zero_weight_are_ignored():
    ws = _make(HealPixWaveletSpectrum, cell_ids=REGION, s1=True)
    n = len(REGION)
    rng = np.random.default_rng(15)
    x = torch.randn(n, dtype=torch.float64)
    hole = _nan_mask(n, 0.2, 16)
    wt = torch.as_tensor(rng.uniform(0.2, 1.0, n))
    wt[hole] = 0.0
    a = ws(x, weights=wt)
    x2 = x.clone()
    x2[hole] = 1e3 * torch.randn(int(hole.sum()), dtype=torch.float64)
    b = ws(x2, weights=wt)                                       # other values under w = 0
    x3 = x.clone()
    x3[hole] = float("nan")
    c = ws(x3, weights=wt)                                       # NaN instead of w = 0
    for o in (b, c):
        for f in ("power", "s1", "monopole", "lowpass_var", "n_eff"):
            torch.testing.assert_close(getattr(o, f), getattr(a, f), equal_nan=True)


def test_a_uniform_taper_changes_nothing():
    # validity does not depend on the weights: a taper of 0.4 (below
    # min_valid) keeps every pixel in every band and only scales n_eff
    ws = _make(HealPixWaveletSpectrum, cell_ids=REGION)
    n = len(REGION)
    x = torch.randn(2, n, dtype=torch.float64)
    x[:, _nan_mask(n, 0.2, 17)] = float("nan")
    a = ws(x)
    b = ws(x, weights=torch.full((n,), 0.4, dtype=torch.float64))
    for f in ("power", "monopole", "lowpass_var"):
        torch.testing.assert_close(getattr(b, f), getattr(a, f), equal_nan=True)
    torch.testing.assert_close(b.n_eff, 0.4 * a.n_eff)


@pytest.mark.parametrize("cls", [HealPixWaveletSpectrum, HealPixIsotropicWaveletSpectrum])
def test_unsorted_cell_ids(cls):
    # base pixels 0, 4 and 5 at level 3: not an arange, then shuffled
    region = np.concatenate([np.arange(0, 64), np.arange(256, 384)]).astype(np.int64)
    perm = np.random.default_rng(18).permutation(len(region))
    x = torch.randn(2, len(region), dtype=torch.float64)
    x[:, _nan_mask(len(region), 0.2, 19)] = float("nan")
    a = _make(cls, cell_ids=region, s1=True)(x)
    b = _make(cls, cell_ids=region[perm], s1=True)(x[:, perm])
    for f in ("power", "s1", "monopole", "lowpass_var", "n_eff"):
        torch.testing.assert_close(getattr(b, f), getattr(a, f), equal_nan=True)


@pytest.mark.parametrize("cls", [HealPixWaveletSpectrum, HealPixIsotropicWaveletSpectrum])
def test_float32_matches_float64(cls):
    """float32, the default, on a regional domain with NaN and a 287 K offset.

    Measured over 20 maps (polar and equatorial domains): the band powers
    agree within the rounding of the float32 input (at most 2.3e-4, at
    level 0).  Two things keep them there: the field is centred before the
    descent (else up to 0.7 % off at the coarse bands), and the geometry of
    HealPixConv(missing="local_mean") is built in float64 (else up to 1.5 %
    off at the finest band, from stencil points at the domain border).
    lowpass_var, the variance of 3 level-0 pixels (down to 1e-4 K^2), is
    limited by the float32 resolution of the mean (3e-5 K at 287 K).
    """
    kw = dict(ellipsoid="sphere", device="cpu", cache_dir=None, cell_ids=REGION)
    n = len(REGION)
    gen = torch.Generator().manual_seed(21)
    x = 287.15 + torch.randn(2, n, dtype=torch.float64, generator=gen)   # SST in kelvin
    x[:, _nan_mask(n, 0.2, 20)] = float("nan")
    o64 = cls(3, dtype=torch.float64, **kw)(x)
    o32 = cls(3, dtype=torch.float32, **kw)(x.float())
    assert o32.power_total.dtype == torch.float32
    torch.testing.assert_close(o32.monopole.double(), o64.monopole, rtol=1e-6, atol=0)
    torch.testing.assert_close(o32.power_total.double(), o64.power_total, rtol=1e-3, atol=0)
    torch.testing.assert_close(o32.lowpass_var.double(), o64.lowpass_var, rtol=2e-2, atol=0)
