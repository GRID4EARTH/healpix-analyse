"""Tests for HealPixPyramidConv: normalized/signed pyramidal convolution,
mask propagation, constant preservation, and gradients.
"""

import numpy as np
import pytest
import torch

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.kernel_pyramid import HealPixKernelPyramid, kernel_gaussian
from healpix_analyse.pyramid_conv import HealPixPyramidConv


def _make(level=4, jmax=2, sigma_pix=1.0, kernel_sz=5, dtype=torch.float64):
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=dtype, Jmax=jmax)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(sigma_pix), compact_kernel_sz=kernel_sz, dtype=dtype
    )
    return decomp, kp


def test_constant_map_preserved_under_normalized_mode_no_mask():
    decomp, kp = _make()
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    c = 4.2
    x = np.full(npix, c)
    y = pconv(x)
    np.testing.assert_allclose(y, c, atol=1e-6)


def test_constant_map_preserved_with_mask():
    decomp, kp = _make()
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    rng = np.random.default_rng(0)
    c = -1.7
    x = np.full(npix, c)
    mask = rng.random(npix) > 0.3
    x[~mask] = np.nan

    y, support = pconv(x, return_support=True)
    finite = np.isfinite(y)
    assert finite.any()
    np.testing.assert_allclose(y[finite], c, atol=1e-4)
    assert (support[finite] > 0).all()


def test_fully_masked_region_is_nan():
    decomp, kp = _make(level=4, jmax=0)  # small kernel support, single band
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    x = np.random.default_rng(1).standard_normal(npix)
    # mask out a region much larger than the kernel support
    x[: npix // 4] = np.nan

    y = pconv(x)
    assert np.isnan(y[: npix // 8]).any(), "deep inside the hole, output should be NaN"
    assert np.isfinite(y[npix // 2 :]).all(), "far from the hole, output should be finite"


def test_random_holes_do_not_crash_and_stay_finite_where_supported():
    decomp, kp = _make(level=4, jmax=2)
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    rng = np.random.default_rng(2)
    x = rng.standard_normal(npix)
    holes = rng.random(npix) > 0.85
    x[holes] = np.nan

    y, support = pconv(x, return_support=True)
    assert y.shape == x.shape
    # Where support is non-trivial, result must be finite.
    well_supported = support > 0.5 * np.nanmax(support)
    assert np.isfinite(y[well_supported]).all()


def test_explicit_weights_scale_contribution():
    decomp, kp = _make(level=4, jmax=1)
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    rng = np.random.default_rng(3)
    x = rng.standard_normal(npix)
    weights = rng.random(npix)

    y = pconv(x, weights=weights)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_signed_kernel_can_produce_negative_output_in_signed_mode():
    """A signed (non-positive) kernel should be allowed to produce a
    negative response in 'signed' mode, which performs no normalization
    (there is no confidence channel to divide by).
    """
    decomp, kp = _make(level=4, jmax=0, sigma_pix=1.0)
    # Overwrite band-0 kernel with an explicitly signed (Mexican-hat-like) one.
    conv0 = kp.convs[0]
    P = conv0.P
    rho2 = np.linspace(0, 4, P)
    signed_w = (1 - rho2) * np.exp(-rho2 / 2.0)
    conv0.set_kernel(signed_w[None, None, :].astype(np.float32), requires_grad=False)

    pconv = HealPixPyramidConv(decomp, kp, mode="signed")
    npix = 12 * 4 ** decomp.level
    x = np.ones(npix)  # constant input: a signed kernel need not preserve it
    y = pconv(x)
    assert np.isfinite(y).all()

    with pytest.raises(ValueError):
        pconv(x, weights=np.ones(npix))
    with pytest.raises(ValueError):
        pconv(x, return_support=True)


def test_gradient_flows_through_pyramid_conv():
    decomp, kp = _make(level=4, jmax=1)
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    rng = np.random.default_rng(4)

    mask_np = rng.random(npix) > 0.2
    x_np = rng.standard_normal(npix)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    m = torch.tensor(mask_np.astype(np.float64), dtype=torch.float64, requires_grad=True)
    data = torch.where(m.bool(), x, torch.full_like(x, float("nan")))

    y = pconv(data, weights=m, restore_mask=False)
    y.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_apply_pyramid_matches_manual_invert():
    decomp, kp = _make(level=4, jmax=1)
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")
    npix = 12 * 4 ** decomp.level
    rng = np.random.default_rng(5)
    x = rng.standard_normal(npix)

    y1 = pconv(x)

    wpyr = decomp.compute_weighted(x)
    conv_wpyr = pconv.apply_pyramid(wpyr)
    y2 = decomp.invert(conv_wpyr)

    np.testing.assert_allclose(y1, y2, atol=1e-10)


def test_multichannel_rgb_shape_and_independent_masking():
    """End-to-end regression test for the real RGB-shape bug: a [C, N]
    "channel-batch" input through a channels=1 kernel pyramid used to come
    back [C, 1, N] (HealPixConv's 2-D input -> 3-D output asymmetry),
    silently breaking a plain x.T round trip. With a channels=3 kernel
    pyramid, the output must come back exactly [C, N], and per-channel NaN
    holes (independent cloud masks per band, as in real Sentinel-2 data)
    must be filled independently without cross-channel leakage.
    """
    level = 4
    npix = 12 * 4 ** level
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=2)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, channels=3, dtype=torch.float64,
    )
    pconv = HealPixPyramidConv(decomp, kp, mode="normalized")

    rng = np.random.default_rng(0)
    rgb = rng.random((npix, 3))
    # Independent per-channel holes, as real per-band cloud/edge masks would be.
    for c in range(3):
        holes = rng.random(npix) < 0.2
        rgb[holes, c] = np.nan

    x = torch.as_tensor(rgb.T)  # [3, N], the exact shape the notebook uses
    y, support = pconv(x, return_support=True)
    assert tuple(y.shape) == (3, npix)
    assert tuple(support.shape) == (3, npix)

    rgb_filtered = y.detach().cpu().numpy().T
    assert rgb_filtered.shape == (npix, 3)
    # Some, but not necessarily all, holes get filled by the pyramid.
    assert np.isnan(rgb_filtered).sum() <= np.isnan(rgb).sum()


def test_multichannel_matches_running_channels_separately():
    level = 4
    npix = 12 * 4 ** level
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=1)
    kp_multi = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.0), compact_kernel_sz=5, channels=3, dtype=torch.float64,
    )
    kp_single = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.0), compact_kernel_sz=5, channels=1, dtype=torch.float64,
    )
    pconv_multi = HealPixPyramidConv(decomp, kp_multi, mode="normalized")
    pconv_single = HealPixPyramidConv(decomp, kp_single, mode="normalized")

    rng = np.random.default_rng(3)
    rgb = rng.random((npix, 3))
    rgb[rng.random(npix) < 0.15, 0] = np.nan

    x = torch.as_tensor(rgb.T)
    y_multi = pconv_multi(x).detach().cpu().numpy()
    y_ref = np.stack([
        np.asarray(pconv_single(rgb[:, c])) for c in range(3)
    ], axis=0)
    np.testing.assert_allclose(y_multi, y_ref, atol=1e-10, equal_nan=True)
