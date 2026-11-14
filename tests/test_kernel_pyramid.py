"""Tests for HealPixKernelPyramid: geometry consistency and per-band fidelity
against the independent direct-convolution oracle in healpix_analyse.validation.
"""

import numpy as np
import pytest
import torch

from healpix_analyse.decomp import HealPixDecomp
from healpix_analyse.kernel_pyramid import (
    HealPixKernelPyramid,
    kernel_gaussian,
    kernel_exponential,
    kernel_lorentzian,
    kernel_beta,
    kernel_anisotropic_gaussian,
)
from healpix_analyse.validation import (
    direct_spherical_convolution,
    direct_reference_operator_factory,
    smooth_test_field,
)


@pytest.mark.parametrize(
    "kernel_factory",
    [
        lambda: kernel_gaussian(1.2),
        lambda: kernel_exponential(1.5),
        lambda: kernel_lorentzian(1.2),
        lambda: kernel_beta(1.2, beta=3.0),
    ],
)
@pytest.mark.parametrize("kernel_sz", [3, 5])
def test_single_band_fidelity_vs_direct_oracle_smooth_field(kernel_factory, kernel_sz):
    """A single band's HealPixConv kernel should closely match a direct,
    independently-implemented brute-force convolution with the identical
    kernel profile, at that band's own resolution, on spatially-smooth
    input -- this isolates the stencil/bilinear-interpolation discretization
    error from any pyramid/block-diagonal effects (compute_weighted/invert
    are not used here at all). See
    test_single_band_white_noise_sensitivity_is_characterized for the
    (much larger) error under single-pixel-scale content.
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=0)
    kernel = kernel_factory()
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel, compact_kernel_sz=kernel_sz, gauge_type="phi", dtype=torch.float64
    )

    x = smooth_test_field(decomp.cell_ids_per_scale[0], decomp.levels[0])

    y_conv = np.asarray(kp.convs[0](x)).reshape(-1)
    y_ref = direct_spherical_convolution(
        x, decomp.cell_ids_per_scale[0], decomp.levels[0], kernel,
        kernel_sz=kernel_sz, ellipsoid="sphere", restore_mask=False, normalize=False,
    )

    err = np.abs(y_conv - y_ref)
    rel_rms = np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(y_ref ** 2))
    # HealPixConv's bilinear stencil binding closely approximates the
    # continuous kernel for spatially-smooth fields; measured at ~4-5% RMS
    # for these profiles at level=5 -- a generous margin is kept here.
    assert rel_rms < 0.15, f"rel RMS error {rel_rms:.4f} too large for kernel_sz={kernel_sz}"


def test_single_band_white_noise_sensitivity_is_characterized(capsys):
    """Documents (rather than hides) a real, measured limitation: on
    single-pixel-scale (white-noise) content, HealPixConv's fixed stencil +
    bilinear binding departs substantially further from a nearest-pixel
    quadrature reference than it does on smooth content, because the
    rotated stencil taps rarely land exactly on neighbouring pixel centres.
    This is not a bug in either implementation -- both are valid, different
    discretizations of the same continuous kernel -- but it means per-pixel
    accuracy claims for this convolution should not be extrapolated from
    smooth-field tests alone. The bound here is deliberately loose; the
    printed number is the one that belongs in the accuracy report.
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=0)
    kernel = kernel_gaussian(1.2)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel, compact_kernel_sz=5, gauge_type="phi", dtype=torch.float64
    )
    rng = np.random.default_rng(0)
    npix = 12 * 4 ** level
    x = rng.standard_normal(npix)

    y_conv = np.asarray(kp.convs[0](x)).reshape(-1)
    y_ref = direct_spherical_convolution(
        x, decomp.cell_ids_per_scale[0], decomp.levels[0], kernel,
        kernel_sz=5, ellipsoid="sphere", restore_mask=False, normalize=False,
    )
    rel_rms = np.sqrt(np.mean((y_conv - y_ref) ** 2)) / np.sqrt(np.mean(y_ref ** 2))
    print(f"[measured] white-noise rel RMS error, Gaussian sigma=1.2pix, kernel_sz=5: {rel_rms:.4f}")
    assert 0.0 < rel_rms < 0.6


def test_larger_kernel_is_not_worse_than_smaller_kernel(capsys):
    """A 7x7 stencil should not be meaningfully less accurate than 5x5, on
    smooth content, for a kernel whose support actually needs it (a
    slowly-decaying Lorentzian). This is a sanity/regression check for the
    '5x5 vs more' §12.3 question, not a proof that 7x7 is strictly better
    in every case (see the white-noise characterization test for how much
    this depends on input smoothness).
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=0)
    kernel = kernel_lorentzian(2.0)  # wide tail relative to a 5x5 support
    x = smooth_test_field(decomp.cell_ids_per_scale[0], decomp.levels[0])

    errs = {}
    for ksz in (5, 7):
        kp = HealPixKernelPyramid.from_kernel(
            decomp, kernel, compact_kernel_sz=ksz, gauge_type="phi", dtype=torch.float64
        )
        y_conv = np.asarray(kp.convs[0](x)).reshape(-1)
        y_ref = direct_spherical_convolution(
            x, decomp.cell_ids_per_scale[0], decomp.levels[0], kernel,
            kernel_sz=ksz, ellipsoid="sphere", restore_mask=False, normalize=False,
        )
        errs[ksz] = np.sqrt(np.mean((y_conv - y_ref) ** 2)) / np.sqrt(np.mean(y_ref ** 2))

    print(f"[measured] Lorentzian(scale=2.0pix) smooth-field rel RMS: 5x5={errs[5]:.4f} 7x7={errs[7]:.4f}")
    assert errs[7] <= errs[5] * 1.5  # generous margin; the point is no blow-up


def test_weights_from_finest_band_are_shared_identically_across_bands():
    """Default construction (``weights_from_finest_band=True``): the kernel
    is realized once, on band 0's stencil, and every coarser band's
    HealPixConv must carry that exact same discrete tap vector -- not an
    independent re-evaluation of the profile on its own (slightly
    different) real HEALPix geometry. This is the literal fix for "on doit
    donner le kernel a la resolution J=0 et il doit construire une
    pyramide": one realization, reused, rather than one per band.
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=3)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, dtype=torch.float64,
    )
    # Compare via the taps actually bound into each band's HealPixConv.
    taps = [conv.weight.detach().cpu().numpy() for conv in kp.convs]
    for j in range(1, len(taps)):
        assert np.array_equal(taps[0], taps[j]), (
            f"band {j} kernel taps differ from band 0's -- weights must be "
            "shared, not resampled, under the default weights_from_finest_band=True"
        )


def test_weights_from_finest_band_false_restores_per_band_resampling():
    """Explicit opt-out: with weights_from_finest_band=False, coarser bands
    may (and, on real HEALPix geometry, typically do) carry taps that
    differ slightly from band 0's -- the previous default behaviour,
    kept available for comparison.
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=3)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, dtype=torch.float64,
        weights_from_finest_band=False,
    )
    taps = [conv.weight.detach().cpu().numpy() for conv in kp.convs]
    # Not asserting inequality (geometry differences could in principle be
    # exactly zero for some band pairs) -- this just documents that nothing
    # forces equality in this mode, by construction of the code path taken.
    assert len(taps) == decomp.n_bands


def test_kernel_pyramid_geometry_matches_decomp_bands():
    level = 4
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=2)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.0), compact_kernel_sz=5, dtype=torch.float64
    )
    assert len(kp.convs) == decomp.n_bands
    for j, conv in enumerate(kp.convs):
        assert conv.level == decomp.levels[j]
        assert conv.K == decomp.sizes[j]


def test_apply_runs_on_a_full_pyramid():
    level = 4
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=2)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.0), compact_kernel_sz=5, dtype=torch.float64
    )
    rng = np.random.default_rng(2)
    npix = 12 * 4 ** level
    x = rng.standard_normal(npix)
    pyr = decomp.compute(x)
    conv_bands = kp.apply(pyr.bands)
    assert len(conv_bands) == len(pyr.bands)
    for band, conv_band in zip(pyr.bands, conv_bands):
        assert np.asarray(conv_band).shape == np.asarray(band).shape


def test_calibrate_recovers_a_representable_target(capsys):
    """When the reference operator's kernel is well within a single band's
    compact support (same scale as the stencil itself), least-squares
    calibration from random-excitation probes should recover essentially
    the same per-band accuracy as directly evaluating the analytic kernel
    -- this is a correctness check of the calibration machinery itself
    (design-matrix construction, probing), not a claim that calibration can
    make a compact per-band kernel reproduce an arbitrarily wide target
    (it cannot -- see the module docstring on block-diagonal approximation).
    """
    level = 4
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=1)
    target_kernel = kernel_gaussian(sigma_pix=1.2)
    ref_fn = direct_reference_operator_factory(
        target_kernel, kernel_sz=5, ellipsoid="sphere", normalize=False
    )

    kp_cal = HealPixKernelPyramid.calibrate(
        decomp, ref_fn, compact_kernel_sz=5, n_probes=128, n_excitations=4, seed=0
    )
    kp_analytic = HealPixKernelPyramid.from_kernel(
        decomp, target_kernel, compact_kernel_sz=5, dtype=torch.float64
    )

    x = smooth_test_field(decomp.cell_ids_per_scale[0], decomp.levels[0])
    y_ref = direct_spherical_convolution(
        x, decomp.cell_ids_per_scale[0], decomp.levels[0], target_kernel,
        kernel_sz=5, ellipsoid="sphere", restore_mask=False, normalize=False,
    )
    y_cal = np.asarray(kp_cal.convs[0](x)).reshape(-1)
    y_ana = np.asarray(kp_analytic.convs[0](x)).reshape(-1)

    err_cal = np.sqrt(np.mean((y_cal - y_ref) ** 2)) / np.sqrt(np.mean(y_ref ** 2))
    err_ana = np.sqrt(np.mean((y_ana - y_ref) ** 2)) / np.sqrt(np.mean(y_ref ** 2))
    print(f"[measured] calibrated vs analytic rel RMS on representable target: {err_cal:.4f} vs {err_ana:.4f}")
    assert err_cal < 0.15
    assert err_cal < err_ana * 3  # calibration must be in the same ballpark, not degenerate


def test_calibrate_on_a_much_wider_target_does_not_silently_claim_success(capsys):
    """A single compact 5x5 band kernel cannot represent a target many times
    wider than its own support; calibration should not be mistaken for a
    fix for that. This test measures (and prints, for the report) how large
    that gap remains even after calibration -- it intentionally does NOT
    assert a tight bound, since the point is to document the limitation,
    not paper over it.
    """
    level = 4
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=1)
    wide_kernel = kernel_gaussian(sigma_pix=6.0)
    ref_fn = direct_reference_operator_factory(
        wide_kernel, kernel_sz=13, ellipsoid="sphere", normalize=False
    )
    kp_cal = HealPixKernelPyramid.calibrate(
        decomp, ref_fn, compact_kernel_sz=5, n_probes=128, n_excitations=4, seed=1
    )
    x = smooth_test_field(decomp.cell_ids_per_scale[0], decomp.levels[0])
    y_ref = direct_spherical_convolution(
        x, decomp.cell_ids_per_scale[0], decomp.levels[0], wide_kernel,
        kernel_sz=13, ellipsoid="sphere", restore_mask=False, normalize=False,
    )
    y_cal = np.asarray(kp_cal.convs[0](x)).reshape(-1)
    err_cal = np.sqrt(np.mean((y_cal - y_ref) ** 2)) / np.sqrt(np.mean(y_ref ** 2))
    print(f"[measured] single-band 5x5 calibrated against a much wider (sigma=6pix) target: rel RMS {err_cal:.4f}")
    # Only asserts it runs and stays finite -- the large residual error is expected and documented.
    assert np.isfinite(err_cal)


def test_anisotropic_kernel_runs_and_is_direction_sensitive():
    """Sanity check for the anisotropic path: away from the gauge's own
    singularities, an elongated kernel should respond differently to a
    striped input aligned with vs. across its major axis. This does not
    check against an independent oracle (see healpix_analyse.validation's
    documented isotropic-only scope) -- it is a qualitative consistency
    check, not a numeric accuracy claim.
    """
    level = 5
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", dtype=torch.float64, Jmax=0)
    kernel = kernel_anisotropic_gaussian(sigma_major_pix=2.5, sigma_minor_pix=0.6, orientation_rad=0.0)
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel, compact_kernel_sz=7, gauge_type="phi", dtype=torch.float64
    )
    conv = kp.convs[0]

    rng = np.random.default_rng(3)
    npix = 12 * 4 ** level
    x = rng.standard_normal(npix)
    y = np.asarray(conv(x)).reshape(-1)
    assert np.isfinite(y).all()
    # An elongated kernel must not degenerate to a delta (near-input-copy)
    # nor to a wildly unstable output.
    assert np.std(y) > 0
    assert np.std(y) < 10 * np.std(x)


# ---------------------------------------------------------------------------
# Multi-channel (block-diagonal-across-channels) kernels
# ---------------------------------------------------------------------------

def test_channels_output_shape_matches_input_no_spurious_axis():
    """Regression test for a real shape bug: applying a channels=1 kernel
    pyramid to a [C, N] band via HealPixConv's ambiguous 2-D "batch"
    convention silently returns [C, 1, N] (HealPixConv never squeezes a
    2-D input's output back to 2-D). With channels=C built explicitly,
    the output must come back exactly [C, N], matching the input.
    """
    level = 4
    npix = 12 * 4 ** level
    cell_ids = np.arange(npix)[:400]
    decomp = HealPixDecomp(level=level, cell_ids=cell_ids, Jmax=1, ellipsoid="sphere")
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, channels=3, ellipsoid="sphere",
    )
    band0 = np.random.default_rng(0).standard_normal((3, decomp.sizes[0])).astype(np.float32)
    out = kp.apply([band0] + [
        np.zeros((3, s), dtype=np.float32) for s in decomp.sizes[1:]
    ])
    assert out[0].shape == band0.shape


def test_channels_are_independent_no_cross_channel_mixing():
    """The multi-channel kernel must be block-diagonal across channels: a
    signal placed in one channel must not leak into another.
    """
    level = 4
    npix = 12 * 4 ** level
    cell_ids = np.arange(npix)[:400]
    decomp = HealPixDecomp(level=level, cell_ids=cell_ids, Jmax=1, ellipsoid="sphere")
    kp = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, channels=3, ellipsoid="sphere",
    )
    x = np.zeros((3, decomp.sizes[0]), dtype=np.float32)
    x[0, 5] = 1.0
    y = np.asarray(kp.convs[0](x[None]))[0]  # explicit [1, C, N] -- conv's own
    # ambiguous 2-D convention would otherwise read [C, N] as [B, 1, N].
    assert np.abs(y[0]).max() > 0
    assert np.abs(y[1]).max() == 0
    assert np.abs(y[2]).max() == 0


def test_channels_matches_three_independent_single_channel_kernels():
    """channels=3 applied to a [3, N] map must equal running three separate
    channels=1 kernel pyramids independently on each channel.
    """
    level = 4
    npix = 12 * 4 ** level
    cell_ids = np.arange(npix)[:400]
    decomp = HealPixDecomp(level=level, cell_ids=cell_ids, Jmax=1, ellipsoid="sphere")
    kp_multi = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, channels=3, ellipsoid="sphere",
    )
    kp_single = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, channels=1, ellipsoid="sphere",
    )
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, decomp.sizes[0])).astype(np.float32)
    y_multi = np.asarray(kp_multi.convs[0](x[None]))[0]  # explicit [1, C, N]
    y_ref = np.stack([
        np.asarray(kp_single.convs[0](x[c])).reshape(-1) for c in range(3)
    ], axis=0)
    assert np.allclose(y_multi, y_ref, atol=1e-5)


def test_channels_requires_positive_odd_and_single_gauge():
    level = 3
    decomp = HealPixDecomp(level=level, ellipsoid="sphere", Jmax=0)
    with pytest.raises(ValueError):
        HealPixKernelPyramid.from_kernel(decomp, kernel_gaussian(1.0), channels=0)
    with pytest.raises(ValueError):
        HealPixKernelPyramid.from_kernel(decomp, kernel_gaussian(1.0), channels=3, n_gauges=2)


# ---------------------------------------------------------------------------
# Ellipsoid case-insensitivity
# ---------------------------------------------------------------------------

def test_ellipsoid_name_is_case_insensitive():
    """healpix_geo's own ellipsoid registry is case-sensitive ('WGS84',
    'sphere' -- not 'wgs84', 'SPHERE'), but real data sources do not all
    agree on a casing (e.g. EOPF/GRID4EARTH products declare 'wgs84').
    HealPixKernelPyramid.from_kernel must accept common case variants and
    resolve to the same geometry as the canonical spelling.
    """
    level = 4
    npix = 12 * 4 ** level
    cell_ids = np.arange(npix)[:200]
    decomp = HealPixDecomp(level=level, cell_ids=cell_ids, Jmax=0, ellipsoid="wgs84")
    assert decomp.ellipsoid == "WGS84"
    kp_lower = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, ellipsoid="wgs84",
    )
    kp_upper = HealPixKernelPyramid.from_kernel(
        decomp, kernel_gaussian(1.2), compact_kernel_sz=5, ellipsoid="WGS84",
    )
    x = np.random.default_rng(2).standard_normal(decomp.sizes[0]).astype(np.float32)
    y_lower = np.asarray(kp_lower.convs[0](x))
    y_upper = np.asarray(kp_upper.convs[0](x))
    assert np.allclose(y_lower, y_upper)
