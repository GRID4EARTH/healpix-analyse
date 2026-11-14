"""Tests for :class:`healpix_analyse.wide_conv.HealPixWideConv`.

Kept at a small level so the whole file runs in seconds; the accuracy claims
are checked at the same level the notebook uses in spirit, just smaller.
"""
import numpy as np
import pytest
import torch

import healpix_geo.nested as hgn

from healpix_analyse.wide_conv import HealPixWideConv

LEVEL = 10
SIDE = 128
JMAX = 4


def _square_domain(level=LEVEL, side=SIDE, lon=0.0, lat=-20.0):
    """An exact `side` x `side` block of cells centred on (lon, lat).

    The default point sits well inside base face 4 at ``LEVEL`` (i=j=249 of
    1024), so the kernel images used below never run into a face edge.
    """
    centre = int(np.asarray(hgn.lonlat_to_healpix([lon], [lat], level))[0])
    face, i_c, j_c = [
        int(v[0]) for v in hgn.healpix_to_base_cell_coordinates([centre], level)
    ]
    half = side // 2
    jj, ii = np.meshgrid(
        np.arange(j_c - half, j_c + half), np.arange(i_c - half, i_c + half), indexing="ij"
    )
    cells = hgn.base_cell_coordinates_to_healpix(
        np.full(ii.size, face), ii.ravel(), jj.ravel(), level
    ).astype(np.int64)
    return np.sort(cells), (face, i_c, j_c)


def _exp_kernel_image(n=24, scale_pix=4.0):
    d = np.arange(-n, n + 1)
    dj, di = np.meshgrid(d, d, indexing="ij")
    return np.exp(-np.hypot(di, dj) / scale_pix)


def test_rejects_malformed_kernel_images():
    with pytest.raises(ValueError):
        HealPixWideConv(np.zeros((4, 4)), LEVEL)            # even side
    with pytest.raises(ValueError):
        HealPixWideConv(np.zeros((5, 7)), LEVEL)            # not square
    with pytest.raises(ValueError):
        HealPixWideConv(np.zeros(5), LEVEL)                 # not 2-D
    with pytest.raises(ValueError):
        HealPixWideConv(np.zeros((5, 5)), LEVEL, compact_kernel_sz=4)


def test_convolving_a_dirac_returns_the_kernel(capsys):
    """The defining property: conv(dirac) == the kernel image, laid on the grid."""
    cell_ids, _ = _square_domain()
    conv = HealPixWideConv(_exp_kernel_image(), LEVEL, Jmax=JMAX, compact_kernel_sz=5)

    centre = conv.reference_centre(cell_ids)
    x = np.zeros(cell_ids.size)
    x[np.searchsorted(cell_ids, centre)] = 1.0

    y = conv(x, cell_ids)
    K = conv.kernel_as_field(cell_ids, centre_cell=centre)

    rel = np.sqrt(np.mean((y - K) ** 2)) / np.sqrt(np.mean(K ** 2))
    print(f"[measured] level {LEVEL}, {SIDE}x{SIDE}, Jmax={JMAX}, 5x5: "
          f"rel RMS(conv(dirac) - K) = {rel:.4f}")
    assert rel < 0.25, f"convolving a Dirac should give back the kernel; rel RMS {rel}"
    # and it must be much better than what one compact 5x5 band alone can do
    single = HealPixWideConv(_exp_kernel_image(), LEVEL, Jmax=0, compact_kernel_sz=5)
    y1 = single(x, cell_ids)
    rel1 = np.sqrt(np.mean((y1 - K) ** 2)) / np.sqrt(np.mean(K ** 2))
    print(f"[measured] same kernel, Jmax=0 (single 5x5 band): rel RMS = {rel1:.4f}")
    assert rel < rel1 / 2, "the pyramid must beat a single compact band on a wide kernel"


def test_accepts_leading_dimensions_and_preserves_shape():
    cell_ids, _ = _square_domain()
    conv = HealPixWideConv(_exp_kernel_image(n=12), LEVEL, Jmax=3)

    x1 = np.random.default_rng(0).standard_normal(cell_ids.size)
    y1 = conv(x1, cell_ids)
    assert y1.shape == x1.shape

    for leading in [(3,), (2, 3)]:
        xn = np.broadcast_to(x1, leading + (cell_ids.size,)).copy()
        yn = conv(xn, cell_ids)
        assert yn.shape == xn.shape
        # every leading slice is filtered independently and identically
        for row in yn.reshape(-1, cell_ids.size):
            assert np.allclose(row, y1, atol=1e-10)


def test_linearity():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3)
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(cell_ids.size), rng.standard_normal(cell_ids.size)
    ya, yb = conv(a, cell_ids), conv(b, cell_ids)
    yab = conv(2.0 * a - 3.0 * b, cell_ids)
    assert np.allclose(yab, 2.0 * ya - 3.0 * yb, atol=1e-8)


def test_torch_input_returns_torch_on_the_same_device_and_dtype():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3)
    x = torch.zeros(cell_ids.size, dtype=torch.float32)
    x[0] = 1.0

    y = conv(x, cell_ids)
    assert torch.is_tensor(y)
    assert y.dtype == torch.float32
    assert y.device == x.device
    assert y.shape == x.shape

    # numpy in -> numpy out, same values
    y_np = conv(x.numpy(), cell_ids)
    assert isinstance(y_np, np.ndarray)
    assert np.allclose(y_np, y.numpy(), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available")
def test_gpu_tensor_round_trips():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3)
    x = torch.zeros(cell_ids.size, dtype=torch.float32, device="cuda")
    x[0] = 1.0
    y = conv(x, cell_ids)
    assert y.is_cuda and y.shape == x.shape


def test_domain_is_cached_and_refit_only_once():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3)
    x = np.zeros(cell_ids.size)
    x[0] = 1.0
    conv(x, cell_ids)
    first = conv._cache[conv._last_key]
    conv(x, cell_ids)
    assert conv._cache[conv._last_key] is first, "second call on the same domain refitted"
    assert len(conv._cache) == 1


def test_fit_residuals_are_reported_per_band():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3, compact_kernel_sz=5)
    conv(np.zeros(cell_ids.size), cell_ids)
    res = conv.fit_residuals()
    assert len(res) == conv.decomp.n_bands
    assert all(0.0 <= r < 1.5 for r in res)
    kernels = conv.band_kernels()
    assert len(kernels) == conv.decomp.n_bands
    assert all(k.shape == (5, 5) for k in kernels)


def test_wrong_pixel_count_is_rejected():
    cell_ids, _ = _square_domain(side=64)
    conv = HealPixWideConv(_exp_kernel_image(n=8), LEVEL, Jmax=3)
    with pytest.raises(ValueError):
        conv(np.zeros(cell_ids.size + 1), cell_ids)


def test_kernel_clipped_by_the_domain_warns():
    cell_ids, _ = _square_domain(side=32)
    conv = HealPixWideConv(_exp_kernel_image(n=24), LEVEL, Jmax=2)   # 49x49 kernel, 32x32 domain
    with pytest.warns(RuntimeWarning, match="outside the domain"):
        conv.kernel_as_field(cell_ids)


def test_kernel_clipped_by_a_face_edge_warns():
    """A domain hugging a base-face edge clips the kernel rather than wrapping it."""
    level, side, n = 8, 32, 12
    edge = 2 ** level - 1                      # last row of the face
    face, i_c, j_c = 4, 16, edge - 2
    jj, ii = np.meshgrid(
        np.arange(j_c - side // 2, j_c + side // 2),
        np.arange(i_c - side // 2, i_c + side // 2), indexing="ij",
    )
    keep = (ii >= 0) & (ii <= edge) & (jj >= 0) & (jj <= edge)
    cells = np.sort(hgn.base_cell_coordinates_to_healpix(
        np.full(int(keep.sum()), face), ii[keep].ravel(), jj[keep].ravel(), level
    ).astype(np.int64))
    conv = HealPixWideConv(_exp_kernel_image(n=n), level, Jmax=2)
    with pytest.warns(RuntimeWarning, match="fall off base face"):
        conv.kernel_as_field(cells, centre_cell=int(hgn.base_cell_coordinates_to_healpix(
            [face], [i_c], [j_c], level)[0]))


# ---------------------------------------------------------------------------
# The three ways of specifying the kernel
# ---------------------------------------------------------------------------

# Two reference points, deliberately: the HEALPix lattice is a *square metric
# grid* in the equatorial belt (|lat| < 41.8 deg) and visibly sheared in the
# polar caps. Everything about metric kernels only bites in the second case.
BELT_LON, BELT_LAT = 0.0, -20.0     # equatorial belt; also _square_domain's centre
CAP_LON, CAP_LAT = 20.0, 65.0       # polar cap, well inside base face 0 at LEVEL
PIX_M = np.sqrt(4 * np.pi / (12 * (2 ** LEVEL) ** 2)) * 6371008.8   # ~6.4 km at level 10


def _lattice_steps(lon, lat, level=LEVEL):
    x_m, y_m = HealPixWideConv.lattice_offsets_m(level, lon, lat, 1)
    return np.hypot(x_m[1, 2], y_m[1, 2]), np.hypot(x_m[2, 1], y_m[2, 1])


def test_lattice_offsets_are_metric_and_centred():
    n = 4
    x_m, y_m = HealPixWideConv.lattice_offsets_m(LEVEL, CAP_LON, CAP_LAT, n)
    assert x_m.shape == y_m.shape == (2 * n + 1, 2 * n + 1)
    assert abs(x_m[n, n]) < 1e-9 and abs(y_m[n, n]) < 1e-9      # centre is the origin

    # hypot(x, y) must be the true great-circle distance to each lattice cell
    centre = int(np.asarray(hgn.lonlat_to_healpix([CAP_LON], [CAP_LAT], LEVEL))[0])
    face, i_c, j_c = [
        int(v[0]) for v in hgn.healpix_to_base_cell_coordinates([centre], LEVEL)
    ]
    d = np.arange(-n, n + 1)
    dj, di = np.meshgrid(d, d, indexing="ij")
    cells = hgn.base_cell_coordinates_to_healpix(
        np.full(di.size, face), (i_c + di).ravel(), (j_c + dj).ravel(), LEVEL
    ).astype(np.int64)
    lon, lat = [np.asarray(v) for v in hgn.healpix_to_lonlat(cells.tolist(), LEVEL)]
    clon, clat = [float(v[0]) for v in hgn.healpix_to_lonlat([centre], LEVEL)]
    l1, p1, l2, p2 = np.radians(lon), np.radians(lat), np.radians(clon), np.radians(clat)
    hav = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    r_true = 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1))) * 6371008.8

    assert np.allclose(np.hypot(x_m, y_m).ravel(), r_true, rtol=1e-9, atol=1e-6)


def test_lattice_is_square_in_the_belt_and_sheared_in_the_cap(capsys):
    """Why `from_function`/`from_grid` exist -- and when they are unnecessary."""
    si_b, sj_b = _lattice_steps(BELT_LON, BELT_LAT)
    si_c, sj_c = _lattice_steps(CAP_LON, CAP_LAT)
    print(f"[measured] level {LEVEL}, equatorial belt ({BELT_LON}, {BELT_LAT}): "
          f"i-step {si_b:.0f} m, j-step {sj_b:.0f} m, ratio {max(si_b, sj_b)/min(si_b, sj_b):.3f}")
    print(f"[measured] level {LEVEL}, polar cap     ({CAP_LON}, {CAP_LAT}): "
          f"i-step {si_c:.0f} m, j-step {sj_c:.0f} m, ratio {max(si_c, sj_c)/min(si_c, sj_c):.3f}")
    assert np.isclose(si_b, sj_b, rtol=0.01), "the belt lattice should be metrically square"
    assert not np.isclose(si_c, sj_c, rtol=0.05), "the cap lattice should be sheared"


def test_lattice_offsets_reject_a_lattice_running_off_the_face():
    with pytest.raises(ValueError, match="runs off base face"):
        HealPixWideConv.lattice_offsets_m(4, BELT_LON, BELT_LAT, n=100)   # face is 16x16


def test_from_function_is_isotropic_in_metres_not_in_pixels():
    n, r0 = 12, 4.0 * PIX_M
    conv = HealPixWideConv.from_function(
        lambda x, y: np.exp(-np.hypot(x, y) / r0), LEVEL, n, CAP_LON, CAP_LAT, Jmax=3
    )
    x_m, y_m = HealPixWideConv.lattice_offsets_m(LEVEL, CAP_LON, CAP_LAT, n)
    assert np.allclose(conv.kernel_image, np.exp(-np.hypot(x_m, y_m) / r0))

    # in the cap, a naive (i, j)-indexed kernel of the "same" shape is distorted
    d = np.arange(-n, n + 1)
    dj, di = np.meshgrid(d, d, indexing="ij")
    step_i = np.hypot(x_m, y_m)[n, n + 1]
    naive = np.exp(-np.hypot(di, dj) * step_i / r0)
    rel = np.linalg.norm(naive - conv.kernel_image) / np.linalg.norm(conv.kernel_image)
    print(f"[measured] polar cap: (i, j)-indexed vs metric kernel differ by {rel:.3f}")
    assert rel > 0.1, (
        "indexing by (i, j) should visibly distort a kernel defined in metres; "
        f"got only {rel:.3f} relative difference"
    )


def test_from_function_rejects_a_badly_shaped_return():
    with pytest.raises(ValueError, match="shaped like its inputs"):
        HealPixWideConv.from_function(
            lambda x, y: np.zeros(3), LEVEL, 4, BELT_LON, BELT_LAT
        )


def test_from_grid_bilinear_matches_the_same_kernel_evaluated_directly():
    """A raster in metres, resampled, must agree with evaluating the function."""
    n, r0 = 10, 4.0 * PIX_M
    m, step = 201, PIX_M / 4.0            # raster covers ~50 pixels, 4 samples per pixel
    ax = (np.arange(m) - (m - 1) / 2) * step
    gx, gy = np.meshgrid(ax, ax)
    raster = np.exp(-np.hypot(gx, gy) / r0)

    from_raster = HealPixWideConv.from_grid(
        raster, step, LEVEL, n, CAP_LON, CAP_LAT, Jmax=3
    )
    exact = HealPixWideConv.from_function(
        lambda x, y: np.exp(-np.hypot(x, y) / r0), LEVEL, n, CAP_LON, CAP_LAT, Jmax=3
    )
    rel = (np.linalg.norm(from_raster.kernel_image - exact.kernel_image)
           / np.linalg.norm(exact.kernel_image))
    print(f"[measured] bilinear raster vs exact evaluation: rel {rel:.5f}")
    assert rel < 5e-3, f"bilinear resampling should track the exact kernel; rel {rel:.4f}"


def test_from_grid_warns_when_the_raster_is_too_small():
    raster = np.ones((5, 5))
    with pytest.warns(RuntimeWarning, match="outside the raster"):
        HealPixWideConv.from_grid(raster, PIX_M / 10, LEVEL, n=12,
                                   lon=BELT_LON, lat=BELT_LAT, Jmax=2)


def test_from_grid_rejects_bad_inputs():
    with pytest.raises(ValueError, match="grid must be 2-D"):
        HealPixWideConv.from_grid(np.ones(5), 10.0, LEVEL, 4, BELT_LON, BELT_LAT)
    with pytest.raises(ValueError, match="must be positive"):
        HealPixWideConv.from_grid(np.ones((5, 5)), -1.0, LEVEL, 4, BELT_LON, BELT_LAT)


def test_all_three_recipes_convolve_a_dirac_into_their_own_kernel():
    """Whatever the recipe, conv(dirac) must reproduce that recipe's own kernel."""
    cell_ids, _ = _square_domain()                      # centred on the belt point
    n, r0 = 16, 4.0 * PIX_M
    x_m, y_m = HealPixWideConv.lattice_offsets_m(LEVEL, BELT_LON, BELT_LAT, n)

    m, step = 401, PIX_M / 4.0
    ax = (np.arange(m) - (m - 1) / 2) * step
    gx, gy = np.meshgrid(ax, ax)

    recipes = {
        "1: image": HealPixWideConv(
            np.exp(-np.hypot(x_m, y_m) / r0), LEVEL, Jmax=JMAX),
        "2: function of x, y in m": HealPixWideConv.from_function(
            lambda x, y: np.exp(-np.hypot(x, y) / r0), LEVEL, n,
            BELT_LON, BELT_LAT, Jmax=JMAX),
        "3: raster in m, bilinear": HealPixWideConv.from_grid(
            np.exp(-np.hypot(gx, gy) / r0), step, LEVEL, n,
            BELT_LON, BELT_LAT, Jmax=JMAX),
    }
    for name, conv in recipes.items():
        centre = conv.reference_centre(cell_ids)
        x = np.zeros(cell_ids.size)
        x[np.searchsorted(cell_ids, centre)] = 1.0
        y = conv(x, cell_ids)
        K = conv.kernel_as_field(cell_ids, centre_cell=centre)
        rel = np.sqrt(np.mean((y - K) ** 2)) / np.sqrt(np.mean(K ** 2))
        print(f"[measured] recipe '{name}': rel RMS(conv(dirac) - K) = {rel:.4f}")
        assert rel < 0.25, f"recipe '{name}' failed to reproduce its own kernel ({rel:.3f})"


def test_recipes_1_and_2_agree_in_the_equatorial_belt():
    """Where the lattice is metrically square, (i, j) indexing is already right."""
    n, r0 = 10, 4.0 * PIX_M
    x_m, y_m = HealPixWideConv.lattice_offsets_m(LEVEL, BELT_LON, BELT_LAT, n)
    step_i = np.hypot(x_m, y_m)[n, n + 1]
    d = np.arange(-n, n + 1)
    dj, di = np.meshgrid(d, d, indexing="ij")

    naive = np.exp(-np.hypot(di, dj) * step_i / r0)
    metric = np.exp(-np.hypot(x_m, y_m) / r0)
    rel = np.linalg.norm(naive - metric) / np.linalg.norm(metric)
    print(f"[measured] equatorial belt: (i, j)-indexed vs metric kernel differ by {rel:.4f}")
    assert rel < 0.02, "in the belt the two constructions should essentially coincide"


def test_from_radial_is_isotropic_on_the_ground():
    n, r0 = 12, 4.0 * PIX_M
    conv = HealPixWideConv.from_radial(
        lambda r: np.exp(-r / r0), LEVEL, n, CAP_LON, CAP_LAT, Jmax=3
    )
    x_m, y_m = HealPixWideConv.lattice_offsets_m(LEVEL, CAP_LON, CAP_LAT, n)
    assert np.allclose(conv.kernel_image, np.exp(-np.hypot(x_m, y_m) / r0))

    # cells at equal distance on the ground must carry equal weight, even
    # though they sit at very different (i, j) offsets in the polar cap
    r = np.hypot(x_m, y_m)
    ref = r[n, n + 2]                                   # some radius
    ring = np.abs(r - ref) < 0.02 * ref
    assert ring.sum() >= 4
    vals = conv.kernel_image[ring]
    assert np.ptp(vals) / vals.mean() < 0.05


def test_from_radial_matches_from_function_for_an_isotropic_kernel():
    n, r0 = 10, 3.0 * PIX_M
    radial = HealPixWideConv.from_radial(
        lambda r: np.exp(-r / r0), LEVEL, n, CAP_LON, CAP_LAT, Jmax=3)
    xy = HealPixWideConv.from_function(
        lambda x, y: np.exp(-np.hypot(x, y) / r0), LEVEL, n, CAP_LON, CAP_LAT, Jmax=3)
    assert np.allclose(radial.kernel_image, xy.kernel_image)


def test_from_radial_rejects_a_badly_shaped_return():
    with pytest.raises(ValueError, match="shaped like its input"):
        HealPixWideConv.from_radial(lambda r: np.zeros(3), LEVEL, 4, CAP_LON, CAP_LAT)


def test_from_radial_convolves_a_dirac_into_its_own_kernel(capsys):
    cell_ids, _ = _square_domain()
    r0 = 4.0 * PIX_M
    conv = HealPixWideConv.from_radial(
        lambda r: np.exp(-r / r0), LEVEL, 16, BELT_LON, BELT_LAT,
        Jmax=JMAX, compact_kernel_sz=5)
    centre = conv.reference_centre(cell_ids)
    x = np.zeros(cell_ids.size)
    x[np.searchsorted(cell_ids, centre)] = 1.0
    y = conv(x, cell_ids)
    K = conv.kernel_as_field(cell_ids, centre_cell=centre)
    rel = np.sqrt(np.mean((y - K) ** 2)) / np.sqrt(np.mean(K ** 2))
    print(f"[measured] recipe '4: fn(r) in m': rel RMS(conv(dirac) - K) = {rel:.4f}")
    assert rel < 0.25
