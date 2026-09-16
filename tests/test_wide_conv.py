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
