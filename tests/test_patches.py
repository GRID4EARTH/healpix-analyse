"""Tests of HealPixPatches (issue #109, section 2)."""

import numpy as np
import pytest
import torch

import healpix_geo
import healpix_analyse._patches_backend as pb
from healpix_analyse.dino import nested_to_tiles
from healpix_analyse.healpix_interp import get_interp_val
from healpix_analyse.patches import HealPixPatches

F64 = dict(device="cpu", dtype=torch.float64)
CUDA_TRITON = pytest.mark.skipif(not pb.triton_available(), reason="needs CUDA and Triton")


def _lonlat(ids, level):
    lon, lat = healpix_geo.nested.healpix_to_lonlat(np.asarray(ids), level, ellipsoid="sphere")
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _smooth_field(level, ids=None):
    ids = np.arange(12 * 4**level) if ids is None else ids
    lon, lat = np.radians(_lonlat(ids, level))
    return np.cos(lat) ** 2 * np.sin(3 * lon) + np.sin(2 * lat)


def _regional_ids(level, seed=0):
    """Cells of base faces 0-4 at ``level``, shuffled."""
    ids = np.arange(5 * 4**level)
    return np.random.default_rng(seed).permutation(ids)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def test_gauge_values_match_independent_sampling():
    """phi gauge: pixel (a, b) is the point at angular offset (u_b, u_a) cells
    along (e_theta, e_phi), interpolated bilinearly."""
    L, N = 4, 4
    hp = HealPixPatches(L, N, **F64)
    x = np.random.default_rng(1).standard_normal(hp.K)
    y = hp(torch.as_tensor(x))[:, 0, 0].numpy()                     # [M, N, N]

    lon, lat = np.radians(_lonlat(hp.centre_ids.numpy(), L - 2))
    th = np.pi / 2 - lat
    n = np.stack([np.sin(th) * np.cos(lon), np.sin(th) * np.sin(lon), np.cos(th)], -1)
    e_th = np.stack([np.cos(th) * np.cos(lon), np.cos(th) * np.sin(lon), -np.sin(th)], -1)
    e_ph = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    u = np.arange(N) - (N - 1) / 2
    alpha = np.sqrt(4 * np.pi / (12 * 4**L))
    ref = np.empty_like(y)
    for a in range(N):
        for b in range(N):
            d = np.hypot(u[a], u[b]) * alpha
            az = np.arctan2(u[a], u[b])
            v = (np.cos(d) * n
                 + np.sin(d) * (np.cos(az) * e_th + np.sin(az) * e_ph))
            vlat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
            vlon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
            ref[:, a, b] = get_interp_val(x, vlon, vlat, L)
    np.testing.assert_allclose(y, ref, atol=1e-10)


def test_nested_matches_nested_to_tiles():
    L, N = 4, 4
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 12 * 4**L)).astype(np.float32)
    hp = HealPixPatches(L, N, mode="nested", device="cpu")
    y = hp(torch.as_tensor(x)[None])[0, :, 0]                       # [M, C, N, N]
    tiles, parents, _, _ = nested_to_tiles(x.T, np.arange(x.shape[1]), L, L - 2)
    np.testing.assert_array_equal(parents, hp.centre_ids.numpy())
    np.testing.assert_array_equal(y.numpy(), tiles)


@pytest.mark.parametrize("mode", ["gauge", "nested"])
def test_patch_size_one_is_identity(mode):
    hp = HealPixPatches(3, 1, mode=mode, **F64)
    x = torch.randn(2, 3, hp.K, dtype=torch.float64)
    y = hp(x)
    assert y.shape == (2, hp.K, 1, 3, 1, 1)
    torch.testing.assert_close(y.reshape(2, hp.K, 3).transpose(1, 2), x, atol=1e-12, rtol=0)


def test_output_shapes_and_numpy():
    hp = HealPixPatches(3, 2, n_gauges=3, device="cpu")
    x = np.random.default_rng(0).standard_normal(hp.K).astype(np.float32)
    y = hp(x)
    assert isinstance(y, np.ndarray) and y.shape == (hp.M, 3, 1, 2, 2)
    assert hp(torch.as_tensor(x)[None]).shape == (1, hp.M, 3, 1, 2, 2)
    assert hp(torch.as_tensor(x)[None, None]).shape == (1, hp.M, 3, 1, 2, 2)
    assert hp.coverage.shape == (hp.M, 3, 2, 2)
    assert hp.M == 12 * 4**2


def test_tokens_are_free_views():
    hp = HealPixPatches(3, 2, n_gauges=2, device="cpu")
    y = hp(torch.randn(2, 4, hp.K))
    assert y.is_contiguous()
    assert y.flatten(2).shape == (2, hp.M, 2 * 4 * 4)
    assert y.flatten(2).data_ptr() == y.data_ptr()


# ---------------------------------------------------------------------------
# Adjoint identity <P x, y> = <x, fold y>
# ---------------------------------------------------------------------------

def _cases():
    reg = _regional_ids(3)
    return [
        dict(mode="gauge", centres="coarse", cell_ids=None),
        dict(mode="gauge", centres="all", cell_ids=None),
        dict(mode="gauge", centres="coarse", cell_ids=reg),
        dict(mode="gauge", centres="all", cell_ids=reg[:200]),
        dict(mode="nested", centres="coarse", cell_ids=None),
        dict(mode="nested", centres="coarse", cell_ids=reg[:300]),
    ]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c['mode']}-{c['centres']}-"
                         f"{'full' if c['cell_ids'] is None else len(c['cell_ids'])}")
@pytest.mark.parametrize("sel", [None, "shared", "per_sample"])
def test_adjoint_identity(case, sel):
    hp = HealPixPatches(3, 2, n_gauges=2, **case, **F64)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, hp.K, dtype=torch.float64, generator=g)
    index = None
    if sel == "shared":
        index = torch.randperm(hp.M, generator=g)[: max(1, hp.M // 3)]
    elif sel == "per_sample":
        index = torch.stack([torch.randperm(hp.M, generator=g)[: max(1, hp.M // 3)]
                             for _ in range(2)])
    y = hp(x, index=index)
    z = torch.randn(y.shape, dtype=torch.float64, generator=g)
    lhs = (y * z).sum()
    rhs = (x * hp.fold(z, index=index, norm="none")).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-12)


def test_fold_norms():
    hp = HealPixPatches(4, 4, centres="all", n_gauges=2, **F64)
    c = torch.full((1, 2, hp.K), 3.0, dtype=torch.float64)
    torch.testing.assert_close(hp.fold(hp(c), norm="weight"), c, atol=1e-12, rtol=0)
    count = hp.fold(torch.ones(1, hp.M, 2, 1, 4, 4, dtype=torch.float64), norm="none")
    count_ones = hp._geometry().norm("count")
    assert torch.all(count_ones >= 1)                       # every cell is read
    raw = hp.fold(hp(c), norm="none")
    torch.testing.assert_close(hp.fold(hp(c), norm="count"), raw / count_ones, atol=1e-12, rtol=0)
    torch.testing.assert_close(count[0, 0], hp._geometry().norm("weight"), atol=1e-12, rtol=0)


def test_fold_norm_with_index_counts_selected_patches_only():
    hp = HealPixPatches(3, 2, n_gauges=1, **F64)
    index = torch.tensor([[0, 5, 9], [3, 4, 100]])
    c = torch.ones(2, 1, hp.K, dtype=torch.float64)
    out = hp.fold(hp(c, index=index), index=index, norm="weight")
    read = hp.fold(torch.ones(2, 3, 1, 1, 2, 2, dtype=torch.float64), index=index, norm="none")
    torch.testing.assert_close(out, (read > 0).to(out.dtype), atol=1e-12, rtol=0)


def test_nested_fold_inverts_extraction():
    hp = HealPixPatches(4, 8, mode="nested", **F64)
    x = torch.randn(2, 3, hp.K, dtype=torch.float64)
    for norm in ("weight", "count", "none"):
        torch.testing.assert_close(hp.fold(hp(x), norm=norm), x, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,centres,regional", [
    ("gauge", "coarse", False), ("gauge", "all", True),
    ("nested", "coarse", False), ("nested", "coarse", True)])
def test_gradcheck(mode, centres, regional):
    ids = _regional_ids(2)[:90] if regional else None
    hp = HealPixPatches(2, 2, mode=mode, centres=centres, cell_ids=ids, n_gauges=2, **F64)
    x = torch.randn(2, 2, hp.K, dtype=torch.float64, requires_grad=True)
    index = torch.tensor([[1, 3, 0], [2, 2, 5]]) % hp.M
    assert torch.autograd.gradcheck(lambda t: hp(t), (x,))
    assert torch.autograd.gradcheck(lambda t: hp(t, index=index), (x,))
    y = torch.randn(2, hp.M, hp.G, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: hp.fold(t), (y,))
    ys = torch.randn(2, 3, hp.G, 2, 2, 2, dtype=torch.float64, requires_grad=True)
    distinct = torch.tensor([[1, 3, 0], [2, 4, 5]]) % hp.M
    assert torch.autograd.gradcheck(lambda t: hp.fold(t, index=distinct), (ys,))
    # a repeated centre: the plain adjoint sums the copies, a normalised fold raises
    assert torch.autograd.gradcheck(lambda t: hp.fold(t, index=index, norm="none"), (ys,))
    with pytest.raises(ValueError):
        hp.fold(ys, index=torch.tensor([[1, 1, 2]]))
    assert torch.autograd.gradgradcheck(lambda t: hp(t, index=index) ** 2, (x,))
    assert torch.autograd.gradgradcheck(lambda t: hp(t) ** 2, (x,))   # nested fast path too


# ---------------------------------------------------------------------------
# Geometry: gauge equivariance, nested/gauge consistency, regional domains
# ---------------------------------------------------------------------------

def test_gauge_shift_is_a_quarter_turn():
    """With G = 4, gauge g + 2 is gauge g rotated by 90 degrees."""
    hp = HealPixPatches(5, 4, n_gauges=4, **F64)
    y = hp(torch.randn(1, 1, hp.K, dtype=torch.float64))[0, :, :, 0]
    for g in (0, 1):
        torch.testing.assert_close(y[:, g + 2], torch.rot90(y[:, g], 1, (-2, -1)),
                                   atol=1e-10, rtol=0)


def test_gauge_shift_with_projected_ref():
    hp = HealPixPatches(4, 2, n_gauges=2, centres="all", gauge_type="projected_ref",
                        singularity_lonlat=(20.0, 5.0), **F64)
    y = hp(torch.randn(1, 1, hp.K, dtype=torch.float64))[0, :, :, 0]
    torch.testing.assert_close(y[:, 1], torch.rot90(y[:, 0], 1, (-2, -1)), atol=1e-10, rtol=0)


def test_cosmo_gauge_rotates_the_other_way_in_the_south():
    hp = HealPixPatches(4, 4, centres="all", n_gauges=4, gauge_type="cosmo", **F64)
    y = hp(torch.randn(1, 1, hp.K, dtype=torch.float64))[0, :, :, 0]
    _, lat = _lonlat(hp.centre_ids.numpy(), 4)
    north, south = torch.as_tensor(lat > 0), torch.as_tensor(lat < 0)
    torch.testing.assert_close(y[north, 2], torch.rot90(y[north, 0], 1, (-2, -1)),
                               atol=1e-10, rtol=0)
    torch.testing.assert_close(y[south, 2], torch.rot90(y[south, 0], 3, (-2, -1)),
                               atol=1e-10, rtol=0)


def test_nested_and_gauge_agree_on_coarse_centres():
    """Same centres, same area: equal patch means for a smooth field; in the
    equatorial belt the NESTED tile is the gauge patch at 45 degrees (G = 4,
    g = 1), rotated by 270 degrees and mirrored."""
    L, N = 5, 4
    f = torch.as_tensor(_smooth_field(L)).view(1, 1, -1)
    hg = HealPixPatches(L, N, n_gauges=4, **F64)
    hn = HealPixPatches(L, N, mode="nested", **F64)
    torch.testing.assert_close(hg.centre_ids, hn.centre_ids)
    yg = hg(f)[0, :, :, 0]
    yn = hn(f)[0, :, 0, 0]
    assert float((yg[:, 0].mean((-2, -1)) - yn.mean((-2, -1))).abs().max()) < 0.03
    _, clat = _lonlat(hg.centre_ids.numpy(), L - 2)
    eq = torch.as_tensor(np.abs(clat) < 30)
    match = torch.rot90(yg[:, 1], 3, (-2, -1)).flip(-1)
    err = float((match[eq] - yn[eq]).abs().mean())
    other = float((yg[eq, 0] - yn[eq]).abs().mean())
    assert err < 0.01 < other


@pytest.mark.parametrize("centres", ["coarse", "all"])
def test_regional_domain_matches_full_sphere(centres):
    """An unsorted regional domain gives the full-sphere patches wherever the
    patch lies inside it."""
    L = 4
    ids = _regional_ids(L)
    full = HealPixPatches(L, 4, centres=centres, n_gauges=2, **F64)
    reg = HealPixPatches(L, 4, centres=centres, cell_ids=ids, n_gauges=2, **F64)
    x = torch.randn(2, 3, full.K, dtype=torch.float64)
    yf = full(x)
    yr = reg(x[..., torch.as_tensor(ids)])
    inside = reg.coverage.flatten(1).min(1).values > 1 - 1e-12
    assert inside.sum() > reg.M // 8
    torch.testing.assert_close(yr[:, inside], yf[:, reg.centre_ids[inside]], atol=1e-12, rtol=0)
    # outside the domain: zero weights, coverage < 1
    assert float(reg.coverage.min()) < 1


def test_nested_regional_incomplete_parents():
    L = 3
    ids = _regional_ids(L)[:150]
    hp = HealPixPatches(L, 4, mode="nested", cell_ids=ids, **F64)
    assert not hp.contiguous
    x = torch.randn(1, 2, hp.K, dtype=torch.float64)
    y = hp(x)
    full = HealPixPatches(L, 4, mode="nested", **F64)
    xf = torch.zeros(1, 2, full.K, dtype=torch.float64)
    xf[..., torch.as_tensor(ids)] = x
    torch.testing.assert_close(y, full(xf)[:, hp.centre_ids])
    assert int(hp.coverage.sum()) == len(ids)
    torch.testing.assert_close(hp.fold(y), x)


def test_nested_contiguous_fast_path_and_zorder():
    L, N = 4, 4
    # sorted domain made of complete level-2 cells
    parents = np.array([3, 7, 8, 40, 41])
    ids = (parents[:, None] * 16 + np.arange(16)).ravel()
    hp = HealPixPatches(L, N, mode="nested", cell_ids=ids, **F64)
    assert hp.contiguous
    x = torch.randn(2, 3, hp.K, dtype=torch.float64)
    fast = hp(x)
    generic = hp(x, index=torch.arange(hp.M))
    torch.testing.assert_close(fast, generic, atol=0, rtol=0)
    z = hp(x, zorder=True)
    assert z.shape == (2, hp.M, 1, 3, N * N) and z.data_ptr() == x.data_ptr()
    torch.testing.assert_close(z[..., hp._perm], fast.flatten(-2), atol=0, rtol=0)
    with pytest.raises(ValueError):
        HealPixPatches(L, N, cell_ids=ids, **F64)(x, zorder=True)


def test_gauge_coarse_coverage_full_sphere():
    """On the full sphere every pixel is inside the domain; the fold reaches
    every cell for N = 4 except a few next to the poles."""
    hp = HealPixPatches(6, 4, **F64)
    assert float(hp.coverage.min()) == 1.0
    unread = int((hp._geometry().norm("count") == 0).sum())
    assert unread <= 16


def test_errors():
    with pytest.raises(ValueError):
        HealPixPatches(3, 3)
    with pytest.raises(ValueError):
        HealPixPatches(2, 8)
    with pytest.raises(ValueError):
        HealPixPatches(3, 2, mode="nested", centres="all")
    hp = HealPixPatches(2, 2, **F64)
    with pytest.raises(ValueError):
        hp(torch.zeros(5))
    with pytest.raises(ValueError):
        hp(torch.zeros(1, 1, hp.K), index=[hp.M])
    with pytest.raises(ValueError):
        hp.fold(torch.zeros(1, 3, 1, 1, 2, 2))


def test_module_to_rebuilds_geometry():
    hp = HealPixPatches(3, 2, **F64)
    x = torch.randn(1, 1, hp.K, dtype=torch.float64)
    y = hp(x)
    hp.float()
    torch.testing.assert_close(hp(x.float()), y.float(), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Triton backend (CUDA)
# ---------------------------------------------------------------------------

@CUDA_TRITON
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float16, 2e-3),
                                       (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("mode,centres", [("gauge", "coarse"), ("gauge", "all"),
                                          ("nested", "coarse")])
def test_triton_matches_torch(dtype, tol, mode, centres):
    ids = _regional_ids(5)[:3000]
    kw = dict(mode=mode, centres=centres, cell_ids=ids, n_gauges=2 if mode == "gauge" else 1)
    ht = HealPixPatches(5, 4, device="cuda", **kw)
    hr = HealPixPatches(5, 4, device="cuda", backend="torch", **kw)
    x = torch.randn(2, 5, ht.K, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().float().requires_grad_()
    index = torch.stack([torch.randperm(ht.M)[: ht.M // 4] for _ in range(2)]).cuda()
    for idx in (None, index):
        y = ht(x, index=idx)
        yr = hr(xr, index=idx)
        torch.testing.assert_close(y.float(), yr, atol=tol, rtol=tol)
        gy = torch.randn_like(yr)
        gx, = torch.autograd.grad(y, x, gy.to(dtype))
        gxr, = torch.autograd.grad(yr, xr, gy)
        torch.testing.assert_close(gx.float(), gxr, atol=tol * 10, rtol=tol)
        for norm in ("weight", "count", "none"):
            torch.testing.assert_close(ht.fold(gy.to(dtype), index=idx, norm=norm).float(),
                                       hr.fold(gy, index=idx, norm=norm), atol=tol * 10, rtol=tol)


@CUDA_TRITON
def test_triton_backward_is_deterministic():
    hp = HealPixPatches(7, 4, centres="all", n_gauges=2, device="cuda")
    x = torch.randn(2, 8, hp.K, device="cuda", requires_grad=True)
    gy = torch.randn(2, hp.M, 2, 8, 4, 4, device="cuda")
    g1, = torch.autograd.grad(hp(x), x, gy)
    g2, = torch.autograd.grad(hp(x), x, gy)
    assert torch.equal(g1, g2)


@CUDA_TRITON
@pytest.mark.parametrize("C", [3, 16])
def test_nested_fast_path_cuda_deterministic(C):
    """Full-sphere NESTED fast path on CUDA: extraction, backward and fold are
    permutations (gathers), so two runs give the same bits and fold inverts
    the extraction exactly."""
    hp = HealPixPatches(7, 8, mode="nested", device="cuda")
    assert hp.contiguous
    x = torch.randn(2, C, hp.K, device="cuda", requires_grad=True)
    gy = torch.randn(2, hp.M, 1, C, 8, 8, device="cuda")
    y1 = hp(x)
    g1, = torch.autograd.grad(y1, x, gy)
    y2 = hp(x)
    g2, = torch.autograd.grad(y2, x, gy)
    assert torch.equal(y1, y2) and torch.equal(g1, g2)
    assert torch.equal(hp.fold(y1.detach()), x.detach())
    assert torch.equal(g1, hp.fold(gy))
    hr = HealPixPatches(7, 8, mode="nested", device="cuda", backend="torch")
    torch.testing.assert_close(g1, hr.fold(gy, index=torch.arange(hp.M, device="cuda"),
                                           norm="none"), atol=0, rtol=0)


@CUDA_TRITON
@pytest.mark.parametrize("C", [1, 3, 5, 16])
def test_triton_fold_deterministic_all_layouts(C):
    """The fold reads a channels-last copy for every C: two runs agree bitwise
    and match the PyTorch fold."""
    hp = HealPixPatches(6, 4, centres="all", n_gauges=2, device="cuda")
    hr = HealPixPatches(6, 4, centres="all", n_gauges=2, device="cuda", backend="torch")
    y = torch.randn(2, hp.M, 2, C, 4, 4, device="cuda")
    for norm in ("none", "weight", "count"):
        a, b = hp.fold(y, norm=norm), hp.fold(y, norm=norm)
        assert torch.equal(a, b)
        torch.testing.assert_close(a, hr.fold(y, norm=norm), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shift", [None, 0, 2])
@pytest.mark.parametrize("mode,centres,regional", [
    ("gauge", "coarse", False), ("gauge", "all", False), ("nested", "coarse", True)])
def test_footprint_marks_exactly_the_blocks_read(mode, centres, regional, shift, monkeypatch):
    """The mark of a selection is exactly the set of blocks of 2**shift cells
    that its patches read: the masked fold may skip the others."""
    monkeypatch.setattr(pb, "MARK_SHIFT", shift)
    ids = _regional_ids(3)[:500] if regional else None
    hp = HealPixPatches(3, 2, mode=mode, centres=centres, cell_ids=ids, n_gauges=2, **F64)
    g = hp._geometry()
    sel = torch.stack([torch.randperm(hp.M)[: hp.M // 4] for _ in range(2)]).int()
    mark, used = pb.selection_mark(g, sel)
    if shift is not None:
        assert used == shift
    assert g.footprint()[0].shape[1] <= pb.MARK_MAX_F or shift is not None
    mark = mark[:, :-1].bool()
    for b in range(2):
        rows = (sel[b].long()[:, None] * g.G * g.P + torch.arange(g.G * g.P)).reshape(-1)
        read = torch.zeros(g.K, dtype=torch.bool)
        nz = (g.w[:, rows] != 0).reshape(-1)
        read[g.idx[:, rows].long().reshape(-1)[nz]] = True
        cells = read.view(-1)
        blocks = torch.zeros_like(mark[b])
        blocks[torch.nonzero(cells).view(-1) >> used] = True
        assert torch.equal(mark[b], blocks)


@CUDA_TRITON
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode,centres", [("gauge", "coarse"), ("gauge", "all"),
                                          ("nested", "coarse")])
def test_masked_fold_skip_is_bitwise_identical(dtype, mode, centres):
    """Skipping the cells no selected patch reads changes nothing, bit for bit:
    the fold, its normalisation (through the backward) and the extraction
    backward, for a shared and a per-sample selection."""
    ids = _regional_ids(6)[:20000] if mode == "nested" else None
    hp = HealPixPatches(6, 4, mode=mode, centres=centres, cell_ids=ids, n_gauges=2,
                        device="cuda")
    G = hp.G
    shared = torch.randperm(hp.M, device="cuda")[: hp.M // 4]
    per = torch.stack([torch.randperm(hp.M, device="cuda")[: hp.M // 2] for _ in range(2)])
    x = torch.randn(2, 5, hp.K, device="cuda", dtype=dtype)
    try:
        for index in (shared, per):
            y = torch.randn(2, index.shape[-1], G, 5, 4, 4, device="cuda", dtype=dtype)
            out = {}
            for skip in (False, True):
                pb.MASKED_FOLD_SKIP = skip
                folds = []
                for norm in ("none", "weight", "count"):
                    yy = y.clone().requires_grad_()
                    f = hp.fold(yy, index=index, norm=norm)
                    gy, = torch.autograd.grad(f, yy, torch.ones_like(f))
                    folds += [f, gy]
                xx = x.clone().requires_grad_()
                gx, = torch.autograd.grad(hp(xx, index=index), xx,
                                          torch.ones(2, index.shape[-1], G, 5, 4, 4,
                                                     device="cuda", dtype=dtype))
                out[skip] = folds + [gx]
            for a, b in zip(out[False], out[True]):
                assert torch.equal(a, b)
            # the masked fold equals the full fold of the patches expanded
            # with zeros (the fold of #113), bit for bit
            sel = index.view(1, -1) if index.dim() == 1 else index
            full = pb._expand_selected(y, sel.int(), hp.M)
            torch.testing.assert_close(hp.fold(y, index=index, norm="none"),
                                       hp.fold(full, norm="none"), atol=0, rtol=0)
    finally:
        pb.MASKED_FOLD_SKIP = "auto"


@CUDA_TRITON
def test_triton_double_backward():
    hp = HealPixPatches(5, 4, device="cuda")
    x = torch.randn(1, 3, hp.K, device="cuda", requires_grad=True)
    g, = torch.autograd.grad((hp(x) ** 2).sum(), x, create_graph=True)
    gg, = torch.autograd.grad(g.sum(), x)
    hr = HealPixPatches(5, 4, device="cuda", backend="torch")
    xr = x.detach().requires_grad_()
    gr, = torch.autograd.grad((hr(xr) ** 2).sum(), xr, create_graph=True)
    ggr, = torch.autograd.grad(gr.sum(), xr)
    torch.testing.assert_close(gg, ggr, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# pixel_size: a step in metres, independent of the level of the map
# ---------------------------------------------------------------------------

def _cell_metres(level, ellipsoid="sphere"):
    from healpix_analyse._ellipsoid import authalic_radius
    return authalic_radius(ellipsoid) * np.sqrt(4 * np.pi / (12 * 4**level))


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
def test_pixel_size_of_one_cell_is_the_default(ellipsoid):
    L, N = 5, 4
    ids = _regional_ids(L)
    ref = HealPixPatches(L, N, cell_ids=ids, centres=L - 1, ellipsoid=ellipsoid, **F64)
    hp = HealPixPatches(L, N, cell_ids=ids, centres=L - 1, ellipsoid=ellipsoid,
                        pixel_size=_cell_metres(L, ellipsoid), **F64)
    assert hp.centre_level == ref.centre_level and hp.M == ref.M
    x = torch.as_tensor(_smooth_field(L, ids))
    torch.testing.assert_close(hp(x), ref(x), atol=1e-12, rtol=0)
    torch.testing.assert_close(hp.coverage, ref.coverage, atol=1e-12, rtol=0)
    assert "pixel_size=" in repr(hp) and "pixel_size=" not in repr(ref)


def test_pixel_size_grid_has_a_constant_step_from_the_pole():
    """Every grid point at its exact geodesic distance from the pole; along
    the two axes through the centre the step is exactly the pixel size."""
    from healpix_analyse.patches import _patch_grid
    N, step = 8, 3e-3
    v = _patch_grid(N, 2**10, step).reshape(N, N, 3)
    u = np.arange(N) - (N - 1) / 2
    d = np.arctan2(np.hypot(v[..., 0], v[..., 1]), v[..., 2])
    np.testing.assert_allclose(d, step * np.hypot(u[None, :], u[:, None]), rtol=1e-12, atol=1e-18)
    w = _patch_grid(N + 1, 2**10, step).reshape(N + 1, N + 1, 3)   # odd: an axis through the pole
    c = N // 2
    for line in (w[c, :], w[:, c]):
        gap = np.arctan2(np.linalg.norm(np.cross(line[1:], line[:-1]), axis=-1),
                         (line[1:] * line[:-1]).sum(-1))
        np.testing.assert_allclose(gap, step, rtol=1e-10)


def test_pixel_size_values_match_independent_sampling():
    """A map at level L read with pixel_size = 1.6 cells: the patch is the
    bilinear sample at offsets 1.6 * u cells, as for the default step."""
    L, N, f = 6, 4, 1.6
    hp = HealPixPatches(L, N, centres=L - 2, pixel_size=f * _cell_metres(L), **F64)
    hp_ids = hp.centre_ids.numpy()
    x = np.random.default_rng(2).standard_normal(hp.K)
    y = hp(torch.as_tensor(x))[:, 0, 0].numpy()
    lon, lat = np.radians(_lonlat(hp_ids, L - 2))
    th = np.pi / 2 - lat
    n = np.stack([np.sin(th) * np.cos(lon), np.sin(th) * np.sin(lon), np.cos(th)], -1)
    e_th = np.stack([np.cos(th) * np.cos(lon), np.cos(th) * np.sin(lon), -np.sin(th)], -1)
    e_ph = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    u = np.arange(N) - (N - 1) / 2
    alpha = f * np.sqrt(4 * np.pi / (12 * 4**L))
    ref = np.empty_like(y)
    for a in range(N):
        for b in range(N):
            d = np.hypot(u[a], u[b]) * alpha
            az = np.arctan2(u[a], u[b])
            v = np.cos(d) * n + np.sin(d) * (np.cos(az) * e_th + np.sin(az) * e_ph)
            vlat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
            vlon = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
            ref[:, a, b] = get_interp_val(x, vlon, vlat, L)
    np.testing.assert_allclose(y, ref, atol=1e-10)


def test_pixel_size_same_ground_grid_at_any_map_level():
    """The same pixel size at levels L and L + 1 samples the same points:
    on a smooth field the patches agree to the interpolation error."""
    N, d = 8, 0.7 * _cell_metres(6)
    c = np.arange(20) * 37                                         # level-3 centres
    a = HealPixPatches(6, N, centres=3, centre_ids=c, pixel_size=d, **F64)
    b = HealPixPatches(7, N, centres=3, centre_ids=c, pixel_size=d, **F64)
    ya = a(torch.as_tensor(_smooth_field(6)))
    yb = b(torch.as_tensor(_smooth_field(7)))
    assert float((ya - yb).abs().max()) < 0.02
    assert float((ya - yb).abs().max()) < 0.2 * float(ya.std())


def test_pixel_size_coarse_centres():
    L, N = 10, 16
    cell = _cell_metres(L)
    ids = np.arange(4**6)                     # the level-10 cells of level-4 cell 0
    assert HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], centres="coarse", pixel_size=cell,
                          **F64).centre_level == L - 4
    # 16 px of 1.6 cells = 25.6 cells: coarsest level not larger than the patch is L - 4
    assert HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], pixel_size=1.6 * cell, **F64).centre_level == L - 4
    assert HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], pixel_size=2.0 * cell, **F64).centre_level == L - 5
    assert HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], pixel_size=0.25 * cell, **F64).centre_level == L - 2
    assert HealPixPatches(L, 1, cell_ids=ids, centre_ids=[0], pixel_size=0.5 * cell, **F64).centre_level == L
    # centres coarser than the coarse level are refused, finer ones accepted
    HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], centres=L - 5, pixel_size=2.0 * cell, **F64)
    with pytest.raises(ValueError):
        HealPixPatches(L, N, cell_ids=ids, centre_ids=[0], centres=L - 6, pixel_size=2.0 * cell, **F64)


def test_pixel_size_errors():
    with pytest.raises(ValueError):
        HealPixPatches(3, 2, mode="nested", pixel_size=10.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            HealPixPatches(3, 2, pixel_size=bad)
