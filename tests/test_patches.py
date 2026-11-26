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
