"""
HealPixDown / HealPixUp: maxpool on regional domains (#93), reference test of
the Gaussian matrix (#61), and the CSR operator built once (#59, point 2).
"""

import numpy as np
import pytest
import torch

import healpix_geo
from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp


def _regional_ids(level, frac, seed):
    """A random, shuffled subset of the cells of ``level``."""
    rng = np.random.default_rng(seed)
    n = 12 * 4**level
    ids = np.flatnonzero(rng.random(n) < frac)
    return rng.permutation(ids)


# ---------------------------------------------------------------------------
# #93: maxpool with incomplete parents
# ---------------------------------------------------------------------------

def test_maxpool_issue_93_example():
    down = HealPixDown(2, mode="maxpool", cell_ids=[100, 4, 5, 6], device="cpu")
    y, ids = down(np.array([50.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(ids, [1, 25])
    np.testing.assert_array_equal(y, [3.0, 50.0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_maxpool_regional_matches_brute_force(seed):
    level = 3
    ids = _regional_ids(level, 0.4, seed)
    rng = np.random.default_rng(100 + seed)
    # negative values: a child borrowed from another parent would show
    x = -rng.random((2, ids.size)) * 10.0 - 1.0

    down = HealPixDown(level, mode="maxpool", cell_ids=ids, device="cpu",
                       dtype=torch.float64)
    y, ids_out = down(x)

    np.testing.assert_array_equal(ids_out, np.unique(ids // 4))
    expected = np.full((2, ids_out.size), -np.inf)
    row = {p: i for i, p in enumerate(ids_out)}
    for j, c in enumerate(ids):
        i = row[c // 4]
        expected[:, i] = np.maximum(expected[:, i], x[:, j])
    np.testing.assert_array_equal(y, expected)


def test_maxpool_full_sphere_unchanged():
    level = 2
    x = np.random.default_rng(0).standard_normal(12 * 4**level)
    y, _ = HealPixDown(level, mode="maxpool", device="cpu",
                       dtype=torch.float64)(x)
    np.testing.assert_array_equal(y, x.reshape(-1, 4).max(axis=1))


# ---------------------------------------------------------------------------
# #61: reference test of _build_smooth_matrix
# ---------------------------------------------------------------------------

def _unit(lon_deg, lat_deg):
    lon, lat = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _brute_force_smooth(down):
    """Dense [N_out, N_in] matrix of HealPixDown 'smooth', by brute force.

    Every pair (coarse centre, fine centre) is tested against the radius;
    nothing comes from cone_coverage.
    """
    lev_in, lev_out = down.level_in, down.level_out
    ids_in = (down._cell_ids_in if down.partial
              else np.arange(down.N_in_full))
    ids_out = down.cell_ids_out
    lon_o, lat_o = healpix_geo.nested.healpix_to_lonlat(
        ids_out, lev_out, ellipsoid=down.ellipsoid)
    lon_i, lat_i = healpix_geo.nested.healpix_to_lonlat(
        ids_in, lev_in, ellipsoid=down.ellipsoid)
    u_o, u_i = _unit(lon_o, lat_o), _unit(lon_i, lat_i)
    cross = np.linalg.norm(np.cross(u_o[:, None], u_i[None, :]), axis=-1)
    gamma = np.arctan2(cross, u_o @ u_i.T)                    # [N_out, N_in]
    w = np.exp(-0.5 * (gamma / down.sigma_rad) ** 2)
    w[gamma > down.radius_rad] = 0.0
    assert (w.sum(axis=1) > 0).all()     # no fallback row in these cases
    if down.weight_norm == "l1":
        w /= w.sum(axis=1, keepdims=True)
    elif down.weight_norm == "l2":
        w /= np.sqrt((w * w).sum(axis=1, keepdims=True))
    return w


@pytest.mark.parametrize("ellipsoid", ["sphere", "WGS84"])
@pytest.mark.parametrize("weight_norm", ["l1", "l2", "none"])
@pytest.mark.parametrize("regional", [False, True])
def test_smooth_matrix_matches_brute_force(ellipsoid, weight_norm, regional):
    level = 3
    ids = _regional_ids(level, 0.5, 7) if regional else None
    down = HealPixDown(level, mode="smooth", ellipsoid=ellipsoid,
                       weight_norm=weight_norm, cell_ids=ids,
                       device="cpu", dtype=torch.float64)
    M = down.sparse_matrix().to_dense().numpy()
    np.testing.assert_allclose(M, _brute_force_smooth(down),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("radius_deg", [12.0, 20.0])
def test_smooth_matrix_brute_force_other_radius(radius_deg):
    level = 2
    down = HealPixDown(level, mode="smooth", ellipsoid="sphere",
                       radius_deg=radius_deg, cell_ids=_regional_ids(level, 0.6, 3),
                       device="cpu", dtype=torch.float64)
    M = down.sparse_matrix().to_dense().numpy()
    np.testing.assert_allclose(M, _brute_force_smooth(down),
                               rtol=1e-12, atol=1e-14)


def test_smooth_matrix_fallback_rows():
    """No fine centre within the radius: uniform weights over the cells of
    the cone present in cell_ids, which include the present children."""
    level = 2
    ids = _regional_ids(level, 0.6, 3)
    down = HealPixDown(level, mode="smooth", ellipsoid="sphere", radius_deg=6.0,
                       cell_ids=ids, device="cpu", dtype=torch.float64)
    M = down.sparse_matrix().to_dense().numpy()
    np.testing.assert_allclose(M.sum(axis=1), 1.0, rtol=1e-12)
    for i, p in enumerate(down.cell_ids_out):
        row = M[i][M[i] > 0]
        np.testing.assert_allclose(row, row[0], rtol=1e-12)
        children = np.flatnonzero(ids // 4 == p)
        assert (M[i, children] > 0).all()


# ---------------------------------------------------------------------------
# #59 point 2: CSR operator built once
# ---------------------------------------------------------------------------

def _operators(regional):
    level = 3
    ids = _regional_ids(level, 0.5, 11) if regional else None
    kw = dict(device="cpu", dtype=torch.float64)
    down = HealPixDown(level, mode="smooth", cell_ids=ids, **kw)
    cons = HealPixDown(level, mode="smooth_conservative", cell_ids=ids, **kw)
    up = HealPixUp(level - 1, cell_ids=None if ids is None
                   else np.unique(ids // 4), **kw)
    paired = HealPixUp(level - 1, paired_down=down, up_norm="adjoint", **kw)
    return down, cons, up, paired


@pytest.mark.parametrize("regional", [False, True])
def test_forward_matches_sparse_matrix(regional):
    down, _, up, paired = _operators(regional)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, down.N_in))
    y, _ = down(x)
    M = down.sparse_matrix().to_dense().numpy()
    np.testing.assert_allclose(y, x @ M.T, rtol=1e-12, atol=1e-13)

    z = rng.standard_normal((3, paired.N_in))
    np.testing.assert_allclose(paired(z)[0], z @ M, rtol=1e-12, atol=1e-13)

    MT = up._sparse_coo().to_dense().numpy()          # [N_out, N_in]
    z = rng.standard_normal(up.N_in)
    yu, _ = up(z)
    np.testing.assert_allclose(yu, (MT @ z) / up._col_sum.numpy(),
                               rtol=1e-12, atol=1e-13)


def test_forward_builds_no_sparse_tensor(monkeypatch):
    down, cons, up, _ = _operators(regional=True)
    x = torch.randn(2, down.N_in, dtype=torch.float64)
    xu = torch.randn(2, up.N_in, dtype=torch.float64)

    def _fail(*a, **k):
        raise AssertionError("sparse_coo_tensor called in forward")

    monkeypatch.setattr(torch, "sparse_coo_tensor", _fail)
    down(x)
    cons(x)
    up(xu)


@pytest.mark.parametrize("regional", [False, True])
def test_gradcheck(regional):
    down, cons, up, paired = _operators(regional)
    for op in (down, cons, up, paired):
        x = torch.randn(2, op.N_in, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: op(t)[0], (x,))
    x = torch.randn(2, down.N_in, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(lambda t: down(t)[0] ** 2, (x,))


def test_gradient_is_transpose():
    down, _, _, _ = _operators(regional=True)
    x = torch.randn(down.N_in, dtype=torch.float64, requires_grad=True)
    g = torch.randn(down.N_out, dtype=torch.float64)
    down(x)[0].backward(g)
    M = down.sparse_matrix().to_dense()
    torch.testing.assert_close(x.grad, M.T @ g)


def test_buffers_follow_to_and_state_dict():
    level = 2
    down = HealPixDown(level, device="cpu", dtype=torch.float32)
    up = HealPixUp(level - 1, device="cpu", dtype=torch.float32)
    x = np.random.default_rng(0).standard_normal(down.N_in)
    y32, _ = down(x)

    down.double()
    up.double()
    assert down._M_val.dtype == torch.float64
    y64, _ = down(x)
    assert y64.dtype == np.float64
    np.testing.assert_allclose(y64, y32, rtol=1e-6, atol=1e-6)
    assert up(y64)[0].dtype == np.float64

    # the matrix is not persisted; old state dicts with COO keys still load
    assert not any(k.startswith("_M") for k in down.state_dict())
    legacy = {"_M_indices": torch.zeros(2, 1, dtype=torch.long),
              "_M_values": torch.zeros(1)}
    down.load_state_dict(legacy, strict=True)
    up.load_state_dict(dict(legacy, _col_sum=up._col_sum, _col_l2=up._col_l2),
                       strict=True)

    # read-only views of the former COO buffers
    M = down.sparse_matrix()
    torch.testing.assert_close(down._M_indices, M.indices())
    torch.testing.assert_close(down._M_values, M.values())


# ---------------------------------------------------------------------------
# dtype rule: every mode follows .to(dtype); 16-bit casts keep the matrix
# ---------------------------------------------------------------------------

DOWN_MODES = ("smooth", "smooth_conservative", "mean", "maxpool")


@pytest.mark.parametrize("half", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("regional", [False, True])
def test_half_precision_cast(half, regional):
    level = 3
    ids = _regional_ids(level, 0.5, 5) if regional else None
    rng = np.random.default_rng(1)
    for mode in DOWN_MODES:
        ref = HealPixDown(level, mode=mode, cell_ids=ids, device="cpu")
        op = HealPixDown(level, mode=mode, cell_ids=ids, device="cpu").to(half)
        assert op.dtype == half
        x = torch.as_tensor(rng.standard_normal((2, op.N_in)), dtype=torch.float32)
        y, _ = op(x.to(half))
        assert y.dtype == half
        y_ref, _ = ref(x.to(half).float())
        torch.testing.assert_close(y.float(), y_ref, rtol=1e-2, atol=1e-2)
        # numpy input: bfloat16 is returned as float32 (numpy has no bfloat16)
        y_np, _ = op(x[0].numpy())
        assert y_np.dtype == (np.float32 if half == torch.bfloat16 else np.float16)
        if mode in ("smooth", "smooth_conservative"):
            assert op._M_val.dtype == torch.float32
        # gradient flows back to the 16-bit input
        if mode != "maxpool":
            xh = x.to(half).requires_grad_(True)
            op(xh)[0].sum().backward()
            assert xh.grad.dtype == half and torch.isfinite(xh.grad.float()).all()

    up_ids = None if ids is None else np.unique(ids // 4)
    for kw in (dict(cell_ids=up_ids), dict(paired_down=HealPixDown(
            level, mode="mean", cell_ids=ids, device="cpu"))):
        ref = HealPixUp(level - 1, device="cpu", **kw)
        up = HealPixUp(level - 1, device="cpu", **kw).to(half)
        z = torch.as_tensor(rng.standard_normal((2, up.N_in)), dtype=torch.float32)
        yu, _ = up(z.to(half))
        assert yu.dtype == half and up._col_sum.dtype == torch.float32
        torch.testing.assert_close(yu.float(), ref(z.to(half).float())[0],
                                   rtol=1e-2, atol=1e-2)
    # back to float32: the kept-precision buffers are unchanged
    up.float()
    torch.testing.assert_close(up(z)[0], ref(z)[0])


@pytest.mark.parametrize("half", [torch.bfloat16, torch.float16])
def test_half_precision_constructor(half):
    down = HealPixDown(2, device="cpu", dtype=half)
    up = HealPixUp(1, device="cpu", dtype=half)
    assert down._M_val.dtype == torch.float32 and up._M_val.dtype == torch.float32
    x = torch.randn(down.N_in).to(half)
    y, _ = down(x)
    assert y.dtype == half and up(y)[0].dtype == half


def test_every_mode_follows_double():
    level = 2
    ids = _regional_ids(level, 0.6, 2)
    x = np.random.default_rng(0).standard_normal(len(ids)).astype(np.float32)
    for mode in DOWN_MODES:
        op = HealPixDown(level, mode=mode, cell_ids=ids, device="cpu").double()
        assert op.dtype == torch.float64
        assert op(x)[0].dtype == np.float64
        assert op(torch.as_tensor(x))[0].dtype == torch.float64
    up = HealPixUp(level - 1, cell_ids=np.unique(ids // 4), device="cpu").double()
    assert up(np.zeros(up.N_in, np.float32))[0].dtype == np.float64
