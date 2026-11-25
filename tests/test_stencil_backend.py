"""Equivalence of the vectorised / fused stencil backends with the reference.

The reference of the neighbourhood reductions is the per-cell loop of
``main`` (kept as ``HealPixNeighbourReducer._reference_reduce``); the
reference of the gradient is ``_torch_gradient_from_geometry`` (the
materialised implementation of ``main``).  Requirements (issue #109):
1e-12 in float64, exact for median / min / max, matching (sub)gradients.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from healpix_analyse import _stencil_backend as sb
from healpix_analyse._neighbourhood import build_neighbourhoods
from healpix_analyse.gradient import (
    HealPixGradient,
    _torch_gradient_from_geometry,
    gradient,
)
from healpix_analyse.neighbour_reduce import HealPixNeighbourReducer

CUDA = torch.cuda.is_available()
TRITON = sb.triton_available()

ALL_REDUCTIONS = ("mean", "sum", "min", "max", "median", "count", "std")
EXACT = ("min", "max", "median", "count")


def _spacing(level):
    return float(np.sqrt(4 * np.pi / (12 * 4**level)) * 6.371e6)


def _regional_ids(level, rng, fraction=0.3):
    n = 12 * 4**level
    start = int(rng.integers(0, n // 2))
    ids = np.arange(start, start + int(fraction * n))
    return rng.permutation(ids)


@pytest.fixture(scope="module")
def reducers():
    rng = np.random.default_rng(1)
    level = 3
    full = np.arange(12 * 4**level)
    regional = _regional_ids(level, rng)
    domain = rng.permutation(regional[: regional.size * 2 // 3])
    out = {
        "full": HealPixNeighbourReducer(full, level, radius_m=1.6 * _spacing(level)),
        "full_wide": HealPixNeighbourReducer(full, level, radius_m=2.6 * _spacing(level)),
        "regional_domain": HealPixNeighbourReducer(
            regional, level, radius_m=1.6 * _spacing(level), domain=domain,
        ),
        "no_self_cone": HealPixNeighbourReducer(
            regional, level, radius_m=1.2 * _spacing(level),
            include_self=False, neighbourhood="cone_coverage",
        ),
    }
    return out


def _field(shape, dtype, rng, special=True):
    x = torch.as_tensor(rng.normal(size=shape)).to(dtype)
    if special and dtype.is_floating_point:
        flat = x.view(-1)
        n = flat.numel()
        flat[rng.integers(0, n, 3)] = float("nan")
        flat[rng.integers(0, n, 3)] = float("inf")
        flat[rng.integers(0, n, 3)] = float("-inf")
        flat[rng.integers(0, n, 6)] = 0.25          # ties
        flat[rng.integers(0, n, 2)] = -0.0
        flat[rng.integers(0, n, 2)] = 0.0
    return x


def _assert_same(actual, expected, reduction, dtype):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if reduction in EXACT or not dtype.is_floating_point:
        a = actual.double() if actual.dtype.is_floating_point else actual
        e = expected.double() if expected.dtype.is_floating_point else expected
        if a.dtype.is_floating_point:
            assert torch.equal(torch.isnan(a), torch.isnan(e))
            fin = ~torch.isnan(a)
            assert torch.equal(a[fin], e[fin])
            assert torch.equal(torch.signbit(a[fin]), torch.signbit(e[fin]))
        else:
            assert torch.equal(a, e)
    else:
        # 16-bit: one rounding step (the kernels accumulate in fp32 and round once)
        tol = {torch.float64: 1e-12, torch.float32: 1e-5, torch.bfloat16: 1.6e-2}.get(dtype, 2e-3)
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol, equal_nan=True)


# ---------------------------------------------------------------------------
# Geometry: the vectorised CSR build equals the per-cell loop of main
# ---------------------------------------------------------------------------


def _main_positions(reducer):
    """Verbatim logic of ``HealPixNeighbourReducer._build_positions`` on main."""
    source = {int(c): p for p, c in enumerate(reducer.cell_ids.tolist())}
    domain_set = set(reducer.domain.tolist())
    if reducer.radius_m == 0:
        geo = [np.asarray([c], dtype=np.uint64) for c in reducer.domain]
    else:
        geo = build_neighbourhoods(
            reducer.domain, reducer.radius_m, reducer.refinement_level,
            neighbourhood=reducer.neighbourhood, ellipsoid=reducer.ellipsoid,
        )
    out = []
    for target, neighbours in zip(reducer.domain, geo, strict=True):
        t = int(target)
        pos = []
        for nb in neighbours:
            nb = int(nb)
            if nb not in domain_set:
                continue
            if not reducer.include_self and nb == t:
                continue
            pos.append(source[nb])
        if reducer.include_self and source[t] not in pos:
            pos.append(source[t])
        out.append(np.asarray(sorted(set(pos)), dtype=np.int64))
    return out


@pytest.mark.parametrize("name", ["full", "full_wide", "regional_domain", "no_self_cone"])
def test_positions_match_main(reducers, name):
    r = reducers[name]
    expected = _main_positions(r)
    actual = r._positions
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


# ---------------------------------------------------------------------------
# Reductions: values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["full", "full_wide", "regional_domain", "no_self_cone"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
@pytest.mark.parametrize("reduction", ALL_REDUCTIONS)
def test_reduction_matches_reference(reducers, name, dtype, reduction):
    r = reducers[name]
    rng = np.random.default_rng(hash((name, reduction)) % 2**32)
    x = _field((3, r.cell_ids.size), dtype, rng)
    if reduction == "std":
        x = torch.nan_to_num(x, posinf=3.0, neginf=-3.0)
    expected = r._reference_reduce(x, reduction)
    actual = r(x, reduction=reduction, backend="torch")
    _assert_same(actual, expected, reduction, dtype)


@pytest.mark.parametrize("reduction", ["sum", "min", "max", "mode", "mean", "median"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int16, torch.uint8, torch.bool])
def test_integer_reductions_match_reference(reducers, reduction, dtype):
    r = reducers["regional_domain"]
    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.integers(0, 4, size=(2, r.cell_ids.size))).to(dtype)
    expected = r._reference_reduce(
        x if reduction not in ("mean", "median") else x.to(torch.float64), reduction,
    )
    actual = r(x, reduction=reduction)
    _assert_same(actual, expected, reduction, dtype)


@pytest.mark.parametrize("reduction", ["any", "all"])
def test_boolean_reductions_match_reference(reducers, reduction):
    r = reducers["regional_domain"]
    x = torch.as_tensor(np.random.default_rng(4).random((2, r.cell_ids.size)) > 0.7)
    assert torch.equal(r(x, reduction=reduction), r._reference_reduce(x, reduction))


def test_numpy_and_leading_dimensions(reducers):
    r = reducers["full"]
    x = np.random.default_rng(5).normal(size=(2, 3, r.cell_ids.size))
    out = r(x, reduction="median")
    assert isinstance(out, np.ndarray) and out.shape == (2, 3, r.domain.size)
    ref = r._reference_reduce(torch.as_tensor(x.reshape(6, -1)), "median")
    np.testing.assert_array_equal(out.reshape(6, -1), ref.numpy())


def test_invalid_backend_raises(reducers):
    with pytest.raises(ValueError, match="backend"):
        reducers["full"](np.zeros(reducers["full"].cell_ids.size), backend="cuda")


# ---------------------------------------------------------------------------
# Reductions: gradients
# ---------------------------------------------------------------------------


def _grads(fn, x, gy):
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    (y * gy).sum().backward()
    return y.detach(), x.grad


@pytest.mark.parametrize("name", ["full", "regional_domain", "no_self_cone"])
@pytest.mark.parametrize("reduction", ["sum", "mean", "min", "max", "median"])
def test_gradient_matches_reference(reducers, name, reduction):
    r = reducers[name]
    rng = np.random.default_rng(7)
    # distinct values: the subgradient is unique
    x = torch.as_tensor(rng.permutation(3 * r.cell_ids.size).reshape(3, -1) * 0.01)
    x[0, rng.integers(0, r.cell_ids.size, 4)] = float("nan")
    gy = torch.as_tensor(rng.normal(size=(3, r.domain.size)))
    y_ref, g_ref = _grads(lambda t: r._reference_reduce(t, reduction), x, gy)
    y, g = _grads(lambda t: r(t, reduction=reduction, backend="torch"), x, gy)
    _assert_same(y, y_ref, reduction, torch.float64)
    torch.testing.assert_close(g, g_ref, rtol=1e-12, atol=1e-12)


def test_ties_go_to_first_cell(reducers):
    r = reducers["full"]
    x = torch.zeros(1, r.cell_ids.size, dtype=torch.float64, requires_grad=True)
    r(x, reduction="max").sum().backward()
    first = torch.as_tensor([p[0] for p in r._positions])
    expected = torch.bincount(first, minlength=r.cell_ids.size).double()
    torch.testing.assert_close(x.grad[0], expected)


@pytest.mark.parametrize("reduction", ["mean", "max", "median"])
def test_gradcheck_and_double_backward(reduction):
    rng = np.random.default_rng(8)
    level = 1
    ids = np.arange(12 * 4**level)
    r = HealPixNeighbourReducer(ids, level, radius_m=1.6 * _spacing(level))
    x = torch.as_tensor(rng.permutation(2 * ids.size).reshape(2, -1) * 0.1)
    x.requires_grad_(True)

    def f(t):
        return r(t, reduction=reduction, backend="torch")

    assert torch.autograd.gradcheck(f, (x,))
    assert torch.autograd.gradgradcheck(f, (x,))


# ---------------------------------------------------------------------------
# Least-squares gradient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gradient_ops():
    rng = np.random.default_rng(11)
    level = 4
    full = np.arange(12 * 4**level)
    regional = _regional_ids(level, rng, 0.2)
    return {
        "full": HealPixGradient(full, level),
        "regional_domain": HealPixGradient(
            regional, level, domain=rng.permutation(regional[: regional.size // 2]),
        ),
    }


def _reference_gradient(op, x):
    return _torch_gradient_from_geometry(
        x, op._geometry, op._neighbour_positions, op._center_positions,
    )


@pytest.mark.parametrize("name", ["full", "regional_domain"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_lsq_gradient_matches_reference(gradient_ops, name, dtype):
    op = gradient_ops[name]
    rng = np.random.default_rng(12)
    x = torch.as_tensor(rng.normal(size=(3, op.cell_ids.size)) * 10).to(dtype)
    x[0, rng.integers(0, op.cell_ids.size, 30)] = float("nan")
    x[1, rng.integers(0, op.cell_ids.size, 5)] = float("inf")
    gu = torch.as_tensor(rng.normal(size=(3, op.domain.size))).to(dtype)
    gw = torch.as_tensor(rng.normal(size=(3, op.domain.size))).to(dtype)

    def run(fn):
        t = x.clone().requires_grad_(True)
        e, n = fn(t)
        (torch.nan_to_num(e * gu) + torch.nan_to_num(n * gw)).sum().backward()
        return e.detach(), n.detach(), t.grad

    ref = run(lambda t: _reference_gradient(op, t))
    new = run(lambda t: sb.lsq_gradient(t, op._gradient_stencil(), backend="torch"))
    tol = dict(rtol=1e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=2e-4, atol=1e-6)
    for a, e in zip(new[:2], ref[:2]):
        assert torch.equal(torch.isnan(a), torch.isnan(e))
        torch.testing.assert_close(a, e, equal_nan=True, **tol)
    scale = ref[2].abs().max()
    torch.testing.assert_close(new[2] / scale, ref[2] / scale, **tol)


def test_lsq_function_matches_reference_1d():
    level = 5
    rng = np.random.default_rng(13)
    ids = _regional_ids(level, rng, 0.05)
    values = torch.as_tensor(rng.normal(size=ids.size))
    op = HealPixGradient(ids, level)
    ref = _reference_gradient(op, values)
    new = gradient(values, ids, level)
    for a, e in zip(new, ref):
        torch.testing.assert_close(a, e, rtol=1e-12, atol=1e-15, equal_nan=True)


def test_lsq_gradcheck_and_double_backward():
    level = 2
    op = HealPixGradient(np.arange(12 * 4**level), level)
    x = torch.as_tensor(np.random.default_rng(14).normal(size=(2, 12 * 4**level)))
    x.requires_grad_(True)
    st = op._gradient_stencil()
    assert torch.autograd.gradcheck(lambda t: sb.lsq_gradient(t, st, backend="torch"), (x,))
    assert torch.autograd.gradgradcheck(lambda t: sb.lsq_gradient(t, st, backend="torch"), (x,))


def test_lsq_numpy_batched(gradient_ops):
    op = gradient_ops["regional_domain"]
    x = np.random.default_rng(15).normal(size=(2, op.cell_ids.size))
    e, n = op(x)
    for row in range(2):
        e1, n1 = op(x[row])
        np.testing.assert_array_equal(e[row], e1)
        np.testing.assert_array_equal(n[row], n1)


# ---------------------------------------------------------------------------
# CUDA / Triton
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("name", ["full", "full_wide", "regional_domain", "no_self_cone"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["sum", "mean", "min", "max", "median"])
def test_triton_reduction_matches_reference(reducers, name, dtype, reduction):
    r = reducers[name]
    rng = np.random.default_rng(21)
    x = _field((3, r.cell_ids.size), dtype, rng)
    expected = r._reference_reduce(x, reduction)
    actual = r(x.cuda(), reduction=reduction, backend="triton").cpu()
    _assert_same(actual, expected, reduction, dtype)

    xd = torch.as_tensor(rng.permutation(3 * r.cell_ids.size).reshape(3, -1) * 0.01).to(dtype)
    gy = torch.as_tensor(rng.normal(size=(3, r.domain.size))).to(dtype)
    _, g_ref = _grads(lambda t: r._reference_reduce(t, reduction), xd, gy)
    _, g1 = _grads(lambda t: r(t, reduction=reduction, backend="triton"), xd.cuda(), gy.cuda())
    _, g2 = _grads(lambda t: r(t, reduction=reduction, backend="triton"), xd.cuda(), gy.cuda())
    assert torch.equal(g1, g2)                        # deterministic
    tol = {torch.float64: 1e-12, torch.float32: 1e-5}.get(dtype, 2e-2)
    torch.testing.assert_close(g1.cpu(), g_ref, rtol=tol, atol=tol)


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
def test_triton_double_backward_uses_torch(reducers):
    r = reducers["full"]
    x = torch.randn(2, r.cell_ids.size, device="cuda", dtype=torch.float64, requires_grad=True)
    y = r(x, reduction="mean", backend="triton")
    (g,) = torch.autograd.grad(y.square().sum(), x, create_graph=True)
    g.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("name", ["full", "regional_domain"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_lsq_gradient_matches_reference(gradient_ops, name, dtype):
    op = gradient_ops[name]
    rng = np.random.default_rng(22)
    x = torch.as_tensor(rng.normal(size=(3, op.cell_ids.size)) * 10).to(dtype)
    x[0, rng.integers(0, op.cell_ids.size, 30)] = float("nan")
    gu = torch.as_tensor(rng.normal(size=(3, op.domain.size))).to(dtype)
    gw = torch.as_tensor(rng.normal(size=(3, op.domain.size))).to(dtype)

    def run(fn, t, a, b):
        t = t.clone().requires_grad_(True)
        e, n = fn(t)
        (torch.nan_to_num(e * a) + torch.nan_to_num(n * b)).sum().backward()
        return e.detach().cpu(), n.detach().cpu(), t.grad.cpu()

    ref = run(lambda t: _reference_gradient(op, t), x, gu, gw)
    new = run(lambda t: op(t, backend="triton"), x.cuda(), gu.cuda(), gw.cuda())
    again = run(lambda t: op(t, backend="triton"), x.cuda(), gu.cuda(), gw.cuda())
    assert torch.equal(new[2], again[2])
    tol = dict(rtol=1e-12, atol=1e-15) if dtype == torch.float64 else dict(rtol=2e-4, atol=1e-6)
    for a, e in zip(new[:2], ref[:2]):
        assert torch.equal(torch.isnan(a), torch.isnan(e))
        s = torch.nan_to_num(e).abs().max()
        torch.testing.assert_close(a / s, e / s, equal_nan=True, **tol)
    scale = ref[2].abs().max()
    torch.testing.assert_close(new[2] / scale, ref[2] / scale, **tol)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_torch_backend_on_cuda(reducers):
    r = reducers["regional_domain"]
    x = torch.randn(2, r.cell_ids.size, dtype=torch.float64)
    for reduction in ("mean", "median", "max", "std", "count"):
        a = r(x.cuda(), reduction=reduction, backend="torch").cpu()
        _assert_same(a, r._reference_reduce(x, reduction), reduction, torch.float64)


# ---------------------------------------------------------------------------
# HealPixDivCurl: fused embedding / contraction
# ---------------------------------------------------------------------------


def _divcurl_main(m, velocity):
    """Verbatim forward of ``HealPixDivCurl`` on main (PyTorch ops)."""
    tensor, leading_shape, _ = m._prepare_velocity(velocity)
    u = tensor[:, 0, :]
    v = tensor[:, 1, :]
    cartesian = u[:, None, :] * m._east[None] + v[:, None, :] * m._north[None]
    d = m.conv(cartesian).reshape(tensor.shape[0], m.n_gauges, 6, m.n_pixels)
    da, db = d[:, :, :3, :], d[:, :, 3:, :]
    a, b = m._axis_a[None], m._axis_b[None]
    div = (a * da).sum(dim=2) + (b * db).sum(dim=2)
    curl = (b * da).sum(dim=2) - (a * db).sum(dim=2)
    out = torch.stack((div, curl), dim=2)
    if m.gauge_reduce == "mean":
        return out.mean(dim=1).reshape(*leading_shape, 2, m.n_pixels)
    return out.reshape(*leading_shape, m.n_gauges, 2, m.n_pixels)


@pytest.mark.parametrize("gauge_reduce", ["mean", "none"])
def test_divcurl_matches_main_cpu(gauge_reduce):
    from healpix_analyse import HealPixDivCurl
    m = HealPixDivCurl(3, n_gauges=2, gauge_reduce=gauge_reduce, dtype=torch.float64,
                       device="cpu", cache_dir=None)
    vel = torch.randn(2, 2, 12 * 4**3, dtype=torch.float64)
    torch.testing.assert_close(m(vel), _divcurl_main(m, vel), rtol=1e-12, atol=1e-15)


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("gauge_reduce", ["mean", "none"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_triton_divcurl_matches_main(gauge_reduce, dtype):
    from healpix_analyse import HealPixDivCurl
    ids = _regional_ids(4, np.random.default_rng(31), 0.2)
    m = HealPixDivCurl(4, ids, n_gauges=2, gauge_reduce=gauge_reduce, dtype=dtype,
                       device="cuda", cache_dir=None)
    vel = torch.randn(3, 2, ids.size, dtype=dtype, device="cuda")

    def run(fn):
        x = vel.clone().requires_grad_(True)
        y = fn(x)
        g = torch.cos(torch.arange(y.numel(), device="cuda", dtype=dtype)).reshape(y.shape)
        (y * g).sum().backward()
        return y.detach(), x.grad

    y_ref, g_ref = run(lambda x: _divcurl_main(m, x))
    y_new, g_new = run(m)
    tol = dict(rtol=1e-12, atol=1e-14) if dtype == torch.float64 else dict(rtol=1e-5, atol=1e-6)
    s = y_ref.abs().max()
    torch.testing.assert_close(y_new / s, y_ref / s, **tol)
    s = g_ref.abs().max()
    torch.testing.assert_close(g_new / s, g_ref / s, **tol)


@pytest.mark.skipif(not TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_lsq_gradient_16bit(gradient_ops, dtype):
    op = gradient_ops["full"]
    x64 = torch.as_tensor(np.random.default_rng(23).normal(size=(2, op.cell_ids.size)))
    ref = _reference_gradient(op, x64)
    x = x64.to(dtype).cuda().requires_grad_(True)
    e, n = op(x)
    assert e.dtype == dtype
    (e.float().sum() + n.float().sum()).backward()
    assert torch.isfinite(x.grad.float()).all()
    for a, r in zip((e, n), ref):
        torch.testing.assert_close(a.double().cpu(), r, rtol=3e-2, atol=3e-2 * r.abs().max().item())
