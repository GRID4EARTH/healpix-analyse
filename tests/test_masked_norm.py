"""Tests of the masked normalisation and pooling layers (issue #109, section 5).

The Triton tests run on CUDA, or on the CPU under the Triton interpreter
(``TRITON_INTERPRET=1``), where they call the autograd Functions directly.
"""

import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from healpix_analyse import (
    HealPixBatchNorm,
    HealPixGlobalPool,
    HealPixGroupNorm,
    HealPixLayerNorm,
    HealPixRegionPool,
    global_pool,
    region_pool,
)
from healpix_analyse import masked_norm as mn

HAS_CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
TRITON_DEVICE = "cuda" if (HAS_CUDA and mn._HAS_TRITON) else ("cpu" if (INTERPRET and mn._HAS_TRITON) else None)
needs_triton = pytest.mark.skipif(TRITON_DEVICE is None,
                                  reason="needs Triton on CUDA or TRITON_INTERPRET=1")


def _rand(B, C, N, dtype=torch.float64, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, C, N, generator=g, dtype=torch.float64) * 2 + 0.5).to(device, dtype)


def _randomise_affine(layer, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64).to(p.dtype))


# ---------------------------------------------------------------------------
# All-valid: identical to torch.nn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
@pytest.mark.parametrize("mask_kind", ["none", "ones_N", "ones_BCN", "bool"])
def test_group_norm_matches_torch(dtype, tol, mask_kind):
    B, C, N, G = 3, 8, 50, 4
    x = _rand(B, C, N, dtype).requires_grad_()
    ref = nn.GroupNorm(G, C, dtype=dtype)
    _randomise_affine(ref)
    hp = HealPixGroupNorm(G, C, dtype=dtype, backend="torch")
    hp.load_state_dict(ref.state_dict())
    mask = {"none": None, "ones_N": torch.ones(N), "ones_BCN": torch.ones(B, C, N),
            "bool": torch.ones(B, N, dtype=torch.bool)}[mask_kind]
    y0 = ref(x)
    y1 = hp(x, mask)
    torch.testing.assert_close(y1, y0, atol=tol, rtol=tol)
    g = torch.randn_like(y0)
    gx0, gw0, gb0 = torch.autograd.grad(y0, (x, ref.weight, ref.bias), g)
    gx1, gw1, gb1 = torch.autograd.grad(y1, (x, hp.weight, hp.bias), g)
    torch.testing.assert_close(gx1, gx0, atol=tol, rtol=tol)
    torch.testing.assert_close(gw1, gw0, atol=tol * 10, rtol=tol)
    torch.testing.assert_close(gb1, gb0, atol=tol * 10, rtol=tol)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_layer_norm_matches_torch(dtype, tol):
    B, C, N = 2, 6, 40
    x = _rand(B, C, N, dtype).requires_grad_()
    ref = nn.LayerNorm(C, dtype=dtype)
    _randomise_affine(ref)
    hp = HealPixLayerNorm(C, dtype=dtype, backend="torch")
    hp.load_state_dict(ref.state_dict())
    y0 = ref(x.transpose(1, 2)).transpose(1, 2)
    y1 = hp(x, torch.ones(N))
    torch.testing.assert_close(y1, y0, atol=tol, rtol=tol)
    g = torch.randn_like(y0)
    gr = torch.autograd.grad(y0, (x, ref.weight, ref.bias), g)
    gh = torch.autograd.grad(y1, (x, hp.weight, hp.bias), g)
    for a, b in zip(gh, gr):
        torch.testing.assert_close(a, b, atol=tol * 10, rtol=tol)


@pytest.mark.parametrize("momentum", [0.1, None])
def test_batch_norm_matches_torch(momentum):
    B, C, N = 4, 5, 30
    dtype, tol = torch.float64, 1e-12
    ref = nn.BatchNorm1d(C, momentum=momentum, dtype=dtype)
    _randomise_affine(ref)
    hp = HealPixBatchNorm(C, momentum=momentum, dtype=dtype, backend="torch")
    hp.load_state_dict(ref.state_dict())
    for step in range(3):
        x = _rand(B, C, N, dtype, seed=step).requires_grad_()
        y0 = ref(x)
        y1 = hp(x, torch.ones(B, N))
        torch.testing.assert_close(y1, y0, atol=tol, rtol=tol)
        g = torch.randn_like(y0)
        gr = torch.autograd.grad(y0, (x, ref.weight, ref.bias), g)
        gh = torch.autograd.grad(y1, (x, hp.weight, hp.bias), g)
        for a, b in zip(gh, gr):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=tol)
    torch.testing.assert_close(hp.running_mean, ref.running_mean, atol=tol, rtol=tol)
    torch.testing.assert_close(hp.running_var, ref.running_var, atol=tol, rtol=tol)
    assert int(hp.num_batches_tracked) == int(ref.num_batches_tracked) == 3
    ref.eval()
    hp.eval()
    x = _rand(B, C, N, dtype, seed=9)
    torch.testing.assert_close(hp(x), ref(x), atol=tol, rtol=tol)


def test_batch_norm_without_running_stats_matches_torch():
    ref = nn.BatchNorm1d(3, track_running_stats=False, dtype=torch.float64)
    hp = HealPixBatchNorm(3, track_running_stats=False, dtype=torch.float64)
    hp.eval()
    ref.eval()
    x = _rand(2, 3, 20)
    torch.testing.assert_close(hp(x), ref(x), atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Masked statistics
# ---------------------------------------------------------------------------

def _explicit_masked_gn(x, valid, G, weight, bias, eps):
    """Loop over (sample, group): statistics of x[valid] only."""
    B, C, N = x.shape
    y = torch.zeros_like(x)
    Cg = C // G
    for b in range(B):
        for g in range(G):
            sl = slice(g * Cg, (g + 1) * Cg)
            xs, vs = x[b, sl], valid[b, sl]
            v = xs[vs]
            mu = v.mean()
            var = v.var(unbiased=False)
            yy = (xs - mu) / torch.sqrt(var + eps)
            yy = yy * weight[sl, None] + bias[sl, None]
            y[b, sl] = torch.where(vs, yy, torch.zeros_like(yy))
    return y


@pytest.mark.parametrize("mask_shape", ["N", "BN", "BCN", "nan"])
def test_group_norm_masked_matches_explicit(mask_shape):
    torch.manual_seed(0)
    B, C, N, G = 2, 6, 64, 3
    x = _rand(B, C, N)
    if mask_shape == "N":
        m = torch.rand(N) > 0.3
        valid = m.expand(B, C, N)
    elif mask_shape == "BN":
        m = torch.rand(B, N) > 0.3
        valid = m[:, None].expand(B, C, N)
    elif mask_shape == "BCN":
        m = torch.rand(B, C, N) > 0.3
        valid = m
    else:
        valid = torch.rand(B, C, N) > 0.3
        m = None
    x_in = torch.where(valid, x, torch.full_like(x, 1e30 if m is not None else float("nan")))
    hp = HealPixGroupNorm(G, C, dtype=torch.float64, backend="torch")
    _randomise_affine(hp)
    y = hp(x_in, m)
    ref = _explicit_masked_gn(x, valid, G, hp.weight.detach(), hp.bias.detach(), hp.eps)
    torch.testing.assert_close(y, ref, atol=1e-12, rtol=1e-12)


def test_masked_cells_never_leak():
    """NaN / inf at masked cells change neither the output nor the grads."""
    B, C, N = 2, 4, 30
    x = _rand(B, C, N).requires_grad_()
    m = torch.ones(N)
    m[::3] = 0
    for layer in (HealPixGroupNorm(2, C, dtype=torch.float64),
                  HealPixLayerNorm(C, dtype=torch.float64),
                  HealPixBatchNorm(C, dtype=torch.float64)):
        x2 = x.detach().clone()
        x2[..., ::3] = float("nan")
        x2[0, 0, 3] = float("inf")
        x2.requires_grad_()
        y1 = layer(x, m)
        y2 = layer(x2, m)
        assert torch.isfinite(y2).all()
        torch.testing.assert_close(y1, y2, atol=0, rtol=0)
        (gx2,) = torch.autograd.grad(y2.square().sum(), x2)
        assert torch.isfinite(gx2).all()
        assert (gx2[..., ::3] == 0).all()


def test_coverage_weights_are_relative():
    """Scaling every weight by the same factor changes nothing; a fractional
    weight equals a duplicated cell in the statistics."""
    B, C, N = 1, 2, 10
    x = _rand(B, C, N)
    hp = HealPixGroupNorm(1, C, dtype=torch.float64, affine=False)
    torch.testing.assert_close(hp(x, torch.full((N,), 0.5)), hp(x), atol=1e-12, rtol=0)
    w = torch.full((N,), 0.5)
    w[0] = 1.0                      # cell 0 counts twice
    xd = torch.cat([x, x[..., :1]], dim=-1)
    yd = hp(xd)
    torch.testing.assert_close(hp(x, w), yd[..., :N], atol=1e-12, rtol=0)


@pytest.mark.parametrize("fill", ["zero", "nan", "passthrough"])
def test_fill_modes(fill):
    B, C, N = 2, 4, 20
    x = _rand(B, C, N)
    x[..., 0] = float("nan")
    x[..., 1] = 7.0
    m = torch.ones(N)
    m[1] = 0
    y = HealPixGroupNorm(2, C, dtype=torch.float64, fill=fill)(x, m)
    if fill == "zero":
        assert (y[..., :2] == 0).all()
    elif fill == "nan":
        assert torch.isnan(y[..., :2]).all()
    else:
        assert torch.isnan(y[..., 0]).all() and (y[..., 1] == 7.0).all()
    assert torch.isfinite(y[..., 2:]).all()


def test_empty_group_is_filled_without_nan():
    x = _rand(1, 4, 10)
    m = torch.ones(1, 4, 10)
    m[0, :2] = 0                    # group 0 fully masked
    y = HealPixGroupNorm(2, 4, dtype=torch.float64)(x, m)
    assert (y[0, :2] == 0).all() and torch.isfinite(y).all()


def test_batch_norm_masked_running_stats():
    B, C, N = 3, 2, 40
    torch.manual_seed(3)
    x = _rand(B, C, N)
    m = torch.rand(B, N) > 0.4
    hp = HealPixBatchNorm(C, momentum=1.0, dtype=torch.float64)
    hp(x, m)
    for c in range(C):
        v = x[:, c][m]
        torch.testing.assert_close(hp.running_mean[c], v.mean(), atol=1e-12, rtol=0)
        torch.testing.assert_close(hp.running_var[c], v.var(unbiased=True), atol=1e-12, rtol=0)
    # A channel without valid cells keeps its running statistics.
    hp2 = HealPixBatchNorm(C, dtype=torch.float64)
    mm = torch.ones(1, C, N)
    mm[0, 1] = 0
    hp2(x, mm)
    assert hp2.running_mean[1] == 0 and hp2.running_var[1] == 1


def test_layer_norm_per_channel_mask():
    B, C, N = 2, 5, 12
    x = _rand(B, C, N)
    m = torch.rand(B, C, N) > 0.3
    y = HealPixLayerNorm(C, dtype=torch.float64, elementwise_affine=False)(x, m)
    for b in range(B):
        for n in range(N):
            v = m[b, :, n]
            if v.any():
                xs = x[b, v, n]
                ref = (xs - xs.mean()) / torch.sqrt(xs.var(unbiased=False) + 1e-5)
                torch.testing.assert_close(y[b, v, n], ref, atol=1e-12, rtol=0)
            assert (y[b, ~v, n] == 0).all()


def test_input_checks():
    with pytest.raises(ValueError):
        HealPixGroupNorm(2, 4, fill="bad")
    with pytest.raises(ValueError):
        HealPixGroupNorm(2, 4, backend="cuda")
    layer = HealPixGroupNorm(2, 4)
    with pytest.raises(ValueError):
        layer(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        layer(torch.zeros(2, 4, 10), torch.ones(4, 10))      # [C, N] is ambiguous
    with pytest.raises(RuntimeError):
        HealPixGroupNorm(2, 4, backend="triton")(torch.zeros(1, 4, 10, dtype=torch.float64))


def test_16bit_input_reduced_in_float32():
    x = _rand(2, 4, 100, torch.float32)
    hp = HealPixGroupNorm(2, 4)
    y16 = hp(x.to(torch.bfloat16))
    assert y16.dtype == torch.bfloat16
    torch.testing.assert_close(y16.float(), hp(x), atol=3e-2, rtol=3e-2)


# ---------------------------------------------------------------------------
# Exact gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fill", ["zero", "passthrough"])
def test_gradcheck_reference(fill):
    torch.manual_seed(0)
    B, C, N = 2, 4, 9
    x = _rand(B, C, N).requires_grad_()
    cov = torch.rand(B, 1, N, dtype=torch.float64)
    cov[..., ::4] = 0
    for layer in (HealPixGroupNorm(2, C, dtype=torch.float64, fill=fill),
                  HealPixLayerNorm(C, dtype=torch.float64, fill=fill),
                  HealPixBatchNorm(C, dtype=torch.float64, fill=fill,
                                   track_running_stats=False)):
        _randomise_affine(layer)
        params = tuple(layer.parameters())
        assert torch.autograd.gradcheck(lambda xx, *p: layer(xx, cov), (x, *params))
        assert torch.autograd.gradgradcheck(lambda xx, *p: layer(xx, cov), (x, *params))


def test_gradcheck_pooling():
    torch.manual_seed(1)
    x = _rand(2, 3, 12).requires_grad_()
    cov = torch.rand(12, dtype=torch.float64)
    cov[0] = 0
    regions = np.array([0, 0, 1, 1, 1, 2, 2, -1, 0, 1, 2, 2])
    for mode in ("mean", "sum", "max"):
        assert torch.autograd.gradcheck(lambda xx: global_pool(xx, cov, mode), (x,))
        assert torch.autograd.gradcheck(lambda xx: region_pool(xx, regions, cov, mode), (x,))


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def test_global_pool_values():
    x = _rand(2, 3, 20)
    torch.testing.assert_close(global_pool(x), x.mean(-1), atol=1e-12, rtol=0)
    torch.testing.assert_close(global_pool(x, mode="max"), x.amax(-1), atol=0, rtol=0)
    torch.testing.assert_close(global_pool(x, mode="sum"), x.sum(-1), atol=1e-12, rtol=0)
    m = torch.zeros(20)
    m[:5] = 1
    xn = x.clone()
    xn[..., 5:] = float("nan")
    torch.testing.assert_close(global_pool(xn), x[..., :5].mean(-1), atol=1e-12, rtol=0)
    torch.testing.assert_close(global_pool(x, m, "max"), x[..., :5].amax(-1), atol=0, rtol=0)
    y, s0 = global_pool(x, torch.zeros(20), return_weight=True)
    assert (y == 0).all() and (s0 == 0).all()
    assert torch.isnan(global_pool(x, torch.zeros(20), fill="nan")).all()
    torch.testing.assert_close(HealPixGlobalPool("max")(x, m), global_pool(x, m, "max"))


def test_region_pool_values_and_module():
    rng = np.random.default_rng(0)
    N = 40
    regions = rng.integers(-1, 5, N)
    x = _rand(2, 3, N)
    m = torch.as_tensor(rng.random(N) > 0.2)
    y = region_pool(x, regions, m, "mean", n_regions=6)
    assert y.shape == (2, 3, 6)
    for r in range(5):
        sel = torch.as_tensor(regions == r) & m
        torch.testing.assert_close(y[..., r], x[..., sel].mean(-1), atol=1e-12, rtol=0)
    assert (y[..., 5] == 0).all()                    # empty region
    pool = HealPixRegionPool(regions, mode="max", n_regions=6)
    ym = pool(x, m)
    for r in range(5):
        sel = torch.as_tensor(regions == r) & m
        torch.testing.assert_close(ym[..., r], x[..., sel].amax(-1), atol=0, rtol=0)


def test_region_pool_from_cell_ids_unsorted():
    level, rl = 3, 1
    rng = np.random.default_rng(1)
    ids = rng.permutation(np.arange(4 ** 3 * 2, 4 ** 3 * 4))   # unsorted regional domain
    x = _rand(1, 2, ids.size)
    pool = HealPixRegionPool.from_cell_ids(ids, level, rl)
    np.testing.assert_array_equal(pool.region_ids, np.unique(ids >> 4))
    y = pool(x)
    for r, pid in enumerate(pool.region_ids):
        sel = torch.as_tensor((ids >> 4) == pid)
        torch.testing.assert_close(y[..., r], x[..., sel].mean(-1), atol=1e-12, rtol=0)
    # A full-sphere pool to level 0 equals a mean Down of 4**level cells per base cell.
    full = np.arange(12 * 4 ** level)
    xf = _rand(1, 1, full.size)
    y0 = HealPixRegionPool.from_cell_ids(full, level, 0)(xf)
    torch.testing.assert_close(y0, xf.view(1, 1, 12, -1).mean(-1), atol=1e-12, rtol=0)


def test_region_pool_follows_to():
    pool = HealPixRegionPool([0, 1, 1, 0])
    pool = pool.to(torch.float64)
    assert pool._idx.dtype == torch.int64


# ---------------------------------------------------------------------------
# Triton backend
# ---------------------------------------------------------------------------

def _triton_call(kind, x, m, weight, bias, G, eps, fill):
    if kind == "layer":
        return mn._LayerNormFunction.apply(x, m, weight, bias, eps, fill)
    return mn._RowsNormFunction.apply(x, m, weight, bias, kind, G, eps, fill)[0]


@needs_triton
@pytest.mark.parametrize("kind", ["group", "batch", "layer"])
@pytest.mark.parametrize("mask_kind", ["none", "N", "BN", "BCN_float", "bool"])
@pytest.mark.parametrize("fill", ["zero", "nan", "passthrough"])
def test_triton_matches_reference(kind, mask_kind, fill):
    dev = TRITON_DEVICE
    B, C, N, G = 2, 8, 3000 if dev == "cuda" else 700, 4
    torch.manual_seed(0)
    x = _rand(B, C, N, torch.float32, device=dev)
    x[0, 1, 5] = float("nan")
    mask = {"none": None,
            "N": (torch.rand(N, device=dev) > 0.2).float(),
            "BN": (torch.rand(B, N, device=dev) > 0.2).float(),
            "BCN_float": torch.rand(B, C, N, device=dev) * (torch.rand(B, C, N, device=dev) > 0.2),
            "bool": torch.rand(B, N, device=dev) > 0.2}[mask_kind]
    w = torch.randn(C, device=dev, requires_grad=True)
    b = torch.randn(C, device=dev, requires_grad=True)
    Gk = G if kind == "group" else 1
    xr = x.clone().requires_grad_()
    xt = x.clone().requires_grad_()
    y0, *_ = mn.masked_norm_reference(xr, mask, kind, Gk, w, b, 1e-5, fill)
    y1 = _triton_call(kind, xt, mn._mask_3d(mask, xt), w, b, Gk, 1e-5, fill)
    torch.testing.assert_close(y1, y0, atol=1e-4, rtol=1e-4, equal_nan=True)
    g = torch.randn_like(y0)
    g0 = torch.autograd.grad(y0, (xr, w, b), g)
    g1 = torch.autograd.grad(y1, (xt, w, b), g)
    for a, c in zip(g1, g0):
        torch.testing.assert_close(a, c, atol=2e-3, rtol=1e-3, equal_nan=True)


@needs_triton
@pytest.mark.parametrize("kind", ["group", "batch", "layer"])
def test_triton_deterministic(kind):
    dev = TRITON_DEVICE
    x = _rand(4, 8, 5000 if dev == "cuda" else 900, torch.float32, device=dev)
    m = torch.rand(x.shape[-1], device=dev) > 0.3
    w = torch.randn(8, device=dev, requires_grad=True)
    b = torch.randn(8, device=dev, requires_grad=True)
    outs = []
    for _ in range(2):
        xt = x.clone().requires_grad_()
        y = _triton_call(kind, xt, mn._mask_3d(m, xt), w, b, 2 if kind == "group" else 1,
                         1e-5, "zero")
        gs = torch.autograd.grad(y.square().sum(), (xt, w, b))
        outs.append((y, *gs))
    for a, c in zip(*outs):
        assert torch.equal(a, c)


@needs_triton
def test_triton_double_backward_falls_back_to_reference():
    dev = TRITON_DEVICE
    x = _rand(2, 4, 300, torch.float32, device=dev).requires_grad_()
    w = torch.randn(4, device=dev, requires_grad=True)
    b = torch.randn(4, device=dev, requires_grad=True)
    y = _triton_call("group", x, None, w, b, 2, 1e-5, "zero")
    (gx,) = torch.autograd.grad(y.sin().sum(), x, create_graph=True)
    (gw,) = torch.autograd.grad(gx.square().sum(), w)
    xr = x.detach().requires_grad_()
    yr, *_ = mn.masked_norm_reference(xr, None, "group", 2, w, b, 1e-5, "zero")
    (gxr,) = torch.autograd.grad(yr.sin().sum(), xr, create_graph=True)
    (gwr,) = torch.autograd.grad(gxr.square().sum(), w)
    torch.testing.assert_close(gw, gwr, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not (HAS_CUDA and mn._HAS_TRITON), reason="needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_16bit_layers(dtype):
    dev = "cuda"
    x = _rand(2, 8, 4000, torch.float32, device=dev)
    m = torch.rand(4000, device=dev) > 0.2
    for cls, args in ((HealPixGroupNorm, (4, 8)), (HealPixLayerNorm, (8,)),
                      (HealPixBatchNorm, (8,))):
        a = cls(*args, device=dev)
        r = cls(*args, device=dev, backend="torch")
        y = a(x.to(dtype), m)
        assert y.dtype == dtype
        torch.testing.assert_close(y.float(), r(x.to(dtype), m).float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not (HAS_CUDA and mn._HAS_TRITON), reason="needs CUDA and Triton")
def test_layers_on_cuda_auto_use_triton_and_match():
    dev = "cuda"
    x = _rand(3, 8, 5000, torch.float32, device=dev)
    m = torch.rand(3, 5000, device=dev) > 0.3
    for cls, args in ((HealPixGroupNorm, (4, 8)), (HealPixLayerNorm, (8,)),
                      (HealPixBatchNorm, (8,))):
        a = cls(*args, device=dev)
        r = cls(*args, device=dev, backend="torch")
        torch.testing.assert_close(a(x, m), r(x, m), atol=1e-4, rtol=1e-4)
        if cls is HealPixBatchNorm:
            torch.testing.assert_close(a.running_var, r.running_var, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["group", "batch", "layer"])
@pytest.mark.parametrize("mask_kind", ["none", "N_bool", "BN_float", "BCN_float"])
@pytest.mark.parametrize("fill", ["zero", "nan", "passthrough"])
@pytest.mark.parametrize("affine", [True, False])
def test_torch_backend_matches_reference(kind, mask_kind, fill, affine):
    """The few-pass PyTorch backend (hand-written backward) equals the
    plain reference to 1e-12 in float64, outputs and gradients."""
    torch.manual_seed(0)
    B, C, N, G = 3, 6, 50, 3
    x = _rand(B, C, N)
    x[1, 2, 7] = float("nan")
    mask = {"none": None,
            "N_bool": torch.rand(N) > 0.3,
            "BN_float": torch.rand(B, N, dtype=torch.float64) * (torch.rand(B, N) > 0.2),
            "BCN_float": torch.rand(B, C, N, dtype=torch.float64)}[mask_kind]
    w = torch.randn(C, dtype=torch.float64, requires_grad=True) if affine else None
    b = torch.randn(C, dtype=torch.float64, requires_grad=True) if affine else None
    Gk = G if kind == "group" else 1
    xr = x.clone().requires_grad_()
    xt = x.clone().requires_grad_()
    y0, mu0, var0, _ = mn.masked_norm_reference(xr, mask, kind, Gk, w, b, 1e-5, fill)
    y1, mu1, var1, _ = mn._TorchNormFunction.apply(xt, mn._mask_3d(mask, xt), w, b, kind,
                                                   Gk, 1e-5, fill)
    torch.testing.assert_close(y1, y0, atol=1e-12, rtol=1e-12, equal_nan=True)
    torch.testing.assert_close(var1.reshape(-1), var0.reshape(-1), atol=1e-12, rtol=1e-12)
    g = torch.randn_like(y0)
    ins0 = (xr,) + ((w, b) if affine else ())
    ins1 = (xt,) + ((w, b) if affine else ())
    for a, c in zip(torch.autograd.grad(y1, ins1, g), torch.autograd.grad(y0, ins0, g)):
        torch.testing.assert_close(a, c, atol=1e-11, rtol=1e-11, equal_nan=True)


@needs_triton
@pytest.mark.parametrize("C", [8, 600])
def test_triton_layer_norm_tile_and_loop_kernels(C):
    """C <= 512 runs the one-tile kernels, wider C the channel loop."""
    dev = TRITON_DEVICE
    N = 300
    x = _rand(2, C, N, torch.float32, device=dev)
    m = torch.rand(2, C, N, device=dev) > 0.2
    w = torch.randn(C, device=dev, requires_grad=True)
    b = torch.randn(C, device=dev, requires_grad=True)
    xr, xt = x.clone().requires_grad_(), x.clone().requires_grad_()
    y0, *_ = mn.masked_norm_reference(xr, m, "layer", 1, w, b, 1e-5, "zero")
    y1 = mn._LayerNormFunction.apply(xt, mn._mask_3d(m, xt), w, b, 1e-5, "zero")
    torch.testing.assert_close(y1, y0, atol=1e-4, rtol=1e-4)
    g = torch.randn_like(y0)
    for a, c in zip(torch.autograd.grad(y1, (xt, w, b), g), torch.autograd.grad(y0, (xr, w, b), g)):
        torch.testing.assert_close(a, c, atol=2e-3, rtol=1e-3)


@needs_triton
@pytest.mark.parametrize("fill", ["zero", "passthrough"])
def test_triton_batch_norm_eval(fill):
    dev = TRITON_DEVICE
    C = 6
    ref = HealPixBatchNorm(C, fill=fill, backend="torch").to(dev)
    tri = HealPixBatchNorm(C, fill=fill, backend="triton").to(dev)
    with torch.no_grad():
        for layer in (ref, tri):
            layer.running_mean.copy_(torch.linspace(-1, 1, C))
            layer.running_var.copy_(torch.linspace(0.5, 2, C))
            layer.weight.copy_(torch.linspace(1, 2, C))
            layer.bias.copy_(torch.linspace(-0.5, 0.5, C))
    ref.eval()
    tri.eval()
    x = _rand(2, C, 500, torch.float32, device=dev)
    m = torch.rand(500, device=dev) > 0.3
    xr, xt = x.clone().requires_grad_(), x.clone().requires_grad_()
    y0, y1 = ref(xr, m), tri(xt, m)
    torch.testing.assert_close(y1, y0, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(y0)
    a = torch.autograd.grad(y1, (xt, tri.weight, tri.bias), g)
    c = torch.autograd.grad(y0, (xr, ref.weight, ref.bias), g)
    for u, v in zip(a, c):
        torch.testing.assert_close(u, v, atol=1e-3, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("mode", ["mean", "sum", "max"])
@pytest.mark.parametrize("mask_kind", ["none", "N", "BCN_float"])
def test_triton_pooling_matches_torch(mode, mask_kind):
    dev = TRITON_DEVICE
    torch.manual_seed(0)
    B, C, N = 2, 3, 2000
    x = _rand(B, C, N, torch.float32, device=dev)
    x[0, 0, 3] = float("nan")
    mask = {"none": None, "N": torch.rand(N, device=dev) > 0.3,
            "BCN_float": torch.rand(B, C, N, device=dev)}[mask_kind]
    regions = torch.randint(-1, 7, (N,))
    regions[:5] = 8                      # region 8, region 7 empty
    for fn in (lambda xx, be: global_pool(xx, mask, mode, return_weight=True, backend=be),
               lambda xx, be: region_pool(xx, regions, mask, mode, n_regions=9,
                                          return_weight=True, backend=be)):
        xr, xt = x.clone().requires_grad_(), x.clone().requires_grad_()
        y0, s0 = fn(xr, "torch")
        y1, s1 = fn(xt, "triton")
        torch.testing.assert_close(s1.float(), s0.float(), atol=1e-3, rtol=1e-5)
        torch.testing.assert_close(y1, y0, atol=1e-4, rtol=1e-4)
        g = torch.randn_like(y0)
        (a,) = torch.autograd.grad(y1, xt, g)
        (c,) = torch.autograd.grad(y0, xr, g)
        torch.testing.assert_close(a, c, atol=1e-5, rtol=1e-4)


@needs_triton
def test_triton_region_pool_module_and_determinism():
    dev = TRITON_DEVICE
    level = 4
    rng = np.random.default_rng(0)
    ids = rng.permutation(np.arange(256 * 3, 256 * 5))
    pool_t = HealPixRegionPool.from_cell_ids(ids, level, 2, backend="triton").to(dev)
    pool_r = HealPixRegionPool.from_cell_ids(ids, level, 2, backend="torch").to(dev)
    x = _rand(3, 4, ids.size, torch.float32, device=dev)
    m = torch.rand(3, ids.size, device=dev) > 0.2
    torch.testing.assert_close(pool_t(x, m), pool_r(x, m), atol=1e-5, rtol=1e-5)
    outs = []
    for _ in range(2):
        xt = x.clone().requires_grad_()
        y = pool_t(xt, m)
        (gx,) = torch.autograd.grad(y.square().sum(), xt)
        outs.append((y, gx))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@pytest.mark.skipif(not (HAS_CUDA and mn._HAS_TRITON), reason="needs CUDA and Triton")
def test_triton_large_grids():
    """More than 65 535 cell blocks per sample: the grids are 1-D."""
    dev = "cuda"
    N = 2_100_000
    x = torch.randn(1, 8, N, device=dev)
    m = torch.rand(N, device=dev) > 0.3
    for cls, args in ((HealPixLayerNorm, (8,)), (HealPixGroupNorm, (2, 8))):
        a = cls(*args, device=dev)
        r = cls(*args, device=dev, backend="torch")
        xa, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yr = a(xa, m), r(xr, m)
        torch.testing.assert_close(ya, yr, atol=1e-4, rtol=1e-4)
        (ga,) = torch.autograd.grad(ya.sum(), xa)
        (gr,) = torch.autograd.grad(yr.sum(), xr)
        torch.testing.assert_close(ga, gr, atol=1e-3, rtol=1e-3)


@needs_triton
@pytest.mark.parametrize("mode", ["mean", "max"])
def test_triton_region_pool_contiguous_regions(mode):
    """Sorted equal regions (full sphere): the contiguous-slice kernel."""
    dev = TRITON_DEVICE
    level = 4
    ids = np.arange(12 * 4 ** level)
    pool_t = HealPixRegionPool.from_cell_ids(ids, level, 2, mode=mode, backend="triton").to(dev)
    pool_r = HealPixRegionPool.from_cell_ids(ids, level, 2, mode=mode, backend="torch").to(dev)
    assert pool_t._ident
    x = _rand(2, 3, ids.size, torch.float32, device=dev)
    m = torch.rand(2, ids.size, device=dev) > 0.3
    xt, xr = x.clone().requires_grad_(), x.clone().requires_grad_()
    yt, yr = pool_t(xt, m), pool_r(xr, m)
    torch.testing.assert_close(yt, yr, atol=1e-5, rtol=1e-5)
    g = torch.randn_like(yt)
    (a,) = torch.autograd.grad(yt, xt, g)
    (c,) = torch.autograd.grad(yr, xr, g)
    torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)


def test_batch_norm_momentum_none_skips_empty_batches():
    """A batch without valid cells does not count in the cumulative average,
    per channel; the per-channel count is rebuilt from a loaded state_dict."""
    torch.manual_seed(0)
    C, N = 2, 50
    bn = HealPixBatchNorm(C, momentum=None, dtype=torch.float64)
    x1 = _rand(2, C, N, seed=1) + 5.0
    bn(x1, torch.zeros(N, dtype=torch.bool))          # nothing valid
    assert int(bn.num_batches_tracked) == 0
    bn(x1)
    torch.testing.assert_close(bn.running_mean, x1.mean((0, 2)), atol=1e-12, rtol=0)
    assert int(bn.num_batches_tracked) == 1
    # channel 1 masked in the next batch: channel 0 averages 2 batches, channel 1 one
    x2 = _rand(2, C, N, seed=2) - 3.0
    m = torch.ones(1, C, N)
    m[0, 1] = 0
    bn(x2, m)
    assert bn.num_batches_tracked_per_channel.tolist() == [2, 1]
    torch.testing.assert_close(bn.running_mean[0], (x1[:, 0].mean() + x2[:, 0].mean()) / 2,
                               atol=1e-12, rtol=0)
    torch.testing.assert_close(bn.running_mean[1], x1[:, 1].mean(), atol=1e-12, rtol=0)
    bn2 = HealPixBatchNorm(C, momentum=None, dtype=torch.float64)
    bn2.load_state_dict(bn.state_dict())
    assert bn2.num_batches_tracked_per_channel.tolist() == [2, 2]
    assert set(bn.state_dict()) == set(nn.BatchNorm1d(C).state_dict())


_XPROC_SCRIPT = r"""
import hashlib, sys, numpy as np, torch
from healpix_analyse import (HealPixGroupNorm, HealPixLayerNorm, HealPixBatchNorm,
                             HealPixRegionPool, global_pool)
torch.manual_seed(0)
dev = "cuda"
B, C, N = 3, 16, 300_000
x = torch.randn(B, C, N, device=dev)
m = torch.rand(B, N, device=dev) > 0.3
ids = np.random.default_rng(0).permutation(np.arange(N))
h = hashlib.sha256()
layers = [HealPixGroupNorm(4, C), HealPixLayerNorm(C), HealPixBatchNorm(C)]
fns = [lambda xx, l=l.to(dev): l(xx, m) for l in layers]
fns.append(lambda xx: global_pool(xx, m))
pool_u = HealPixRegionPool(ids // 16).to(dev)
pool_s = HealPixRegionPool(np.arange(N) // 16).to(dev)
fns.append(lambda xx: pool_u(xx, m))
fns.append(lambda xx: pool_s(xx, m))
for f in fns:
    for dt in (torch.float32, torch.bfloat16):
        xx = x.to(dt).clone().requires_grad_()
        y = f(xx)
        (g,) = torch.autograd.grad((y.float() * 1.7).sin().sum(), xx)
        h.update(y.detach().float().cpu().numpy().tobytes()); h.update(g.float().cpu().numpy().tobytes())
print(h.hexdigest())
"""


@pytest.mark.skipif(not (HAS_CUDA and mn._HAS_TRITON), reason="needs CUDA and Triton")
def test_triton_bitwise_reproducible_across_processes(tmp_path):
    """Two fresh processes (separate Triton caches, so the element-wise
    kernels re-tune) give bit-identical outputs and gradients."""
    import subprocess
    import sys

    script = tmp_path / "run.py"
    script.write_text(_XPROC_SCRIPT)
    digests = []
    for k in range(2):
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"cache{k}"))
        out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        digests.append(out.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1]
