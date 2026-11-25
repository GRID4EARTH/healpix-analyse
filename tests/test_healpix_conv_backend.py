"""HealPixConv backends against the implementation of main before issue #90.

The reference below is a verbatim copy of the geometry build and of the
gather + bmm forward of ``HealPixConv`` on main (0326f2f), with autograd for
the gradients.  It materialises the gathered ``[B, C, G, K, P]`` tensor, so it
is only used on small cases.

* PyTorch backend: float64, outputs and gradients (x, weight, bias) to 1e-12.
* Triton backend (CUDA): float32 to 1e-5; skipped without CUDA or Triton.
"""

from __future__ import annotations

import math

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_analyse import _conv_backend as cb
from healpix_analyse.convol import (
    HealPixConv,
    _bind_support_batched,
    _build_rotation_matrices,
    _geometry_cache_key,
    _get_interp_weights,
    _local_kernel_grid,
)

HAS_CUDA = torch.cuda.is_available()
needs_triton = pytest.mark.skipif(
    not cb.triton_available(), reason="needs CUDA and the optional triton package"
)
needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")


# ---------------------------------------------------------------------------
# Reference: main before #90
# ---------------------------------------------------------------------------

def reference_geometry(conv, cell_ids, dtype, device, return_support=False):
    """pos_safe, w_norm [4, G, K*P] and sort_order [K] (and support [G, K*P]), as on main."""
    level, G, P, K = conv.level, conv.G, conv.P, conv.K
    ids_np = (np.arange(12 * 4**level, dtype=np.int64) if cell_ids is None
              else np.asarray(cell_ids, dtype=np.int64).ravel())
    lon, lat = healpix_geo.nested.healpix_to_lonlat(
        ids_np.tolist(), level, ellipsoid=conv.ellipsoid
    )
    th = np.deg2rad(90.0 - np.asarray(lat, dtype=np.float64))
    ph = np.deg2rad(np.asarray(lon, dtype=np.float64))
    R_tot = _build_rotation_matrices(th, ph, G, conv.gauge_type, device, dtype,
                                     ref_direction=conv.ref_direction)
    vec_t = torch.as_tensor(_local_kernel_grid(conv.kernel_sz, conv.nside),
                            device=device, dtype=dtype)
    flat = torch.einsum("kgij,pj->kgpi", R_tot, vec_t).reshape(-1, 3)
    idx_flat, w_flat = _get_interp_weights(conv.nside, flat, True, device, dtype)
    idx_all = idx_flat.view(4, K, G, P).permute(2, 0, 1, 3).reshape(G, 4, K * P)
    w_all = w_flat.view(4, K, G, P).permute(2, 0, 1, 3).reshape(G, 4, K * P)
    ids_sorted_t = torch.as_tensor(np.sort(ids_np), device=device, dtype=torch.long)
    sort_order = torch.as_tensor(np.argsort(ids_np), device=device, dtype=torch.long)
    pos, w, support = _bind_support_batched(idx_all.clone(), w_all.clone(), ids_sorted_t,
                                            conv.kernel_sz, K, P, device, return_support=True)
    geom = (pos.permute(1, 0, 2).contiguous(), w.permute(1, 0, 2).contiguous(), sort_order)
    return geom + (support,) if return_support else geom


def reference_forward(conv, geom, t):
    """HealPixConv.forward of main: gather [B, C, G, K, P] + bmm (autograd)."""
    pos, wn, so = geom
    G, P, K = conv.G, conv.P, conv.K
    B, C_in, _ = t.shape
    C_out = conv.out_channels
    t_sorted = t[:, :, so]
    gathered = t.new_zeros(B, C_in, G, K, P)
    for j in range(4):
        vf = t_sorted.index_select(2, pos[j].reshape(-1))
        gathered = gathered + vf.view(B, C_in, G, K, P) * wn[j].to(t.dtype).view(1, 1, G, K, P)
    g_mat = gathered.permute(2, 0, 3, 1, 4).contiguous().view(G, B * K, C_in * P)
    W_mat = conv.weight.permute(0, 1, 3, 2).contiguous().view(G, C_in * P, C_out)
    y = torch.bmm(g_mat, W_mat).view(G, B, K, C_out).permute(1, 0, 3, 2).reshape(B, G * C_out, K)
    return y + conv.bias.view(1, -1, 1)


def reference_local_mean_forward(conv, geom, support, t, mask=None):
    """HealPixConv.forward of main (3510979) with missing="local_mean" (autograd)."""
    pos, wn, so = geom
    B, C, N = t.shape
    G, P, K = conv.G, conv.P, conv.K
    wn = wn.to(t.dtype)
    sup = support.to(t.dtype).view(1, 1, G, K, P)
    v = torch.isfinite(t).to(t.dtype)
    if mask is not None:
        v = v * torch.as_tensor(mask, dtype=t.dtype).expand(B, C, N)
    xz = torch.where(v > 0, t, torch.zeros_like(t))
    vx = (xz * v)[:, :, so]
    vs = v[:, :, so]
    ivx = t.new_zeros(B, C, G, K, P)
    iv = t.new_zeros(B, C, G, K, P)
    for j in range(4):
        pj = pos[j].reshape(-1)
        wj = wn[j].view(1, 1, G, K, P)
        ivx = ivx + vx.index_select(2, pj).view(B, C, G, K, P) * wj
        iv = iv + vs.index_select(2, pj).view(B, C, G, K, P) * wj
    cov = iv * sup
    a = conv._local_mean_w.to(t.dtype).view(1, 1, 1, 1, P)
    num = (a * ivx * sup).sum(-1, keepdim=True)
    den = (a * cov).sum(-1, keepdim=True)
    has_den = den > 1e-6
    xbar = torch.where(has_den, num / torch.where(has_den, den, torch.ones_like(den)),
                       torch.zeros_like(den))
    has = cov > 1e-6
    z = torch.where(has, ivx / torch.where(has, iv, torch.ones_like(iv)), xbar.expand_as(ivx))
    g_mat = z.permute(2, 0, 3, 1, 4).contiguous().view(G, B * K, C * P)
    W_mat = conv.weight.permute(0, 1, 3, 2).contiguous().view(G, C * P, conv.out_channels)
    y = torch.bmm(g_mat, W_mat).view(G, B, K, -1).permute(1, 0, 3, 2).reshape(B, -1, K)
    return y + conv.bias.view(1, -1, 1)


def nested_patch(level, m, face=4):
    """A square NESTED patch of 4**m cells at `level` inside base face `face`."""
    n = 4**m
    start = face * 4**level + (4**level // 2 // n) * n
    return np.arange(start, start + n, dtype=np.int64)


def unsorted_patch(level, m, seed=3):
    ids = nested_patch(level, m)
    return ids[np.random.default_rng(seed).permutation(ids.size)]


CASES = {
    # full sphere, "phi" gauge
    "full": dict(level=3, cell_ids=None, G=4, k=3, B=2, C=3, Co=2, gauge="phi"),
    # unsorted regional domain, another gauge and a 5x5 stencil
    "regional": dict(level=5, cell_ids=unsorted_patch(5, 4), G=2, k=5, B=3, C=2, Co=3,
                     gauge="projected_ref"),
}


def make_conv(cs, dtype, device, **kw):
    torch.manual_seed(0)
    conv = HealPixConv(cs["level"], cs["C"], cs["Co"], kernel_sz=cs["k"], n_gauges=cs["G"],
                       gauge_type=cs["gauge"], cell_ids=cs["cell_ids"], device=device,
                       dtype=dtype, cache_dir=None, **kw)
    with torch.no_grad():
        conv.weight.normal_()
        conv.bias.normal_()
    return conv


def run(conv, x0, fwd):
    """y, grad x, grad weight, grad bias of sum(y * sin(arange))."""
    x = x0.clone().requires_grad_(True)
    conv.zero_grad(set_to_none=True)
    y = fwd(x)
    gy = torch.sin(torch.arange(y.numel(), dtype=torch.float64, device=y.device)).view_as(y)
    (y * gy.to(y.dtype)).sum().backward()
    return (y.detach(), x.grad.detach(), conv.weight.grad.detach().clone(),
            conv.bias.grad.detach().clone())


def rel_errs(got, ref):
    return [float((a.double() - b.double()).abs().max() / b.double().abs().max())
            for a, b in zip(got, ref)]


# ---------------------------------------------------------------------------
# PyTorch backend, float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_geometry_matches_reference(case):
    cs = CASES[case]
    conv = make_conv(cs, torch.float64, "cpu")
    pos, w, so = reference_geometry(conv, cs["cell_ids"], torch.float64, torch.device("cpu"))
    assert torch.equal(conv._sort_order, so)
    assert torch.equal(conv._pos_safe, pos)          # legacy view of the tap geometry
    assert torch.equal(conv._w_norm, w)
    # folded indices: the input order, not the sorted one
    assert torch.equal(conv._tap_idx.long(), so[pos.view(4, conv.G, conv.K, conv.P)]
                       .permute(3, 0, 1, 2))
    assert conv._tap_idx.dtype == torch.int32


@pytest.mark.parametrize("case", sorted(CASES))
def test_torch_backend_matches_reference(case):
    cs = CASES[case]
    conv = make_conv(cs, torch.float64, "cpu", backend="torch")
    geom = reference_geometry(conv, cs["cell_ids"], torch.float64, torch.device("cpu"))
    x0 = torch.randn(cs["B"], cs["C"], conv.K, dtype=torch.float64)
    ref = run(conv, x0, lambda x: reference_forward(conv, geom, x))
    got = run(conv, x0, conv)
    errs = rel_errs(got, ref)
    assert max(errs) < 1e-12, errs


@pytest.mark.parametrize("with_mask", [False, True])
def test_local_mean_matches_main(with_mask):
    """missing="local_mean" keeps the reference path of main (3510979): same
    outputs and gradients (x, weight, bias), NaN and masked pixels included."""
    cs = CASES["regional"]
    conv = make_conv(cs, torch.float64, "cpu", missing="local_mean",
                     local_mean_weights=np.linspace(1.0, 2.0, cs["k"] ** 2))
    geom = reference_geometry(conv, cs["cell_ids"], torch.float64, torch.device("cpu"),
                              return_support=True)
    gen = torch.Generator().manual_seed(4)
    x0 = torch.randn(cs["B"], cs["C"], conv.K, dtype=torch.float64, generator=gen)
    x0[:, :, ::9] = float("nan")
    mask = None
    if with_mask:
        mask = (torch.rand(cs["B"], 1, conv.K, generator=gen, dtype=torch.float64) > 0.2).double()
    ref = run(conv, x0, lambda x: reference_local_mean_forward(conv, geom[:3], geom[3], x, mask))
    got = run(conv, x0, lambda x: conv(x, mask=mask))
    for a, b in zip(got, ref):
        assert torch.isfinite(a).all()
    errs = rel_errs(got, ref)
    assert max(errs) < 1e-12, errs


def test_local_mean_stencil_cache():
    """The gather layout of missing="local_mean" is built once, follows
    module.to() and is not pickled; results are unchanged."""
    import pickle

    cs = CASES["regional"]
    conv = make_conv(cs, torch.float32, "cpu", missing="local_mean")
    x0 = torch.randn(cs["B"], cs["C"], conv.K)
    x0[:, :, ::9] = float("nan")
    y = conv(x0)
    pos, wn = conv._lm_stencil
    assert pos.dtype == torch.int64 and wn.dtype == torch.float32
    conv(x0)
    assert conv._lm_stencil[0] is pos                     # reused, not rebuilt
    clone = pickle.loads(pickle.dumps(conv))
    assert clone._lm_stencil is None
    assert torch.equal(clone(x0), y)
    conv.double()
    assert conv._lm_stencil is None
    y64 = conv(x0.double())
    assert conv._lm_stencil[1].dtype == torch.float64
    torch.testing.assert_close(y64.float(), y, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_legacy_whole_module_pickle():
    """A layer pickled whole before #90 holds _pos_safe / _w_norm buffers."""
    import pickle

    conv = HealPixConv(2, 2, 3, kernel_sz=3, n_gauges=2, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    x = torch.randn(1, 2, conv.K, dtype=torch.float64)
    y = conv(x)
    state = conv.__dict__.copy()
    bufs = dict(state["_buffers"])
    bufs["_pos_safe"], bufs["_w_norm"] = conv._pos_safe, conv._w_norm
    del bufs["_tap_idx"], bufs["_tap_w"]
    state["_buffers"] = bufs
    for name in ("backend", "deterministic", "_tmap"):
        state.pop(name, None)
    old = HealPixConv.__new__(HealPixConv)
    old.__setstate__(state)
    old = pickle.loads(pickle.dumps(old))
    assert torch.equal(old(x), y)
    assert old.backend == "auto" and old.deterministic is True


def test_gradcheck_level2():
    conv = HealPixConv(2, 2, 2, kernel_sz=3, n_gauges=2, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    x = torch.randn(2, 2, conv.K, dtype=torch.float64, requires_grad=True)
    W = torch.randn_like(conv.weight).requires_grad_(True)
    b = torch.randn_like(conv.bias).requires_grad_(True)

    def f(x, W, b):
        return cb.gauge_conv(x, W, b, conv._tap_idx, conv._tap_w, backend="torch")

    assert torch.autograd.gradcheck(f, (x, W, b), eps=1e-6, atol=1e-8)


def test_gradgradcheck_torch_backend():
    """Second-order gradients (create_graph=True, e.g. a gradient penalty)."""
    conv = HealPixConv(1, 2, 2, kernel_sz=3, n_gauges=2, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    x = torch.randn(1, 2, conv.K, dtype=torch.float64, requires_grad=True)
    W = torch.randn_like(conv.weight).requires_grad_(True)
    b = torch.randn_like(conv.bias).requires_grad_(True)

    def f(x, W, b):
        return cb.gauge_conv(x, W, b, conv._tap_idx, conv._tap_w, backend="torch")

    assert torch.autograd.gradgradcheck(f, (x, W, b), eps=1e-6, atol=1e-8)


def _rot90_taps(kernel_sz):
    """p -> p' such that the grid point of p' is the one of p rotated by +90 deg."""
    grid = np.arange(kernel_sz) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    xx, yy = xx.ravel(), yy.ravel()
    where = {(a, b): i for i, (a, b) in enumerate(zip(xx, yy))}
    return np.array([where[(-yy[p], xx[p])] for p in range(xx.size)])


@pytest.mark.parametrize("backend", ["torch", pytest.param("triton", marks=needs_triton)])
def test_gauge_equivariance(backend):
    """Gauge g + G/2 is gauge g rotated by 90 deg: with the same kernel W, its
    output equals that of gauge g with the kernel rotated by 90 deg."""
    G, k = 4, 5
    device = "cuda" if backend == "triton" else "cpu"
    dtype = torch.float32 if backend == "triton" else torch.float64
    tol = 1e-5 if backend == "triton" else 1e-9
    # geometry built in float64 in both cases: a float32 build rounds the two
    # rotated stencils differently (~1e-5), which is not what is tested here
    conv = HealPixConv(4, 2, 3, kernel_sz=k, n_gauges=G, gauge_type="projected_ref",
                       singularity_lonlat=(20.0, 5.0), device=device, dtype=torch.float64,
                       cache_dir=None, backend=backend).to(dtype)
    rng = np.random.default_rng(5)
    W = rng.standard_normal((2, 3, k * k))
    x = torch.as_tensor(rng.standard_normal((2, 2, conv.K)), dtype=dtype, device=device)
    with torch.no_grad():
        y = conv.set_kernel(W)(x)
        perm = _rot90_taps(k)
        W_rot = np.empty_like(W)
        W_rot[..., perm] = W                  # W_rot[p'] = W[p]
        y_rot = conv.set_kernel(W_rot)(x)
    Co = 3
    for g in range(G // 2):
        a = y[:, (g + G // 2) * Co:(g + G // 2 + 1) * Co]
        b = y_rot[:, g * Co:(g + 1) * Co]
        err = float((a - b).abs().max() / b.abs().max())
        assert err < tol, (g, err)


# ---------------------------------------------------------------------------
# Phase 1: device, dtype, cache, state_dict
# ---------------------------------------------------------------------------

def test_module_to_follows_dtype_and_device():
    conv = HealPixConv(2, 2, 3, kernel_sz=3, n_gauges=2, device="cpu", cache_dir=None)
    conv.double()
    assert conv.dtype == torch.float64 and conv._tap_w.dtype == torch.float64
    assert conv._tap_idx.dtype == torch.int32
    y = conv(torch.randn(1, 2, conv.K, dtype=torch.float64))
    assert y.dtype == torch.float64
    if HAS_CUDA:
        conv.to("cuda")
        assert conv.device.type == "cuda" and conv._tap_idx.is_cuda
        y = conv(torch.randn(1, 2, conv.K, dtype=torch.float64))   # moved to the layer
        assert y.is_cuda


def test_use_norm_follows_dtype():
    conv = HealPixConv(2, 2, 4, n_gauges=2, use_norm=True, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    assert conv.norm.weight.dtype == torch.float64
    assert conv(torch.randn(1, 2, conv.K, dtype=torch.float64)).dtype == torch.float64


def test_set_kernel_keeps_float64():
    conv = HealPixConv(2, 2, 3, kernel_sz=3, n_gauges=2, device="cpu",
                       dtype=torch.float64, cache_dir=None)
    W = np.random.default_rng(0).standard_normal((2, 2, 3, 9))
    b = np.random.default_rng(1).standard_normal(6)
    conv.set_kernel(W, bias=b)
    assert np.array_equal(conv.weight.detach().numpy(), W)
    assert np.array_equal(conv.bias.detach().numpy(), b)
    conv.set_kernel(torch.as_tensor(W[0]))         # [C_in, C_out, P] broadcast, tensor input
    assert np.array_equal(conv.weight.detach().numpy()[1], W[0])


def test_cache_keeps_float64(tmp_path):
    kw = dict(level=3, in_channels=1, out_channels=1, kernel_sz=3, n_gauges=2,
              cell_ids=unsorted_patch(3, 2), device="cpu", dtype=torch.float64)
    fresh = HealPixConv(cache_dir=None, **kw)
    first = HealPixConv(cache_dir=tmp_path, **kw)     # computes and writes
    files = list(tmp_path.glob("healpixconv_*.pt"))
    assert len(files) == 1
    second = HealPixConv(cache_dir=tmp_path, **kw)    # loads
    for conv in (first, second):
        assert conv._tap_w.dtype == torch.float64
        assert torch.equal(conv._tap_w, fresh._tap_w)
        assert torch.equal(conv._tap_idx, fresh._tap_idx)
    # a float32 layer has its own cache entry
    HealPixConv(cache_dir=tmp_path, **{**kw, "dtype": torch.float32})
    assert len(list(tmp_path.glob("healpixconv_*.pt"))) == 2


def test_cache_key_uses_order_and_precision():
    ids = nested_patch(4, 2)
    args = dict(nside=16, kernel_sz=3, G=2, gauge_type="phi", ref_direction=None,
                nest=True, ellipsoid="WGS84")
    k1 = _geometry_cache_key(cell_ids=ids, **args)
    assert k1 == _geometry_cache_key(cell_ids=ids.tolist(), **args)
    assert k1 != _geometry_cache_key(cell_ids=ids[::-1].copy(), **args)
    assert k1 != _geometry_cache_key(cell_ids=ids, geom_dtype=torch.float64, **args)
    assert k1 != _geometry_cache_key(cell_ids=None, **args)


def test_state_dict_without_geometry_and_legacy_load():
    kw = dict(level=2, in_channels=2, out_channels=3, kernel_sz=3, n_gauges=2,
              device="cpu", cache_dir=None)
    conv = HealPixConv(**kw)
    sd = conv.state_dict()
    assert not any(k.endswith(("_tap_idx", "_tap_w", "_pos_safe", "_w_norm")) for k in sd)
    # a state_dict written before #90 carries the stencil: it still loads strictly
    legacy = dict(sd)
    legacy["_pos_safe"] = conv._pos_safe.clone()
    legacy["_w_norm"] = conv._w_norm.clone()
    other = HealPixConv(**kw)
    other.load_state_dict(legacy, strict=True)
    assert torch.equal(other.weight, conv.weight)


def test_backend_argument():
    with pytest.raises(ValueError):
        HealPixConv(1, 1, 1, device="cpu", cache_dir=None, backend="cuda")
    conv = HealPixConv(1, 1, 1, device="cpu", cache_dir=None)
    if not cb.triton_available():
        conv.backend = "triton"
        with pytest.raises(RuntimeError):
            conv(torch.randn(1, 1, conv.K))


def test_deterministic_is_the_default():
    """deterministic=True by default (#90, open question 3); False is the opt-in."""
    import inspect

    assert HealPixConv(1, 1, 1, device="cpu", cache_dir=None).deterministic is True
    assert HealPixConv.deterministic is True
    assert inspect.signature(cb.gauge_conv).parameters["deterministic"].default is True
    conv = HealPixConv(1, 1, 1, device="cpu", cache_dir=None, deterministic=False)
    assert conv.deterministic is False


def test_transposed_map_is_the_transpose():
    conv = make_conv(CASES["regional"], torch.float64, "cpu")
    ptr, base, tk, tw = cb.build_transposed_map(conv._tap_idx, conv._tap_w, conv.K)
    P, _, G, K = conv._tap_idx.shape
    dense = torch.zeros(P, G, K, K, dtype=torch.float64)   # [p, g, target, source]
    for p in range(P):
        for j in range(4):
            dense[p].view(G * K, K).index_put_(
                (torch.arange(G * K), conv._tap_idx[p, j].reshape(-1).long()),
                conv._tap_w[p, j].reshape(-1), accumulate=True)
    dense_t = torch.zeros_like(dense)
    for p in range(P):
        counts = (ptr[p, 1:] - ptr[p, :-1]).long()            # entries per (g, source)
        rows = torch.repeat_interleave(torch.arange(G * K), counts)
        e = slice(int(base[p]), int(base[p + 1]))
        assert int(counts.sum()) == e.stop - e.start
        dense_t[p].index_put_((rows // K, tk[e].long(), rows % K), tw[e], accumulate=True)
    # same entries, summed in another order: equal to round-off
    torch.testing.assert_close(dense, dense_t, rtol=0, atol=1e-15)
    assert (tw != 0).all()


# ---------------------------------------------------------------------------
# Triton backend, float32
# ---------------------------------------------------------------------------

@needs_triton
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("case", sorted(CASES))
def test_triton_matches_reference(case, deterministic):
    cs = CASES[case]
    dev = torch.device("cuda")
    conv = make_conv(cs, torch.float32, dev, backend="triton", deterministic=deterministic)
    geom = reference_geometry(conv, cs["cell_ids"], torch.float32, dev)
    x0 = torch.randn(cs["B"], cs["C"], conv.K, device=dev)
    ref = run(conv, x0, lambda x: reference_forward(conv, geom, x))
    got = run(conv, x0, conv)
    errs = rel_errs(got, ref)
    assert max(errs) < 1e-5, errs


@needs_triton
@pytest.mark.parametrize("C, Co", [(1, 1), (16, 16), (24, 40), (70, 9)])
def test_triton_channel_tiles(C, Co):
    """Channel counts below, at and above the tile sizes (masking, several tiles)."""
    dev = torch.device("cuda")
    cs = dict(CASES["regional"], C=C, Co=Co)
    conv = make_conv(cs, torch.float32, dev, backend="triton")
    geom = reference_geometry(conv, cs["cell_ids"], torch.float32, dev)
    x0 = torch.randn(2, C, conv.K, device=dev)
    for det in (False, True):
        conv.deterministic = det
        errs = rel_errs(run(conv, x0, conv), run(conv, x0, lambda x: reference_forward(conv, geom, x)))
        assert max(errs) < 1e-5, (det, errs)


@needs_triton
def test_triton_deterministic_is_bitwise_reproducible():
    cs = CASES["full"]
    dev = torch.device("cuda")
    conv = make_conv(dict(cs, C=16, Co=16, B=4), torch.float32, dev, backend="triton")
    assert conv.deterministic                       # the default
    x0 = torch.randn(4, 16, conv.K, device=dev)
    a = run(conv, x0, conv)
    b = run(conv, x0, conv)
    # without a cached map, gauge_conv builds it at the backward pass: same bits
    c = run(conv, x0, lambda x: cb.gauge_conv(x, conv.weight, conv.bias, conv._tap_idx,
                                              conv._tap_w, backend="triton"))
    for u, v, t in zip(a, b, c):
        assert torch.equal(u, v) and torch.equal(u, t)


@needs_triton
@pytest.mark.parametrize("deterministic", [False, True])
def test_triton_double_backward_matches_torch(deterministic):
    """create_graph=True on the Triton path keeps the second-order terms: a
    gradient penalty reaches the weights as with the PyTorch backend."""
    cs = CASES["regional"]
    dev = torch.device("cuda")
    conv = make_conv(cs, torch.float32, dev, deterministic=deterministic)
    x0 = torch.randn(cs["B"], cs["C"], conv.K, device=dev)

    def penalty(backend):
        conv.backend = backend
        conv.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        y = conv(x)
        gy = torch.sin(torch.arange(y.numel(), dtype=torch.float64, device=dev)).view_as(y)
        loss = (y * gy.float()).sum() + y.square().sum()
        gx, = torch.autograd.grad(loss, x, create_graph=True)
        (loss + gx.pow(2).sum()).backward()
        return gx.detach(), x.grad.detach(), conv.weight.grad.clone(), conv.bias.grad.clone()

    got, ref = penalty("triton"), penalty("torch")
    errs = rel_errs(got, ref)
    assert max(errs) < 1e-5, errs
    # the penalty does reach the weights
    conv.backend = "triton"
    conv.zero_grad(set_to_none=True)
    x = x0.clone().requires_grad_(True)
    y = conv(x)
    gy = torch.sin(torch.arange(y.numel(), dtype=torch.float64, device=dev)).view_as(y)
    ((y * gy.float()).sum() + y.square().sum()).backward()
    assert (conv.weight.grad - got[2]).abs().max() > 1e-3 * got[2].abs().max()


@needs_cuda
def test_torch_backend_on_cuda_is_reproducible_under_deterministic_algorithms():
    """deterministic= is a Triton option; the PyTorch backend on CUDA
    (index_add_) gives the same bits under use_deterministic_algorithms."""
    import warnings

    cs = CASES["regional"]
    dev = torch.device("cuda")
    conv = make_conv(dict(cs, C=16, Co=16), torch.float32, dev, backend="torch")
    x0 = torch.randn(cs["B"], 16, conv.K, device=dev)
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = run(conv, x0, conv)
            b = run(conv, x0, conv)
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    for u, v in zip(a, b):
        assert torch.equal(u, v)


@needs_triton
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_16bit(dtype, deterministic):
    cs = CASES["regional"]
    dev = torch.device("cuda")
    conv = make_conv(dict(cs, C=16, Co=16), torch.float32, dev, deterministic=deterministic)
    geom = reference_geometry(conv, cs["cell_ids"], torch.float32, dev)
    x0 = torch.randn(cs["B"], 16, conv.K, device=dev)
    ref = run(conv, x0, lambda x: reference_forward(conv, geom, x))
    half = make_conv(dict(cs, C=16, Co=16), torch.float32, dev,
                     deterministic=deterministic).to(dtype)
    with torch.no_grad():
        half.weight.copy_(conv.weight)
        half.bias.copy_(conv.bias)
    got = run(half, x0.to(dtype), half)
    assert got[0].dtype == dtype
    errs = rel_errs(got, ref)
    assert max(errs) < 3e-2, errs
    # autocast: float32 layer, 16-bit compute, fp32 accumulation
    with torch.autocast("cuda", dtype=dtype):
        got = run(conv, x0, conv)
    assert got[0].dtype == dtype and conv.weight.grad.dtype == torch.float32
    errs = rel_errs(got, ref)
    assert max(errs) < 3e-2, errs


@needs_cuda
def test_float64_on_cuda_runs_on_torch_backend():
    cs = CASES["regional"]
    dev = torch.device("cuda")
    conv = make_conv(cs, torch.float64, dev)
    assert not cb._use_triton(torch.zeros(1, device=dev, dtype=torch.float64), "auto")
    geom = reference_geometry(conv, cs["cell_ids"], torch.float64, dev)
    x0 = torch.randn(cs["B"], cs["C"], conv.K, dtype=torch.float64, device=dev)
    errs = rel_errs(run(conv, x0, conv), run(conv, x0, lambda x: reference_forward(conv, geom, x)))
    assert max(errs) < 1e-12, errs
