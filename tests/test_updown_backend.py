"""
HealPixDown / HealPixUp compute backends (#109, section 1).

Every backend is compared with the reference implementation
(``backend="csr"``: CSR matrix, ``torch.sparse.mm`` and child gather):

* contiguous NESTED fast path (``mean`` / ``maxpool`` Down, ``expand`` Up):
  bit-identical values;
* ELL gather (Gaussian modes, every Up): equal to rounding, in float64,
  float32 and bfloat16, values and gradients, on the full sphere and on
  unsorted regional domains.
"""

import os

import numpy as np
import pytest
import torch

from healpix_analyse import _updown_backend as _ub
from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp

LEVEL = 4
DOWN_MODES = ("smooth", "smooth_conservative", "mean", "maxpool")
UP_NORMS = ("adjoint", "col_l1", "diag_l2")
TOL = {torch.float64: dict(rtol=1e-12, atol=1e-12),
       torch.float32: dict(rtol=1e-5, atol=1e-5),
       torch.bfloat16: dict(rtol=2e-2, atol=2e-2)}


def _unsorted_ids(level, frac=0.5, seed=0):
    """A shuffled subset of the cells of ``level``: incomplete parents."""
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(rng.random(12 * 4**level) < frac)
    return rng.permutation(ids)


def _complete_sorted_ids(level, seed=0):
    """Sorted cells of a random half of the parents, all children present."""
    rng = np.random.default_rng(seed)
    parents = np.flatnonzero(rng.random(12 * 4**(level - 1)) < 0.5)
    return (4 * parents[:, None] + np.arange(4)).ravel()


DOMAINS = {
    "full": lambda: None,
    "sorted_complete": lambda: _complete_sorted_ids(LEVEL),
    "unsorted": lambda: _unsorted_ids(LEVEL),
}


def _pair(cls, *args, **kw):
    return cls(*args, device="cpu", **kw), cls(*args, device="cpu", backend="csr", **kw)


def _value_and_grad(op, x, g):
    x = x.detach().clone().requires_grad_(True)
    y = op(x)[0]
    (gx,) = torch.autograd.grad(y, x, g)
    return y.detach(), gx


def _data(n_rows, n, dtype, seed=0, nan=False):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n, generator=gen, dtype=torch.float64)
    if nan:
        x[0, ::7] = float("nan")
    return x.to(dtype)


# ---------------------------------------------------------------------------
# contiguous NESTED blocks
# ---------------------------------------------------------------------------

def test_nested_blocks_detection():
    assert _ub.nested_blocks(None)
    assert _ub.nested_blocks(np.arange(16, 32))
    assert _ub.nested_blocks(_complete_sorted_ids(3))
    assert not _ub.nested_blocks(np.arange(16, 32)[::-1])          # unsorted
    assert not _ub.nested_blocks(np.arange(17, 33))                # misaligned
    assert not _ub.nested_blocks(np.array([0, 1, 2, 3, 4, 5, 6, 8]))  # incomplete
    assert not _ub.nested_blocks(np.array([4, 5, 6, 7, 0, 1, 2, 3]))  # parents unsorted
    assert not _ub.nested_blocks(np.arange(6))


@pytest.mark.parametrize("domain", ["full", "sorted_complete"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode,skipna", [("mean", False), ("mean", True),
                                         ("maxpool", False)])
def test_block_down_bit_identical(domain, dtype, mode, skipna):
    ids = DOMAINS[domain]()
    fast, ref = _pair(HealPixDown, LEVEL, mode=mode, skipna=skipna,
                      cell_ids=ids, dtype=dtype)
    assert fast._blocks and not ref._blocks
    # no index array is stored
    assert not any(n.startswith("_children") for n, _ in fast.named_buffers())
    x = _data(3, fast.N_in, dtype, nan=skipna)
    g = _data(3, fast.N_out, dtype, seed=1)
    y, gx = _value_and_grad(fast, x, g)
    y_ref, gx_ref = _value_and_grad(ref, x, g)
    assert torch.equal(y.isnan(), y_ref.isnan())
    assert torch.equal(torch.nan_to_num(y), torch.nan_to_num(y_ref))
    torch.testing.assert_close(gx, gx_ref, equal_nan=True, rtol=0, atol=0)
    np.testing.assert_array_equal(fast.cell_ids_out, ref.cell_ids_out)
    if mode == "mean":
        np.testing.assert_array_equal(fast.coverage, ref.coverage)
        torch.testing.assert_close(fast.sparse_matrix().to_dense(),
                                   ref.sparse_matrix().to_dense(), rtol=0, atol=0)
        if skipna:
            assert torch.equal(fast.valid_fraction(x), ref.valid_fraction(x))


@pytest.mark.parametrize("mode", ["mean", "maxpool"])
def test_unsorted_domain_keeps_gather(mode):
    ids = _unsorted_ids(LEVEL)
    fast, ref = _pair(HealPixDown, LEVEL, mode=mode, cell_ids=ids,
                      dtype=torch.float64)
    assert not fast._blocks
    x = _data(2, fast.N_in, torch.float64)
    assert torch.equal(fast(x)[0], ref(x)[0])


@pytest.mark.parametrize("domain", ["full", "sorted_complete"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("up_norm", UP_NORMS)
def test_expand_up_bit_identical(domain, dtype, up_norm):
    ids = DOMAINS[domain]()
    down = HealPixDown(LEVEL, mode="mean", cell_ids=ids, device="cpu", dtype=dtype)
    fast, ref = _pair(HealPixUp, LEVEL - 1, paired_down=down, up_norm=up_norm,
                      dtype=dtype)
    assert fast._expand
    assert not any(n.startswith("_ell") for n, _ in fast.named_buffers())
    x = _data(3, fast.N_in, dtype)
    g = _data(3, fast.N_out, dtype, seed=1)
    y, gx = _value_and_grad(fast, x, g)
    y_ref, gx_ref = _value_and_grad(ref, x, g)
    assert torch.equal(y, y_ref)
    torch.testing.assert_close(gx, gx_ref, **TOL[dtype])
    np.testing.assert_array_equal(fast.cell_ids_out, ref.cell_ids_out)
    torch.testing.assert_close(fast._sparse_coo().to_dense(),
                               ref._sparse_coo().to_dense(), rtol=0, atol=0)


def test_expand_up_not_used_for_incomplete_parents():
    down = HealPixDown(LEVEL, mode="mean", cell_ids=_unsorted_ids(LEVEL),
                       device="cpu", dtype=torch.float64)
    up = HealPixUp(LEVEL - 1, paired_down=down, device="cpu", dtype=torch.float64)
    assert not up._expand and up._ell_idx.shape[1] == 1     # one entry per row


# ---------------------------------------------------------------------------
# ELL gather against the CSR reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["full", "unsorted"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["smooth", "smooth_conservative"])
def test_ell_down_matches_csr(domain, dtype, mode):
    ids = DOMAINS[domain]()
    fast, ref = _pair(HealPixDown, LEVEL, mode=mode, cell_ids=ids, dtype=dtype)
    x = _data(3, fast.N_in, dtype)
    g = _data(3, fast.N_out, dtype, seed=1)
    y, gx = _value_and_grad(fast, x, g)
    y_ref, gx_ref = _value_and_grad(ref, x, g)
    assert y.dtype == dtype and gx.dtype == dtype
    torch.testing.assert_close(y, y_ref, **TOL[dtype])
    torch.testing.assert_close(gx, gx_ref, **TOL[dtype])
    torch.testing.assert_close(fast.sparse_matrix().to_dense(),
                               ref.sparse_matrix().to_dense(), rtol=0, atol=0)


def _up_kwargs(kind, domain, dtype):
    ids = DOMAINS[domain]()
    if kind == "cell_ids":
        return dict(cell_ids=None if ids is None else np.unique(ids // 4))
    down = HealPixDown(LEVEL, mode=kind, cell_ids=ids, device="cpu", dtype=dtype)
    return dict(paired_down=down)


@pytest.mark.parametrize("domain", ["full", "unsorted"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("up_norm", UP_NORMS)
@pytest.mark.parametrize("kind", ["cell_ids", "smooth", "smooth_conservative", "mean"])
def test_ell_up_matches_csr(domain, dtype, up_norm, kind):
    kw = _up_kwargs(kind, domain, dtype)
    fast, ref = _pair(HealPixUp, LEVEL - 1, up_norm=up_norm, dtype=dtype, **kw)
    x = _data(3, fast.N_in, dtype)
    g = _data(3, fast.N_out, dtype, seed=1)
    y, gx = _value_and_grad(fast, x, g)
    y_ref, gx_ref = _value_and_grad(ref, x, g)
    assert y.dtype == dtype and gx.dtype == dtype
    torch.testing.assert_close(y, y_ref, **TOL[dtype])
    torch.testing.assert_close(gx, gx_ref, **TOL[dtype])


def test_nan_does_not_leak_through_padding():
    ids = _unsorted_ids(LEVEL, 0.3, 4)          # rows of very different fan-in
    fast, ref = _pair(HealPixDown, LEVEL, cell_ids=ids, dtype=torch.float64)
    assert (fast._ell_idx == fast.N_in).any()               # there is padding
    x = _data(2, fast.N_in, torch.float64, nan=True)
    y, y_ref = fast(x)[0], ref(x)[0]
    assert torch.equal(y.isnan(), y_ref.isnan())
    torch.testing.assert_close(y, y_ref, equal_nan=True)


@pytest.mark.parametrize("domain", ["full", "unsorted"])
def test_gradcheck(domain):
    ids = DOMAINS[domain]()
    kw = dict(device="cpu", dtype=torch.float64)
    ops = [HealPixDown(LEVEL - 1, mode=m, cell_ids=None if ids is None else
                       np.unique(ids // 4), **kw) for m in DOWN_MODES]
    up_ids = None if ids is None else np.unique(ids // 16)
    ops += [HealPixUp(LEVEL - 2, cell_ids=up_ids, up_norm=n, **kw) for n in UP_NORMS]
    ops.append(HealPixUp(LEVEL - 2, paired_down=ops[2], **kw))     # mean pair
    for op in ops:
        x = torch.randn(2, op.N_in, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: op(t)[0], (x,))
        if not (isinstance(op, HealPixDown) and op.mode == "maxpool"):
            assert torch.autograd.gradgradcheck(lambda t: op(t)[0] ** 2, (x,))


# ---------------------------------------------------------------------------
# layout: [B, C, N] in, [B, C, N_out] out, contiguous, no copy of the input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", DOWN_MODES)
def test_batch_channel_layout(mode):
    down = HealPixDown(LEVEL, mode=mode, device="cpu", dtype=torch.float64)
    up = HealPixUp(LEVEL - 1, device="cpu", dtype=torch.float64)
    x = _data(6, down.N_in, torch.float64).view(2, 3, down.N_in)
    y = down(x)[0]
    assert y.shape == (2, 3, down.N_out) and y.is_contiguous()
    torch.testing.assert_close(y.reshape(6, -1), down(x.reshape(6, -1))[0],
                               rtol=0, atol=0)
    z = up(y)[0]
    assert z.shape == (2, 3, up.N_out) and z.is_contiguous()


def test_backend_validation():
    with pytest.raises(ValueError, match="backend"):
        HealPixDown(2, backend="cusparse", device="cpu")
    if not _ub.triton_available():
        op = HealPixDown(2, backend="triton", device="cpu")
        with pytest.raises(RuntimeError, match="Triton"):
            op(torch.randn(op.N_in))


def test_ell_roundtrip():
    down = HealPixDown(LEVEL, cell_ids=_unsorted_ids(LEVEL), device="cpu",
                       dtype=torch.float64)
    M = down.sparse_matrix()
    (fwd, bwd) = down._ell_ops()
    MT = _ub.ell_to_coo(bwd.idx, bwd.w, bwd.n_cols)
    torch.testing.assert_close(MT.to_dense(), M.to_dense().T, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Triton kernel (CUDA, or the CPU interpreter with TRITON_INTERPRET=1)
# ---------------------------------------------------------------------------

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_triton = pytest.mark.skipif(
    not (_ub._HAS_TRITON and (torch.cuda.is_available() or _INTERPRET)),
    reason="needs Triton with CUDA or TRITON_INTERPRET=1",
)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["down", "up_l1"])
def test_triton_kernel_matches_torch(dtype, kind):
    dev = _device()
    ids = _unsorted_ids(LEVEL)
    if kind == "down":
        op = HealPixDown(LEVEL, cell_ids=ids, device=dev, dtype=torch.float32)
        div = None
    else:
        op = HealPixUp(LEVEL - 1, cell_ids=np.unique(ids // 4), device=dev,
                       dtype=torch.float32)
        div = op._col_sum.clamp_min(op.eps)
    fwd, bwd = op._ell_ops()
    x = torch.randn(5, fwd.n_cols, device=dev).to(dtype)
    y_t = _ub._ell_triton(x, fwd.idx, fwd.w, div, dtype)
    y_r = _ub._ell_torch(x.double(), fwd.idx, fwd.w.double(),
                         None if div is None else div.double(), torch.float64)
    tol = TOL[torch.float32] if dtype == torch.float32 else TOL[torch.bfloat16]
    torch.testing.assert_close(y_t.double(), y_r, **tol)
    # deterministic: the same bits at every run
    assert torch.equal(y_t, _ub._ell_triton(x, fwd.idx, fwd.w, div, dtype))
    # backward operator
    g = torch.randn(5, op._M_size[0], device=dev).to(dtype)
    gx_t = _ub._ell_triton(g, bwd.idx, bwd.w, None, dtype)
    gx_r = _ub._ell_torch(g.double(), bwd.idx, bwd.w.double(), None, torch.float64)
    torch.testing.assert_close(gx_t.double(), gx_r, **tol)


@pytest.mark.skipif(not _ub.triton_available(), reason="needs Triton and CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("domain", ["full", "unsorted"])
def test_triton_layers_match_csr(dtype, domain):
    ids = DOMAINS[domain]()
    for mode in ("smooth", "smooth_conservative"):
        fast = HealPixDown(LEVEL, mode=mode, cell_ids=ids, device="cuda",
                           dtype=dtype, backend="triton")
        ref = HealPixDown(LEVEL, mode=mode, cell_ids=ids, device="cuda",
                          dtype=dtype, backend="csr")
        x = torch.randn(4, fast.N_in, device="cuda").to(dtype)
        g = torch.randn(4, fast.N_out, device="cuda").to(dtype)
        y, gx = _value_and_grad(fast, x, g)
        y_ref, gx_ref = _value_and_grad(ref, x, g)
        torch.testing.assert_close(y, y_ref, **TOL[dtype])
        torch.testing.assert_close(gx, gx_ref, **TOL[dtype])
        # deterministic backward
        assert torch.equal(gx, _value_and_grad(fast, x, g)[1])
    up_ids = None if ids is None else np.unique(ids // 4)
    for up_norm in UP_NORMS:
        fast = HealPixUp(LEVEL - 1, cell_ids=up_ids, up_norm=up_norm,
                         device="cuda", dtype=dtype, backend="triton")
        ref = HealPixUp(LEVEL - 1, cell_ids=up_ids, up_norm=up_norm,
                        device="cuda", dtype=dtype, backend="csr")
        x = torch.randn(4, fast.N_in, device="cuda").to(dtype)
        g = torch.randn(4, fast.N_out, device="cuda").to(dtype)
        y, gx = _value_and_grad(fast, x, g)
        y_ref, gx_ref = _value_and_grad(ref, x, g)
        torch.testing.assert_close(y, y_ref, **TOL[dtype])
        torch.testing.assert_close(gx, gx_ref, **TOL[dtype])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fast_paths_bit_identical(dtype):
    for mode in ("mean", "maxpool"):
        fast = HealPixDown(LEVEL, mode=mode, device="cuda", dtype=dtype)
        ref = HealPixDown(LEVEL, mode=mode, device="cuda", dtype=dtype, backend="csr")
        x = torch.randn(4, fast.N_in, device="cuda").to(dtype)
        assert torch.equal(fast(x)[0], ref(x)[0])
    down = HealPixDown(LEVEL, mode="mean", device="cuda", dtype=dtype)
    for up_norm in UP_NORMS:
        fast = HealPixUp(LEVEL - 1, paired_down=down, up_norm=up_norm,
                         device="cuda", dtype=dtype)
        ref = HealPixUp(LEVEL - 1, paired_down=down, up_norm=up_norm,
                        device="cuda", dtype=dtype, backend="csr")
        assert fast._expand
        x = torch.randn(4, fast.N_in, device="cuda").to(dtype)
        assert torch.equal(fast(x)[0], ref(x)[0])


@needs_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_kernels_match_reference(dtype):
    """The fused maxpool, expand and smooth_conservative kernels against
    their PyTorch forms (values bit for bit where the reference is exact)."""
    dev = _device()
    gen = torch.Generator().manual_seed(3)
    # maxpool: values identical, NaN propagated, gradient to the maximum
    x = torch.randn(3, 4 * 300, generator=gen).to(dtype).to(dev)
    x[0, 5] = float("nan")
    xr = x.detach().requires_grad_(True)
    y = _ub._Max4.apply(xr)
    ref = x.reshape(3, -1, 4).max(dim=-1)
    assert torch.equal(y.isnan(), ref.values.isnan())
    assert torch.equal(torch.nan_to_num(y), torch.nan_to_num(ref.values))
    g = torch.randn(y.shape, generator=gen).to(dtype).to(dev)
    (gx,) = torch.autograd.grad(y, xr, g)
    gref = torch.zeros(3, 300, 4, dtype=dtype, device=dev).scatter_(
        -1, ref.indices[..., None], g[..., None]).reshape(3, -1)
    assert torch.equal(gx, gref)
    # expand: bit-identical forward, gradient to rounding
    v = torch.full((300,), 0.25, device=dev)
    d = torch.rand(300, generator=gen).to(dev) + 0.1
    for dd in (None, d):
        z = torch.randn(3, 300, generator=gen).to(dtype).to(dev).requires_grad_(True)
        y = _ub._Expand4.apply(z, v, dd, 1e-12, dtype)
        y_ref = (z.float() * v) if dd is None else (z.float() * v) / dd.clamp_min(1e-12)
        y_ref = y_ref.to(dtype).repeat_interleave(4, dim=1)
        if dtype == torch.float32 or not _INTERPRET:
            assert torch.equal(y.detach(), y_ref.detach())
        else:   # the interpreter emulates the fp32 -> bf16 rounding in numpy
            torch.testing.assert_close(y, y_ref, **TOL[dtype])
        g = torch.randn(y.shape, generator=gen).to(dtype).to(dev)
        (gz,) = torch.autograd.grad(y, z, g)
        (gz_ref,) = torch.autograd.grad(y_ref, z, g)
        torch.testing.assert_close(gz, gz_ref, **TOL[dtype])
    # smooth_conservative, NaN included
    op = HealPixDown(LEVEL, mode="smooth_conservative", cell_ids=_unsorted_ids(LEVEL),
                     device=dev, dtype=torch.float32)
    fwd, bwd = op._ell_ops()
    x = torch.randn(3, op.N_in, generator=gen).to(dtype).to(dev)
    x[1, ::11] = float("nan")
    xr = x.detach().requires_grad_(True)
    y = _ub._ConservativeEll.apply(xr, fwd, bwd, op._coverage, dtype)
    xr2 = x.detach().requires_grad_(True)
    y_ref = _ub._conservative_reference(xr2, fwd, bwd, op._coverage,
                                        torch.float32).to(dtype)
    torch.testing.assert_close(y, y_ref, equal_nan=True, **TOL[dtype])
    g = torch.randn(y.shape, generator=gen).to(dtype).to(dev)
    (gx,) = torch.autograd.grad(y, xr, g)
    # gradient against a float64 evaluation (the bf16 reference rounds twice)
    x64 = x.double().requires_grad_(True)
    y64 = _ub._conservative_reference(x64, fwd, bwd, op._coverage, torch.float64)
    (gx64,) = torch.autograd.grad(y64, x64, g.double())
    scale = gx64.abs().max().item()
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(gx.double(), gx64, rtol=0, atol=tol * scale)
