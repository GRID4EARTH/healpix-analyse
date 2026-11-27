"""HealPixConv(missing=...): default unchanged, local-mean completion.

missing="local_mean" must treat a NaN pixel exactly like a pixel outside
cell_ids, give finite outputs and gradients, and leave the default
behaviour ("centre") untouched.
"""

import numpy as np
import pytest
import torch

from healpix_analyse.convol import HealPixConv

LEVEL = 3
DOMAIN = np.arange(64 * 2, dtype=np.int64)         # base pixels 0 and 1 at level 3


def _conv(cell_ids=DOMAIN, cache_dir=None, dtype=torch.float64, **kw):
    conv = HealPixConv(level=LEVEL, in_channels=1, out_channels=2, kernel_sz=3,
                       n_gauges=2, cell_ids=cell_ids, device="cpu",
                       dtype=dtype, cache_dir=cache_dir, **kw)
    rng = np.random.default_rng(3)
    conv.set_kernel(rng.standard_normal((1, 2, 9)), bias=[0.1, -0.2, 0.3, 0.0])
    return conv


def test_default_is_centre_and_state_dict_unchanged():
    a, b = _conv(), _conv(missing="centre")
    x = torch.randn(2, len(DOMAIN), dtype=torch.float64)
    assert torch.equal(a(x), b(x))
    # the stencil geometry is rebuilt at construction, not stored (#90)
    assert set(a.state_dict()) == {"weight", "bias", "_sort_order", "_inv_order"}
    assert set(_conv(missing="local_mean").state_dict()) == set(a.state_dict())
    with pytest.raises(ValueError):
        a(x, mask=torch.ones(len(DOMAIN)))


def test_local_mean_equals_centre_without_missing_pixels():
    # full sphere, no NaN: every stencil value is an exact bilinear interpolation
    a = _conv(cell_ids=None)
    b = _conv(cell_ids=None, missing="local_mean")
    x = torch.randn(3, 12 * 4 ** LEVEL, dtype=torch.float64)
    torch.testing.assert_close(a(x), b(x), rtol=1e-12, atol=1e-12)


def test_nan_is_completed_like_a_pixel_outside_the_domain():
    holes = np.array([5, 40, 41, 77, 100])
    x = torch.randn(len(DOMAIN), dtype=torch.float64)
    x_nan = x.clone()
    x_nan[holes] = float("nan")
    keep = np.setdiff1d(np.arange(len(DOMAIN)), holes)
    y_nan = _conv(missing="local_mean")(x_nan)                  # [G*C_out, N]
    y_out = _conv(cell_ids=DOMAIN[keep], missing="local_mean")(x[keep])
    torch.testing.assert_close(y_nan[:, keep], y_out, rtol=1e-10, atol=1e-10)
    assert torch.isfinite(y_nan).all()


def test_constant_field_with_nan_and_border():
    conv = _conv(missing="local_mean")
    x = torch.full((len(DOMAIN),), 2.5, dtype=torch.float64)
    x[::9] = float("nan")
    y = conv(x)
    w = conv.weight.detach()                                    # [G, 1, C_out, P]
    expected = 2.5 * w.sum(-1).reshape(-1) + conv.bias.detach()
    torch.testing.assert_close(y, expected[:, None].expand_as(y))
    # the default mode lets the NaN through, as before
    assert torch.isnan(_conv()(x)).any()


def test_mask_argument_and_gradients():
    conv = _conv(missing="local_mean")
    x = torch.randn(2, len(DOMAIN), dtype=torch.float64)
    x[0, 3] = float("nan")
    mask = torch.ones(len(DOMAIN), dtype=torch.float64)
    mask[10:20] = 0.0
    xg = x.clone().requires_grad_(True)
    y = conv(xg, mask=mask)
    (y ** 2).sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(xg.grad).all()
    assert xg.grad[0, 3] == 0 and torch.all(xg.grad[:, 10:20] == 0)
    # a masked pixel and a NaN pixel are the same thing
    x2 = x.clone()
    x2[:, 10:20] = float("nan")
    torch.testing.assert_close(conv(x2), y.detach())


def test_completed_stencil_coverage():
    conv = _conv(missing="local_mean")
    z, cov = conv.completed_stencil(torch.ones(len(DOMAIN), dtype=torch.float64))
    assert z.shape == cov.shape == (1, 1, 2, len(DOMAIN), 9)
    assert torch.all(z == 1.0)
    assert torch.all((cov >= 0) & (cov <= 1 + 1e-12))
    assert (cov < 1).any()                                       # border of the domain


def test_local_mean_weights_validation():
    with pytest.raises(ValueError):
        _conv(missing="local_mean", local_mean_weights=np.ones(4))
    with pytest.raises(ValueError):
        _conv(local_mean_weights=np.ones(9))
    with pytest.raises(ValueError):
        _conv(missing="zero")


def test_mask_shapes():
    conv = _conv(missing="local_mean")
    n = len(DOMAIN)
    rng = np.random.default_rng(11)
    x = torch.randn(2, 3, n, dtype=torch.float64)
    m = torch.as_tensor(rng.random((2, 3, n)) > 0.3, dtype=torch.float64)
    # per channel [B, C, N]: each channel uses its own mask
    z, cov = conv.completed_stencil(x, m)
    for b in range(2):
        for c in range(3):
            zc, cc = conv.completed_stencil(x[b, c], m[b, c])
            torch.testing.assert_close(z[b, c], zc[0, 0])
            torch.testing.assert_close(cov[b, c], cc[0, 0])
    # [N] applies to every sample and channel, [B, N] to every channel
    full = m[0, 0].expand(2, 3, n)
    torch.testing.assert_close(conv.completed_stencil(x, m[0, 0])[0],
                               conv.completed_stencil(x, full)[0])
    per_b = m[:, 0]
    torch.testing.assert_close(conv.completed_stencil(x, per_b)[0],
                               conv.completed_stencil(x, per_b[:, None].expand(2, 3, n))[0])
    # B and C may be 1
    torch.testing.assert_close(conv.completed_stencil(x, m[:1, :1])[0],
                               conv.completed_stencil(x, full)[0])
    # a [C, N] mask with C != B, or any other shape, raises
    for bad in (m[0], m[..., :-1], m[None], m[:, :2]):
        with pytest.raises(ValueError, match="mask must have shape"):
            conv.completed_stencil(x, bad)
    with pytest.raises(ValueError, match="mask must have shape"):
        conv(x[:, :1], mask=m[0])                                 # [3, N] for B = 2


def test_layer_pickled_before_missing_still_runs():
    conv = _conv()
    x = torch.randn(len(DOMAIN), dtype=torch.float64)
    y = conv(x)
    del conv.missing                    # as unpickled from an older version
    assert conv.missing == "centre"
    torch.testing.assert_close(conv(x), y)
    assert "missing" not in repr(conv)


def test_cache_paths(tmp_path):
    def load(p):
        return torch.load(p, map_location="cpu", weights_only=True)

    x = torch.randn(2, len(DOMAIN), dtype=torch.float64)
    xn = x.clone()
    xn[0, ::7] = float("nan")
    # every layer builds its geometry in float64 (#109): local_mean, float32
    # and float64 layers share one file, which holds the support
    _conv(cache_dir=tmp_path)
    (lm_file,) = tmp_path.glob("healpixconv_*.pt")
    _conv(cache_dir=tmp_path, missing="local_mean")
    assert set(tmp_path.glob("healpixconv_*.pt")) == {lm_file}
    assert "support" in load(lm_file)
    _conv(cache_dir=tmp_path, dtype=torch.float32)
    _conv(cache_dir=tmp_path, dtype=torch.float32, missing="local_mean")
    assert set(tmp_path.glob("healpixconv_*.pt")) == {lm_file}
    # a local_mean file without the support is recomputed once, and rewritten
    data = load(lm_file)
    del data["support"]
    torch.save(data, lm_file)
    ref = _conv(missing="local_mean")
    a = _conv(cache_dir=tmp_path, missing="local_mean")
    torch.testing.assert_close(a(xn), ref(xn), rtol=1e-12, atol=1e-12)
    assert "support" in load(lm_file)
    # then read back (the cache keeps the float64 weights)
    b = _conv(cache_dir=tmp_path, missing="local_mean")
    torch.testing.assert_close(b(xn), ref(xn), rtol=1e-12, atol=1e-12)
    # the default mode reads its own file
    torch.testing.assert_close(_conv(cache_dir=tmp_path)(x), _conv()(x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("level", [3, 6])
def test_local_mean_float32_matches_float64_at_the_border(level):
    # A float32 geometry gives stencil points just outside the domain
    # spurious valid fractions (5e-4 at level 6 near the pole), which switch
    # them from the local mean to one neighbour: outputs off by up to 4.5
    # here.  The geometry of local_mean is therefore built in float64.
    dom = np.arange(2 * 4 ** level, dtype=np.int64)              # base pixels 0, 1
    gen = torch.Generator().manual_seed(22)
    x = torch.randn(2, len(dom), dtype=torch.float64, generator=gen)
    x[:, ::11] = float("nan")
    w = torch.randn(1, 2, 25, dtype=torch.float64, generator=gen)
    out = []
    for dtype in (torch.float64, torch.float32):
        conv = HealPixConv(level=level, in_channels=1, out_channels=2,
                           kernel_sz=5, n_gauges=4, cell_ids=dom,
                           device="cpu", dtype=dtype, cache_dir=None,
                           missing="local_mean")
        conv.set_kernel(w.numpy())
        out.append(conv(x.to(dtype)).double())
    torch.testing.assert_close(out[1], out[0], rtol=1e-5, atol=1e-5)
