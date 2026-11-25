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


def _conv(cell_ids=DOMAIN, **kw):
    conv = HealPixConv(level=LEVEL, in_channels=1, out_channels=2, kernel_sz=3,
                       n_gauges=2, cell_ids=cell_ids, device="cpu",
                       dtype=torch.float64, cache_dir=None, **kw)
    rng = np.random.default_rng(3)
    conv.set_kernel(rng.standard_normal((1, 2, 9)), bias=[0.1, -0.2, 0.3, 0.0])
    return conv


def test_default_is_centre_and_state_dict_unchanged():
    a, b = _conv(), _conv(missing="centre")
    x = torch.randn(2, len(DOMAIN), dtype=torch.float64)
    assert torch.equal(a(x), b(x))
    assert set(a.state_dict()) == {"weight", "bias", "_sort_order", "_inv_order",
                                   "_pos_safe", "_w_norm"}
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
