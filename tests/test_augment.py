"""Batched augmentation (healpix_analyse.augment, issue #109 section 8).

* The torch HEALPix lookup used on the device matches healpix-geo
  (RING -> NESTED) and ``get_interp_weights`` (cells exactly, weights 1e-12).
* The identity transform is exact (bit for bit, float64 geometry).
* A rotation followed by its inverse (two gathers) brings a smooth field back.
* The full-sphere SO(3) rotation preserves the mean and the power of a smooth
  field and matches the analytic rotation.
* Same seed -> same draw and same output; gradients (gradcheck); Triton
  against the torch reference on CUDA.
"""

from __future__ import annotations

import math

import healpix_geo
import numpy as np
import pytest
import torch

from healpix_analyse import _augment_backend as ab
from healpix_analyse.augment import (
    AdditiveNoise,
    HealPixAugment,
    RandomCrop,
    RandomFlip,
    RandomGaugeRotation,
    RandomMask,
    RandomRotation,
    RandomShift,
    RandomSO3,
    RandomZoom,
    Rotation3D,
    TangentLinear,
    gauge_rotate,
)
from healpix_analyse.healpix_interp import get_interp_weights

needs_triton = pytest.mark.skipif(
    not ab.triton_available(), reason="needs CUDA and the optional triton package"
)

LEVEL = 8
# A regional domain: the 4**6 = 4096 children of one level-2 cell, level 8.
BLOCK = np.arange(4**6, dtype=np.int64) + 37 * 4**6


def _xyz(ids, level):
    lon, lat = healpix_geo.nested.healpix_to_lonlat(np.asarray(ids, dtype=np.uint64), level)
    lon = np.radians(np.asarray(lon))
    lat = np.radians(np.asarray(lat))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def _smooth(xyz):
    """Band-limited field (degree <= 3 in x, y, z, so l <= 3)."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return 0.3 + x * y - 0.5 * z**2 + 0.8 * x * z * y + 0.4 * y - 0.2 * x**3


def _rot2(a):
    return [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]


# ---------------------------------------------------------------------------
# Device lookup against healpix-geo / get_interp_weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2, 3, 5])
def test_ring_to_nest_all_cells(order):
    ring = np.arange(12 * 4**order, dtype=np.uint64)
    ref = np.asarray(healpix_geo.ring.to_nested(ring, order)).astype(np.int64)
    got = ab.ring_to_nest(torch.from_numpy(ring.astype(np.int64)), order).numpy()
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("order", [10, 17, 24])
def test_ring_to_nest_random(order):
    ring = np.random.default_rng(order).integers(0, 12 * 4**order, 20000).astype(np.uint64)
    ref = np.asarray(healpix_geo.ring.to_nested(ring, order)).astype(np.int64)
    got = ab.ring_to_nest(torch.from_numpy(ring.astype(np.int64)), order).numpy()
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("order", [0, 3, 10])
def test_bilinear_matches_get_interp_weights(order):
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 360, 20000)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 20000)))
    lat[:6] = [90, -90, 89.99, -89.99, 41.8103149, -41.8103149]  # poles, cap/belt edge
    pix, w = get_interp_weights(lon, lat, order)
    v = np.stack([np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
                  np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
                  np.sin(np.radians(lat))], -1)
    pg, wg = ab.bilinear_nested(torch.from_numpy(v), order)
    np.testing.assert_array_equal(pg.numpy(), pix.astype(np.int64))
    np.testing.assert_allclose(wg.numpy(), w, atol=1e-11)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_identity_exact_regional(dtype):
    ids = np.random.default_rng(0).permutation(BLOCK)        # any order
    aug = HealPixAugment([RandomRotation((0.0, 0.0)), RandomZoom((1.0, 1.0))], LEVEL, ids)
    x = torch.randn(3, 2, ids.size, dtype=dtype)
    lab = torch.randint(0, 7, (3, ids.size))
    out = aug(x, labels=lab)
    assert torch.equal(out.x, x)
    assert torch.equal(out.labels, lab)
    assert out.valid.all()


def test_identity_exact_full_sphere():
    aug = HealPixAugment([Rotation3D(torch.eye(3))], 4)
    x = torch.randn(2, 12 * 4**4, dtype=torch.float64)
    out = aug(x)
    assert torch.equal(out.x, x)
    assert out.valid.all()


def test_nan_input_propagates_and_identity_keeps_nan():
    aug = HealPixAugment([RandomRotation((0.0, 0.0))], LEVEL, BLOCK)
    x = torch.randn(1, 1, BLOCK.size, dtype=torch.float64)
    x[0, 0, 100] = float("nan")
    out = aug(x)
    assert torch.isnan(out.x[0, 0, 100])
    assert torch.isnan(out.x).sum() == 1


# ---------------------------------------------------------------------------
# Round trips and the full-sphere rotation
# ---------------------------------------------------------------------------

def test_rotation_then_inverse_regional():
    """Rotate by 0.7 rad about the centre, then by -0.7 rad (two gathers).

    The field varies on the scale of the domain (64 cells): two bilinear
    interpolations leave an error of about 6e-5 of the field range on the
    cells that stay inside the domain; the test states 2e-4.  (With
    ``min_support=0.5`` the renormalised border cells reach about 6e-3.)
    """
    f = _smooth(_xyz(BLOCK, LEVEL) * 8.0 - 8.0 * _xyz(BLOCK, LEVEL).mean(0))
    x = torch.as_tensor(f).view(1, 1, -1)
    fwd = HealPixAugment([TangentLinear(_rot2(0.7))], LEVEL, BLOCK)
    inv = HealPixAugment([TangentLinear(_rot2(-0.7))], LEVEL, BLOCK)
    y = fwd(x)
    z = inv(y.x)
    ok = z.valid[0] & ~torch.isnan(z.x[0, 0])
    assert ok.float().mean() > 0.5
    err = (z.x[0, 0][ok] - x[0, 0][ok]).abs().max() / (f.max() - f.min())
    assert err < 2e-4, float(err)


def test_shift_then_inverse_regional():
    f = _smooth(_xyz(BLOCK, LEVEL) * 8.0 - 8.0 * _xyz(BLOCK, LEVEL).mean(0))
    x = torch.as_tensor(f).view(1, 1, -1)
    aug = HealPixAugment([RandomShift((0.03, 0.03))], LEVEL, BLOCK)
    plan = aug.sample(1, torch.Generator().manual_seed(3))
    y = plan.apply(x)
    # The same rotation, transposed: the inverse shift.
    Rm = _plan_rotation(aug, 3)
    back = HealPixAugment([Rotation3D(Rm.T)], LEVEL, BLOCK)(y.x)
    ok = back.valid[0] & ~torch.isnan(back.x[0, 0])
    assert ok.float().mean() > 0.5
    err = (back.x[0, 0][ok] - x[0, 0][ok]).abs().max() / (f.max() - f.min())
    assert err < 2e-4, float(err)


def _plan_rotation(aug, seed):
    t = aug.transforms[0]
    M, _ = t.draw(1, torch.Generator().manual_seed(seed), aug.grid)
    return M[0]


def test_so3_full_sphere_mean_power_and_analytic():
    """SO(3) rotation of an l <= 3 field at level 6.

    Mean and power (mean square) are preserved to 1e-3 (relative), and the
    rotated field matches f(R p) to 2e-3 of the field range.
    """
    level = 6
    xyz = _xyz(np.arange(12 * 4**level), level)
    f = _smooth(xyz)
    aug = HealPixAugment([RandomSO3()], level)
    g = torch.Generator().manual_seed(11)
    plan = aug.sample(4, g)
    out = plan.apply(torch.as_tensor(f).expand(4, -1).clone())
    assert out.valid.all()
    q = plan.params["0:RandomSO3.quaternion"]
    for b in range(4):
        R = _quat_to_mat(q[b].numpy())
        exact = _smooth(xyz @ R.T)
        y = out.x[b].numpy()
        assert abs(y.mean() - f.mean()) < 1e-3 * np.abs(f).mean()
        assert abs((y**2).mean() / (f**2).mean() - 1) < 1e-3
        assert np.abs(y - exact).max() < 2e-3 * (f.max() - f.min())


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_so3_then_inverse_full_sphere():
    """R then R^T at level 6 (two gathers): 2.3e-3 of the field range at
    worst (the HEALPix bilinear interpolation is least accurate near the
    poles); the test states 5e-3."""
    level = 6
    f = _smooth(_xyz(np.arange(12 * 4**level), level))
    R = _quat_to_mat(np.array([0.3, -0.5, 0.6, 0.2]) / np.linalg.norm([0.3, -0.5, 0.6, 0.2]))
    x = torch.as_tensor(f).view(1, -1)
    y = HealPixAugment([Rotation3D(R)], level)(x).x
    z = HealPixAugment([Rotation3D(R.T)], level)(y).x
    assert (z - x).abs().max() < 5e-3 * (f.max() - f.min())


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_fusion_into_one_gather():
    aug = HealPixAugment([RandomShift(0.02), RandomRotation(), RandomZoom(), RandomFlip()], LEVEL, BLOCK)
    assert len(aug.sample(2, torch.Generator().manual_seed(0)).gather_maps) == 1
    aug2 = HealPixAugment([RandomZoom(), RandomShift(0.02)], LEVEL, BLOCK)
    assert len(aug2.sample(2, torch.Generator().manual_seed(0)).gather_maps) == 2


def test_fused_matches_sequential():
    """Rotation then zoom in one gather vs two gathers (smooth field)."""
    f = _smooth(_xyz(BLOCK, LEVEL) * 8.0 - 8.0 * _xyz(BLOCK, LEVEL).mean(0))
    x = torch.as_tensor(f).view(1, 1, -1)
    rz = [[1 / 1.2, 0], [0, 1 / 1.2]]
    fused = HealPixAugment([TangentLinear(_rot2(0.4)), TangentLinear(rz)], LEVEL, BLOCK)(x)
    a = HealPixAugment([TangentLinear(_rot2(0.4))], LEVEL, BLOCK)(x)
    b = HealPixAugment([TangentLinear(rz)], LEVEL, BLOCK)(a.x)
    ok = fused.valid[0] & b.valid[0] & ~torch.isnan(b.x[0, 0])
    err = (fused.x[0, 0][ok] - b.x[0, 0][ok]).abs().max() / (f.max() - f.min())
    assert err < 2e-3


def test_order_of_source_maps():
    """Rotation by a then rotation by b equals rotation by a + b (one gather)."""
    f = _smooth(_xyz(BLOCK, LEVEL) * 8.0 - 8.0 * _xyz(BLOCK, LEVEL).mean(0))
    x = torch.as_tensor(f).view(1, 1, -1)
    two = HealPixAugment([TangentLinear(_rot2(0.3)), TangentLinear(_rot2(0.5))], LEVEL, BLOCK)(x)
    one = HealPixAugment([TangentLinear(_rot2(0.8))], LEVEL, BLOCK)(x)
    torch.testing.assert_close(two.x, one.x, equal_nan=True, rtol=0, atol=1e-12)


@pytest.mark.parametrize("antialias", ["supersample", "pyramid", None])
def test_zoom_out_constant_and_smooth(antialias):
    aug = HealPixAugment([RandomZoom((0.5, 0.5))], LEVEL, BLOCK, antialias=antialias)
    out = aug(torch.full((1, 1, BLOCK.size), 2.5, dtype=torch.float64))
    v = out.x[0, 0][out.valid[0]]
    assert v.numel() > 0.2 * BLOCK.size
    torch.testing.assert_close(v, torch.full_like(v, 2.5), rtol=0, atol=1e-12)


def test_zoom_out_antialias_damps_checkerboard():
    """A cell-scale pattern zoomed out by 4 is averaged away by the filters."""
    x = torch.as_tensor(np.where(BLOCK % 2 == 0, 1.0, -1.0)).view(1, 1, -1)
    res = {}
    for aa in ["supersample", "pyramid", None]:
        out = HealPixAugment([RandomZoom((0.25, 0.25))], LEVEL, BLOCK, antialias=aa)(x)
        res[aa] = out.x[0, 0][out.valid[0]].abs().mean().item()
    assert res["supersample"] < 0.5 * res[None]
    assert res["pyramid"] < 0.5 * res[None]


# ---------------------------------------------------------------------------
# Labels, masks, crops, gauges, noise
# ---------------------------------------------------------------------------

def test_labels_nearest_and_fill():
    aug = HealPixAugment([RandomShift((0.05, 0.05)), RandomRotation()], LEVEL, BLOCK)
    lab = torch.randint(0, 5, (4, BLOCK.size))
    msk = torch.rand(4, BLOCK.size) > 0.5
    out = aug(torch.randn(4, BLOCK.size), labels={"lab": lab, "m": msk},
              generator=torch.Generator().manual_seed(2))
    L, M = out.labels["lab"], out.labels["m"]
    assert L.dtype == lab.dtype and M.dtype == torch.bool
    assert ((L == -1) == ~out.valid).all()
    assert (M[~out.valid] == False).all()  # noqa: E712
    assert set(L[out.valid].unique().tolist()) <= set(range(5))


def test_crop_nested_prefix():
    ids = np.random.default_rng(0).permutation(BLOCK)
    aug = HealPixAugment([RandomCrop(256)], LEVEL, ids)
    x = torch.as_tensor(ids, dtype=torch.float64).view(1, -1).expand(5, -1)
    out = aug(x, generator=torch.Generator().manual_seed(1))
    assert out.x.shape == (5, 256)
    cid = out.cell_ids
    assert ((cid[:, 0] % 256) == 0).all()
    assert (cid - cid[:, :1] == torch.arange(256)).all()
    torch.testing.assert_close(out.x, cid.to(torch.float64))


def test_crop_full_sphere():
    aug = HealPixAugment([RandomCrop(64)], 3)
    x = torch.arange(12 * 64, dtype=torch.float32).view(1, 1, -1).expand(3, 2, -1)
    out = aug(x, generator=torch.Generator().manual_seed(0))
    assert out.x.shape == (3, 2, 64)
    torch.testing.assert_close(out.x[:, 0], out.cell_ids.float())


@pytest.mark.parametrize("block", [1, 16])
def test_mask_ratio_and_mae_indices(block):
    aug = HealPixAugment([RandomMask(0.75, block=block, fill=0.0)], LEVEL, BLOCK)
    x = torch.randn(3, 2, BLOCK.size) + 10.0
    out = aug(x, generator=torch.Generator().manual_seed(4))
    n_tok = BLOCK.size // block
    n_keep = n_tok // 4
    assert (out.mask.sum(1) == (n_tok - n_keep) * block).all()
    assert (out.x[:, 0][out.mask] == 0).all() and (out.x[:, 0][~out.mask] > 5).all()
    tok = out.mask.view(3, n_tok, block)
    assert (tok.all(-1) | ~tok.any(-1)).all()                  # whole tokens
    kept = torch.gather(~tok[..., 0], 1, out.ids_keep)
    assert kept.all() and out.ids_keep.shape == (3, n_keep)
    assert out.ids_restore.shape == (3, n_tok)


def test_gauge_rotate_roundtrip_and_flip():
    G = 4
    x = torch.randn(2, G * 3, 50)
    y = gauge_rotate(x, torch.tensor([1, 3]), G)
    torch.testing.assert_close(y.view(2, G, 3, 50)[0, 1], x.view(2, G, 3, 50)[0, 0])
    torch.testing.assert_close(gauge_rotate(y, torch.tensor([-1, -3]), G), x)
    f = gauge_rotate(x, 0, G, flip=True)
    torch.testing.assert_close(gauge_rotate(f, 0, G, flip=True), x)
    aug = HealPixAugment([RandomGaugeRotation(G, flip=0.5)], LEVEL, BLOCK)
    out = aug(torch.randn(3, G * 2, BLOCK.size))
    assert out.x.shape == (3, G * 2, BLOCK.size)


def test_noise_per_sample_std_and_replay():
    aug = HealPixAugment([AdditiveNoise((0.1, 2.0))], LEVEL, BLOCK)
    plan = aug.sample(4, torch.Generator().manual_seed(5))
    x = torch.zeros(4, 1, BLOCK.size)
    a, b = plan.apply(x).x, plan.apply(x).x
    assert torch.equal(a, b)
    std = plan.params["0:AdditiveNoise.noise_std"]
    torch.testing.assert_close(a.std(-1).squeeze(1).double(), std, rtol=0.05, atol=0)


# ---------------------------------------------------------------------------
# Determinism, gradients, errors
# ---------------------------------------------------------------------------

def _pipeline():
    return [RandomShift(0.03), RandomRotation(), RandomZoom((0.7, 1.3)), RandomFlip(),
            AdditiveNoise(0.05), RandomMask(0.5, block=16)]


def test_determinism_fixed_seed():
    aug = HealPixAugment(_pipeline(), LEVEL, BLOCK)
    x = torch.randn(4, 2, BLOCK.size)
    a = aug(x, generator=torch.Generator().manual_seed(7))
    b = aug(x, generator=torch.Generator().manual_seed(7))
    c = aug(x, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a.x.nan_to_num(123.0), b.x.nan_to_num(123.0))
    assert torch.equal(a.mask, b.mask)
    assert not torch.equal(a.x.nan_to_num(123.0), c.x.nan_to_num(123.0))


def test_per_sample_parameters_differ():
    aug = HealPixAugment([RandomRotation()], LEVEL, BLOCK)
    p = aug.sample(8, torch.Generator().manual_seed(0)).params["0:RandomRotation.angle"]
    assert p.unique().numel() == 8


def test_gradcheck_small():
    ids = np.arange(4**3, dtype=np.int64) + 5 * 4**3
    aug = HealPixAugment([RandomShift((0.02, 0.02)), RandomRotation(), RandomZoom((0.6, 0.6))],
                         5, ids, fill=0.0)
    plan = aug.sample(2, torch.Generator().manual_seed(0))
    x = torch.randn(2, 2, ids.size, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: plan.apply(t).x, (x,))


def test_gradient_of_pipeline_is_finite():
    aug = HealPixAugment(_pipeline()[:4], LEVEL, BLOCK, fill=0.0)
    x = torch.randn(2, 1, BLOCK.size, requires_grad=True)
    aug(x, generator=torch.Generator().manual_seed(0)).x.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_errors():
    with pytest.raises(ValueError):
        HealPixAugment([RandomRotation()], 4)                  # tangent map on the full sphere
    with pytest.raises(ValueError):
        HealPixAugment([RandomCrop(16), RandomRotation()], LEVEL, BLOCK)
    with pytest.raises(ValueError):
        RandomCrop(10)
    aug = HealPixAugment([RandomShift(0.1)], 3)
    with pytest.raises(ValueError):
        aug.sample(2)


# ---------------------------------------------------------------------------
# Triton (CUDA)
# ---------------------------------------------------------------------------

@needs_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("antialias", ["supersample", "pyramid"])
def test_triton_matches_torch(dtype, antialias):
    dev = torch.device("cuda")
    aug = HealPixAugment([RandomShift(0.03), RandomRotation(), RandomZoom((0.5, 1.3)), RandomFlip()],
                         LEVEL, BLOCK, antialias=antialias).to(dev)
    plan = aug.sample(6, torch.Generator().manual_seed(1))
    x = torch.randn(6, 3, BLOCK.size, device=dev, dtype=dtype)
    aug.backend = "triton"
    yt = plan.apply(x).x
    aug.backend = "torch"
    yr = plan.apply(x).x
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(yt, yr, rtol=tol, atol=tol, equal_nan=True)


@needs_triton
@pytest.mark.parametrize("deterministic", [True, False])
def test_triton_backward(deterministic):
    dev = torch.device("cuda")
    aug = HealPixAugment([RandomShift(0.03), RandomRotation(), RandomZoom((0.5, 2.0))],
                         LEVEL, BLOCK, fill=0.0, deterministic=deterministic).to(dev)
    plan = aug.sample(4, torch.Generator().manual_seed(2))
    x = torch.randn(4, 3, BLOCK.size, device=dev, requires_grad=True)
    gy = torch.randn(4, 3, BLOCK.size, device=dev)
    aug.backend = "triton"
    (gt,) = torch.autograd.grad(plan.apply(x).x, x, gy)
    aug.backend = "torch"
    (gr,) = torch.autograd.grad(plan.apply(x).x, x, gy)
    torch.testing.assert_close(gt, gr, rtol=1e-5, atol=1e-5)
    if deterministic:
        aug.backend = "triton"
        (g2,) = torch.autograd.grad(plan.apply(x).x, x, gy)
        assert torch.equal(gt, g2)


@needs_triton
def test_triton_double_backward():
    dev = torch.device("cuda")
    aug = HealPixAugment([RandomRotation()], LEVEL, BLOCK, fill=0.0, backend="triton").to(dev)
    plan = aug.sample(2, torch.Generator().manual_seed(3))
    x = torch.randn(2, 1, BLOCK.size, device=dev, requires_grad=True)
    y = plan.apply(x).x
    (g,) = torch.autograd.grad((y**2).sum(), x, create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), x)
    assert torch.isfinite(gg).all()


@needs_triton
def test_cuda_plan_matches_cpu_plan():
    """The same seed gives the same transformation on the CPU and on CUDA."""
    aug = HealPixAugment([RandomShift(0.03), RandomRotation(), RandomZoom((0.7, 1.3))],
                         LEVEL, BLOCK, fill=0.0, backend="torch")
    x = torch.randn(3, 1, BLOCK.size, dtype=torch.float64)
    y_cpu = aug.sample(3, torch.Generator().manual_seed(9)).apply(x).x
    aug.to("cuda")
    y_gpu = aug.sample(3, torch.Generator().manual_seed(9)).apply(x.cuda()).x
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, rtol=0, atol=1e-9)
