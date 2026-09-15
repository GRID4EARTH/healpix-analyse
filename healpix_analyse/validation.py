"""A direct, non-pyramidal spherical convolution used only as a validation
and calibration oracle for :mod:`healpix_analyse.kernel_pyramid` /
:mod:`healpix_analyse.pyramid_conv`.

This module is deliberately **independent** of :class:`~healpix_analyse.convol.HealPixConv`:
it does not reuse its bilinear stencil-binding or per-pixel gauge-rotation
machinery. Instead it locates true pixel centres with ``healpix_geo``,
searches neighbours with a KD-tree on unit vectors, and evaluates a kernel
against the *exact* great-circle angular distance. This way, comparing the
pyramidal path against this module checks the pyramidal path itself
(including ``HealPixConv``'s own approximations), rather than the pyramidal
method validating itself against a copy of its own machinery.

Scope and honest limitations
-----------------------------
- Only **isotropic** kernel profiles (``kernel(rho_pix, phi) -> weight`` that
  does not depend on ``phi``) are supported here. Reproducing an independent
  oracle for the anisotropic, gauge-dependent case would require
  re-implementing ``HealPixConv``'s own rotation-matrix conventions, which
  would no longer be an independent check; anisotropic kernels are validated
  qualitatively instead (see the validation notes), not against this module.
- This is a plain Python/NumPy implementation with an explicit neighbour
  loop, intended for correctness on the modest map sizes used in tests and
  calibration -- not a fast or GPU-capable production path. It intentionally
  trades speed for being simple enough to audit by eye.
- Distances are true geodesic (great-circle) angles on the given
  ``ellipsoid`` (use ``"sphere"`` to match :mod:`healpix_analyse.kernel_pyramid`'s
  own default), converted to the same "pixel units" convention
  (``rho_pix = rho_rad / alpha_pix``) used throughout
  :mod:`healpix_analyse.kernel_pyramid`, so the *same* kernel callables can be
  passed to both without rescaling.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import healpix_geo
from scipy.spatial import cKDTree

IsoKernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def smooth_test_field(cell_ids: np.ndarray, level: int, ellipsoid: str = "sphere") -> np.ndarray:
    """A fixed, spatially-smooth (low-degree, non-bandlimited-exactly-zero
    but slowly varying) synthetic map used as a validation test field.

    Not a substitute for white-noise/random tests -- see the module
    docstring: HealPixConv's stencil binds data by *interpolation*, which
    closely reproduces a continuous kernel's action on smooth content but
    measurably departs from a nearest-pixel quadrature reference on
    single-pixel-scale (white-noise) content. Both regimes are real and are
    reported separately rather than folded into one number.
    """
    lon_deg, lat_deg = healpix_geo.nested.healpix_to_lonlat(
        np.asarray(cell_ids, dtype=np.int64).tolist(), level, ellipsoid=ellipsoid
    )
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.sin(3.0 * lon) * np.cos(2.0 * lat) + 0.5 * np.sin(lat)


def _pixel_unit_vectors(cell_ids: np.ndarray, level: int, ellipsoid: str) -> np.ndarray:
    lon_deg, lat_deg = healpix_geo.nested.healpix_to_lonlat(
        cell_ids.tolist(), level, ellipsoid=ellipsoid
    )
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def direct_spherical_convolution(
    x: np.ndarray,
    cell_ids: np.ndarray,
    level: int,
    kernel: IsoKernelFn,
    *,
    weights: Optional[np.ndarray] = None,
    kernel_sz: int = 5,
    ellipsoid: str = "sphere",
    eps: float = 1e-8,
    restore_mask: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Direct (brute-force, non-pyramidal) spherical convolution.

    For every pixel ``i``, sums ``kernel(rho_pix, 0) * m_j * x_j`` over
    neighbours ``j`` within the kernel's compact support (matching
    ``compact_kernel_sz`` in pixel units).

    ``normalize=True`` (default) additionally divides by the summed weight
    ``sum(kernel * m)`` -- the normalized-convolution convention used by
    :class:`~healpix_analyse.pyramid_conv.HealPixPyramidConv` in its
    ``"normalized"`` mode, appropriate when comparing against a *masked*
    result. ``normalize=False`` returns the **raw** weighted sum with no
    division, matching :class:`~healpix_analyse.convol.HealPixConv`'s own
    literal tap-weighted-sum semantics (``HealPixConv`` never normalizes its
    kernel internally) -- use this when validating a single
    :class:`~healpix_analyse.kernel_pyramid.HealPixKernelPyramid` band
    directly, or when calibrating raw kernel taps, so both sides of the
    comparison mean the same operator. ``weights`` defaults to a 0/1 mask
    from ``isfinite(x)``.

    Returns a ``[len(cell_ids)]`` array. When ``normalize=True``, pixels
    with no supported neighbours (summed weight below ``eps``) are NaN if
    ``restore_mask`` else 0; ``normalize=False`` never produces NaN/fill
    values on its own (missing values are simply excluded via ``m``).
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.shape[0] != cell_ids.shape[0]:
        raise ValueError("x and cell_ids must have the same length")

    nside = 2 ** int(level)
    alpha_pix = np.sqrt(4.0 * np.pi / (12.0 * nside ** 2))
    max_rho_pix = (kernel_sz // 2) * np.sqrt(2.0) + 0.5
    radius_rad = min(max_rho_pix * alpha_pix, np.pi)

    vecs = _pixel_unit_vectors(cell_ids, level, ellipsoid)
    tree = cKDTree(vecs)
    chord = 2.0 * np.sin(radius_rad / 2.0)
    neighbour_lists = tree.query_ball_point(vecs, r=chord)

    if weights is None:
        finite = np.isfinite(x)
        m = finite.astype(np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != cell_ids.shape[0]:
            raise ValueError("weights and cell_ids must have the same length")
        finite = np.isfinite(x) & np.isfinite(w)
        m = np.where(finite, w, 0.0)
    x_safe = np.where(finite, x, 0.0)

    fill = float("nan") if (restore_mask and normalize) else 0.0
    y = np.full(cell_ids.shape[0], fill, dtype=np.float64)
    for i, nbrs in enumerate(neighbour_lists):
        nbrs = np.asarray(nbrs, dtype=np.int64)
        cosang = np.clip(vecs[nbrs] @ vecs[i], -1.0, 1.0)
        rho_rad = np.arccos(cosang)
        rho_pix = rho_rad / alpha_pix
        w_k = np.asarray(kernel(rho_pix, np.zeros_like(rho_pix)), dtype=np.float64)
        wm = w_k * m[nbrs]
        num = float(np.dot(wm, x_safe[nbrs]))
        if not normalize:
            y[i] = num
            continue
        den = wm.sum()
        if den > eps:
            y[i] = num / den
    return y


def direct_reference_operator_factory(
    kernel: IsoKernelFn,
    *,
    kernel_sz: int = 5,
    ellipsoid: str = "sphere",
    weights: Optional[np.ndarray] = None,
    normalize: bool = False,
):
    """Build a ``reference_fn(cell_ids, level) -> operator`` usable with
    :meth:`healpix_analyse.kernel_pyramid.HealPixKernelPyramid.calibrate`,
    wrapping :func:`direct_spherical_convolution` with a fixed kernel.

    ``normalize=False`` (default) matches ``HealPixConv``'s raw,
    unnormalized tap-weighted-sum convention, which is what
    :meth:`~healpix_analyse.kernel_pyramid.HealPixKernelPyramid.calibrate`
    fits its raw kernel taps against; pass ``normalize=True`` only if the
    calibration target is itself meant to be a normalized operator.
    """

    def reference_fn(cell_ids, level):
        def operator(x):
            return direct_spherical_convolution(
                x, cell_ids, level, kernel,
                weights=weights, kernel_sz=kernel_sz, ellipsoid=ellipsoid,
                restore_mask=False, normalize=normalize,
            )

        return operator

    return reference_fn


__all__ = [
    "direct_spherical_convolution",
    "direct_reference_operator_factory",
    "smooth_test_field",
]
