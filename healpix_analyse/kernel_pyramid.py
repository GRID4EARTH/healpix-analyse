"""Compact per-band kernels for pyramidal HEALPix convolution.

Theoretical status (read before using)
---------------------------------------
Let ``W`` be the analysis operator of a :class:`~healpix_analyse.decomp.HealPixDecomp`
(the stacked Down/Up chain) and ``S`` its synthesis operator, so that
``S W = I`` exactly (this is what makes :meth:`HealPixDecomp.invert` an exact
inverse of :meth:`HealPixDecomp.compute`).  Convolving a map ``x`` with a
target operator ``K`` and then decomposing, or decomposing and then acting on
the pyramid coefficients, are *not* interchangeable in general.  Write the
"acting on coefficients" operator as ``B_K`` such that the intended pipeline
is

``y = S · B_K · W · x``

which reduces to plain convolution (``y = K x``) only for the specific choice
``B_K = W K S`` -- and that operator is, in general, **dense across bands**:
detail at one scale can leak into neighbouring scales through ``B_K``.

:class:`HealPixKernelPyramid` builds a **block-diagonal approximation** of
``B_K``: one small, compact ``HealPixConv`` kernel acting independently on
each pyramid band.  This is a deliberate simplification, not a mathematical
identity, and it is *not* the same claim as "the pyramid reconstructs
exactly" (that property belongs to ``S W = I`` alone and holds regardless of
which kernel, if any, is inserted).  The approximation error of using a
block-diagonal ``B_K`` in place of the true (block, inter-band-coupled)
operator must be measured against a direct, non-pyramidal reference
convolution -- see :mod:`healpix_analyse.validation` -- and is reported,
band by band and kernel-size by kernel-size, in the accompanying validation
notes rather than assumed to be small.

This module does not claim to reproduce a specific published construction
(e.g. the least-squares boundary-matched kernels of Farbman, Fattal &
Lischinski, "Convolution Pyramids", ACM TOG 2011); it provides analytic
per-band kernels evaluated on the exact stencil geometry used at runtime by
:class:`~healpix_analyse.convol.HealPixConv`, plus an optional, explicitly
approximate least-squares calibration helper against a direct reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

from healpix_analyse.convol import HealPixConv, _local_kernel_grid
from healpix_analyse.decomp import HealPixDecomp

KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""A kernel profile ``fn(rho_pix, phi_rad) -> weight``.

``rho_pix`` is the radial distance from the stencil centre, in *pixel*
units (i.e. ``sqrt(du**2 + dv**2)`` on the ``kernel_sz x kernel_sz`` index
grid, matching :func:`healpix_analyse.convol._local_kernel_grid`'s own
angular spacing ``alpha_pix``); ``phi_rad`` is the azimuth in the fixed
North-Pole stencil frame, in radians. Using pixel units (rather than a
physical angle) keeps a kernel's *shape in pixels* identical at every
pyramid band; its physical footprint then grows automatically from band to
band because the pixel size itself grows.
"""


# ---------------------------------------------------------------------------
# Built-in isotropic / anisotropic kernel families
# ---------------------------------------------------------------------------

def kernel_gaussian(sigma_pix: float) -> KernelFn:
    """Isotropic Gaussian, ``exp(-rho^2 / (2 sigma^2))``. Control/reference case."""
    sigma_pix = float(sigma_pix)

    def _fn(rho, phi):
        return np.exp(-0.5 * (rho / sigma_pix) ** 2)

    return _fn


def kernel_exponential(scale_pix: float) -> KernelFn:
    """Isotropic exponential ("K_exp"), ``exp(-rho / scale)``. Heavier tail than Gaussian."""
    scale_pix = float(scale_pix)

    def _fn(rho, phi):
        return np.exp(-rho / scale_pix)

    return _fn


def kernel_lorentzian(scale_pix: float) -> KernelFn:
    """Isotropic Lorentzian ("K_lor"), ``1 / (1 + (rho/scale)^2)``. Slowly-decaying tail."""
    scale_pix = float(scale_pix)

    def _fn(rho, phi):
        return 1.0 / (1.0 + (rho / scale_pix) ** 2)

    return _fn


def kernel_beta(scale_pix: float, beta: float = 2.0) -> KernelFn:
    """Isotropic generalized-Lorentzian/"Moffat" profile ("K_beta"),
    ``(1 + (rho/scale)^2)^(-beta/2)``. ``beta=2`` reduces to :func:`kernel_lorentzian`;
    larger ``beta`` decays faster (approaches Gaussian-like compactness).
    """
    scale_pix = float(scale_pix)
    beta = float(beta)

    def _fn(rho, phi):
        return (1.0 + (rho / scale_pix) ** 2) ** (-beta / 2.0)

    return _fn


def kernel_anisotropic_gaussian(
    sigma_major_pix: float,
    sigma_minor_pix: float,
    orientation_rad: float = 0.0,
) -> KernelFn:
    """Anisotropic scalar-field Gaussian ``K(rho, phi)``, elongated along
    ``orientation_rad`` (measured from the stencil's local x-axis) with
    principal widths ``sigma_major_pix``/``sigma_minor_pix``.

    Evaluated in the fixed North-Pole stencil frame, this is combined with
    :class:`HealPixConv`'s per-pixel gauge rotation, so the anisotropy is
    expressed consistently in every output pixel's *local tangent frame*
    (the same convention ``HealPixConv`` itself uses) -- not in a single
    global frame that would be meaningless on a sphere.
    """
    sigma_major_pix = float(sigma_major_pix)
    sigma_minor_pix = float(sigma_minor_pix)
    orientation_rad = float(orientation_rad)

    def _fn(rho, phi):
        dphi = phi - orientation_rad
        u = rho * np.cos(dphi) / sigma_major_pix
        v = rho * np.sin(dphi) / sigma_minor_pix
        return np.exp(-0.5 * (u ** 2 + v ** 2))

    return _fn


def _stencil_pixel_polar(kernel_sz: int, nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rho_pix, phi_rad)`` for the exact stencil ``HealPixConv`` uses.

    Recomputed from :func:`_local_kernel_grid` (rather than re-deriving the
    ``kernel_sz x kernel_sz`` index grid independently) so that kernel
    coefficients are guaranteed to land on precisely the same stencil
    positions ``HealPixConv`` will bind data to at runtime, at this band's
    resolution.
    """
    grid = _local_kernel_grid(kernel_sz, nside)  # [P, 3], unit vectors at N. pole
    z = np.clip(grid[:, 2], -1.0, 1.0)
    dtheta = np.arccos(z)
    dphi = np.arctan2(grid[:, 1], grid[:, 0])
    alpha_pix = np.sqrt(4.0 * np.pi / (12.0 * nside ** 2))
    rho_pix = dtheta / alpha_pix
    return rho_pix, dphi


def _block_diagonal_kernel(w: np.ndarray, channels: int) -> np.ndarray:
    """``[P]`` taps -> ``[channels, channels, P]``, zero off the diagonal.

    Every channel gets the identical profile ``w``; there is no cross-channel
    mixing (this is what "block-diagonal across channels" means here).
    """
    P = w.shape[0]
    W = np.zeros((channels, channels, P), dtype=np.float64)
    idx = np.arange(channels)
    W[idx, idx, :] = w[None, :]
    return W


@dataclass
class HealPixKernelPyramid:
    """One fixed, non-learnable :class:`HealPixConv` per pyramid band.

    Built from :meth:`from_kernel` (or :meth:`calibrate`, for an optional
    data-driven refinement). Shares the exact stencil geometry, gauge
    convention, and cell-id domain of a given
    :class:`~healpix_analyse.decomp.HealPixDecomp`, band for band, so that
    :meth:`apply` can be composed directly with
    :meth:`HealPixDecomp.compute`/:meth:`~HealPixDecomp.compute_weighted` and
    :meth:`~HealPixDecomp.invert`.

    This is a **block-diagonal** approximation -- see the module docstring
    (block-diagonal *across pyramid bands*; see also ``channels`` below for
    the unrelated, additional sense in which multi-channel kernels here are
    block-diagonal *across channels*).

    Multiple channels (``channels > 1``, e.g. 3 for RGB) are supported as an
    explicit second, independent block-diagonal structure: one
    ``in_channels=channels, out_channels=channels`` :class:`HealPixConv` per
    band, whose ``[channels, channels, P]`` kernel is zero off the diagonal
    -- i.e. every channel is filtered independently with the *same* profile,
    with no cross-channel mixing. This is deliberately not full
    ``HealPixConv`` generality (a dense ``[C_out, C_in, P]`` kernel, or
    per-channel gauge counts) -- the point is only to filter several
    co-registered scalar fields (e.g. R, G, B) through the same fixed
    kernel pyramid in one call, cheaply and without the shape pitfall
    documented on :meth:`apply`.
    """

    decomp: HealPixDecomp
    convs: tuple[HealPixConv, ...]
    kernel_sz: int
    gauge_type: str
    n_gauges: int
    channels: int = 1

    @classmethod
    def from_kernel(
        cls,
        decomp: HealPixDecomp,
        kernel: KernelFn,
        *,
        compact_kernel_sz: int = 5,
        gauge_type: str = "phi",
        n_gauges: int = 1,
        singularity_lonlat=None,
        ref_direction=None,
        bands: Optional[Sequence[int]] = None,
        channels: int = 1,
        ellipsoid: str = "sphere",
        dtype: Optional[torch.dtype] = None,
        device=None,
    ) -> "HealPixKernelPyramid":
        """Build one fixed ``HealPixConv`` per band from an analytic kernel profile.

        Parameters
        ----------
        decomp : HealPixDecomp
            The pyramid whose bands this kernel pyramid will act on.
        kernel : KernelFn
            ``fn(rho_pix, phi_rad) -> weight``, evaluated once per band on
            that band's own stencil geometry (see :func:`_stencil_pixel_polar`).
            The *same* callable is reused unchanged at every band, so the
            kernel's compact support (in pixel units) is identical band to
            band; its physical footprint grows with the coarsened pixel size.
        compact_kernel_sz : int, default 5
            Odd stencil size, e.g. 5 for a 5x5 stencil (``P = 25`` taps).
        bands : sequence of int, optional
            Restrict construction to these band indices (0 = finest detail,
            ``decomp.n_scales`` = coarse residual). Defaults to every band.
        channels : int, default 1
            Number of independent scalar channels each band's kernel will
            act on (e.g. 3 for RGB), built as an ``in_channels=out_channels=
            channels`` ``HealPixConv`` per band whose kernel is block-diagonal
            across channels -- the same profile applied to every channel with
            no cross-channel mixing (see the class docstring). Requires
            ``n_gauges=1`` (not checked against ``n_gauges`` passed here
            unless ``channels>1``, in which case a value other than 1 raises,
            since the general ``[G, C, C, P]`` case with ``G>1`` cannot be
            reshaped back to a plain ``[channels, N]`` band -- a currently
            unaddressed limitation, not a silent approximation). Pass data
            shaped ``[..., channels, N]`` to :meth:`~healpix_analyse.pyramid_conv.HealPixPyramidConv.forward`
            (see :meth:`apply`).
        ellipsoid : str, default "sphere"
            Geometry passed to every band's ``HealPixConv``. This module
            defaults to the true sphere (see the module docstring, section
            A.5), which is *not* the same convention some real data stores
            use -- e.g. EOPF HEALPix products declare ``"wgs84"``. Pass the
            same ellipsoid the data (and ``decomp``, if it uses one other
            than its own default) actually use, or geometry is silently
            mislabeled. Accepted case-insensitively (canonicalized against
            ``healpix_geo``'s own registry).
        """
        if compact_kernel_sz < 1 or compact_kernel_sz % 2 == 0:
            raise ValueError("compact_kernel_sz must be a positive odd integer")
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be a positive integer")
        if channels > 1 and n_gauges != 1:
            raise ValueError(
                "channels > 1 requires n_gauges == 1 (multi-gauge, "
                "multi-channel kernels are not supported here -- see the "
                "class docstring)"
            )
        dtype = dtype or decomp.dtype
        device = device if device is not None else decomp.device
        band_indices = range(decomp.n_bands) if bands is None else list(bands)

        convs: list[Optional[HealPixConv]] = [None] * decomp.n_bands
        for j in band_indices:
            level_j = decomp.levels[j]
            nside_j = 2 ** level_j
            rho_pix, phi_rad = _stencil_pixel_polar(compact_kernel_sz, nside_j)
            w = np.asarray(kernel(rho_pix, phi_rad), dtype=np.float64)
            if w.shape != rho_pix.shape:
                raise ValueError(
                    "kernel(rho_pix, phi_rad) must return an array the same "
                    f"shape as its inputs ({rho_pix.shape}); got {w.shape}"
                )
            cell_ids = decomp.cell_ids_per_scale[j] if decomp.partial else None
            conv = HealPixConv(
                level=level_j,
                in_channels=channels,
                out_channels=channels,
                kernel_sz=compact_kernel_sz,
                n_gauges=n_gauges,
                gauge_type=gauge_type,
                singularity_lonlat=singularity_lonlat,
                ref_direction=ref_direction,
                cell_ids=cell_ids,
                ellipsoid=ellipsoid,
                dtype=dtype,
                device=device,
            )
            conv.set_kernel(_block_diagonal_kernel(w, channels), requires_grad=False)
            convs[j] = conv

        return cls(
            decomp=decomp,
            convs=tuple(convs),
            kernel_sz=compact_kernel_sz,
            gauge_type=gauge_type,
            n_gauges=n_gauges,
            channels=channels,
        )

    @classmethod
    def calibrate(
        cls,
        decomp: HealPixDecomp,
        reference_fn: Callable[[np.ndarray, int], np.ndarray],
        *,
        compact_kernel_sz: int = 5,
        gauge_type: str = "phi",
        n_probes: int = 128,
        n_excitations: int = 4,
        seed: int = 0,
        ridge: float = 1e-6,
        channels: int = 1,
        ellipsoid: str = "sphere",
        dtype: Optional[torch.dtype] = None,
        device=None,
    ) -> "HealPixKernelPyramid":
        """Fit one kernel per band by least squares against a reference operator.

        This is an explicitly *approximate*, single-band-at-a-time
        calibration: it does **not** account for inter-band coupling (the
        same block-diagonal caveat as :meth:`from_kernel` applies), and its
        accuracy should be checked the same way, against the same
        reference.

        For each band ``j``, this solves for the ``P`` kernel taps ``w``
        that minimise ``sum_i (sum_p w_p * x_interp[i, p] - target[i])^2``
        over a set of probe output pixels ``i`` and ``n_excitations``
        independent random excitation fields, where ``x_interp[i, p]`` is
        the value ``HealPixConv`` itself bilinear-interpolates for tap ``p``
        of pixel ``i`` (obtained by running the *same* ``HealPixConv`` with
        a one-hot kernel, so the fit is against the exact runtime stencil
        geometry) and ``target = reference_fn(...)`` applied to the same
        excitation field. Using a random excitation (rather than probing
        with per-pixel delta inputs placed *at* each probe) is required for
        correctness: for an off-centre tap, a delta exactly at the probe
        pixel does not generally coincide with that tap's rotated,
        bilinear-bound sample location, so per-probe deltas would recover
        the wrong design matrix.

        ``reference_fn(cell_ids, level) -> operator`` must return a callable
        usable as ``operator(x) -> y`` mapping a ``[len(cell_ids)]`` map to
        a ``[len(cell_ids)]`` map -- i.e. it should already be
        restricted/consistent with this decomposition's own geometry
        (masks, units, gauge) for a fair comparison.

        ``channels``, as in :meth:`from_kernel`: the single scalar profile
        fitted per band is replicated block-diagonally across ``channels``
        independent channels (no cross-channel mixing); fitting itself is
        always done against a scalar reference (one profile per channel, not
        one fit per channel), so ``reference_fn`` need not know about
        channels at all.
        """
        channels = int(channels)
        if channels < 1:
            raise ValueError("channels must be a positive integer")
        rng = np.random.default_rng(seed)
        dtype = dtype or decomp.dtype
        device = device if device is not None else decomp.device
        convs: list[Optional[HealPixConv]] = [None] * decomp.n_bands

        for j in range(decomp.n_bands):
            level_j = decomp.levels[j]
            cell_ids = decomp.cell_ids_per_scale[j]
            n = len(cell_ids)
            op = reference_fn(cell_ids, level_j)

            rho_pix, _phi_rad = _stencil_pixel_polar(compact_kernel_sz, 2 ** level_j)
            P = rho_pix.size

            conv = HealPixConv(
                level=level_j, in_channels=1, out_channels=1,
                kernel_sz=compact_kernel_sz, n_gauges=1, gauge_type=gauge_type,
                cell_ids=cell_ids if decomp.partial else None,
                ellipsoid=ellipsoid, dtype=dtype, device=device,
            )

            n_probes_j = min(n_probes, n)
            basis = np.eye(P)
            A_rows = []
            b_rows = []
            for _ in range(max(1, n_excitations)):
                x_exc = rng.standard_normal(n)
                probe_idx = rng.choice(n, size=n_probes_j, replace=False)

                target_full = np.asarray(op(x_exc)).reshape(-1)
                b_rows.append(target_full[probe_idx])

                A_e = np.zeros((n_probes_j, P), dtype=np.float64)
                for tap in range(P):
                    conv.set_kernel(basis[tap][None, None, :].astype(np.float32), requires_grad=False)
                    y_tap = np.asarray(conv(x_exc)).reshape(-1)  # x_interp[:, tap]
                    A_e[:, tap] = y_tap[probe_idx]
                A_rows.append(A_e)

            A = np.concatenate(A_rows, axis=0)
            b = np.concatenate(b_rows, axis=0)
            reg = ridge * np.eye(P)
            w_fit, *_ = np.linalg.lstsq(A.T @ A + reg, A.T @ b, rcond=None)

            if channels == 1:
                conv.set_kernel(w_fit[None, None, :].astype(np.float32), requires_grad=False)
            else:
                # Refit was against a single-channel probing conv (the
                # reference is scalar); rebuild a channels x channels
                # block-diagonal conv to hold the fitted profile.
                conv = HealPixConv(
                    level=level_j, in_channels=channels, out_channels=channels,
                    kernel_sz=compact_kernel_sz, n_gauges=1, gauge_type=gauge_type,
                    cell_ids=cell_ids if decomp.partial else None,
                    ellipsoid=ellipsoid, dtype=dtype, device=device,
                )
                conv.set_kernel(_block_diagonal_kernel(w_fit, channels), requires_grad=False)
            convs[j] = conv

        return cls(
            decomp=decomp, convs=tuple(convs), kernel_sz=compact_kernel_sz,
            gauge_type=gauge_type, n_gauges=1, channels=channels,
        )

    def apply(self, bands: Sequence) -> tuple:
        """Apply each band's fixed kernel to the matching pyramid band.

        With ``channels=1`` (the default), each ``band`` is passed straight
        to its ``HealPixConv`` -- see that class's own ``forward`` for its
        exact input/output shape convention, including a known asymmetry: a
        2-D ``[B, N]`` input always returns a 3-D ``[B, 1, N]`` output, never
        squeezed back to 2-D. This method does not paper over that, so a
        band shaped ``[C, N]`` used as a poor-man's channel batch (e.g. RGB
        packed as ``[3, N]``) comes back ``[3, 1, N]``, not ``[3, N]`` --
        build the kernel pyramid with ``channels=3`` instead (see the class
        docstring) to get real, correctly shaped multi-channel support.

        With ``channels>1``, each ``band`` must be shaped ``[..., channels,
        N]``; this method reshapes it to the ``[B, channels, N]`` its
        ``HealPixConv`` (built with matching ``in_channels=out_channels=
        channels``) actually expects, applies it, and reshapes the result
        back to ``[..., channels, N]`` -- any leading batch dims are
        otherwise untouched.
        """
        if len(bands) != len(self.convs):
            raise ValueError(
                f"Expected {len(self.convs)} bands, got {len(bands)}"
            )
        if self.channels == 1:
            return tuple(conv(band) for conv, band in zip(self.convs, bands))
        return tuple(
            self._apply_multichannel(conv, band)
            for conv, band in zip(self.convs, bands)
        )

    def _apply_multichannel(self, conv: HealPixConv, band):
        is_numpy = isinstance(band, np.ndarray)
        t = torch.as_tensor(band) if is_numpy else band
        if t.ndim < 2 or t.shape[-2] != self.channels:
            raise ValueError(
                f"With channels={self.channels}, each band must have shape "
                f"[..., {self.channels}, N]; got {tuple(t.shape)}"
            )
        n = t.shape[-1]
        batch_shape = tuple(t.shape[:-2])
        b = 1
        for s in batch_shape:
            b *= int(s)
        y3 = conv(t.reshape(b, self.channels, n))  # [b, channels, n] (n_gauges == 1)
        y = y3.reshape(*batch_shape, self.channels, n)
        if is_numpy:
            y = y.detach().cpu().numpy()
        return y


__all__ = [
    "HealPixKernelPyramid",
    "KernelFn",
    "kernel_gaussian",
    "kernel_exponential",
    "kernel_lorentzian",
    "kernel_beta",
    "kernel_anisotropic_gaussian",
]
