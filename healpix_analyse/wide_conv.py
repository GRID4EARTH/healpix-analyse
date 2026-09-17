"""Convolution by a *wide* kernel, given as an image, via a scale pyramid.

Motivation
----------
:class:`~healpix_analyse.convol.HealPixConv` convolves with a compact
stencil: a ``kernel_sz x kernel_sz`` neighbourhood, so its reach is a couple
of pixels. A kernel that is genuinely wide -- an exponential with a 10-pixel
scale, a Lorentzian, anything with a long tail -- cannot be written as one
compact stencil at all; the stencil would have to be as wide as the kernel,
which is exactly what nobody wants to pay for.

The way out is the "convolution pyramids" idea (Farbman, Fattal &
Lischinski, ACM TOG 2011): decompose the signal into a Laplacian pyramid,
convolve each band with its *own small* kernel, and synthesize. A wide
kernel is wide only in the finest band's units; at band ``j`` the pixels are
``2**j`` times larger, so the same physical width is ``2**j`` times fewer
pixels. Past a few stages, a very wide kernel is a handful of pixels across.

What this module adds
---------------------
:class:`HealPixWideConv` is the single-call front end for that:

    kernel_image = ...                      # (2n+1, 2n+1) at `level`
    conv = HealPixWideConv(kernel_image, level, Jmax=6)
    y = conv(x, cell_ids)                   # x: [N] or [..., N] -> same shape

You hand it the kernel as an **image** at one level -- the thing you can
plot and reason about -- and it works out, by itself, the small per-band
kernels that reproduce it through the pyramid. There is no per-band kernel
to write by hand and no pyramid bookkeeping in the caller.

How the per-band kernels are obtained
-------------------------------------
Not by sampling ``band_j(K)`` around the centre: the input to band ``j`` is
``band_j(delta)``, a small Laplacian bump rather than a Dirac, so convolving
it by ``band_j(K)``'s own values does not give ``band_j(K)`` back (measured:
~99% wrong). They are *fitted*: for each band, the ``compact_kernel_sz**2``
taps ``w_j`` that best satisfy

    band_j(delta)  (*)  w_j   ~=   band_j(K)

in the least-squares sense, probed through the real
:class:`~healpix_analyse.convol.HealPixConv` so the stencil geometry used to
fit is exactly the one used to apply. Since synthesis is the exact inverse
of analysis (``S W = I``), if every band's fit were exact the result would
be ``K`` exactly -- so the per-band residuals, available from
:meth:`HealPixWideConv.fit_residuals`, are the whole error budget of the
method.

The fit needs a domain, so it happens on the **first call**, on that call's
own ``cell_ids`` (and is cached per domain afterwards). That also keeps it
honest about HEALPix geometry: pixel shape depends on latitude, and fitting
on the data's own patch beats fitting on a reference patch somewhere else.

Scope
-----
Deliberately single-channel and mask-free: one kernel, applied independently
to every leading dimension of ``x``. Channel mixing (``[C_in, C_out, P]``
kernels) and NaN/weight-aware filtering live in
:class:`~healpix_analyse.kernel_pyramid.HealPixKernelPyramid` /
:class:`~healpix_analyse.pyramid_conv.HealPixPyramidConv`; folding them in
here would complicate the one thing this class is for.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch

import healpix_geo.nested as hgn

from healpix_analyse.convol import HealPixConv
from healpix_analyse.decomp import HealPixDecomp

__all__ = ["HealPixWideConv"]


def _as_numpy(t):
    """Torch tensor (possibly on GPU) or array-like -> numpy array."""
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _domain_key(cell_ids: np.ndarray) -> str:
    return hashlib.blake2b(
        np.ascontiguousarray(cell_ids, dtype=np.int64).tobytes(), digest_size=16
    ).hexdigest()


class HealPixWideConv:
    """Convolve by a kernel given as a ``(2n+1, 2n+1)`` image, via a pyramid.

    Parameters
    ----------
    kernel_image : array-like, shape ``(2n+1, 2n+1)``
        The kernel, as an image on the HEALPix base-face integer lattice at
        ``level``: ``kernel_image[a, b]`` is the weight at offset
        ``(di, dj) = (b - n, a - n)`` from the centre, where ``i``/``j`` are
        the base-cell-local coordinates of
        :func:`healpix_geo.nested.healpix_to_base_cell_coordinates` (so rows
        are ``j``, columns are ``i`` -- the same convention as unfolding a
        NESTED tile into a square image). Must be square with an odd side.
    level : int
        The HEALPix level ``kernel_image`` is sampled at. ``cell_ids`` passed
        to :meth:`~HealPixWideConv.__call__` must be at this level.
    Jmax : int, default 6
        Number of pyramid stages. The kernel's reach grows roughly like
        ``compact_kernel_sz//2 * 2**Jmax`` finest-band pixels, so this is the
        knob that has to be large enough for the kernel you are asking for.
    compact_kernel_sz : int, default 5
        Odd size of each band's fitted stencil.
    gauge_type, ellipsoid
        Passed through to each band's :class:`HealPixConv`.
    ridge : float, default 1e-12
        Tikhonov regularization on the per-band least-squares fit.
    dtype : torch.dtype, default ``torch.float64``
        Working precision. ``float64`` is recommended: the fit is a small
        normal-equations solve and the coarse bands' taps are large.
    device : optional
        Torch device for the pyramid and the convolutions.

    Notes
    -----
    Not mask-aware: NaN in ``x`` propagates. Use
    :class:`~healpix_analyse.pyramid_conv.HealPixPyramidConv` in
    ``mode="normalized"`` for hole-filling on real, gappy data.

    Examples
    --------
    >>> import numpy as np, healpix_geo.nested as hgn
    >>> from healpix_analyse.wide_conv import HealPixWideConv
    >>> level, n = 10, 8
    >>> di, dj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    >>> kernel_image = np.exp(-np.hypot(di, dj) / 3.0)          # 17x17
    >>> conv = HealPixWideConv(kernel_image, level, Jmax=3)     # doctest: +SKIP
    >>> y = conv(x, cell_ids)                                   # doctest: +SKIP
    """

    def __init__(
        self,
        kernel_image,
        level: int,
        *,
        Jmax: int = 6,
        compact_kernel_sz: int = 5,
        gauge_type: str = "phi",
        ellipsoid: str = "sphere",
        ridge: float = 1e-12,
        dtype: torch.dtype = torch.float64,
        device=None,
    ):
        k = np.asarray(_as_numpy(kernel_image), dtype=np.float64)
        if k.ndim != 2 or k.shape[0] != k.shape[1]:
            raise ValueError(f"kernel_image must be square 2-D; got shape {k.shape}")
        if k.shape[0] % 2 == 0:
            raise ValueError(f"kernel_image side must be odd (2n+1); got {k.shape[0]}")
        if compact_kernel_sz < 1 or compact_kernel_sz % 2 == 0:
            raise ValueError("compact_kernel_sz must be a positive odd integer")

        self.kernel_image = k
        self.n = k.shape[0] // 2
        self.level = int(level)
        self.Jmax = int(Jmax)
        self.compact_kernel_sz = int(compact_kernel_sz)
        self.gauge_type = gauge_type
        self.ellipsoid = ellipsoid
        self.ridge = float(ridge)
        self.dtype = dtype
        self.device = device

        self._cache: dict = {}        # domain key -> fitted state
        self._last_key: Optional[str] = None

    # -- alternative ways of specifying the kernel ------------------------

    @staticmethod
    def lattice_offsets_m(level: int, lon: float, lat: float, n: int, *,
                          r_earth: float = 6371008.8):
        """Metric offsets of the ``(2n+1, 2n+1)`` lattice around ``(lon, lat)``.

        Returns ``(x_m, y_m)``, each ``(2n+1, 2n+1)``: the position of every
        lattice cell in a local east/north frame centred on the cell that
        contains ``(lon, lat)``, in metres, using an azimuthal-equidistant
        projection (so ``hypot(x_m, y_m)`` is the exact great-circle distance).

        This is where the HEALPix deformation becomes visible and usable: the
        ``(i, j)`` lattice is **not** a square metric grid -- its axes are
        neither orthogonal nor equally scaled, and how much they are sheared
        depends on latitude. Anything defined in metres (a function of
        ``x, y``, a raster with a pixel size in metres) must be evaluated at
        these positions rather than indexed by ``(i, j)``.

        Parameters
        ----------
        level : int
        lon, lat : float
            Reference point, degrees. The lattice is centred on the cell
            containing it.
        n : int
            Half-size; the returned arrays are ``(2n+1, 2n+1)``.
        r_earth : float, default 6371008.8
            Sphere radius used to turn angles into metres.
        """
        centre = int(np.asarray(hgn.lonlat_to_healpix([float(lon)], [float(lat)], level))[0])
        face, i_c, j_c = [
            int(v[0]) for v in hgn.healpix_to_base_cell_coordinates([centre], level)
        ]
        side = 2 ** level
        d = np.arange(-n, n + 1)
        dj, di = np.meshgrid(d, d, indexing="ij")           # rows = j, cols = i
        ti, tj = (i_c + di).ravel(), (j_c + dj).ravel()
        on_face = (ti >= 0) & (ti < side) & (tj >= 0) & (tj < side)
        if not on_face.all():
            raise ValueError(
                f"the {2*n+1}x{2*n+1} lattice around (lon={lon}, lat={lat}) runs off base "
                f"face {face} (centre at i={i_c}, j={j_c} of {side}); use a smaller n or a "
                "reference point further from the face edge"
            )
        cells = hgn.base_cell_coordinates_to_healpix(
            np.full(ti.size, face), ti, tj, level
        ).astype(np.int64)

        clon, clat = [float(v[0]) for v in hgn.healpix_to_lonlat([centre], level)]
        plon, plat = [np.asarray(v) for v in hgn.healpix_to_lonlat(cells.tolist(), level)]
        lo, la = np.radians(plon), np.radians(plat)
        lo0, la0 = np.radians(clon), np.radians(clat)

        v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        east = np.array([-np.sin(lo0), np.cos(lo0), 0.0])
        north = np.array([-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)])
        up = np.array([np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)])

        e, nn, u = east @ v, north @ v, up @ v
        rho = r_earth * np.arccos(np.clip(u, -1.0, 1.0))     # exact great-circle distance
        norm = np.hypot(e, nn)
        scale = np.divide(rho, norm, out=np.zeros_like(rho), where=norm > 0)
        shape = (2 * n + 1, 2 * n + 1)
        return (e * scale).reshape(shape), (nn * scale).reshape(shape)

    @classmethod
    def from_function(cls, fn, level: int, n: int, lon: float, lat: float, *,
                      r_earth: float = 6371008.8, **kwargs) -> "HealPixWideConv":
        """Build from a kernel defined as ``fn(x_m, y_m)``, metres.

        ``fn`` is evaluated at each lattice cell's true metric offset from the
        centre (see :meth:`lattice_offsets_m`), so a kernel written in metres
        keeps its intended physical shape even though the ``(i, j)`` lattice
        it lands on is sheared.

        Parameters
        ----------
        fn : callable
            ``fn(x_m, y_m) -> weight``, applied element-wise to two
            ``(2n+1, 2n+1)`` arrays (east and north offsets, metres).
        level, n, lon, lat
            As for :meth:`lattice_offsets_m`.
        **kwargs
            Passed to the constructor (``Jmax``, ``compact_kernel_sz``, ...).

        Examples
        --------
        >>> conv = HealPixWideConv.from_function(                  # doctest: +SKIP
        ...     lambda x, y: np.exp(-np.hypot(x, y) / 500.0),      # 500 m e-folding
        ...     level=17, n=64, lon=2.3198, lat=48.8704, Jmax=6)
        """
        x_m, y_m = cls.lattice_offsets_m(level, lon, lat, n, r_earth=r_earth)
        kernel_image = np.asarray(fn(x_m, y_m), dtype=np.float64)
        if kernel_image.shape != x_m.shape:
            raise ValueError(
                f"fn must return an array shaped like its inputs {x_m.shape}; "
                f"got {kernel_image.shape}"
            )
        return cls(kernel_image, level, **kwargs)

    @classmethod
    def from_radial(cls, fn, level: int, n: int, lon: float, lat: float, *,
                    r_earth: float = 6371008.8, **kwargs) -> "HealPixWideConv":
        """Build from an isotropic kernel given as ``fn(r_m)``, metres.

        The common case: the kernel depends only on the distance to its own
        centre. ``r`` is the exact great-circle distance in metres from the
        centre cell to each lattice cell (see :meth:`lattice_offsets_m`), so
        the kernel is isotropic **on the ground**, not in pixel indices --
        which in the polar caps is a materially different object, since one
        step along ``i`` and one along ``j`` are not the same number of metres
        there.

        Parameters
        ----------
        fn : callable
            ``fn(r_m) -> weight``, applied element-wise to a ``(2n+1, 2n+1)``
            array of distances in metres.
        level, n, lon, lat
            As for :meth:`lattice_offsets_m`.
        **kwargs
            Passed to the constructor (``Jmax``, ``compact_kernel_sz``, ...).

        Examples
        --------
        >>> conv = HealPixWideConv.from_radial(                    # doctest: +SKIP
        ...     lambda r: np.exp(-r / 500.0),                      # 500 m e-folding
        ...     level=17, n=64, lon=2.3198, lat=48.8704, Jmax=6)
        """
        x_m, y_m = cls.lattice_offsets_m(level, lon, lat, n, r_earth=r_earth)
        r_m = np.hypot(x_m, y_m)
        kernel_image = np.asarray(fn(r_m), dtype=np.float64)
        if kernel_image.shape != r_m.shape:
            raise ValueError(
                f"fn must return an array shaped like its input {r_m.shape}; "
                f"got {kernel_image.shape}"
            )
        return cls(kernel_image, level, **kwargs)

    @classmethod
    def from_grid(cls, grid, pixel_size_m, level: int, n: int, lon: float, lat: float, *,
                  fill: float = 0.0, r_earth: float = 6371008.8,
                  **kwargs) -> "HealPixWideConv":
        """Build from a kernel given as a **square raster in metres**.

        ``grid`` is a regular raster whose pixels are ``pixel_size_m`` metres
        apart, centred on its own middle pixel. It is **bilinearly
        interpolated** onto the lattice cells' true metric positions (see
        :meth:`lattice_offsets_m`), which is what accounts for the HEALPix
        deformation: dropping such a raster straight onto ``(i, j)`` indices
        would silently shear it, because the lattice is not a square metric
        grid.

        Parameters
        ----------
        grid : array-like, 2-D
            The kernel raster, ``grid[b, a]`` at ``(x, y) = ((a - cx) * sx,
            (b - cy) * sy)`` metres from its centre. Any shape; odd sides put
            the centre exactly on a pixel.
        pixel_size_m : float or (float, float)
            Raster spacing in metres, ``sx`` or ``(sx, sy)``.
        level, n, lon, lat
            As for :meth:`lattice_offsets_m`. ``n`` sets the size of the
            HEALPix-side kernel image the raster is resampled onto; make it
            large enough to cover the raster's own extent.
        fill : float, default 0.0
            Value for lattice cells falling outside the raster.
        """
        g = np.asarray(_as_numpy(grid), dtype=np.float64)
        if g.ndim != 2:
            raise ValueError(f"grid must be 2-D; got shape {g.shape}")
        sx, sy = (pixel_size_m, pixel_size_m) if np.isscalar(pixel_size_m) else pixel_size_m
        sx, sy = float(sx), float(sy)
        if sx <= 0 or sy <= 0:
            raise ValueError("pixel_size_m must be positive")

        x_m, y_m = cls.lattice_offsets_m(level, lon, lat, n, r_earth=r_earth)
        ny, nx = g.shape
        # fractional pixel coordinates of each query point in the raster
        fx = x_m / sx + (nx - 1) / 2.0
        fy = y_m / sy + (ny - 1) / 2.0

        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        tx, ty = fx - x0, fy - y0
        inside = (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny)
        xi = np.clip(x0, 0, nx - 2)
        yi = np.clip(y0, 0, ny - 2)

        kernel_image = (
            g[yi, xi] * (1 - tx) * (1 - ty)
            + g[yi, xi + 1] * tx * (1 - ty)
            + g[yi + 1, xi] * (1 - tx) * ty
            + g[yi + 1, xi + 1] * tx * ty
        )
        kernel_image = np.where(inside, kernel_image, fill)
        if not inside.all():
            import warnings
            warnings.warn(
                f"{int((~inside).sum())} of {inside.size} lattice cells fall outside the "
                f"raster ({nx}x{ny} at {sx:.4g}x{sy:.4g} m, i.e. "
                f"{nx * sx / 1000:.3g}x{ny * sy / 1000:.3g} km) and were set to {fill}. "
                "Use a smaller n, or a raster covering more ground.",
                RuntimeWarning, stacklevel=2,
            )
        return cls(kernel_image, level, **kwargs)

    # -- public -----------------------------------------------------------

    def __call__(self, x, cell_ids):
        """Convolve ``x`` over ``cell_ids``.

        Parameters
        ----------
        x : array-like, shape ``[N]`` or ``[..., N]``
            Values on ``cell_ids``. Numpy arrays and torch tensors (CPU or
            GPU) are both accepted.
        cell_ids : array-like of int, shape ``[N]``
            HEALPix NESTED cell ids at ``self.level``.

        Returns
        -------
        y : same shape as ``x``
            Numpy in, numpy out; torch in, torch out on the input's own
            device.
        """
        cell_ids = np.ascontiguousarray(_as_numpy(cell_ids), dtype=np.int64)
        state = self._prepare(cell_ids)
        decomp = state["decomp"]
        convs = state["convs"]

        was_torch = torch.is_tensor(x)
        in_device = x.device if was_torch else None
        in_dtype = x.dtype if was_torch else None

        t = torch.as_tensor(_as_numpy(x), dtype=self.dtype, device=self.device)
        if t.shape[-1] != cell_ids.size:
            raise ValueError(
                f"x has {t.shape[-1]} pixels on its last axis but cell_ids has "
                f"{cell_ids.size}"
            )
        leading = t.shape[:-1]
        flat = t.reshape(-1, cell_ids.size)          # [B, N]

        bands = decomp.compute(flat).bands
        filtered = []
        for j, band in enumerate(bands):
            out = convs[j](band)                      # [B, N_j] -> [B, 1, N_j]
            out = torch.as_tensor(out, dtype=self.dtype, device=self.device)
            filtered.append(out.reshape(flat.shape[0], -1))
        y = decomp.invert(filtered)
        y = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        y = y.reshape(*leading, cell_ids.size)

        if was_torch:
            return y.to(device=in_device, dtype=in_dtype)
        return y.detach().cpu().numpy()

    def kernel_as_field(self, cell_ids, centre_cell: Optional[int] = None) -> np.ndarray:
        """The kernel image laid onto ``cell_ids``, centred on one cell.

        This is the reference a convolution must reproduce: convolving a
        Dirac placed on ``centre_cell`` with this operator should give back
        this field. Useful for validation -- and for getting the centring
        right, since a one-pixel offset between the Dirac and the reference
        is easily mistaken for a much larger error than it is.

        Parameters
        ----------
        cell_ids : array-like of int, shape ``[N]``
        centre_cell : int, optional
            Where to centre the kernel. Defaults to the same cell the fit
            uses (the domain's own centroid, :meth:`reference_centre`).
        """
        cell_ids = np.ascontiguousarray(_as_numpy(cell_ids), dtype=np.int64)
        if centre_cell is None:
            centre_cell = self._centre_cell(cell_ids)
        return self._kernel_field(cell_ids, int(centre_cell))

    def reference_centre(self, cell_ids) -> int:
        """The cell the per-band fit is centred on (the domain's centroid)."""
        return self._centre_cell(
            np.ascontiguousarray(_as_numpy(cell_ids), dtype=np.int64)
        )

    def band_kernels(self, cell_ids=None) -> tuple:
        """The fitted per-band stencils, each ``(compact_kernel_sz,) * 2``."""
        state = self._state(cell_ids)
        s = self.compact_kernel_sz
        return tuple(w.reshape(s, s) for w in state["weights"])

    def fit_residuals(self, cell_ids=None) -> tuple:
        """Per-band relative residual of the fit -- the method's error budget.

        ``||band_j(delta) (*) w_j - band_j(K)|| / ||band_j(K)||``, fine band
        first. If these were all zero the convolution would reproduce the
        kernel image exactly.
        """
        return tuple(self._state(cell_ids)["residuals"])

    @property
    def decomp(self) -> HealPixDecomp:
        """The pyramid built for the most recent domain."""
        return self._state(None)["decomp"]

    # -- internals --------------------------------------------------------

    def _state(self, cell_ids):
        if cell_ids is not None:
            return self._prepare(
                np.ascontiguousarray(_as_numpy(cell_ids), dtype=np.int64)
            )
        if self._last_key is None:
            raise RuntimeError(
                "nothing fitted yet -- call the operator once (or pass cell_ids)"
            )
        return self._cache[self._last_key]

    def _prepare(self, cell_ids: np.ndarray) -> dict:
        key = _domain_key(cell_ids)
        self._last_key = key
        if key in self._cache:
            return self._cache[key]

        decomp = HealPixDecomp(
            level=self.level, cell_ids=cell_ids, Jmax=self.Jmax,
            ellipsoid=self.ellipsoid, dtype=self.dtype, device=self.device,
        )
        centre_cell = self._centre_cell(cell_ids)
        kernel_field = self._kernel_field(cell_ids, centre_cell)

        dirac = np.zeros(cell_ids.size, dtype=np.float64)
        dirac[np.searchsorted(cell_ids, centre_cell)] = 1.0

        pyr_d = decomp.compute(torch.as_tensor(dirac, dtype=self.dtype, device=self.device))
        pyr_K = decomp.compute(
            torch.as_tensor(kernel_field, dtype=self.dtype, device=self.device)
        )

        P = self.compact_kernel_sz ** 2
        eye = np.eye(P)
        convs, weights, residuals = [], [], []
        for j in range(decomp.n_bands):
            ids_j = decomp.cell_ids_per_scale[j]
            conv = HealPixConv(
                level=decomp.levels[j], in_channels=1, out_channels=1,
                kernel_sz=self.compact_kernel_sz, n_gauges=1,
                gauge_type=self.gauge_type, cell_ids=ids_j,
                ellipsoid=self.ellipsoid, dtype=self.dtype, device=self.device,
            )
            b_j = pyr_d.bands[j]
            k_j = _as_numpy(pyr_K.bands[j]).reshape(-1)

            A = np.empty((k_j.size, P), dtype=np.float64)
            for p in range(P):
                conv.set_kernel(eye[p][None, None, :], requires_grad=False)
                A[:, p] = _as_numpy(conv(b_j)).reshape(-1)

            w, *_ = np.linalg.lstsq(
                A.T @ A + self.ridge * np.eye(P), A.T @ k_j, rcond=None
            )
            residuals.append(
                float(np.linalg.norm(A @ w - k_j) / max(np.linalg.norm(k_j), 1e-300))
            )
            conv.set_kernel(w[None, None, :], requires_grad=False)
            convs.append(conv)
            weights.append(w)

        state = {
            "decomp": decomp, "convs": convs, "weights": weights,
            "residuals": residuals, "centre_cell": centre_cell,
            "kernel_field": kernel_field,
        }
        self._cache[key] = state
        return state

    def _centre_cell(self, cell_ids: np.ndarray) -> int:
        """The domain cell closest to its own centroid (wraparound-safe)."""
        lon, lat = [np.asarray(v) for v in hgn.healpix_to_lonlat(cell_ids.tolist(), self.level)]
        lon_r, lat_r = np.radians(lon), np.radians(lat)
        # average unit vectors: immune to the lon=+/-180 discontinuity
        vx = np.mean(np.cos(lat_r) * np.cos(lon_r))
        vy = np.mean(np.cos(lat_r) * np.sin(lon_r))
        vz = np.mean(np.sin(lat_r))
        lon0 = np.degrees(np.arctan2(vy, vx))
        lat0 = np.degrees(np.arctan2(vz, np.hypot(vx, vy)))
        l1, p1, l2, p2 = lon_r, lat_r, np.radians(lon0), np.radians(lat0)
        hav = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
        return int(cell_ids[int(np.argmin(hav))])

    def _kernel_field(self, cell_ids: np.ndarray, centre_cell: int) -> np.ndarray:
        """Lay `kernel_image` onto the domain, centred on `centre_cell`."""
        face, i_c, j_c = [
            int(v[0]) for v in hgn.healpix_to_base_cell_coordinates([centre_cell], self.level)
        ]
        import warnings

        n, side = self.n, 2 ** self.level
        d = np.arange(-n, n + 1)
        dj, di = np.meshgrid(d, d, indexing="ij")            # rows = j, cols = i
        tgt_i, tgt_j = (i_c + di).ravel(), (j_c + dj).ravel()

        # A kernel pixel stepping off the base face would wrap onto a
        # neighbouring face, where (i, j) no longer means the same direction.
        # Those are dropped rather than mis-placed.
        on_face = (tgt_i >= 0) & (tgt_i < side) & (tgt_j >= 0) & (tgt_j < side)
        if not on_face.all():
            warnings.warn(
                f"{int((~on_face).sum())} of {tgt_i.size} kernel pixels fall off base face "
                f"{face} (centre at i={i_c}, j={j_c}, face is {side}x{side}) and are dropped: "
                "the kernel is clipped by the face boundary. Centre the domain further from "
                "the face edge, or use a smaller kernel image.",
                RuntimeWarning, stacklevel=3,
            )

        target = np.full(tgt_i.size, -1, dtype=np.int64)
        target[on_face] = hgn.base_cell_coordinates_to_healpix(
            np.full(int(on_face.sum()), face), tgt_i[on_face], tgt_j[on_face], self.level
        ).astype(np.int64)

        field = np.zeros(cell_ids.size, dtype=np.float64)
        pos = np.clip(np.searchsorted(cell_ids, target), 0, cell_ids.size - 1)
        inside = on_face & (cell_ids[pos] == target)         # also drop cells outside the domain
        field[pos[inside]] = self.kernel_image.ravel()[inside]

        off_domain = int((on_face & ~inside).sum())
        if off_domain:
            warnings.warn(
                f"{off_domain} of {tgt_i.size} kernel pixels fall outside the domain -- the "
                "kernel is clipped by the domain boundary, so the fit (and the result) will "
                "be biased. Use a larger domain.",
                RuntimeWarning, stacklevel=3,
            )
        return field

    def __repr__(self) -> str:
        s = self.compact_kernel_sz
        return (
            f"HealPixWideConv(kernel_image={2*self.n+1}x{2*self.n+1}, level={self.level}, "
            f"Jmax={self.Jmax}, compact_kernel_sz={s}, domains_cached={len(self._cache)})"
        )
