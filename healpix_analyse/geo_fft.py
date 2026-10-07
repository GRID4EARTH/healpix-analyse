"""FFT convolution on the HEALPix face lattice with a geodesically sampled kernel.

Motivation
----------
Inside one base face, the NESTED cells of a HEALPix level form a regular
``(i, j)`` lattice: the children of a parent cell are a square block, and any
set of cells of one face sits in a rectangle of that lattice.  A convolution
on such a block can therefore be done by a planar FFT, at ``N log N`` cost,
with zero padding for the cells that are absent or missing.

The catch is that the lattice is **not a square grid**.  The two lattice
steps make an angle of 99 degrees at the centre of an equatorial face, and in
the polar caps the lattice is a sheared parallelogram whose sides differ in
length by up to a factor two.  A kernel written as ``k(sqrt(di**2 + dj**2))``
-- the face treated as a square grid -- is therefore not the kernel one asked
for: on Sentinel-2 tiles at level 20 it is off by 5-30 % of the standard
deviation of the exact result, and by up to 50 % under cloud masks.

:class:`HealPixGeoFFT` removes that error by sampling the kernel on the
**geodesic distances** (and azimuths) from the central cell of the block to
every cell of the kernel window.  The local distortion of the lattice is then
built into the kernel image, and the FFT applies the exact operator at the
centre; what remains is the drift of the cell geometry between the centre
and the edges of the block, negligible on a tile small compared with the
scale of that drift (kilometres at level 20) and measurable only on patches
tens of degrees wide.  Missing data are handled by normalized convolution,
``K*(w x) / K*w``, computed as a ratio of two FFTs: this is the exact
normalized convolution for the kernel image used, so gaps add no error of
their own.

Scope
-----
One base face.  A block that straddles several faces, a region tens of
degrees wide or the full sphere have no single Fourier kernel; use
:class:`~healpix_analyse.wide_conv.HealPixWideConv` or
:class:`~healpix_analyse.pyramid_conv.HealPixPyramidConv` there, and
:class:`~healpix_analyse.healpix_sht.HEALPixSHT` for full-sky maps.

Quick start
-----------
::

    from healpix_analyse import HealPixGeoFFT

    conv = HealPixGeoFFT(level=20, cell_ids=cells, kernel="exponential", scale=100.0)   # metres
    y = conv(x)                     # x: [N] with NaN where missing -> normalized convolution

    # any kernel: a callable of the geodesic distance (metres here) and the azimuth from north
    aniso = lambda rho, phi: np.exp(-(rho * np.cos(phi) / 300.0) ** 2 - (rho * np.sin(phi) / 100.0) ** 2)
    conv = HealPixGeoFFT(level=20, cell_ids=cells, kernel=aniso, radius=1000.0)
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

import healpix_geo.nested as hgn

from healpix_analyse._ellipsoid import canonicalize_ellipsoid

ArrayLike = Union[np.ndarray, torch.Tensor]
KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

R_EARTH_M = 6371008.8          # mean Earth radius, as in HealPixWideConv.lattice_offsets_m

__all__ = [
    "HealPixGeoFFT",
    "geo_fft_convolve",
    "KERNELS",
    "register_kernel",
    "make_kernel",
    "lattice_geometry",
]


# ---------------------------------------------------------------------------
# Predefined kernels.  Every factory takes the scale (in the units the user
# chose) and returns k(rho, phi) with k(0) = 1 (except the signed DoG).
# ---------------------------------------------------------------------------

def _gaussian(scale: float) -> KernelFn:
    return lambda rho, phi: np.exp(-0.5 * (rho / scale) ** 2)


def _exponential(scale: float) -> KernelFn:
    return lambda rho, phi: np.exp(-rho / scale)


def _matern(scale: float, nu: float = 1.5) -> KernelFn:
    from scipy.special import gamma, kv

    def fn(rho, phi):
        x = np.sqrt(2.0 * nu) * np.asarray(rho, float) / scale
        out = np.ones_like(x)
        m = x > 0
        out[m] = (2.0 ** (1.0 - nu) / gamma(nu)) * x[m] ** nu * kv(nu, x[m])
        return out
    return fn


def _lorentzian(scale: float) -> KernelFn:
    return lambda rho, phi: 1.0 / (1.0 + (rho / scale) ** 2)


def _moffat(scale: float, beta: float = 2.5) -> KernelFn:
    return lambda rho, phi: (1.0 + (rho / scale) ** 2) ** (-beta)


def _voigt(scale: float, gamma: Optional[float] = None) -> KernelFn:
    from scipy.special import wofz
    g = scale if gamma is None else gamma

    def fn(rho, phi):
        z = (np.asarray(rho, float) + 1j * g) / (scale * np.sqrt(2.0))
        v = wofz(z).real
        v0 = wofz(1j * g / (scale * np.sqrt(2.0))).real
        return v / v0
    return fn


def _screened_poisson(scale: float, r_core: float = 0.5) -> KernelFn:
    """K0(rho / scale) with a regularised core: rho is floored at r_core * scale."""
    from scipy.special import k0

    def fn(rho, phi):
        r = np.maximum(np.asarray(rho, float), r_core * scale)
        return k0(r / scale) / k0(r_core)
    return fn


def _top_hat(scale: float) -> KernelFn:
    return lambda rho, phi: (np.asarray(rho, float) <= scale).astype(float)


def _dog(scale: float, ratio: float = 2.0) -> KernelFn:
    """Difference of Gaussians (signed, zero mean): a band-pass kernel."""
    s1, s2 = scale, ratio * scale
    return lambda rho, phi: (np.exp(-0.5 * (rho / s1) ** 2) / s1 ** 2
                             - np.exp(-0.5 * (rho / s2) ** 2) / s2 ** 2) * s1 ** 2


KERNELS: Dict[str, Callable[..., KernelFn]] = {
    "gaussian": _gaussian,
    "exponential": _exponential,
    "matern": _matern,
    "lorentzian": _lorentzian,
    "moffat": _moffat,
    "voigt": _voigt,
    "screened_poisson": _screened_poisson,
    "top_hat": _top_hat,
    "dog": _dog,
}
SIGNED_KERNELS = {"dog"}


def register_kernel(name: str, factory: Callable[..., KernelFn]) -> None:
    """Add a predefined kernel: ``factory(scale, **kwargs) -> k(rho, phi)``."""
    KERNELS[str(name)] = factory


def make_kernel(kernel: Union[str, KernelFn], scale: Optional[float] = None, **kwargs) -> KernelFn:
    """Resolve ``kernel`` to a callable ``k(rho, phi)``.

    ``kernel`` is either a name of :data:`KERNELS` (then ``scale`` is required,
    in the user's units, and ``kwargs`` are the family's extra parameters) or a
    callable, returned unchanged.
    """
    if callable(kernel):
        return kernel
    name = str(kernel).lower()
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; known: {sorted(KERNELS)} or a callable k(rho, phi)")
    if scale is None:
        raise ValueError(f"kernel {name!r} needs a scale")
    return KERNELS[name](float(scale), **kwargs)


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------

def _unit_vectors(lon_deg, lat_deg):
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _geodesic_rad(lon1, lat1, lon2, lat2):
    a, b = np.radians(lat1), np.radians(lat2)
    d = np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(np.radians(lon1 - lon2))
    return np.arccos(np.clip(d, -1.0, 1.0))


def _bearing_rad(lon1, lat1, lon2, lat2):
    """Initial bearing from point 1 to point 2, radians, 0 = north, pi/2 = east."""
    a, b = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(b), np.cos(a) * np.sin(b) - np.sin(a) * np.cos(b) * np.cos(dl))


def _lattice_lonlat(face: int, i, j, level: int, ellipsoid: str):
    """lon, lat (degrees) of the cells at base-face coordinates (i, j) of ``face``."""
    i = np.asarray(i, dtype=np.int64).ravel()
    j = np.asarray(j, dtype=np.int64).ravel()
    cells = hgn.base_cell_coordinates_to_healpix(np.full(i.size, face, dtype=np.int64), i, j, level)
    lon, lat = hgn.healpix_to_lonlat(np.asarray(cells).tolist(), level, ellipsoid=ellipsoid)
    return np.asarray(lon, float), np.asarray(lat, float)


def lattice_geometry(level: int, face: int, i: int, j: int, *, ellipsoid: str = "sphere") -> dict:
    """Local geometry of the face lattice at cell ``(face, i, j)``.

    Returns the lengths of the ``i`` and ``j`` steps in units of the cell
    size ``alpha = sqrt(4 pi / N)`` and the angle between them, in degrees.
    In the equatorial zone the two steps have equal length and only the angle
    departs from 90 degrees (99 degrees at the equator); in the polar caps the
    steps also differ in length.
    """
    side = 2 ** level
    i2, j2 = min(i + 1, side - 1), min(j + 1, side - 1)
    i0, j0 = (i if i2 > i else i - 1), (j if j2 > j else j - 1)
    lon, lat = _lattice_lonlat(face, [i0, i2, i0, i2], [j0, j0, j2, j2], level, ellipsoid)
    alpha = math.sqrt(4.0 * math.pi / (12.0 * 4 ** level))
    li = float(_geodesic_rad(lon[0], lat[0], lon[1], lat[1]) / alpha)
    lj = float(_geodesic_rad(lon[0], lat[0], lon[2], lat[2]) / alpha)
    ld = float(_geodesic_rad(lon[0], lat[0], lon[3], lat[3]) / alpha)
    cos_t = (ld ** 2 - li ** 2 - lj ** 2) / (2.0 * li * lj)
    return dict(step_i=li, step_j=lj, angle_deg=float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))),
                cell_size_rad=alpha)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

_UNIT_TO_RAD = {"m": 1.0 / R_EARTH_M, "km": 1000.0 / R_EARTH_M, "deg": math.pi / 180.0}


class HealPixGeoFFT(nn.Module):
    """FFT convolution of a HEALPix block inside one base face, with the kernel
    sampled on geodesic distances from the block centre.

    Parameters
    ----------
    level : int
        HEALPix level (``nside = 2**level``).
    cell_ids : array-like of int, shape [N]
        NESTED cell identifiers at ``level``, all inside one base face, in any
        order; need not be contiguous.  The operator works on the bounding
        rectangle of these cells in base-face coordinates; lattice positions
        that are not in ``cell_ids`` count as missing.
    kernel : str or callable
        A predefined kernel (:data:`KERNELS`: ``"gaussian"``, ``"exponential"``,
        ``"matern"``, ``"lorentzian"``, ``"moffat"``, ``"voigt"``,
        ``"screened_poisson"``, ``"top_hat"``, ``"dog"``) or any callable
        ``k(rho, phi)`` of the geodesic distance ``rho`` (in ``units``) and the
        azimuth ``phi`` from north (radians, clockwise), evaluated on arrays.
    scale : float, optional
        Scale parameter of a predefined kernel, in ``units``.
    radius : float, optional
        Truncation radius of the kernel, in ``units``.  Default: the distance
        at which the kernel (at azimuth 0) falls below ``1e-3`` of its value
        at the origin, capped so that the window fits in the base face.  The
        same truncated kernel is what :func:`~healpix_analyse.validation.direct_spherical_convolution`
        applies when given the same ``radius``.
    units : {"m", "km", "deg", "pix"}, default "m"
        Units of ``scale``, ``radius`` and of ``rho`` in a callable kernel.
        ``"pix"`` is the cell size ``sqrt(4 pi / N)`` of ``level``.
    kernel_kwargs : dict, optional
        Extra parameters of a predefined family (``nu`` for Matérn, ``beta`` for
        Moffat, ``gamma`` for Voigt, ``ratio`` for the DoG, ``r_core`` for the
        screened Poisson kernel).
    normalized : bool, default True
        Normalized convolution ``K*(w x) / K*w``: NaN in ``x`` (or zero
        ``weights``) are missing data, and the result is NaN where the support
        ``K*w`` falls below ``support_eps`` of its maximum.  ``False``: the raw
        sum with zeros at the missing cells.  A signed kernel forces ``False``.
    support_eps : float, default 1e-6
    ellipsoid : str, default "sphere"
    dtype : torch.dtype, default torch.float64
    device : torch.device or str, optional
    warn_extent_deg : float, default 10.0
        A block wider than this (great-circle distance between opposite
        corners) triggers a warning: the cell geometry then varies noticeably
        across the block and the pyramid is the better tool.

    Attributes
    ----------
    kernel_image : np.ndarray, shape (2n+1, 2n+1)
        The kernel actually applied, sampled at the lattice positions around
        the central cell (``[dj + n, di + n]``).
    geometry : dict
        ``step_i``, ``step_j`` (cell units), ``angle_deg`` of the lattice at the
        centre, ``window`` (``n``), ``centre_lonlat``, ``face``, ``shape``.
    extent_deg : float
        Great-circle extent of the block.
    """

    def __init__(self, level: int, cell_ids, kernel: Union[str, KernelFn], *,
                 scale: Optional[float] = None, radius: Optional[float] = None,
                 units: str = "m", kernel_kwargs: Optional[dict] = None,
                 normalized: bool = True, support_eps: float = 1e-6,
                 ellipsoid: str = "sphere", dtype: torch.dtype = torch.float64,
                 device: Optional[Union[str, torch.device]] = None,
                 warn_extent_deg: float = 10.0) -> None:
        super().__init__()
        self.level = int(level)
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)
        self.dtype = dtype
        self.device_ = torch.device(device) if device is not None else torch.device("cpu")
        self.support_eps = float(support_eps)
        self.units = str(units)
        if self.units not in ("m", "km", "deg", "pix"):
            raise ValueError("units must be 'm', 'km', 'deg' or 'pix'")

        # -- domain: one face, bounding rectangle in base-face coordinates
        cells = np.asarray(cell_ids, dtype=np.int64).ravel()
        if cells.size == 0:
            raise ValueError("cell_ids is empty")
        if np.unique(cells).size != cells.size:
            raise ValueError("cell_ids must be unique")
        face, ii, jj = [np.asarray(v, dtype=np.int64) for v in hgn.healpix_to_base_cell_coordinates(cells.tolist(), self.level)]
        faces = np.unique(face)
        if faces.size != 1:
            raise ValueError(
                f"cell_ids span {faces.size} base faces ({faces.tolist()}); HealPixGeoFFT works inside one "
                "face -- use HealPixWideConv / HealPixPyramidConv across faces")
        self.face = int(faces[0])
        self.i0, self.j0 = int(ii.min()), int(jj.min())
        self.W, self.H = int(ii.max() - self.i0 + 1), int(jj.max() - self.j0 + 1)     # image [H=j, W=i]
        self.n_cells = int(cells.size)
        self._rows = torch.as_tensor(jj - self.j0, dtype=torch.long, device=self.device_)
        self._cols = torch.as_tensor(ii - self.i0, dtype=torch.long, device=self.device_)
        self.cell_ids_ = cells

        # -- centre and local geometry
        side = 2 ** self.level
        ic, jc = self.i0 + self.W // 2, self.j0 + self.H // 2
        self.geometry = lattice_geometry(self.level, self.face, ic, jc, ellipsoid=self.ellipsoid)
        alpha = self.geometry["cell_size_rad"]
        lon_c, lat_c = _lattice_lonlat(self.face, [ic], [jc], self.level, self.ellipsoid)
        self.geometry.update(centre_lonlat=(float(lon_c[0]), float(lat_c[0])), face=self.face,
                             shape=(self.H, self.W), i0=self.i0, j0=self.j0)
        lon_e, lat_e = _lattice_lonlat(self.face, [self.i0, self.i0 + self.W - 1], [self.j0, self.j0 + self.H - 1],
                                       self.level, self.ellipsoid)
        self.extent_deg = float(np.degrees(_geodesic_rad(lon_e[0], lat_e[0], lon_e[1], lat_e[1])))
        if self.extent_deg > warn_extent_deg:
            warnings.warn(
                f"the block spans {self.extent_deg:.1f} degrees: the cell geometry varies across it and the "
                "kernel sampled at the centre drifts towards the edges; consider HealPixWideConv (pyramid)",
                RuntimeWarning, stacklevel=2)

        # -- units: one unit of the user's choice = this many cells
        self.unit_per_cell = (1.0 if self.units == "pix" else _UNIT_TO_RAD[self.units] / alpha)   # user unit -> cells
        self.cells_per_unit = self.unit_per_cell
        self.kernel_fn = make_kernel(kernel, scale, **(kernel_kwargs or {}))
        self.kernel_name = kernel if isinstance(kernel, str) else getattr(kernel, "__name__", "callable")
        self.scale = scale
        signed = isinstance(kernel, str) and kernel.lower() in SIGNED_KERNELS
        self.normalized = bool(normalized) and not signed

        # -- truncation radius and window
        s = math.sin(math.radians(self.geometry["angle_deg"]))
        step_min_cells = min(self.geometry["step_i"], self.geometry["step_j"]) * s
        n_max = min(ic, jc, side - 1 - ic, side - 1 - jc)              # window must stay on the face
        if radius is None:
            n_block = max(self.H, self.W)                                # default search: up to the block size
            radius = self._default_radius(min(n_max, n_block) * step_min_cells / self.cells_per_unit)
        self.radius = float(radius)
        n = int(math.ceil(self.radius * self.cells_per_unit / step_min_cells))
        if n > n_max:
            raise ValueError(
                f"the kernel window (half-size {n} lattice steps for radius {self.radius} {self.units}) runs off "
                f"base face {self.face}; the largest radius here is {n_max * step_min_cells / self.cells_per_unit:.4g} "
                f"{self.units}")
        self.window = n

        # -- kernel image: geodesic distance and azimuth from the centre to every lattice position
        d = np.arange(-n, n + 1)
        dj, di = np.meshgrid(d, d, indexing="ij")
        lon, lat = _lattice_lonlat(self.face, ic + di, jc + dj, self.level, self.ellipsoid)
        rho_rad = _geodesic_rad(lon_c[0], lat_c[0], lon, lat).reshape(2 * n + 1, 2 * n + 1)
        phi = _bearing_rad(lon_c[0], lat_c[0], lon, lat).reshape(2 * n + 1, 2 * n + 1)
        rho = rho_rad / (_UNIT_TO_RAD[self.units] if self.units != "pix" else alpha)
        k = np.asarray(self.kernel_fn(rho, phi), dtype=np.float64)
        k = np.where(rho <= self.radius, k, 0.0)
        self.kernel_image = k
        self.geometry["window"] = n

        # -- FFT plan: zero-padded linear convolution of an [H, W] image with the (2n+1)^2 kernel
        self._Hp = _fast_len(self.H + 2 * n)
        self._Wp = _fast_len(self.W + 2 * n)
        kpad = torch.zeros(self._Hp, self._Wp, dtype=self.dtype, device=self.device_)
        kpad[: 2 * n + 1, : 2 * n + 1] = torch.as_tensor(k, dtype=self.dtype, device=self.device_)
        kpad = torch.roll(kpad, shifts=(-n, -n), dims=(0, 1))          # kernel centre at the origin
        self.register_buffer("_kfft", torch.fft.rfft2(kpad), persistent=False)

    # ------------------------------------------------------------------ helpers
    def _default_radius(self, radius_max: float) -> float:
        """Distance (user units) where |k| at azimuth 0 first falls below 1e-3 of |k(0)|.

        Searched on [0, radius_max] with three successive refinements, so that
        the result is accurate to ~1e-9 radius_max even when radius_max is the
        distance to the face edge.
        """
        k0 = float(np.abs(np.asarray(self.kernel_fn(np.zeros(1), np.zeros(1)), dtype=np.float64))[0])
        if not k0 > 0:
            k0 = float(np.abs(self.kernel_fn(np.linspace(0, radius_max, 4097), np.zeros(4097))).max())
        lo, hi = 0.0, float(radius_max)
        for _ in range(3):
            r = np.linspace(lo, hi, 4097)
            k = np.abs(np.asarray(self.kernel_fn(r, np.zeros_like(r)), dtype=np.float64))
            below = np.flatnonzero(k < 1e-3 * k0)
            if below.size == 0:
                return float(radius_max)
            i = int(below[0])
            lo, hi = (r[i - 1] if i > 0 else 0.0), r[i]
        return float(hi)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N] -> [B, H, W] with zeros at absent lattice positions."""
        img = torch.zeros(x.shape[0], self.H, self.W, dtype=self.dtype, device=self.device_)
        img[:, self._rows, self._cols] = x
        return img

    def _conv(self, img: torch.Tensor) -> torch.Tensor:
        """Linear convolution of [B, H, W] by the kernel image, 'same' output."""
        B = img.shape[0]
        pad = torch.zeros(B, self._Hp, self._Wp, dtype=self.dtype, device=self.device_)
        pad[:, : self.H, : self.W] = img
        y = torch.fft.irfft2(torch.fft.rfft2(pad) * self._kfft, s=(self._Hp, self._Wp))
        return y[:, : self.H, : self.W]

    def _prepare(self, x: ArrayLike):
        is_numpy = isinstance(x, np.ndarray)
        t = torch.as_tensor(x, device=self.device_).to(self.dtype)
        shape = t.shape
        if shape[-1] != self.n_cells:
            raise ValueError(f"expected {self.n_cells} values on the last axis, got {shape[-1]}")
        return t.reshape(-1, self.n_cells), shape, is_numpy

    def _restore(self, y: torch.Tensor, shape, is_numpy: bool):
        y = y.reshape(shape)
        return y.detach().cpu().numpy() if is_numpy else y

    # ------------------------------------------------------------------ public API
    def raw(self, x: ArrayLike) -> ArrayLike:
        """Raw convolution ``K*x`` at the cells; NaN count as zero."""
        t, shape, is_numpy = self._prepare(x)
        img = self._to_image(torch.nan_to_num(t))
        y = self._conv(img)[:, self._rows, self._cols]
        return self._restore(y, shape, is_numpy)

    def support(self, weights: Optional[ArrayLike] = None, x: Optional[ArrayLike] = None) -> ArrayLike:
        """Support map ``K*w`` at the cells (``w`` = ``weights``, or ``isfinite(x)``)."""
        if weights is None:
            if x is None:
                w = torch.ones(1, self.n_cells, dtype=self.dtype, device=self.device_)
                return self._restore(self._conv(self._to_image(w))[:, self._rows, self._cols], (self.n_cells,), True)
            t, shape, is_numpy = self._prepare(x)
            w = torch.isfinite(t).to(self.dtype)
        else:
            w, shape, is_numpy = self._prepare(weights)
            w = torch.nan_to_num(w)
        return self._restore(self._conv(self._to_image(w))[:, self._rows, self._cols], shape, is_numpy)

    def forward(self, x: ArrayLike, weights: Optional[ArrayLike] = None) -> ArrayLike:
        """Convolve ``x`` ([N], [B, N] or [B, C, N]; numpy or torch).

        Normalized mode (default): ``K*(w x) / K*w`` with ``w = weights`` (or
        1 where ``x`` is finite), NaN where the support is below
        ``support_eps`` of its maximum.  Raw mode: ``K*x`` with zeros at the
        missing cells.
        """
        t, shape, is_numpy = self._prepare(x)
        finite = torch.isfinite(t)
        if not self.normalized:
            y = self._conv(self._to_image(torch.where(finite, t, torch.zeros_like(t))))[:, self._rows, self._cols]
            return self._restore(y, shape, is_numpy)
        if weights is None:
            w = finite.to(self.dtype)
        else:
            w, _, _ = self._prepare(weights)
            w = torch.where(finite & torch.isfinite(w), w, torch.zeros_like(w))
            if w.shape[0] == 1 and t.shape[0] > 1:
                w = w.expand_as(t)
        num = self._conv(self._to_image(torch.where(finite, t, torch.zeros_like(t)) * w))[:, self._rows, self._cols]
        den = self._conv(self._to_image(w))[:, self._rows, self._cols]
        thr = self.support_eps * den.abs().amax(dim=-1, keepdim=True)
        ok = den > thr
        y = torch.where(ok, num / torch.where(ok, den, torch.ones_like(den)), torch.full_like(num, float("nan")))
        return self._restore(y, shape, is_numpy)

    def extra_repr(self) -> str:
        g = self.geometry
        return (f"level={self.level}, face={self.face}, block={self.H}x{self.W}, cells={self.n_cells}, "
                f"kernel={self.kernel_name}, scale={self.scale} {self.units}, radius={self.radius:.4g} {self.units}, "
                f"window={self.window}, lattice angle={g['angle_deg']:.1f} deg, steps={g['step_i']:.2f}/{g['step_j']:.2f}, "
                f"extent={self.extent_deg:.3f} deg, normalized={self.normalized}")


def _fast_len(n: int) -> int:
    """Smallest 2^a 3^b 5^c >= n (fast FFT length)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def geo_fft_convolve(x: ArrayLike, cell_ids, level: int, kernel: Union[str, KernelFn], **kwargs) -> ArrayLike:
    """One-shot :class:`HealPixGeoFFT`: build the operator and apply it to ``x``.

    ``kwargs`` are passed to :class:`HealPixGeoFFT` (``scale``, ``radius``,
    ``units``, ``normalized``, ...).  Build the class once instead when the
    same kernel is applied to several maps of the same block.
    """
    return HealPixGeoFFT(level, cell_ids, kernel, **kwargs)(x)
