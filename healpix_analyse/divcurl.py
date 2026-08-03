"""Local, gauge-aware divergence and curl on multiscale HEALPix maps.

The differential stencil is evaluated by :class:`HealPixConv`.  Horizontal
east/north components are first embedded in a global Cartesian tangent vector.
This makes the sampled channels independent of the local coordinate frame.
Two fixed derivative-of-Gaussian kernels then estimate directional derivatives
along the gauge axes, which are contracted to the intrinsic surface divergence
and outward-normal scalar curl.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Optional, Sequence, Union

import healpix_geo
import numpy as np
import torch
import torch.nn as nn

from healpix_analyse.convol import (
    HealPixConv,
    _DEFAULT_CACHE_DIR,
    _build_rotation_matrices,
)
from healpix_analyse.decomp import HealPixDecomp, HealPixPyramid


ArrayLike = Union[np.ndarray, torch.Tensor]


def _select_component(array: ArrayLike, component: int) -> ArrayLike:
    if isinstance(array, np.ndarray):
        return np.take(array, component, axis=-2)
    return array.select(-2, component)


@dataclass(frozen=True)
class HealPixDivCurlPyramid(Sequence[ArrayLike]):
    """List-like divergence/curl maps returned for a HEALPix pyramid.

    Every band has shape ``[..., 2, N_j]`` after gauge reduction, with
    channel zero containing divergence and channel one containing curl.  With
    ``gauge_reduce="none"``, the shape is ``[..., G, 2, N_j]``.
    """

    bands: tuple[ArrayLike, ...]
    cell_ids: tuple[np.ndarray, ...]
    levels: tuple[int, ...]
    pixel_spacing_m: tuple[float, ...]
    gauge_reduce: str

    def __len__(self) -> int:
        return len(self.bands)

    def __getitem__(self, index):
        return self.bands[index]

    def __iter__(self) -> Iterator[ArrayLike]:
        return iter(self.bands)

    @property
    def div(self) -> tuple[ArrayLike, ...]:
        """Divergence maps ordered from fine to coarse."""
        return tuple(_select_component(band, 0) for band in self.bands)

    @property
    def curl(self) -> tuple[ArrayLike, ...]:
        """Outward-normal scalar-curl maps ordered from fine to coarse."""
        return tuple(_select_component(band, 1) for band in self.bands)


def _as_cell_ids(cell_ids, level: int) -> np.ndarray:
    if cell_ids is None:
        return np.arange(12 * 4**level, dtype=np.int64)
    if torch.is_tensor(cell_ids):
        ids = cell_ids.detach().cpu().numpy()
    else:
        ids = np.asarray(cell_ids)
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError("cell_ids must not be empty")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must contain unique identifiers")
    npix = 12 * 4**level
    if np.any(ids < 0) or np.any(ids >= npix):
        raise ValueError(f"cell_ids contain identifiers outside level={level}")
    return ids


def _derivative_kernels(
    kernel_sz: int,
    sigma_pix: float,
    pixel_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-moment-normalised derivatives along the kernel x/y axes."""
    grid = np.arange(kernel_sz, dtype=np.float64) - kernel_sz // 2
    xx, yy = np.meshgrid(grid, grid)
    gaussian = np.exp(-0.5 * (xx * xx + yy * yy) / sigma_pix**2)

    denom_x = np.sum(xx * xx * gaussian)
    denom_y = np.sum(yy * yy * gaussian)
    derivative_x = xx * gaussian / (denom_x * pixel_spacing_m)
    derivative_y = yy * gaussian / (denom_y * pixel_spacing_m)

    # Enforce exact zero mean against floating-point accumulation.  This is
    # useful for constant fields and keeps the kernels strictly antisymmetric.
    derivative_x -= derivative_x.mean()
    derivative_y -= derivative_y.mean()
    return derivative_x.ravel(), derivative_y.ravel()


class HealPixDivCurl(nn.Module):
    """Estimate divergence and curl at one HEALPix resolution.

    Parameters
    ----------
    level : int
        Grid4Earth/HEALPix level, with ``nside = 2**level``.
    cell_ids : array-like, optional
        NESTED identifiers in the same order as the input velocity samples.
        ``None`` selects the full sphere.
    kernel_sz : int, default=3
        Odd derivative-of-Gaussian stencil size.  Must be at least three.
    sigma_pix : float, optional
        Gaussian width in native pixels.  The default is ``kernel_sz / 3``.
    n_gauges : int, default=1
        Number of rotated HealPixConv gauges.  Multiple gauges provide
        orientation-averaged estimates when ``gauge_reduce="mean"``.
    gauge_type : str, default="phi"
        Gauge convention passed directly to :class:`HealPixConv`.
    gauge_reduce : {"mean", "none"}, default="mean"
        Average the gauge estimates or retain an explicit gauge dimension.
    radius_m : float, default=6371008.8
        Sphere radius used to convert the angular native-pixel spacing to
        metres.  Velocity in m/s then produces divergence and curl in 1/s.

    Notes
    -----
    Input channels are eastward ``u`` and northward ``v``.  They are embedded
    as a three-component Cartesian tangent vector before convolution.  If
    ``a`` and ``b`` are the two oriented gauge axes and ``V`` is this vector,
    the returned scalars are

    ``div = a . d_a(V) + b . d_b(V)``

    ``curl = b . d_a(V) - a . d_b(V)``.

    This contraction includes the change of the local vector basis implicitly
    and is invariant to a simultaneous rotation of the derivative stencil and
    gauge frame, apart from interpolation/discretisation errors.
    """

    def __init__(
        self,
        level: int,
        cell_ids=None,
        *,
        kernel_sz: int = 3,
        sigma_pix: Optional[float] = None,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        gauge_reduce: str = "mean",
        radius_m: float = 6_371_008.8,
        singularity_lonlat=None,
        ref_direction=None,
        ellipsoid: str = "sphere",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir=_DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__()

        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0")
        if (
            isinstance(kernel_sz, bool)
            or int(kernel_sz) != kernel_sz
            or int(kernel_sz) < 3
            or int(kernel_sz) % 2 == 0
        ):
            raise ValueError("kernel_sz must be an odd integer >= 3")
        if isinstance(n_gauges, bool) or int(n_gauges) != n_gauges or int(n_gauges) < 1:
            raise ValueError("n_gauges must be an integer >= 1")
        if gauge_reduce not in ("mean", "none"):
            raise ValueError("gauge_reduce must be 'mean' or 'none'")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if not np.isfinite(radius_m) or float(radius_m) <= 0:
            raise ValueError("radius_m must be a positive finite value")

        self.level = int(level)
        self.nside = 2**self.level
        self.kernel_sz = int(kernel_sz)
        self.sigma_pix = (
            float(sigma_pix) if sigma_pix is not None else self.kernel_sz / 3.0
        )
        if not np.isfinite(self.sigma_pix) or self.sigma_pix <= 0:
            raise ValueError("sigma_pix must be a positive finite value")
        self.n_gauges = int(n_gauges)
        self.gauge_type = str(gauge_type)
        self.gauge_reduce = gauge_reduce
        self.radius_m = float(radius_m)
        self.ellipsoid = str(ellipsoid)
        self.dtype = dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        ids = _as_cell_ids(cell_ids, self.level)
        self.partial = cell_ids is not None
        self.n_pixels = len(ids)
        self._cell_ids = ids.copy()

        alpha_pix = math.sqrt(4.0 * math.pi / (12.0 * self.nside**2))
        self.pixel_spacing_rad = alpha_pix
        self.pixel_spacing_m = self.radius_m * alpha_pix

        lon, lat = healpix_geo.nested.healpix_to_lonlat(
            ids.tolist(), self.level, ellipsoid=self.ellipsoid
        )
        lon_rad = np.deg2rad(np.asarray(lon, dtype=np.float64))
        lat_rad = np.deg2rad(np.asarray(lat, dtype=np.float64))
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)

        east = np.stack(
            [-sin_lon, cos_lon, np.zeros_like(lon_rad)], axis=0
        )
        north = np.stack(
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=0
        )
        self.register_buffer(
            "_east",
            torch.as_tensor(east, dtype=dtype, device=resolved_device),
        )
        self.register_buffer(
            "_north",
            torch.as_tensor(north, dtype=dtype, device=resolved_device),
        )
        self.register_buffer(
            "_dtype_anchor",
            torch.empty(0, dtype=dtype, device=resolved_device),
        )

        self.conv = HealPixConv(
            level=self.level,
            in_channels=3,
            out_channels=6,
            kernel_sz=self.kernel_sz,
            n_gauges=self.n_gauges,
            gauge_type=self.gauge_type,
            singularity_lonlat=singularity_lonlat,
            ref_direction=ref_direction,
            cell_ids=ids if self.partial else None,
            nest=True,
            use_norm=False,
            device=resolved_device,
            ellipsoid=self.ellipsoid,
            dtype=dtype,
            cache_dir=cache_dir,
        )

        theta = 0.5 * math.pi - lat_rad
        rotations = _build_rotation_matrices(
            theta,
            lon_rad,
            self.n_gauges,
            self.gauge_type,
            resolved_device,
            dtype,
            ref_direction=self.conv.ref_direction,
        )
        # Columns zero and one are precisely the two tangent axes used to
        # rotate the North-Pole kernel grid in HealPixConv.
        self.register_buffer(
            "_axis_a", rotations[..., :, 0].permute(1, 2, 0).contiguous()
        )
        self.register_buffer(
            "_axis_b", rotations[..., :, 1].permute(1, 2, 0).contiguous()
        )

        derivative_a, derivative_b = _derivative_kernels(
            self.kernel_sz, self.sigma_pix, self.pixel_spacing_m
        )
        self.register_buffer(
            "_derivative_a",
            torch.as_tensor(
                derivative_a.reshape(self.kernel_sz, self.kernel_sz),
                dtype=dtype,
                device=resolved_device,
            ),
        )
        self.register_buffer(
            "_derivative_b",
            torch.as_tensor(
                derivative_b.reshape(self.kernel_sz, self.kernel_sz),
                dtype=dtype,
                device=resolved_device,
            ),
        )
        weight = np.zeros((3, 6, self.kernel_sz**2), dtype=np.float64)
        for component in range(3):
            weight[component, component, :] = derivative_a
            weight[component, 3 + component, :] = derivative_b
        self.conv.set_kernel(weight, bias=None, requires_grad=False)

    @property
    def device(self) -> torch.device:
        return self._dtype_anchor.device

    @property
    def cell_ids(self) -> np.ndarray:
        return self._cell_ids.copy()

    @property
    def derivative_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fixed physical derivative kernels along the two gauge axes."""
        return self._derivative_a.clone(), self._derivative_b.clone()

    def _apply(self, fn):
        result = super()._apply(fn)
        self.dtype = self._dtype_anchor.dtype
        self.conv.device = self.device
        self.conv.dtype = self.dtype
        return result

    def _prepare_velocity(
        self, velocity: ArrayLike
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        is_numpy = isinstance(velocity, np.ndarray)
        if not is_numpy and not torch.is_tensor(velocity):
            raise TypeError("velocity must be a numpy.ndarray or torch.Tensor")
        if (np.iscomplexobj(velocity) if is_numpy else torch.is_complex(velocity)):
            raise TypeError("velocity must be real-valued")
        tensor = torch.as_tensor(velocity, device=self.device).to(dtype=self.dtype)
        if (
            tensor.ndim < 2
            or tensor.shape[-2] != 2
            or tensor.shape[-1] != self.n_pixels
        ):
            raise ValueError(
                "velocity must have shape [..., 2, N] with "
                f"N={self.n_pixels}; got {tuple(tensor.shape)}"
            )
        leading_shape = tuple(tensor.shape[:-2])
        return tensor.reshape(-1, 2, self.n_pixels), leading_shape, is_numpy

    def forward(self, velocity: ArrayLike) -> ArrayLike:
        """Return divergence/curl for eastward/northward velocity channels."""
        tensor, leading_shape, is_numpy = self._prepare_velocity(velocity)
        u = tensor[:, 0, :]
        v = tensor[:, 1, :]
        cartesian = (
            u[:, None, :] * self._east[None, :, :]
            + v[:, None, :] * self._north[None, :, :]
        )

        derivatives = self.conv(cartesian)
        derivatives = derivatives.reshape(
            tensor.shape[0], self.n_gauges, 6, self.n_pixels
        )
        derivative_a = derivatives[:, :, :3, :]
        derivative_b = derivatives[:, :, 3:, :]
        axis_a = self._axis_a[None, :, :, :]
        axis_b = self._axis_b[None, :, :, :]

        divergence = (axis_a * derivative_a).sum(dim=2) + (
            axis_b * derivative_b
        ).sum(dim=2)
        curl = (axis_b * derivative_a).sum(dim=2) - (
            axis_a * derivative_b
        ).sum(dim=2)
        output = torch.stack((divergence, curl), dim=2)  # [B,G,2,N]

        if self.gauge_reduce == "mean":
            output = output.mean(dim=1)
            output = output.reshape(*leading_shape, 2, self.n_pixels)
        else:
            output = output.reshape(
                *leading_shape, self.n_gauges, 2, self.n_pixels
            )
        if is_numpy:
            return output.detach().cpu().numpy()
        return output

    def compute(self, velocity: ArrayLike) -> ArrayLike:
        """Alias for :meth:`forward`."""
        return self.forward(velocity)

    def extra_repr(self) -> str:
        domain = "partial" if self.partial else "full-sky"
        return (
            f"level={self.level}, cells={self.n_pixels}, domain={domain}, "
            f"kernel_sz={self.kernel_sz}, sigma_pix={self.sigma_pix:g}, "
            f"gauges={self.n_gauges}, gauge_type={self.gauge_type!r}, "
            f"gauge_reduce={self.gauge_reduce!r}, "
            f"spacing={self.pixel_spacing_m:.3f} m"
        )


class HealPixMultiScaleDivCurl(nn.Module):
    """Apply :class:`HealPixDivCurl` to every band of a decomposition.

    The native pixel spacing doubles approximately at every Down operation.
    Each derivative kernel is divided by its own physical spacing, making the
    returned values comparable and expressed in the same physical units.
    """

    def __init__(
        self,
        decomp: HealPixDecomp,
        *,
        kernel_sz: int = 3,
        sigma_pix: Optional[float] = None,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        gauge_reduce: str = "mean",
        radius_m: float = 6_371_008.8,
        singularity_lonlat=None,
        ref_direction=None,
        ellipsoid: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir=_DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__()
        if not isinstance(decomp, HealPixDecomp):
            raise TypeError("decomp must be a HealPixDecomp instance")

        self.levels = tuple(decomp.levels)
        self.cell_ids_per_scale = tuple(ids.copy() for ids in decomp.cell_ids)
        self.n_bands = len(self.levels)
        self.gauge_reduce = gauge_reduce
        if ellipsoid is None:
            ellipsoid = decomp.ellipsoid
        if dtype is None:
            dtype = decomp.dtype
        if device is None:
            device = decomp.device

        self.layers = nn.ModuleList(
            [
                HealPixDivCurl(
                    level=level,
                    cell_ids=ids if decomp.partial else None,
                    kernel_sz=kernel_sz,
                    sigma_pix=sigma_pix,
                    n_gauges=n_gauges,
                    gauge_type=gauge_type,
                    gauge_reduce=gauge_reduce,
                    radius_m=radius_m,
                    singularity_lonlat=singularity_lonlat,
                    ref_direction=ref_direction,
                    ellipsoid=ellipsoid,
                    dtype=dtype,
                    device=device,
                    cache_dir=cache_dir,
                )
                for level, ids in zip(self.levels, self.cell_ids_per_scale)
            ]
        )
        self.pixel_spacing_m = tuple(
            layer.pixel_spacing_m for layer in self.layers
        )

    def _validate_metadata(self, pyramid: HealPixPyramid) -> None:
        if tuple(pyramid.levels) != self.levels:
            raise ValueError(
                f"Pyramid levels {tuple(pyramid.levels)} do not match {self.levels}"
            )
        if len(pyramid.cell_ids) != self.n_bands:
            raise ValueError("Pyramid contains an invalid number of cell-id arrays")
        for index, (actual, expected) in enumerate(
            zip(pyramid.cell_ids, self.cell_ids_per_scale)
        ):
            if not np.array_equal(np.asarray(actual), expected):
                raise ValueError(f"Pyramid cell_ids do not match at band {index}")

    def forward(
        self, pyramid: Union[HealPixPyramid, Sequence[ArrayLike]]
    ) -> HealPixDivCurlPyramid:
        """Return one divergence/curl map for every native-resolution band."""
        if isinstance(pyramid, HealPixPyramid):
            self._validate_metadata(pyramid)
            bands = pyramid.bands
        else:
            bands = tuple(pyramid)
        if len(bands) != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, got {len(bands)}")

        outputs = tuple(layer(band) for layer, band in zip(self.layers, bands))
        return HealPixDivCurlPyramid(
            bands=outputs,
            cell_ids=tuple(ids.copy() for ids in self.cell_ids_per_scale),
            levels=self.levels,
            pixel_spacing_m=self.pixel_spacing_m,
            gauge_reduce=self.gauge_reduce,
        )

    def compute(
        self, pyramid: Union[HealPixPyramid, Sequence[ArrayLike]]
    ) -> HealPixDivCurlPyramid:
        """Alias for :meth:`forward`."""
        return self.forward(pyramid)

    def extra_repr(self) -> str:
        return (
            f"bands={self.n_bands}, levels={self.levels}, "
            f"spacing_m={tuple(round(value, 3) for value in self.pixel_spacing_m)}, "
            f"gauge_reduce={self.gauge_reduce!r}"
        )


__all__ = [
    "HealPixDivCurl",
    "HealPixMultiScaleDivCurl",
    "HealPixDivCurlPyramid",
]
