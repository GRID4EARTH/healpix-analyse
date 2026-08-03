"""Fast local 2-D FFTs for data sampled on HEALPix cells.

The HEALPix cell centres are projected onto a square gnomonic grid.  The
geometry is computed once by :class:`LocalFFT`; subsequent projections and
transforms use differentiable PyTorch scatter/gather operations.

The FFT itself is exactly invertible on the projected grid.  The default
HEALPix -> grid -> HEALPix round trip is approximate because it uses fast
bilinear gridding in both directions.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import healpix_geo
import numpy as np
import torch
import torch.nn as nn


ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_complex_input(value: ArrayLike) -> bool:
    if torch.is_tensor(value):
        return torch.is_complex(value)
    return bool(np.iscomplexobj(value))


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise ValueError("dtype must be torch.float32 or torch.float64")


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


class LocalFFT(nn.Module):
    """A reusable local flat-sky FFT for a set of NESTED HEALPix cells.

    Parameters
    ----------
    cell_ids : array-like, shape [N]
        Unique HEALPix cell identifiers in NESTED ordering.
    level : int
        HEALPix level, with ``nside = 2**level``.
    ellipsoid : str, default="sphere"
        Geometry passed to :mod:`healpix_geo` when locating cell centres.
    max_patch_radius_deg : float, default=10
        Maximum angular distance of a cell centre from the spherical patch
        centre.  A single gnomonic plane is rejected above this radius.
    pixel_size_rad : float, optional
        Grid spacing in gnomonic coordinates.  The default is the square root
        of the HEALPix pixel area, ``sqrt(pi/3) / nside``.
    grid_size : int, optional
        Side of the square projected grid.  By default, the smallest power of
        two that covers the patch with a one-pixel margin is used.
    norm : {"forward", "backward", "ortho"}, default="ortho"
        Normalisation passed to :func:`torch.fft.fft2` and ``ifft2``.
    dtype : torch.dtype, default=torch.float32
        Real dtype used for geometry and real-valued input maps.
    device : torch.device or str, optional
        Device of the cached projection geometry.  CUDA is selected when it
        is available, otherwise CPU.

    Notes
    -----
    The tangent frame is built from three-dimensional unit vectors, not from
    longitude/latitude differences.  It is therefore well defined at both
    geographic poles and across the 0/360-degree meridian.

    For torch inputs, gradients propagate through projection, FFT, IFFT and
    back-projection with respect to the data.  The discrete HEALPix geometry
    is fixed and is not differentiable with respect to ``cell_ids``.
    """

    def __init__(
        self,
        cell_ids: ArrayLike,
        level: int,
        *,
        ellipsoid: str = "sphere",
        max_patch_radius_deg: float = 10.0,
        pixel_size_rad: Optional[float] = None,
        grid_size: Optional[int] = None,
        norm: str = "ortho",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if norm not in ("forward", "backward", "ortho"):
            raise ValueError("norm must be 'forward', 'backward', or 'ortho'")
        if int(level) != level or int(level) < 0:
            raise ValueError("level must be a non-negative integer")
        if max_patch_radius_deg <= 0 or max_patch_radius_deg >= 90:
            raise ValueError("max_patch_radius_deg must be in the interval (0, 90)")

        self.level = int(level)
        self.nside = 2**self.level
        self.ellipsoid = str(ellipsoid)
        self.max_patch_radius_deg = float(max_patch_radius_deg)
        self.norm = norm
        self.dtype = dtype
        self.cdtype = _complex_dtype(dtype)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        if torch.is_tensor(cell_ids):
            ids_np = cell_ids.detach().cpu().numpy()
        else:
            ids_np = np.asarray(cell_ids)
        ids_np = np.asarray(ids_np, dtype=np.int64).ravel()
        if ids_np.size == 0:
            raise ValueError("cell_ids must contain at least one cell")
        if np.unique(ids_np).size != ids_np.size:
            raise ValueError("cell_ids must be unique")
        npix = 12 * 4**self.level
        if np.any(ids_np < 0) or np.any(ids_np >= npix):
            raise ValueError(
                f"cell_ids must be valid NESTED identifiers at level {self.level} "
                f"(expected values in [0, {npix - 1}])"
            )

        self.n_cells = int(ids_np.size)
        self.register_buffer(
            "cell_ids", torch.as_tensor(ids_np, dtype=torch.long, device=resolved_device)
        )

        lon_deg, lat_deg = healpix_geo.nested.healpix_to_lonlat(
            ids_np, self.level, ellipsoid=self.ellipsoid
        )
        lon_deg = np.asarray(lon_deg, dtype=np.float64).ravel()
        lat_deg = np.asarray(lat_deg, dtype=np.float64).ravel()

        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        cos_lat = np.cos(lat)
        vectors = np.column_stack(
            (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat))
        )

        centre = vectors.mean(axis=0)
        centre_norm = float(np.linalg.norm(centre))
        if centre_norm < 1e-12:
            raise ValueError(
                "The spherical centre of cell_ids is undefined; the patch is not local."
            )
        centre /= centre_norm

        centre_dot = np.clip(vectors @ centre, -1.0, 1.0)
        angular_distance = np.arccos(centre_dot)
        radius_deg = float(np.rad2deg(angular_distance.max()))
        if radius_deg > self.max_patch_radius_deg + 1e-10:
            raise ValueError(
                f"The patch radius is {radius_deg:.6g} degrees, greater than "
                f"max_patch_radius_deg={self.max_patch_radius_deg:.6g}. "
                "Split the cells into smaller local patches."
            )

        # East/north-like tangent axes.  The fallback makes the frame stable
        # when the patch centre is at either geographic pole.
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(centre @ reference)) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        axis_x = np.cross(reference, centre)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(centre, axis_x)
        axis_y /= np.linalg.norm(axis_y)

        denominator = vectors @ centre
        projected_x = (vectors @ axis_x) / denominator
        projected_y = (vectors @ axis_y) / denominator

        if pixel_size_rad is None:
            pixel_size_rad = math.sqrt(math.pi / 3.0) / self.nside
        self.pixel_size_rad = float(pixel_size_rad)
        if not math.isfinite(self.pixel_size_rad) or self.pixel_size_rad <= 0:
            raise ValueError("pixel_size_rad must be finite and strictly positive")

        extent = float(max(np.max(np.abs(projected_x)), np.max(np.abs(projected_y))))
        required_size = max(4, int(math.ceil(2.0 * extent / self.pixel_size_rad)) + 3)
        if grid_size is None:
            grid_size = _next_power_of_two(required_size)
        elif int(grid_size) != grid_size or int(grid_size) < required_size:
            raise ValueError(
                f"grid_size must be an integer >= {required_size} for this patch"
            )
        self.grid_size = int(grid_size)
        self.grid_shape = (self.grid_size, self.grid_size)

        centre_index = 0.5 * (self.grid_size - 1)
        grid_x = projected_x / self.pixel_size_rad + centre_index
        grid_y = projected_y / self.pixel_size_rad + centre_index
        x0 = np.floor(grid_x).astype(np.int64)
        y0 = np.floor(grid_y).astype(np.int64)
        if (
            np.any(x0 < 0)
            or np.any(y0 < 0)
            or np.any(x0 + 1 >= self.grid_size)
            or np.any(y0 + 1 >= self.grid_size)
        ):
            raise RuntimeError("Internal error: the projected grid margin is insufficient")

        fraction_x = grid_x - x0
        fraction_y = grid_y - y0
        indices = np.stack(
            (
                y0 * self.grid_size + x0,
                y0 * self.grid_size + (x0 + 1),
                (y0 + 1) * self.grid_size + x0,
                (y0 + 1) * self.grid_size + (x0 + 1),
            ),
            axis=1,
        )
        weights = np.stack(
            (
                (1.0 - fraction_x) * (1.0 - fraction_y),
                fraction_x * (1.0 - fraction_y),
                (1.0 - fraction_x) * fraction_y,
                fraction_x * fraction_y,
            ),
            axis=1,
        )

        indices_t = torch.as_tensor(indices, dtype=torch.long, device=resolved_device)
        weights_t = torch.as_tensor(weights, dtype=self.dtype, device=resolved_device)
        self.register_buffer("_grid_indices", indices_t)
        self.register_buffer("_grid_weights", weights_t)

        density = torch.zeros(
            self.grid_size * self.grid_size, dtype=self.dtype, device=resolved_device
        )
        density.scatter_add_(0, indices_t.reshape(-1), weights_t.reshape(-1))
        self.register_buffer("_grid_density", density)
        inverse_density = torch.where(
            density > 0, density.reciprocal(), torch.zeros_like(density)
        )
        self.register_buffer("_grid_inverse_density", inverse_density)

        self.centre_lon_deg = float(np.rad2deg(math.atan2(centre[1], centre[0])) % 360.0)
        self.centre_lat_deg = float(np.rad2deg(math.asin(np.clip(centre[2], -1.0, 1.0))))
        self.patch_radius_deg = radius_deg
        self.register_buffer(
            "projected_x",
            torch.as_tensor(projected_x, dtype=self.dtype, device=resolved_device),
        )
        self.register_buffer(
            "projected_y",
            torch.as_tensor(projected_y, dtype=self.dtype, device=resolved_device),
        )

    @property
    def device(self) -> torch.device:
        """Current device of the cached geometry buffers."""
        return self.cell_ids.device

    @property
    def coverage_mask(self) -> torch.Tensor:
        """Boolean ``[grid_size, grid_size]`` mask reached by input cells."""
        return (self._grid_density > 0).reshape(self.grid_shape)

    @property
    def grid_density(self) -> torch.Tensor:
        """Accumulated interpolation weight on the projected square grid."""
        return self._grid_density.reshape(self.grid_shape)

    def _input_tensor(self, data: ArrayLike) -> tuple[torch.Tensor, bool]:
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy and not torch.is_tensor(data):
            raise TypeError("data must be a numpy.ndarray or torch.Tensor")
        target_dtype = self.cdtype if _is_complex_input(data) else self.dtype
        tensor = torch.as_tensor(data, device=self.cell_ids.device).to(dtype=target_dtype)
        if tensor.ndim < 1 or tensor.shape[-1] != self.n_cells:
            raise ValueError(
                f"The last data dimension must contain {self.n_cells} HEALPix values, "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor, is_numpy

    @staticmethod
    def _restore_type(tensor: torch.Tensor, is_numpy: bool) -> ArrayLike:
        if is_numpy:
            return tensor.detach().cpu().numpy()
        return tensor

    def _project_tensor(self, data: torch.Tensor) -> torch.Tensor:
        leading_shape = data.shape[:-1]
        flat_data = data.reshape(-1, self.n_cells)
        batch_size = flat_data.shape[0]
        sources = (flat_data.unsqueeze(-1) * self._grid_weights).reshape(batch_size, -1)
        indices = self._grid_indices.reshape(1, -1).expand(batch_size, -1)
        projected = torch.zeros(
            (batch_size, self.grid_size * self.grid_size),
            dtype=flat_data.dtype,
            device=flat_data.device,
        )
        projected.scatter_add_(1, indices, sources)
        projected = projected * self._grid_inverse_density
        return projected.reshape(*leading_shape, self.grid_size, self.grid_size)

    def project(self, data: ArrayLike) -> ArrayLike:
        """Bilinearly scatter HEALPix values onto the cached square grid."""
        tensor, is_numpy = self._input_tensor(data)
        return self._restore_type(self._project_tensor(tensor), is_numpy)

    def _unproject_tensor(self, grid: torch.Tensor) -> torch.Tensor:
        if grid.ndim < 2 or tuple(grid.shape[-2:]) != self.grid_shape:
            raise ValueError(
                f"The last two grid dimensions must be {self.grid_shape}, "
                f"got shape {tuple(grid.shape)}"
            )
        leading_shape = grid.shape[:-2]
        flat_grid = grid.reshape(-1, self.grid_size * self.grid_size)
        batch_size = flat_grid.shape[0]
        indices = self._grid_indices.reshape(1, -1).expand(batch_size, -1)
        samples = torch.gather(flat_grid, 1, indices).reshape(
            batch_size, self.n_cells, 4
        )
        samples = (samples * self._grid_weights).sum(dim=-1)
        return samples.reshape(*leading_shape, self.n_cells)

    def unproject(self, grid: ArrayLike) -> ArrayLike:
        """Bilinearly sample a projected grid at the HEALPix cell centres."""
        is_numpy = isinstance(grid, np.ndarray)
        if not is_numpy and not torch.is_tensor(grid):
            raise TypeError("grid must be a numpy.ndarray or torch.Tensor")
        target_dtype = self.cdtype if _is_complex_input(grid) else self.dtype
        tensor = torch.as_tensor(grid, device=self.cell_ids.device).to(dtype=target_dtype)
        return self._restore_type(self._unproject_tensor(tensor), is_numpy)

    def fft(self, data: ArrayLike) -> ArrayLike:
        """Project HEALPix data and compute its unshifted two-dimensional FFT."""
        tensor, is_numpy = self._input_tensor(data)
        spectrum = torch.fft.fft2(self._project_tensor(tensor), norm=self.norm)
        return self._restore_type(spectrum, is_numpy)

    def ifft(self, spectrum: ArrayLike, *, real_output: bool = True) -> ArrayLike:
        """Compute the inverse FFT and sample it at the input HEALPix cells.

        This is a fast approximate inverse of :meth:`fft`.  Set
        ``real_output=False`` when transforming genuinely complex-valued data.
        """
        is_numpy = isinstance(spectrum, np.ndarray)
        if not is_numpy and not torch.is_tensor(spectrum):
            raise TypeError("spectrum must be a numpy.ndarray or torch.Tensor")
        tensor = torch.as_tensor(spectrum, device=self.cell_ids.device).to(
            dtype=self.cdtype
        )
        if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != self.grid_shape:
            raise ValueError(
                f"The last two spectrum dimensions must be {self.grid_shape}, "
                f"got shape {tuple(tensor.shape)}"
            )
        grid = torch.fft.ifft2(tensor, norm=self.norm)
        reconstructed = self._unproject_tensor(grid)
        if real_output:
            reconstructed = reconstructed.real
        return self._restore_type(reconstructed, is_numpy)

    def extra_repr(self) -> str:
        return (
            f"level={self.level}, n_cells={self.n_cells}, "
            f"grid_shape={self.grid_shape}, radius={self.patch_radius_deg:.4g} deg, "
            f"ellipsoid={self.ellipsoid!r}, dtype={self.dtype}"
        )


def fft(
    cell_ids: ArrayLike,
    level: int,
    data: ArrayLike,
    *,
    return_transform: bool = False,
    **transform_kwargs,
):
    """Build a :class:`LocalFFT` and compute a local 2-D FFT.

    Use ``return_transform=True`` to return ``(spectrum, transform)``.  Reusing
    that transform for IFFT and additional maps avoids rebuilding geometry.
    """
    transform = LocalFFT(cell_ids, level, **transform_kwargs)
    spectrum = transform.fft(data)
    if return_transform:
        return spectrum, transform
    return spectrum


def ifft(
    spectrum: ArrayLike,
    transform: Optional[LocalFFT] = None,
    *,
    cell_ids: Optional[ArrayLike] = None,
    level: Optional[int] = None,
    real_output: bool = True,
    **transform_kwargs,
) -> ArrayLike:
    """Compute a local inverse FFT using cached or newly built geometry.

    Prefer passing the :class:`LocalFFT` returned by :func:`fft`.  Alternatively
    provide both ``cell_ids`` and ``level`` to rebuild the deterministic grid.
    """
    if transform is None:
        if cell_ids is None or level is None:
            raise ValueError("Pass transform, or provide both cell_ids and level")
        transform = LocalFFT(cell_ids, level, **transform_kwargs)
    elif cell_ids is not None or level is not None or transform_kwargs:
        raise ValueError(
            "cell_ids, level and transform options must be omitted when transform is given"
        )
    return transform.ifft(spectrum, real_output=real_output)


__all__ = ["LocalFFT", "fft", "ifft"]
