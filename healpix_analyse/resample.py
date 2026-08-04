"""Resampling helpers for HEALPix and regular latitude/longitude grids."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp


ArrayLike = Union[np.ndarray, torch.Tensor]


def _validate_level(level, name: str) -> int:
    if level is None:
        raise ValueError(
            f"{name} must be specified; use the corresponding cell_ids=None "
            "to represent a full-sphere map"
        )
    if isinstance(level, bool) or int(level) != level or int(level) < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return int(level)


def _validate_ids(cell_ids, *, level: int, name: str) -> Optional[np.ndarray]:
    if cell_ids is None:
        return None
    if torch.is_tensor(cell_ids):
        ids = cell_ids.detach().cpu().numpy()
    else:
        ids = np.asarray(cell_ids)
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.unique(ids).size != ids.size:
        raise ValueError(f"{name} must contain unique identifiers")
    npix = 12 * 4**level
    if np.any(ids < 0) or np.any(ids >= npix):
        raise ValueError(
            f"{name} contain identifiers outside level={level} "
            f"(expected values in [0, {npix - 1}])"
        )
    return ids


def _positions_for_ids(
    source_ids: np.ndarray,
    requested_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source positions and matching requested-output positions."""
    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, requested_ids)
    safe = np.clip(positions, 0, len(sorted_ids) - 1)
    available = (positions < len(sorted_ids)) & (
        sorted_ids[safe] == requested_ids
    )
    return order[safe[available]], np.flatnonzero(available)


class HealPixResampler(nn.Module):
    """Reusable NESTED HEALPix resampler based on ``HealPixDown``/``Up``.

    Parameters
    ----------
    in_level, out_level : int
        Input and output Grid4Earth/HEALPix levels.  The corresponding NSIDE
        values are ``2**in_level`` and ``2**out_level``.
    in_cell_ids : array-like, optional
        NESTED identifiers matching the last dimension of the input data.
        ``None`` means that the input is a complete sphere in canonical
        NESTED order.  ``in_level`` is still mandatory.
    out_cell_ids : array-like, optional
        Exact output cells, in the desired output order.  ``None`` requests
        the complete output sphere in canonical NESTED order.
    ellipsoid : str, default="WGS84"
        Geometry passed to every Down/Up operator.
    radius_deg, sigma_deg : float, optional
        Optional fixed Gaussian parameters passed to each resolution step.
        When omitted, every step uses its native resolution-dependent values.
    weight_norm : {"l1", "l2", "none"}, default="l1"
        Normalisation used by ``HealPixDown``.
    up_norm : {"adjoint", "col_l1", "diag_l2"}, default="col_l1"
        Normalisation used by ``HealPixUp``.
    dtype, device : optional
        PyTorch computation options.  CUDA is selected when available.

    Notes
    -----
    Data may have shape ``[..., N_in]``.  NumPy input returns NumPy output;
    Torch input remains differentiable and returns a tensor on ``device``.

    Missing or non-finite samples are excluded through normalised convolution.
    A target with at least one non-zero weighted support remains calculable;
    a target with no support is filled with NaN.

    Upsampling a requested subset first maps every requested output cell to its
    NESTED ancestor at ``in_level``.  Only available ancestors and their full
    descendant trees are propagated, after which the exact requested cells are
    gathered in their original order.
    """

    def __init__(
        self,
        in_level: int,
        out_level: int,
        in_cell_ids=None,
        out_cell_ids=None,
        *,
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.in_level = _validate_level(in_level, "in_level")
        self.out_level = _validate_level(out_level, "out_level")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if not np.isfinite(eps) or float(eps) <= 0:
            raise ValueError("eps must be a positive finite value")
        self.dtype = dtype
        self.eps = float(eps)
        self.ellipsoid = str(ellipsoid)
        self.weight_norm = str(weight_norm).lower().strip()
        self.up_norm = str(up_norm).lower().strip()
        if self.weight_norm not in ("l1", "l2", "none"):
            raise ValueError("weight_norm must be 'l1', 'l2', or 'none'")
        if self.up_norm not in ("adjoint", "col_l1", "diag_l2"):
            raise ValueError(
                "up_norm must be 'adjoint', 'col_l1', or 'diag_l2'"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        input_ids = _validate_ids(
            in_cell_ids, level=self.in_level, name="in_cell_ids"
        )
        requested_ids = _validate_ids(
            out_cell_ids, level=self.out_level, name="out_cell_ids"
        )
        self.input_full_sphere = input_ids is None
        self.output_full_sphere = requested_ids is None
        self.n_input = (
            12 * 4**self.in_level if input_ids is None else len(input_ids)
        )
        if requested_ids is None:
            requested_ids = np.arange(
                12 * 4**self.out_level, dtype=np.int64
            )
        self.n_output = len(requested_ids)
        self._input_ids = None if input_ids is None else input_ids.copy()
        self._output_ids = requested_ids.copy()

        self.register_buffer(
            "_dtype_anchor",
            torch.empty(0, dtype=dtype, device=resolved_device),
        )

        if self.in_level == self.out_level:
            self.direction = "identity"
        elif self.in_level > self.out_level:
            self.direction = "down"
        else:
            self.direction = "up"

        self.layers = nn.ModuleList()
        self._full_weight_names: list[str] = []
        self.empty_route = False

        if self.direction == "identity":
            if input_ids is None:
                source_positions = requested_ids.copy()
                output_positions = np.arange(self.n_output, dtype=np.int64)
            else:
                source_positions, output_positions = _positions_for_ids(
                    input_ids, requested_ids
                )
            self._register_long_buffer(
                "_route_input_positions", source_positions, resolved_device
            )
            self._register_long_buffer(
                "_route_output_positions", output_positions, resolved_device
            )
            self._register_long_buffer(
                "_final_source_positions",
                np.arange(len(source_positions), dtype=np.int64),
                resolved_device,
            )

        elif self.direction == "down":
            if input_ids is None:
                route_input_positions = np.arange(0, dtype=np.int64)
                current_ids = None
                current_size = self.n_input
            else:
                sort_order = np.argsort(input_ids)
                route_input_positions = sort_order.astype(np.int64)
                current_ids = input_ids[sort_order]
                current_size = len(current_ids)
            self._register_long_buffer(
                "_route_input_positions",
                route_input_positions,
                resolved_device,
            )

            current_level = self.in_level
            while current_level > self.out_level:
                layer = HealPixDown(
                    level=current_level,
                    mode="smooth",
                    ellipsoid=self.ellipsoid,
                    radius_deg=radius_deg,
                    sigma_deg=sigma_deg,
                    weight_norm=self.weight_norm,
                    cell_ids=current_ids,
                    dtype=dtype,
                    device=resolved_device,
                )
                self.layers.append(layer)
                self._register_full_weight(layer, current_size, resolved_device)
                current_ids = (
                    None
                    if current_ids is None
                    else np.asarray(layer.cell_ids_out, dtype=np.int64).copy()
                )
                current_size = layer.N_out
                current_level -= 1

            if current_ids is None:
                final_source_positions = requested_ids.copy()
                output_positions = np.arange(self.n_output, dtype=np.int64)
            else:
                final_source_positions, output_positions = _positions_for_ids(
                    current_ids, requested_ids
                )
            self._register_long_buffer(
                "_final_source_positions",
                final_source_positions,
                resolved_device,
            )
            self._register_long_buffer(
                "_route_output_positions", output_positions, resolved_device
            )

        else:  # up
            level_delta = self.out_level - self.in_level
            ancestors = requested_ids // (4**level_delta)
            needed_ancestors = np.unique(ancestors).astype(np.int64)

            route_full = self.input_full_sphere and self.output_full_sphere
            if route_full:
                route_input_positions = np.arange(0, dtype=np.int64)
                current_ids = None
                current_size = self.n_input
            elif input_ids is None:
                current_ids = needed_ancestors
                route_input_positions = current_ids.copy()
                current_size = len(current_ids)
            else:
                source_positions, _ = _positions_for_ids(
                    input_ids, needed_ancestors
                )
                # Recreate the selected IDs in the same sorted order expected
                # by the partial HealPixUp geometry.
                selected_ids = input_ids[source_positions]
                sort_order = np.argsort(selected_ids)
                current_ids = selected_ids[sort_order]
                route_input_positions = source_positions[sort_order]
                current_size = len(current_ids)

            self.empty_route = current_size == 0
            self._register_long_buffer(
                "_route_input_positions",
                route_input_positions,
                resolved_device,
            )

            current_level = self.in_level
            while current_level < self.out_level and not self.empty_route:
                layer = HealPixUp(
                    level=current_level,
                    ellipsoid=self.ellipsoid,
                    radius_deg=radius_deg,
                    sigma_deg=sigma_deg,
                    weight_norm=self.weight_norm,
                    up_norm=self.up_norm,
                    cell_ids=current_ids,
                    dtype=dtype,
                    device=resolved_device,
                )
                self.layers.append(layer)
                self._register_full_weight(layer, current_size, resolved_device)
                current_ids = (
                    None
                    if current_ids is None
                    else np.asarray(layer.cell_ids_out, dtype=np.int64).copy()
                )
                current_size = layer.N_out
                current_level += 1

            if self.empty_route:
                final_source_positions = np.empty(0, dtype=np.int64)
                output_positions = np.empty(0, dtype=np.int64)
            elif current_ids is None:
                final_source_positions = requested_ids.copy()
                output_positions = np.arange(self.n_output, dtype=np.int64)
            else:
                final_source_positions, output_positions = _positions_for_ids(
                    current_ids, requested_ids
                )
            self._register_long_buffer(
                "_final_source_positions",
                final_source_positions,
                resolved_device,
            )
            self._register_long_buffer(
                "_route_output_positions", output_positions, resolved_device
            )

    def _register_long_buffer(
        self, name: str, values: np.ndarray, device: torch.device
    ) -> None:
        self.register_buffer(
            name,
            torch.as_tensor(values, dtype=torch.long, device=device),
        )

    def _register_full_weight(
        self,
        layer: nn.Module,
        input_size: int,
        device: torch.device,
    ) -> None:
        with torch.no_grad():
            ones = torch.ones((1, input_size), dtype=self.dtype, device=device)
            full_weight, _ = layer(ones)
        name = f"_full_weight_{len(self._full_weight_names)}"
        self.register_buffer(name, full_weight.detach())
        self._full_weight_names.append(name)

    @property
    def device(self) -> torch.device:
        return self._dtype_anchor.device

    @property
    def in_cell_ids(self) -> Optional[np.ndarray]:
        """Input identifiers, or ``None`` for a full-sphere input."""
        return None if self._input_ids is None else self._input_ids.copy()

    @property
    def out_cell_ids(self) -> np.ndarray:
        """Exact output identifiers in returned-data order."""
        return self._output_ids.copy()

    def _apply(self, fn):
        result = super()._apply(fn)
        self.dtype = self._dtype_anchor.dtype
        for layer in self.layers:
            layer.device = self.device
            layer.dtype = self.dtype
        return result

    def _prepare_data(
        self, data: ArrayLike
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy and not torch.is_tensor(data):
            raise TypeError("in_data must be a numpy.ndarray or torch.Tensor")
        if np.iscomplexobj(data) if is_numpy else torch.is_complex(data):
            raise TypeError("in_data must be real-valued")
        tensor = torch.as_tensor(data, device=self.device).to(dtype=self.dtype)
        if tensor.ndim < 1 or tensor.shape[-1] != self.n_input:
            raise ValueError(
                f"in_data must end with {self.n_input} HEALPix values; "
                f"got shape {tuple(tensor.shape)}"
            )
        leading_shape = tuple(tensor.shape[:-1])
        return tensor.reshape(-1, self.n_input), leading_shape, is_numpy

    def _apply_layer_with_missing(
        self,
        layer: nn.Module,
        values: torch.Tensor,
        full_weight: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.isfinite(values)
        cleaned = torch.where(valid, values, torch.zeros_like(values))
        raw, _ = layer(cleaned)
        coverage, _ = layer(valid.to(dtype=values.dtype))
        full = full_weight.to(device=values.device, dtype=values.dtype)
        supported = coverage.abs() > self.eps
        safe_coverage = torch.where(
            supported, coverage, torch.ones_like(coverage)
        )
        normalised = raw * full / safe_coverage
        return torch.where(
            supported, normalised, torch.full_like(normalised, torch.nan)
        )

    def forward(self, in_data: ArrayLike) -> tuple[ArrayLike, np.ndarray]:
        """Resample data and return ``(out_data, out_cell_ids)``."""
        values, leading_shape, is_numpy = self._prepare_data(in_data)
        output = torch.full(
            (values.shape[0], self.n_output),
            torch.nan,
            dtype=self.dtype,
            device=self.device,
        )

        if not self.empty_route:
            if self.direction == "identity":
                route_values = values.index_select(
                    -1, self._route_input_positions
                )
            elif self.input_full_sphere and self._route_input_positions.numel() == 0:
                route_values = values
            else:
                route_values = values.index_select(
                    -1, self._route_input_positions
                )

            for index, layer in enumerate(self.layers):
                route_values = self._apply_layer_with_missing(
                    layer,
                    route_values,
                    getattr(self, self._full_weight_names[index]),
                )

            selected = route_values.index_select(
                -1, self._final_source_positions
            )
            output[:, self._route_output_positions] = selected

        restored = output.reshape(*leading_shape, self.n_output)
        if is_numpy:
            restored = restored.detach().cpu().numpy()
        return restored, self.out_cell_ids

    def compute(self, in_data: ArrayLike) -> tuple[ArrayLike, np.ndarray]:
        """Alias for :meth:`forward`."""
        return self.forward(in_data)

    def extra_repr(self) -> str:
        return (
            f"in_level={self.in_level}, out_level={self.out_level}, "
            f"direction={self.direction!r}, input_pixels={self.n_input}, "
            f"output_pixels={self.n_output}, steps={len(self.layers)}, "
            f"input_full_sphere={self.input_full_sphere}, "
            f"output_full_sphere={self.output_full_sphere}"
        )


def resample_healpix(
    in_data: ArrayLike,
    *,
    in_level: int,
    out_level: int,
    in_cell_ids=None,
    out_cell_ids=None,
    ellipsoid: str = "WGS84",
    radius_deg: Optional[float] = None,
    sigma_deg: Optional[float] = None,
    weight_norm: str = "l1",
    up_norm: str = "col_l1",
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[str, torch.device]] = None,
    eps: float = 1e-12,
) -> tuple[ArrayLike, np.ndarray]:
    """One-shot NESTED HEALPix resampling convenience function.

    Construct :class:`HealPixResampler` directly when several maps share the
    same input/output geometry, so the sparse operators can be reused.
    """
    operator = HealPixResampler(
        in_level=in_level,
        out_level=out_level,
        in_cell_ids=in_cell_ids,
        out_cell_ids=out_cell_ids,
        ellipsoid=ellipsoid,
        radius_deg=radius_deg,
        sigma_deg=sigma_deg,
        weight_norm=weight_norm,
        up_norm=up_norm,
        dtype=dtype,
        device=device,
        eps=eps,
    )
    return operator(in_data)


def resample_to_latlon_grid(lat, lon, data, method="linear"):
    """Resample HEALPix data onto a regular latitude/longitude grid."""
    # SciPy is optional for the HEALPix-to-HEALPix operators above.
    from scipy.interpolate import griddata

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon.min(), lon.max(), data.shape[1]),
        np.linspace(lat.min(), lat.max(), data.shape[0]),
    )
    points = np.column_stack((lat.flatten(), lon.flatten()))
    data_resampled = griddata(
        points,
        data.flatten(),
        (lat_grid.flatten(), lon_grid.flatten()),
        method=method,
        fill_value=data.mean(),
    )
    return data_resampled.reshape(lon_grid.shape)


__all__ = [
    "HealPixResampler",
    "resample_healpix",
    "resample_to_latlon_grid",
]
