"""Memory-efficient large-receptive-field convolution on HEALPix maps.

``LargeConv`` replaces a large fine-resolution stencil by a matched sequence
of smooth downsamplings, a compact :class:`HealPixConv`, and the corresponding
upsamplings.  The compact kernel is learned end-to-end through the complete
multiresolution operator.
"""

from __future__ import annotations

import math
import operator
import warnings
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from healpix_analyse.convol import HealPixConv, _DEFAULT_CACHE_DIR
from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp
from healpix_analyse._ellipsoid import canonicalize_ellipsoid


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_int(name: str, value, minimum: int) -> int:
    """Return ``value`` as an int, rejecting bools, floats, nan and inf."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        as_int = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}") from None
    if as_int < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(as_int)


def _kernel_plan(
    kernel_sz: int,
    max_compact_kernel_sz: int,
    n_levels: Optional[int] = None,
) -> tuple[int, int]:
    """Return ``(n_levels, compact_kernel_sz)`` from fine-scale radii.

    With ``n_levels=None`` the smallest number of levels whose compact kernel
    fits in ``max_compact_kernel_sz`` is chosen.  An explicit ``n_levels``
    is used as given and ``max_compact_kernel_sz`` is not applied to it.
    """
    if int(kernel_sz) != kernel_sz or int(kernel_sz) < 1 or int(kernel_sz) % 2 == 0:
        raise ValueError("kernel_sz must be a positive odd integer")
    if (
        int(max_compact_kernel_sz) != max_compact_kernel_sz
        or int(max_compact_kernel_sz) < 1
        or int(max_compact_kernel_sz) % 2 == 0
    ):
        raise ValueError("max_compact_kernel_sz must be a positive odd integer")

    fine_radius = (int(kernel_sz) - 1) // 2
    max_radius = (int(max_compact_kernel_sz) - 1) // 2
    if n_levels is not None:
        n_levels = _as_int("n_levels", n_levels, 0)
        return n_levels, 2 * math.ceil(fine_radius / (2**n_levels)) + 1
    if fine_radius == 0:
        return 0, 1
    if max_radius == 0:
        raise ValueError(
            "max_compact_kernel_sz=1 can only represent kernel_sz=1"
        )

    n_levels = 0
    compact_radius = fine_radius
    while compact_radius > max_radius:
        n_levels += 1
        compact_radius = math.ceil(fine_radius / (2**n_levels))
    return n_levels, 2 * compact_radius + 1


def _prepare_input(
    x: ArrayLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    in_channels: int,
    n_pixels: int,
) -> tuple[torch.Tensor, bool, bool]:
    """Normalise input to ``[B, C, N]`` and retain restoration metadata."""
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        tensor = torch.as_tensor(x, dtype=dtype, device=device)
    elif torch.is_tensor(x):
        tensor = x.to(device=device, dtype=dtype)
    else:
        raise TypeError("x must be a numpy.ndarray or torch.Tensor")

    was_1d = tensor.ndim == 1
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim != 3:
        raise ValueError(
            "Input must have shape [N], [B, N] or [B, C, N], "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[1] != in_channels:
        raise ValueError(
            f"Expected in_channels={in_channels}, got {tensor.shape[1]}"
        )
    if tensor.shape[2] != n_pixels:
        raise ValueError(f"Expected {n_pixels} HEALPix values, got {tensor.shape[2]}")
    return tensor, is_numpy, was_1d


def _restore_output(
    tensor: torch.Tensor,
    *,
    is_numpy: bool,
    was_1d: bool,
) -> ArrayLike:
    if was_1d:
        tensor = tensor.squeeze(0)
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
    if is_numpy:
        return tensor.detach().cpu().numpy()
    return tensor


class LargeConv(nn.Module):
    """Learn a large effective HEALPix convolution through a compact kernel.

    The linear part of the layer is

    ``Up[0] ... Up[L-1] @ Conv_compact @ Down[L-1] ... Down[0]``.

    ``kernel_sz`` describes the requested fine-resolution receptive-field
    diameter.  The number of levels and compact odd kernel size are selected
    automatically so that ``compact_kernel_sz <= max_compact_kernel_sz``.
    For example, ``kernel_sz=33`` and ``max_compact_kernel_sz=7`` produce
    three down/up levels and a compact ``5 x 5`` kernel.

    Parameters
    ----------
    level : int
        Fine input Grid4Earth/HEALPix level (integer >= 0).  Internally,
        ``nside = 2**level``.
    in_channels, out_channels : int
        Input and output channels per gauge.
    kernel_sz : int
        Requested positive odd fine-resolution kernel diameter.
    max_compact_kernel_sz : int, optional
        Maximum positive odd kernel diameter at the coarse resolution
        (7 when neither this nor ``n_levels`` is given).  Cannot be combined
        with ``n_levels``.
    n_levels : int, optional
        Number of Down/Up levels.  By default it is chosen from
        ``kernel_sz`` and ``max_compact_kernel_sz`` as described above.
        When given, it is used as is and the compact kernel size follows
        from ``kernel_sz``.  Each level makes the output at least as wide
        as :func:`down_up_blur`; to build a Gaussian of a given width use
        :func:`gaussian_plan`.
    n_gauges, gauge_type, singularity_lonlat, ref_direction :
        Passed to the internal :class:`HealPixConv`.
    cell_ids : array-like, optional
        Fine-resolution NESTED cell identifiers.  Partial patches are sorted
        internally and outputs are restored to the original input order.
    ellipsoid : str, default="WGS84"
        Shared by every Down, Up and convolution geometry operation.
    radius_deg, sigma_deg, weight_norm :
        Identical smoothing parameters used in every matched Down/Up pair.
    up_norm : {"adjoint", "col_l1", "diag_l2"}, default="col_l1"
        Upsampling normalisation.  ``col_l1`` preserves constant fields when
        paired with ``weight_norm="l1"``.
    use_norm : bool, default=False
        Apply GroupNorm and ReLU after the final fine-resolution upsampling.
    dtype, device, cache_dir :
        PyTorch and HealPixConv execution options.

    Notes
    -----
    The layer learns the compact kernel directly; it does not claim exact
    equivalence to an arbitrary supplied large kernel.  Downsampling removes
    high-frequency information, so the representable large kernels are those
    in the multiresolution subspace defined by the selected Down/Up operators.
    """

    def __init__(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int,
        *,
        max_compact_kernel_sz: Optional[int] = None,
        n_levels: Optional[int] = None,
        n_gauges: int = 1,
        gauge_type: str = "phi",
        singularity_lonlat: Optional[tuple[float, float]] = None,
        ref_direction=None,
        cell_ids=None,
        nest: bool = True,
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        use_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: "Path | str | None" = _DEFAULT_CACHE_DIR,
    ) -> None:
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        super().__init__()

        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0")
        self.level = int(level)
        self.nside = 2 ** self.level
        if int(in_channels) < 1 or int(out_channels) < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if not nest:
            raise ValueError("LargeConv currently requires NESTED HEALPix ordering")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_sz = int(kernel_sz)
        if n_levels is not None and max_compact_kernel_sz is not None:
            raise ValueError(
                "n_levels and max_compact_kernel_sz cannot both be given: "
                "n_levels fixes the number of levels and the compact kernel "
                "size follows from kernel_sz"
            )
        self.max_compact_kernel_sz = (
            7 if max_compact_kernel_sz is None else int(max_compact_kernel_sz)
        )
        self.n_levels, self.compact_kernel_sz = _kernel_plan(
            self.kernel_sz, self.max_compact_kernel_sz, n_levels
        )
        if self.nside // (2**self.n_levels) < 1:
            raise ValueError(
                f"kernel_sz={self.kernel_sz} requires {self.n_levels} downsamplings, "
                f"which is incompatible with nside={self.nside}"
            )

        self.coarse_nside = self.nside // (2**self.n_levels)
        compact_radius = (self.compact_kernel_sz - 1) // 2
        self.effective_kernel_sz = 2 * compact_radius * (2**self.n_levels) + 1
        self.G = max(1, int(n_gauges))
        self.ellipsoid = str(ellipsoid)
        self.weight_norm = str(weight_norm)
        self.up_norm = str(up_norm)
        self.radius_deg = radius_deg
        self.sigma_deg = sigma_deg
        self.use_norm = bool(use_norm)
        self.dtype = dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        self.partial = cell_ids is not None
        if self.partial:
            original_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
            if original_ids.size == 0:
                raise ValueError("cell_ids must not be empty")
            if np.unique(original_ids).size != original_ids.size:
                raise ValueError("cell_ids must be unique")
            npix = 12 * self.nside**2
            if np.any(original_ids < 0) or np.any(original_ids >= npix):
                raise ValueError("cell_ids contain invalid identifiers")
            input_sort = np.argsort(original_ids)
            input_unsort = np.argsort(input_sort)
            fine_ids = original_ids[input_sort]
        else:
            fine_ids = np.arange(12 * self.nside**2, dtype=np.int64)
            input_sort = np.arange(fine_ids.size, dtype=np.int64)
            input_unsort = input_sort.copy()

        self.n_pixels = int(fine_ids.size)
        self.register_buffer(
            "cell_ids",
            torch.as_tensor(fine_ids, dtype=torch.long, device=resolved_device),
        )
        self.register_buffer(
            "_input_sort",
            torch.as_tensor(input_sort, dtype=torch.long, device=resolved_device),
        )
        self.register_buffer(
            "_input_unsort",
            torch.as_tensor(input_unsort, dtype=torch.long, device=resolved_device),
        )

        self.down_layers = nn.ModuleList()
        level_ids = [fine_ids]
        current_ids = fine_ids
        current_level = self.level
        for _ in range(self.n_levels):
            down = HealPixDown(
                level=current_level,
                mode="smooth",
                ellipsoid=self.ellipsoid,
                radius_deg=radius_deg,
                sigma_deg=sigma_deg,
                weight_norm=self.weight_norm,
                cell_ids=current_ids if self.partial else None,
                device=resolved_device,
                dtype=self.dtype,
            )
            self.down_layers.append(down)
            current_ids = np.asarray(down.cell_ids_out, dtype=np.int64)
            level_ids.append(current_ids)
            current_level -= 1

        self.conv = HealPixConv(
            level=current_level,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_sz=self.compact_kernel_sz,
            n_gauges=self.G,
            gauge_type=gauge_type,
            singularity_lonlat=singularity_lonlat,
            ref_direction=ref_direction,
            cell_ids=current_ids if self.partial else None,
            nest=True,
            use_norm=False,
            device=resolved_device,
            ellipsoid=self.ellipsoid,
            dtype=self.dtype,
            cache_dir=cache_dir,
        )
        with torch.no_grad():
            self.conv.bias.zero_()
        self.conv.bias.requires_grad_(False)

        self.up_layers = nn.ModuleList()
        for fine_stage in range(self.n_levels - 1, -1, -1):
            up = HealPixUp(
                level=self.level - fine_stage - 1,
                ellipsoid=self.ellipsoid,
                up_norm=self.up_norm,
                device=resolved_device,
                dtype=self.dtype,
                paired_down=self.down_layers[fine_stage],
            )
            self.up_layers.append(up)

        self.bias = nn.Parameter(
            torch.zeros(self.G * self.out_channels, device=resolved_device, dtype=dtype)
        )
        if self.use_norm:
            total_channels = self.G * self.out_channels
            groups = min(8, total_channels)
            while total_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, total_channels).to(
                device=resolved_device, dtype=dtype
            )
        else:
            self.norm = None

    @property
    def device(self) -> torch.device:
        """Current device of the layer buffers."""
        return self.cell_ids.device

    @property
    def weight(self) -> nn.Parameter:
        """Learnable compact convolution kernel."""
        return self.conv.weight

    def _apply(self, fn):
        result = super()._apply(fn)
        current_device = self.cell_ids.device
        current_dtype = self.conv.weight.dtype
        self.dtype = current_dtype
        for module in [*self.down_layers, self.conv, *self.up_layers]:
            module.device = current_device
            module.dtype = current_dtype
        return result

    def set_compact_kernel(self, weight, bias=None, requires_grad=False):
        """Set compact weights with the same shapes accepted by HealPixConv."""
        self.conv.set_kernel(weight, bias=None, requires_grad=requires_grad)
        self.conv.bias.requires_grad_(False)
        with torch.no_grad():
            if bias is None:
                self.bias.zero_()
            else:
                bias_tensor = torch.as_tensor(
                    bias, dtype=self.dtype, device=self.device
                ).ravel()
                if bias_tensor.numel() != self.bias.numel():
                    raise ValueError(f"bias must contain {self.bias.numel()} values")
                self.bias.copy_(bias_tensor)
        self.bias.requires_grad_(requires_grad)
        return self

    def set_gaussian_kernel(self, compact_sigma):
        """Set an isotropic Gaussian compact kernel, normalised to sum 1.

        ``compact_sigma`` is the width in cells of the coarse level (one
        axis), or a :class:`GaussianPlan`, in which case the layer must have
        the plan's ``n_levels`` and a compact kernel at least ``plan.kernel_sz``
        wide (build it with ``LargeConv.from_gaussian``, or pass
        ``n_levels=plan.n_levels`` and ``plan.fine_kernel_sz``), and the
        level, ellipsoid, ``radius_deg``, ``sigma_deg``, ``weight_norm`` and
        ``up_norm`` the plan was measured with.  The Gaussian
        is sampled on the whole compact kernel grid and must fit in it
        (radius of at least four sigmas).  Each channel is smoothed on its
        own, so ``in_channels`` must equal ``out_channels``.
        """
        if isinstance(compact_sigma, GaussianPlan):
            plan = compact_sigma
            if self.n_levels != plan.n_levels:
                raise ValueError(
                    f"the plan needs n_levels={plan.n_levels}, this layer has "
                    f"{self.n_levels}: build it with n_levels=plan.n_levels and "
                    "kernel_sz=plan.fine_kernel_sz, or use LargeConv.from_gaussian"
                )
            mine = _layer_options(self)
            if mine != plan.options:
                names = ("level", "ellipsoid", "radius_deg", "sigma_deg",
                         "weight_norm", "up_norm")
                diff = ", ".join(
                    f"{k}={a!r} (plan: {b!r})"
                    for k, a, b in zip(names, mine, plan.options) if a != b
                )
                raise ValueError(
                    f"the plan was measured with other options than this layer: {diff}; "
                    "the width would differ, use LargeConv.from_gaussian or the "
                    "same options in gaussian_plan"
                )
            compact_sigma = plan.compact_sigma
        if self.in_channels != self.out_channels:
            raise ValueError("set_gaussian_kernel needs in_channels == out_channels")
        compact_sigma = float(compact_sigma)
        if not (math.isfinite(compact_sigma) and compact_sigma > 0.0):
            raise ValueError("compact_sigma must be a positive finite number")
        radius = (self.compact_kernel_sz - 1) // 2
        if radius < math.ceil(4.0 * compact_sigma):
            raise ValueError(
                f"the compact kernel ({self.compact_kernel_sz} x {self.compact_kernel_sz}) "
                f"is too small for a Gaussian of {compact_sigma:.4g} coarse cells: it "
                f"needs a radius of {math.ceil(4.0 * compact_sigma)}; build the layer "
                "with kernel_sz=plan.fine_kernel_sz (not plan.kernel_sz)"
            )
        grid = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(grid, grid)
        w = np.exp(-(xx**2 + yy**2).ravel() / (2.0 * compact_sigma**2))
        w /= w.sum()
        weight = np.zeros((self.in_channels, self.out_channels, w.size))
        for c in range(self.in_channels):
            weight[c, c] = w
        return self.set_compact_kernel(weight)

    @classmethod
    def from_gaussian(
        cls,
        level: int,
        sigma: Optional[float] = None,
        cell_ids=None,
        *,
        sigma_m: Optional[float] = None,
        channels: int = 1,
        reference_cell: Optional[int] = None,
        max_levels: Optional[int] = None,
        patch_depth: int = 8,
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "LargeConv":
        """Gaussian smoothing layer of width ``sigma`` (radians, one-axis RMS).

        The width is given in radians as ``sigma`` or in metres as
        ``sigma_m`` (the standard deviation of ``gaussian_filter``, converted
        with its cell size: square root of the cell area on the authalic
        sphere).

        One call: it computes :func:`gaussian_plan`, builds the layer with the
        same options and sets the kernel.  The calibration location is the
        block that contains ``reference_cell``; by default a cell at the
        centre of ``cell_ids`` (the equatorial block when ``cell_ids`` is
        ``None``).  ``channels`` smooths each channel on its own; other
        keyword arguments (``n_gauges``, ``gauge_type``, ``cache_dir`` ...)
        go to :class:`LargeConv`.  See :func:`gaussian_plan` for the accuracy
        and the limits.
        """
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        if reference_cell is None and cell_ids is not None:
            # here, so that a wide-domain warning points at the caller
            reference_cell = _centre_cell(level, cell_ids, str(ellipsoid))
        plan = gaussian_plan(
            level, sigma, sigma_m=sigma_m, max_levels=max_levels, ellipsoid=ellipsoid,
            radius_deg=radius_deg, sigma_deg=sigma_deg, weight_norm=weight_norm,
            up_norm=up_norm, patch_depth=patch_depth,
            reference_cell=reference_cell, cell_ids=cell_ids,
        )
        layer = cls(
            level, channels, channels, plan.fine_kernel_sz, n_levels=plan.n_levels,
            cell_ids=cell_ids, ellipsoid=ellipsoid, radius_deg=radius_deg,
            sigma_deg=sigma_deg, weight_norm=weight_norm, up_norm=up_norm,
            dtype=dtype, device=device, **kwargs,
        )
        layer.set_gaussian_kernel(plan)
        layer.gaussian_plan = plan
        return layer

    def forward(self, x: ArrayLike) -> ArrayLike:
        tensor, is_numpy, was_1d = _prepare_input(
            x,
            device=self.device,
            dtype=self.dtype,
            in_channels=self.in_channels,
            n_pixels=self.n_pixels,
        )
        tensor = tensor.index_select(-1, self._input_sort)
        batch_size, channels, _ = tensor.shape

        for down in self.down_layers:
            flattened = tensor.reshape(batch_size * channels, tensor.shape[-1])
            flattened, _ = down(flattened)
            tensor = flattened.reshape(batch_size, channels, -1)

        tensor = self.conv(tensor)
        output_channels = self.G * self.out_channels

        for up in self.up_layers:
            flattened = tensor.reshape(batch_size * output_channels, tensor.shape[-1])
            flattened, _ = up(flattened)
            tensor = flattened.reshape(batch_size, output_channels, -1)

        tensor = tensor.index_select(-1, self._input_unsort)
        tensor = tensor + self.bias.view(1, -1, 1)
        if self.use_norm and self.norm is not None:
            tensor = F.relu(self.norm(tensor), inplace=False)
        return _restore_output(tensor, is_numpy=is_numpy, was_1d=was_1d)

    def extra_repr(self) -> str:
        return (
            f"level={self.level}, nside={self.nside}, in={self.in_channels}, out={self.out_channels}, "
            f"requested_kernel={self.kernel_sz}, compact_kernel={self.compact_kernel_sz}, "
            f"levels={self.n_levels}, effective_kernel={self.effective_kernel_sz}, "
            f"gauges={self.G}, partial={self.partial}"
        )


_PATCH_MARGIN = 3  # n_levels <= patch_depth - 3: blur ~ 2**n cells << 2**depth
_COMPACT_SIGMA_START = 0.5  # coarse cells: smallest compact sigma used by gaussian_plan
_COMPACT_SIGMA_MAX = 2.25  # coarse cells: largest compact sigma calibrated
_COMPACT_SIGMA_STEP = 0.05
# The calibration measures the width on one patch, with zero padding at its
# border: the tail beyond the border is lost and biases the measured width,
# so the plan would set a compact Gaussian that is too wide on a larger
# domain.  Measured at level 10 (calibration patch 256 x 256, response
# re-measured on the 512 x 512 block around it, same centre cell): border at
# 4 sigmas or more -> at most 3e-4 relative; 3.2 -> 4.4e-3; 2.1 -> 0.11.
_EDGE_SIGMAS = 4.0
# Above this angular reach of cell_ids from the chosen calibration cell, warn:
# calibrated at the equator, 55 degrees latitude is up to 22 % too wide.
_WIDE_DOMAIN_DEG = 30.0


def _check_options(level, n_levels, patch_depth):
    level = _as_int("level", level, 0)
    n_levels = _as_int("n_levels", n_levels, 0)
    patch_depth = _as_int("patch_depth", patch_depth, 1)
    depth = min(patch_depth, level)
    if n_levels > level:
        raise ValueError(f"n_levels={n_levels} exceeds level={level}")
    if n_levels > depth - _PATCH_MARGIN:
        raise ValueError(
            f"n_levels={n_levels} is too large for a patch of 4**{depth} cells "
            f"(level={level}, patch_depth={patch_depth}): the blur would reach "
            f"the patch edge; at most {max(depth - _PATCH_MARGIN, 0)} levels "
            "can be measured, use a larger level or patch_depth"
        )
    return level, n_levels, patch_depth, depth


@lru_cache(maxsize=16)
def _patch(level, depth, ellipsoid, parent=None):
    """Cells of a 2**depth square patch, centre cell, angular distances.

    By default the patch is the one in the middle of base face 5 (equatorial).
    ``parent`` is the index of the aligned block (at level ``level - depth``)
    that makes the patch.
    """
    import healpix_geo.nested as hgn

    parent_level = level - depth
    if parent is None:
        parent = 5 if parent_level == 0 else 5 * 4**parent_level + 3 * 4 ** (parent_level - 1)
    ids = parent * 4**depth + np.arange(4**depth, dtype=np.int64)
    lon, lat = hgn.healpix_to_lonlat(ids, level, ellipsoid=ellipsoid)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    centre = int(np.argmax(xyz @ xyz.mean(axis=0)))
    dist = np.arccos(np.clip(xyz @ xyz[centre], -1.0, 1.0))
    return ids, centre, dist


@lru_cache(maxsize=16)
def _edge_distance(level, depth, ellipsoid, parent=None):
    """Angular distance from the centre cell of a patch to its border cells."""
    ids, centre, dist = _patch(level, depth, ellipsoid, parent)
    local = np.arange(4**depth, dtype=np.int64)
    x = np.zeros_like(local)
    y = np.zeros_like(local)
    for b in range(depth):
        x |= ((local >> (2 * b)) & 1) << b
        y |= ((local >> (2 * b + 1)) & 1) << b
    side = 2**depth - 1
    border = (x == 0) | (x == side) | (y == 0) | (y == side)
    return float(dist[border].min())


def _layer_options(layer):
    return (layer.level, layer.ellipsoid, getattr(layer, "radius_deg", None),
            getattr(layer, "sigma_deg", None), layer.weight_norm, layer.up_norm)


def _impulse_width(layer, centre, dist, n_cells):
    impulse = np.zeros(n_cells)
    impulse[centre] = 1.0
    w = np.asarray(layer(impulse), dtype=np.float64).ravel()
    return float(np.sqrt(np.sum(w * dist**2) / (2.0 * np.sum(w))))


def _measure_layer(level, n_levels, depth, kernel_sz, options):
    ellipsoid, radius_deg, sigma_deg, weight_norm, up_norm, parent = options
    ids, centre, dist = _patch(level, depth, ellipsoid, parent)
    layer = LargeConv(
        level, 1, 1, kernel_sz, n_levels=n_levels, cell_ids=ids,
        ellipsoid=ellipsoid, radius_deg=radius_deg, sigma_deg=sigma_deg,
        weight_norm=weight_norm, up_norm=up_norm,
        dtype=torch.float64, device="cpu", cache_dir=None,
    )
    return layer, centre, dist, ids.size


def _centre_cell(level, cell_ids, ellipsoid, stacklevel=2):
    """NESTED id of the cell of ``cell_ids`` closest to their mean direction.

    A wide domain gives a warning attributed ``stacklevel`` frames above
    this function (2: the caller of the public function that called it).
    """
    import healpix_geo.nested as hgn

    ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError("cell_ids must not be empty")
    if ids.min() < 0 or ids.max() >= 12 * 4**level:
        raise ValueError("cell_ids contain invalid identifiers")
    lon, lat = hgn.healpix_to_lonlat(ids, level, ellipsoid=ellipsoid)
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )
    k = int(np.argmax(xyz @ xyz.mean(axis=0)))
    spread = float(np.degrees(np.arccos(np.clip(np.min(xyz @ xyz[k]), -1.0, 1.0))))
    if spread > _WIDE_DOMAIN_DEG:
        warnings.warn(
            f"cell_ids reach {spread:.0f} degrees from the cell chosen for the "
            f"calibration ({int(ids[k])}); the width is calibrated there only and "
            "can be off by about 20 % at other latitudes (see docs/conv/large_conv.md); "
            "pass reference_cell, or smooth each region with its own layer",
            stacklevel=stacklevel + 1,
        )
    return int(ids[k])


def _options(
    ellipsoid, radius_deg, sigma_deg, weight_norm, up_norm, level, depth, reference_cell
):
    parent_level = level - depth
    parent = 5 if parent_level == 0 else 5 * 4**parent_level + 3 * 4 ** (parent_level - 1)
    if reference_cell is not None:
        cell = _as_int("reference_cell", reference_cell, 0)
        if cell >= 12 * 4**level:
            raise ValueError(f"reference_cell must be below 12 * 4**{level}")
        parent = cell // 4**depth
    return (str(ellipsoid), radius_deg, sigma_deg, str(weight_norm), str(up_norm), parent)


@lru_cache(maxsize=64)
def _blur_cached(level, n_levels, depth, options):
    layer, centre, dist, n = _measure_layer(level, n_levels, depth, 1, options)
    layer.set_compact_kernel(np.ones((1, 1, 1)))
    return _impulse_width(layer, centre, dist, n)


def down_up_blur(
    level: int,
    n_levels: int,
    *,
    ellipsoid: str = "WGS84",
    radius_deg: Optional[float] = None,
    sigma_deg: Optional[float] = None,
    weight_norm: str = "l1",
    up_norm: str = "col_l1",
    patch_depth: int = 8,
    reference_cell: Optional[int] = None,
) -> float:
    """Narrowest response of a :class:`LargeConv` with ``n_levels`` levels.

    The compact kernel is point-like (``kernel_sz=1``, weight 1) and the
    response to a single cell is measured.  It therefore contains the
    ``n_levels`` Down/Up pairs **and** the compact ``HealPixConv`` itself,
    which is the identity for a point-like kernel: at ``n_levels=0`` the
    width is 0.  The result is in **radians**, as the one-axis RMS width
    ``sqrt(sum(w d**2) / (2 sum(w)))`` of the response ``w`` with ``d`` the
    angular distance to the impulse cell (not a FWHM; FWHM is 2.355 times
    larger for a Gaussian).  To read it in cells divide by the nominal cell
    size ``sqrt(4 pi / 12) / 2**level``.

    It is measured on a patch of ``4**patch_depth`` cells, by default in the
    middle of base face 5 (near the equator), or on the aligned block that
    contains ``reference_cell`` (a NESTED id at ``level``), and cached per
    ``(level, n_levels, options)``.  ``n_levels`` must be at
    most ``min(patch_depth, level) - 3``, otherwise the blur would reach the
    patch edge and a ``ValueError`` is raised.
    """
    ellipsoid = canonicalize_ellipsoid(ellipsoid)
    level, n_levels, patch_depth, depth = _check_options(level, n_levels, patch_depth)
    opts = _options(
        ellipsoid, radius_deg, sigma_deg, weight_norm, up_norm, level, depth, reference_cell
    )
    return _blur_cached(level, n_levels, depth, opts)


class GaussianPlan(NamedTuple):
    """Result of :func:`gaussian_plan`."""

    n_levels: int
    """Number of Down/Up levels, for ``LargeConv(n_levels=...)``."""
    compact_sigma: float
    """Gaussian width in coarse cells, for ``LargeConv.set_gaussian_kernel``."""
    kernel_sz: int
    """Odd compact kernel size that covers 5 compact sigmas."""
    fine_kernel_sz: int
    """``kernel_sz`` argument of ``LargeConv`` that gives that compact size."""
    sigma: float
    """The requested width (radians)."""
    options: tuple
    """``(level, ellipsoid, radius_deg, sigma_deg, weight_norm, up_norm)`` of
    the calibration; ``set_gaussian_kernel`` refuses a layer with others."""


@lru_cache(maxsize=32)
def _calibration(level, n_levels, depth, options):
    """Width (rad) of the response for compact sigma 0, step, ..., max (cached)."""
    radius = math.ceil(5.0 * _COMPACT_SIGMA_MAX)
    kernel_sz = (2 * radius) * 2**n_levels + 1
    layer, centre, dist, n = _measure_layer(level, n_levels, depth, kernel_sz, options)
    assert layer.compact_kernel_sz == 2 * radius + 1
    sigmas = np.arange(
        0.0, _COMPACT_SIGMA_MAX + 0.5 * _COMPACT_SIGMA_STEP, _COMPACT_SIGMA_STEP
    )
    widths = np.empty_like(sigmas)
    for k, sc in enumerate(sigmas):
        if sc == 0.0:
            layer.set_compact_kernel(
                np.eye(1, layer.conv.P, layer.conv.P // 2).reshape(1, 1, -1)
            )
        else:
            layer.set_gaussian_kernel(sc)
        widths[k] = _impulse_width(layer, centre, dist, n)
    start = int(round(_COMPACT_SIGMA_START / _COMPACT_SIGMA_STEP))
    if not np.all(np.diff(widths[start:]) > 0.0):  # pragma: no cover
        raise RuntimeError("calibration of the compact Gaussian is not monotone")
    return sigmas, widths


def _plan_state(level, depth, max_levels):
    top = min(level, depth - _PATCH_MARGIN)
    if max_levels is not None:
        top = min(top, _as_int("max_levels", max_levels, 1))
    if top < 1:
        raise ValueError(
            f"level={level} is too small for LargeConv Gaussians: at least one "
            "level needs a patch of 2**4 cells; use gaussian_filter"
        )
    return top


def _resolve_sigma(level: int, sigma, sigma_m) -> float:
    """Width in radians from ``sigma`` (radians) or ``sigma_m`` (metres).

    Metres are converted with the cell size of ``gaussian_filter`` (square
    root of the cell area on the authalic sphere), so that ``sigma_m`` means
    the same here and there.
    """
    if (sigma is None) == (sigma_m is None):
        raise ValueError(
            "sigma must be given as exactly one of sigma (radians) or sigma_m (metres)"
        )
    if sigma_m is not None:
        from healpix_analyse.radial_filter import _cell_size_m

        try:
            sigma_m = float(sigma_m)
        except (TypeError, ValueError):
            raise ValueError(f"sigma_m must be a positive finite number, got {sigma_m!r}") from None
        if not (math.isfinite(sigma_m) and sigma_m > 0.0):
            raise ValueError(f"sigma_m must be a positive finite number, got {sigma_m!r}")
        level = _as_int("level", level, 0)
        return sigma_m / _cell_size_m(level) * _nominal_cell(level)
    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}") from None
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"sigma must be a positive finite number, got {sigma!r}")
    return sigma


def gaussian_plan(
    level: int,
    sigma: Optional[float] = None,
    *,
    sigma_m: Optional[float] = None,
    max_levels: Optional[int] = None,
    ellipsoid: str = "WGS84",
    radius_deg: Optional[float] = None,
    sigma_deg: Optional[float] = None,
    weight_norm: str = "l1",
    up_norm: str = "col_l1",
    patch_depth: int = 8,
    reference_cell: Optional[int] = None,
    cell_ids=None,
) -> GaussianPlan:
    """Levels and compact Gaussian that give a response of width ``sigma``.

    ``sigma`` is the one-axis RMS width in **radians** (the standard
    deviation of a Gaussian, as in :func:`down_up_blur`); ``sigma_m`` is the
    same width in **metres**, the ``sigma_m`` of ``gaussian_filter``, which
    is converted with the same cell size (square root of the cell area on the
    authalic sphere); give exactly one of them.  This function is for inspection;
    to build a layer use ``LargeConv.from_gaussian(level, sigma, cell_ids)``,
    which is the same as::

        plan = gaussian_plan(level, sigma, cell_ids=cell_ids)
        layer = LargeConv(level, 1, 1, plan.fine_kernel_sz,
                          n_levels=plan.n_levels, cell_ids=cell_ids)
        layer.set_gaussian_kernel(plan)

    Method.  For each number of levels ``n`` the width of the real
    ``LargeConv`` response to one cell is measured for compact Gaussians of
    width ``0, 0.05, ..., 2.25`` coarse cells (cached per level, ``n`` and
    options).  This includes the blur of the Down/Up pairs and of the compact
    convolution and the effect of sampling the Gaussian on the coarse grid,
    so no analytic model is assumed.  The number of levels grows from ``n-1``
    to ``n`` where level ``n`` reaches ``sigma`` with a compact Gaussian of
    0.5 coarse cell (just wide enough that the width still grows with it),
    and the compact width is the one for which the measured width equals
    ``sigma``.  Both branches give ``sigma`` at the switch, so the width is
    continuous and strictly increasing in ``sigma``.

    The smallest width of this class is the one of one level with a compact
    Gaussian of 0.5 coarse cell (about 2.2 cells at level 10).  Below it a
    ``ValueError`` is raised; use ``gaussian_filter`` for narrower
    widths.  The largest is set by ``max_levels`` or the patch
    (``min(level, patch_depth) - 3`` levels), the largest calibrated
    compact width, and the patch border: the width must be at most a quarter
    of the distance from the calibration cell to the border of the patch
    (31.7 cells at level 10 with the default ``patch_depth=8``), otherwise its
    tail is cut there during the calibration and the plan would be too wide
    on a larger domain; ``patch_depth + 1`` doubles that limit.  Above any of
    these a ``ValueError`` is raised.

    The widths are measured on one patch: the aligned block of
    ``4**patch_depth`` cells that contains ``reference_cell`` (a NESTED id at
    ``level``), by default the block of a cell at the centre of ``cell_ids``,
    and the equatorial block when neither is given.  The width is exact for
    the response to the centre cell of that patch (interpolation error of
    the table 4e-4 relative in the benchmark) and an approximation for
    other cells and other places; see ``docs/conv/large_conv.md``.
    :meth:`LargeConv.from_gaussian` does all of this in one call.
    """
    ellipsoid = canonicalize_ellipsoid(ellipsoid)
    level = _as_int("level", level, 0)
    sigma = _resolve_sigma(level, sigma, sigma_m)
    patch_depth = _as_int("patch_depth", patch_depth, 1)
    depth = min(patch_depth, level)
    if reference_cell is None and cell_ids is not None:
        reference_cell = _centre_cell(level, cell_ids, str(ellipsoid))
    opts = _options(
        ellipsoid, radius_deg, sigma_deg, weight_norm, up_norm, level, depth, reference_cell
    )
    top = _plan_state(level, depth, max_levels)
    edge = _edge_distance(level, depth, opts[0], opts[5])
    if sigma > edge / _EDGE_SIGMAS:
        hint = (
            f"use patch_depth={depth + 1} (4 times more cells, slower calibration)"
            if depth < level else "use a larger level"
        )
        raise ValueError(
            f"sigma={_fmt_width(level, sigma)} is too wide for the calibration patch "
            f"(patch_depth={depth}): its border is {_fmt_width(level, edge)} from the "
            f"calibration cell; wider than 1/{_EDGE_SIGMAS:g} of that "
            f"({_fmt_width(level, edge / _EDGE_SIGMAS)}) the tail is cut at the "
            f"border and the width would come out too wide on a larger domain; {hint}"
        )

    start = int(round(_COMPACT_SIGMA_START / _COMPACT_SIGMA_STEP))
    sigmas, widths = _calibration(level, 1, depth, opts)
    lowest = float(widths[start])
    if sigma < lowest:
        raise ValueError(
            f"sigma={_fmt_width(level, sigma)} is below the narrowest Gaussian this "
            f"method gives at level {level} ({_fmt_width(level, lowest)}); use "
            "gaussian_filter for narrower widths"
        )
    n = 1
    while n < top:
        nxt_sigmas, nxt_widths = _calibration(level, n + 1, depth, opts)
        if float(nxt_widths[start]) <= sigma:
            n += 1
            sigmas, widths = nxt_sigmas, nxt_widths
        else:
            break
    if sigma > float(widths[-1]):
        raise ValueError(
            f"sigma={_fmt_width(level, sigma)} is above the widest Gaussian "
            f"available with {n} levels ({_fmt_width(level, float(widths[-1]))}); "
            "raise max_levels or patch_depth, or use a larger level"
        )
    compact_sigma = float(np.interp(sigma, widths[start:], sigmas[start:]))
    radius = max(1, math.ceil(5.0 * compact_sigma))
    return GaussianPlan(
        n_levels=n,
        compact_sigma=compact_sigma,
        kernel_sz=2 * radius + 1,
        fine_kernel_sz=2 * radius * 2**n + 1,
        sigma=sigma,
        options=(level,) + tuple(opts[:5]),
    )


def _fmt_width(level: int, width: float) -> str:
    cell = _nominal_cell(level)
    return f"{width:.6g} rad (= {width / cell:.3g} cells of {cell:.3g} rad)"


def _nominal_cell(level: int) -> float:
    """Nominal cell size ``sqrt(4 pi / 12) / 2**level`` in radians."""
    return math.sqrt(4.0 * math.pi / 12.0) / 2**level



__all__ = ["LargeConv", "GaussianPlan", "down_up_blur", "gaussian_plan"]
