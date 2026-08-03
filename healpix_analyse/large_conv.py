"""Memory-efficient large-receptive-field convolution on HEALPix maps.

``LargeConv`` replaces a large fine-resolution stencil by a matched sequence
of smooth downsamplings, a compact :class:`HealPixConv`, and the corresponding
upsamplings.  The compact kernel is learned end-to-end through the complete
multiresolution operator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from healpix_analyse.convol import HealPixConv, _DEFAULT_CACHE_DIR
from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp


ArrayLike = Union[np.ndarray, torch.Tensor]


def _kernel_plan(kernel_sz: int, max_compact_kernel_sz: int) -> tuple[int, int]:
    """Return ``(n_levels, compact_kernel_sz)`` from fine-scale radii."""
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
    max_compact_kernel_sz : int, default=7
        Maximum positive odd kernel diameter at the coarse resolution.
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
        max_compact_kernel_sz: int = 7,
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
        self.max_compact_kernel_sz = int(max_compact_kernel_sz)
        self.n_levels, self.compact_kernel_sz = _kernel_plan(
            self.kernel_sz, self.max_compact_kernel_sz
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


__all__ = ["LargeConv"]
