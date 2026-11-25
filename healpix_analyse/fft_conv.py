"""Fast large-kernel convolution on local HEALPix patches.

The input cells are projected with :class:`healpix_analyse.fft_local.LocalFFT`,
convolved on its gnomonic grid with a zero-padded 2-D FFT, then sampled back at
the original HEALPix cell centres.  All data-path operations are implemented
with PyTorch and remain differentiable.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from healpix_analyse.fft_local import LocalFFT, _next_power_of_two
from healpix_analyse._ellipsoid import canonicalize_ellipsoid


ArrayLike = Union[np.ndarray, torch.Tensor]


def _prepare_input(
    x: ArrayLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
    in_channels: int,
    n_cells: int,
) -> tuple[torch.Tensor, bool, bool]:
    """Convert input to ``[B, C, N]`` and retain restoration metadata."""
    is_numpy = isinstance(x, np.ndarray)
    if not is_numpy and not torch.is_tensor(x):
        raise TypeError("x must be a numpy.ndarray or torch.Tensor")
    is_complex = bool(np.iscomplexobj(x)) if is_numpy else torch.is_complex(x)
    if is_complex:
        raise TypeError("HealPixFFTConv currently expects real-valued input")

    tensor = torch.as_tensor(x, device=device).to(dtype=dtype)
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
    if tensor.shape[2] != n_cells:
        raise ValueError(f"Expected {n_cells} HEALPix values, got {tensor.shape[2]}")
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


class HealPixFFTConv(nn.Module):
    """Large learned convolution on a local NESTED HEALPix patch.

    Parameters
    ----------
    level : int
        Grid4Earth/HEALPix level.  Internally, ``nside = 2**level``.
    in_channels, out_channels : int
        Input and output feature channels.
    kernel_sz : int, default=33
        Positive odd side length of the learned 2-D tangent-plane kernel.
    cell_ids : array-like
        Unique NESTED cells forming one local patch.  A single gnomonic plane
        cannot represent a full sphere.
    ellipsoid : str, default="sphere"
        Geometry passed to :class:`LocalFFT`.
    max_patch_radius_deg : float, default=10
        Reject patches extending farther from their spherical centre.
    pixel_size_rad, grid_size : optional
        Projected-grid controls forwarded to :class:`LocalFFT`.
    use_norm : bool, default=False
        Apply GroupNorm followed by ReLU after back-projection.
    dtype, device : optional
        PyTorch execution options.  CUDA is selected when available.
    cache_kernel_fft : bool, default=True
        Cache the transformed kernel during no-gradient inference.  Training
        always recomputes it so gradients reach the spatial kernel.
    projection : {"average", "conservative"}, default="average"
        Convention of the HEALPix <-> grid pair, forwarded to
        :class:`LocalFFT`.  With ``"average"`` a constant field stays constant
        and the total is not kept; with ``"conservative"`` a kernel whose
        weights sum to one keeps the total (mass leaving the domain is lost)
        and a constant field is not kept.

    Notes
    -----
    The convolution is linear rather than circular: both the projected data
    and kernel are zero-padded to at least ``grid_size + kernel_sz - 1`` in
    each direction.  This prevents values wrapping across opposite sides of
    the local patch.  Values near the patch boundary still see zero outside
    the supplied domain; provide a halo and crop it when boundary accuracy is
    important.

    The learned weight follows the existing ``HealPixConv`` channel order and
    has shape ``[C_in, C_out, kernel_sz, kernel_sz]``.
    """

    def __init__(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 33,
        *,
        cell_ids,
        ellipsoid: str = "sphere",
        max_patch_radius_deg: float = 10.0,
        pixel_size_rad: Optional[float] = None,
        grid_size: Optional[int] = None,
        use_norm: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        cache_kernel_fft: bool = True,
        projection: str = "average",
    ) -> None:
        ellipsoid = canonicalize_ellipsoid(ellipsoid)
        super().__init__()

        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0")
        if int(in_channels) < 1 or int(out_channels) < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if int(kernel_sz) != kernel_sz or int(kernel_sz) < 1 or int(kernel_sz) % 2 == 0:
            raise ValueError("kernel_sz must be a positive odd integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.level = int(level)
        self.nside = 2**self.level
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_sz = int(kernel_sz)
        self.use_norm = bool(use_norm)
        self.dtype = dtype
        self.cache_kernel_fft = bool(cache_kernel_fft)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        self.transform = LocalFFT(
            cell_ids,
            self.level,
            ellipsoid=ellipsoid,
            max_patch_radius_deg=max_patch_radius_deg,
            pixel_size_rad=pixel_size_rad,
            grid_size=grid_size,
            norm="backward",
            dtype=dtype,
            device=resolved_device,
            projection=projection,
        )
        self.n_cells = self.transform.n_cells
        self.grid_size = self.transform.grid_size
        linear_size = self.grid_size + self.kernel_sz - 1
        fft_size = _next_power_of_two(linear_size)
        self.fft_shape = (fft_size, fft_size)
        self._same_start = self.kernel_sz // 2

        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels,
                self.kernel_sz,
                self.kernel_sz,
                dtype=dtype,
                device=resolved_device,
            )
        )
        fan_in = self.in_channels * self.kernel_sz * self.kernel_sz
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(
            torch.zeros(self.out_channels, dtype=dtype, device=resolved_device)
        )

        if self.use_norm:
            groups = min(8, self.out_channels)
            while self.out_channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, self.out_channels).to(
                device=resolved_device, dtype=dtype
            )
        else:
            self.norm = None

        self._cached_kernel_fft: Optional[torch.Tensor] = None
        self._cached_weight_version: Optional[int] = None

    @property
    def device(self) -> torch.device:
        """Current device of weights and projection buffers."""
        return self.weight.device

    @property
    def cell_ids(self) -> torch.Tensor:
        """NESTED cell identifiers in input/output order."""
        return self.transform.cell_ids

    def _clear_kernel_cache(self) -> None:
        self._cached_kernel_fft = None
        self._cached_weight_version = None

    def _apply(self, fn):
        result = super()._apply(fn)
        self.dtype = self.weight.dtype
        self.transform.dtype = self.weight.dtype
        self.transform.cdtype = (
            torch.complex64 if self.weight.dtype == torch.float32 else torch.complex128
        )
        self._clear_kernel_cache()
        return result

    def train(self, mode: bool = True):
        self._clear_kernel_cache()
        return super().train(mode)

    def set_kernel(self, weight, bias=None, requires_grad: bool = False):
        """Set a spatial kernel using ``[C_in,C_out,K,K]`` or flattened form."""
        tensor = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        if tensor.shape == (
            self.in_channels,
            self.out_channels,
            self.kernel_sz * self.kernel_sz,
        ):
            tensor = tensor.reshape(
                self.in_channels,
                self.out_channels,
                self.kernel_sz,
                self.kernel_sz,
            )
        expected = (
            self.in_channels,
            self.out_channels,
            self.kernel_sz,
            self.kernel_sz,
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(f"weight must have shape {expected}, got {tuple(tensor.shape)}")
        with torch.no_grad():
            self.weight.copy_(tensor)
            if bias is None:
                self.bias.zero_()
            else:
                bias_tensor = torch.as_tensor(
                    bias, dtype=self.dtype, device=self.device
                ).ravel()
                if bias_tensor.numel() != self.out_channels:
                    raise ValueError(
                        f"bias must contain {self.out_channels} values"
                    )
                self.bias.copy_(bias_tensor)
        self.weight.requires_grad_(requires_grad)
        self.bias.requires_grad_(requires_grad)
        self._clear_kernel_cache()
        return self

    def blur_floor_px(self) -> np.ndarray:
        """Per-cell blur floor of the projection pair, in grid pixels.

        See :meth:`healpix_analyse.fft_local.LocalFFT.blur_floor_px`.
        """
        return self.transform.blur_floor_px()

    def set_gaussian(
        self,
        sigma: float,
        *,
        unit: str = "pixel",
        on_below: str = "warn",
        on_truncated: str = "warn",
    ):
        """Set a normalised isotropic Gaussian kernel on every channel pair ``c -> c``.

        The kernel weights sum to one over the ``kernel_sz`` window (after
        truncation), so with ``projection="conservative"`` the total is kept.
        Only the channel pairs ``c -> c`` are filled; cross-channel weights and
        the bias are set to zero.

        Parameters
        ----------
        sigma : float
            Standard deviation per axis.
        unit : {"pixel", "rad"}
            Grid pixels, or radians in the gnomonic plane (divided by
            ``pixel_size_rad``).
        on_below : {"warn", "error", "ignore"}
            Action when ``sigma`` is below the blur floor of the projection
            pair (see :meth:`blur_floor_px`; the median over the cells is
            used).  The kernel is set in every case except ``"error"``.
        on_truncated : {"warn", "error", "ignore"}
            Action when the half-width of the window, ``kernel_sz // 2``, is
            under 3 sigma: the Gaussian is then cut and renormalised, and
            for a very large sigma it is nearly a flat box.  Use
            ``kernel_sz >= 2 * ceil(5 * sigma) + 1`` for an accurate Gaussian.

        Notes
        -----
        ``unit="rad"`` divides by ``pixel_size_rad``, the pixel size of the
        gnomonic plane, which equals the angular distance on the sphere at the
        patch centre only.
        """
        if on_truncated not in ("warn", "error", "ignore"):
            raise ValueError("on_truncated must be 'warn', 'error' or 'ignore'")
        if unit not in ("pixel", "rad"):
            raise ValueError("unit must be 'pixel' or 'rad'")
        sigma_px = float(sigma) / (
            self.transform.pixel_size_rad if unit == "rad" else 1.0
        )
        self.transform.check_width(sigma_px, on_below=on_below, stacklevel=3)
        half = self.kernel_sz // 2
        if half < 3.0 * sigma_px and on_truncated != "ignore":
            message = (
                f"kernel_sz={self.kernel_sz} (half-width {half} pixel) is under "
                f"3 sigma for sigma={sigma_px:.4g} pixel: the Gaussian is "
                "truncated and renormalised, and is not a Gaussian any more. "
                f"Use kernel_sz >= {2 * math.ceil(5 * sigma_px) + 1}."
            )
            if on_truncated == "error":
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        axis = torch.arange(-half, half + 1, dtype=torch.float64)
        if sigma_px > 0:
            profile = torch.exp(-0.5 * (axis / sigma_px) ** 2)
        else:
            profile = (axis == 0).to(torch.float64)
        kernel = profile[:, None] * profile[None, :]
        kernel = kernel / kernel.sum()
        weight = torch.zeros(
            self.in_channels,
            self.out_channels,
            self.kernel_sz,
            self.kernel_sz,
            dtype=torch.float64,
        )
        for channel in range(min(self.in_channels, self.out_channels)):
            weight[channel, channel] = kernel
        return self.set_kernel(weight)

    def _kernel_fft(self) -> torch.Tensor:
        """Return flipped-kernel spectra as ``[C_out,C_in,FH,FW//2+1]``."""
        may_cache = (
            self.cache_kernel_fft
            and not self.training
            and not torch.is_grad_enabled()
        )
        version = self.weight._version
        if (
            may_cache
            and self._cached_kernel_fft is not None
            and self._cached_weight_version == version
        ):
            return self._cached_kernel_fft

        # FFT multiplication implements mathematical convolution.  Flipping
        # converts the learned kernel to the cross-correlation convention used
        # by neural-network convolution layers.
        kernel = self.weight.permute(1, 0, 2, 3).flip((-2, -1))
        spectrum = torch.fft.rfft2(kernel, s=self.fft_shape, norm="backward")
        if may_cache:
            self._cached_kernel_fft = spectrum.detach()
            self._cached_weight_version = version
        return spectrum

    def forward(self, x: ArrayLike) -> ArrayLike:
        """Apply the FFT convolution to ``[N]``, ``[B,N]`` or ``[B,C,N]``."""
        tensor, is_numpy, was_1d = _prepare_input(
            x,
            device=self.device,
            dtype=self.dtype,
            in_channels=self.in_channels,
            n_cells=self.n_cells,
        )
        projected = self.transform._project_tensor(tensor)
        input_fft = torch.fft.rfft2(
            projected, s=self.fft_shape, norm="backward"
        )
        kernel_fft = self._kernel_fft()
        batch_size = input_fft.shape[0]
        freq_h, freq_w = input_fft.shape[-2:]
        # One channel-mixing matrix product per Fourier mode.  The explicit
        # batched matmul maps efficiently to cuBLAS/MKL and avoids a generic
        # complex einsum contraction.
        input_matrix = (
            input_fft.permute(2, 3, 0, 1)
            .contiguous()
            .view(freq_h * freq_w, batch_size, self.in_channels)
        )
        kernel_matrix = (
            kernel_fft.permute(2, 3, 1, 0)
            .contiguous()
            .view(freq_h * freq_w, self.in_channels, self.out_channels)
        )
        output_fft = (
            torch.bmm(input_matrix, kernel_matrix)
            .view(freq_h, freq_w, batch_size, self.out_channels)
            .permute(2, 3, 0, 1)
            .contiguous()
        )
        full = torch.fft.irfft2(
            output_fft, s=self.fft_shape, norm="backward"
        )
        start = self._same_start
        stop = start + self.grid_size
        same_grid = full[..., start:stop, start:stop]
        output = self.transform._unproject_tensor(same_grid)
        output = output + self.bias.view(1, -1, 1)
        if self.use_norm and self.norm is not None:
            output = F.relu(self.norm(output), inplace=False)
        return _restore_output(output, is_numpy=is_numpy, was_1d=was_1d)

    def extra_repr(self) -> str:
        return (
            f"level={self.level}, nside={self.nside}, in={self.in_channels}, "
            f"out={self.out_channels}, kernel_sz={self.kernel_sz}, "
            f"cells={self.n_cells}, grid={self.grid_size}, "
            f"fft_shape={self.fft_shape}, radius={self.transform.patch_radius_deg:.4g} deg"
        )


__all__ = ["HealPixFFTConv"]
