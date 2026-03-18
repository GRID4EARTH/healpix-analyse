"""
convol.py
=========
HEALPix spherical convolution operator for the ``healpix_analyse`` package.

Implements a learnable nearest-neighbor convolution on the HEALPix sphere.

For each target pixel p, the stencil collects:
  - ``kernel_sz = 1``: the pixel itself (1 point).
  - ``kernel_sz = 3``: the pixel + its 8 nearest neighbors = 9 points.

The gathered neighborhood values are mixed with a learned weight tensor
``W[C_in, C_out, K]`` where ``K = kernel_sz**2``.

For pixels at the boundary of a partial-sky patch (where some neighbors
are not in ``cell_ids``), missing neighbors are replaced by the center
pixel value (zero-padding equivalent).

Works on the full sphere or on a partial-sky subset.

Accepts numpy arrays or torch tensors of shape:
    ``[N]``        → treated as single-channel, single-sample
    ``[B, N]``     → single-channel batch
    ``[B, C, N]``  → multi-channel batch

Returns an output of the same type, always with the channel dimension
present: ``[C_out, N]`` or ``[B, C_out, N]``.

Dependencies: numpy, torch, healpy.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import healpy as hp
except ImportError as e:
    raise ImportError(
        "healpy is required by healpix_analyse.convol. "
        "Install it with:  pip install healpy"
    ) from e

from healpix_analyse.down import _restore_output

ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal I/O helper (handles [N], [B,N], [B,C,N])
# ---------------------------------------------------------------------------

def _prepare_input_conv(
    x: ArrayLike,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, bool, bool, bool]:
    """
    Normalise input to a 3-D torch.Tensor [B, C, N].

    Returns
    -------
    t        : torch.Tensor  [B, C, N]
    is_numpy : bool
    was_1d   : bool  -- input was [N]
    was_2d   : bool  -- input was [B, N] or [N] (no explicit channel dim)
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        t = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        t = x.to(device=device, dtype=dtype)

    was_1d = (t.ndim == 1)   # [N]
    was_2d = (t.ndim <= 2)   # [N] or [B, N]

    if t.ndim == 1:
        t = t.unsqueeze(0).unsqueeze(0)   # [1, 1, N]
    elif t.ndim == 2:
        t = t.unsqueeze(1)                # [B, 1, N]
    elif t.ndim != 3:
        raise ValueError(
            f"Input must have shape [N], [B, N], or [B, C, N], got {tuple(t.shape)}"
        )
    return t, is_numpy, was_1d, was_2d


def _restore_output_conv(
    t: torch.Tensor,
    is_numpy: bool,
    was_1d: bool,
    was_2d: bool,
) -> ArrayLike:
    """
    Convert [B, C_out, N] back to the appropriate output format.

    If input was [N] → output is [C_out, N]  (or [N] if C_out=1).
    If input was [B,N] → output is [B, C_out, N].
    If input was [B,C,N] → output is [B, C_out, N].
    """
    if was_1d:
        # Remove batch dim; keep channel dim unless C_out=1
        t = t.squeeze(0)       # [C_out, N]
        if t.shape[0] == 1:
            t = t.squeeze(0)   # [N]  (convenient single-channel output)
    # else: keep [B, C_out, N] or [B, 1, N]

    if is_numpy:
        return t.detach().cpu().numpy()
    return t


# ---------------------------------------------------------------------------
# Stencil building helpers
# ---------------------------------------------------------------------------

def _build_stencil_1ring(
    nside: int,
    cell_ids: Optional[np.ndarray],
    nest: bool = True,
) -> np.ndarray:
    """
    Build a neighbor-index stencil [K, 9] for a 3x3 (8-neighbor + center) kernel.

    For each of the K target pixels (in the order of ``cell_ids``), stores the
    9 column indices into the input pixel array:
        - Column 0 : center pixel itself
        - Columns 1..8 : 8 nearest neighbors from hp.get_all_neighbours

    Parameters
    ----------
    nside    : int
    cell_ids : np.ndarray [K]  or None (full sphere)
    nest     : bool

    Returns
    -------
    stencil : np.ndarray [K, 9], dtype int64
        Column indices into the sorted ``cell_ids`` array (or into the full
        sphere pixel array when ``cell_ids`` is None).
    """
    if cell_ids is None:
        # Full sphere: pixel ids are 0..N-1
        K = 12 * nside * nside
        ids = np.arange(K, dtype=np.int64)
    else:
        ids = np.asarray(cell_ids, dtype=np.int64)
        K   = len(ids)

    # Sorted ids for fast lookup
    ids_sorted = np.sort(ids)

    def _safe_col(pix_ids: np.ndarray) -> np.ndarray:
        """Map absolute pixel ids to column indices; fallback to 0 for missing."""
        idx  = np.searchsorted(ids_sorted, pix_ids)
        idx  = np.clip(idx, 0, len(ids_sorted) - 1)
        mask = ids_sorted[idx] == pix_ids
        # For missing neighbors, use column 0 as a safe fallback
        idx[~mask] = 0
        return idx

    # Center columns
    center_cols = _safe_col(ids)   # [K]

    # 8-neighbor columns
    nbrs = hp.get_all_neighbours(nside, ids.tolist(), nest=nest)
    # hp.get_all_neighbours returns shape (8, K); value -1 = no neighbor
    nbrs = np.asarray(nbrs, dtype=np.int64)   # [8, K]

    # Replace -1 (no neighbor) with center pixel id, then map to column
    for i in range(8):
        missing = nbrs[i] < 0
        nbrs[i, missing] = ids[missing]   # fallback: self

    nbr_cols = np.stack([_safe_col(nbrs[i]) for i in range(8)], axis=0)  # [8, K]

    # Stencil: [K, 9]  (center first, then 8 neighbors)
    stencil = np.stack(
        [center_cols] + [nbr_cols[i] for i in range(8)],
        axis=1
    ).astype(np.int64)

    return stencil


def _build_stencil_center_only(
    nside: int,
    cell_ids: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build a trivial [K, 1] stencil: just the center pixel (kernel_sz=1).
    """
    if cell_ids is None:
        K = 12 * nside * nside
        ids = np.arange(K, dtype=np.int64)
    else:
        ids = np.asarray(cell_ids, dtype=np.int64)
        K   = len(ids)

    # Full sphere: center pixel k maps to column k
    # Partial sky: sorted lookup
    if cell_ids is None:
        return ids.reshape(K, 1)

    ids_sorted = np.sort(ids)
    center_cols = np.searchsorted(ids_sorted, ids).reshape(K, 1).astype(np.int64)
    return center_cols


# ---------------------------------------------------------------------------
# HealPixConv
# ---------------------------------------------------------------------------

class HealPixConv(nn.Module):
    """
    Learnable HEALPix spherical convolution.

    For each target pixel, gathers a local neighborhood of ``K = kernel_sz**2``
    pixels (via ``hp.get_all_neighbours``), then applies a learned linear
    transformation with weight tensor ``W[C_in, C_out, K]``.

    An optional GroupNorm + ReLU activation can be applied after the
    convolution (``use_norm=True``).

    Parameters
    ----------
    nside : int
        HEALPix resolution.
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    kernel_sz : {1, 3}, default 3
        Convolution kernel size.
        *1* -- 1×1 convolution (no spatial mixing).
        *3* -- 3×3: center pixel + 8 nearest neighbors = 9 points.
    use_norm : bool, default False
        If True, apply GroupNorm(min(8, out_channels), out_channels) + ReLU
        after the linear mix.
    cell_ids : array-like of int or None
        Pixel indices (NESTED ordering) of the input sub-map.
        If ``None``, the operator covers the full sphere.
        If provided, ``level`` is also required.
    level : int or None
        HEALPix level such that ``nside = 2**level``.
        Required when ``cell_ids`` is not ``None``.
    nest : bool, default True
        Use NESTED pixel ordering if True, RING if False.
    device : torch.device or str or None
        Device for the learnable parameters and stencil indices.
        Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for the weight parameters.

    Notes
    -----
    - Input shapes accepted: ``[N]``, ``[B, N]``, ``[B, C_in, N]``.
    - Output shapes:
        - ``[N]`` input → ``[N]`` output  (only when ``C_in = C_out = 1``).
        - ``[B, N]`` input → ``[B, C_out, N]`` output.
        - ``[B, C, N]`` input → ``[B, C_out, N]`` output.
    - The stencil index tensor is precomputed and registered as a buffer.

    Examples
    --------
    Full sphere, single channel:

    >>> import numpy as np
    >>> from healpix_analyse.convol import HealPixConv
    >>> nside = 32
    >>> conv = HealPixConv(nside=nside, in_channels=1, out_channels=16)
    >>> x = np.random.randn(12 * nside**2)     # [N]
    >>> y = conv(x)
    >>> y.shape
    (16, 12288)                                # [C_out, N]

    Batch, multi-channel:

    >>> x = np.random.randn(8, 4, 12 * nside**2)  # [B, C_in, N]
    >>> conv = HealPixConv(nside=nside, in_channels=4, out_channels=16)
    >>> y = conv(x)
    >>> y.shape
    (8, 16, 12288)                             # [B, C_out, N]

    Partial sky:

    >>> import healpy as hp
    >>> cell_ids = hp.query_disc(nside, hp.ang2vec(np.pi/2, 0), 0.3, nest=True)
    >>> conv = HealPixConv(nside=nside, in_channels=1, out_channels=8,
    ...                    cell_ids=cell_ids, level=5)
    >>> x = np.random.randn(len(cell_ids))
    >>> y = conv(x)
    >>> y.shape
    (8, len(cell_ids))
    """

    def __init__(
        self,
        nside: int,
        in_channels: int,
        out_channels: int,
        kernel_sz: int = 3,
        use_norm: bool = False,
        cell_ids: Optional[ArrayLike] = None,
        level: Optional[int] = None,
        nest: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # ---- resolve device ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype

        # ---- validate nside ----
        self.nside = int(nside)
        if (self.nside & (self.nside - 1)) != 0 or self.nside < 1:
            raise ValueError("nside must be a positive power of 2.")

        # ---- validate kernel_sz ----
        if kernel_sz not in (1, 3):
            raise ValueError(
                f"kernel_sz must be 1 or 3, got {kernel_sz}. "
                "For larger stencils, consider using SphericalStencil directly."
            )
        self.kernel_sz = int(kernel_sz)
        self.K = self.kernel_sz * self.kernel_sz   # number of stencil points

        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)

        # ---- partial sky ----
        self.partial = cell_ids is not None
        if self.partial:
            if level is None:
                raise ValueError(
                    "level must be provided together with cell_ids "
                    "(nside = 2**level)."
                )
            expected_nside = 2 ** int(level)
            if expected_nside != self.nside:
                raise ValueError(
                    f"Inconsistent level={level} (→ nside={expected_nside}) "
                    f"and nside={self.nside}."
                )
            cell_ids_np = np.asarray(cell_ids, dtype=np.int64).ravel()
        else:
            cell_ids_np = None

        self._cell_ids = cell_ids_np  # None = full sphere
        self.N = len(cell_ids_np) if cell_ids_np is not None else 12 * self.nside ** 2

        # ---- build stencil ----
        if self.kernel_sz == 1:
            stencil = _build_stencil_center_only(self.nside, cell_ids_np)
        else:
            stencil = _build_stencil_1ring(self.nside, cell_ids_np, nest=nest)

        # stencil: [N, K] long tensor
        self.register_buffer(
            "_stencil",
            torch.as_tensor(stencil, dtype=torch.long, device=self.device),
        )

        # ---- learnable kernel ----
        # W[C_in, C_out, K]: for each (input channel, output channel, stencil point)
        self.weight = nn.Parameter(
            torch.empty(self.in_channels, self.out_channels, self.K,
                        device=self.device, dtype=self.dtype)
        )
        nn.init.kaiming_uniform_(
            self.weight.view(self.in_channels, self.out_channels * self.K),
            a=0.0, mode="fan_in", nonlinearity="relu"
        )

        # ---- optional bias ----
        self.bias = nn.Parameter(
            torch.zeros(self.out_channels, device=self.device, dtype=self.dtype)
        )

        # ---- optional GroupNorm + ReLU ----
        self.use_norm = bool(use_norm)
        if self.use_norm:
            n_groups = min(8, self.out_channels)
            while self.out_channels % n_groups != 0 and n_groups > 1:
                n_groups -= 1
            self.norm = nn.GroupNorm(n_groups, self.out_channels)
        else:
            self.norm = None

        self.to(self.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Apply the spherical convolution.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            Shape ``[N]``, ``[B, N]``, or ``[B, C_in, N]``.
            When ``cell_ids`` was provided at construction, ``N`` must equal
            ``len(cell_ids)``.  Otherwise ``N = 12 * nside**2``.

        Returns
        -------
        y : same type as x
            Shape ``[N]`` (only when C_in = C_out = 1 and input was [N]),
            ``[B, C_out, N]`` otherwise.
        """
        t, is_numpy, was_1d, was_2d = _prepare_input_conv(x, self.device, self.dtype)
        B, C_in, N = t.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C_in}."
            )
        if N != self.N:
            raise ValueError(
                f"Expected {self.N} pixels, got {N}."
            )

        # ---- gather neighborhood values ----
        # _stencil: [N, K] → t[:, :, _stencil]: [B, C_in, N, K]
        stencil = self._stencil.to(device=t.device)   # [N, K]
        gathered = t[:, :, stencil]                    # [B, C_in, N, K]

        # ---- apply kernel: y[b,o,n] = sum_{c,k} W[c,o,k] * gathered[b,c,n,k] ----
        # weight: [C_in, C_out, K]
        W = self.weight.to(device=t.device, dtype=t.dtype)
        y = torch.einsum("bcnk,cok->bon", gathered, W)   # [B, C_out, N]

        # ---- bias ----
        b_vec = self.bias.to(device=t.device, dtype=t.dtype)
        y = y + b_vec.view(1, -1, 1)

        # ---- optional norm + activation ----
        if self.use_norm and self.norm is not None:
            norm = self.norm.to(device=t.device, dtype=t.dtype)
            y = norm(y)
            y = F.relu(y, inplace=True)

        return _restore_output_conv(y, is_numpy, was_1d, was_2d)

    # ------------------------------------------------------------------
    # Extra utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"nside={self.nside}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_sz={self.kernel_sz}, K={self.K}, "
            f"partial_sky={self.partial}"
        )
