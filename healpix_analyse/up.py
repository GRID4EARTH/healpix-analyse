"""
up.py
=====
HEALPix resolution-increase operator for the ``healpix_analyse`` package.

Increases the HEALPix resolution by a factor of 2:
    nside_out = nside_in * 2
    N_out     = N_in * 4

The operation is defined as the *adjoint* (transpose) of the smooth
downsampling matrix from :mod:`healpix_analyse.down`, with an optional
diagonal normalisation that improves round-trip consistency.

Works on the full sphere or on a regional domain (``cell_ids``).
Resolution is exposed through the Grid4Earth ``level`` convention, with
``nside = 2**level``.

Accepts numpy arrays or torch tensors of shape ``[N]``, ``[B, N]`` or
``[..., N]`` and returns an output of the same type and leading dimensions
(flattened by a view: a contiguous ``[B*C, N]`` tensor is not copied).

The matrix and its transpose are kept in ELL form (``backend="csr"``: as
CSR buffers, the reference), built once in the constructor: they follow
``.to()`` and no sparse tensor is assembled in ``forward``.  When every fine
cell takes the value of its parent (``paired_down`` a ``mean`` Down on
contiguous NESTED data), no matrix is stored at all: the forward is an
``expand`` of the scaled coarse values, bit-identical to the reference.  See
``healpix_analyse._updown_backend``.

Dependencies: numpy, torch, healpix_geo.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse import _updown_backend as _ub
from healpix_analyse.down import (
    HealPixDown,
    _HALF_DTYPES,
    _SparseOperatorMixin,
    _storage_dtype,
    _prepare_input,
    _restore_output,
)
from healpix_analyse._ellipsoid import canonicalize_ellipsoid

ArrayLike = Union[np.ndarray, torch.Tensor]


class HealPixUp(_SparseOperatorMixin, nn.Module):
    """
    HEALPix upsampling: increase resolution by a factor of 2.

    Implements the *adjoint* of the smooth downsampling operator from
    :class:`healpix_analyse.down.HealPixDown`:

        x_up = M^T @ x

    where ``M`` is the (N_coarse × N_fine) downsampling matrix.

    Optional diagonal normalisation modes improve amplitude preservation
    for constant fields or local energy:

    - ``"adjoint"``  -- raw transpose (M^T x).
    - ``"col_l1"``   -- divide by column sums of M: preserves constant fields
                        when ``HealPixDown`` used ``weight_norm="l1"``.
    - ``"diag_l2"``  -- divide by diagonal of M^T M: least-squares diagonal
                        preconditioner (local amplitude correction).

    Parameters
    ----------
    level : int
        Coarse input Grid4Earth/HEALPix level (integer ≥ 0).  Internally,
        ``nside_in = 2**level`` and the output level is ``level + 1``.
    radius_deg : float or None
        Angular radius passed to :class:`~healpix_analyse.down.HealPixDown`
        for building the transpose matrix.  ``None`` → default.
    sigma_deg : float or None
        Gaussian sigma (degrees).  ``None`` → default.
    weight_norm : {"l1", "l2", "none"}, default "l1"
        Weight normalisation used when building the downsampling matrix.
        Must match the norm used by the paired :class:`HealPixDown`.
    up_norm : {"adjoint", "col_l1", "diag_l2"}, default "col_l1"
        Diagonal normalisation applied after the transpose:
        *adjoint* -- no normalisation.
        *col_l1*  -- preserve constant fields.
        *diag_l2* -- preserve local energy.
    eps : float, default 1e-12
        Small value added to the denominator to avoid division by zero.
    cell_ids : array-like of int or None
        Coarse pixel indices (NESTED) for a regional domain.
        If ``None``, the operator covers the full sphere.
    device : torch.device or str or None
        Device for computation.  Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for the sparse matrix values.
    paired_down : HealPixDown or None
        Reuse the exact sparse matrix and fine-cell domain of an existing
        ``smooth``, ``smooth_conservative`` or ``mean`` downsampler.  This is
        the strictest way to construct a matched Up operator for a partial
        patch.  ``cell_ids``
        must then be omitted; the ellipsoid and weight normalisation are
        inherited.

        With a ``mean`` downsampler and ``up_norm="col_l1"``, every child
        receives its parent's value.  ``"adjoint"`` gives each child
        ``parent / n`` and ``"diag_l2"`` gives ``n * parent``, where ``n``
        is the number of children of that parent present in the fine
        domain (4 on the full sphere).
    backend : {"auto", "triton", "torch", "csr"}, default "auto"
        As for :class:`~healpix_analyse.down.HealPixDown`.  *auto*: when
        every fine cell takes its parent's value (``paired_down`` a ``mean``
        Down on the full sphere, or on sorted ``cell_ids`` with complete
        parents), an ``expand`` of the coarse values with no stored index;
        otherwise the transpose matrix in ELL form, with the Triton kernel
        on CUDA and a PyTorch gather elsewhere.  *csr*: the reference (CSR
        matrix and ``torch.sparse.mm``).

    Examples
    --------
    Full sphere:

    >>> import numpy as np
    >>> from healpix_analyse.up import HealPixUp
    >>> level = 5
    >>> nside = 2**level
    >>> x = np.random.randn(12 * nside**2)      # [N_coarse]
    >>> up = HealPixUp(level=level)
    >>> y, cell_ids_out = up(x)
    >>> y.shape
    (49152,)   # 12 * 64**2

    Regional domain (the cells within 20 degrees of lon = lat = 0):

    >>> from healpix_geo import nested
    >>> cell_ids_coarse, _, _ = nested.cone_coverage((0.0, 0.0), 20.0, level)
    >>> up = HealPixUp(level=level, cell_ids=cell_ids_coarse)
    >>> y, fine_ids = up(x[cell_ids_coarse])
    """

    def __init__(
        self,
        level: int,
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        eps: float = 1e-12,
        ellipsoid: str = "WGS84",
        cell_ids: Optional[ArrayLike] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        paired_down: Optional[HealPixDown] = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.backend = _ub.check_backend(backend)

        # ---- resolve device ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.ellipsoid = (
            paired_down.ellipsoid if paired_down is not None
            else canonicalize_ellipsoid(ellipsoid)
        )
        self.weight_norm = (
            paired_down.weight_norm if paired_down is not None else weight_norm
        )

        # ---- Grid4Earth level -> internal HEALPix nside ----
        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0.")
        self.level_in = int(level)
        self.level_out = self.level_in + 1
        self.level = self.level_in
        self.nside_in = 2 ** self.level_in
        self.nside_out = self.nside_in * 2   # fine resolution
        self.N_in_full  = 12 * self.nside_in  * self.nside_in
        self.N_out_full = 12 * self.nside_out * self.nside_out

        # ---- validate up_norm ----
        self.up_norm = up_norm.lower().strip()
        if self.up_norm not in ("adjoint", "col_l1", "diag_l2"):
            raise ValueError("up_norm must be 'adjoint', 'col_l1', or 'diag_l2'.")
        self.eps = float(eps)

        # ---- regional domain / exact pairing ----
        if paired_down is not None:
            if cell_ids is not None:
                raise ValueError(
                    "cell_ids must be omitted when paired_down is given."
                )
            if paired_down.mode not in ("smooth", "smooth_conservative", "mean"):
                raise ValueError(
                    "paired_down must use mode='smooth', "
                    "'smooth_conservative' or 'mean'."
                )
            if paired_down.nside_out != self.nside_in:
                raise ValueError(
                    f"paired_down outputs nside={paired_down.nside_out}, "
                    f"but HealPixUp expects nside_in={self.nside_in}."
                )
            self.partial = paired_down.partial
            if self.partial:
                self._cell_ids_in = np.asarray(
                    paired_down.cell_ids_out, dtype=np.int64
                ).copy()
                self._cell_ids_out = np.asarray(
                    paired_down._cell_ids_in, dtype=np.int64
                ).copy()
            else:
                self._cell_ids_in = None
                self._cell_ids_out = np.arange(self.N_out_full, dtype=np.int64)
            self.N_in = paired_down.N_out
            self.N_out = paired_down.N_in

        else:
            self.partial = cell_ids is not None

        if paired_down is None and self.partial:
            cell_ids_in = np.asarray(cell_ids, dtype=np.int64).ravel()
            if cell_ids_in.size == 0:
                raise ValueError("cell_ids must not be empty.")
            if np.unique(cell_ids_in).size != cell_ids_in.size:
                raise ValueError("cell_ids must contain unique identifiers.")
            if np.any(cell_ids_in < 0) or np.any(cell_ids_in >= self.N_in_full):
                raise ValueError(
                    f"cell_ids contain identifiers outside level={self.level_in}."
                )
            # Fine output pixels: 4 NESTED children of each coarse input pixel
            cell_ids_out = np.unique(
                (cell_ids_in[:, None] * 4 + np.arange(4)[None, :]).ravel()
            ).astype(np.int64)
            self._cell_ids_in  = cell_ids_in
            self._cell_ids_out = cell_ids_out
            self.N_in  = len(cell_ids_in)
            self.N_out = len(cell_ids_out)
        elif paired_down is None:
            self._cell_ids_in  = None
            self._cell_ids_out = np.arange(self.N_out_full, dtype=np.int64)
            self.N_in  = self.N_in_full
            self.N_out = self.N_out_full

        # ---- build M^T via HealPixDown at fine resolution ----
        # The paired Down operator goes from nside_out (fine) → nside_in (coarse).
        # We build it and immediately transpose it.
        if paired_down is None:
            down = HealPixDown(
                level       = self.level_out,          # fine level
                mode        = "smooth",
                ellipsoid   = self.ellipsoid,
                radius_deg  = radius_deg,
                sigma_deg   = sigma_deg,
                weight_norm = weight_norm,
                cell_ids    = cell_ids_out if self.partial else None,
                device      = self.device,
                dtype       = self.dtype,
                backend     = "csr",       # only its matrix is used
            )
        else:
            down = paired_down

        # Reconstruct Down matrix M_down: [N_in_coarse, N_out_fine]
        vdtype = _storage_dtype(self.dtype)
        M_down = down.sparse_matrix().to(device=self.device, dtype=vdtype).coalesce()

        # Diagonal normalisation vectors (computed from M_down once)
        denom = None
        if self.up_norm in ("col_l1", "diag_l2"):
            idx  = M_down.indices()
            vals = M_down.values()
            cols = idx[1]   # fine-grid column indices

            col_sum = torch.zeros(self.N_out, device=self.device, dtype=vdtype)
            col_l2  = torch.zeros(self.N_out, device=self.device, dtype=vdtype)
            col_sum.scatter_add_(0, cols, vals)
            col_l2.scatter_add_(0, cols, vals * vals)

            self.register_buffer("_col_sum", col_sum)
            self.register_buffer("_col_l2",  col_l2)
            denom = col_sum if self.up_norm == "col_l1" else col_l2
        else:
            self._col_sum = None
            self._col_l2  = None

        # Every fine cell takes (a multiple of) its parent's value, the same
        # for the 4 children: the forward is an expand, no matrix is stored.
        self._expand = (
            self.backend != "csr" and self._expand_scales(M_down, denom)
        )
        if not self._expand:
            # Transpose → M_up: [N_out_fine, N_in_coarse]; the normalisation
            # is folded into the stored transpose used by the backward
            self._register_sparse(
                self._transpose_sparse(M_down),
                div=None if denom is None else denom.clamp_min(self.eps),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_scales(self, M_down: torch.Tensor, denom) -> bool:
        """
        Detect an Up that copies each parent to its 4 contiguous children.

        True when fine cell ``j`` reads coarse cell ``j // 4`` only, with a
        weight and a normalisation that are the same for the 4 children.
        Registers ``_fast_v`` (the weight) and ``_fast_d`` (the raw
        normalisation, or None), one value per coarse cell.
        """
        n_f, n_c = self.N_out, self.N_in
        if n_f != 4 * n_c or M_down._nnz() != n_f:
            return False
        rows, cols = M_down.indices()
        j = torch.arange(n_f, device=rows.device)
        if not (torch.equal(cols, j) and torch.equal(rows, j // 4)):
            return False
        v = M_down.values().view(n_c, 4)
        if not bool((v == v[:, :1]).all()):
            return False
        if denom is not None:
            d = denom.view(n_c, 4)
            if not bool((d == d[:, :1]).all()):
                return False
            self.register_buffer("_fast_d", d[:, 0].clone(), persistent=False)
        else:
            self._fast_d = None
        self.register_buffer("_fast_v", v[:, 0].clone(), persistent=False)
        self._M_size = (n_f, n_c)
        return True

    def _sparse_coo(self) -> torch.Tensor:
        """The stored matrix ``M^T`` (without normalisation) as COO."""
        if not self._expand:
            return super()._sparse_coo()
        n_f, n_c = self._M_size
        j = torch.arange(n_f, device=self._fast_v.device)
        return torch.sparse_coo_tensor(
            torch.stack([j, j // 4]), self._fast_v.repeat_interleave(4),
            size=(n_f, n_c), device=j.device, dtype=self._fast_v.dtype,
        ).coalesce()

    @staticmethod
    def _transpose_sparse(M: torch.Tensor) -> torch.Tensor:
        """Return the transpose of a coalesced sparse COO matrix."""
        M = M.coalesce()
        idx = M.indices()
        R, C = M.size()
        idx_T = torch.stack([idx[1], idx[0]], dim=0)
        return torch.sparse_coo_tensor(
            idx_T, M.values(), size=(C, R),
            device=M.device, dtype=M.dtype
        ).coalesce()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Output cell ids (fine-resolution NESTED indices)."""
        return self._cell_ids_out

    def forward(
        self, x: ArrayLike
    ) -> Tuple[ArrayLike, np.ndarray]:
        """
        Apply the upsampling operator.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor, shape [N], [B, N] or [..., N]
            Input map(s) at coarse resolution ``nside_in``.
            When ``cell_ids`` was given at construction, ``N`` must equal
            ``len(cell_ids)``.  Otherwise ``N = 12 * nside_in**2``.

        Returns
        -------
        y : same type as x, shape [N_out], [B, N_out] or [..., N_out]
            Upsampled map(s) at fine resolution ``nside_out = nside_in * 2``.
        cell_ids_out : np.ndarray
            NESTED pixel indices of the output pixels at ``nside_out``.
        """
        # self.device / self.dtype follow .to(); a 16-bit module computes in
        # the stored precision (float32) and casts the output.  The ELL
        # kernels read 16-bit data as it is (fp32 accumulation).
        comp = self._compute_dtype()
        keep = ()
        if self.backend != "csr" and not self._expand and self.dtype in _HALF_DTYPES:
            keep = (self.dtype,)
        t, is_numpy, lead = _prepare_input(x, self.device, comp, keep)
        B, N = t.shape

        if N != self.N_in:
            raise ValueError(
                f"Expected input with {self.N_in} pixels, got {N}."
            )

        denom = None
        if self.up_norm == "col_l1" and self._col_sum is not None:
            denom = self._col_sum
        elif self.up_norm == "diag_l2" and self._col_l2 is not None:
            denom = self._col_l2
        # up_norm == "adjoint" → no correction

        if self._expand:
            # the operations of the reference on the coarse values (product by
            # the matrix entry, division by the normalisation), then a copy
            # to the 4 children: bit-identical to the CSR product
            y = t * self._fast_v.to(device=t.device)
            if self._fast_d is not None:
                y = y / self._fast_d.to(device=t.device, dtype=t.dtype).clamp_min(self.eps)
            y = y.to(self.dtype)
            y = y.unsqueeze(-1).expand(B, N, 4).reshape(B, 4 * N)

        elif self.backend != "csr":
            div = None
            if denom is not None:
                div = denom.to(device=t.device, dtype=comp).clamp_min(self.eps)
            y = self._apply_sparse(t, div=div, out_dtype=self.dtype)

        else:
            # M^T: [N_out, N_in] CSR, built once;  t: [B, N_in]
            y = self._apply_sparse(t)         # [B, N_out]
            if denom is not None:
                d = denom.to(device=t.device, dtype=t.dtype).clamp_min(self.eps)
                y = y / d.unsqueeze(0)
            y = y.to(self.dtype)

        return _restore_output(y, is_numpy, lead), self._cell_ids_out
