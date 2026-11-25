"""
down.py
=======
HEALPix resolution-reduction operator for the ``healpix_analyse`` package.

Reduces the HEALPix resolution by a factor of 2:
    nside_out = nside_in // 2
    N_out     = N_in // 4

Four modes:
    "smooth"              -- weighted Gaussian average over a disc.
    "smooth_conservative" -- "smooth" plus one global additive correction.
    "maxpool"             -- non-linear max over the 4 direct NESTED children.
    "mean"                -- unweighted mean over the 4 direct NESTED children.

HEALPix cells have equal area, so "mean" and "smooth_conservative" conserve
the global mean of a NaN-free full-sphere field; "smooth" does not.

Works on the full sphere or on a regional domain (``cell_ids``).
Resolution is exposed through the Grid4Earth ``level`` convention, with
``nside = 2**level``.

Accepts numpy arrays or torch tensors of shape ``[N]`` or ``[B, N]`` and
returns an output of the same type and number of dimensions.

The linear modes keep their sparse matrix, and its transpose for the
gradient, as CSR buffers built once in the constructor: they follow
``.to()`` and no sparse tensor is assembled in ``forward``.

Dependencies: numpy, torch, healpix_geo.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import healpix_geo
from healpix_analyse._ellipsoid import canonicalize_ellipsoid

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------

def _prepare_input(
    x: ArrayLike,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, bool, bool]:
    """
    Convert an input array / tensor into a 2-D torch.Tensor [B, N].

    Returns
    -------
    t       : torch.Tensor  [B, N]
    is_numpy: bool   -- True if the input was a numpy array
    was_1d  : bool   -- True if the input had shape [N] (no batch dim)
    """
    is_numpy = isinstance(x, np.ndarray)

    if is_numpy:
        t = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        t = x.to(device=device, dtype=dtype)

    was_1d = (t.ndim == 1)
    if was_1d:
        t = t.unsqueeze(0)  # [1, N]

    if t.ndim != 2:
        raise ValueError(
            f"Input must have shape [N] or [B, N], got {tuple(t.shape)}"
        )
    return t, is_numpy, was_1d


def _restore_output(
    t: torch.Tensor,
    is_numpy: bool,
    was_1d: bool,
) -> ArrayLike:
    """
    Convert a 2-D torch.Tensor [B, N] back to the original format.
    """
    if was_1d:
        t = t.squeeze(0)   # [N]
    if is_numpy:
        if t.dtype == torch.bfloat16:      # numpy has no bfloat16
            t = t.float()
        return t.detach().cpu().numpy()
    return t


# ---------------------------------------------------------------------------
# Sparse operator kept as CSR buffers (built once, follows .to())
# ---------------------------------------------------------------------------

_HALF_DTYPES = (torch.float16, torch.bfloat16)

# Buffers that keep at least float32 when the module is cast to a 16-bit
# dtype: the sparse matrix (CPU sparse kernels have no 16-bit version) and the
# normalisation / coverage vectors.
_KEEP_PRECISION = ("_M_val", "_MT_val", "_coverage", "_col_sum", "_col_l2")


def _storage_dtype(dtype: torch.dtype) -> torch.dtype:
    """dtype of the stored matrix for a module of dtype ``dtype``."""
    return torch.float32 if dtype in _HALF_DTYPES else dtype

class _CsrMatmul(torch.autograd.Function):
    """
    ``y = M @ x`` for a CSR matrix ``M`` and a dense ``x`` ``[N_in, B]``.

    The backward pass is ``M^T @ g`` with the CSR transpose given by the
    caller, so no transpose (and no coalesce) is computed at each call.  The
    backward is itself a ``_CsrMatmul``, hence twice differentiable.  The
    matrix is a constant: no gradient is returned for it.
    """

    @staticmethod
    def forward(ctx, x, M, MT):  # noqa: D102
        ctx.M, ctx.MT = M, MT
        return torch.sparse.mm(M, x)

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        return _CsrMatmul.apply(g, ctx.MT, ctx.M), None, None


class _SparseOperatorMixin:
    """
    Store a sparse ``[R, C]`` matrix and its transpose as CSR buffers.

    The components (``crow_indices``, ``col_indices``, ``values``) are plain
    dense buffers, so they follow ``.to()``, ``.double()``, ``.cuda()``...
    ``forward`` wraps them in CSR tensors without copying.  The buffers are
    not persistent: the matrix is entirely determined by the constructor
    arguments, so it is not written to the ``state_dict``.  The
    ``_M_indices`` / ``_M_values`` keys written by older versions are
    accepted and ignored.

    dtype rule, shared by every mode of ``HealPixDown`` and ``HealPixUp``:
    the output has the module dtype ``self.dtype``, which starts as the
    constructor ``dtype`` and follows ``.to(dtype)``, ``.double()``,
    ``.half()``...  (``self.device`` follows ``.to(device)`` likewise).  With
    a 16-bit dtype the matrix and the normalisation vectors stay in float32
    (or in their previous precision): the linear part is computed there and
    the output is cast to the 16-bit dtype.
    """

    def _apply(self, fn, *args, **kwargs):
        keep = {n: b for n, b in self._buffers.items()
                if n in _KEEP_PRECISION and b is not None}
        probe = fn(torch.zeros((), dtype=self.dtype, device=self.device))
        out = super()._apply(fn, *args, **kwargs)
        self.dtype, self.device = probe.dtype, probe.device
        for n, old in keep.items():
            new = self._buffers[n]
            if new.dtype in _HALF_DTYPES and old.dtype not in _HALF_DTYPES:
                self._buffers[n] = old.to(device=new.device)
        return out

    def _compute_dtype(self) -> torch.dtype:
        """dtype in which the sparse product is computed."""
        return self._M_val.dtype if self.dtype in _HALF_DTYPES else self.dtype

    def _register_sparse(self, M: torch.Tensor) -> None:
        M = M.coalesce().to(_storage_dtype(M.dtype))
        with warnings.catch_warnings():
            # torch warns once per process that CSR support is "beta"
            warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support")
            Mc = M.to_sparse_csr()
            MTc = M.t().coalesce().to_sparse_csr()
        # int32 indices when they fit: half the memory, and the CPU kernel
        # (MKL) would otherwise convert int64 indices at every call
        itype = torch.int32 if max(M.size() + (M._nnz(),)) < 2**31 else torch.int64
        for name, T in (("_M", Mc), ("_MT", MTc)):
            self.register_buffer(f"{name}_crow", T.crow_indices().to(itype), persistent=False)
            self.register_buffer(f"{name}_col", T.col_indices().to(itype), persistent=False)
            self.register_buffer(f"{name}_val", T.values().clone(), persistent=False)
        self._M_size = tuple(M.size())

    def _csr(self, transpose: bool = False) -> torch.Tensor:
        """The stored matrix (or its transpose) as a CSR tensor, no copy."""
        name = "_MT" if transpose else "_M"
        R, C = self._M_size
        return torch.sparse_csr_tensor(
            getattr(self, f"{name}_crow"),
            getattr(self, f"{name}_col"),
            getattr(self, f"{name}_val"),
            size=(C, R) if transpose else (R, C),
        )

    def _apply_sparse(self, t: torch.Tensor) -> torch.Tensor:
        """``M @ t`` for a batch ``t`` of shape ``[B, C]``; returns ``[B, R]``."""
        return _CsrMatmul.apply(t.T, self._csr(), self._csr(transpose=True)).T

    def _sparse_coo(self) -> torch.Tensor:
        """The stored matrix as a coalesced COO tensor."""
        return self._csr().to_sparse().coalesce()

    # Read-only views of the former COO buffers (kept for existing callers).
    @property
    def _M_indices(self) -> torch.Tensor:
        return self._sparse_coo().indices()

    @property
    def _M_values(self) -> torch.Tensor:
        return self._sparse_coo().values()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ("_M_indices", "_M_values"):
            state_dict.pop(prefix + key, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


# ---------------------------------------------------------------------------
# HealPixDown
# ---------------------------------------------------------------------------

class HealPixDown(_SparseOperatorMixin, nn.Module):
    """
    HEALPix downsampling: reduce resolution by a factor of 2.

    Reduces ``level`` → ``level - 1`` using either a
    Gaussian-weighted sparse matrix (``mode="smooth"``), the same smoothing
    with a global correction that conserves the mean
    (``mode="smooth_conservative"``), a max-pooling
    over the 4 direct NESTED children (``mode="maxpool"``), or their
    unweighted mean (``mode="mean"``).

    Can operate on the full sphere or on a regional domain.

    Parameters
    ----------
    level : int
        Input Grid4Earth/HEALPix level.  Must be an integer ≥ 1.  The
        corresponding HEALPix resolution is ``nside_in = 2**level``.
        When ``cell_ids`` is ``None``, the input is expected to have
        ``N = 12 * nside_in**2`` pixels.
    mode : {"smooth", "smooth_conservative", "maxpool", "mean"}, default "smooth"
        Downsampling strategy.
        *smooth*   -- differentiable Gaussian-weighted average (linear).
        *maxpool*  -- non-linear max over 4 NESTED children (fast).
        *mean*     -- unweighted mean over the 4 NESTED children (linear,
        differentiable).  For a NaN-free full-sphere input the global
        mean of the output equals the global mean of the input (with
        NaNs, see ``skipna``), so this is the mode
        to use for a hierarchy of maps.  *smooth* does not have this
        property: its weights are normalised per output cell only, which
        preserves constant fields but not the global mean.
        On a regional domain, a coarse cell is the mean of the children
        present in ``cell_ids``; :attr:`coverage` gives the fraction of
        children present, and the mean of the output weighted by
        :attr:`coverage` equals the mean of the input.  This holds for one
        step: chaining regional-domain operators with incomplete parents gives
        a mean of means.  To stay conservative over several steps, run the
        data and a map of ones through the chain, multiply each step's
        output by that step's :attr:`coverage`, and divide the two at the
        end: the ratio is the mean of the descendants present.
        NaN values propagate to the parent cell unless ``skipna=True``.
        *smooth_conservative* -- *smooth*, then one number is added to every
        output cell so that the mean of the output equals the mean of the
        input: ``y = M x + delta``, ``delta = mean(x) - mean(M x)``.  The
        smoothing, and the way information crosses parent boundaries, are
        those of *smooth*; only a uniform offset (the drift of one step) is
        added.  A constant field is preserved (``delta`` is then zero up to
        round-off).  The shift is the same for every cell, so a negative
        ``delta`` can make a non-negative field (a probability, say)
        slightly negative: positivity is not kept.  On a regional domain the
        means are weighted by :attr:`coverage` (the fraction of the 4
        children present), as for *mean*.  NaN: ``delta`` is built from
        finite values only, so a NaN stays local and the finite output
        cells have the same (coverage-weighted) mean as the finite input
        cells, although they are not the same cells; the mean of a NaN-free
        input is conserved over all cells.  ``weight_norm`` must be
        ``"l1"``.  :meth:`sparse_matrix` is the Gaussian part ``M`` without
        the offset.
    skipna : bool, default False
        Only for ``mode="mean"``.  If True, non-finite children are left
        out of the mean (a NaN-skipping mean); a parent with no finite
        child is NaN.  The number of children used then depends on the
        data: :meth:`valid_fraction` returns it per output cell, and the
        mean of the output weighted by ``valid_fraction(x)`` equals the
        mean of the finite input values.
    radius_deg : float or None
        Angular radius of the Gaussian kernel (degrees).
        Defaults to ~3 times the fine pixel size.
        Only used for the Gaussian modes (``"smooth"``, ``"smooth_conservative"``).
    sigma_deg : float or None
        Gaussian sigma (degrees).  Defaults to ``radius_deg / 2``.
        Only used for the Gaussian modes (``"smooth"``, ``"smooth_conservative"``).
    weight_norm : {"l1", "l2", "none"}, default "l1"
        Normalisation applied per output pixel in the Gaussian modes:
        *l1*   -- sum of weights = 1  (preserves constant fields).
        *l2*   -- sum of squares = 1  (preserves local energy).
        *none* -- raw Gaussian weights (not recommended).
    cell_ids : array-like of int or None
        Pixel indices (NESTED ordering) of the input sub-map.
        If ``None``, the operator covers the full sphere.
    device : torch.device or str or None
        Device for the sparse matrix and computations.
        Defaults to CUDA if available, else CPU.
    dtype : torch.dtype, default torch.float32
        Floating-point dtype for the sparse matrix values.

    Examples
    --------
    Full sphere:

    >>> import numpy as np
    >>> from healpix_analyse.down import HealPixDown
    >>> level = 6
    >>> nside = 2**level
    >>> x = np.random.randn(12 * nside**2)          # [N]
    >>> down = HealPixDown(level=level)
    >>> y, cell_ids_out = down(x)
    >>> y.shape
    (12288,)   # 12 * 32**2

    Regional domain (the cells within 20 degrees of lon = lat = 0):

    >>> from healpix_geo import nested
    >>> cell_ids, _, _ = nested.cone_coverage((0.0, 0.0), 20.0, level)
    >>> down = HealPixDown(level=level, cell_ids=cell_ids)
    >>> y, coarse_ids = down(x[cell_ids])
    """

    def __init__(
        self,
        level: int,
        mode: str = "smooth",
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        cell_ids: Optional[ArrayLike] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        skipna: bool = False,
    ) -> None:
        super().__init__()

        # ---- resolve device ----
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = dtype
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)

        # ---- Grid4Earth level -> internal HEALPix nside ----
        if isinstance(level, bool) or int(level) != level or int(level) < 1:
            raise ValueError("level must be an integer >= 1.")
        self.level_in = int(level)
        self.level_out = self.level_in - 1
        self.level = self.level_in
        self.nside_in = 2 ** self.level_in
        self.nside_out = self.nside_in // 2
        self.N_in_full  = 12 * self.nside_in  * self.nside_in
        self.N_out_full = 12 * self.nside_out * self.nside_out

        # ---- validate mode / weight_norm ----
        self.mode = mode.lower().strip()
        if self.mode not in ("smooth", "smooth_conservative", "maxpool", "mean"):
            raise ValueError(
                "mode must be 'smooth', 'smooth_conservative', 'maxpool' or "
                "'mean'."
            )
        self.skipna = bool(skipna)
        if self.skipna and self.mode != "mean":
            raise ValueError("skipna is only supported with mode='mean'.")
        self.weight_norm = weight_norm.lower().strip()
        if self.weight_norm not in ("l1", "l2", "none"):
            raise ValueError("weight_norm must be 'l1', 'l2', or 'none'.")
        if self.mode == "smooth_conservative" and self.weight_norm != "l1":
            raise ValueError(
                "mode='smooth_conservative' requires weight_norm='l1'."
            )

        # ---- regional domain ----
        self.partial = cell_ids is not None
        if self.partial:
            cell_ids_in = np.asarray(cell_ids, dtype=np.int64).ravel()
            if cell_ids_in.size == 0:
                raise ValueError("cell_ids must not be empty.")
            if np.unique(cell_ids_in).size != cell_ids_in.size:
                raise ValueError("cell_ids must contain unique identifiers.")
            if np.any(cell_ids_in < 0) or np.any(cell_ids_in >= self.N_in_full):
                raise ValueError(
                    f"cell_ids contain identifiers outside level={self.level_in}."
                )
            # Coarse output pixels: parent of each input pixel in NESTED ordering
            cell_ids_out = np.unique(cell_ids_in // 4).astype(np.int64)
            self._cell_ids_in  = cell_ids_in
            self._cell_ids_out = cell_ids_out
            self.N_in  = len(cell_ids_in)
            self.N_out = len(cell_ids_out)
        else:
            self._cell_ids_in  = None
            self._cell_ids_out = np.arange(self.N_out_full, dtype=np.int64)
            self.N_in  = self.N_in_full
            self.N_out = self.N_out_full

        # ---- Gaussian scale (smooth mode) ----
        if self.mode in ("smooth", "smooth_conservative"):
            pix_area = 4.0 * np.pi / self.N_in_full
            pix_deg  = np.degrees(np.sqrt(pix_area))
            self.radius_deg = float(radius_deg) if radius_deg is not None else 3.0 * pix_deg
            self.sigma_deg  = float(sigma_deg)  if sigma_deg  is not None else self.radius_deg / 2.0
            self.radius_rad = np.radians(self.radius_deg)
            self.sigma_rad  = np.radians(self.sigma_deg)

            self._register_sparse(self._build_smooth_matrix())
            if self.mode == "smooth_conservative":
                children = 4 * self._cell_ids_out[:, None] + np.arange(4)
                if self.partial:
                    present = np.isin(children, self._cell_ids_in)
                else:
                    present = np.ones(children.shape, dtype=bool)
                self.register_buffer(
                    "_coverage",
                    torch.as_tensor(present.sum(axis=1) / 4.0,
                                    dtype=torch.float64, device=self.device),
                )
        elif self.mode == "mean":
            self._build_mean_children()
        else:
            # maxpool: precompute children indices
            self._build_maxpool_children()

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
        """Haversine angular distance (radians) between (lat1,lon1) and (lat2,lon2)."""
        dlat = 0.5 * (lat2 - lat1)
        dlon = 0.5 * (lon2 - lon1)
        a = np.clip(
            np.sin(dlat)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon)**2,
            0.0, 1.0
        )
        return 2.0 * np.arcsin(np.sqrt(a))

    def _build_smooth_matrix(self) -> torch.Tensor:
        """
        Build sparse weight matrix M of shape (N_out, N_in):
            y[i_out] = Σ_j  M[i_out, j_in] * x[j_in]

        Vectorised, with one batched ``cone_coverage`` call for all output
        cells.
        """
        cell_ids_out = self._cell_ids_out   # (N_out,) int64
        cell_ids_in  = self._cell_ids_in    # (N_in,)  int64 | None
        N_out  = len(cell_ids_out)
        ilevel = int(np.log2(self.nside_in))
        olevel = int(np.log2(self.nside_out))

        # ------------------------------------------------------------------
        # Regional-domain column lookup  (vectorisable via searchsorted)
        # ------------------------------------------------------------------
        if self.partial:
            # Columns index the *input array*, i.e. the order of cell_ids_in,
            # not the sorted ids used for the lookup.
            order = np.argsort(cell_ids_in, kind="stable")
            ids_in_sorted = cell_ids_in[order]

            def _col_vec(pix_arr: np.ndarray):
                """Map pixel ids → (column_index, availability_mask) — vectorised."""
                idx  = np.searchsorted(ids_in_sorted, pix_arr)
                idx  = np.clip(idx, 0, len(ids_in_sorted) - 1)
                mask = ids_in_sorted[idx] == pix_arr
                return order[idx], mask
        else:
            def _col_vec(pix_arr: np.ndarray):
                return pix_arr.astype(np.int64), np.ones(len(pix_arr), dtype=bool)

        # ------------------------------------------------------------------
        # 1. Batch lon/lat for ALL coarse centres — ONE vectorised API call
        #    (was: N_out separate scalar calls)
        # ------------------------------------------------------------------
        lons0_deg, lats0_deg = healpix_geo.nested.healpix_to_lonlat(
            cell_ids_out.tolist(), olevel, ellipsoid=self.ellipsoid
        )
        lons0_deg = np.asarray(lons0_deg, dtype=np.float64)   # (N_out,)
        lats0_deg = np.asarray(lats0_deg, dtype=np.float64)   # (N_out,)
        lons0_rad = np.deg2rad(lons0_deg)
        lats0_rad = np.deg2rad(lats0_deg)

        # ------------------------------------------------------------------
        # 2. Batched cone_coverage — ONE native call for all coarse centres
        #    (was: N_out calls from a Python thread pool)
        # ------------------------------------------------------------------
        radius_deg = np.rad2deg(self.radius_rad)
        cones, _, _ = healpix_geo.nested.cone_coverage(
            np.column_stack((lons0_deg, lats0_deg)),
            radius_deg, ilevel, ellipsoid=self.ellipsoid,
        )
        cone_offsets = np.asarray(cones.offsets, dtype=np.int64)
        cone_cells   = np.asarray(cones.data, dtype=np.int64)

        all_cands = []
        for i_out in range(N_out):
            cand = cone_cells[cone_offsets[i_out]:cone_offsets[i_out + 1]]
            if cand.size == 0:                                   # fallback: 4 children
                p_out = int(cell_ids_out[i_out])
                cand  = (4 * p_out + np.arange(4, dtype=np.int64))
            all_cands.append(cand)
        # all_cands[i] : array of fine NESTED pixel ids for coarse pixel i

        sizes   = np.array([len(c) for c in all_cands], dtype=np.int64)   # (N_out,)
        offsets = np.concatenate([[0], sizes.cumsum()])                    # (N_out+1,)

        # ------------------------------------------------------------------
        # 3. Batch lon/lat for ALL unique fine candidates — ONE vectorised call
        #    (was: N_out batch calls with overlapping candidates)
        # ------------------------------------------------------------------
        flat_cands = np.concatenate(all_cands)                   # (total_cands,)
        unique_cands, inverse_idx = np.unique(
            flat_cands, return_inverse=True
        )                                                        # (U,), (total_cands,)

        lon_u_deg, lat_u_deg = healpix_geo.nested.healpix_to_lonlat(
            unique_cands.tolist(), ilevel, ellipsoid=self.ellipsoid
        )
        lon_u = np.deg2rad(np.asarray(lon_u_deg, dtype=np.float64))  # (U,)
        lat_u = np.deg2rad(np.asarray(lat_u_deg, dtype=np.float64))  # (U,)

        # ------------------------------------------------------------------
        # 4. Build a padded (N_out × max_K) matrix of candidates
        #    → allows fully vectorised weight computation with numpy broadcasting
        #    (was: a Python for loop with per-pixel numpy ops)
        # ------------------------------------------------------------------
        max_K = int(sizes.max()) if N_out > 0 else 0

        # Padded candidate ids and their inverse indices into unique_cands
        cands_padded = np.zeros((N_out, max_K), dtype=np.int64)    # (N_out, max_K)
        inv_padded   = np.zeros((N_out, max_K), dtype=np.int64)    # (N_out, max_K)
        valid_mask   = np.zeros((N_out, max_K), dtype=bool)        # True = real pixel

        for i_out in range(N_out):
            s = int(sizes[i_out])
            if s == 0:
                continue
            cands_padded[i_out, :s] = all_cands[i_out]
            inv_padded  [i_out, :s] = inverse_idx[offsets[i_out]:offsets[i_out + 1]]
            valid_mask  [i_out, :s] = True

        # ------------------------------------------------------------------
        # 4a. Regional-domain availability mask — vectorised over ALL rows at once
        # ------------------------------------------------------------------
        if self.partial:
            safe_cands_flat = np.where(valid_mask.ravel(), cands_padded.ravel(), 0)
            col_flat, avail_flat = _col_vec(safe_cands_flat)
            col_2d   = col_flat.reshape(N_out, max_K)
            avail_2d = avail_flat.reshape(N_out, max_K) & valid_mask
        else:
            col_2d   = cands_padded.copy()                       # col index == pixel id
            avail_2d = valid_mask.copy()

        # ------------------------------------------------------------------
        # 4b. lon/lat lookup for padded matrix — pure index operation (no API call)
        # ------------------------------------------------------------------
        safe_inv   = np.where(valid_mask, inv_padded, 0)         # 0 = safe dummy
        lon_padded = lon_u[safe_inv]                             # (N_out, max_K)
        lat_padded = lat_u[safe_inv]                             # (N_out, max_K)

        # ------------------------------------------------------------------
        # 4c. Haversine + Gaussian — fully vectorised (N_out, max_K) operations
        # ------------------------------------------------------------------
        gamma = self._haversine(
            lats0_rad[:, None], lons0_rad[:, None],              # (N_out, 1)
            lat_padded, lon_padded,                              # (N_out, max_K)
        )                                                        # (N_out, max_K)

        w = np.exp(-0.5 * (gamma / self.sigma_rad) ** 2)
        w[gamma > self.radius_rad] = 0.0
        w[~avail_2d]               = 0.0                        # unavailable / padding

        # ------------------------------------------------------------------
        # 4d. Fallback: rows where every candidate is masked out
        # ------------------------------------------------------------------
        row_sums    = w.sum(axis=1)                              # (N_out,)
        empty_rows  = row_sums <= 0.0
        n_avail     = avail_2d.sum(axis=1)                      # (N_out,)
        recoverable = empty_rows & (n_avail > 0)

        if recoverable.any():
            # Uniform weight over available slots (mirrors original behaviour)
            unif = np.where(n_avail > 0, 1.0 / np.maximum(n_avail, 1), 0.0)
            w[recoverable] = np.where(
                avail_2d[recoverable],
                unif[recoverable, None],
                0.0,
            )

        truly_empty = empty_rows & (n_avail == 0)
        if truly_empty.any():
            for i_out in np.where(truly_empty)[0]:
                warnings.warn(
                    f"Coarse pixel {cell_ids_out[i_out]} has no available input "
                    "pixels in cell_ids; it will produce zero output.",
                    UserWarning, stacklevel=3,
                )

        # ------------------------------------------------------------------
        # 4e. Normalisation — vectorised
        # ------------------------------------------------------------------
        if self.weight_norm == "l1":
            row_sums = w.sum(axis=1, keepdims=True)              # (N_out, 1)
            w = np.where(row_sums > 0, w / row_sums, w)

        elif self.weight_norm == "l2":
            row_norms = np.sqrt((w * w).sum(axis=1, keepdims=True))
            w = np.where(row_norms > 0, w / row_norms, w)

        # "none": leave weights as-is

        # ------------------------------------------------------------------
        # 5. Extract non-zero COO entries — pre-allocated arrays, no list.append
        #    (was: Python `for j, (col, wv) in enumerate(zip(col_idx, w))`)
        # ------------------------------------------------------------------
        nz_i, nz_k = np.where(w != 0.0)                         # row and slot indices
        rows_arr = nz_i.astype(np.int64)
        cols_arr = col_2d[nz_i, nz_k].astype(np.int64)
        vals_arr = w[nz_i, nz_k].astype(np.float64)

        # ------------------------------------------------------------------
        # 6. Assemble sparse tensor
        # ------------------------------------------------------------------
        rows_t = torch.as_tensor(rows_arr, dtype=torch.long,  device=self.device)
        cols_t = torch.as_tensor(cols_arr, dtype=torch.long,  device=self.device)
        vdtype = _storage_dtype(self.dtype)
        vals_t = torch.as_tensor(vals_arr, dtype=vdtype,  device=self.device)

        M = torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t]),
            vals_t,
            size=(self.N_out, self.N_in),
            device=self.device,
            dtype=vdtype,
        ).coalesce()
        return M


    def _build_maxpool_children(self) -> None:
        """
        Precompute the child index mapping for the maxpool mode.
        Registers buffer ``_children_idx`` of shape [N_out, 4].
        """
        cell_ids_out = self._cell_ids_out  # coarse pixel ids

        if self.partial:
            # Position of each child in the input array (cell_ids need not be
            # sorted).  A child absent from cell_ids is replaced by a present
            # child of the same parent, which leaves the max unchanged; every
            # parent has at least one, by construction of cell_ids_out.
            children = 4 * cell_ids_out[:, None] + np.arange(4, dtype=np.int64)
            order = np.argsort(self._cell_ids_in, kind="stable")
            ids_sorted = self._cell_ids_in[order]
            pos = np.clip(
                np.searchsorted(ids_sorted, children), 0, len(ids_sorted) - 1
            )
            valid = ids_sorted[pos] == children                 # [N_out, 4]
            col = order[pos]
            first = col[np.arange(len(col)), valid.argmax(axis=1)]
            children_idx = np.where(valid, col, first[:, None])
        else:
            children_idx = np.stack([
                4 * cell_ids_out,
                4 * cell_ids_out + 1,
                4 * cell_ids_out + 2,
                4 * cell_ids_out + 3,
            ], axis=1).astype(np.int64)

        self.register_buffer(
            "_children_idx",
            torch.as_tensor(children_idx, dtype=torch.long, device=self.device),
        )

    def _build_mean_children(self) -> None:
        """
        Precompute the child index mapping for the mean mode.

        Registers ``_children_idx`` of shape [N_out, 4] (positions in the
        input array, in the order of ``cell_ids``) and ``_children_valid``
        of shape [N_out, 4] (False where a child is absent from a
        regional ``cell_ids``).
        """
        children = 4 * self._cell_ids_out[:, None] + np.arange(4, dtype=np.int64)

        if self.partial:
            # position of each child in the input array; cell_ids need not
            # be sorted
            order = np.argsort(self._cell_ids_in, kind="stable")
            ids_sorted = self._cell_ids_in[order]
            pos = np.clip(
                np.searchsorted(ids_sorted, children), 0, len(ids_sorted) - 1
            )
            valid = ids_sorted[pos] == children
            children_idx = np.where(valid, order[pos], 0)
        else:
            valid = np.ones(children.shape, dtype=bool)
            children_idx = children

        self.register_buffer(
            "_children_idx",
            torch.as_tensor(children_idx, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_children_valid",
            torch.as_tensor(valid, dtype=torch.bool, device=self.device),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _conserve_mean(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Add ``delta = mean(x) - mean_cov(y)`` to every output cell
        (``mode="smooth_conservative"``), per batch row.

        ``mean_cov`` weights the output cells by :attr:`coverage`, so that
        on the full sphere it is the plain mean.  Only finite values enter
        the means (sums in float64), so a NaN does not spread through the
        offset.
        """
        cov = self._coverage.to(device=y.device)
        fin_x = torch.isfinite(t)
        fin_y = torch.isfinite(y)
        sum_x = torch.where(fin_x, t, torch.zeros_like(t)).double().sum(-1)
        mean_x = sum_x / fin_x.sum(-1)
        wy = cov * fin_y
        sum_y = (torch.where(fin_y, y, torch.zeros_like(y)).double() * cov).sum(-1)
        mean_y = sum_y / wy.sum(-1)
        delta = mean_x - mean_y
        delta = torch.where(torch.isfinite(delta), delta, torch.zeros_like(delta))
        return y + delta.to(y.dtype)[:, None]

    def _gather_children(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gather the 4 NESTED children of every output cell (``mode="mean"``).

        Returns ``(gathered, valid)`` with shape ``[B, N_out, 4]``.  ``valid``
        marks the children that take part in the mean: present in
        ``cell_ids`` and, with ``skipna``, finite.  It is ``None`` when every
        child takes part (full sphere, ``skipna=False``).
        """
        idx = self._children_idx.to(device=t.device)            # [N_out, 4]
        gathered = t[:, idx]                                     # [B, N_out, 4]
        valid = None
        if self.partial:
            valid = self._children_valid.to(device=t.device).expand_as(gathered)
        if self.skipna:
            finite = torch.isfinite(gathered)
            valid = finite if valid is None else valid & finite
        return gathered, valid

    def valid_fraction(self, x: ArrayLike) -> ArrayLike:
        """
        Fraction of the 4 NESTED children of each output cell that enter
        the mean for this input ``x`` (``mode="mean"`` only).

        A child counts when it is present in ``cell_ids`` and its value is
        finite.  With ``skipna=True``, the mean of ``self(x)[0]`` weighted
        by this array equals the mean of the finite values of ``x``.  To
        chain levels conservatively, carry the weights along: downsample
        ``where(isfinite(x), x, 0)`` and ``isfinite(x)`` with a plain
        ``mode="mean"`` operator and divide at the end.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor, shape [N] or [B, N]
            The input that is (or will be) passed to :meth:`forward`.

        Returns
        -------
        same type as x, shape [N_out] or [B, N_out], values in {0, .25, .5, .75, 1}
        """
        if self.mode != "mean":
            raise ValueError("valid_fraction is only defined for mode='mean'.")
        t, is_numpy, was_1d = _prepare_input(x, self.device, self.dtype)
        if t.shape[1] != self.N_in:
            raise ValueError(
                f"Expected input with {self.N_in} pixels, got {t.shape[1]}."
            )
        gathered = t[:, self._children_idx.to(device=t.device)]
        valid = torch.isfinite(gathered)
        if self.partial:
            valid = valid & self._children_valid.to(device=t.device)
        frac = valid.sum(dim=-1).to(t.dtype) / 4.0
        return _restore_output(frac, is_numpy, was_1d)

    def sparse_matrix(self) -> torch.Tensor:
        """
        The linear operator as a coalesced sparse ``[N_out, N_in]`` matrix.

        ``mode="smooth"``: the Gaussian weights.  ``mode="smooth_conservative"``:
        the same Gaussian weights, without the global offset of ``forward``
        (the offset is not sparse).  ``mode="mean"``: weight
        ``1 / n`` on each of the ``n`` children present in ``cell_ids``
        (the data-independent part; ``skipna`` is not represented).  Used
        by :class:`~healpix_analyse.up.HealPixUp` (``paired_down=...``).
        """
        if self.mode in ("smooth", "smooth_conservative"):
            return self._sparse_coo()
        if self.mode != "mean":
            raise ValueError("maxpool is not linear; it has no sparse matrix.")
        idx = self._children_idx
        valid = self._children_valid
        rows = torch.arange(self.N_out, device=idx.device)[:, None].expand_as(idx)
        vdtype = _storage_dtype(self.dtype)
        n = valid.sum(dim=-1, keepdim=True).to(vdtype).expand(idx.shape)
        return torch.sparse_coo_tensor(
            torch.stack([rows[valid], idx[valid]]),
            1.0 / n[valid],
            size=(self.N_out, self.N_in),
            device=idx.device, dtype=vdtype,
        ).coalesce()

    @property
    def coverage(self) -> np.ndarray:
        """
        Fraction of the 4 NESTED children of each output cell that are
        present in ``cell_ids`` (``mode="mean"`` and
        ``"smooth_conservative"``).

        All ones on the full sphere.  On a regional domain, the mean of the
        output weighted by this array equals the mean of the input.  It
        does not depend on the data; for the NaN-skipping mean use
        :meth:`valid_fraction`.
        """
        if self.mode == "smooth_conservative":
            return self._coverage.cpu().numpy()
        if self.mode != "mean":
            raise ValueError(
                "coverage is only defined for mode='mean' or "
                "'smooth_conservative'."
            )
        return self._children_valid.sum(dim=-1).cpu().numpy() / 4.0

    @property
    def cell_ids_out(self) -> np.ndarray:
        """Output cell ids (coarse-resolution NESTED indices)."""
        return self._cell_ids_out

    def forward(
        self, x: ArrayLike
    ) -> Tuple[ArrayLike, np.ndarray]:
        """
        Apply the downsampling operator.

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor, shape [N] or [B, N]
            Input map(s) at fine resolution ``nside_in``.
            When ``cell_ids`` was given at construction, ``N`` must equal
            ``len(cell_ids)``.  Otherwise ``N = 12 * nside_in**2``.

        Returns
        -------
        y : same type as x, shape [N_out] or [B, N_out]
            Downsampled map(s) at coarse resolution ``nside_out``.
        cell_ids_out : np.ndarray
            NESTED pixel indices of the output pixels at ``nside_out``.
        """
        # self.device / self.dtype follow .to(); the sparse product of a
        # 16-bit module is computed in the stored precision (float32)
        linear = self.mode in ("smooth", "smooth_conservative")
        dtype = self._compute_dtype() if linear else self.dtype
        t, is_numpy, was_1d = _prepare_input(x, self.device, dtype)
        B, N = t.shape

        if N != self.N_in:
            raise ValueError(
                f"Expected input with {self.N_in} pixels, got {N}."
            )

        if self.mode in ("smooth", "smooth_conservative"):
            # M: [N_out, N_in] CSR, built once;  t: [B, N_in]
            y = self._apply_sparse(t)       # [B, N_out]
            if self.mode == "smooth_conservative":
                y = self._conserve_mean(t, y)
            y = y.to(self.dtype)

        elif self.mode == "mean":
            gathered, valid = self._gather_children(t)          # [B, N_out, 4]
            if valid is None:
                y = gathered.mean(dim=-1)                       # [B, N_out]
            else:
                gathered = torch.where(
                    valid, gathered, torch.zeros((), dtype=t.dtype, device=t.device)
                )
                y = gathered.sum(dim=-1) / valid.sum(dim=-1).to(t.dtype)

        else:  # maxpool
            idx = self._children_idx.to(device=t.device)   # [N_out, 4]
            gathered = t[:, idx]                            # [B, N_out, 4]
            y = gathered.max(dim=-1).values                 # [B, N_out]

        return _restore_output(y, is_numpy, was_1d), self._cell_ids_out
