"""
minkowski.py
------------
Differentiable Minkowski functionals for 2D images [B, N, N].

The three 2D Minkowski functionals are:
  W0  –  Area            : fraction of "active" pixels
  W1  –  Perimeter       : normalised boundary length
  W2  –  Euler characteristic : topological invariant (components − holes)

All operations are differentiable; gradients flow back through `img`
and optionally through `threshold` (when it is a learned tensor).
"""

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────────────

def _as_threshold(threshold, img: torch.Tensor) -> torch.Tensor:
    """
    Cast `threshold` to a tensor that broadcasts onto img [B, N, N].

    Accepted shapes
    ---------------
    float / Tensor []    →  scalar, same threshold for the whole batch
    Tensor [B]           →  one threshold per image  → reshaped to [B, 1, 1]
    Tensor [1, N, N]     →  shared spatial threshold → direct broadcast
    Tensor [B, N, N]     →  per-image spatial threshold
    Any tensor that broadcasts onto [B, N, N]  (e.g. [B, 1, N], [1, 1, N])

    The tensor may itself carry a gradient (e.g. a learned threshold).
    """
    B = img.shape[0]
    t = (threshold if isinstance(threshold, torch.Tensor)
         else torch.tensor(threshold, dtype=img.dtype, device=img.device))
    # Convenience: [B] alone → [B, 1, 1]
    if t.ndim == 1 and t.shape[0] == B:
        t = t[:, None, None]
    # Raises a clear error if shapes are incompatible
    torch.broadcast_shapes(t.shape, img.shape)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────

def minkowski_functionals(
    img: torch.Tensor,
    threshold: "float | torch.Tensor | None" = None,
    temperature: "float | torch.Tensor" = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Compute the three 2D Minkowski functionals in a differentiable manner.

    Parameters
    ----------
    img : torch.Tensor, shape [B, N, N]
        Batch of square 2D images with values in [0, 1].
    threshold : float | torch.Tensor | None
        Optional soft threshold applied via a sigmoid before computation.
        Accepted shapes — see :func:`_as_threshold` for full details:

        - ``None``          – no thresholding; ``img`` used as soft membership
        - ``float``         – scalar, same threshold for the whole batch
        - ``Tensor []``     – scalar tensor (may carry ``requires_grad``)
        - ``Tensor [B]``    – one threshold per image in the batch
        - ``Tensor [B,N,N]``– per-pixel spatial threshold (e.g. learned)
        - Any tensor broadcastable onto ``[B, N, N]``
    temperature : float | torch.Tensor
        Sharpness of the sigmoid threshold.  Higher → closer to hard binary.

    Returns
    -------
    dict[str, Tensor]
        Keys ``'W0'``, ``'W1'``, ``'W2'``, each of shape ``[B]``.
        Fully differentiable w.r.t. both ``img`` and ``threshold``.

    Notes
    -----
    Mathematical definitions (pixel-complex formulation):

    .. math::

        W_0 = \\frac{1}{N^2} \\sum_{i,j} f_{ij}

        W_1 = \\frac{1}{N^2} \\sum_{i,j} |\\nabla f|_1

        W_2 = \\frac{Q_1 - Q_h - Q_v + Q_f}{N^2}

    where :math:`Q_1` = pixel sum, :math:`Q_h` / :math:`Q_v` = horizontal /
    vertical adjacent-pair products (soft AND), :math:`Q_f` = 2×2 block
    product (4-connectivity formula, Hadwiger 1957).
    """
    B, N, M = img.shape
    assert N == M, f"Image must be square N×N, got {N}×{M}"

    if threshold is not None:
        t = _as_threshold(threshold, img)
        img = torch.sigmoid(temperature * (img - t))

    # ── W0 · Area ─────────────────────────────────────────────────────────────
    W0 = img.mean(dim=(-2, -1))                                    # [B]

    # ── W1 · Perimeter ────────────────────────────────────────────────────────
    # First-order finite differences (L1 gradient norm ≈ boundary length)
    dh = (img[:, :,  1:] - img[:, :, :-1]).abs()                  # [B, N, N-1]
    dv = (img[:, 1:, :]  - img[:, :-1, :]).abs()                  # [B, N-1, N]
    W1 = (dh.sum(dim=(-2, -1)) + dv.sum(dim=(-2, -1))) / (N * N) # [B]

    # ── W2 · Euler characteristic ──────────────────────────────────────────────
    # Pixel-complex formula (soft AND ≈ product):
    #   χ = #vertices − #h-edges − #v-edges + #faces
    Q1 = img.sum(dim=(-2, -1))                                     # vertices

    Qh = (img[:, :,  :-1] * img[:, :,  1:]).sum(dim=(-2, -1))    # h-edges
    Qv = (img[:, :-1, :]  * img[:, 1:, :] ).sum(dim=(-2, -1))    # v-edges

    Qf = (img[:, :-1, :-1]   # top-left
        * img[:, :-1,  1:]   # top-right
        * img[:,  1:, :-1]   # bottom-left
        * img[:,  1:,  1:]   # bottom-right
         ).sum(dim=(-2, -1))                                       # 2×2 faces

    W2 = (Q1 - Qh - Qv + Qf) / (N * N)                           # [B]

    return {"W0": W0, "W1": W1, "W2": W2}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-threshold version (Minkowski curves)
# ──────────────────────────────────────────────────────────────────────────────

def minkowski_curves(
    img: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: "float | torch.Tensor" = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Compute Minkowski functionals at multiple thresholds (Minkowski curves).

    Useful for cosmological field analysis, texture description, or as
    rich topological feature vectors for machine learning.

    Parameters
    ----------
    img : torch.Tensor, shape [B, N, N]
        Batch of square 2D images.
    thresholds : torch.Tensor
        - Shape ``[T]``    – same threshold grid for every image in the batch.
        - Shape ``[B, T]`` – a different threshold grid per image
          (e.g. per-sample learned thresholds).
    temperature : float | torch.Tensor
        Sigmoid sharpness.  Scalar or broadcastable onto ``[B, T, N, N]``.

    Returns
    -------
    dict[str, Tensor]
        Keys ``'W0'``, ``'W1'``, ``'W2'``, each of shape ``[B, T]``.
        Fully differentiable w.r.t. both ``img`` and ``thresholds``.
    """
    B, N, _ = img.shape

    # Normalise thresholds → [B, T]
    t = thresholds
    if t.ndim == 1:
        T = t.shape[0]
        t = t.unsqueeze(0).expand(B, T)            # [B, T]
    else:
        assert t.ndim == 2 and t.shape[0] == B, (
            f"thresholds must be [T] or [B, T], got {thresholds.shape}")
        T = t.shape[1]

    # [B, 1, N, N] − [B, T, 1, 1]  →  [B, T, N, N]
    img_exp = img.unsqueeze(1)                     # [B, 1, N, N]
    t_exp   = t.view(B, T, 1, 1)                  # [B, T, 1, 1]
    soft    = torch.sigmoid(temperature * (img_exp - t_exp))  # [B, T, N, N]

    # Flatten batch and threshold dims, reuse minkowski_functionals
    flat = soft.view(B * T, N, N)                 # [(B·T), N, N]
    mf   = minkowski_functionals(flat)

    return {k: v.view(B, T) for k, v in mf.items()}


# ══════════════════════════════════════════════════════════════════════════════
# HEALPix version — data shape [..., Npix]
# ══════════════════════════════════════════════════════════════════════════════

def build_healpix_adjacency(
    nside: int,
    cell_ids=None,
    nest: bool = True,
    device=None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Pre-compute the edge and triangle adjacency for a HEALPix map.

    This is a **one-time setup** call.  Pass the returned tensors to
    :func:`minkowski_functionals_healpix` and :func:`minkowski_curves_healpix`.
    The computation uses healpy and is **not** differentiable, but the
    Minkowski functions that consume the output **are** differentiable.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter.
    cell_ids : array-like or None
        Global HEALPix pixel indices of the map.
        ``None`` = full sky (all ``12 * nside²`` pixels).
    nest : bool
        ``True`` for NESTED ordering (default), ``False`` for RING.
    device : torch.device or None
        Target device for the returned tensors.

    Returns
    -------
    edges : LongTensor [E, 2]
        Unique pairs of adjacent **local** pixel indices (i < j).
        "Adjacent" = sharing a boundary in the HEALPix tessellation
        (up to 8 neighbours per pixel).
    triangles : LongTensor [F, 3]
        Unique triples of mutually adjacent local indices (i < j < k).
        Used to compute the Euler characteristic.

    Notes
    -----
    The Euler characteristic formula on the pixel complex is:

    .. math::

        \\chi = Q_1 - Q_{\\text{edges}} + Q_{\\text{triangles}}

    For the full sphere with all pixels active this recovers
    :math:`\\chi(\\mathbb{S}^2) = 2` (Euler characteristic of the sphere).
    """
    import healpy as hp
    import numpy as np

    if cell_ids is None:
        cell_ids = np.arange(hp.nside2npix(nside))
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    Npix = len(cell_ids)

    # Map global → local index for fast lookup
    gid_to_lid = {int(g): l for l, g in enumerate(cell_ids)}

    edges_set: set = set()
    nbrs_local: list = []          # nbrs_local[i] = set of local neighbours of pixel i

    for lid, gid in enumerate(cell_ids):
        nbr_global = hp.get_all_neighbours(nside, int(gid), nest=nest)
        local_nbrs: set = set()
        for ng in nbr_global:
            if ng >= 0 and ng in gid_to_lid:
                nll = gid_to_lid[int(ng)]
                edges_set.add((min(lid, nll), max(lid, nll)))
                local_nbrs.add(nll)
        nbrs_local.append(local_nbrs)

    edges = torch.tensor(sorted(edges_set), dtype=torch.long)  # [E, 2]

    # Triangles: for each edge (i, j), find k ∈ N(i) ∩ N(j) with i < j < k
    triangles_set: set = set()
    for i, j in edges_set:
        for k in nbrs_local[i] & nbrs_local[j]:
            tri = tuple(sorted((i, j, k)))
            triangles_set.add(tri)

    triangles = torch.tensor(sorted(triangles_set), dtype=torch.long)  # [F, 3]

    if device is not None:
        edges     = edges.to(device)
        triangles = triangles.to(device)

    return edges, triangles


def _healpix_as_threshold(threshold, x: torch.Tensor) -> torch.Tensor:
    """
    Cast threshold to a tensor broadcastable on x [B, Npix].

    Accepted shapes (B = product of all leading dims, already flattened):
      float / Tensor[]  →  scalar
      Tensor [B]        →  one threshold per sample  → [B, 1]
      Tensor [B, Npix]  →  spatial (pixel-wise)      → direct
      Any tensor broadcastable on [B, Npix]
    """
    B = x.shape[0]
    t = (threshold if isinstance(threshold, torch.Tensor)
         else torch.tensor(threshold, dtype=x.dtype, device=x.device))
    if t.ndim == 1 and t.shape[0] == B:
        t = t[:, None]   # [B] → [B, 1]
    torch.broadcast_shapes(t.shape, x.shape)
    return t


def minkowski_functionals_healpix(
    img: torch.Tensor,
    edges: torch.Tensor,
    triangles: torch.Tensor,
    threshold: "float | torch.Tensor | None" = None,
    temperature: "float | torch.Tensor" = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Differentiable Minkowski functionals for HEALPix maps.

    Analogue of :func:`minkowski_functionals` for the spherical pixel graph
    instead of a 2D regular grid.

    Parameters
    ----------
    img : torch.Tensor, shape [..., Npix]
        Batch of HEALPix maps.  Any number of leading dimensions is accepted.
    edges : LongTensor [E, 2]
        Adjacent pixel-pair indices from :func:`build_healpix_adjacency`.
    triangles : LongTensor [F, 3]
        Adjacent pixel-triple indices from :func:`build_healpix_adjacency`.
    threshold : float | torch.Tensor | None
        Soft threshold applied via sigmoid.  Accepted shapes (with
        B = product of leading dims of ``img``):

        - ``None``         – no thresholding
        - ``float``        – scalar
        - ``Tensor []``    – scalar tensor (may carry ``requires_grad``)
        - ``Tensor [B]``   – one threshold per sample
        - ``Tensor [B, Npix]`` – spatial (pixel-wise) threshold
        - Any tensor broadcastable on ``[B, Npix]``
    temperature : float | torch.Tensor
        Sigmoid sharpness.

    Returns
    -------
    dict[str, Tensor]
        Keys ``'W0'``, ``'W1'``, ``'W2'``, each of shape ``[...]``
        (same leading dims as ``img``).  Fully differentiable w.r.t.
        ``img`` and ``threshold``.

    Notes
    -----
    Formulas on the pixel graph:

    .. math::

        W_0 = \\frac{1}{N_{\\text{pix}}} \\sum_i f_i

        W_1 = \\frac{1}{N_{\\text{pix}}} \\sum_{(i,j)\\in E} |f_i - f_j|

        W_2 = \\frac{Q_1 - Q_E + Q_F}{N_{\\text{pix}}}

    where :math:`Q_1`, :math:`Q_E`, :math:`Q_F` are the sums of
    soft-AND over vertices, edges, and triangular faces respectively.
    """
    *leading, Npix = img.shape
    B = int(torch.prod(torch.tensor(leading)).item()) if leading else 1
    x = img.reshape(B, Npix)                           # [B, Npix]

    if threshold is not None:
        t = _healpix_as_threshold(threshold, x)
        x = torch.sigmoid(temperature * (x - t))

    # ── W0 · Area ─────────────────────────────────────────────────────────────
    W0 = x.mean(dim=-1)                                # [B]

    # ── W1 · Perimeter ────────────────────────────────────────────────────────
    xi = x[:, edges[:, 0]]                             # [B, E]
    xj = x[:, edges[:, 1]]                             # [B, E]
    W1 = (xi - xj).abs().sum(dim=-1) / Npix           # [B]

    # ── W2 · Euler characteristic ─────────────────────────────────────────────
    # Pixel-complex formula on the graph: χ = V − E + F
    Q1 = x.sum(dim=-1)                                 # [B]  vertices
    Qe = (x[:, edges[:,     0]] * x[:, edges[:,     1]]).sum(dim=-1)              # [B]  edges
    if triangles.shape[0] > 0:
        Qf = (x[:, triangles[:, 0]] * x[:, triangles[:, 1]]
            * x[:, triangles[:, 2]]).sum(dim=-1)       # [B]  triangular faces
    else:
        Qf = torch.zeros(B, dtype=x.dtype, device=x.device)
    W2 = (Q1 - Qe + Qf) / Npix                        # [B]

    # ── Restore leading dims ──────────────────────────────────────────────────
    def _restore(v):
        return v.reshape(*leading) if leading else v.squeeze(0)

    return {"W0": _restore(W0), "W1": _restore(W1), "W2": _restore(W2)}


def minkowski_curves_healpix(
    img: torch.Tensor,
    edges: torch.Tensor,
    triangles: torch.Tensor,
    thresholds: torch.Tensor,
    temperature: "float | torch.Tensor" = 20.0,
) -> "dict[str, torch.Tensor]":
    """
    Minkowski curves for HEALPix maps — functionals at multiple thresholds.

    Parameters
    ----------
    img : torch.Tensor, shape [..., Npix]
    edges : LongTensor [E, 2]
    triangles : LongTensor [F, 3]
    thresholds : Tensor [T] or [B, T]
        - ``[T]``    – same grid for every sample in the batch
        - ``[B, T]`` – one grid per sample (B = product of leading dims)
    temperature : float | torch.Tensor

    Returns
    -------
    dict[str, Tensor]
        Keys ``'W0'``, ``'W1'``, ``'W2'``, each of shape ``[..., T]``.
        Differentiable w.r.t. ``img`` and ``thresholds``.
    """
    *leading, Npix = img.shape
    B = int(torch.prod(torch.tensor(leading)).item()) if leading else 1
    x = img.reshape(B, Npix)                           # [B, Npix]

    t = thresholds
    if t.ndim == 1:
        T = t.shape[0]
        t = t.unsqueeze(0).expand(B, T)               # [B, T]
    else:
        assert t.ndim == 2 and t.shape[0] == B, (
            f"thresholds must be [T] or [B, T], got {thresholds.shape}")
        T = t.shape[1]

    # [B, 1, Npix] − [B, T, 1] → [B, T, Npix]
    soft = torch.sigmoid(
        temperature * (x.unsqueeze(1) - t.view(B, T, 1))
    )                                                  # [B, T, Npix]

    # Flatten (B, T) → (B*T,) and reuse minkowski_functionals_healpix
    flat = soft.view(B * T, Npix)
    mf   = minkowski_functionals_healpix(flat, edges, triangles)

    return {k: v.view(*leading, T) if leading else v.view(B, T)
            for k, v in mf.items()}
