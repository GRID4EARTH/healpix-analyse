"""
dino.py
=======
DINOv3 (SAT-493M) embeddings of NESTED HEALPix data for the
``healpix_analyse`` package.

The idea
--------
A ViT is not a "grid" architecture: after the patch embedding it only sees a
set of tokens.  In the NESTED HEALPix scheme every cell at ``parent_level``
contains exactly ``4**(level - parent_level)`` cells of ``level``, stored
contiguously and ordered by a Morton (Z-order) code of their local ``(x, y)``
face coordinates.  A NESTED block can therefore be reshaped *exactly* into a
square image of side ``2**(level - parent_level)`` pixels, fed to an
unmodified DINOv3 network, and every 16x16 patch token maps back to one
HEALPix cell at ``level - 4``.

This module provides

- :func:`nested_to_tiles`   NESTED block  -> square tiles ``[M, C, S, S]``
- :func:`tiles_to_nested`   the exact inverse (for debugging / round trips)
- :func:`load_dinov3_sat`   load a DINOv3 SAT-493M backbone (hub or HF)
- :func:`GetDINOV3SAT`      the user-facing function

Orientation
-----------
Inside a HEALPix base face the local axes ``(x, y)`` are rotated by 45
degrees with respect to the local East/North directions: the "north" corner
of a face is at ``(x, y) = (max, max)``.  Tiles are built with
``row = S - 1 - y`` and ``col = x`` so that North points to the top-right
corner of the image.  The orientation is consistent for all tiles of the
same base face and rotated for the polar faces, which is a mild nuisance for
a satellite backbone (no gravity direction) and should be handled by
rotation augmentations when fine-tuning.

Dependencies: numpy, torch, and either the ``dinov3`` torch-hub repo
(``facebookresearch/dinov3``) or ``transformers`` (HF weights).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, torch.Tensor]

# DINOv3 SAT-493M normalisation constants (from the DINOv3 model card;
# they differ from the ImageNet/LVD-1689M values used by the web models).
SAT493M_MEAN: Tuple[float, float, float] = (0.430, 0.411, 0.296)
SAT493M_STD: Tuple[float, float, float] = (0.213, 0.156, 0.143)

# DINOv3 patch size (all released models).
DINOV3_PATCH: int = 16
DINOV3_PATCH_LEVELS: int = 4          # 16 px  == 4 HEALPix levels

_HF_SAT_IDS = {
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-sat493m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-sat493m",
}


# ---------------------------------------------------------------------------
# Morton (Z-order) helpers -- pure integer arithmetic, no healpy needed
# ---------------------------------------------------------------------------

_M1 = np.int64(0x5555555555555555)
_M2 = np.int64(0x3333333333333333)
_M4 = np.int64(0x0F0F0F0F0F0F0F0F)
_M8 = np.int64(0x00FF00FF00FF00FF)
_M16 = np.int64(0x0000FFFF0000FFFF)
_M32 = np.int64(0x00000000FFFFFFFF)


def _compact_bits(v: np.ndarray) -> np.ndarray:
    """Keep the even bits of ``v`` and pack them (inverse of :func:`_spread_bits`)."""
    v = v & _M1
    v = (v | (v >> 1)) & _M2
    v = (v | (v >> 2)) & _M4
    v = (v | (v >> 4)) & _M8
    v = (v | (v >> 8)) & _M16
    v = (v | (v >> 16)) & _M32
    return v


def _spread_bits(v: np.ndarray) -> np.ndarray:
    """Spread the 32 low bits of ``v`` onto the even bit positions."""
    v = v & _M32
    v = (v | (v << 16)) & _M16
    v = (v | (v << 8)) & _M8
    v = (v | (v << 4)) & _M4
    v = (v | (v << 2)) & _M2
    v = (v | (v << 1)) & _M1
    return v


def nested_to_xy(rel_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a *relative* NESTED index into local ``(x, y)`` coordinates.

    ``rel_id`` is the index of a cell inside its ancestor block
    (``cell_id - parent_id * 4**k``); the result satisfies
    ``0 <= x, y < 2**k``.  This is the HEALPix ``nest2xyf`` convention:
    ``x`` occupies the even bits, ``y`` the odd bits.
    """
    rel_id = np.asarray(rel_id, dtype=np.int64)
    return _compact_bits(rel_id), _compact_bits(rel_id >> 1)


def xy_to_nested(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`nested_to_xy`."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    return _spread_bits(x) | (_spread_bits(y) << 1)


# ---------------------------------------------------------------------------
# NESTED block <-> square tile
# ---------------------------------------------------------------------------

def _as_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _deduplicate(
    d: np.ndarray,
    ids: np.ndarray,
    how: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse repeated cell ids.

    Data projected from another grid (UTM, swath, ...) onto HEALPix regularly
    contains a few cells hit by more than one source pixel.  Rather than
    rejecting such an input, the duplicated rows are averaged (NaN-aware), or
    the first occurrence is kept.
    """
    uniq_ids, inv = np.unique(ids, return_inverse=True)
    n_dup = ids.shape[0] - uniq_ids.shape[0]
    if n_dup == 0:
        return d, ids
    if how == "error":
        raise ValueError(
            f"cell_id contains {n_dup} duplicate ids "
            "(pass duplicates='mean' or 'first' to aggregate them)"
        )
    warnings.warn(
        f"cell_id contains {n_dup} duplicate ids out of {ids.shape[0]}; "
        f"aggregating with duplicates={how!r}",
        RuntimeWarning,
        stacklevel=3,
    )
    if how == "first":
        _, first = np.unique(ids, return_index=True)
        return d[first], uniq_ids
    if how != "mean":
        raise ValueError("duplicates must be 'mean', 'first' or 'error'")

    finite = np.isfinite(d)
    acc = np.zeros((uniq_ids.shape[0], d.shape[1]), dtype=np.float64)
    cnt = np.zeros((uniq_ids.shape[0], d.shape[1]), dtype=np.int64)
    np.add.at(acc, inv, np.where(finite, d, 0.0))
    np.add.at(cnt, inv, finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, acc / cnt, np.nan).astype(np.float32)
    return out, uniq_ids


def nested_to_tiles(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    fill: str = "mean",
    duplicates: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder a NESTED HEALPix dataset into square tiles, one per
    ``parent_level`` cell present in ``cell_id``.

    Parameters
    ----------
    data : array [N, C] or [N]
        Pixel values at HEALPix ``level``.  NaN marks a missing pixel.
    cell_id : int array [N]
        NESTED cell ids at ``level``.  Duplicates are not allowed.
    level : int
        Resolution of ``data`` (``nside = 2**level``).
    parent_level : int
        Resolution of the tiles.  ``k = level - parent_level >= 1``; the
        tiles have side ``S = 2**k``.
    fill : {"mean", "zero", "nan"}
        Value given to the pixels of a tile that are absent from ``cell_id``
        (or NaN): the per-tile, per-band mean of the available pixels,
        zero, or NaN.
    duplicates : {"mean", "first", "error"}
        What to do when the same cell id appears several times, which happens
        when data projected from another grid puts two source pixels in the
        same HEALPix cell: average them (NaN-aware), keep the first one, or
        raise.

    Returns
    -------
    tiles : float32 array [M, C, S, S]
    parent_ids : int64 array [M]
        NESTED ids of the tiles at ``parent_level`` (sorted).
    valid : bool array [M, S, S]
        Availability mask of every pixel.
    coverage : float array [M]
        Fraction of available pixels per tile.
    """
    k = int(level) - int(parent_level)
    if k < 1:
        raise ValueError(
            f"level ({level}) must be larger than parent_level ({parent_level})"
        )
    if k > 15:
        raise ValueError("level - parent_level > 15 is not supported")
    S = 1 << k

    d = _as_numpy(data)
    squeeze = d.ndim == 1
    if squeeze:
        d = d[:, None]
    if d.ndim != 2:
        raise ValueError(f"data must have shape [N, C] or [N], got {d.shape}")
    d = d.astype(np.float32, copy=False)

    ids = _as_numpy(cell_id).astype(np.int64)
    if ids.shape != (d.shape[0],):
        raise ValueError("cell_id must have shape [N] matching data")

    d, ids = _deduplicate(d, ids, duplicates)

    parent = ids >> (2 * k)
    parent_ids, tile_idx = np.unique(parent, return_inverse=True)
    rel = ids - (parent_ids[tile_idx] << (2 * k))
    x, y = nested_to_xy(rel)
    row = (S - 1) - y
    col = x

    M, C = parent_ids.shape[0], d.shape[1]
    tiles = np.full((M, S, S, C), np.nan, dtype=np.float32)
    valid = np.zeros((M, S, S), dtype=bool)

    tiles[tile_idx, row, col] = d
    valid[tile_idx, row, col] = np.isfinite(d).all(axis=1)

    coverage = valid.reshape(M, -1).mean(axis=1)

    if fill == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m = np.nanmean(tiles.reshape(M, S * S, C), axis=1)      # [M, C]
        m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
        bad = ~np.isfinite(tiles)                                   # [M,S,S,C]
        tiles = np.where(bad, np.broadcast_to(m[:, None, None, :], tiles.shape), tiles)
    elif fill == "zero":
        tiles = np.nan_to_num(tiles, nan=0.0)
    elif fill != "nan":
        raise ValueError("fill must be 'mean', 'zero' or 'nan'")

    tiles = np.ascontiguousarray(np.transpose(tiles, (0, 3, 1, 2)))   # [M,C,S,S]
    return tiles, parent_ids, valid, coverage


def tiles_to_nested(
    tiles: np.ndarray,
    parent_ids: np.ndarray,
    parent_level: int,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of :func:`nested_to_tiles` (full tiles, no mask).

    Parameters
    ----------
    tiles : array [M, C, S, S]
    parent_ids : int array [M]
    parent_level, level : int

    Returns
    -------
    data : array [M * S * S, C]
    cell_id : int64 array [M * S * S]
        NESTED ids at ``level``, in tile-major order.
    """
    k = int(level) - int(parent_level)
    M, C, S, S2 = tiles.shape
    if S != S2 or S != (1 << k):
        raise ValueError("tile side does not match level - parent_level")
    cell_id = tile_grid_cell_ids(np.asarray(parent_ids), parent_level, level)
    data = np.transpose(tiles, (0, 2, 3, 1)).reshape(M * S * S, C)
    return data, cell_id.reshape(-1)


def tile_grid_cell_ids(
    parent_ids: np.ndarray,
    parent_level: int,
    sub_level: int,
) -> np.ndarray:
    """
    NESTED ids (at ``sub_level``) of the ``[S, S]`` image grid of every tile.

    Returns an int64 array ``[M, S, S]`` indexed as ``[tile, row, col]`` with
    the same orientation convention as :func:`nested_to_tiles`.
    """
    k = int(sub_level) - int(parent_level)
    if k < 0:
        raise ValueError("sub_level must be >= parent_level")
    S = 1 << k
    row, col = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
    rel = xy_to_nested(col, (S - 1) - row)                          # [S, S]
    parent_ids = np.asarray(parent_ids, dtype=np.int64)
    return (parent_ids[:, None, None] << (2 * k)) + rel[None]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov3_sat(
    model_name: str = "dinov3_vitl16",
    weights: Optional[str] = None,
    *,
    source: str = "auto",
    repo: str = "facebookresearch/dinov3",
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Load a DINOv3 SAT-493M backbone in evaluation mode.

    Parameters
    ----------
    model_name : {"dinov3_vitl16", "dinov3_vit7b16"}
        SAT-493M is released for ViT-L/16 and ViT-7B/16 only.
    weights : str, optional
        Path to the ``.pth`` checkpoint (torch-hub route; DINOv3 weights are
        gated and must be downloaded after accepting the licence) or a
        Hugging Face model id / local directory (HF route).
    source : {"auto", "hub", "hf"}
        "hub" uses ``torch.hub.load(repo, model_name, weights=weights)``;
        "hf" uses ``transformers.AutoModel``; "auto" tries hub then hf.
    repo : str
        Torch-hub repo or a local clone of ``facebookresearch/dinov3``.
    device : str or torch.device, optional
    """
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if weights is None and source != "hf":
        raise ValueError(
            "DINOv3 SAT-493M weights are gated and are not downloaded automatically. "
            "Without `weights`, torch-hub would fall back to the *web* (LVD-1689M) "
            "checkpoint, which is a different model and is also gated (HTTP 403).\n"
            "Request access on https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
            "(accept the DINOv3 License), download e.g. "
            "dinov3_vitl16_pretrain_sat493m-<hash>.pth, and pass its local path (or the "
            "personalised download URL) as `weights`."
        )
    errors = []
    if source in ("auto", "hub"):
        try:
            src = "local" if (repo.startswith((".", "/", "~")) or ":" in repo[:3]) else "github"
            model = torch.hub.load(repo, model_name, source=src, weights=weights)
            return model.eval().to(device)
        except Exception as e:                         # noqa: BLE001
            errors.append(f"hub: {e!r}")
            if source == "hub":
                raise
    if source in ("auto", "hf"):
        try:
            from transformers import AutoModel
            hf_id = weights if weights is not None else _HF_SAT_IDS[model_name]
            model = AutoModel.from_pretrained(hf_id)
            return model.eval().to(device)
        except Exception as e:                         # noqa: BLE001
            errors.append(f"hf: {e!r}")
    raise RuntimeError("Could not load DINOv3: " + " | ".join(errors))


def _forward(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the backbone and return ``(cls [B, D], patches [B, P, D])`` for both
    the torch-hub (``forward_features``) and the HF (``AutoModel``) APIs.
    """
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    out = model(pixel_values=x)
    h = out.last_hidden_state                                       # [B, 1+R+P, D]
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return h[:, 0], h[:, 1 + n_reg:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DINOEmbedding:
    """
    Result of :func:`GetDINOV3SAT`.

    Attributes
    ----------
    embedding : array [M, Ndino]
        One vector per tile of ``parent_level`` (CLS token by default).
    cell_id : int64 array [M]
        NESTED ids of the tiles at ``parent_level``.
    parent_level : int
    coverage : float array [M]
        Fraction of ``level`` pixels available in each tile.
    patch_embedding : array [M * P, Ndino] or None
        Patch tokens (``return_patches=True``); one per HEALPix cell at
        ``patch_level = level - 4``.
    patch_cell_id : int64 array [M * P] or None
    patch_level : int or None
    """

    embedding: np.ndarray
    cell_id: np.ndarray
    parent_level: int
    coverage: np.ndarray
    patch_embedding: Optional[np.ndarray] = None
    patch_cell_id: Optional[np.ndarray] = None
    patch_level: Optional[int] = None


def GetDINOV3SAT(
    data: ArrayLike,
    cell_id: ArrayLike,
    level: int,
    parent_level: int,
    *,
    model: Optional[nn.Module] = None,
    weights: Optional[str] = None,
    model_name: str = "dinov3_vitl16",
    bands: Sequence[int] = (0, 1, 2),
    mean: Sequence[float] = SAT493M_MEAN,
    std: Sequence[float] = SAT493M_STD,
    pooling: str = "cls",
    return_patches: bool = False,
    min_coverage: float = 0.0,
    fill: str = "mean",
    duplicates: str = "mean",
    batch_size: int = 16,
    device: Optional[Union[str, torch.device]] = None,
    autocast: bool = True,
) -> DINOEmbedding:
    """
    DINOv3 SAT-493M embeddings of NESTED HEALPix data.

    Every HEALPix cell of ``parent_level`` that has at least one pixel in
    ``cell_id`` becomes one square image of side ``2**(level - parent_level)``
    pixels (see :func:`nested_to_tiles`), which is passed *unchanged* to the
    DINOv3 backbone.

    Parameters
    ----------
    data : array [N, C]
        Reflectances at ``level``, expected in ``[0, 1]`` (Sentinel-2
        ``DN / 10000`` for instance).  NaN marks a missing pixel.
    cell_id : int array [N]
        NESTED cell ids at ``level``.
    level : int
        Resolution of ``data``.
    parent_level : int
        Resolution of the DINO tiles; ``level - parent_level >= 4`` is
        required (one DINO patch is 16 px = 4 HEALPix levels).  The natural
        choice is ``level - 8`` (256 px tiles, 16x16 patch tokens).
    model : nn.Module, optional
        A backbone already loaded with :func:`load_dinov3_sat`.  When
        ``None`` the model is loaded from ``weights`` / ``model_name``.
    weights, model_name :
        Passed to :func:`load_dinov3_sat` when ``model`` is ``None``.
    bands : sequence of 3 int
        Indices of the (R, G, B) bands in ``data`` -- Sentinel-2 B04, B03, B02.
    mean, std : sequence of 3 float
        Normalisation constants (SAT-493M defaults).
    pooling : {"cls", "mean", "cls+mean"}
        Tile vector: CLS token, mean of the patch tokens, or their
        concatenation (``2 * Ndino``).
    return_patches : bool
        Also return the patch tokens, one per cell at ``level - 4``.
    min_coverage : float
        Drop tiles whose fraction of available pixels is below this value.
    fill : {"mean", "zero"}
        How missing pixels are filled before normalisation.
    duplicates : {"mean", "first", "error"}
        How repeated cell ids are aggregated (see :func:`nested_to_tiles`).
    batch_size : int
    device : str or torch.device, optional
    autocast : bool
        Use bfloat16 autocast on CUDA.

    Returns
    -------
    DINOEmbedding
        ``result.embedding`` has shape ``[M, Ndino]`` where ``M`` is the
        number of ``parent_level`` tiles kept and ``result.cell_id`` gives
        their NESTED ids.
    """
    k = int(level) - int(parent_level)
    if k < DINOV3_PATCH_LEVELS:
        raise ValueError(
            f"level - parent_level must be >= {DINOV3_PATCH_LEVELS} "
            f"(one DINOv3 patch = {DINOV3_PATCH} px), got {k}"
        )
    if len(bands) != 3:
        raise ValueError("DINOv3 SAT expects 3 bands (R, G, B)")
    if fill == "nan":
        raise ValueError("fill='nan' cannot be fed to the network")

    d = _as_numpy(data)
    if d.ndim == 1:
        d = d[:, None]
    d = d[:, list(bands)]

    tiles, parent_ids, _valid, coverage = nested_to_tiles(
        d, cell_id, level, parent_level, fill=fill, duplicates=duplicates
    )
    keep = coverage >= float(min_coverage)
    tiles, parent_ids, coverage = tiles[keep], parent_ids[keep], coverage[keep]
    M = tiles.shape[0]

    if model is None:
        model = load_dinov3_sat(model_name, weights, device=device)
    dev = next(model.parameters()).device if device is None else torch.device(device)
    model = model.to(dev).eval()

    mean_t = torch.tensor(mean, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=dev).view(1, 3, 1, 1)

    cls_out, patch_out = [], []
    use_ac = autocast and dev.type == "cuda"
    with torch.inference_mode():
        for i in range(0, M, batch_size):
            x = torch.from_numpy(tiles[i:i + batch_size]).to(dev)
            x = (x - mean_t) / std_t
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_ac):
                cls, patches = _forward(model, x)
            cls, patches = cls.float(), patches.float()
            if pooling == "cls":
                v = cls
            elif pooling == "mean":
                v = patches.mean(dim=1)
            elif pooling == "cls+mean":
                v = torch.cat([cls, patches.mean(dim=1)], dim=1)
            else:
                raise ValueError("pooling must be 'cls', 'mean' or 'cls+mean'")
            cls_out.append(v.cpu())
            if return_patches:
                patch_out.append(patches.cpu())

    D = cls_out[0].shape[1] if cls_out else 0
    embedding = torch.cat(cls_out).numpy() if cls_out else np.zeros((0, D), np.float32)

    res = DINOEmbedding(
        embedding=embedding,
        cell_id=parent_ids,
        parent_level=int(parent_level),
        coverage=coverage,
    )
    if return_patches:
        patch_level = int(level) - DINOV3_PATCH_LEVELS
        grid = tile_grid_cell_ids(parent_ids, parent_level, patch_level)   # [M, G, G]
        if patch_out:
            p = torch.cat(patch_out).numpy()                                 # [M, G*G, D]
            if p.shape[1] != grid.shape[1] * grid.shape[2]:
                raise RuntimeError(
                    "unexpected number of patch tokens; is the model patch size 16?"
                )
            res.patch_embedding = p.reshape(-1, p.shape[-1])
        else:
            res.patch_embedding = np.zeros((0, D), np.float32)
        res.patch_cell_id = grid.reshape(-1)
        res.patch_level = patch_level
    return res


__all__ = [
    "GetDINOV3SAT",
    "DINOEmbedding",
    "load_dinov3_sat",
    "nested_to_tiles",
    "tiles_to_nested",
    "tile_grid_cell_ids",
    "nested_to_xy",
    "xy_to_nested",
    "SAT493M_MEAN",
    "SAT493M_STD",
]
