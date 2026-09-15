"""Exactly reconstructing multiscale decomposition for HEALPix maps.

``HealPixDecomp`` builds a local Laplacian pyramid from matched
:class:`HealPixDown` and :class:`HealPixUp` operators.  Every detail band is a
fine map minus the prediction obtained by downsampling and upsampling it.  The
same stored Up operator is reused during synthesis, making reconstruction
algebraically exact up to floating-point round-off, including on masked maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp
from healpix_analyse._ellipsoid import canonicalize_ellipsoid


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class HealPixPyramid(Sequence[ArrayLike]):
    """List-like multiscale coefficients returned by :meth:`HealPixDecomp.compute`.

    ``bands[:-1]`` are detail maps from fine to coarse and ``bands[-1]`` is
    the final low-pass map.  ``cell_ids[i]`` and ``levels[i]`` describe the
    pixel domain of ``bands[i]``.  The object behaves like a read-only
    sequence, so existing list-oriented code can iterate over it directly.
    """

    bands: tuple[ArrayLike, ...]
    cell_ids: tuple[np.ndarray, ...]
    levels: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.bands)

    def __getitem__(self, index):
        return self.bands[index]

    def __iter__(self) -> Iterator[ArrayLike]:
        return iter(self.bands)

    @property
    def details(self) -> tuple[ArrayLike, ...]:
        """All band-pass/detail maps, ordered from fine to coarse."""
        return self.bands[:-1]

    @property
    def coarse(self) -> ArrayLike:
        """Final low-pass scaling map."""
        return self.bands[-1]

    @property
    def images(self) -> tuple[ArrayLike, ...]:
        """Alias for :attr:`bands`."""
        return self.bands


@dataclass(frozen=True)
class HealPixWeightedPyramid:
    """Paired analysis of confidence-weighted data and confidence weights.

    Returned by :meth:`HealPixDecomp.compute_weighted`.  ``q`` is the pyramid
    obtained by running the *same* linear analysis (:class:`HealPixDown`/
    :class:`HealPixUp` chain) on ``weights * x`` (with missing values replaced
    by zero), and ``m`` is the pyramid obtained by running that identical
    analysis on ``weights`` alone.  Because the analysis operator ``W`` is
    data-independent, ``q`` and ``m`` share ``cell_ids``/``levels`` band for
    band.

    The detail bands of ``m`` are *signed correction terms*, not per-band
    confidences: they are not bounded in ``[0, 1]`` and must not be clipped,
    thresholded, or divided into individually.  Only a full synthesis of both
    channels followed by a single division -- performed by
    :meth:`HealPixDecomp.invert` when given a ``HealPixWeightedPyramid``, or
    by a kernel-pyramid convolution applied identically to both channels --
    recovers a valid normalized result.
    """

    q: HealPixPyramid
    m: HealPixPyramid

    @property
    def cell_ids(self) -> tuple[np.ndarray, ...]:
        return self.q.cell_ids

    @property
    def levels(self) -> tuple[int, ...]:
        return self.q.levels


def _as_numpy_ids(cell_ids, *, level: int) -> np.ndarray:
    if torch.is_tensor(cell_ids):
        ids = cell_ids.detach().cpu().numpy()
    else:
        ids = np.asarray(cell_ids)
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise ValueError("cell_ids must not be empty")
    if np.unique(ids).size != ids.size:
        raise ValueError("cell_ids must contain unique identifiers")
    npix = 12 * 4**level
    if np.any(ids < 0) or np.any(ids >= npix):
        raise ValueError(
            f"cell_ids contain identifiers outside level={level} "
            f"(expected values in [0, {npix - 1}])"
        )
    return ids


class HealPixDecomp(nn.Module):
    """Local, exactly reconstructing HEALPix multiscale decomposition.

    Parameters
    ----------
    level : int
        Fine input Grid4Earth/HEALPix level, with ``nside = 2**level``.
    cell_ids : array-like, optional
        NESTED identifiers of a masked/partial map, in the same order as the
        input data.  ``None`` selects the complete sphere.
    Jmax : int, default=-1
        Number of Down operations.  ``-1`` uses every available scale down to
        level zero.  ``0`` returns a pyramid containing only the input map.
    ellipsoid : str, default="WGS84"
        Geometry used by every analysis filter.
    radius_deg, sigma_deg : float, optional
        Gaussian smoothing parameters passed to every ``HealPixDown``.  When
        omitted, each level uses its own resolution-dependent defaults.
    weight_norm : {"l1", "l2", "none"}, default="l1"
        Down-filter normalisation.  ``l1`` preserves constant maps locally.
    up_norm : {"adjoint", "col_l1", "diag_l2"}, default="col_l1"
        Matched synthesis normalisation.  Exact pyramid reconstruction does
        not depend on this choice because the prediction is stored as a
        residual, but it changes the interpretation of individual bands.
    dtype, device : optional
        PyTorch execution options.  CUDA is selected when available.

    Notes
    -----
    At scale ``j`` the analysis is

    ``coarse[j+1] = Down[j](coarse[j])``

    ``detail[j] = coarse[j] - Up[j](coarse[j+1])``.

    Synthesis applies the exact reverse recursion

    ``coarse[j] = detail[j] + Up[j](coarse[j+1])``.

    Every ``HealPixUp`` is constructed with ``paired_down=...`` and therefore
    reuses the exact sparse matrix and fine-cell domain of its analysis step.
    """

    def __init__(
        self,
        level: int,
        cell_ids=None,
        Jmax: int = -1,
        *,
        ellipsoid: str = "WGS84",
        radius_deg: Optional[float] = None,
        sigma_deg: Optional[float] = None,
        weight_norm: str = "l1",
        up_norm: str = "col_l1",
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if isinstance(level, bool) or int(level) != level or int(level) < 0:
            raise ValueError("level must be an integer >= 0")
        if isinstance(Jmax, bool) or int(Jmax) != Jmax or int(Jmax) < -1:
            raise ValueError("Jmax must be -1 or a non-negative integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        self.level = int(level)
        requested_jmax = int(Jmax)
        if requested_jmax == -1:
            n_scales = self.level
        elif requested_jmax > self.level:
            raise ValueError(
                f"Jmax={requested_jmax} requests more Down operations than "
                f"available from level={self.level}"
            )
        else:
            n_scales = requested_jmax
        self.Jmax = requested_jmax
        self.n_scales = n_scales
        self.n_bands = n_scales + 1
        self.ellipsoid = canonicalize_ellipsoid(ellipsoid)
        self.weight_norm = str(weight_norm)
        self.up_norm = str(up_norm)
        self.dtype = dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_device = torch.device(device)

        self.partial = cell_ids is not None
        if self.partial:
            original_ids = _as_numpy_ids(cell_ids, level=self.level)
            sort_order = np.argsort(original_ids)
            inverse_order = np.argsort(sort_order)
            fine_ids = original_ids[sort_order]
        else:
            fine_ids = np.arange(12 * 4**self.level, dtype=np.int64)
            sort_order = np.arange(fine_ids.size, dtype=np.int64)
            inverse_order = sort_order.copy()

        self.n_pixels = int(fine_ids.size)
        self.register_buffer(
            "_input_sort",
            torch.as_tensor(sort_order, dtype=torch.long, device=resolved_device),
        )
        self.register_buffer(
            "_input_unsort",
            torch.as_tensor(inverse_order, dtype=torch.long, device=resolved_device),
        )
        self.register_buffer(
            "_dtype_anchor",
            torch.empty(0, dtype=dtype, device=resolved_device),
        )

        down_layers = []
        up_layers = []
        ids_per_scale = [fine_ids.copy()]
        current_ids = fine_ids
        current_level = self.level
        for _ in range(self.n_scales):
            down = HealPixDown(
                level=current_level,
                mode="smooth",
                ellipsoid=self.ellipsoid,
                radius_deg=radius_deg,
                sigma_deg=sigma_deg,
                weight_norm=self.weight_norm,
                cell_ids=current_ids if self.partial else None,
                device=resolved_device,
                dtype=dtype,
            )
            up = HealPixUp(
                level=current_level - 1,
                up_norm=self.up_norm,
                device=resolved_device,
                dtype=dtype,
                paired_down=down,
            )
            down_layers.append(down)
            up_layers.append(up)
            current_ids = np.asarray(down.cell_ids_out, dtype=np.int64).copy()
            ids_per_scale.append(current_ids)
            current_level -= 1

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)
        self.levels = tuple(self.level - j for j in range(self.n_bands))
        self.cell_ids_per_scale = tuple(ids.copy() for ids in ids_per_scale)
        self.sizes = tuple(len(ids) for ids in self.cell_ids_per_scale)

    @property
    def device(self) -> torch.device:
        """Current device of the decomposition operators."""
        return self._input_sort.device

    @property
    def cell_ids(self) -> tuple[np.ndarray, ...]:
        """Canonical NESTED identifiers for every returned band."""
        return tuple(ids.copy() for ids in self.cell_ids_per_scale)

    def _apply(self, fn):
        result = super()._apply(fn)
        self.dtype = self._dtype_anchor.dtype
        for module in [*self.down_layers, *self.up_layers]:
            module.device = self.device
            module.dtype = self.dtype
        return result

    def _prepare_map(
        self,
        data: ArrayLike,
        *,
        expected_size: int,
        name: str,
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy and not torch.is_tensor(data):
            raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")
        is_complex = bool(np.iscomplexobj(data)) if is_numpy else torch.is_complex(data)
        if is_complex:
            raise TypeError("HealPixDecomp currently expects real-valued maps")
        tensor = torch.as_tensor(data, device=self.device).to(dtype=self.dtype)
        if tensor.ndim < 1 or tensor.shape[-1] != expected_size:
            raise ValueError(
                f"{name} must end with {expected_size} HEALPix values, "
                f"got shape {tuple(tensor.shape)}"
            )
        leading_shape = tuple(tensor.shape[:-1])
        return tensor.reshape(-1, expected_size), leading_shape, is_numpy

    @staticmethod
    def _restore_band(
        tensor: torch.Tensor,
        *,
        leading_shape: tuple[int, ...],
        is_numpy: bool,
    ) -> ArrayLike:
        restored = tensor.reshape(*leading_shape, tensor.shape[-1])
        if is_numpy:
            return restored.detach().cpu().numpy()
        return restored

    def _analyze(self, current: torch.Tensor) -> list[torch.Tensor]:
        """Run the fine-to-coarse analysis recursion on a prepared tensor.

        ``current`` must already be shaped ``[..., n_pixels]`` in the input's
        *external* NESTED order.  Returns bands in each scale's own internal
        (sorted) cell-id order, fine to coarse, matching
        :attr:`cell_ids_per_scale`. This is the shared core used by both
        :meth:`compute` and :meth:`compute_weighted`, so the identical linear
        operator ``W`` is applied whatever channel (data or weights) is
        passed in.
        """
        current = current.index_select(-1, self._input_sort)
        bands: list[torch.Tensor] = []

        for j, (down, up) in enumerate(zip(self.down_layers, self.up_layers)):
            coarse, coarse_ids = down(current)
            prediction, fine_ids = up(coarse)
            if not np.array_equal(coarse_ids, self.cell_ids_per_scale[j + 1]):
                raise RuntimeError(f"Down returned inconsistent cell_ids at scale {j}")
            if not np.array_equal(fine_ids, self.cell_ids_per_scale[j]):
                raise RuntimeError(f"Up returned inconsistent cell_ids at scale {j}")
            bands.append(current - prediction)
            current = coarse
        bands.append(current)
        return bands

    def _synthesize(self, prepared: Sequence[torch.Tensor]) -> torch.Tensor:
        """Run the coarse-to-fine synthesis recursion.

        ``prepared`` must be ``n_bands`` tensors in each scale's internal
        (sorted) cell-id order (as returned by :meth:`_analyze`). Returns a
        tensor still in that finest scale's internal order -- callers apply
        ``index_select(-1, self._input_unsort)`` to restore external order.
        """
        current = prepared[-1]
        for j in range(self.n_scales - 1, -1, -1):
            prediction, fine_ids = self.up_layers[j](current)
            if not np.array_equal(fine_ids, self.cell_ids_per_scale[j]):
                raise RuntimeError(f"Up returned inconsistent cell_ids at scale {j}")
            current = prepared[j] + prediction
        return current

    def compute(self, data: ArrayLike) -> HealPixPyramid:
        """Decompose a map into fine-to-coarse details and one coarse map.

        The returned object is list-like and contains ``n_scales + 1`` images.
        Arbitrary leading dimensions are supported, for example ``[B,C,N]``.
        """
        current, leading_shape, is_numpy = self._prepare_map(
            data, expected_size=self.n_pixels, name="data"
        )
        bands = self._analyze(current)

        restored_bands = tuple(
            self._restore_band(
                band, leading_shape=leading_shape, is_numpy=is_numpy
            )
            for band in bands
        )
        return HealPixPyramid(
            bands=restored_bands,
            cell_ids=self.cell_ids,
            levels=self.levels,
        )

    def compute_weighted(
        self, data: ArrayLike, weights: Optional[ArrayLike] = None
    ) -> HealPixWeightedPyramid:
        """Decompose a map that may contain NaN/missing values.

        Non-finite entries of ``data`` (and, when given, non-finite or
        implicitly-absent entries of ``weights``) are treated as missing.
        ``weights`` defaults to a 0/1 confidence mask built from
        ``isfinite(data)`` when omitted, i.e. plain unweighted missing-data
        handling.

        Internally this runs the *same* analysis operator ``W`` (the stored
        Down/Up chain) on ``q = weights * data`` (missing values zeroed) and
        on ``m = weights`` (also zeroed at missing positions), and returns
        both resulting pyramids paired in a :class:`HealPixWeightedPyramid`.
        Reconstruct with :meth:`invert`, which performs the single division
        ``S(q) / S(m)`` after synthesis, per the normalized-convolution
        convention -- never divide the two pyramids band by band.

        The boolean missing/valid decision itself carries no gradient (as
        with any masking based on ``isfinite``); the surviving finite values
        of ``data``/``weights`` remain differentiable.
        """
        tensor, leading_shape, is_numpy = self._prepare_map(
            data, expected_size=self.n_pixels, name="data"
        )
        finite_data = torch.isfinite(tensor)
        if weights is None:
            valid = finite_data
            m = finite_data.to(dtype=self.dtype)
        else:
            wtensor, wshape, w_is_numpy = self._prepare_map(
                weights, expected_size=self.n_pixels, name="weights"
            )
            if wshape != leading_shape:
                raise ValueError(
                    "weights must have the same leading shape as data; got "
                    f"{wshape} vs {leading_shape}"
                )
            valid = finite_data & torch.isfinite(wtensor)
            m = torch.where(valid, wtensor, torch.zeros_like(wtensor))
            is_numpy = is_numpy or w_is_numpy

        x_safe = torch.where(valid, tensor, torch.zeros_like(tensor))
        q = x_safe * m

        q_bands = self._analyze(q)
        m_bands = self._analyze(m)

        def _restore_all(bands: list[torch.Tensor]) -> tuple[ArrayLike, ...]:
            return tuple(
                self._restore_band(band, leading_shape=leading_shape, is_numpy=is_numpy)
                for band in bands
            )

        q_pyramid = HealPixPyramid(
            bands=_restore_all(q_bands), cell_ids=self.cell_ids, levels=self.levels
        )
        m_pyramid = HealPixPyramid(
            bands=_restore_all(m_bands), cell_ids=self.cell_ids, levels=self.levels
        )
        return HealPixWeightedPyramid(q=q_pyramid, m=m_pyramid)

    def forward(self, data: ArrayLike) -> HealPixPyramid:
        """Alias for :meth:`compute`, enabling normal ``nn.Module`` usage."""
        return self.compute(data)

    def _validate_pyramid_metadata(self, pyramid: HealPixPyramid) -> None:
        if tuple(pyramid.levels) != self.levels:
            raise ValueError(
                f"Pyramid levels {tuple(pyramid.levels)} do not match {self.levels}"
            )
        if len(pyramid.cell_ids) != self.n_bands:
            raise ValueError("Pyramid contains an invalid number of cell-id arrays")
        for j, (actual, expected) in enumerate(
            zip(pyramid.cell_ids, self.cell_ids_per_scale)
        ):
            if not np.array_equal(np.asarray(actual), expected):
                raise ValueError(f"Pyramid cell_ids do not match at band {j}")

    def _prepare_decomposition(
        self,
        decomposition: Union[HealPixPyramid, Sequence[ArrayLike]],
    ) -> tuple[list[torch.Tensor], tuple[int, ...], bool]:
        if isinstance(decomposition, HealPixPyramid):
            self._validate_pyramid_metadata(decomposition)
            bands = decomposition.bands
        else:
            bands = tuple(decomposition)
        if len(bands) != self.n_bands:
            raise ValueError(
                f"Expected {self.n_bands} bands, got {len(bands)}"
            )

        prepared: list[torch.Tensor] = []
        leading_shape: Optional[tuple[int, ...]] = None
        all_numpy = True
        for j, (band, size) in enumerate(zip(bands, self.sizes)):
            tensor, shape, is_numpy = self._prepare_map(
                band, expected_size=size, name=f"band[{j}]"
            )
            if leading_shape is None:
                leading_shape = shape
            elif shape != leading_shape:
                raise ValueError(
                    f"All bands must have the same leading shape; band[0] has "
                    f"{leading_shape}, band[{j}] has {shape}"
                )
            all_numpy = all_numpy and is_numpy
            prepared.append(tensor)

        assert leading_shape is not None
        return prepared, leading_shape, all_numpy

    def invert(
        self,
        decomposition: Union[HealPixPyramid, HealPixWeightedPyramid, Sequence[ArrayLike]],
        *,
        restore_mask: bool = True,
        eps: float = 1e-8,
    ) -> ArrayLike:
        """Reconstruct the original map from multiscale coefficients.

        Given a plain :class:`HealPixPyramid` (or a bare sequence of bands),
        this is the exact algebraic reverse of :meth:`compute`, unaffected by
        ``restore_mask``/``eps``.

        Given a :class:`HealPixWeightedPyramid` (from :meth:`compute_weighted`),
        both channels are synthesized with the *same* Up chain and combined
        with a single division ``y = S(q) / S(m)`` -- never per-band. Output
        pixels whose synthesized weight ``S(m)`` falls at or below ``eps``
        have no reliable support; when ``restore_mask`` is True (default)
        they are set to NaN, otherwise to 0.
        """
        if isinstance(decomposition, HealPixWeightedPyramid):
            return self._invert_weighted(decomposition, restore_mask=restore_mask, eps=eps)

        prepared, leading_shape, all_numpy = self._prepare_decomposition(
            decomposition
        )
        current = self._synthesize(prepared)
        current = current.index_select(-1, self._input_unsort)
        return self._restore_band(
            current, leading_shape=leading_shape, is_numpy=all_numpy
        )

    def _invert_weighted(
        self,
        weighted: HealPixWeightedPyramid,
        *,
        restore_mask: bool,
        eps: float,
    ) -> ArrayLike:
        q_prepared, leading_shape, q_is_numpy = self._prepare_decomposition(weighted.q)
        m_prepared, m_leading_shape, m_is_numpy = self._prepare_decomposition(weighted.m)
        if m_leading_shape != leading_shape:
            raise ValueError(
                "q and m pyramids of a HealPixWeightedPyramid must share the "
                f"same leading shape; got {leading_shape} vs {m_leading_shape}"
            )
        is_numpy = q_is_numpy or m_is_numpy

        q_syn = self._synthesize(q_prepared).index_select(-1, self._input_unsort)
        m_syn = self._synthesize(m_prepared).index_select(-1, self._input_unsort)

        valid = m_syn > eps
        safe_m = torch.where(valid, m_syn, torch.ones_like(m_syn))
        y = q_syn / safe_m
        fill = float("nan") if restore_mask else 0.0
        y = torch.where(valid, y, torch.full_like(y, fill))

        return self._restore_band(y, leading_shape=leading_shape, is_numpy=is_numpy)

    def expand(
        self,
        decomposition: Union[HealPixPyramid, Sequence[ArrayLike]],
    ) -> tuple[ArrayLike, ...]:
        """Lift every band to the fine grid so their direct sum reconstructs.

        ``compute`` deliberately returns compact native-resolution bands.
        This method applies the stored synthesis operators to each band
        independently, restores the original fine-cell order, and returns
        ``n_bands`` fine-resolution components.  Their elementwise sum equals
        :meth:`invert` up to floating-point round-off.
        """
        prepared, leading_shape, all_numpy = self._prepare_decomposition(
            decomposition
        )
        components: list[ArrayLike] = []
        for band_index, band in enumerate(prepared):
            current = band
            for scale in range(band_index - 1, -1, -1):
                current, fine_ids = self.up_layers[scale](current)
                if not np.array_equal(
                    fine_ids, self.cell_ids_per_scale[scale]
                ):
                    raise RuntimeError(
                        f"Up returned inconsistent cell_ids at scale {scale}"
                    )
            current = current.index_select(-1, self._input_unsort)
            components.append(
                self._restore_band(
                    current,
                    leading_shape=leading_shape,
                    is_numpy=all_numpy,
                )
            )
        return tuple(components)

    def extra_repr(self) -> str:
        domain = "partial" if self.partial else "full-sky"
        return (
            f"level={self.level}, Jmax={self.Jmax}, scales={self.n_scales}, "
            f"bands={self.n_bands}, domain={domain}, sizes={self.sizes}, "
            f"filter='symmetric Gaussian', up_norm={self.up_norm!r}"
        )


__all__ = ["HealPixDecomp", "HealPixPyramid", "HealPixWeightedPyramid"]
