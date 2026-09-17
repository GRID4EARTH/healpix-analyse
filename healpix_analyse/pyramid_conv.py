"""Pyramidal convolution on HEALPix maps: block-diagonal kernel pyramid +
:class:`~healpix_analyse.decomp.HealPixDecomp`, with correct NaN/weight
propagation.

See :mod:`healpix_analyse.kernel_pyramid` for the precise sense in which a
per-band kernel is an *approximation* (block-diagonal in ``B_K``), and for
the mandatory validation step against :mod:`healpix_analyse.validation`.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from healpix_analyse.decomp import HealPixDecomp, HealPixPyramid, HealPixWeightedPyramid
from healpix_analyse.kernel_pyramid import HealPixKernelPyramid

ArrayLike = Union[np.ndarray, torch.Tensor]


class HealPixPyramidConv(nn.Module):
    """Apply a :class:`~healpix_analyse.kernel_pyramid.HealPixKernelPyramid`
    through a matched :class:`~healpix_analyse.decomp.HealPixDecomp`.

    Parameters
    ----------
    decomp : HealPixDecomp
        Must be the same decomposition the kernel pyramid was built from
        (same levels, cell-id domains).
    kernel_pyramid : HealPixKernelPyramid
    mode : {"normalized", "signed"}, default "normalized"
        ``"normalized"`` (default): NaN/weight-aware. Internally runs
        :meth:`~healpix_analyse.decomp.HealPixDecomp.compute_weighted` on the input, applies the
        kernel pyramid identically to the data channel ``q`` and the weight
        channel ``m``, synthesizes both, and divides **once** --
        ``y = S(B_K q) / S(B_K m)`` -- never band by band. This is a strict
        superset of plain filtering: on fully finite data with no
        ``weights`` given, ``m`` is uniformly 1 and the division is a no-op
        up to floating-point round-off, so this mode is safe to use as the
        default even without missing data.
        ``"signed"``: no masking at all -- runs the kernel pyramid directly
        on :meth:`~healpix_analyse.decomp.HealPixDecomp.compute`/:meth:`~healpix_analyse.decomp.HealPixDecomp.invert`. Only
        meaningful on fully finite input; useful for validating kernels that
        are not required to stay positive (signed kernels), where dividing
        by a synthesized weight would be the wrong operation.
    """

    def __init__(
        self,
        decomp: HealPixDecomp,
        kernel_pyramid: HealPixKernelPyramid,
        *,
        mode: str = "normalized",
    ) -> None:
        super().__init__()
        if mode not in ("normalized", "signed"):
            raise ValueError("mode must be 'normalized' or 'signed'")
        if kernel_pyramid.decomp is not decomp:
            # Not fatal (levels/cell_ids equality is what actually matters),
            # but almost always a mistake, so fail loudly.
            if (
                tuple(kernel_pyramid.decomp.levels) != tuple(decomp.levels)
                or kernel_pyramid.decomp.n_bands != decomp.n_bands
            ):
                raise ValueError(
                    "kernel_pyramid was not built from a decomposition "
                    "compatible with `decomp` (levels/band count differ)"
                )
        self.decomp = decomp
        self.kernel_pyramid = kernel_pyramid
        self.mode = mode

    def apply_pyramid(
        self, pyramid: Union[HealPixPyramid, HealPixWeightedPyramid]
    ) -> Union[HealPixPyramid, HealPixWeightedPyramid]:
        """Apply the per-band kernels to an already-computed pyramid.

        Does **not** synthesize or divide -- use :meth:`forward` (or
        :meth:`~healpix_analyse.decomp.HealPixDecomp.invert`) for that. Accepts either a plain
        :class:`~healpix_analyse.decomp.HealPixPyramid` or a
        :class:`~healpix_analyse.decomp.HealPixWeightedPyramid` (in which
        case the *same* per-band kernels are applied to both the ``q`` and
        ``m`` channels, as required for a valid normalized-convolution
        result).
        """
        if isinstance(pyramid, HealPixWeightedPyramid):
            q_bands = self.kernel_pyramid.apply(pyramid.q.bands)
            m_bands = self.kernel_pyramid.apply(pyramid.m.bands)
            q = HealPixPyramid(bands=q_bands, cell_ids=pyramid.cell_ids, levels=pyramid.levels)
            m = HealPixPyramid(bands=m_bands, cell_ids=pyramid.cell_ids, levels=pyramid.levels)
            return HealPixWeightedPyramid(q=q, m=m)

        bands = pyramid.bands if isinstance(pyramid, HealPixPyramid) else tuple(pyramid)
        conv_bands = self.kernel_pyramid.apply(bands)
        cell_ids = pyramid.cell_ids if isinstance(pyramid, HealPixPyramid) else self.decomp.cell_ids
        levels = pyramid.levels if isinstance(pyramid, HealPixPyramid) else self.decomp.levels
        return HealPixPyramid(bands=conv_bands, cell_ids=cell_ids, levels=levels)

    def forward(
        self,
        x: ArrayLike,
        weights: Optional[ArrayLike] = None,
        *,
        return_support: bool = False,
        restore_mask: bool = True,
        eps: float = 1e-8,
    ):
        """Convolve ``x`` through the pyramid.

        In ``"normalized"`` mode (default), returns the masked/weighted
        result ``y = S(B_K q) / S(B_K m)``; when ``return_support=True``,
        also returns the synthesized support map ``S(B_K m)`` (the
        post-convolution confidence/coverage, useful for diagnosing where
        the result is under-supported without being a per-band clip).

        In ``"signed"`` mode, ``weights``/``return_support`` are not
        accepted (raises ``ValueError``) -- there is no weight channel to
        report support from.
        """
        if self.mode == "signed":
            if weights is not None:
                raise ValueError("mode='signed' does not accept `weights`")
            if return_support:
                raise ValueError("mode='signed' has no support map to return")
            pyramid = self.decomp.compute(x)
            conv_pyramid = self.apply_pyramid(pyramid)
            return self.decomp.invert(conv_pyramid)

        wpyramid = self.decomp.compute_weighted(x, weights=weights)
        conv_wpyramid = self.apply_pyramid(wpyramid)
        y = self.decomp.invert(conv_wpyramid, restore_mask=restore_mask, eps=eps)
        if return_support:
            support = self.decomp.invert(conv_wpyramid.m)
            return y, support
        return y


__all__ = ["HealPixPyramidConv"]
