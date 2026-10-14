"""Partial-sky operators must not depend on the storage order of cell_ids.

Position ``i`` of the input is pixel ``cell_ids[i]``; the output must follow
the same convention whatever the order of ``cell_ids``.
"""

import numpy as np
import pytest
import torch

from healpix_analyse.convol import HealPixConv
from healpix_analyse.down import HealPixDown

LEVEL = 3
NPIX = 12 * (2**LEVEL) ** 2


@pytest.fixture
def permuted():
    ids = np.arange(NPIX)
    perm = np.random.default_rng(1).permutation(NPIX)
    x = torch.randn(NPIX, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    return ids, perm, x


def test_conv_output_follows_cell_ids_order(permuted):
    ids, perm, x = permuted
    kwargs = dict(
        level=LEVEL, in_channels=1, out_channels=2, kernel_sz=3, n_gauges=2,
        nest=True, device="cpu", dtype=torch.float64, cache_dir=None,
    )
    weights = np.random.default_rng(2).standard_normal((1, 2, 9))
    sorted_conv = HealPixConv(cell_ids=ids, **kwargs)
    unsorted_conv = HealPixConv(cell_ids=ids[perm], **kwargs)
    sorted_conv.set_kernel(weights)
    unsorted_conv.set_kernel(weights)
    y_sorted = sorted_conv(x[None, None, :])
    y_unsorted = unsorted_conv(x[perm][None, None, :])
    torch.testing.assert_close(y_unsorted[..., np.argsort(perm)], y_sorted)


@pytest.mark.parametrize("mode", ["smooth", "maxpool"])
def test_down_output_follows_cell_ids_order(permuted, mode):
    ids, perm, x = permuted
    kwargs = dict(level=LEVEL, mode=mode, device="cpu", dtype=torch.float64)
    sorted_down = HealPixDown(cell_ids=ids, **kwargs)
    unsorted_down = HealPixDown(cell_ids=ids[perm], **kwargs)
    y_sorted, out_sorted = sorted_down(x[None, :])
    y_unsorted, out_unsorted = unsorted_down(x[perm][None, :])
    assert np.array_equal(out_sorted, out_unsorted)
    torch.testing.assert_close(y_unsorted, y_sorted)
