"""get_interp_weights: longitudes a hair below 0 fold into [0, 2 pi)."""

import numpy as np

from healpix_analyse.healpix_interp import get_interp_weights


def test_tiny_negative_longitude_matches_zero():
    for lon_neg in (-6.28e-15, -1e-14, -5e-16):
        p0, w0 = get_interp_weights(np.array([0.0]), np.array([5.379]), 6)
        p1, w1 = get_interp_weights(np.array([lon_neg]), np.array([5.379]), 6)
        order0, order1 = np.argsort(p0[0]), np.argsort(p1[0])
        np.testing.assert_array_equal(p0[0][order0], p1[0][order1])
        np.testing.assert_allclose(w0[0][order0], w1[0][order1], atol=1e-12)
