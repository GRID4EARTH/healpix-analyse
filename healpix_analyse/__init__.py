"""
healpix_analyse
===============
PyTorch-based signal analysis tools for HEALPix maps.

Components
----------
- healpix_analyse.down       : HealPixDown — resolution reduction (Gaussian smooth or maxpool)
- healpix_analyse.up         : HealPixUp   — resolution increase (adjoint of smooth downsampling)
- healpix_analyse.alm_latlon : ring-based SHT for arbitrary lat/lon grids (map2alm, anafast)
- healpix_analyse.alm        : local spherical harmonic coefficients (AlmCoeffs, AlmTransform)
- healpix_analyse.healpix_sht: ring-FFT SHT optimised for full-sky HEALPix maps
- healpix_analyse.powerspectra: angular power spectra on HEALPix subsets
- healpix_analyse.convol     : gauge-equivariant spherical convolution
- healpix_analyse.resample   : grid resampling helpers

Public re-exports
-----------------
The symbols below are the primary user-facing API.  Import them directly::

    from healpix_analyse import HealPixDown, HealPixUp
    from healpix_analyse import build_rings_from_latlon, anafast_latlon, map2alm_latlon
"""

from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp

from healpix_analyse.alm_latlon import (
    build_rings_from_latlon,
    anafast_latlon,
    map2alm_latlon,
    alm2map_latlon,
    compute_weights,
    grid_summary,
)

from healpix_analyse.alm import AlmCoeffs

__all__ = [
    # Multi-resolution operators
    "HealPixDown",
    "HealPixUp",
    # Spherical harmonic transforms (arbitrary lat/lon ring grids)
    "build_rings_from_latlon",
    "anafast_latlon",
    "map2alm_latlon",
    "alm2map_latlon",
    "compute_weights",
    "grid_summary",
    # ALM containers
    "AlmCoeffs",
]
