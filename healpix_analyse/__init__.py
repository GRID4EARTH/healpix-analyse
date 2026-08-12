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
- healpix_analyse.fft_local  : fast gnomonic 2D FFT on local HEALPix patches
- healpix_analyse.powerspectra: angular power spectra on HEALPix subsets
- healpix_analyse.convol     : gauge-equivariant spherical convolution
- healpix_analyse.large_conv : multiresolution large-kernel convolution
- healpix_analyse.fft_conv   : FFT-accelerated large-kernel local convolution
- healpix_analyse.decomp     : exactly reconstructing local multiscale pyramid
- healpix_analyse.divcurl    : gauge-aware multiscale divergence and curl
- healpix_analyse.resample   : grid resampling helpers
- healpix_analyse.minkowski  : differentiable Minkowski functionals for 2D images

Public re-exports
-----------------
The symbols below are the primary user-facing API.  Import them directly::

    from healpix_analyse import HealPixDown, HealPixUp
    from healpix_analyse import build_rings_from_latlon, anafast_latlon, map2alm_latlon
    from healpix_analyse import minkowski_functionals, minkowski_curves
"""

from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp
from healpix_analyse.large_conv import LargeConv
from healpix_analyse.fft_conv import HealPixFFTConv
from healpix_analyse.decomp import HealPixDecomp, HealPixPyramid
from healpix_analyse.divcurl import (
    HealPixDivCurl,
    HealPixMultiScaleDivCurl,
    HealPixDivCurlPyramid,
)
from healpix_analyse.resample import HealPixResampler, resample_healpix

from healpix_analyse.alm_latlon import (
    build_rings_from_latlon,
    anafast_latlon,
    map2alm_latlon,
    alm2map_latlon,
    compute_weights,
    grid_summary,
)

from healpix_analyse.alm import AlmCoeffs

from healpix_analyse.fft_local import (
    LocalFFT,
    fft as local_fft,
    ifft as local_ifft,
    ps as local_ps,
)

from healpix_analyse.minkowski import (
    minkowski_functionals,
    minkowski_curves,
    build_healpix_adjacency,
    minkowski_functionals_healpix,
    minkowski_curves_healpix,
)

from healpix_analyse.neighbour_reduce import (
    HealPixNeighbourReducer,
    max_filter,
    mean_filter,
    median_filter,
    min_filter,
    neighbour_reduce,
)

from healpix_analyse.gradient import (
    directional_derivative,
    gradient,
    gradient_magnitude,
)

__all__ = [
    # Multi-resolution operators
    "HealPixDown",
    "HealPixUp",
    "LargeConv",
    "HealPixFFTConv",
    "HealPixDecomp",
    "HealPixPyramid",
    "HealPixDivCurl",
    "HealPixMultiScaleDivCurl",
    "HealPixDivCurlPyramid",
    "HealPixResampler",
    "resample_healpix",
    # Spherical harmonic transforms (arbitrary lat/lon ring grids)
    "build_rings_from_latlon",
    "anafast_latlon",
    "map2alm_latlon",
    "alm2map_latlon",
    "compute_weights",
    "grid_summary",
    # ALM containers
    "AlmCoeffs",
    # Local flat-sky FFT
    "LocalFFT",
    "local_fft",
    "local_ifft",
    "local_ps",
    # Minkowski functionals — 2D planar
    "minkowski_functionals",
    "minkowski_curves",
    # Minkowski functionals — HEALPix spherical
    "build_healpix_adjacency",
    "minkowski_functionals_healpix",
    "minkowski_curves_healpix",
    # HEALPix neighbourhood reductions
    "HealPixNeighbourReducer",
    "neighbour_reduce",
    "median_filter",
    "mean_filter",
    "min_filter",
    "max_filter",
    # HEALPix scalar-field gradients
    "gradient",
    "gradient_magnitude",
    "directional_derivative",
]
