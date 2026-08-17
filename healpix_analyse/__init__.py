"""
healpix_analyse
===============

Signal-analysis and spatial-analysis tools for HEALPix data.

The package contains operators for:

- multi-resolution HEALPix processing,
- spherical-harmonic analysis,
- local FFT analysis,
- Minkowski functionals,
- physical-radius neighbourhood reductions,
- connected-component analysis on HEALPix topology.

Public re-exports
-----------------
The symbols imported below form the main user-facing API.

Examples
--------
Multi-resolution operators::

    from healpix_analyse import HealPixDown, HealPixUp

Neighbourhood reductions::

    from healpix_analyse import neighbour_reduce, median_filter

Connected components::

    from healpix_analyse import (
        connected_components,
        component_size,
        component_area,
        remove_small_components,
    )
"""


# ---------------------------------------------------------------------------
# Multi-resolution operators
# ---------------------------------------------------------------------------

from healpix_analyse.down import HealPixDown
from healpix_analyse.up import HealPixUp
from healpix_analyse.large_conv import LargeConv
from healpix_analyse.fft_conv import HealPixFFTConv

from healpix_analyse.decomp import (
    HealPixDecomp,
    HealPixPyramid,
)

from healpix_analyse.divcurl import (
    HealPixDivCurl,
    HealPixMultiScaleDivCurl,
    HealPixDivCurlPyramid,
)

from healpix_analyse.resample import (
    HealPixResampler,
    resample_healpix,
)


# ---------------------------------------------------------------------------
# Spherical harmonic transforms
# ---------------------------------------------------------------------------

from healpix_analyse.alm_latlon import (
    build_rings_from_latlon,
    anafast_latlon,
    map2alm_latlon,
    alm2map_latlon,
    compute_weights,
    grid_summary,
)

from healpix_analyse.alm import AlmCoeffs


# ---------------------------------------------------------------------------
# Local flat-sky FFT
# ---------------------------------------------------------------------------

from healpix_analyse.fft_local import (
    LocalFFT,
    fft as local_fft,
    ifft as local_ifft,
    ps as local_ps,
)


# ---------------------------------------------------------------------------
# Minkowski functionals
# ---------------------------------------------------------------------------

from healpix_analyse.minkowski import (
    minkowski_functionals,
    minkowski_curves,
    build_healpix_adjacency,
    minkowski_functionals_healpix,
    minkowski_curves_healpix,
)


# ---------------------------------------------------------------------------
# HEALPix physical-radius neighbourhood reductions
#
# These operations use physical/geodesic neighbourhood definitions rather
# than immediate HEALPix topological adjacency.
# ---------------------------------------------------------------------------

from healpix_analyse.neighbour_reduce import (
    HealPixNeighbourReducer,
    neighbour_reduce,
    median_filter,
    mean_filter,
    min_filter,
    max_filter,
)

from healpix_analyse.gradient import (
    directional_derivative,
    gradient,
    gradient_magnitude,
)

from .directional_filter import directional_filter

# ---------------------------------------------------------------------------
# HEALPix connected components
#
# Connected components use immediate NESTED HEALPix topology.
#
# connectivity="edge"
#     Edge-sharing HEALPix cells only.
#     This is the HEALPix analogue of Cartesian 4-connectivity.
#
# connectivity="edge_or_vertex"
#     Cells sharing either an edge or a vertex.
#     This is the HEALPix analogue of Cartesian 8-connectivity.
#
# The topology backend itself is intentionally private.  The current
# implementation uses healpy temporarily through ``_topology.py`` and is
# intended to move to healpix-geo / CDSHEALPix once directional neighbour
# access is available there.
# ---------------------------------------------------------------------------

from healpix_analyse.components import (
    connected_components,
    component_size,
    component_area,
    healpix_cell_area,
    remove_small_components,
)

from healpix_analyse.radial_filter import (
    configure_radial_filter_cache,
    gaussian_filter,
    radial_filter,
    radial_filter_cache_info,
)

__all__ = [
    # ------------------------------------------------------------------
    # Multi-resolution operators
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Spherical harmonic transforms
    # ------------------------------------------------------------------
    "build_rings_from_latlon",
    "anafast_latlon",
    "map2alm_latlon",
    "alm2map_latlon",
    "compute_weights",
    "grid_summary",

    # ------------------------------------------------------------------
    # ALM containers
    # ------------------------------------------------------------------
    "AlmCoeffs",

    # ------------------------------------------------------------------
    # Local flat-sky FFT
    # ------------------------------------------------------------------
    "LocalFFT",
    "local_fft",
    "local_ifft",
    "local_ps",

    # ------------------------------------------------------------------
    # Minkowski functionals — 2D planar
    # ------------------------------------------------------------------
    "minkowski_functionals",
    "minkowski_curves",

    # ------------------------------------------------------------------
    # Minkowski functionals — HEALPix
    # ------------------------------------------------------------------
    "build_healpix_adjacency",
    "minkowski_functionals_healpix",
    "minkowski_curves_healpix",

    # ------------------------------------------------------------------
    # HEALPix neighbourhood reductions
    # ------------------------------------------------------------------
    "HealPixNeighbourReducer",
    "neighbour_reduce",
    "median_filter",
    "mean_filter",
    "min_filter",
    "max_filter",

    # ------------------------------------------------------------------
    # HEALPix geographical weighted filtering
    # ------------------------------------------------------------------
    "directional_filter",

    # ------------------------------------------------------------------
    # HEALPix radial filtering
    # ------------------------------------------------------------------
    "radial_filter",
    "gaussian_filter",
    "configure_radial_filter_cache",
    "radial_filter_cache_info",

    # ------------------------------------------------------------------
    # HEALPix scalar-field gradients
    # ------------------------------------------------------------------
    "gradient",
    "gradient_magnitude",
    "directional_derivative",

    # ------------------------------------------------------------------
    # HEALPix connected components
    # ------------------------------------------------------------------
    "connected_components",
    "component_size",
    "component_area",
    "healpix_cell_area",
    "remove_small_components",
]
