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
from healpix_analyse.large_conv import GaussianPlan, LargeConv, down_up_blur, gaussian_plan
from healpix_analyse.fft_conv import HealPixFFTConv
from healpix_analyse.geo_fft import HealPixGeoFFT, geo_fft_convolve

from healpix_analyse.decomp import (
    HealPixDecomp,
    HealPixPyramid,
    HealPixWeightedPyramid,
)

from healpix_analyse.kernel_pyramid import (
    HealPixKernelPyramid,
    kernel_gaussian,
    kernel_exponential,
    kernel_lorentzian,
    kernel_beta,
    kernel_anisotropic_gaussian,
)

from healpix_analyse.pyramid_conv import HealPixPyramidConv

from healpix_analyse.wide_conv import HealPixWideConv

from healpix_analyse.divcurl import (
    HealPixDivCurl,
    HealPixMultiScaleDivCurl,
    HealPixDivCurlPyramid,
)

from healpix_analyse.resample import (
    HealPixResampler,
    resample_healpix,
    resample_to_latlon_grid,
)
from healpix_analyse.healpix_interp import get_interp_val, get_interp_weights
from healpix_analyse.convol import HealPixConv
from healpix_analyse.attention import (
    DEFAULT_WINDOW_SHIFT,
    WINDOW_SHIFTS,
    HealPixNeighbourhoodAttention,
    HealPixWindowAttention,
)
from healpix_analyse.morphology import binary_dilation, binary_erosion


# ---------------------------------------------------------------------------
# Spherical harmonic transforms
# ---------------------------------------------------------------------------

from healpix_analyse.healpix_sht import HEALPixSHT
from healpix_analyse.alm_latlon import (
    build_rings_from_latlon,
    anafast_latlon,
    map2alm_latlon,
    alm2map_latlon,
    compute_weights,
    grid_summary,
)


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
# Wavelet power spectrum (NaN- and mask-aware, differentiable)
# ---------------------------------------------------------------------------

from healpix_analyse.wavelet_spectrum import (
    HealPixWaveletSpectrum,
    HealPixIsotropicWaveletSpectrum,
    wavelet_power_spectrum,
    wavelet_morlet,
    wavelet_morlet_isotropic,
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
    HealPixGradient,
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
# The topology backend itself is intentionally private. ``_topology.py``
# delegates direction-preserving immediate-neighbour access to healpix-geo /
# CDSHEALPix.
# ---------------------------------------------------------------------------

from healpix_analyse.components import (
    connected_components,
    component_size,
    component_area,
    healpix_cell_area,
    remove_small_components,
)

from healpix_analyse.multiorder_filter import multiorder_filter_operator
from healpix_analyse.radial_filter import (
    FilterOperator,
    apply_filter_operator,
    configure_radial_filter_cache,
    gaussian_filter,
    gaussian_filter_operator,
    radial_filter,
    radial_filter_operator,
    radial_filter_cache_info,
)

# ---------------------------------------------------------------------------
# Foundation-model embeddings (DINOv3 SAT-493M on NESTED blocks)
#
# The backbone is loaded at run time from Meta's own distribution and is
# subject to the DINOv3 License; nothing of it is vendored here (see
# docs/other/dino.md).
# ---------------------------------------------------------------------------

from healpix_analyse.dino import (
    GetDINOV3SAT,
    DINOEmbedding,
    load_dinov3_sat,
    nested_to_tiles,
    tiles_to_nested,
)

# ---------------------------------------------------------------------------
# Batched data augmentation (GPU, per-sample random draws)
# ---------------------------------------------------------------------------

from healpix_analyse.augment import (
    HealPixAugment,
    AugmentPlan,
    AugmentResult,
    RandomRotation,
    RandomSO3,
    RandomShift,
    RandomZoom,
    RandomFlip,
    Rotation3D,
    TangentLinear,
    RandomGaugeRotation,
    AdditiveNoise,
    RandomMask,
    RandomCrop,
    gauge_rotate,
)

# ---------------------------------------------------------------------------
# AI layers: masked normalisation, pooling and positional encodings (#109)
# ---------------------------------------------------------------------------

from healpix_analyse.masked_norm import (
    HealPixGroupNorm,
    HealPixLayerNorm,
    HealPixBatchNorm,
    HealPixGlobalPool,
    HealPixRegionPool,
    global_pool,
    region_pool,
)
from healpix_analyse.positional_encoding import (
    HealPixPositionalEncoding,
    healpix_positional_encoding,
    positional_encoding_cache_info,
    clear_positional_encoding_cache,
)

__all__ = [
    # ------------------------------------------------------------------
    # Multi-resolution operators
    # ------------------------------------------------------------------
    "HealPixDown",
    "HealPixUp",
    "LargeConv",
    "down_up_blur",
    "gaussian_plan",
    "GaussianPlan",
    "HealPixFFTConv",
    "HealPixGeoFFT",
    "geo_fft_convolve",
    "HealPixDecomp",
    "HealPixWideConv",
    "HealPixPyramid",
    "HealPixWeightedPyramid",
    "HealPixKernelPyramid",
    "HealPixPyramidConv",
    "kernel_gaussian",
    "kernel_exponential",
    "kernel_lorentzian",
    "kernel_beta",
    "kernel_anisotropic_gaussian",
    "HealPixDivCurl",
    "HealPixMultiScaleDivCurl",
    "HealPixDivCurlPyramid",
    "HealPixResampler",
    "resample_healpix",
    "resample_to_latlon_grid",
    "get_interp_val",
    "get_interp_weights",
    "HealPixConv",
    "HealPixWindowAttention",
    "HealPixNeighbourhoodAttention",
    "WINDOW_SHIFTS",
    "DEFAULT_WINDOW_SHIFT",
    "binary_dilation",
    "binary_erosion",

    # ------------------------------------------------------------------
    # Spherical harmonic transforms
    # ------------------------------------------------------------------
    "HEALPixSHT",
    "build_rings_from_latlon",
    "anafast_latlon",
    "map2alm_latlon",
    "alm2map_latlon",
    "compute_weights",
    "grid_summary",

    # ------------------------------------------------------------------
    # Local flat-sky FFT
    # ------------------------------------------------------------------
    "LocalFFT",
    "local_fft",
    "local_ifft",
    "local_ps",
    # ------------------------------------------------------------------
    # Wavelet power spectrum
    # ------------------------------------------------------------------
    "HealPixWaveletSpectrum",
    "HealPixIsotropicWaveletSpectrum",
    "wavelet_power_spectrum",
    "wavelet_morlet",
    "wavelet_morlet_isotropic",

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
    "FilterOperator",
    "apply_filter_operator",
    "gaussian_filter_operator",
    "multiorder_filter_operator",
    "radial_filter_operator",
    "configure_radial_filter_cache",
    "radial_filter_cache_info",

    # ------------------------------------------------------------------
    # HEALPix scalar-field gradients
    # ------------------------------------------------------------------
    "gradient",
    "gradient_magnitude",
    "directional_derivative",
    "HealPixGradient",

    # ------------------------------------------------------------------
    # HEALPix connected components
    # ------------------------------------------------------------------
    "connected_components",
    "component_size",
    "component_area",
    "healpix_cell_area",
    "remove_small_components",

    # ------------------------------------------------------------------
    # DINOv3 embeddings of NESTED blocks
    # ------------------------------------------------------------------
    "GetDINOV3SAT",
    "DINOEmbedding",
    "load_dinov3_sat",
    "nested_to_tiles",
    "tiles_to_nested",

    # ------------------------------------------------------------------
    # Batched data augmentation
    # ------------------------------------------------------------------
    "HealPixAugment",
    "AugmentPlan",
    "AugmentResult",
    "RandomRotation",
    "RandomSO3",
    "RandomShift",
    "RandomZoom",
    "RandomFlip",
    "Rotation3D",
    "TangentLinear",
    "RandomGaugeRotation",
    "AdditiveNoise",
    "RandomMask",
    "RandomCrop",
    "gauge_rotate",

    # ------------------------------------------------------------------
    # AI layers: masked normalisation, pooling, positional encodings
    # ------------------------------------------------------------------
    "HealPixGroupNorm",
    "HealPixLayerNorm",
    "HealPixBatchNorm",
    "HealPixGlobalPool",
    "HealPixRegionPool",
    "global_pool",
    "region_pool",
    "HealPixPositionalEncoding",
    "healpix_positional_encoding",
    "positional_encoding_cache_info",
    "clear_positional_encoding_cache",
]
