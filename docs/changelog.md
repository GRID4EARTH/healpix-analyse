# Changelog

## Unreleased

### Changed
- HEALPix-facing public APIs now take the Grid4Earth `level` directly and
  derive `nside = 2**level` internally. This applies to `HealPixConv`,
  `LargeConv`, `HealPixDown`, `HealPixUp`, `HEALPixSHT`, localized ALM
  transforms, and `build_healpix_adjacency`.

### Fixed
- `HealPixConv` geometry cache keys retain the input `cell_ids` order, so a
  permutation of the same partial domain cannot reuse incompatible sorting
  buffers.

### Added
- `HealPixDivCurl` and `HealPixMultiScaleDivCurl`: fixed gauge-aware
  derivative kernels for divergence and curl at every decomposition scale.
- `HealPixDecomp` and `HealPixPyramid`: exactly reconstructing local
  Laplacian pyramids with cell identifiers retained at every scale.
- `HealPixFFTConv`: differentiable, zero-padded FFT convolution for very large
  learned kernels on local pole-safe gnomonic HEALPix patches.
- `HEALPixSHT`: ring-based full-sky spherical harmonic transform with spin support (spin-0, spin-1, spin-2)
- `alm_latlon`: SHT for arbitrary iso-latitude grids (ERA5, regular lat/lon, HEALPix)
- `HealPixConv`: gauge-equivariant spherical convolution on HEALPix maps
- `HealPixDown` / `HealPixUp`: multi-resolution operators (smooth and max-pool modes)
- `powerspectra` / `powerspectra_lonlat`: isotropic 1D power spectrum estimation
- `LocalizedFlatSkyAlm`: flat-sky approximation for localized SHT on large patches
