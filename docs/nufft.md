
# Non-uniform FFT

The {mod}`nufft` module allows to compute two-dimensional FFTs on any HEALPix
input.
It relies on the [fiNUFFT](https://github.com/flatironinstitute/finufft) package to compute fast Fourier transform from non-equispaced input .
Install with `pip install finufft`.

## Algorithm

This module is a thin wrapper around
[finufft.nufft2d1](https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1).

**Input coordinates:** Input coordinates should both be of shape (M), M being
the number of samples. The module will automatically map the coordinates to the
$[-\pi; \pi]$ interval that fiNUFFT expects. We leave to the user the
responsibility to convert the input coordinates into appropriate units and
projection.

**Input data:** Input data can be of shape (M), or (K, M). In the latter case,
fiNUFFT will compute FFTs in batches, saving some overhead time. It will
automatically be converted to a complex data type.

**Output frequencies:** The user can specify the number of frequencies (or modes) to compute for each dimension. For the 1D case (to simplify) if we take a coordinate $x$ that span a domain of size $L$, and retrieve $N$ frequencies. fiNUFFT will compute the frequencies as integer multiple of $1/L$, in $[-\frac{N}{2L}; \frac{N}{2L}]$ (right side excluded if N is even).

N should be chosen to follow the Nyqist criterion. Since
the HEALPix grid is close to a regular grid, taking as many modes as they are samples along a direction is a reasonable choice.

**Precision:** The NUFFT algorithm keeps the complexity low
($\mathcal{O}(N\log{N} + M)$) by making a number of approximations. The user can
directly specify the relative tolerance to achieve. Smaller values give better
accuracy but slower computations.

:::{attention}
Despite the "inverse" NUFFT function being available, contrary to the
traditional FFT it is not the true inverse operation (it should rather be called
an adjoint transform). Reconstructing a signal is possible, but the amplitudes
will not match correctly. It is recommended to treat it as an inverse problem, solving
for the signal amplitudes, for instance using conjugate gradients. Since the HEALPix grid
is close to regular, the system should be well conditioned. See [1] for details.
:::

## Example

For example here we select data inside a polygon and compute a FFT:
```python
from healpix_geo.auto import Grid

grid = Grid(level=12, indexing_scheme="nested", ellipsoid="WGS84")

# Select a region from a polygon
region_vertices = np.array([[2.0, 48.5], [2.5, 50], [1.0, 52.0], [0.5, 50.5]])
region_cell_ids, _ , _ = healpix_geo.auto.polygon_coverage(region_vertices, grid=grid)
region_data = ...

lon, lat = healpix_geo.auto.healpix_to_lonlat(region_cell_ids, grid=grid)

weights = windowing_weights_polygon(lon, lat, region_vertices, length=0.5)

# TODO: Add example for passing to UTM coordinates

fx, fy, fft = nufft.nufft(
    lon,
    lat,
    (region_data - np.mean(region_data)) * weights,
    n_frequencies=[200, 200],
)

fig, ax = plt.subplots()
ax.pcolormesh(fx, fy, np.abs(fft)**2, norm="log")
ax.set_xlabel("[$°^{-1}$]")
ax.set_ylabel("[$°^{-1}$]")
```


## References

- [1] A parallel non-uniform fast Fourier transform library based on an ``exponential of semicircle'' kernel. A. H. Barnett, J. F. Magland, and L. af Klinteberg. SIAM J. Sci. Comput. 41(5), C479-C504 (2019).

- [2] cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs, Yu-hsuan Shih, Garrett Wright, Joakim Andén, Johannes Blaschke, Alex H. Barnett, PDSEC2021 workshop of the IPDPS2021 conference. https://arxiv.org/abs/2102.08463 
