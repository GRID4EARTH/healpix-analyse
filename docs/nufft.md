# Non-uniform FFT

The `healpix_analyse.nufft` module computes two-dimensional Fourier transforms
of samples that are not on a regular grid, such as the centres of the HEALPix
cells of a regional domain.
It is a thin wrapper around
[finufft.nufft2d1](https://finufft.readthedocs.io/en/latest/python.html#finufft.nufft2d1)
from the [fiNUFFT](https://github.com/flatironinstitute/finufft) package [1],
an optional dependency:

```bash
pip install "healpix-analyse[nufft]"   # or: pip install finufft
```

## What it computes

For samples $c_j$ at $(x_j, y_j)$, `nufft` returns

$$
F(f_x, f_y) = \sum_j c_j \, \exp\left(-2 i \pi \left[f_x (x_j - x_0) + f_y (y_j - y_0)\right]\right)
$$

with $x_0, y_0$ the smallest coordinates. This is the sign convention of
`numpy.fft`, and the phase reference is the corner $(x_0, y_0)$ of the domain.

**Input coordinates:** `x` and `y` both have shape (M), M being the number of
samples. They are used as given, so the frequencies are in cycles per unit of
`x` and `y`. It is up to the user to convert the coordinates to suitable units
and projection: with longitudes and latitudes in degrees the frequencies are in
cycles per degree, and a degree of longitude is shorter than a degree of
latitude; after a projection to metres (for instance UTM with `pyproj`) they
are in cycles per metre.

**Input data:** `data` can be of shape (M), or (K, M). In the latter case,
fiNUFFT computes the K transforms in one batch, saving some overhead time. The
data are converted to a complex type.

**Output:** the coefficients have shape (Ny, Nx), or (K, Ny, Nx) for batched
data, with Nx and Ny the numbers of frequencies requested with
`n_frequencies=(Nx, Ny)`. The frequencies `fx` and `fy` are returned in
increasing order.

**Frequencies and period:** the transform treats the data as periodic, with a
period $L$ along each coordinate that must be larger than the extent of the
coordinates. With $N$ modes, the frequencies are $k / L$ for
$k = -\lfloor N/2 \rfloor, \dots, N - \lfloor N/2 \rfloor - 1$, the order of
`np.fft.fftshift(np.fft.fftfreq(N, d=L / N))`.
By default $L$ is the extent of the coordinate plus the median spacing between
its distinct values. On a regular grid of $n$ samples with spacing $dx$ this is
$n \, dx$, and the result equals `np.fft.fftshift(np.fft.fft2(...))` of the
gridded data. For scattered samples such as HEALPix cell centres the median
spacing is small and the period is close to the extent; pass
`period=(Lx, Ly)` to choose it, for instance the extent plus one cell size.

**Number of modes:** frequencies above the Nyquist frequency of the sampling,
$1 / (2 \, dx)$ for a sample spacing $dx$, are aliased. Since the HEALPix grid is
close to a regular grid, taking as many modes along each direction as there are
cells across the domain is a reasonable choice.

**Precision:** the NUFFT algorithm keeps the complexity low
($\mathcal{O}(N\log{N} + M)$) by making a number of approximations. The user can
directly specify the relative tolerance `rtol` to achieve. Smaller values give
better accuracy but slower computations.

**Windowing:** since the data are treated as periodic, a jump at the border of
the domain spreads power over all frequencies. `windowing_weights_polygon` and
`windowing_weights_bbox` return Tukey tapers: 0 outside the polygon (or box)
and on its border, rising to 1 inside over a distance `length` following a
cosine. The polygon can be concave and its vertices can be given in either
orientation; distances are Euclidean, in the units of the coordinates.

:::{attention}
fiNUFFT also provides the type-2 transform (`finufft.nufft2d2`, not wrapped
here), which evaluates a Fourier series at the sample points. Contrary to the
regular FFT, it is not the inverse of `nufft` but its adjoint: applied to the
coefficients, it does not give back the samples with the right amplitudes.
Reconstructing a field from its coefficients should rather be treated as an
inverse problem, solving for the sample values, for instance with conjugate
gradients. Since the HEALPix grid is close to regular, the system should be
well conditioned.
:::

## Example

Here we select the cells inside a polygon, taper the data and compute their
power spectrum:

```python
import healpix_geo.nested as hgn
import matplotlib.pyplot as plt
import numpy as np

from healpix_analyse.nufft import nufft, windowing_weights_polygon

level = 12  # cells of about 1.6 km

# A regional domain, as (lon, lat) vertices in degrees
vertices = np.array([[2.0, 48.5], [2.5, 50.0], [1.0, 52.0], [0.5, 50.5]])
cell_ids, _, _ = hgn.polygon_coverage(vertices, level, ellipsoid="WGS84")
lon, lat = hgn.healpix_to_lonlat(cell_ids, level, ellipsoid="WGS84")

# A synthetic field: a wave of 0.25 degree wavelength along longitude, plus noise
rng = np.random.default_rng(0)
data = np.cos(2 * np.pi * lon / 0.25) + 0.5 * rng.standard_normal(lon.size)

# Taper over 0.2 degree inside the polygon, after removing the mean
weights = windowing_weights_polygon(lon, lat, vertices, length=0.2)

# About as many modes as cells across the domain in each direction
fx, fy, spectrum = nufft(
    lon, lat, (data - data.mean()) * weights, n_frequencies=(100, 270)
)

fig, ax = plt.subplots()
ax.pcolormesh(fx, fy, np.abs(spectrum) ** 2, norm="log")
ax.set_xlabel("longitude frequency [cycles per degree]")
ax.set_ylabel("latitude frequency [cycles per degree]")
```

The power peaks at $\pm 4$ cycles per degree of longitude, at zero latitude
frequency.

## References

- [1] A parallel non-uniform fast Fourier transform library based on an ``exponential of semicircle'' kernel. A. H. Barnett, J. F. Magland, and L. af Klinteberg. SIAM J. Sci. Comput. 41(5), C479-C504 (2019).

- [2] cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs, Yu-hsuan Shih, Garrett Wright, Joakim Andén, Johannes Blaschke, Alex H. Barnett, PDSEC2021 workshop of the IPDPS2021 conference. https://arxiv.org/abs/2102.08463
