# Local two-dimensional FFT on HEALPix patches

`healpix_analyse.fft_local` computes a conventional two-dimensional FFT after
projecting a local set of HEALPix samples onto a square gnomonic grid. It is
intended for local flat-sky analysis, image-like diagnostics and local power
spectra—not as a replacement for a spherical harmonic transform on large
regions or the full sphere.

The implementation is:

- compatible with NESTED HEALPix identifiers;
- safe at both poles and across the 0/360-degree meridian;
- based on PyTorch and usable on CPU or CUDA;
- differentiable with respect to the input values;
- reusable: all geometric and interpolation tables are computed once;
- compatible with NumPy arrays and PyTorch tensors;
- able to process arbitrary leading batch dimensions `[..., N]`.

## Quick start

```python
import torch
from healpix_analyse.fft_local import LocalFFT

transform = LocalFFT(
    cell_ids,
    level=12,
    ellipsoid="sphere",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

spectrum = transform.fft(data)
reconstructed = transform.ifft(spectrum)

print(transform.grid_shape)
print(transform.patch_radius_deg)
```

`cell_ids` contains unique NESTED identifiers at the same `level`, and the
last dimension of `data` must follow exactly the same order.

For several fields defined on the same cells, construct `LocalFFT` only once:

```python
spectra = [transform.fft(field) for field in fields]
```

The geometry construction uses `healpix-geo` and NumPy on the CPU. Projection,
FFT, IFFT and back-projection then run on the device holding the transform's
buffers.

## Functional interface

The module also exposes two convenience functions:

```python
from healpix_analyse.fft_local import fft, ifft

spectrum, transform = fft(
    cell_ids,
    level,
    data,
    return_transform=True,
    device="cuda",
)
reconstructed = ifft(spectrum, transform)
```

Passing the transform to `ifft` is recommended. It guarantees that the same
projection grid is used and avoids rebuilding the geometry. It is also
possible to rebuild it deterministically:

```python
reconstructed = ifft(
    spectrum,
    cell_ids=cell_ids,
    level=level,
    device="cuda",
)
```

## Geometry

### Spherical centre

The longitude and latitude of every HEALPix cell centre are converted to a
three-dimensional unit vector $\boldsymbol n_i$. The centre of the local patch
is the normalised vector mean

$$
\boldsymbol n_0 =
\frac{\sum_i \boldsymbol n_i}
     {\left\|\sum_i \boldsymbol n_i\right\|}.
$$

This avoids averaging longitude directly, which would fail for a patch
crossing 0/360 degrees.

### Tangent frame and pole handling

Two orthonormal vectors $\boldsymbol e_x$ and $\boldsymbol e_y$ are built in
the plane perpendicular to $\boldsymbol n_0$. Away from the poles, the global
north direction is used to orient the frame. Close to a pole, a Cartesian
reference axis is selected instead. Consequently the construction never
divides by `cos(latitude)` and has no geographic pole singularity.

The gnomonic coordinates of cell $i$ are

$$
x_i = \frac{\boldsymbol n_i \cdot \boldsymbol e_x}
           {\boldsymbol n_i \cdot \boldsymbol n_0},
\qquad
y_i = \frac{\boldsymbol n_i \cdot \boldsymbol e_y}
           {\boldsymbol n_i \cdot \boldsymbol n_0}.
$$

The coordinates are dimensionless tangent-plane coordinates (`tan(angle)`).
For small patches they are numerically close to angular offsets in radians.

### Patch-size guard

The default maximum angular distance from the computed centre is 10 degrees:

```python
LocalFFT(cell_ids, level, max_patch_radius_deg=10.0)
```

A larger patch raises `ValueError`. Gnomonic projection is mathematically
defined up to 90 degrees from its centre, but distortion grows rapidly before
that limit. Splitting large regions into overlapping local patches is safer
for spectral analysis.

Coverage functions can include cells whose centres lie slightly outside the
requested region because the cells themselves intersect its boundary. For
example, a nominal five-degree cone may have a measured cell-centre radius of
approximately 5.3 degrees.

## Projected grid

The default grid spacing is derived from the square root of the HEALPix pixel
area:

$$
\Delta = \frac{\sqrt{\pi/3}}{n_{\rm side}},
\qquad n_{\rm side}=2^{\text{level}}.
$$

This is a nominal linear scale, not an assertion that HEALPix centres form a
Cartesian lattice. It can be overridden with `pixel_size_rad`.

The grid is square, centred on the tangent origin, and includes a one-pixel
margin around all projected centres. By default its side is rounded up to the
next power of two for efficient FFT execution. A fixed size can be requested
with `grid_size`; an error is raised when it is too small.

Useful geometry attributes are:

| Attribute | Meaning |
|---|---|
| `n_cells` | Number of input HEALPix cells |
| `grid_size`, `grid_shape` | Side and shape of the square FFT grid |
| `centre_lon_deg`, `centre_lat_deg` | Computed spherical centre |
| `patch_radius_deg` | Maximum angular cell-centre distance |
| `pixel_size_rad` | Gnomonic grid spacing |
| `projected_x`, `projected_y` | Position of every input cell in the tangent plane |
| `coverage_mask` | Boolean grid mask reached by at least one input cell |
| `grid_density` | Sum of interpolation weights on every grid point |

## Bilinear gridding

Each HEALPix value contributes to the four surrounding grid points with
bilinear weights $w_{ig}$. Overlapping contributions are normalised by their
accumulated density:

$$
G_g =
\frac{\sum_i w_{ig}d_i}{\sum_i w_{ig}}.
$$

Uncovered grid points are zero. The operation is implemented with cached
indices and `torch.scatter_add`, so it is fast, batched and differentiable
with respect to $d_i$.

The projected grid can be inspected directly:

```python
grid = transform.project(data)
back_projected = transform.unproject(grid)
mask = transform.coverage_mask
```

## FFT conventions

`LocalFFT.fft` returns the unshifted complex FFT:

```python
spectrum = transform.fft(data)
display_spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
```

The default normalisation is `norm="ortho"`. With this convention, Parseval's
identity on the projected grid has no extra grid-size factor:

$$
\sum_g |G_g|^2 = \sum_k |F_k|^2.
$$

`"forward"` and `"backward"` are also accepted and are passed unchanged to
`torch.fft.fft2` and `torch.fft.ifft2`.

For a grid spacing $\Delta$, frequency axes in cycles per tangent unit are
obtained with:

```python
frequency = torch.fft.fftfreq(
    transform.grid_size,
    d=transform.pixel_size_rad,
    device=transform.device,
)
```

On a sphere of radius $R$, a small-patch physical approximation uses
`d = R * transform.pixel_size_rad`, giving frequencies in cycles per metre.

## Meaning of IFFT and reconstruction error

The FFT/IFFT pair is exact, up to floating-point precision, on the projected
grid:

```python
grid = transform.project(data)
grid_roundtrip = torch.fft.ifft2(
    torch.fft.fft2(grid, norm=transform.norm),
    norm=transform.norm,
).real
```

The complete HEALPix round trip is different:

```text
HEALPix values -> bilinear grid -> FFT -> IFFT -> bilinear sampling
```

The last bilinear sampling is a fast approximate inverse, not the matrix
pseudoinverse of the forward gridding operation. Therefore
`transform.ifft(transform.fft(data))` is generally approximate. A constant
field is preserved, while spatially varying fields are slightly smoothed.

Example results for smooth level-7 patches in float64 are:

| Location | Relative RMS error | Constant-field maximum error |
|---|---:|---:|
| Equator | 0.968% | $4.44\,10^{-16}$ |
| Across 0/360 degrees | 1.006% | $4.44\,10^{-16}$ |
| North pole | 1.684% | $4.44\,10^{-16}$ |
| South pole | 1.684% | $4.44\,10^{-16}$ |

These figures validate the coordinate construction but are not universal
accuracy guarantees. Reconstruction error depends on the field bandwidth,
HEALPix level, patch shape and `pixel_size_rad`.

For genuinely complex input data, request a complex reconstruction:

```python
reconstructed = transform.ifft(spectrum, real_output=False)
```

## Power-spectrum recommendations

The dominant systematic effect in a local power spectrum is normally the
HEALPix-to-grid interpolation, followed by the finite patch boundary. The
following checks are recommended:

1. remove the mean before computing the spectrum;
2. inspect `coverage_mask` and avoid interpreting zero padding as data;
3. compare several `pixel_size_rad` values;
4. test plane waves of increasing spatial frequency;
5. estimate the projection transfer function with white noise or simulations;
6. compare equivalent signals at the equator and near the poles;
7. use a window for power-spectrum estimation when boundary leakage matters.

A Hann window reduces edge leakage but changes amplitudes and makes the
windowed transform unsuitable for direct map reconstruction:

```python
grid = transform.project(data)
window_1d = torch.hann_window(
    transform.grid_size,
    periodic=False,
    device=transform.device,
    dtype=grid.dtype,
)
window = window_1d[:, None] * window_1d[None, :]
windowed_spectrum = torch.fft.fft2(
    (grid - grid[transform.coverage_mask].mean()) * window,
    norm="ortho",
)
```

Window-energy and mask corrections must be included before comparing absolute
power between patches.

## NumPy, batches, CUDA and autograd

NumPy input produces NumPy output. Torch input produces Torch output on the
transform's device:

```python
# [N]
spectrum = transform.fft(single_map)

# [batch, channel, N]
spectrum = transform.fft(batch)
# -> [batch, channel, grid_size, grid_size]
```

The transform is an `nn.Module`; its cached buffers move normally between
devices:

```python
transform = transform.to("cuda")
```

Gradients propagate to Torch input values:

```python
data = torch.randn(
    4, transform.n_cells,
    device=transform.device,
    requires_grad=True,
)
loss = transform.fft(data).abs().square().mean()
loss.backward()
assert torch.isfinite(data.grad).all()
```

Cell identifiers and projection geometry are discrete precomputed quantities;
gradients are not defined with respect to them.

## Sentinel-2 example

`Notebooks/fft_sentinel2_test.ipynb` loads the real Sentinel-2 arrays produced
by `healpix-compress/test/compress_test.ipynb`:

- B04 red reflectance;
- B08 near-infrared reflectance;
- one complete `1024 x 1024` NESTED tile at level 20 per domain;
- urban, water, forest, agriculture, snow/ice, clouds and ocean scenes.

Set `HEALPIX_COMPRESS_REPO` if the repositories are not siblings:

```bash
export HEALPIX_COMPRESS_REPO=/path/to/healpix-compress
```

The notebook reports reconstruction errors, verifies constant preservation,
displays the tangent-plane projection and compares radial spectra for B04,
B08 and NDVI.

## Errors and limitations

- `cell_ids` must be unique, valid NESTED identifiers at one level.
- A single transform assumes one connected local patch.
- The default 10-degree radius is deliberately conservative.
- Missing cells and irregular patch boundaries appear as a spatial mask and
  affect the spectrum.
- The current IFFT is a fast interpolating inverse, not a least-squares
  reconstruction.
- `float32` is recommended for throughput; `float64` is useful for numerical
  validation.
- The local tangent-plane spectrum should not be interpreted as a full-sky
  spherical $C_\ell$ without an explicit flat-sky-to-spherical calibration.

