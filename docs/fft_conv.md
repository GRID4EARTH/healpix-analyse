# FFT-accelerated convolution on local HEALPix patches

`HealPixFFTConv` applies a learned, potentially very large convolution kernel
to data stored on a local NESTED HEALPix patch. It reuses the pole-safe
gnomonic projection from `LocalFFT` and evaluates the planar convolution with
`torch.fft.rfft2`/`irfft2`.

```text
HEALPix values → gnomonic projection → zero-padded FFT convolution
               → inverse FFT → HEALPix back-projection
```

The geometry is built once. Projection, convolution and back-projection run
on CPU or CUDA and are differentiable with respect to the input and kernel.

## Quick start

```python
import torch
from healpix_analyse import HealPixFFTConv

layer = HealPixFFTConv(
    level=12,
    in_channels=4,
    out_channels=8,
    kernel_sz=65,
    cell_ids=cell_ids,
    ellipsoid="sphere",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

x = torch.randn(2, 4, len(cell_ids), device=layer.device)
y = layer(x)
print(y.shape)  # [2, 8, len(cell_ids)]
```

`cell_ids` must contain unique NESTED identifiers at `level` and must describe
one local patch. The default maximum cell-centre radius is 10 degrees, as for
`LocalFFT`. The implementation is safe at the poles and across the longitude
wrap because the tangent frame is constructed from three-dimensional vectors.

## Interface

```python
HealPixFFTConv(
    level,
    in_channels,
    out_channels,
    kernel_sz=33,
    cell_ids=cell_ids,
    ellipsoid="sphere",
    max_patch_radius_deg=10.0,
    pixel_size_rad=None,
    grid_size=None,
    use_norm=False,
    dtype=torch.float32,
    device=None,
    cache_kernel_fft=True,
)
```

| Parameter | Meaning |
|---|---|
| `level` | Grid4Earth resolution; internally `nside = 2**level` |
| `in_channels`, `out_channels` | Input and output feature channels |
| `kernel_sz` | Positive odd tangent-grid kernel size |
| `cell_ids` | Local NESTED HEALPix domain and data order |
| `ellipsoid` | Geometry used by `healpix_geo` |
| `max_patch_radius_deg` | Gnomonic distortion guard |
| `pixel_size_rad`, `grid_size` | Optional `LocalFFT` grid overrides |
| `use_norm` | Apply GroupNorm and ReLU at the HEALPix output |
| `cache_kernel_fft` | Cache the kernel spectrum during no-gradient inference |
| `projection` | `"average"` (default, keeps constants) or `"conservative"` (keeps the total) |

Accepted input shapes are `[N]`, `[B, N]` for a single input channel, and
`[B, C_in, N]`. The output contains `C_out` channels and follows the same
NumPy/Torch return-type convention as the other operators.

Useful attributes include:

```python
layer.transform.patch_radius_deg
layer.grid_size
layer.fft_shape
layer.weight.shape  # [C_in, C_out, K, K]
layer.cell_ids
```

## Linear convolution rather than circular wrap

A raw FFT product evaluates a periodic convolution. That would incorrectly
connect the west/east and north/south sides of the local tangent grid.
`HealPixFFTConv` instead pads both operands to

$$
N_{\rm linear} = N_{\rm grid} + K - 1
$$

and rounds this size up to the next power of two. After the inverse FFT, the
central `N_grid × N_grid` region is selected. The result therefore matches a
standard 2-D cross-correlation with `padding=K//2` and zeros outside the grid.

For input spectra $X_c$ and flipped kernel spectra $H_{oc}$, the multi-channel
operation is

$$
Y_o(k_x,k_y) = \sum_c X_c(k_x,k_y)H_{oc}(k_x,k_y).
$$

The channel contraction is performed directly in Fourier space. Real FFTs
are used, so only the non-redundant half-spectrum is stored.

## Kernel convention and initialisation

The learned spatial parameter has shape
`[C_in, C_out, kernel_sz, kernel_sz]`, consistent with the channel ordering of
`HealPixConv`. `set_kernel` accepts this shape or the flattened equivalent
`[C_in, C_out, kernel_sz**2]`:

```python
kernel = torch.zeros(1, 1, 65, 65)
kernel[0, 0, 32, 32] = 1
layer.set_kernel(kernel, requires_grad=False)
```

The layer implements the neural-network cross-correlation convention. The
internal spatial flip is handled automatically before transforming the
kernel.

During training, the kernel FFT is recomputed on every forward pass so
autograd can update the spatial weights. During `eval()` under
`torch.no_grad()` or `torch.inference_mode()`, its spectrum is cached and
reused until the weight changes.

## Accuracy and boundaries

The FFT convolution itself agrees with direct `conv2d` to floating-point
precision on the projected grid. The complete HEALPix operator also contains
the approximate interpolation sequence

```text
HEALPix → grid → HEALPix.
```

Consequently, even an identity kernel returns the same approximate round trip
as `LocalFFT.project` followed by `LocalFFT.unproject`, rather than an exact
copy of arbitrary HEALPix data.

Uncovered grid points and values beyond the supplied patch are zero. For
scientific output near a boundary, supply a halo at least half the kernel
width and crop the halo after convolution. A smooth apodization can also
reduce edge discontinuities.

## Constants, totals and the blur floor

The HEALPix → grid → HEALPix pair has two conventions, chosen with
`projection=` (also accepted by `LocalFFT`). A convolution cannot keep both a
constant and the total at the edge of the domain, and in the interior it keeps
only one of them.

| `projection=` | Scatter | Back-projection | Constant field | Total (sum over cells) |
|---|---|---|---|---|
| `"average"` (default) | divided by the accumulated weight | bilinear gather | kept | **not kept** |
| `"conservative"` | plain bilinear scatter | bilinear gather of the grid divided by the accumulated weight | **not kept** | kept |

- With `"average"` a uniform field stays uniform, but the response to a single
  cell does not sum to one. Measured at level 10 on an 8125-cell domain with a
  Gaussian kernel of 1 to 3 pixels, the sum of the response of an interior cell
  lies between 0.79 and 1.06, and up to 1.4 at the domain edge.
- With `"conservative"` and a kernel whose weights sum to one, the response of
  a cell with at least `5 sigma + 2` pixels of margin sums to 1 to better than
  1e-8 (float64). Mass that the kernel moves outside the covered grid points is
  lost, not reflected: the sum of the response of an edge cell is below one
  (0.39 to 0.98 for edge cells at 1 to 3 pixels in the same domain), never above. Use this
  option for densities and probability maps; a caller that renormalises
  afterwards should not need to.
- The two steps of either convention are not transposes of one another. What
  holds for `"conservative"` is that the round trip (scatter, then divide and
  gather) is a symmetric matrix whose column sums are one at every cell. The
  round trip is the same matrix for both conventions; they differ in where the
  convolution acts (on the plain scatter, or on its average).
- Without a kernel (identity), the conservative pair keeps both the total and a
  constant; with a kernel, a constant is changed by as much as the total is
  changed with `"average"` (0.79 to 1.06 inside the same domain).
- The total is kept by the projection pair and the convolution only. A
  non-zero `bias` (added after the back-projection) or `use_norm=True` breaks
  it; `set_gaussian` sets the bias to zero. Grid points inside the patch that
  no cell covers (a domain with holes, such as land masked out of an ocean
  domain) take mass like the grid points outside the domain, and it is lost.

The default results are unchanged. `set_gaussian` sets a normalised Gaussian
kernel on every channel pair `c → c`:

```python
layer = HealPixFFTConv(level, 1, 1, kernel_sz=21, cell_ids=cell_ids,
                       projection="conservative")
layer.set_gaussian(2.0)                  # pixels of the projected grid
layer.set_gaussian(0.002, unit="rad")    # or radians in the tangent plane
```

`set_gaussian` fills only the channel pairs `c → c` (other pairs and the bias
are zero). `unit="rad"` divides by `pixel_size_rad`, the pixel size of the
gnomonic plane, which equals the angular distance on the sphere at the patch
centre only. The kernel is normalised after truncation to `kernel_sz`; when
`kernel_sz // 2` is under 3 sigma it warns (`on_truncated="error"` or
`"ignore"` to change that), because for a large sigma the truncated kernel is
nearly a flat box. Use `kernel_sz >= 2 * ceil(5 * sigma) + 1`.

**Blur floor.** Projection followed by back-projection spreads a single cell
over its neighbours, even with an identity kernel. The nominal standard
deviation per axis of this response is $\sqrt{1/3} = 0.577$ pixel
(`PROJECTION_BLUR_FLOOR_PX` in `healpix_analyse.fft_local`), which is
`0.00058` rad at level 10. It is exactly the same for both conventions, because
the round trip is the same matrix. It is not the same for every cell: on the
level-10, 8125-cell domain of the benchmark it ranges from 0.23 to 0.69 pixel
over the cells (median 0.569), with the position of the cell centre inside its
grid pixel; and it is not isotropic: along the two axes of the gnomonic plane
the medians are 0.568 and 0.591 pixel. The widths add in quadrature,
$\sqrt{\sigma^2 + 0.58^2}$ (to 0.03 pixel from 0.5 pixel upwards), so
Gaussian widths below the floor cannot be represented; at 0.25 pixel the
effective width is 0.57 pixel. `layer.blur_floor_px()` returns the exact value
for each cell (`per_axis=True` for the two axes). `set_gaussian` (or
`layer.transform.check_width(sigma_px)`) warns when the requested width is below
the **median** floor, recomputed at each call, so a cell whose own floor is
larger is not flagged; the warning points at the caller's line. Pass
`on_below="error"` to raise instead, or `"ignore"`. For such small widths use
`gaussian_filter(..., width_rule="matched")` ({doc}`radial_filter`), which
reaches widths below one cell; with `normalization="full_disc"` it also keeps
the total and loses mass leaving the domain, as `projection="conservative"`
does here.

`benchmarks/benchmark_fft_total.py` reproduces these numbers.

## Performance

For a projected grid of side $N$ and kernel side $K$:

- direct convolution costs approximately $O(C_{in}C_{out}N^2K^2)$;
- FFT convolution costs approximately
  $O((C_{in}+C_{out})M^2\log M + C_{in}C_{out}M^2)$,
  where $M$ is the padded FFT side.

Small kernels such as `3×3` or `5×5` are normally faster with `HealPixConv` or
`torch.conv2d`. FFT convolution becomes attractive for large kernels,
especially for batches and repeated inference on a GPU.

## `HealPixFFTConv` or `LargeConv`?

| Property | `HealPixFFTConv` | `LargeConv` |
|---|---|---|
| Domain | One local gnomonic patch | Full sphere or partial NESTED patch |
| Large-kernel representation | Full learned planar kernel | Compact learned multiresolution kernel |
| High-frequency kernel detail | Retained on projected grid | Reduced by Down operations |
| Main approximation | Projection/back-projection | Multiresolution subspace |
| Best use | Very large kernels on local imagery | Wide spherical context and memory control |

`HealPixFFTConv` has one global tangent-frame orientation for the patch. It is
not a gauge-equivariant spherical convolution. Use `HealPixConv` when gauge
equivariance is required.
