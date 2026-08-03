# Large receptive-field convolution

`LargeConv` provides large effective convolution kernels without constructing
the expensive fine-resolution interpolation stencil required by a direct
`HealPixConv`.

It applies a matched multiresolution operator:

$$
y = U_1 U_2 \cdots U_L\; C_{W_c}\;D_L\cdots D_2D_1 x + b,
$$

where:

- $D_i$ is a smooth `HealPixDown`;
- $C_{W_c}$ is a compact trainable `HealPixConv`;
- $U_i$ is the `HealPixUp` constructed from the same smoothing parameters;
- $b$ is added after returning to the fine resolution.

The compact kernel is learned end-to-end through the complete chain. Its
effective fine-resolution kernel therefore includes the exact Down/Up
operators used at inference.

## Quick start

```python
import torch
from healpix_analyse import LargeConv

layer = LargeConv(
    level=8,  # nside = 256
    in_channels=8,
    out_channels=16,
    kernel_sz=33,
    max_compact_kernel_sz=7,
    cell_ids=cell_ids,
    ellipsoid="sphere",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

x = torch.randn(4, 8, len(cell_ids), device=layer.device)
y = layer(x)
print(y.shape)  # [4, 16, len(cell_ids)] for one gauge
```

## Selecting the resolution and compact kernel

Kernel size is converted using its radius rather than its diameter. For a
requested fine kernel $K_f$:

$$
r_f = \frac{K_f-1}{2}, \qquad
r_c(L) = \left\lceil\frac{r_f}{2^L}\right\rceil, \qquad
K_c(L)=2r_c(L)+1.
$$

The smallest $L$ satisfying

$$K_c(L) \leq K_{\max}$$

is selected automatically.

For `kernel_sz=33` and `max_compact_kernel_sz=7`:

| Down levels | Coarse radius | Compact kernel | Accepted? |
|---:|---:|---:|---:|
| 0 | 16 | 33 | no |
| 1 | 8 | 17 | no |
| 2 | 4 | 9 | no |
| 3 | 2 | 5 | yes |

Therefore a requested `33×33` receptive field uses three Down operations, a
`5×5` compact convolution, and three Up operations. A `7×7` compact kernel at
that level would have a nominal fine diameter of `49×49`.

The selected plan is exposed through:

```python
print(layer.n_levels)                 # 3
print(layer.compact_kernel_sz)        # 5
print(layer.coarse_nside)             # nside // 8
print(layer.effective_kernel_sz)      # 33
```

When the requested kernel already fits within the configured maximum, no
resolution change is performed and `LargeConv` reduces to one `HealPixConv`
plus its fine-resolution bias and optional normalisation.

## Why the compact kernel is learned directly

Naively resizing a `33×33` array to `5×5` would ignore:

- the smoothing transfer function of `HealPixDown`;
- the transpose and normalisation used by `HealPixUp`;
- HEALPix geometry and bilinear stencil interpolation;
- gauge orientation;
- partial-patch boundaries.

Learning $W_c$ through the whole chain optimises the actual operator
$U^LC_{W_c}D^L$. Gradients pass through every sparse matrix multiplication,
the compact convolution, and each interpolation operation.

The representation is intentionally lower rank than an arbitrary direct
large kernel. Downsampling removes high-frequency information, so an
oscillatory fine kernel cannot in general be reproduced exactly. Large smooth
kernels and learned contextual filters are the intended use cases.

## Matched Down and Up operators

Every pair receives the same:

- `ellipsoid`;
- `radius_deg`;
- `sigma_deg`;
- `weight_norm`;
- floating-point dtype and device.

In addition, every `HealPixUp` is constructed with `paired_down=down_layer`.
It transposes the already stored sparse Down matrix instead of rebuilding an
operator on a potentially different set of fine children. The pairing is
therefore exact even for irregular partial patches.

Only `HealPixDown(mode="smooth")` is used. Max pooling is nonlinear and cannot
define a linear effective convolution kernel.

`up_norm="col_l1"` with `weight_norm="l1"` is the default combination. It
preserves constant fields and is a suitable default for learned features.

```python
layer = LargeConv(
    level=7,
    in_channels=4,
    out_channels=8,
    kernel_sz=65,
    weight_norm="l1",
    up_norm="col_l1",
)
```

Other supported combinations are inherited from `HealPixDown` and
`HealPixUp`:

- `up_norm="adjoint"` gives the raw transpose of Down;
- `weight_norm="l2", up_norm="diag_l2"` favours local energy consistency.

## Bias, normalisation and activation

The internal compact convolution has a fixed zero bias. `LargeConv.bias` is
added only after the final Up operation, so it represents a true
fine-resolution bias even when partial-patch boundary normalisation is not
uniform.

When `use_norm=True`, GroupNorm and ReLU are also applied at the final fine
resolution:

```text
Down × L → compact convolution → Up × L → bias → GroupNorm → ReLU
```

Applying these nonlinear operations at the coarse level would define a
different operator and would prevent interpretation as one effective large
convolution.

## Input and output shapes

| Input | Interpretation | Output |
|---|---|---|
| `[N]` | one single-channel map | `[N]` when total output channels = 1, otherwise `[C_out, N]` |
| `[B, N]` | batch of single-channel maps | `[B, G*C_out, N]` |
| `[B, C_in, N]` | general batch | `[B, G*C_out, N]` |

NumPy input returns NumPy output. Torch input returns Torch output and retains
autograd connectivity.

## Partial HEALPix patches

`cell_ids` must be unique NESTED identifiers at the fine level. `LargeConv`:

1. sorts identifiers internally to match the sparse Down matrices;
2. builds the parent cell set at every coarse level;
3. builds each Up operator from the exact corresponding parent set;
4. crops extra children produced by partial-patch upsampling;
5. restores the original `cell_ids` order before returning.

```python
layer = LargeConv(
    level=6,
    in_channels=1,
    out_channels=2,
    kernel_sz=33,
    cell_ids=cell_ids,
)
```

As with every local convolution, values close to the patch boundary have less
context. Supply a halo around the requested output region and crop the final
result when accurate boundary responses are important.

## Gauges

The internal convolution supports the same gauge options as `HealPixConv`:

- `"phi"`;
- `"cosmo"`;
- `"projected_ref"`;
- `"two_ref"`.

`n_gauges`, `singularity_lonlat` and `ref_direction` are forwarded unchanged.
The output contains `n_gauges * out_channels` channels.

## Kernel access and initialisation

The compact trainable parameter is available as `layer.weight` and has shape

```text
[n_gauges, in_channels, out_channels, compact_kernel_sz**2]
```

A fixed or custom initial compact kernel can be installed with:

```python
layer.set_compact_kernel(weight, bias=bias, requires_grad=True)
```

The provided array describes the compact coarse kernel, not a fine `33×33`
kernel. Fitting a compact kernel from a pre-existing large fine kernel requires
an explicit operator-matching optimisation and is outside the current API.

## Computational benefit

A direct convolution stores and gathers values proportional to

$$K_{\text{fine pixels}}\,K_f^2.$$

After $L$ levels, the dominant compact convolution scales approximately as

$$\frac{K_{\text{fine pixels}}}{4^L}\,K_c^2,$$

plus sparse Down/Up operations. For `33×33 → 5×5` with three levels, the
convolution gather term is reduced nominally by

$$
4^3\frac{33^2}{5^2} \approx 2788.
$$

Actual speedup depends on batch size, channels, patch shape, device and the
sparse-matrix overhead.

## Limitations

- The effective kernel is approximate in the sense of representational
  bandwidth; it is exactly the operator learned through the selected chain.
- `nside` must remain at least one after all selected downsamplings.
- Only NESTED ordering is currently supported.
- Geometry construction can be expensive; the internal `HealPixConv` cache is
  enabled by default.
- Full-sphere operation at very high `nside` still requires substantial memory
  for Down/Up sparse matrices.
- Partial patches should include a halo for reliable edge values.

See `Notebooks/large_conv_test.ipynb` for configuration, constant-field,
ordering, gradient, receptive-field and training tests.
