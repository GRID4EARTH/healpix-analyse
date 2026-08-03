# Local multiscale HEALPix decomposition

`HealPixDecomp` constructs an exactly reconstructing, wavelet-like Laplacian
pyramid for full-sky or masked NESTED HEALPix maps. It is designed for local
geophysical data such as sea-surface temperature or ocean velocity, where
land and coastline masks make global Fourier decompositions unsuitable.

The implementation uses only local sparse operators:

```text
analysis:  current ── Down ──> coarse
               └── current - Up(coarse) ──> detail

synthesis: detail + Up(coarse) ──> current
```

Every Up operator reuses the exact sparse matrix and cell domain of its paired
Down operator. Cell identifiers are retained explicitly at every scale.

## Quick start

```python
import torch
from healpix_analyse import HealPixDecomp

decomp = HealPixDecomp(
    level=10,
    cell_ids=ocean_cell_ids,
    Jmax=5,
    ellipsoid="sphere",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# u_v may have shape [2, N] or [B, 2, N]
pyramid = decomp.compute(u_v)
reconstructed = decomp.invert(pyramid)

print(pyramid.levels)       # (10, 9, 8, 7, 6, 5)
print([x.shape for x in pyramid])
print(torch.max(torch.abs(reconstructed - u_v)))
```

`Jmax=5` means five Down operations and therefore six returned images: five
detail maps followed by the final coarse map. `Jmax=-1` uses every available
scale down to level zero. `Jmax=0` returns one image and acts as the identity.

## Analysis and exact synthesis

Let $x_j$ be the low-pass map at level $L-j$, $D_j$ the local Gaussian Down
operator and $U_j$ its exactly paired Up operator. Analysis computes

$$
x_{j+1}=D_jx_j,
\qquad
w_j=x_j-U_jx_{j+1}.
$$

The returned sequence is

$$
(w_0,w_1,\ldots,w_{J-1},x_J).
$$

Each successive image is stored on a smaller HEALPix domain. Synthesis uses

$$
x_j=w_j+U_jx_{j+1}
$$

from coarse to fine. Because $w_j$ was defined using the same stored $U_j$,
this is an algebraic identity independent of whether $U_jD_j$ is itself a
perfect inverse. Reconstruction is therefore exact up to floating-point
round-off.

For an explicit sum at the finest resolution:

```python
components = decomp.expand(pyramid)
direct_sum = torch.stack(components).sum(dim=0)
```

`expand` applies all required Up operators to every band separately. The
elementwise sum of these fine-resolution components equals `invert(pyramid)`.

## The returned `HealPixPyramid`

`compute` returns a list-like `HealPixPyramid`. It can be indexed and iterated
like a tuple while retaining the geometry needed for safe reconstruction:

| Attribute | Description |
|---|---|
| `bands`, `images` | Detail bands followed by the final coarse image |
| `details` | All detail bands (`bands[:-1]`) |
| `coarse` | Final low-pass image (`bands[-1]`) |
| `levels[j]` | Grid4Earth/HEALPix level of band `j` |
| `cell_ids[j]` | NESTED identifiers matching the last dimension of band `j` |

The decomposition validates this metadata during inversion. Passing a
pyramid generated from another mask, resolution or operator raises an error
instead of silently mixing incompatible cells.

Input data may have any leading dimensions as long as the final dimension is
the HEALPix pixel axis. This includes:

- `[N]` for one scalar map;
- `[2, N]` for a velocity pair `(u, v)`;
- `[B, C, N]` for batched multi-channel simulations.

NumPy input produces NumPy bands and NumPy reconstruction. Torch input stays
on the selected device and retains autograd connectivity.

## Cell ordering and masked domains

For partial maps, `cell_ids` may be in any order. `HealPixDecomp`:

1. records the original input order;
2. sorts the fine identifiers into a canonical internal order;
3. computes the unique NESTED parents at every coarser level;
4. constructs each Up directly from its paired Down matrix;
5. stores the exact identifiers for every band;
6. restores the original fine order after `invert` and `expand`.

This is essential for irregular ocean masks. A coarse coastal cell may have
only a subset of its fine children present. The Down filter uses only
available cells and renormalises its weights locally; synthesis returns
exactly to that same available fine-cell set. No artificial land values are
introduced.

## Filtering

The analysis filter is the angularly symmetric Gaussian used by
`HealPixDown(mode="smooth")`. With omitted `radius_deg` and `sigma_deg`, each
level selects defaults proportional to its own HEALPix pixel size. A fixed
angular filter can instead be supplied explicitly.

The default combination is:

```python
weight_norm="l1"
up_norm="col_l1"
```

It preserves constant fields locally and normally produces near-zero detail
bands for a constant unmasked or masked map. Other supported combinations
are `up_norm="adjoint"` and the energy-oriented
`weight_norm="l2", up_norm="diag_l2"`.

Changing the normalisation changes individual detail maps but never the
reconstruction identity, because the same prediction is subtracted during
analysis and added during synthesis.

## Interpretation and limitations

This is a Laplacian pyramid with wavelet-like detail bands. It is not claimed
to be an orthogonal or tight wavelet basis:

- detail bands are generally correlated;
- their squared energies do not necessarily sum to input energy;
- masks and local renormalisation make the filters spatially varying near
  coastlines;
- a coefficient at a given scale must always be interpreted with its stored
  `cell_ids` and level.

These properties are intentional for masked climate fields: locality and
exact synthesis are prioritised over global spectral orthogonality.

`(u, v)` are decomposed as two ordinary channels. The separate
`HealPixMultiScaleDivCurl` operator applies fixed gauge-aware `HealPixConv`
derivative kernels to every band, using the physical scale and exact cell
geometry stored by this pyramid. See [Multiscale divergence and
curl](divcurl.md).

## API

```python
HealPixDecomp(
    level,
    cell_ids=None,
    Jmax=-1,
    ellipsoid="WGS84",
    radius_deg=None,
    sigma_deg=None,
    weight_norm="l1",
    up_norm="col_l1",
    dtype=torch.float32,
    device=None,
)
```

Main methods:

```python
pyramid = decomp.compute(data)
pyramid = decomp(data)             # same operation through nn.Module.forward
data = decomp.invert(pyramid)
fine_components = decomp.expand(pyramid)
```
