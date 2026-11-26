# HealPixPatches — patch tokens and their fold

`HealPixPatches` cuts a field stored on HEALPix cells into square patches,
the tokens of ViT, MAE, JEPA and other token-based models, and folds patches
back onto the cells (the adjoint, for MAE decoders and dense heads). It is a
PyTorch `nn.Module`, differentiable, and runs on the full sphere or on a
regional domain.

```{include} ../_generated/glance/patches.md
```

---

## Quick start

```python
import torch
from healpix_analyse import HealPixPatches

level = 10                               # data at level 10 (12.6 M cells)
patches = HealPixPatches(level, patch_size=16)   # 16 x 16 tokens, centres at level 6

x = torch.randn(4, 3, 12 * 4**level)     # [B, C, K]
y = patches(x)                           # [4, 49152, 1, 3, 16, 16]  = [B, M, G, C, N, N]
tokens = y.flatten(2)                    # [4, 49152, 768], a free view

# MAE: keep 25 % of the tokens, a different draw per sample
keep = torch.rand(4, patches.M).argsort(1)[:, : patches.M // 4]
visible = patches(x, index=keep)         # gathers only those patches

# dense head: back to the cells
x_rec = patches.fold(y)                  # [4, 3, K]
```

The centre of patch `m` is the NESTED cell `patches.centre_ids[m]` at level
`patches.centre_level`.

## Patches, centres and modes

Patch sizes are powers of 2, `N = 2**p`. The patches are centred:

- `centres="coarse"` (default): on the cells of level `L − p`, one patch per
  coarse cell. On a regional domain, the coarse cells are the parents of the
  domain cells. This is the ViT tiling, aligned with NESTED.
- `centres="all"`: on every cell of the domain, in the order of `cell_ids`
  (overlapping patches, `mode="gauge"` only).

Two modes give the pixels of a patch:

`mode="gauge"` (default)
: An `N × N` grid in the tangent plane of the centre, oriented by the gauge
  and sampled by bilinear interpolation, exactly like a
  {doc}`HealPixConv <convol>` kernel: the same rotation matrices, the same
  gauge types (`phi`, `cosmo`, `projected_ref`, `two_ref`) and `n_gauges`
  orientations `g·π/G`, the spacing of one cell of level `L`. The grid
  offsets are `i − (N − 1)/2`, so an even `N` is centred on the centre of the
  cell. A shared patch embedding sees the same orientation everywhere on the
  sphere, and with `G = 4` gauge `g + 2` is gauge `g` rotated by 90° (a
  `torch.rot90` of the patch, `k=1`). With `gauge_type="cosmo"` the gauge
  angle changes sign across the equator, so at southern centres the rotation
  goes the other way: gauge `g + 2` is `torch.rot90(..., k=3)` of gauge `g`.
  The grid matches the stencil of `HealPixConv(ellipsoid="sphere")`; with
  the default `ellipsoid="WGS84"`, `HealPixConv` places its stencil at
  geodetic latitudes, shifted from the cell centres of the grid, while
  `HealPixPatches` always uses the sphere of the grid.

`mode="nested"`
: The `4**p` descendants of each coarse cell, as a `2**p × 2**p` image: row
  `N − 1 − y`, column `x` in the face coordinates of HEALPix. Face-oriented
  rather than gauge-oriented: the cheap option. On the full sphere, or on a
  sorted domain made of complete coarse cells, the extraction is a view of
  the data plus one gather, and `patches(x, zorder=True)` returns the
  zero-copy view `[B, M, 1, C, 4**p]`, the pixels in NESTED order (a linear
  patch embedding does not depend on the order of the pixels).

In the equatorial belt a coarse cell is close to a square tilted by 45°: the
NESTED tile is then close to the gauge patch at 45° (`n_gauges=4`, `g = 1`),
rotated and mirrored. Near the poles, and along the boundaries of the base
cells, the two differ more.

### How this differs from `nested_to_tiles`

{func}`~healpix_analyse.dino.nested_to_tiles` (the DINOv3 pipeline) and
`mode="nested"` cut the same tiles in the same layout. `nested_to_tiles` works
on NumPy arrays, fills the missing pixels with a mean and returns a validity
mask; `HealPixPatches` keeps the data on the device, is differentiable, has
the fold and the masked sampling, and returns 0 with a `coverage` of 0 for
the missing pixels.

Both are face-oriented: the axes of a tile follow the base cell of HEALPix
it belongs to, so the orientation of the content changes from one base cell
to the next (and by a quarter turn or more across the poles). A network that
shares one patch embedding then sees rotated inputs depending on where the
patch lies. `mode="gauge"` removes this: the orientation follows the gauge,
which is continuous except at its two singular points.

## Output layout

`patches(x)` returns `[B, M, G, C, N, N]` for an input `[B, C, K]`
(`[B, M, G, 1, N, N]` for `[B, K]`, `[M, G, 1, N, N]` for `[K]`; NumPy in,
NumPy out). The output is contiguous, so `y.flatten(2)` (one token of
`G·C·N²` values per centre) and `y.flatten(3)` (`G` tokens per centre) are
free views.

## Regional domains and missing cells

`cell_ids` gives the NESTED cells of the domain at level `L`, in the order of
the data, in any order. In `mode="gauge"`, a grid point whose 4 bilinear
neighbours are not all in the domain uses the present ones, renormalised; a
point with none of them is 0. `patches.coverage` (`[M, G, N, N]`) holds the
fraction of the bilinear weight inside the domain: 1 inside, between 0 and 1
at the edge. In `mode="nested"`, descendants absent from the domain are 0,
with coverage 0. NaN values are not handled and propagate: replace them, and
use the coverage or your own mask.

## Masked token sampling

`patches(x, index=...)` gathers only the patches listed: positions in
`centre_ids`, shape `[Ms]` (the same subset for every sample) or `[B, Ms]` (one
draw per sample, as an MAE draws them). The output is `[B, Ms, G, C, N, N]`;
nothing is computed for the other patches. `fold(y, index=...)` takes the same
index.

## Fold

`patches.fold(y, index=None, norm=None)` scatter-adds the patches back to the
cells, `[B, Ms, G, C, N, N] → [B, C, K]`. The normalisation is an option
(`fold_norm` at construction, or `norm=` per call):

- `"weight"` (default): each cell is divided by the sum of the weights with
  which the patches read it, so a constant field comes back unchanged;
- `"count"`: by the number of patch samples that read it (the overlap count);
- `"none"`: the plain adjoint, `⟨patches(x), y⟩ = ⟨x, fold(y)⟩`.

With `index`, the normalisation counts only the patches given, per sample.
A cell that no patch reads is 0. With `mode="nested"` on a complete domain
every cell is read once, and the three agree: `fold` inverts the extraction.

With `centres="coarse"` and `mode="gauge"` the square grids do not tile the
coarse cells exactly: a coarse cell is a curved diamond, the grid a square
aligned with the gauge. With `N = 4`, `fold` reaches every cell except up to
16 cells next to the poles; with `N = 8` at level 6, 1952 of the 49 152
cells (the tips of the diamonds) are read by no patch. Use `centres="all"`
or `mode="nested"` when every cell must be reconstructed.

## Backends and cost

The geometry is a sparse map stored once: `J` int32 positions and `J`
weights per patch pixel (`J = 4` in `mode="gauge"`, 1 in `mode="nested"`),
that is `32·M·G·N²` bytes in `mode="gauge"` (`M·N² = K` for
`centres="coarse"`, `K·N²` for `centres="all"`). It is built at construction
(not cached on disk).

`backend="auto"` runs fused Triton kernels on CUDA for float32, float16 and
bfloat16 when the optional `triton` package is installed (16-bit data
accumulate in fp32), and PyTorch otherwise (CPU, float64). The extraction is
one gather kernel; the fold, which is also the backward pass of the
extraction, is a gather on the transposed map (for each cell, the patch
pixels that read it), built at the first fold or backward pass and kept: no
atomics, so the gradients are the same at every run. The backward of the
fold is an extraction, and both are differentiable again
(`create_graph=True`). The PyTorch backend uses `index_select` and
`index_add_`; on CUDA, `index_add_` is deterministic only under
`torch.use_deterministic_algorithms(True)`.

With `index`, the Triton fold reads the selected patches through an inverse
map (the position of each centre in the selection, or −1), so nothing is
written for the patches left out. It visits only the cells that the selected
patches read: at construction, each centre stores the list of cells its patch
reads (its footprint), and each call marks the cells of the selected
footprints. Its cost follows the number of selected patches, and the result
is the same, bit for bit, as a walk over the whole transposed map. With
overlapping patches (`centres="all"`) every cell is read by many patches, so
there is little to skip. The PyTorch backend writes the selected patches into
a zero tensor of all `M` patches, then folds it. A normalised fold needs distinct centres in each row of
`index`.

`benchmarks/benchmark_patches.py` measures the extraction, its backward and
the fold against a plain gather of the same size.

## Reference

Parameters and shapes: the docstrings of
{py:class}`~healpix_analyse.patches.HealPixPatches`,
{py:meth}`~healpix_analyse.patches.HealPixPatches.forward` and
{py:meth}`~healpix_analyse.patches.HealPixPatches.fold`.
