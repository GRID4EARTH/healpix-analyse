# Attention layers — windows and neighbourhoods

Two local self-attention layers for transformer models on HEALPix grids, on
the full sphere or on a regional domain:

- `HealPixWindowAttention`: full attention inside NESTED blocks of
  `4**window_level` cells (Swin-like windows);
- `HealPixNeighbourhoodAttention`: each cell attends to the taps of its
  gauge-oriented stencil, the stencil of {doc}`convol` (the HEALPix analogue
  of neighbourhood attention).

Global attention on patch tokens (ViT, MAE) needs no layer of its own: it is
standard attention applied to the patch tokens.

Both layers take `[B, C, N]` tensors (batch, channels, cells in the order of
`cell_ids`, NESTED on the full sphere), project them to queries, keys and
values with `num_heads` heads, and project the attention output back to `C`
channels.

## Window attention

```{include} ../_generated/glance/window_attention.md
```

The queries of a window are the `4**w` descendants of one cell of level
`level - w`. In NESTED order they are contiguous: on the full sphere, and on
a regional domain whose `cell_ids` are sorted and made of complete blocks,
the query windows need no index. Any other domain (unsorted, or with
incomplete blocks) pads its windows to a common length and masks the
padding.

On CUDA (float32, float16, bfloat16, and under `torch.autocast`) fused
Triton kernels in the style of flash attention run every kind of window:

- they read the packed output of the `qkv` projection in place and write
  its gradient in place, so no permuted copy of q, k, v is made;
- the rows of padded or overlapping windows are gathered in registers and
  the padding is masked there, with no gathered copy and no mask tensor;
- q kᵀ and p v run on tensor cores (`tl.dot`), with fp32 accumulation and
  an online softmax; the tile sizes are autotuned;
- the backward is deterministic: one kernel for the query gradient, one for
  the key and value gradients of each block of keys. When windows overlap
  (`"coarse_centred"`), each window slot writes a partial sum, and a third
  kernel sums the slots of each cell in a fixed order. No atomics.

Elsewhere (CPU, float64, attention dropout, `backend="torch"`), the windows
are gathered and passed to `torch.nn.functional.scaled_dot_product_attention`.

```python
import torch
from healpix_analyse import HealPixWindowAttention

layers = torch.nn.Sequential(*[
    HealPixWindowAttention(level=8, embed_dim=128, num_heads=4, window_level=3)
    for i in range(4)
])
y = layers(torch.randn(2, 128, 12 * 4**8))      # [2, 128, N]
```

### Replacing Swin's shifted windows

Swin alternates its window grid with one shifted by half a window, so that
information crosses the window borders. HEALPix blocks have no such shift.
The option `shift` selects the replacement. The maintainer chose the windows
centred on the cells of a coarser level, which overlap across block borders
(issue #109): `"coarse_centred"` is the default, and the two other options
stay available.

| `shift` | windows | cost | receptive field |
|---|---|---|---|
| `"coarse_centred"` (default) | queries: a block of `4**w` cells; keys: the cells within `(1 + halo) r_c` of the block centre, `r_c` the distance to the farthest cell centre of the block plus one cell spacing | keys per window, in block sizes, at `halo=0.25`: 3.8 (w = 1), 3.7 (2), 3.5 (3), 3.4 (4); at `halo=0`: 3.1, 2.3, 2.2, 2.2 | grows at every layer, across block borders |
| `"none"` | blocks of `4**w` cells at every layer | the cheapest | one block, whatever the depth |
| `"alternate_level"` | `shifted=False`: blocks of `4**w`; `shifted=True`: blocks of `4**(w+1)` | the shifted layers attend over 4× longer windows (4× the cost) | one block of `4**(w+1)` cells after 2 layers; it grows further only through Down/Up, as in the hierarchical stages of Swin |

`shift=None` takes the module constant `DEFAULT_WINDOW_SHIFT`
(`"coarse_centred"`), so the default can be changed in one place.
`benchmarks/benchmark_attention.py receptive` prints the receptive field of
one cell after 1 to 6 layers for each option. At level 8 with windows of 64
cells (cell size 0.23°), as cells / radius in cell sizes:

| `shift` | 1 layer | 2 layers | 3 layers | 4 layers | 6 layers |
|---|---|---|---|---|---|
| `"coarse_centred"` | 210 / 8.8 | 974 / 20.9 | 2265 / 34.1 | 4142 / 49.6 | 9761 / 77.1 |
| `"alternate_level"` | 64 / 6.3 | 256 / 17.8 | 256 / 17.8 | 256 / 17.8 | 256 / 17.8 |
| `"none"` | 64 / 6.3 | 64 / 6.3 | 64 / 6.3 | 64 / 6.3 | 64 / 6.3 |

(for a cell next to the centre of its block; layers alternate `shifted=False`
and `True`).

## Neighbourhood attention

```{include} ../_generated/glance/neighbourhood_attention.md
```

Cell `n` attends to the `P = kernel_sz**2` taps of the stencil of
`HealPixConv` around it. The key and the value of a tap are the bilinear
interpolations of the keys and values of the 4 cells around the tap point,
with the weights of the convolution:

```text
s[h, g, n, p] = <q[h, n], k_p[h, g, n]> / sqrt(D) + bias[h, p]
out[g, n]     = proj( sum_p softmax_p(s) v_p[g, n] )
```

The stencil is oriented by the gauge, so one relative position bias table
`[num_heads, P]` serves the whole sphere. With `n_gauges = G`, the stencil
turns by `pi / G` from one gauge to the next and the output stacks the G
gauges, `[B, G*C, N]`, as `HealPixConv` does. The gauge options
(`gauge_type`, `singularity_lonlat`, `ref_direction`) are those of
`HealPixConv`, and the geometry shares its disk cache.

On a regional domain, a tap whose 4 cells are all outside the domain is left
out of the softmax (`outside="mask"`, the default) or reads the centre cell
(`outside="centre"`, the rule of `HealPixConv`). A tap with some cells inside
uses the renormalised weights of those cells.

```python
from healpix_analyse import HealPixNeighbourhoodAttention

na = HealPixNeighbourhoodAttention(level=8, embed_dim=128, num_heads=4, kernel_sz=3)
y = na(torch.randn(2, 128, 12 * 4**8))          # [2, 128, N]
```

### Backends

`backend="auto"` runs fused Triton kernels on CUDA for float32, float16 and
bfloat16 (and under `torch.autocast`) when the optional `triton` package is
installed, and a PyTorch loop over the taps otherwise (CPU, float64).

- **Forward (Triton):** one program per block of cells, batch, gauge and head
  gathers the 4 cells of each tap in registers, then runs an online softmax
  over the taps. The gathered keys and values `[B, H, G, N, P, D]` are never
  stored; the layer keeps the output and one log-sum-exp per cell and head.
- **Backward (Triton), deterministic:** a first kernel recomputes the scores
  and writes the query gradient, the attention weights and the score
  gradients (`P` values per cell, head and gauge). A second kernel gathers
  the key and value gradients of each cell from the transposed stencil map,
  in a fixed order: no atomics, the same bits at every run. The map is built
  at the first backward pass and kept (about the size of the geometry).
- **PyTorch:** one tap at a time, with a hand-written backward. On CUDA its
  `index_add_` is deterministic only under
  `torch.use_deterministic_algorithms(True)`.
- A double backward (`create_graph=True`) runs the PyTorch backward on every
  device.

`neighbourhood_attention_naive` in `healpix_analyse._attention_backend`
materialises the gathered keys and values; it is the reference of the tests
and the baseline of the benchmark.

## Cost

- Window attention: `N * 4**w` scores per head with `"none"`; with
  `"coarse_centred"` 3.4–3.8 times more at `halo=0.25` and 2.2–3.1 times
  more at `halo=0` (see the table above), memory linear in `N`. The Triton backward
  of overlapping windows keeps one fp32 partial sum per key slot (twice the
  size of k and v per slot). On the PyTorch path, the backward of the
  gathers accumulates with `index_put_`, whose order on CUDA is fixed only
  under `torch.use_deterministic_algorithms(True)`.
- Neighbourhood attention: `N * P` scores per head and gauge, linear in `N`;
  each score reads 4 rows of keys and 4 of values. The geometry is the one of
  a `HealPixConv` with the same arguments.
- `benchmarks/benchmark_attention.py` measures both layers against dense
  attention and against the naive gather.
