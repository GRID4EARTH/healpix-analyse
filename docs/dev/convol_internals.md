# HealPixConv — how the geometry is built

What happens between the constructor and the forward pass, and the private
helpers that do it. You do not need this page to use the layer or to look up an argument
({doc}`/conv/convol`) -- it is here
for anyone modifying the operator or checking its correctness.

---

## The three stages

The layer operates in three stages that are all precomputed at construction
time. The forward pass reduces to a gather and one small matrix product per
stencil tap.

### Stage A — kernel definition and rotation

A `kernel_sz × kernel_sz` grid of `P = kernel_sz²` unit vectors is placed
at the **North Pole** (`z = 1`) with angular spacing equal to one HEALPix
pixel width:

```
alpha_pix = sqrt(4π / (12 · nside²))   ← angular size of one pixel

stencil point (i, j):
    dtheta = sqrt(i² + j²) · alpha_pix
    dphi   = atan2(j, i)
    vec    = [sin(dtheta)·cos(dphi),
              sin(dtheta)·sin(dphi),
              cos(dtheta)]
```

This stencil template is **fixed** and never changes. For every target
pixel `k` (colatitude `θ_k`, longitude `φ_k`) and every gauge orientation
`g`, a total rotation matrix is assembled as:

```
R_total[k, g] = R_gauge(α_g)  @  Rz(φ_k)  @  Ry(θ_k)
                └─ gauge roll ─┘  └─── carry North Pole → pixel k ───┘
```

where `R_gauge(α_g)` is the Rodrigues rotation around the surface normal
`n_k` by the gauge angle `α_g` (see {doc}`/conv/convol`).

Each stencil point is then rotated to its position on the sphere around
pixel `k`:

```
rotated[k, g, p] = R_total[k, g] @ vec_pole[p]    ∈ ℝ³
```

producing a tensor of shape `[K, G, P, 3]`.

### Stage B — bilinear binding

For each of the `K × G × P` rotated direction vectors, the function
`get_interp_weights` returns the 4 nearest HEALPix pixel indices and their
bilinear interpolation weights:

```
idx[4, K·G·P]    — absolute NESTED pixel ids of the 4 neighbours
w  [4, K·G·P]    — bilinear weights (sum to 1 per stencil point)
```

For regional-domain inputs, neighbours outside the domain are masked and
weights are renormalised; stencil points with no available neighbour fall back
to the centre pixel. The result is stored tap-major, as
`_tap_idx [P, 4, G, K]` (int32 positions of the neighbours in the order of the
input: the sort permutation of the domain is folded in) and
`_tap_w [P, 4, G, K]`.

### Stage C — forward pass

**Important:** it is the **data** that is brought to the fixed kernel, not
the kernel that is deformed per pixel.

At inference, for each stencil point `p` of pixel `k` under gauge `g`, the
signal value is obtained by bilinear interpolation of the input map:

```
x_interp[b, c, g, k, p] = Σ_{j=0}^{3}  w[g, j, k, p] · x[b, c, nbr[g, j, k, p]]
```

The interpolated values are then contracted with the learned kernel:

```
y[b, g·C_out + o, k] = Σ_{c, p}  W[g, c, o, p] · x_interp[b, c, g, k, p]
```

In index notation: `"bcgkp, gcop -> bgok"`.

`x_interp` is never stored whole. The PyTorch backend loops over the taps:

```python
for p in range(P):                                 # x: [K, B, C_in]
    a = sum(w[p, j, :, :, None, None] * x[idx[p, j]] for j in range(4))
    y += bmm(a.view(G, K * B, C_in), W[..., p])    # [G, K·B, C_out]
```

and the Triton backend fuses the same loop into one kernel, building each
tap's tile of `x_interp` in registers. Both have a hand-written backward that
recomputes the tile instead of storing it.

---

## Private helpers

These functions are not part of the public API but are documented here for
developers who want to extend or debug the module.

### `_local_kernel_grid(kernel_sz, nside) → np.ndarray [P, 3]`

Builds the `kernel_sz × kernel_sz` stencil at the North Pole. Angular
spacing equals one pixel width (`sqrt(4π / (12·nside²))`). The centre
point (index `kernel_sz//2 · (kernel_sz + 1)`) maps to the exact North
Pole direction `[0, 0, 1]`.

### `_build_rotation_matrices(th, ph, G, gauge_type, ...) → torch.Tensor [K, G, 3, 3]`

Assembles the full rotation matrix `R_total = R_gauge(α_g) @ Rz(φ) @ Ry(θ)`
for every pixel and gauge. `R_gauge` is computed via the Rodrigues formula
(axis-angle rotation around the surface normal `n`):

```
R_gauge = I·cos(α) + K_skew·sin(α) + (n⊗n)·(1 − cos(α))
```

where `K_skew` is the skew-symmetric matrix of `n`.

### `_get_interp_weights(nside, vecs, nest, device, dtype) → (idx [4, M], w [4, M])`

Converts `M` direction vectors to bilinear interpolation weights and
neighbour indices on the HEALPix grid. Internally delegates to
`healpix_analyse.healpix_interp.get_interp_weights` (one vectorised call,
no Python loop).

### `_bind_support_batched(idx_t, w_t, ids_sorted, ...) → (pos_safe, w_norm)`

Maps absolute NESTED pixel ids to column indices within the current pixel
patch via `torch.searchsorted`. Handles three edge cases: neighbours
outside the patch (zeroed), stencil points with no valid neighbour (fall
back to the centre pixel), and zero-sum weight columns (assign weight 1 to
the first present neighbour). All G gauges are processed in a single
`searchsorted` call over the full `[G · 4 · K · P]` index array.

