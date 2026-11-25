# `AlmCoeffs`, a container for coefficients

`AlmCoeffs` (`healpix_analyse.alm`) is a dataclass that holds spherical
harmonic coefficients with their indices and metadata.

```{include} ../_generated/glance/alm_coeffs.md
```

```python
from healpix_analyse import AlmCoeffs

c = AlmCoeffs(alm=alm, l=l, m=m, meta={"grid": "ERA5"})
c.n_modes, c.is_batched
c_gpu = c.to("cuda")       # also .cpu(), .cuda(), .clone()
```

Fields: `alm` (a complex `torch.Tensor` of shape `[K]` or `[B, K]`), `l`
and `m` (1-D tensors of degrees and orders, `m >= 0`), `meta` (a
dictionary). The constructor validates types and shapes.

No function of the package returns an `AlmCoeffs`: `HEALPixSHT.map2alm` and
`map2alm_latlon` return plain tensors in the `healpy` layout. Whether to use
this container in those functions or to drop it is an open question of the
documentation overhaul.

The same module defines `AlmTransform`, the experimental flat transform used
by `powerspectra` ({doc}`powerspectra`); it is not exported.
