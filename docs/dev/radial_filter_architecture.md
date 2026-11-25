# Radial filter: implementation architecture

Moved from the user guide of {doc}`/conv/radial_filter`.

The implementation deliberately reuses the shared geometry introduced for neighbouring HEALPix operators and the shared weighted aggregation used by directional filtering.

```text
physical neighbourhood
        |
        v
processing-domain restriction
        |
        v
relative_geometry_from_neighbourhoods(...)
        |
        v
distance_m
        |
        v
kernel(distance_m)
        |
        v
weights
        |
        v
weighted_neighbourhood_reduce(...)
        |
        v
filtered values
```

`radial_filter` does not independently implement:

- HEALPix neighbourhood padding,
- WGS84 geodesic distance calculation,
- neighbour-value gathering,
- generic weighted normalization,
- Torch device/autograd handling.

Those responsibilities remain in the shared private layers so that radial and directional operators behave consistently.
