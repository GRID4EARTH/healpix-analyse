# Migrating projected-raster directional operations to HEALPix

Background for {doc}`directional_filter`: why it takes metres and a compass
bearing rather than pixel offsets, and how a Sentinel-2 L2A processing step
moves across.

## The Sentinel-2 case

The EOPF Sentinel-2 MSI Detailed Processing Model puts part of the chain in the
[Projected Geometry Processor](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/dpm/projected_geometry/projected-geometry-processor.html),
which works in UTM geometry: the end of L1C mask processing, and the L2A scene
classification and atmospheric correction.

The [L2A scene-classification interface](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/api/s2msi.projected_geometry_processor.l2a_scene_classification_processor.html)
takes cloud-shadow parameters that are all physical: a `resolution` in metres,
the solar azimuth `solaz`, the solar zenith `solze_noclip`, cloud-height
thresholds in metres. Terrain slope and cast shadow come through the same
[projected-geometry workflow](https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/ppb/projected_geometry.html).

## Where the meaning gets lost

On a UTM raster, that physical statement is turned into array arithmetic:
an azimuth and a ground distance, divided by the pixel size, become a `(dx, dy)`
displacement and then a row/column shift.

That conversion is an *implementation* of the geographical operation. The
trouble starts when it is mistaken for its definition, because the natural
translation to HEALPix then looks like "move N cell indices" or "take the third
neighbour the topology routine returns". Neither carries any compass
information: HEALPix index ordering and neighbour-list ordering do not define
East or North, and cell shape varies with latitude.

## What to do instead

Keep the physical quantities and skip the raster step entirely. The solar
azimuth and the search distance go straight into `directional_filter()`, which
derives WGS84 distances and forward bearings itself and hands both to your
kernel.

A shadow search becomes the kernel, and reads like the physical statement it
came from — *look about 400 m away, in the sun's direction, within ±10°*:

```python
def shadow_displacement(distance_m, relative_bearing_rad,
                        target_m=400.0, tol_m=50.0, tol_deg=10.0):
    return ((np.abs(distance_m - target_m) <= tol_m)
            & (np.abs(relative_bearing_rad) <= np.deg2rad(tol_deg))).astype(float)

shadow_response = directional_filter(
    cloud_probability, cell_ids, level,
    max_distance_m=shadow_distance_m,
    azimuth_rad=shadow_azimuth_rad,
    kernel=shadow_displacement,
    domain=domain,
)
```

The HEALPix version is not trying to reproduce the UTM row/column
representation. It preserves what that representation was encoding.

## Where the boundary sits

The mission keeps its physics; the library keeps the geometry.

`healpix-analyse` provides a generic geographical directional operator and
nothing else — no cloud-height model, no illumination model, no Sentinel-2
rule. S2MSI supplies the direction, the distance and the kernel that express
its own shadow and illumination logic. Embedding those in a general-purpose
library would tie it to one mission and one processing baseline.

## How the pieces fit

Three layers, each reusable on its own:

- `_neighbourhood.py` finds the cells within a physical radius and computes
  WGS84 distances and forward azimuths.
- `directional_filter.py` turns a forward azimuth into a bearing relative to
  the requested direction, and evaluates your kernel.
- `_weighted_neighbourhood.py` gathers the neighbour values, handles validity
  and NaN, applies the weights, normalises, and aggregates in NumPy or Torch.

{doc}`radial_filter` sits in the same stack with a `kernel(distance_m)` and no
bearing — same search, same aggregation, different spatial semantics.

## The rule, in one line

> Do not translate a physically directional Earth-observation operation into
> HEALPix index directions.

Distance stays in metres on WGS84, direction stays a geographical forward
azimuth with 0 at North, support stays a physical radius, and the boundary
stays an explicit processing domain.
