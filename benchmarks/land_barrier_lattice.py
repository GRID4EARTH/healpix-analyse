"""Metrication of the 8-connected HEALPix lattice: D0 / straight distance.

D0 is the shortest lattice path (every cell passable) between two cell
centres, straight the geodesic distance.  The maximum over the pairs of a
disc bounds ``_SEARCH_FACTOR`` of ``healpix_analyse/_land_barrier.py``.
Whole sphere; the search disc is 1.6 times the evaluated disc so that the
shortest paths of the evaluated pairs stay inside it.  Writes
``land_barrier_lattice.json``: per level and disc radius (in cells) the
maximum, mean and 99th percentile of D0 / straight over pairs with
straight <= radius, overall and by absolute-latitude band.
"""
import json
from pathlib import Path

import numpy as np
from healpix_geo import nested

from healpix_analyse import _land_barrier as lb
from healpix_analyse._neighbourhood import build_metric_neighbourhood_geometry
from healpix_analyse.radial_filter import _cell_size_m

CASES = {3: (1.5, 3.0, 6.0), 4: (1.5, 3.0, 6.0), 5: (1.5, 3.0, 6.0), 6: (1.5, 3.0), 7: (3.0,)}
out = {}
for level, radii in CASES.items():
    cells = np.arange(12 * 4**level, dtype=np.uint64)
    lat = np.asarray(nested.healpix_to_lonlat(cells, level)[1])  # degrees
    table = lb.domain_neighbour_table(cells, level)
    length = lb.edge_lengths(cells, table, level)
    allowed = lb.passable_edges(table, np.zeros(cells.size, bool))
    cell = _cell_size_m(level)
    out[str(level)] = {}
    for radius_cells in radii:
        radius = radius_cells * cell
        geom = build_metric_neighbourhood_geometry(cells, 1.6 * radius, level)
        rows = np.repeat(np.arange(cells.size), np.diff(geom.row_offsets))
        cols = geom.neighbour_indices.astype(np.int64)
        _, free = lb.path_distances(rows, cols, cells.size, table, length, allowed)
        ok = (rows != cols) & (geom.distance_m <= radius)
        ratio = free[ok] / geom.distance_m[ok]
        band = np.abs(lat[rows[ok]]) // 15 * 15
        entry = dict(
            pairs=int(ok.sum()),
            max=round(float(ratio.max()), 4),
            mean=round(float(ratio.mean()), 4),
            p99=round(float(np.quantile(ratio, 0.99)), 4),
            unreached=int(np.isinf(ratio).sum()),
            max_by_abs_lat_band_deg={
                int(b): round(float(ratio[band == b].max()), 3) for b in np.unique(band)
            },
        )
        out[str(level)][str(radius_cells)] = entry
        print(level, radius_cells, entry, flush=True)
Path(__file__).with_name("land_barrier_lattice.json").write_text(json.dumps(out, indent=1))
