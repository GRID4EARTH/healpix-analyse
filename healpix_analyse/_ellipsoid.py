"""Tiny internal helper: normalize the case of a ``healpix_geo`` ellipsoid name.

``healpix_geo``'s ellipsoid registry (``Ellipsoid::named()`` on the Rust
side) matches names **case-sensitively**: it accepts e.g. ``"WGS84"`` and
``"sphere"``, but rejects ``"wgs84"`` or ``"SPHERE"`` outright with
``ValueError: Operator '<name>' not foundEllipsoid::named()``. Real data
sources do not all agree on a casing -- for example EOPF/GRID4EARTH
Sentinel-2 HEALPix products declare their ellipsoid as the lowercase
string ``"wgs84"``. Rather than have every caller (and every notebook)
remember to re-case that string by hand, the public constructors and
functions of this package that accept an ``ellipsoid`` argument pass it
through :func:`canonicalize_ellipsoid` on entry, so ``"wgs84"``,
``"Wgs84"``, ``"Sphere"`` or ``"SPHERE"`` work wherever ``"WGS84"`` or
``"sphere"`` do. Names ``healpix_geo`` does not know under any of these
casings are passed through unchanged and fail with ``healpix_geo``'s own
error at the point of use.

Known exception: ``healpix_analyse.alm.AlmTransform`` does not canonicalize
yet (``powerspectra`` does it before calling it).
"""

from __future__ import annotations

import numpy as np
from healpix_geo._healpix_geo_python import resolve_ellipsoid

__all__ = ["canonicalize_ellipsoid", "authalic_scale"]


def canonicalize_ellipsoid(ellipsoid):
    """Return ``ellipsoid`` re-cased to whatever spelling ``healpix_geo``'s
    own registry accepts, trying the string as given, then upper case,
    lower case, and title case, in that order.

    This does **not** hardcode the list of valid ellipsoid names (that list
    belongs to ``healpix_geo`` and may grow); it only tries a few common
    case variants and keeps the first one ``healpix_geo`` itself accepts.
    If none of them resolve, the original string is returned unchanged, so
    a genuinely unknown name still fails with ``healpix_geo``'s own, more
    informative error at the point of use, rather than a confusing one
    raised from here.

    Anything that is not a string (for instance a ``dict`` of ellipsoid
    parameters, which ``healpix_geo`` also accepts) is returned unchanged.
    """
    if not isinstance(ellipsoid, str):
        return ellipsoid
    s = ellipsoid
    for candidate in (s, s.upper(), s.lower(), s.title()):
        try:
            resolve_ellipsoid(candidate)
            return candidate
        except Exception:
            continue
    return s


def authalic_scale(lat_geodetic_deg, ellipsoid):
    """Scale factor ``k`` of the authalic sphere at the given geodetic latitudes.

    A HEALPix grid on an ellipsoid is the HEALPix grid of its authalic sphere
    (radius ``R_a``, same area), reached through the authalic latitude
    ``beta``.  That map keeps areas but not shapes: a ground displacement of
    ``ds`` metres becomes ``k * ds`` along the meridian and ``ds / k`` along
    the parallel on the authalic sphere, with

        k = N(phi) cos(phi) / (R_a cos(beta)),

    ``N`` the prime-vertical radius of curvature.  ``k`` is 1 on a sphere; on
    WGS84 it runs from 1 at the poles to 1.0011 at the equator.  Returns an
    array of the shape of ``lat_geodetic_deg``.
    """
    phi = np.radians(np.asarray(lat_geodetic_deg, dtype=np.float64))
    params = resolve_ellipsoid(canonicalize_ellipsoid(ellipsoid)) if isinstance(
        ellipsoid, str) else dict(ellipsoid)
    inv_f = params.get("inverse_flattening")
    if inv_f is None or not np.isfinite(inv_f) or inv_f == 0:
        return np.ones_like(phi)
    a = 1.0                                     # k is a ratio: any unit
    f = 1.0 / float(inv_f)
    e2 = f * (2.0 - f)
    e = np.sqrt(e2)
    q = lambda s: (1 - e2) * (s / (1 - e2 * s * s)
                              - np.log((1 - e * s) / (1 + e * s)) / (2 * e))
    qp = q(1.0)
    r_a = a * np.sqrt(qp / 2.0)
    # At the poles both cosines vanish and cos(beta), from 1 - sin(beta)**2,
    # loses all its digits: k is taken at 89.99 degrees there, where it is
    # within 1e-8 of its polar limit.
    lim = np.radians(89.99)
    phi = np.clip(phi, -lim, lim)
    sinb = np.clip(q(np.sin(phi)) / qp, -1.0, 1.0)
    cosb = np.sqrt((1.0 - sinb) * (1.0 + sinb))
    n_rad = a / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    return n_rad * np.cos(phi) / (r_a * cosb)
