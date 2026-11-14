"""Tiny internal helper: normalize the case of a ``healpix_geo`` ellipsoid name.

``healpix_geo``'s ellipsoid registry (``Ellipsoid::named()`` on the Rust
side) matches names **case-sensitively**: it accepts e.g. ``"WGS84"`` and
``"sphere"``, but rejects ``"wgs84"`` or ``"SPHERE"`` outright with
``ValueError: Operator '<name>' not foundEllipsoid::named()``. Real data
sources do not all agree on a casing -- for example EOPF/GRID4EARTH
Sentinel-2 HEALPix products declare their ellipsoid as the lowercase
string ``"wgs84"``. Rather than have every caller (and every notebook)
remember to re-case that string by hand, every constructor in this package
that accepts an ``ellipsoid`` argument canonicalizes it once, here, at
construction time.
"""

from __future__ import annotations

from healpix_geo._healpix_geo_python import resolve_ellipsoid

__all__ = ["canonicalize_ellipsoid"]


def canonicalize_ellipsoid(ellipsoid: str) -> str:
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
    """
    s = str(ellipsoid)
    for candidate in (s, s.upper(), s.lower(), s.title()):
        try:
            resolve_ellipsoid(candidate)
            return candidate
        except Exception:
            continue
    return s
