r"""Non-uniform FFT for data sampled on HEALPix cells.

This relies on the `fiNUFFT <https://github.com/flatironinstitute/finufft>`__
library, an optional dependency: ``pip install "healpix-analyse[nufft]"``.

Convention
----------
For samples :math:`c_j` at :math:`(x_j, y_j)`, :func:`nufft` returns

.. math::

    F(f_x, f_y) = \sum_j c_j \exp\left(-2 i \pi \left[f_x (x_j - x_0)
    + f_y (y_j - y_0)\right]\right)

with :math:`x_0, y_0` the smallest coordinates and :math:`f_x = k / L_x`,
:math:`f_y = l / L_y` for integers :math:`k, l`, :math:`L_x, L_y` being the
periods. This is the sign of :func:`numpy.fft.fft2`. On a regular grid of
``nx x ny`` samples with spacings ``dx, dy``, the default periods are
``nx * dx`` and ``ny * dy`` and the result equals
``np.fft.fftshift(np.fft.fft2(data.reshape(ny, nx)))``.
"""

import itertools
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

try:
    import finufft
except ImportError as exc:
    raise ImportError(
        "healpix_analyse.nufft requires the optional `finufft` package. "
        'Install it with `pip install "healpix-analyse[nufft]"` '
        "(or `pip install finufft`)."
    ) from exc


def windowing_weights_polygon(
    x: npt.ArrayLike, y: npt.ArrayLike, vertices: npt.ArrayLike, length: float
) -> npt.NDArray[np.float64]:
    """Return weights for a Tukey apodization window defined by a polygon.

    Weights equal 0 outside the polygon and on its border. Inside, they rise to
    1 over a distance *length* from the border following a cosine:
    ``(1 - cos(pi * d / length)) / 2`` at a distance ``d < length`` from the
    nearest edge, 1 beyond. Weights of all segments are multiplied together.

    The polygon can be concave and its vertices can be given in either
    orientation. It must be simple (edges do not cross); inside and outside are
    decided with the even-odd rule. Distances are Euclidean in the units of
    *x* and *y*.

    Parameters
    ----------
    x, y:
        Coordinates for which to compute windowing weights, of the same shape.
    vertices:
        Array of shape (N, 2) with the (x, y) coordinates of the vertices. The
        ring can be given closed (last vertex equal to the first) or open.
    length:
        Length of the transition between the polygon border (where weights equal
        0) and its interior (where weights equal 1). In the same units as *x* and
        *y*.

    Returns
    -------
    weights:
        Array of float64 with the shape of *x*.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not length > 0:
        raise ValueError("length must be positive.")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2).")
    # close the ring
    if (vertices[0] != vertices[-1]).any():
        vertices = np.concatenate([vertices, vertices[:1]])
    if len(vertices) < 4:
        raise ValueError("A polygon needs at least 3 distinct vertices.")

    inside = np.zeros(x.shape, dtype=bool)
    weights = np.ones(x.shape, dtype=np.float64)
    for (x1, y1), (x2, y2) in itertools.pairwise(vertices):
        # even-odd rule: does the ray from (x, y) towards +x cross this edge?
        if y1 != y2:
            crosses = (y1 > y) != (y2 > y)
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (x < x_cross)

        # distance to the segment (not to the infinite line)
        sx, sy = x2 - x1, y2 - y1
        seg2 = sx * sx + sy * sy
        t = 0.0
        if seg2 > 0:
            t = np.clip(((x - x1) * sx + (y - y1) * sy) / seg2, 0.0, 1.0)
        dist = np.sqrt((x - x1 - t * sx) ** 2 + (y - y1 - t * sy) ** 2)

        taper = 0.5 * (1.0 - np.cos(np.pi * dist / length))
        weights *= np.where(dist < length, taper, 1.0)

    return np.where(inside, weights, 0.0)


def windowing_weights_bbox(
    x: npt.ArrayLike, y: npt.ArrayLike, bbox: Sequence[float], length: float
) -> npt.NDArray[np.float64]:
    """Return weights for a Tukey apodization window defined by a BBox.

    Weights equal 0 outside the box and on its border. Inside, they rise to 1
    over a distance *length* from the border following a cosine (see
    :func:`windowing_weights_polygon`).

    Parameters
    ----------
    x, y:
        Coordinates for which to compute windowing weights.
    bbox:
        Bbox definition as [xmin, ymin, xmax, ymax].
    length:
        Length of the transition between the box border (where weights equal 0)
        and its interior (where weights equal 1). In the same units as *x* and
        *y*.
    """
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(
            "bbox must be [xmin, ymin, xmax, ymax] with xmin < xmax and ymin < ymax."
        )
    vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return windowing_weights_polygon(x, y, vertices, length=length)


def get_bounds(*coordinates: npt.ArrayLike) -> list[tuple[float, float]]:
    """Return tuple of minimum and maximum values for each coordinate."""
    return [(float(np.min(x)), float(np.max(x))) for x in coordinates]


def _unique_float(x: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Return unique float values, up to a tolerance."""
    # See https://stackoverflow.com/a/38050528
    as_int = np.floor(x.ravel() / rtol).astype("int")
    return (np.unique(as_int) + 0.5) * rtol


def _is_regular(x: np.ndarray, y: np.ndarray) -> bool:
    """Return if coordinates correspond to a regular grid.

    A regular grid is rectangular, not rotated, but not necessarily uniformly
    spaced. Typically coordinates created by ``np.meshgrid(x, y)``.
    """
    x1d = _unique_float(x)
    y1d = _unique_float(y)

    return x1d.size * y1d.size == x.size


def period_vertices(
    x_vertices: npt.NDArray, y_vertices: npt.NDArray
) -> tuple[float, float]:
    """Return the exact period from the extent of HEALPix cells.

    Parameters
    ----------
    x_vertices, y_vertices : array-like of float
        Coordinates of the vertices of HEALPix cells, as given by the function
        :func:`healpix_geo.auto.vertices`, with an eventual projection applied.
        The last axis must be of size 4.

    Returns
    -------
    period_x, period_y : float
        Tuple of the period for both dimensions.
    """
    for x in (x_vertices, y_vertices):
        if x.ndim < 2 or x.shape[-1] != 4:
            raise ValueError(
                "Vertices coordinates must have their last axis of size 4 "
                f"(received shape {x.shape})."
            )

    xmin = np.min(x_vertices[..., 3])
    xmax = np.max(x_vertices[..., 1])
    ymin = np.min(y_vertices[..., 0])
    ymax = np.max(y_vertices[..., 2])
    return xmax - xmin, ymax - ymin


def period_area(
    n_samples: int, bounds: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """Return a period computed from the estimated average pixel size.

    This estimates the average pixel size from the extent of both coordinates
    and the number of samples. This will overestimate the average pixel size for
    anything that is not rectangular. For rectangular grids, this slightly
    underestimates the pixel size as the coordinates extent are limited to cell
    centers (*ie* we are missing the border) and using another method would be
    recommended.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    bounds : list of tuple of float
        The minimum and maximum values of each coordinates, as given by
        :func:`get_bounds`.

    Returns
    -------
    period_x, period_y : float
        Tuple of the period for both dimensions.
    """
    if len(bounds) != 2:
        raise ValueError("Not as many bounds as coordinates.")

    (xmin, xmax), (ymin, ymax) = bounds
    rect_area = (xmax - xmin) * (ymax - ymin)
    avg_pixel_size = np.sqrt(rect_area / n_samples)
    return (xmax - xmin + avg_pixel_size), (ymax - ymin + avg_pixel_size)


def period_median(x: npt.ArrayLike) -> float:
    """Return the default period of the transform along one coordinate.

    The period is the extent of the coordinate plus its typical sample spacing,
    taken as the median gap between consecutive distinct values. On a regular
    grid of ``n`` samples with spacing ``dx`` this is ``n * dx``, the period of
    :func:`numpy.fft.fft`: the first and last samples are one spacing apart on
    the periodic domain instead of coinciding.

    For scattered samples or rotated grids, such as HEALPix cell centres, the
    median gap is not representative of the pixel size. Other methods to find
    the period should be used.

    Parameters
    ----------
    x : array-like of float
        Coordinates along one dimension.

    Returns
    -------
    period : float
        Period along one dimension.

    Raises
    ------
    ValueError
        If the coordinate takes a single value (zero extent).
    """
    values = _unique_float(np.asarray(x, dtype=np.float64))
    if values.size < 2:
        raise ValueError(
            "A coordinate takes a single value: its extent is zero and the "
            "transform is not defined along it."
        )
    return float(values[-1] - values[0] + np.median(np.diff(values)))


def normalize_coordinates(
    coordinates: Sequence[npt.ArrayLike],
    bounds: Sequence[tuple[float, float]],
    periods: Sequence[float],
) -> list[npt.NDArray[np.float64]]:
    """Map coordinates onto the [0, 2 pi) interval expected by fiNUFFT.

    ``x`` is mapped to ``2 pi (x - xmin) / period``, so that the phase reference
    is the smallest coordinate.

    Parameters
    ----------
    coordinates:
        List of arrays of coordinates for each dimension.
    bounds:
        List of min and max values for each coordinate.
    periods:
        Period of the transform for each coordinate. Must be larger than the
        extent ``max - min``, otherwise samples at both ends overlap.

    Returns
    -------
    coordinates_norm:
        List of arrays of normalized coordinates for each dimension.
    """
    if not len(coordinates) == len(bounds) == len(periods):
        raise ValueError("Not as many coordinates as bounds and periods.")

    coordinates_norm = []
    for x, (xmin, xmax), period in zip(coordinates, bounds, periods, strict=True):
        if not period > xmax - xmin:
            raise ValueError(
                f"The period ({period}) must be larger than the extent of the "
                f"coordinates ({xmax - xmin}), otherwise samples at both ends "
                "overlap."
            )
        x = np.asarray(x, dtype=np.float64)
        coordinates_norm.append(2 * np.pi * (x - xmin) / period)
    return coordinates_norm


def get_frequencies(
    n_frequencies: Sequence[int], periods: Sequence[float]
) -> list[npt.NDArray[np.float64]]:
    """Return frequencies for each dimension.

    For ``n`` modes and a period ``L`` the frequencies are ``k / L`` for
    ``k = -(n // 2), ..., n - n // 2 - 1``, as
    ``np.fft.fftshift(np.fft.fftfreq(n, d=L / n))``. They are in cycles per unit
    of the coordinates.

    Parameters
    ----------
    n_frequencies:
        List of number of modes to keep for each dimension.
    periods:
        Period of the transform for each dimension.
    """
    if len(n_frequencies) != len(periods):
        raise ValueError("Not as many periods as dimensions.")
    return [
        np.arange(-(n // 2), n - n // 2) / period
        for n, period in zip(n_frequencies, periods, strict=True)
    ]


def nufft(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    data: npt.ArrayLike,
    *,
    n_frequencies: Sequence[int],
    period: Sequence[float] | Literal["auto", "median", "area"] = "auto",
    rtol: float = 1e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """Compute a two-dimensional non-uniform FFT.

    This is a thin wrapper around ``finufft.nufft2d1``. It returns

    ``F[l, k] = sum_j data[j] * exp(-2i pi (fx[k] (x[j] - x0) + fy[l] (y[j] - y0)))``

    with ``x0, y0`` the smallest coordinates: the sign of :func:`numpy.fft.fft2`.
    On a regular grid of ``nx x ny`` samples with ``n_frequencies=(nx, ny)``
    and the default periods, the result equals
    ``np.fft.fftshift(np.fft.fft2(data.reshape(ny, nx)))``.

    Parameters
    ----------
    x, y:
        Coordinates defining the input data sampling. Both of shape (M,), finite,
        and each with at least two distinct values. They are mapped to the
        periodic interval [0, 2 pi) with :func:`normalize_coordinates`.
    data:
        Data to compute FFT on. Can be of shape (M,) or (K, M) in which case the
        NUFFT is batched. It must be finite. It is converted to complex128.
    n_frequencies:
        Number of frequencies to keep for each dimension (Nx, Ny).
    period:
        Period of the transform along x and y, in the units of the coordinates;
        it must be larger than the extent of the coordinates. Can also be a
        method to guess the period values:

        - "median": Uses :func:`period_median`, best suited for rectangular grids.
        - "area": Uses :func:`period_area`, best suited for non-rectangular grids.
        - "auto" (default): If the grid is detected as regular, uses "median",
          otherwise "area".
    rtol:
        Relative tolerance. Smaller values give better accuracy but slower
        computations.

    Returns
    -------
    fx, fy:
        Arrays of frequencies, of lengths Nx and Ny given by *n_frequencies*,
        in increasing order (see :func:`get_frequencies`).
    fft:
        Array of complex Fourier coefficients, with shape (Ny, Nx), or
        (K, Ny, Nx) for batched data.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D and have the same shape.")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite.")

    data = np.asarray(data)
    if data.ndim not in (1, 2) or data.shape[-1] != x.size:
        raise ValueError("data must have shape (M,) or (K, M).")
    if not np.isfinite(data).all():
        raise ValueError("data must be finite.")

    n_frequencies = tuple(int(n) for n in n_frequencies)
    if len(n_frequencies) != 2 or min(n_frequencies) < 1:
        raise ValueError("n_frequencies must be two positive integers (Nx, Ny).")

    bounds = get_bounds(x, y)

    if isinstance(period, str):
        if period == "auto":
            if _is_regular(x, y):
                periods = (period_median(x), period_median(y))
            else:
                periods = period_area(x.size, bounds)
        elif period == "median":
            periods = (period_median(x), period_median(y))
        elif period == "area":
            periods = period_area(x.size, bounds)
        else:
            raise KeyError(
                "Invalid period '{}', must be 'auto', 'median', 'area' or a sequence of floats."
            )
    else:
        periods = [float(p) for p in period]
        if len(periods) != 2:
            raise ValueError("period must be two values (Lx, Ly).")

    xy_norm = normalize_coordinates([x, y], bounds, periods)
    fx, fy = get_frequencies(n_frequencies, periods)

    fhat = finufft.nufft2d1(
        *xy_norm,
        np.ascontiguousarray(data, dtype=np.complex128),
        n_modes=n_frequencies,
        eps=rtol,
        isign=-1,
    )
    # finufft returns (..., Nx, Ny): swap the last two axes, batched or not
    return fx, fy, np.swapaxes(fhat, -1, -2)
