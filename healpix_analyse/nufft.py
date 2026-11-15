"""Non-Uniform FFT for data sampled on HEALPix cells.

This relies on the `fiNUFFT <https://github.com/flatironinstitute/finufft>__` library.
"""

import itertools
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

try:
    import finufft
except ImportError as exc:
    raise ImportError(
        "HEALPix NUFFT requires the `finufft` package. "
        "Install with `pip install finufft`."
    ) from exc


_Shape = TypeVar("_Shape", bound=tuple[int, ...])


def windowing_weights_polygon(
    x: np.ndarray[_Shape], y: np.ndarray[_Shape], vertices: npt.ArrayLike, length: float
) -> np.ndarray[_Shape, np.dtype[np.float32]]:
    """Return weights for a Tukey apodization window defined by a polygon.

    Weights outside the polygon equal 0. Weights transition to a value of 1 inside the
    polygon over a distance *length* following a cosine function.

    Parameters
    ----------
    x, y:
        Coordinates for which to compute windowing weights.
    vertices:
        Array of coordinates of shape (N, 2) that define the polygon. Points must be
        defined in **counter-clockwise** order.
    length:
        Length of the transition between the polygon border (where weights equal 0) and
        its interior (where weights equal 1). In the same units as *x* and *y*.
    """
    vertices = np.asarray(vertices)
    # close loop
    if (vertices[0] != vertices[-1]).any():
        vertices = np.concat([vertices, vertices[[0]]])

    # sort vertices counter-clockwise
    v1 = vertices[:-1]
    v2 = vertices[1:]
    area = np.sum((v2[:, 0] - v1[:, 0]) * (v2[:, 1] + v1[:, 1]))
    if area > 0:
        vertices = vertices[::-1]

    weights = np.ones_like(x)
    for segment_pair in itertools.pairwise(vertices):
        segment = segment_pair[1] - segment_pair[0]
        x_rel = x - segment_pair[0][0]
        y_rel = y - segment_pair[0][1]
        norm = np.linalg.norm(segment)
        cross = segment[0] * y_rel - segment[1] * x_rel
        dist = cross / norm
        w = np.where(dist < length, 0.5 * (1 - np.cos(np.pi * dist / length)), 1)
        w = np.where(dist > 0, w, 0)
        weights *= w

    return weights


def windowing_weights_bbox(
    x: np.ndarray[_Shape], y: np.ndarray[_Shape], bbox: Sequence[float], length: float
) -> np.ndarray[_Shape, np.dtype[np.float32]]:
    """Return weights for a Tukey apodization window defined by a BBox.

    Weights outside the box equal 0. Weights transition to a value of 1 inside the box
    over a distance *length* following a cosine function.

    Parameters
    ----------
    x, y:
        Coordinates for which to compute windowing weights.
    bbox:
        Bbox definition as [xmin, ymin, xmax, ymax].
    length:
        Length of the transition between the box border (where weights equal 0) and its
        interior (where weights equal 1). In the same units as *x* and *y*.
    """
    xmin, ymin, xmax, ymax = bbox
    vertices = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return windowing_weights_polygon(x, y, vertices, length=length)


def get_bounds(*coordinates: npt.NDArray) -> list[tuple[float, float]]:
    """Return tuple of minimum and maximum values for each coordinate."""
    bounds = []
    for x in coordinates:
        bounds.append((np.min(x), np.max(x)))
    return bounds


def normalize_coordinates(
    coordinates: Sequence[npt.NDArray], bounds: list[tuple[float, float]]
) -> list[npt.NDArray]:
    """Map coordinates into the [-pi, pi] interval.

    Parameters
    ----------
    coordinates:
        List of arrays of coordinates for each dimension.
    bounds:
        List of min and max values for each coordinate.

    Returns
    -------
    coordinates_norm:
        List of arrays of normalized coordinates for each dimension.
    """
    if len(coordinates) != len(bounds):
        raise IndexError("Not as many coordinates as bounds.")

    coordinates_norm = []
    for x, (xmin, xmax) in zip(coordinates, bounds, strict=True):
        x_norm = (x - (xmin + xmax) / 2) / (xmax - xmin)
        x_norm = 2 * np.pi * x_norm
        coordinates_norm.append(x_norm)

    return coordinates_norm


def get_frequencies(
    n_frequencies: Sequence[int], bounds: Sequence[tuple[float, float]]
) -> list[npt.NDArray]:
    """Return frequencies for each dimensions.

    Parameters
    ----------
    n_frequencies:
        List of number of modes to keep for each dimension.
    bounds:
        List of min and max values of the corresponding coordinates for each dimension.
    """
    if len(n_frequencies) != len(bounds):
        raise IndexError("Not as many bounds as dimensions.")

    frequencies = []
    for n, (xmin, xmax) in zip(n_frequencies, bounds, strict=True):
        if n % 2 == 0:
            f = np.arange(-n // 2, n // 2)
        else:
            f = np.arange(-(n - 1) // 2, (n - 1) // 2 + 1)
        frequencies.append(f / (xmax - xmin))
    return frequencies


def nufft(
    x: np.ndarray[tuple[int]],
    y: np.ndarray[tuple[int]],
    data: npt.NDArray,
    *,
    n_frequencies: Sequence[int],
    rtol: float = 1e-6,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.complex128]
]:
    """Compute Non-uniform FFT.

    This is a thin wrapper around `finufft.nufft2d1`.

    Parameters
    ----------
    x, y:
        Coordinates defining the input data sampling. Both of shape (M,). They will be
        automatically mapped to the [-pi;pi] interval.
    data:
        Data to compute FFT on. Can be of shape (M) or (K, M) in which case the NUFFT
        will be batched. It will automatically be converted to complex type.
    n_frequencies:
        Number of frequencies to keep for each dimension (Nx, Ny).
    rtol:
        Relative tolerance. Smaller values give better accuracy but slower computations.

    Returns
    -------
    fx, fy:
        Arrays of frequencies, with shapes Nx and Ny given by *n_frequencies*.
    fft:
        Array of complex Fourier coefficients, with shape (Ny, Nx) or (K, Ny, Nx) if
        applicable.
    """
    if x.ndim > 1 or x.shape != y.shape:
        raise IndexError("x and y must be 1D and have the same shape.")

    M = x.size  # noqa: N806
    if data.ndim > 2 or data.shape[-1] != M:
        raise IndexError("data must have shape (M) or (K, M)")
    K = data.shape[0] if data.ndim == 2 else None  # noqa: N806

    bounds = get_bounds(x, y)
    xy_norm = normalize_coordinates([x, y], bounds)
    fx, fy = get_frequencies(n_frequencies, bounds)

    fhat = finufft.nufft2d1(
        *xy_norm, c=data.astype("complex128"), n_modes=tuple(n_frequencies), eps=rtol
    )
    fhat = fhat.reshape([K, *n_frequencies[::-1]]) if K is not None else fhat.T

    return fx, fy, fhat
