"""Small, predictable NumPy helpers."""

from __future__ import annotations

import numpy as np
from ._registry import api
from ._shared import finite_array, positive_int
from .exceptions import ConfigurationError, ShapeError


@api("Arrays", "x = pb.array([[1, 2], [3, 4]])")
def array(values, dtype=float):
    """Create a NumPy array; dtype defaults to float for numerical learning."""
    return np.asarray(values, dtype=dtype)


@api("Arrays", "x = pb.zeros((2, 3))")
def zeros(shape, dtype=float):
    """Create a zero-filled NumPy array with the given shape and dtype."""
    return np.zeros(shape, dtype=dtype)


@api("Arrays", "x = pb.ones((2, 3))")
def ones(shape, dtype=float):
    """Create a one-filled NumPy array with the given shape and dtype."""
    return np.ones(shape, dtype=dtype)


@api("Arrays", "x = pb.random_array((3, 4), seed=7)")
def random_array(shape, *, distribution="normal", seed=42):
    """Generate seeded standard normal or [0,1) uniform values without global RNG changes."""
    rng = np.random.default_rng(seed)
    if distribution == "normal":
        return rng.normal(size=shape)
    if distribution == "uniform":
        return rng.random(shape)
    raise ConfigurationError("distribution must be normal or uniform.")


@api("Arrays", "x = pb.reshape([1, 2, 3, 4], (2, 2))")
def reshape(values, shape):
    """Reshape values; one -1 dimension can infer its size automatically."""
    try:
        return np.asarray(values).reshape(shape)
    except ValueError as exc:
        raise ShapeError(f"Cannot reshape to {shape}: {exc}") from exc


@api("Arrays", "x = pb.flatten([[1, 2], [3, 4]])")
def flatten(values):
    """Return a copied 1D array from values of any dimensionality."""
    return np.asarray(values).flatten()


@api("Arrays", "encoded, labels = pb.one_hot_encode(['cat', 'dog', 'cat'], return_labels=True)")
def one_hot_encode(values, *, return_labels=False):
    """Encode 1D numeric or string labels; optionally return ordered class values."""
    values = np.asarray(values)
    if values.ndim != 1 or not values.size:
        raise ShapeError("Expected a nonempty 1D label array.")
    labels, inverse = np.unique(values, return_inverse=True)
    result = np.eye(len(labels))[inverse]
    return (result, labels) if return_labels else result


@api("Arrays", "ratio = pb.safe_divide([1, 2], [0, 2], default=0)")
def safe_divide(numerator, denominator, *, default=0.0):
    """Broadcast division, replacing zero denominators with default rather than infinity."""
    a, b = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(a.shape, default, dtype=float)
    return np.divide(a, b, out=result, where=b != 0)


@api("Arrays", "smoothed = pb.moving_average([1, 2, 3, 4], window=2)")
def moving_average(values, window=3):
    """Compute valid-window means of a finite 1D sequence (length n-window+1)."""
    positive_int(window, "window")
    x = finite_array(values, ndim=1)
    if window > len(x):
        raise ShapeError("window must not exceed the sequence length.")
    return np.convolve(x, np.ones(window) / window, mode="valid")


@api("Arrays", "scaled = pb.standardize([[1, 3], [2, 5], [3, 7]])")
def standardize(values, *, axis=0):
    """Center and scale finite values to unit standard deviation; constant slices become 0."""
    x = finite_array(values)
    scale = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(scale == 0, 1, scale)


@api("Arrays", "scaled = pb.minmax([10, 20, 30])")
def minmax(values, *, axis=0):
    """Map finite values to [0,1] by axis; constant slices become 0."""
    x = finite_array(values)
    low, high = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
    return safe_divide(x - low, high - low)


@api("Arrays", "corr = pb.correlation([[1, 3], [2, 5], [3, 7]])")
def correlation(values):
    """Pearson correlation matrix between columns of a 2D finite array.

    Constant columns have undefined correlation and NumPy returns NaN for them.
    """
    x = finite_array(values, ndim=2)
    if len(x) < 2:
        raise ShapeError("Correlation requires at least two rows.")
    return np.corrcoef(x, rowvar=False)


@api("Arrays", "d = pb.distance([0, 0], [3, 4])")
def distance(a, b, *, metric="euclidean", axis=-1):
    """Compute euclidean/manhattan/cosine distance between broadcast-compatible vectors.

    Cosine distance requires nonzero vector norms; zero vectors raise DataError.
    """
    x, y = finite_array(a), finite_array(b)
    if metric == "euclidean":
        return np.sqrt(np.sum((x - y) ** 2, axis=axis))
    if metric == "manhattan":
        return np.sum(np.abs(x - y), axis=axis)
    if metric == "cosine":
        denominator = np.linalg.norm(x, axis=axis) * np.linalg.norm(y, axis=axis)
        if np.any(denominator == 0):
            from .exceptions import DataError

            raise DataError("Cosine distance is undefined for zero vectors.")
        return 1 - np.clip(np.sum(x * y, axis=axis) / denominator, -1, 1)
    raise ConfigurationError("metric must be euclidean, manhattan, or cosine.")


@api("Arrays", "indices, values = pb.top_k([0.2, 0.8, 0.4], k=2)")
def top_k(values, k=3):
    """Return indices and values of the k largest elements, descending and stable on ties."""
    x = finite_array(values, ndim=1)
    positive_int(k, "k")
    if k > len(x):
        raise ShapeError("k cannot exceed the array length.")
    indices = np.argsort(-x, kind="stable")[:k]
    return indices, x[indices]


@api("Arrays", "stats = pb.array_stats([1, 2, 3, 4])")
def array_stats(values):
    """Return shape, mean, std, median, min, max, and finite count without mutating values."""
    x = finite_array(values)
    return {
        "shape": x.shape,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "count": int(x.size),
    }
