from __future__ import annotations

import difflib
import warnings
import numpy as np


ALIASES = {
    "sigmod": "sigmoid",
    "sigmiod": "sigmoid",
    "reluu": "relu",
    "reul": "relu",
    "tan": "tanh",
    "soft max": "softmax",
    "soft_max": "softmax",
    "bce": "binary_crossentropy",
    "binarycrossentropy": "binary_crossentropy",
    "binary_cross_entropy": "binary_crossentropy",
    "cce": "categorical_crossentropy",
    "crossentropy": "categorical_crossentropy",
    "cross_entropy": "categorical_crossentropy",
    "mae": "mae",
    "mean_absolute_error": "mae",
    "mse": "mse",
    "mean_squared_error": "mse",
    "adm": "adam",
    "adma": "adam",
    "sgdoptimizer": "sgd",
}


def canonical_name(value, choices, kind="value"):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    raw = value.strip().lower().replace("-", "_")
    raw = ALIASES.get(raw, raw)
    if raw in choices:
        return raw
    match = difflib.get_close_matches(raw, list(choices), n=1, cutoff=0.55)
    if match:
        warnings.warn(
            f"Pybms auto-corrected {kind} '{value}' -> '{match[0]}'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return match[0]
    return raw


def pop_fuzzy(kwargs, canonical, aliases=(), default=None):
    if canonical in kwargs:
        return kwargs.pop(canonical)
    for alias in aliases:
        if alias in kwargs:
            warnings.warn(
                f"Pybms auto-corrected argument '{alias}' -> '{canonical}'.",
                RuntimeWarning,
                stacklevel=3,
            )
            return kwargs.pop(alias)
    keys = list(kwargs)
    match = difflib.get_close_matches(canonical, keys, n=1, cutoff=0.70)
    if match:
        key = match[0]
        warnings.warn(
            f"Pybms auto-corrected argument '{key}' -> '{canonical}'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return kwargs.pop(key)
    return default


def ensure_2d_x(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 0:
        raise ValueError("X must contain samples, not a scalar.")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        original = X.shape
        X = X.reshape(X.shape[0], -1)
        warnings.warn(
            f"Pybms flattened X automatically from shape {original} to {X.shape}. "
            "Version 0.1 uses dense layers only.",
            RuntimeWarning,
            stacklevel=3,
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values. Clean the data before training.")
    if not X.size or not len(X):
        raise ValueError("X must contain rows and features.")
    return X


def infer_task(y):
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        if (
            np.issubdtype(y_arr.dtype, np.number)
            and np.all(np.isin(y_arr, [0, 1]))
            and np.allclose(y_arr.sum(axis=1), 1)
        ):
            return "multiclass"
        return "regression"

    flat = y_arr.reshape(-1)
    if flat.size == 0:
        raise ValueError("y is empty.")
    unique = np.unique(flat)
    if unique.size == 2:
        return "binary"

    if not np.issubdtype(flat.dtype, np.number):
        return "multiclass"
    is_integer_like = np.all(np.isclose(flat, np.round(flat)))
    if is_integer_like and unique.size <= max(20, int(np.sqrt(max(4, flat.size))) + 2):
        return "multiclass"
    return "regression"


def one_hot(y, num_classes=None):
    raw = np.asarray(y).reshape(-1)
    if (
        not raw.size
        or not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
        or not np.allclose(raw, np.round(raw))
    ):
        raise ValueError("Class labels must be nonempty finite integers.")
    y = raw.astype(int)
    if np.any(y < 0):
        raise ValueError("Class labels must be non-negative integers.")
    if num_classes is None:
        num_classes = int(y.max()) + 1
    if num_classes <= int(y.max()):
        raise ValueError("num_classes must exceed the largest label.")
    out = np.zeros((len(y), num_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out
