"""Common validation; all table transformations return copies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .exceptions import DataError, ShapeError


def frame(data) -> pd.DataFrame:
    try:
        result = data.copy(deep=True) if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    except (TypeError, ValueError) as exc:
        raise DataError("Expected a table, dict of columns, or 2D array.") from exc
    if not result.columns.is_unique:
        raise DataError("Column names must be unique.")
    return result


def columns(df, selection=None, numeric=False):
    names = list(df.select_dtypes(include="number").columns) if numeric else list(df.columns)
    if selection is not None:
        names = [selection] if isinstance(selection, (str, int)) else list(selection)
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise DataError(f"Columns not found: {missing}. Available: {list(df.columns)}")
    if numeric and any(not pd.api.types.is_numeric_dtype(df[name]) for name in names):
        raise DataError("This operation requires numeric columns.")
    return names


def finite_array(values, *, ndim=None):
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise DataError("Expected numeric values.") from exc
    if not result.size or not np.isfinite(result).all():
        raise DataError("Values must be nonempty and finite (no NaN or infinity).")
    if ndim is not None and result.ndim != ndim:
        raise ShapeError(f"Expected {ndim} dimensions; received shape {result.shape}.")
    return result


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise DataError(f"{name} must be a positive integer.")
    return int(value)
