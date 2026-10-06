"""Explicit table cleaning for exploration; train() learns preprocessing per fold."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, normalize

from ._registry import api
from ._shared import columns, frame
from .exceptions import ConfigurationError, DataError


def _selected(df, selection=None, exclude=None, numeric=False):
    return [c for c in columns(df, selection, numeric) if c != exclude]


@api("Cleaning", "clean, report = pb.clean_data(df, target='target', report=True)")
def clean_data(data, *, target=None, duplicates=True, fill=True, report=False):
    """Trim strings, map blanks/infinities to missing, deduplicate, and impute.

    target is excluded from imputation/type coercion and its name is preserved.
    Numeric strings are converted only when all nonmissing values parse. Return
    (table, report) when report=True. Use for exploration; pass raw data to train
    so learned medians/modes stay inside the training folds.
    """
    df = frame(data)
    if target is not None:
        columns(df, target)
    before = {"rows": len(df), "missing": int(df.isna().sum().sum())}
    features = _selected(df, exclude=target)
    df = trim_strings(df, names=features)
    for name in features:
        if pd.api.types.is_object_dtype(df[name]) or pd.api.types.is_string_dtype(df[name]):
            df[name] = df[name].replace(r"^\s*$", np.nan, regex=True)
            nonmissing = df[name].dropna()
            parsed = pd.to_numeric(nonmissing, errors="coerce")
            if len(parsed) and parsed.notna().all():
                df[name] = pd.to_numeric(df[name], errors="coerce")
    numeric = _selected(df, exclude=target, numeric=True)
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
    if duplicates:
        df = drop_duplicates(df)
    if fill:
        df = fill_missing(df, names=features)
    details = {
        "before": before,
        "after": {"rows": len(df), "missing": int(df.isna().sum().sum())},
        "dropped_duplicates": before["rows"] - len(df),
        "target": target,
    }
    return (df, details) if report else df


@api("Cleaning", "df = pb.clean_names(df)")
def clean_names(data):
    """Convert column names to unique snake_case labels, preserving column order."""
    df, used, names = frame(data), set(), []
    for raw in df.columns:
        base = re.sub(r"\W+", "_", str(raw).strip().lower()).strip("_") or "column"
        candidate, suffix = base, 2
        while candidate in used:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        used.add(candidate)
        names.append(candidate)
    df.columns = names
    return df


@api("Cleaning", "df = pb.fill_missing(df, strategy='auto')")
def fill_missing(data, *, strategy="auto", value=None, names=None):
    """Fill numeric medians and category modes, or use mean/mode/constant.

    names selects columns. Empty numeric columns use 0, empty categories use
    '__missing__'. For constant strategy, value must be supplied.
    """
    if strategy not in {"auto", "median", "mean", "mode", "constant"}:
        raise ConfigurationError("strategy must be auto, median, mean, mode, or constant.")
    if strategy == "constant" and value is None:
        raise ConfigurationError("constant strategy requires value.")
    df = frame(data)
    for name in columns(df, names):
        series = (
            df[name].replace([np.inf, -np.inf], np.nan)
            if pd.api.types.is_numeric_dtype(df[name])
            else df[name]
        )
        numeric = pd.api.types.is_numeric_dtype(series)
        if strategy == "constant":
            replacement = value
        elif numeric and strategy in {"auto", "median", "mean"}:
            replacement = (
                (series.mean() if strategy == "mean" else series.median())
                if series.notna().any()
                else 0.0
            )
            if pd.isna(replacement):
                replacement = 0.0
        else:
            mode = series.mode(dropna=True)
            replacement = mode.iloc[0] if len(mode) else (0 if numeric else "__missing__")
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
            and replacement not in series.cat.categories
        ):
            series = series.cat.add_categories([replacement])
        try:
            df[name] = series.fillna(replacement)
        except TypeError:
            df[name] = series.astype(float if numeric else object).fillna(replacement)
    return df


@api("Cleaning", "df = pb.drop_missing(df, names=['target'])")
def drop_missing(data, *, names=None, how="any"):
    """Drop rows with missing values in selected columns; how is any or all."""
    df = frame(data)
    return df.dropna(subset=columns(df, names), how=how).reset_index(drop=True)


@api("Cleaning", "report = pb.missing_report(df)")
def missing_report(data):
    """Return missing counts and fractions for every column, largest first."""
    df = frame(data)
    return pd.DataFrame({"count": df.isna().sum(), "fraction": df.isna().mean()}).sort_values(
        "count", ascending=False
    )


@api("Cleaning", "df = pb.drop_duplicates(df)")
def drop_duplicates(data, *, names=None, keep="first"):
    """Drop identical rows or duplicate keys; names selects the comparison columns."""
    df = frame(data)
    return df.drop_duplicates(subset=columns(df, names), keep=keep).reset_index(drop=True)


@api("Cleaning", "df = pb.coerce_numbers(df, names=['price'])")
def coerce_numbers(data, *, names=None, errors="coerce"):
    """Convert selected columns to numbers; errors='coerce' maps invalid text to NaN."""
    if errors not in {"coerce", "raise"}:
        raise ConfigurationError("errors must be coerce or raise.")
    df = frame(data)
    for name in columns(df, names):
        df[name] = pd.to_numeric(df[name], errors=errors)
    return df


@api("Cleaning", "df = pb.coerce_dates(df, names=['date'], format='%Y-%m-%d')")
def coerce_dates(data, *, names, format=None, errors="coerce", utc=False):
    """Parse selected datetime columns; give format to disambiguate day/month order."""
    df = frame(data)
    for name in columns(df, names):
        df[name] = pd.to_datetime(df[name], format=format, errors=errors, utc=utc)
    return df


@api("Cleaning", "df = pb.trim_strings(df, lower=True)")
def trim_strings(data, *, names=None, lower=False):
    """Strip surrounding whitespace from string cells; optionally lowercase text."""
    df = frame(data)
    for name in columns(df, names):
        if pd.api.types.is_object_dtype(df[name]) or pd.api.types.is_string_dtype(df[name]):
            df[name] = df[name].map(
                lambda v: (v.strip().lower() if lower else v.strip()) if isinstance(v, str) else v
            )
    return df


@api("Cleaning", "df = pb.encode_categories(df, names=['city'])")
def encode_categories(data, *, names=None, drop_first=False):
    """One-hot encode selected categories for exploration, treating missing as a level.

    With names=None choose nonnumeric columns. For train/test data use train()
    which fits one encoder on training rows and handles unseen categories.
    """
    df = frame(data)
    chosen = (
        columns(df, names)
        if names is not None
        else list(df.select_dtypes(exclude="number").columns)
    )
    return pd.get_dummies(df, columns=chosen, drop_first=drop_first, dummy_na=True, dtype=float)


@api("Cleaning", "scaled, scaler = pb.scale_data(df, return_scaler=True)")
def scale_data(data, *, names=None, method="standard", return_scaler=False):
    """Scale numeric columns by standard/minmax/robust; optionally return the fitted scaler.

    Fit only on training rows when evaluating models. This helper is intended
    for exploration; automatic training fits scalers inside each CV fold.
    """
    methods = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    if method not in methods:
        raise ConfigurationError(f"method must be one of {list(methods)}.")
    df = frame(data)
    chosen = columns(df, names, numeric=True)
    if not chosen:
        raise DataError("No numeric columns to scale.")
    scaler = methods[method]()
    df[chosen] = scaler.fit_transform(df[chosen])
    return (df, scaler) if return_scaler else df


@api("Cleaning", "df = pb.normalize_data(df, norm='l2')")
def normalize_data(data, *, names=None, norm="l2"):
    """Normalize each row's numeric vector to unit l1/l2/max norm."""
    df = frame(data)
    chosen = columns(df, names, numeric=True)
    if not chosen:
        raise DataError("No numeric columns to normalize.")
    df[chosen] = normalize(df[chosen], norm=norm)
    return df


def _bounds(df, names, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ConfigurationError("factor must be positive and finite.")
    chosen = columns(df, names, numeric=True)
    q1, q3 = df[chosen].quantile(0.25), df[chosen].quantile(0.75)
    return chosen, q1 - factor * (q3 - q1), q3 + factor * (q3 - q1)


@api("Cleaning", "df = pb.clip_outliers(df, factor=1.5)")
def clip_outliers(data, *, names=None, factor=1.5):
    """Winsorize numeric columns at Q1/Q3 ± factor*IQR; missing values stay missing."""
    df = frame(data)
    chosen, lower, upper = _bounds(df, names, factor)
    df[chosen] = df[chosen].clip(lower=lower, upper=upper, axis=1)
    return df


@api("Cleaning", "df = pb.drop_outliers(df, names=['price'])")
def drop_outliers(data, *, names=None, factor=1.5):
    """Drop rows outside IQR bounds in any selected numeric column."""
    df = frame(data)
    chosen, lower, upper = _bounds(df, names, factor)
    mask = ((df[chosen] < lower) | (df[chosen] > upper)).any(axis=1)
    return df.loc[~mask].reset_index(drop=True)


@api("Cleaning", "report = pb.outlier_report(df)")
def outlier_report(data, *, names=None, factor=1.5):
    """Return lower/upper IQR limits and outlier count for each numeric column."""
    df = frame(data)
    chosen, lower, upper = _bounds(df, names, factor)
    counts = ((df[chosen] < lower) | (df[chosen] > upper)).sum()
    return pd.DataFrame({"lower": lower, "upper": upper, "count": counts})


@api("Cleaning", "df = pb.drop_constant_columns(df, exclude=['target'])")
def drop_constant_columns(data, *, exclude=()):
    """Remove columns with at most one nonmissing value; exclude preserves names."""
    df = frame(data)
    return df.drop(columns=[c for c in df if c not in exclude and df[c].nunique(dropna=True) <= 1])


@api("Cleaning", "df = pb.drop_sparse_columns(df, threshold=0.8)")
def drop_sparse_columns(data, *, threshold=0.8, exclude=()):
    """Remove columns whose missing fraction exceeds threshold [0,1]."""
    if not 0 <= threshold <= 1:
        raise ConfigurationError("threshold must be between 0 and 1.")
    df = frame(data)
    return df.drop(columns=[c for c in df if c not in exclude and df[c].isna().mean() > threshold])


@api("Cleaning", "df = pb.replace_values(df, {'?': None, 'N/A': None})")
def replace_values(data, mapping, *, names=None):
    """Replace exact values using a dict, optionally only in selected columns."""
    df = frame(data)
    chosen = columns(df, names)
    df[chosen] = df[chosen].replace(mapping)
    return df


@api("Cleaning", "train_df = pb.balance_data(train_df, target='target')")
def balance_data(data, target="target", *, method="over", seed=42):
    """Randomly over/undersample classes to equal counts; use ONLY after splitting.

    Oversampling repeats minority rows; it does not create synthetic examples.
    Never balance evaluation rows. Automatic train() uses class weights instead.
    """
    df = frame(data)
    columns(df, target)
    if method not in {"over", "under"}:
        raise ConfigurationError("method must be over or under.")
    if df.empty or df[target].isna().any() or df[target].nunique() < 2:
        raise DataError("Need at least two classes and no missing targets.")
    groups = list(df.groupby(target, sort=False, observed=True))
    count = (max if method == "over" else min)(len(group) for _, group in groups)
    result = pd.concat(
        [
            group.sample(n=count, replace=count > len(group), random_state=seed)
            for _, group in groups
        ],
        ignore_index=True,
    )
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


@api("Cleaning", "report = pb.validate_data(df, target='target')")
def validate_data(data, *, target=None, strict=False):
    """Check emptiness, missing/infinite values and target existence; strict raises DataError."""
    df, issues = frame(data), []
    if df.empty:
        issues.append("Table is empty.")
    if df.isna().any().any():
        issues.append("Table contains missing values.")
    if np.isinf(df.select_dtypes(include="number").to_numpy(dtype=float, na_value=np.nan)).any():
        issues.append("Numeric columns contain infinity.")
    if target is not None and target not in df:
        issues.append(f"Target '{target}' does not exist.")
    report = {"valid": not issues, "issues": issues, "rows": len(df), "columns": len(df.columns)}
    if strict and issues:
        raise DataError(" ".join(issues))
    return report
