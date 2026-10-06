"""Local/URL datasets, reproducible synthetic data, and table operations."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from ._registry import api
from ._shared import columns, frame, positive_int
from .exceptions import DataError, NotSupportedError


@dataclass
class DataSplit:
    """X_train, X_test, y_train, y_test; also supports tuple unpacking."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def __iter__(self):
        return iter((self.X_train, self.X_test, self.y_train, self.y_test))


def _read(source, kind, **kwargs):
    readers = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "xlsx": pd.read_excel,
        "xls": pd.read_excel,
        "parquet": pd.read_parquet,
    }
    if kind not in readers:
        raise NotSupportedError(f"Unsupported format '{kind}'. Use CSV, JSON, XLSX, or Parquet.")
    try:
        return frame(readers[kind](source, **kwargs))
    except ImportError as exc:
        raise NotSupportedError("Install optional readers: pip install 'pybms[io]'.") from exc
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        raise DataError(f"Could not read {kind.upper()} dataset: {exc}") from exc


def _format(source):
    suffix = Path(urlparse(str(source)).path).suffix.lower().lstrip(".")
    return {"tsv": "csv", "jsonl": "json", "ndjson": "json"}.get(suffix, suffix)


@api("Datasets", "df = pb.load_data('iris')")
def load_data(source="iris", *, format=None, **kwargs):
    """Load a builtin name, local path, HTTP(S) URL, DataFrame, dict, or array.

    format overrides the extension. Reader options such as sep/header/lines pass
    through to pandas. Builtin datasets include a 'target' column.
    """
    if isinstance(source, (pd.DataFrame, dict, list, tuple, np.ndarray)):
        return frame(source)
    if str(source) in {"iris", "wine", "breast_cancer", "diabetes", "digits"}:
        return sample_data(str(source))
    if urlparse(str(source)).scheme in {"http", "https"}:
        return from_url(str(source), format=format, **kwargs)
    kind = format or _format(source)
    if str(source).lower().endswith(".tsv"):
        kwargs.setdefault("sep", "\t")
    if str(source).lower().endswith((".jsonl", ".ndjson")):
        kwargs.setdefault("lines", True)
    return _read(source, kind, **kwargs)


@api("Datasets", "df = pb.from_url('https://example.org/data.csv')")
def from_url(url: str, *, format=None, timeout=20, max_bytes=50_000_000, **kwargs):
    """Download a direct CSV/JSON/Excel/Parquet URL with timeout and byte limit.

    GitHub blob links become raw links. Query strings are supported. For URLs
    without an extension supply format, or use a recognized Content-Type header.
    JSON must describe records or columns; HTML pages are not datasets.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DataError("Dataset URL must use http:// or https://.")
    positive_int(max_bytes, "max_bytes")
    if not np.isfinite(timeout) or timeout <= 0:
        raise DataError("timeout must be positive and finite.")
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        url = "https://raw.githubusercontent.com" + parsed.path.replace("/blob/", "/", 1)
    try:
        with urlopen(
            Request(url, headers={"User-Agent": "Pybms/0.2"}), timeout=timeout
        ) as response:
            body = response.read(max_bytes + 1)
            content_type = response.headers.get_content_type()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DataError(f"Dataset download failed: {exc}") from exc
    if len(body) > max_bytes:
        raise DataError(f"Dataset exceeds max_bytes={max_bytes}.")
    extension = _format(url)
    kind = (
        format
        or (extension if extension in {"csv", "json", "xlsx", "xls", "parquet"} else None)
        or {
            "text/csv": "csv",
            "application/json": "json",
            "application/vnd.apache.parquet": "parquet",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        }.get(content_type)
    )
    if kind is None or content_type == "text/html":
        raise DataError("Use a direct dataset download URL; specify format='csv' if needed.")
    if urlparse(url).path.lower().endswith(".tsv"):
        kwargs.setdefault("sep", "\t")
    if urlparse(url).path.lower().endswith((".jsonl", ".ndjson")):
        kwargs.setdefault("lines", True)
    return _read(BytesIO(body), kind, **kwargs)


@api("Datasets", "df = pb.read_csv('data.csv')")
def read_csv(source, **kwargs):
    """Read CSV from a path or URL; optional sep/header/encoding go to pandas."""
    return load_data(source, format="csv", **kwargs)


@api("Datasets", "df = pb.read_json('[{\"x\": 1}]')")
def read_json(source, **kwargs):
    """Read JSON records/columns from a path, URL, or literal JSON string."""
    if isinstance(source, str) and source.lstrip().startswith(("[", "{")):
        try:
            json.loads(source)
        except json.JSONDecodeError as exc:
            raise DataError(f"Invalid JSON: {exc}") from exc
        return _read(StringIO(source), "json", **kwargs)
    return load_data(source, format="json", **kwargs)


@api("Datasets", "df = pb.read_excel('data.xlsx')")
def read_excel(source, **kwargs):
    """Read one Excel sheet; sheet_name defaults to the first sheet; needs [io]."""
    if kwargs.get("sheet_name", 0) is None:
        raise DataError("Select one sheet_name; sheet_name=None returns multiple tables.")
    return load_data(source, format="xlsx", **kwargs)


@api("Datasets", "df = pb.read_parquet('data.parquet')")
def read_parquet(source, **kwargs):
    """Read Parquet from a path or URL; requires the optional [io] extra."""
    return load_data(source, format="parquet", **kwargs)


@api("Datasets", "path = pb.save_data(df, 'dataset.csv')")
def save_data(data, path, *, index=False):
    """Save a table as CSV, JSON records, XLSX, or Parquet; return a Path."""
    df, path = frame(data), Path(path)
    kind = _format(path)
    try:
        if kind == "csv":
            df.to_csv(path, index=index, sep="\t" if path.suffix == ".tsv" else ",")
        elif kind == "json":
            df.to_json(path, orient="records", lines=path.suffix in {".jsonl", ".ndjson"})
        elif kind == "xlsx":
            df.to_excel(path, index=index)
        elif kind == "parquet":
            df.to_parquet(path, index=index)
        else:
            raise NotSupportedError("Save using .csv, .json, .jsonl, .xlsx, or .parquet.")
    except ImportError as exc:
        raise NotSupportedError("Install optional readers: pip install 'pybms[io]'.") from exc
    return path


@api("Datasets", "df = pb.sample_data('wine')")
def sample_data(name="iris"):
    """Load offline iris/wine/breast_cancer/diabetes/digits as a DataFrame."""
    loaders = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
        "digits": datasets.load_digits,
    }
    if name not in loaders:
        raise DataError(f"Unknown sample dataset '{name}'. Choose {list(loaders)}.")
    bunch = loaders[name](as_frame=True)
    return bunch.frame.copy()


@api("Datasets", "df = pb.random_data(rows=100, columns=4, seed=7)")
def random_data(rows=100, columns=4, *, seed=42, missing=0.0):
    """Generate a normal-distributed numeric table; missing is a fraction [0,1]."""
    positive_int(rows, "rows")
    positive_int(columns, "columns")
    if not 0 <= missing <= 1:
        raise DataError("missing must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, columns))
    values[rng.random(values.shape) < missing] = np.nan
    return pd.DataFrame(values, columns=[f"feature_{i}" for i in range(columns)])


@api("Datasets", "df = pb.classification_data(rows=200, features=5)")
def classification_data(rows=100, features=4, *, classes=2, seed=42, **kwargs):
    """Create a synthetic classification table with a 'target' column.

    Automatically choose informative/redundant feature counts; sklearn generator
    options such as class_sep, weights, and flip_y can be supplied explicitly.
    """
    positive_int(rows, "rows")
    positive_int(features, "features")
    if classes < 2:
        raise DataError("classes must be >= 2.")
    kwargs.setdefault("n_clusters_per_class", 1)
    required = int(np.ceil(np.log2(classes * kwargs["n_clusters_per_class"])))
    kwargs.setdefault("n_informative", min(features, max(required, features // 2)))
    kwargs.setdefault("n_redundant", 0)
    try:
        X, y = datasets.make_classification(
            n_samples=rows, n_features=features, n_classes=classes, random_state=seed, **kwargs
        )
    except ValueError as exc:
        raise DataError(f"Cannot generate classification data: {exc}") from exc
    result = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)])
    return result.assign(target=y)


@api("Datasets", "df = pb.regression_data(rows=200, features=3, noise=0.1)")
def regression_data(rows=100, features=4, *, noise=1.0, seed=42, **kwargs):
    """Generate a reproducible regression table with continuous 'target'."""
    positive_int(rows, "rows")
    positive_int(features, "features")
    kwargs.setdefault("n_informative", features)
    X, y = datasets.make_regression(
        n_samples=rows, n_features=features, noise=noise, random_state=seed, **kwargs
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)]).assign(target=y)


@api("Datasets", "df = pb.cluster_data(rows=100, centers=3)")
def cluster_data(rows=100, features=2, *, centers=3, seed=42, **kwargs):
    """Generate Gaussian blobs; 'target' contains the known cluster labels."""
    positive_int(rows, "rows")
    positive_int(features, "features")
    X, y = datasets.make_blobs(
        n_samples=rows, n_features=features, centers=centers, random_state=seed, **kwargs
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)]).assign(target=y)


@api("Datasets", "split = pb.split_data(df, target='target')")
def split_data(data, target="target", *, test_size=0.2, seed=42, stratify=False):
    """Split a table by target name; return DataSplit or unpack its four values.

    Use stratify=True for classification. For time series use chronological
    splitting yourself; this function assumes independent shuffled rows.
    """
    df = frame(data)
    columns(df, target)
    if df.empty or len(df.columns) < 2:
        raise DataError("Need rows and at least one feature plus the target.")
    if df[target].isna().any():
        raise DataError("Target contains missing values. Drop those rows before splitting.")
    try:
        values = train_test_split(
            df.drop(columns=target),
            df[target],
            test_size=test_size,
            random_state=seed,
            stratify=df[target] if stratify else None,
        )
    except ValueError as exc:
        raise DataError(f"Cannot split data: {exc}") from exc
    return DataSplit(*values)


@api("Datasets", "df = pb.shuffle_data(df, seed=7)")
def shuffle_data(data, *, seed=42):
    """Shuffle rows reproducibly and reset the index."""
    return frame(data).sample(frac=1, random_state=seed).reset_index(drop=True)


@api("Datasets", "batches = list(pb.batch_data(df, size=32))")
def batch_data(data, size=32, *, shuffle=False, seed=42):
    """Yield copied mini-batch tables including the final partial batch."""
    positive_int(size, "size")
    df = shuffle_data(data, seed=seed) if shuffle else frame(data)
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size].copy()


@api("Datasets", "report = pb.describe_data(df)")
def describe_data(data):
    """Return row/column counts, dtypes, missing counts, and pandas summary."""
    df = frame(data)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "summary": df.describe(include="all") if len(df.columns) else pd.DataFrame(),
    }


@api("Datasets", "first_rows = pb.preview_data(df, rows=5)")
def preview_data(data, rows=5):
    """Return the first rows as a new DataFrame (never prints the entire dataset)."""
    positive_int(rows, "rows")
    return frame(data).head(rows)


@api("Datasets", "features = pb.select_columns(df, ['feature_0', 'feature_1'])")
def select_columns(data, names):
    """Select column names in the given order; raise DataError for missing names."""
    df = frame(data)
    return df.loc[:, columns(df, names)].copy()


@api("Datasets", "merged = pb.join_data(left, right, on='id')")
def join_data(left, right, *, on, how="inner", validate="one_to_one"):
    """Merge two tables; validate defaults to one_to_one to catch row multiplication."""
    try:
        return frame(left).merge(frame(right), on=on, how=how, validate=validate)
    except (ValueError, KeyError) as exc:
        raise DataError(f"Could not join tables: {exc}") from exc


@api("Datasets", "combined = pb.concat_data([df, df])")
def concat_data(tables, *, axis=0):
    """Concatenate tables by rows (axis=0) or columns (axis=1)."""
    values = [frame(table) for table in tables]
    if not values:
        raise DataError("Provide at least one table.")
    return frame(pd.concat(values, axis=axis, ignore_index=(axis == 0)))
