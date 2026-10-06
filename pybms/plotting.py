"""Matplotlib plots that return Axes (pair_plot returns a Figure). No automatic show()."""

from __future__ import annotations

import numpy as np
from ._registry import api
from ._shared import columns, frame, positive_int
from .exceptions import ConfigurationError, DataError


def _axes(ax=None, title=None):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    if title:
        ax.set_title(title)
    return ax


def _xy(data, x, y):
    df = frame(data)
    numeric = list(df.select_dtypes(include="number").columns)
    if x is None:
        if not numeric:
            raise DataError("Choose an x column; no numeric columns were found.")
        x = numeric[0]
    if y is None:
        remaining = [c for c in numeric if c != x]
        if not remaining:
            raise DataError("Choose a y column; two columns are required.")
        y = remaining[0]
    columns(df, [x, y])
    return df, x, y


@api("Plots", "ax = pb.plot(df)")
def plot(data, x=None, y=None, *, kind="auto", **kwargs):
    """Choose histogram for one numeric column or scatter for two; explicit kind is allowed.

    Supported kinds are line/scatter/bar/hist/box/heatmap/pie. Returns Axes;
    call ax.figure.savefig('plot.png') or matplotlib.pyplot.show() yourself.
    """
    df = frame(data)
    if kind == "auto":
        numeric = df.select_dtypes(include="number").columns
        kind = "scatter" if len(numeric) >= 2 else "hist" if len(numeric) else "bar"
    if kind in {"line", "scatter", "bar"}:
        return {"line": line_plot, "scatter": scatter_plot, "bar": bar_plot}[kind](
            df, x, y, **kwargs
        )
    if kind in {"hist", "box"}:
        return {"hist": hist_plot, "box": box_plot}[kind](df, names=x, **kwargs)
    if kind == "heatmap":
        return heatmap(df, **kwargs)
    if kind == "pie":
        return pie_plot(df, name=x, **kwargs)
    raise ConfigurationError("Unknown plot kind. Use auto/line/scatter/bar/hist/box/heatmap/pie.")


@api("Plots", "ax = pb.line_plot(df, x='feature_0', y='feature_1')")
def line_plot(data, x=None, y=None, *, ax=None, title=None, **kwargs):
    """Plot y against x in row order; x/y default to the first two numeric columns."""
    df, x, y = _xy(data, x, y)
    ax = _axes(ax, title)
    ax.plot(df[x], df[y], **kwargs)
    ax.set(xlabel=str(x), ylabel=str(y))
    return ax


@api("Plots", "ax = pb.scatter_plot(df, x='feature_0', y='feature_1')")
def scatter_plot(data, x=None, y=None, *, ax=None, title=None, **kwargs):
    """Scatter two columns; Matplotlib options such as c/s/alpha pass through."""
    df, x, y = _xy(data, x, y)
    ax = _axes(ax, title)
    ax.scatter(df[x], df[y], **kwargs)
    ax.set(xlabel=str(x), ylabel=str(y))
    return ax


@api("Plots", "ax = pb.bar_plot(df, x='target')")
def bar_plot(data, x=None, y=None, *, ax=None, title=None, **kwargs):
    """Bar chart of x/y, or category counts of x when y is omitted."""
    df = frame(data)
    if not len(df.columns):
        raise DataError("Bar plot needs a column.")
    x = df.columns[0] if x is None else x
    columns(df, [x] if y is None else [x, y])
    ax = _axes(ax, title)
    if y is None:
        counts = df[x].value_counts(dropna=False)
        labels, values = counts.index.astype(str), counts.to_numpy()
    else:
        labels, values = df[x].astype(str), df[y]
    ax.bar(labels, values, **kwargs)
    ax.set(xlabel=str(x), ylabel="count" if y is None else str(y))
    return ax


@api("Plots", "ax = pb.hist_plot(df, names='feature_0', bins=15)")
def hist_plot(data, *, names=None, bins="auto", ax=None, title=None, **kwargs):
    """Overlay histograms of selected numeric columns; bins defaults to NumPy auto bins."""
    df = frame(data)
    chosen = columns(df, names, numeric=True)
    if not chosen:
        raise DataError("Histogram needs numeric columns.")
    ax = _axes(ax, title)
    kwargs.setdefault("alpha", 0.6)
    for name in chosen:
        values = df[name].dropna()
        if len(values):
            ax.hist(values, bins=bins, label=str(name), **kwargs)
    ax.set(xlabel="value", ylabel="count")
    ax.legend()
    return ax


@api("Plots", "ax = pb.box_plot(df, names=['feature_0', 'feature_1'])")
def box_plot(data, *, names=None, ax=None, title=None, **kwargs):
    """Draw distributions of selected numeric columns as labeled box plots."""
    df = frame(data)
    chosen = columns(df, names, numeric=True)
    if not chosen:
        raise DataError("Box plot needs numeric columns.")
    ax = _axes(ax, title)
    ax.boxplot(
        [df[c].dropna().to_numpy() for c in chosen], tick_labels=[str(c) for c in chosen], **kwargs
    )
    return ax


@api("Plots", "ax = pb.heatmap(df)")
def heatmap(data, *, correlation=True, ax=None, title=None, annotate=False, **kwargs):
    """Show numeric column correlations, or a numeric matrix when correlation=False."""
    df = frame(data).select_dtypes(include="number")
    if not len(df.columns) or not len(df):
        raise DataError("Heatmap needs a nonempty numeric matrix.")
    values = df.corr() if correlation else df
    ax = _axes(ax, title)
    kwargs.setdefault("cmap", "coolwarm" if correlation else "viridis")
    if correlation:
        kwargs.setdefault("vmin", -1)
        kwargs.setdefault("vmax", 1)
    rendered = ax.imshow(values.to_numpy(), aspect="auto", **kwargs)
    ax.set_xticks(
        range(len(values.columns)), [str(c) for c in values.columns], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(values.index)), [str(i) for i in values.index])
    ax.figure.colorbar(rendered, ax=ax, label="correlation" if correlation else "value")
    if annotate:
        for row, col in np.ndindex(values.shape):
            ax.text(col, row, f"{values.iloc[row, col]:.2f}", ha="center", va="center")
    return ax


@api("Plots", "ax = pb.pie_plot(df, name='target')")
def pie_plot(data, name=None, *, ax=None, title=None, **kwargs):
    """Plot category proportions as a pie, using the first column by default."""
    df = frame(data)
    if df.empty:
        raise DataError("Pie plot needs rows and a column.")
    name = df.columns[0] if name is None else name
    columns(df, name)
    counts = df[name].value_counts(dropna=False)
    ax = _axes(ax, title)
    kwargs.setdefault("autopct", "%1.1f%%")
    ax.pie(counts.to_numpy(), labels=counts.index.astype(str), **kwargs)
    return ax


@api("Plots", "fig = pb.pair_plot(df, max_columns=3)")
def pair_plot(data, *, names=None, max_columns=4, title=None):
    """Return a scatter-matrix Figure of up to eight numeric columns; diagonal histograms."""
    positive_int(max_columns, "max_columns")
    if max_columns > 8:
        raise ConfigurationError("max_columns must be <= 8 to keep figures readable.")
    from pandas.plotting import scatter_matrix

    df = frame(data)
    chosen = columns(df, names, numeric=True)[:max_columns]
    if not chosen:
        raise DataError("Pair plot needs numeric columns.")
    axes = scatter_matrix(df[chosen], diagonal="hist", figsize=(3 * len(chosen), 3 * len(chosen)))
    fig = axes[0, 0].figure
    if title:
        fig.suptitle(title)
    return fig


@api("Plots", "ax = pb.missing_plot(df)")
def missing_plot(data, *, ax=None, title="Missing values"):
    """Plot missing-value fractions by column as horizontal bars."""
    df = frame(data)
    values = df.isna().mean().sort_values()
    ax = _axes(ax, title)
    ax.barh(values.index.astype(str), values.to_numpy())
    ax.set(xlabel="fraction missing", xlim=(0, 1))
    return ax


@api("Plots", "ax = pb.class_plot(df, target='target')")
def class_plot(data, target="target", *, ax=None, title="Class distribution"):
    """Draw labeled target class counts; useful for identifying class imbalance."""
    return bar_plot(data, x=target, ax=ax, title=title)


@api("Plots", "ax = pb.loss_plot(network.history_)")
def loss_plot(history, *, ax=None, title="Training loss"):
    """Plot training/validation loss from History, dict, or a Model with history_."""
    if hasattr(history, "history_"):
        history = history.history_
    if "loss" not in history:
        raise DataError("History must contain a loss sequence.")
    ax = _axes(ax, title)
    for key in ["loss", "val_loss"]:
        if key in history and len(history[key]):
            ax.plot(np.arange(1, len(history[key]) + 1), history[key], label=key)
    ax.set(xlabel="epoch", ylabel="loss")
    ax.legend()
    return ax


@api("Plots", "ax = pb.confusion_plot([0, 1, 1], [0, 0, 1])")
def confusion_plot(y_true, y_pred, *, normalize=None, ax=None, title="Confusion matrix", **kwargs):
    """Draw a classification confusion matrix; normalize can be true/pred/all."""
    from sklearn.metrics import ConfusionMatrixDisplay

    ax = _axes(ax, title)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=normalize, ax=ax, **kwargs)
    return ax


@api("Plots", "ax = pb.residual_plot([1, 2, 3], [1.1, 1.8, 3.2])")
def residual_plot(y_true, y_pred, *, ax=None, title="Residuals"):
    """Scatter prediction versus actual-minus-predicted residual with a zero reference."""
    actual, predicted = (
        np.asarray(y_true, dtype=float).reshape(-1),
        np.asarray(y_pred, dtype=float).reshape(-1),
    )
    if not len(actual) or actual.shape != predicted.shape:
        raise DataError("Actual and predicted values must have matching nonempty shapes.")
    ax = _axes(ax, title)
    ax.scatter(predicted, actual - predicted, alpha=0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="prediction", ylabel="actual - prediction")
    return ax


@api("Plots", "ax = pb.feature_plot(importance)")
def feature_plot(importance, *, top=10, ax=None, title="Feature importance"):
    """Plot a feature_importance() table, or a dict mapping feature names to importances."""
    positive_int(top, "top")
    if isinstance(importance, dict):
        import pandas as pd

        df = pd.DataFrame({"feature": list(importance), "importance": list(importance.values())})
    else:
        df = frame(importance)
    columns(df, ["feature", "importance"])
    df = df.nlargest(top, "importance").sort_values("importance")
    ax = _axes(ax, title)
    ax.barh(df["feature"].astype(str), df["importance"], xerr=df["std"] if "std" in df else None)
    ax.set_xlabel("importance")
    return ax
