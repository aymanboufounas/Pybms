from __future__ import annotations

import numpy as np
from .exceptions import UnknownLossError
from .utils import canonical_name

EPS = 1e-12


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_true, y_pred):
    n = max(1, y_true.shape[0])
    return 2.0 * (y_pred - y_true) / n


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def mae_grad(y_true, y_pred):
    n = max(1, y_true.shape[0])
    return np.sign(y_pred - y_true) / n


def binary_crossentropy(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_crossentropy_grad(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1.0 - EPS)
    y = np.asarray(y_true, dtype=float)
    n = max(1, y.shape[0])
    return ((p - y) / (p * (1.0 - p))) / n


def categorical_crossentropy(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1.0)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def categorical_crossentropy_grad(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1.0)
    y = np.asarray(y_true, dtype=float)
    n = max(1, y.shape[0])
    return -(y / p) / n


LOSSES = {
    "mse": (mse, mse_grad),
    "mae": (mae, mae_grad),
    "binary_crossentropy": (binary_crossentropy, binary_crossentropy_grad),
    "categorical_crossentropy": (categorical_crossentropy, categorical_crossentropy_grad),
}


def get_loss(name):
    name = canonical_name(name, LOSSES.keys(), "loss")
    if name not in LOSSES:
        raise UnknownLossError(f"Unknown loss '{name}'. Available: {', '.join(LOSSES)}.")
    return name, LOSSES[name][0], LOSSES[name][1]
