from __future__ import annotations

import numpy as np
from .activations import get_activation, backward_activation
from .exceptions import ShapeError
from .utils import pop_fuzzy


class Dense:
    def __init__(self, units=None, activation="linear", input_dim=None, **kwargs):
        if units is None:
            units = pop_fuzzy(kwargs, "units", aliases=("unit", "neurons", "nodes", "output_size"))
        activation = pop_fuzzy(
            kwargs,
            "activation",
            aliases=("activaton", "activtion", "active", "act"),
            default=activation,
        )
        input_dim = pop_fuzzy(
            kwargs,
            "input_dim",
            aliases=("input", "input_size", "inputs", "in_features"),
            default=input_dim,
        )
        if kwargs:
            raise TypeError(f"Unknown Dense arguments: {', '.join(kwargs)}")
        if units is None:
            raise TypeError("Dense needs 'units', e.g. Dense(16, activation='relu').")
        try:
            units = int(units)
        except Exception as exc:
            raise TypeError("Dense units must be an integer.") from exc
        if units <= 0:
            raise ValueError("Dense units must be > 0.")

        self.units = units
        self.input_dim = int(input_dim) if input_dim is not None else None
        self.activation_name, self.activation, _ = get_activation(activation)
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self._x = None
        self._z = None
        self._a = None

        if self.input_dim is not None:
            self.build(self.input_dim)

    def build(self, input_dim):
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ShapeError("input_dim must be > 0.")
        self.input_dim = input_dim
        if self.activation_name == "relu":
            scale = np.sqrt(2.0 / input_dim)
        else:
            scale = np.sqrt(1.0 / input_dim)
        self.W = np.random.randn(input_dim, self.units) * scale
        self.b = np.zeros((1, self.units), dtype=float)
        return self

    @property
    def built(self):
        return self.W is not None

    def forward(self, x, training=False):
        if not self.built:
            self.build(x.shape[1])
        if x.shape[1] != self.input_dim:
            raise ShapeError(
                f"Dense expected {self.input_dim} features but received {x.shape[1]}."
            )
        z = x @ self.W + self.b
        a = self.activation(z)
        if training:
            self._x, self._z, self._a = x, z, a
        return a

    def backward(self, upstream):
        if self._x is None:
            raise RuntimeError("Dense.backward() called before a training forward pass.")
        dz = backward_activation(self.activation_name, self._z, self._a, upstream)
        self.dW = self._x.T @ dz
        self.db = np.sum(dz, axis=0, keepdims=True)
        return dz @ self.W.T

    def count_params(self):
        if not self.built:
            return 0
        return int(self.W.size + self.b.size)

    def __repr__(self):
        return f"Dense(units={self.units}, activation='{self.activation_name}')"
