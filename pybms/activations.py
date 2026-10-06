from __future__ import annotations

import numpy as np
from .exceptions import UnknownActivationError
from .utils import canonical_name


def linear(x):
    return x


def linear_grad(x, y=None):
    return np.ones_like(x)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x, y=None):
    s = sigmoid(x) if y is None else y
    return s * (1.0 - s)


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x, y=None):
    return (x > 0).astype(float)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x, y=None):
    t = np.tanh(x) if y is None else y
    return 1.0 - t * t


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=1, keepdims=True)


ACTIVATIONS = {
    "linear": (linear, linear_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "softmax": (softmax, None),
}


def get_activation(name):
    name = canonical_name(name or "linear", ACTIVATIONS.keys(), "activation")
    if name not in ACTIVATIONS:
        raise UnknownActivationError(
            f"Unknown activation '{name}'. Available: {', '.join(ACTIVATIONS)}."
        )
    return name, ACTIVATIONS[name][0], ACTIVATIONS[name][1]


def backward_activation(name, z, a, upstream):
    if name == "softmax":
        return a * (upstream - np.sum(upstream * a, axis=1, keepdims=True))
    _, _, grad_fn = get_activation(name)
    return upstream * grad_fn(z, a)
