from __future__ import annotations

import numpy as np
from .exceptions import UnknownOptimizerError
from .utils import canonical_name


class SGD:
    name = "sgd"

    def __init__(self, lr=0.01, learning_rate=None, momentum=0.0):
        self.lr = float(learning_rate if learning_rate is not None else lr)
        self.momentum = float(momentum)
        self.velocity = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W") or layer.dW is None:
                continue
            if self.momentum:
                vw = self.velocity.get((i, "W"), np.zeros_like(layer.W))
                vb = self.velocity.get((i, "b"), np.zeros_like(layer.b))
                vw = self.momentum * vw - self.lr * layer.dW
                vb = self.momentum * vb - self.lr * layer.db
                self.velocity[(i, "W")] = vw
                self.velocity[(i, "b")] = vb
                layer.W += vw
                layer.b += vb
            else:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


class Adam:
    name = "adam"

    def __init__(self, lr=0.001, learning_rate=None, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = float(learning_rate if learning_rate is not None else lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W") or layer.dW is None:
                continue
            for name, grad in (("W", layer.dW), ("b", layer.db)):
                param = getattr(layer, name)
                key = (i, name)
                m = self.m.get(key, np.zeros_like(param))
                v = self.v.get(key, np.zeros_like(param))
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
                self.m[key] = m
                self.v[key] = v
                m_hat = m / (1.0 - self.beta1 ** self.t)
                v_hat = v / (1.0 - self.beta2 ** self.t)
                setattr(layer, name, param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


def get_optimizer(value="adam", learning_rate=None):
    if hasattr(value, "step"):
        return value
    name = canonical_name(value or "adam", OPTIMIZERS.keys(), "optimizer")
    if name not in OPTIMIZERS:
        raise UnknownOptimizerError(
            f"Unknown optimizer '{value}'. Available: {', '.join(OPTIMIZERS)}."
        )
    cls = OPTIMIZERS[name]
    kwargs = {}
    if learning_rate is not None:
        kwargs["learning_rate"] = learning_rate
    return cls(**kwargs)
