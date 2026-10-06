from __future__ import annotations

import difflib
import warnings
import numpy as np

from .exceptions import (
    ConfigurationError,
    DataError,
    InvalidLayerError,
    ModelNotBuiltError,
    TrainingError,
)
from .layers import Dense
from .losses import get_loss
from .optimizers import get_optimizer
from .utils import ensure_2d_x, infer_task, one_hot, pop_fuzzy


class History(dict):
    @property
    def history(self):
        return self


class Model:
    """Forgiving dense neural-network model."""

    _friendly_methods = {
        "add", "compile", "fit", "predict", "evaluate", "summary",
        "count_params", "info", "build", "auto"
    }

    def __init__(self, layers=None, task="auto", name="PybmsModel", strict=False):
        self.layers = []
        self.name = name
        self.task = task
        self.strict = bool(strict)
        self.optimizer = None
        self.loss_name = None
        self.loss_fn = None
        self.loss_grad = None
        self.compiled = False
        self.input_dim = None
        self.output_dim = None
        self._class_values = None
        self._trained = False
        self._auto_created = False

        if layers:
            for layer in layers:
                self.add(layer)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        match = difflib.get_close_matches(name, self._friendly_methods, n=1, cutoff=0.60)
        if match:
            corrected = match[0]
            warnings.warn(
                f"Pybms auto-corrected method '{name}' -> '{corrected}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            return object.__getattribute__(self, corrected)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Try one of: {', '.join(sorted(self._friendly_methods))}."
        )

    @classmethod
    def auto(cls, X=None, y=None, **kwargs):
        model = cls(task=kwargs.pop("task", "auto"), **kwargs)
        if X is not None and y is not None:
            model._prepare_and_auto_build(X, y)
        return model

    def add(self, layer=None, *args, **kwargs):
        if isinstance(layer, Dense):
            obj = layer
        elif isinstance(layer, int):
            obj = Dense(layer, *args, **kwargs)
        elif isinstance(layer, dict):
            spec = dict(layer)
            kind = str(spec.pop("type", spec.pop("layer", "dense"))).lower()
            if difflib.get_close_matches(kind, ["dense"], n=1, cutoff=0.55):
                obj = Dense(**spec)
            else:
                raise InvalidLayerError("Pybms 0.1 currently supports Dense layers only.")
        elif isinstance(layer, str):
            kind = difflib.get_close_matches(layer.lower(), ["dense"], n=1, cutoff=0.50)
            if not kind:
                raise InvalidLayerError(
                    f"Unknown layer '{layer}'. Pybms 0.1 currently supports Dense."
                )
            if layer.lower() != "dense":
                warnings.warn(
                    f"Pybms auto-corrected layer '{layer}' -> 'dense'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            obj = Dense(*args, **kwargs)
        elif layer is None:
            units = pop_fuzzy(kwargs, "units", aliases=("unit", "neurons", "nodes"))
            if units is None:
                raise InvalidLayerError("Use model.add(8, activation='relu') or model.add(Dense(...)).")
            obj = Dense(units, **kwargs)
        else:
            raise InvalidLayerError(
                "Unsupported layer. Use Dense(...), an integer unit count, a dict, or 'dense'."
            )

        if self.layers and self.layers[-1].built and not obj.built:
            obj.build(self.layers[-1].units)
        self.layers.append(obj)
        return self

    def build(self, input_dim):
        if not self.layers:
            raise ModelNotBuiltError("No layers exist yet. Add layers or call fit(X, y) for auto-build.")
        current = int(input_dim)
        for layer in self.layers:
            if not layer.built:
                layer.build(current)
            elif layer.input_dim != current:
                raise ConfigurationError(
                    f"Layer {layer} expects {layer.input_dim} inputs but previous layer outputs {current}."
                )
            current = layer.units
        self.input_dim = int(input_dim)
        self.output_dim = current
        return self

    def compile(self, optimizer="adam", loss=None, learning_rate=None, **kwargs):
        optimizer = pop_fuzzy(
            kwargs, "optimizer", aliases=("optimzer", "optmizer", "optimiser", "opt"), default=optimizer
        )
        loss = pop_fuzzy(kwargs, "loss", aliases=("los", "losses", "criterion"), default=loss)
        learning_rate = pop_fuzzy(
            kwargs,
            "learning_rate",
            aliases=("lr", "learn_rate", "lern_rate", "learningrate"),
            default=learning_rate,
        )
        if kwargs:
            raise TypeError(f"Unknown compile arguments: {', '.join(kwargs)}")

        if loss is None:
            if self.task == "binary":
                loss = "binary_crossentropy"
            elif self.task == "multiclass":
                loss = "categorical_crossentropy"
            elif self.task == "regression":
                loss = "mse"
            else:
                loss = "mse"

        self.loss_name, self.loss_fn, self.loss_grad = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, learning_rate)
        self.compiled = True
        return self

    def _prepare_xy(self, X, y=None, training=False):
        try:
            X = ensure_2d_x(X)
        except Exception as exc:
            raise DataError(str(exc)) from exc
        if y is None:
            return X, None
        y_arr = np.asarray(y)
        if len(y_arr) != len(X):
            raise DataError(f"X has {len(X)} samples but y has {len(y_arr)}.")

        if training:
            if self.task == "auto":
                self.task = infer_task(y_arr)
            if self.task == "binary":
                values = np.unique(y_arr.reshape(-1))
                if values.size != 2:
                    if self.strict:
                        raise DataError("Binary task requires exactly 2 target classes.")
                    warnings.warn("Target no longer looks binary; switching task automatically.", RuntimeWarning)
                    self.task = infer_task(y_arr)
                else:
                    self._class_values = values
                    mapped = (y_arr.reshape(-1) == values[-1]).astype(float)
                    y_arr = mapped.reshape(-1, 1)
            if self.task == "multiclass":
                if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
                    values = np.unique(y_arr.reshape(-1))
                    self._class_values = values
                    index = {v: i for i, v in enumerate(values)}
                    encoded = np.array([index[v] for v in y_arr.reshape(-1)], dtype=int)
                    y_arr = one_hot(encoded, len(values))
                else:
                    y_arr = y_arr.astype(float)
            elif self.task == "regression":
                y_arr = y_arr.astype(float)
                if y_arr.ndim == 1:
                    y_arr = y_arr.reshape(-1, 1)
        else:
            y_arr = y_arr.astype(float)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
        return X, y_arr

    def _prepare_and_auto_build(self, X, y):
        X, y2 = self._prepare_xy(X, y, training=True)
        self._ensure_architecture(X, y2)
        return X, y2

    def _ensure_architecture(self, X, y):
        input_dim = X.shape[1]
        target_dim = y.shape[1]
        task = self.task

        if not self.layers:
            h1 = max(8, min(128, input_dim * 2))
            h2 = max(4, min(64, h1 // 2))
            self.layers = [Dense(h1, activation="relu")]
            if input_dim >= 4:
                self.layers.append(Dense(h2, activation="relu"))
            if task == "binary":
                self.layers.append(Dense(1, activation="sigmoid"))
            elif task == "multiclass":
                self.layers.append(Dense(target_dim, activation="softmax"))
            else:
                self.layers.append(Dense(target_dim, activation="linear"))
            self._auto_created = True

        self.build(input_dim)
        expected_units = target_dim
        expected_activation = {
            "binary": "sigmoid",
            "multiclass": "softmax",
            "regression": "linear",
        }[task]

        last = self.layers[-1]
        if last.units != expected_units:
            if self.strict:
                raise ConfigurationError(
                    f"Output layer has {last.units} units but target requires {expected_units}."
                )
            warnings.warn(
                f"Pybms fixed output units automatically: {last.units} -> {expected_units}.",
                RuntimeWarning,
                stacklevel=3,
            )
            prev = input_dim if len(self.layers) == 1 else self.layers[-2].units
            self.layers[-1] = Dense(expected_units, activation=expected_activation, input_dim=prev)
            last = self.layers[-1]
        elif last.activation_name != expected_activation and not self.strict:
            warnings.warn(
                f"Pybms changed final activation '{last.activation_name}' -> '{expected_activation}' "
                f"for task '{task}'.",
                RuntimeWarning,
                stacklevel=3,
            )
            prev = input_dim if len(self.layers) == 1 else self.layers[-2].units
            self.layers[-1] = Dense(expected_units, activation=expected_activation, input_dim=prev)

        self.input_dim = input_dim
        self.output_dim = expected_units
        return self

    def _forward(self, X, training=False):
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def _backward(self, grad):
        upstream = grad
        for layer in reversed(self.layers):
            upstream = layer.backward(upstream)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=1, shuffle=True, **kwargs):
        epochs = pop_fuzzy(kwargs, "epochs", aliases=("epoch", "epocs", "epoachs"), default=epochs)
        batch_size = pop_fuzzy(
            kwargs, "batch_size", aliases=("batch", "batchsize", "batch_siz"), default=batch_size
        )
        verbose = pop_fuzzy(kwargs, "verbose", aliases=("verbos", "verbosity"), default=verbose)
        if kwargs:
            raise TypeError(f"Unknown fit arguments: {', '.join(kwargs)}")

        try:
            epochs = int(epochs)
            batch_size = int(batch_size)
        except Exception as exc:
            raise TrainingError("epochs and batch_size must be integers.") from exc
        if epochs <= 0 or batch_size <= 0:
            raise TrainingError("epochs and batch_size must be > 0.")

        X, y = self._prepare_xy(X, y, training=True)
        self._ensure_architecture(X, y)

        if not self.compiled:
            self.compile(loss=None, optimizer="adam")

        if verbose:
            print("\nPybms auto setup")
            print(f"  task: {self.task}")
            print(f"  loss: {self.loss_name}")
            print(f"  optimizer: {getattr(self.optimizer, 'name', type(self.optimizer).__name__)}")
            self.summary()

        history = History(loss=[])
        n = len(X)
        batch_size = min(batch_size, n)

        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(n)
                X_epoch, y_epoch = X[idx], y[idx]
            else:
                X_epoch, y_epoch = X, y

            losses = []
            for start in range(0, n, batch_size):
                xb = X_epoch[start : start + batch_size]
                yb = y_epoch[start : start + batch_size]
                pred = self._forward(xb, training=True)
                loss = self.loss_fn(yb, pred)
                if not np.isfinite(loss):
                    raise TrainingError(
                        "Loss became NaN/Inf. Try a smaller learning rate or inspect your data."
                    )
                grad = self.loss_grad(yb, pred)
                self._backward(grad)
                self.optimizer.step(self.layers)
                losses.append(loss)

            epoch_loss = float(np.mean(losses))
            history["loss"].append(epoch_loss)
            if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        self._trained = True
        return history

    def predict(self, X, classes=False, **kwargs):
        classes = pop_fuzzy(kwargs, "classes", aliases=("class", "labels"), default=classes)
        if kwargs:
            raise TypeError(f"Unknown predict arguments: {', '.join(kwargs)}")
        if not self.layers:
            raise ModelNotBuiltError("Model has no layers. Train it first or add layers.")
        X, _ = self._prepare_xy(X)
        pred = self._forward(X, training=False)
        if not classes:
            return pred
        if self.task == "binary":
            idx = (pred.reshape(-1) >= 0.5).astype(int)
            if self._class_values is not None:
                return np.asarray(self._class_values)[idx]
            return idx
        if self.task == "multiclass":
            idx = np.argmax(pred, axis=1)
            if self._class_values is not None:
                return np.asarray(self._class_values)[idx]
            return idx
        return pred

    def evaluate(self, X, y, verbose=1):
        X, y2 = self._prepare_xy(X, y, training=False)
        if self.task == "binary" and self._class_values is not None:
            y2 = (np.asarray(y).reshape(-1) == self._class_values[-1]).astype(float).reshape(-1, 1)
        elif self.task == "multiclass" and (np.asarray(y).ndim == 1 or np.asarray(y).shape[-1] == 1):
            values = self._class_values if self._class_values is not None else np.unique(y)
            index = {v: i for i, v in enumerate(values)}
            y2 = one_hot(np.array([index[v] for v in np.asarray(y).reshape(-1)]), len(values))
        pred = self._forward(X, training=False)
        loss = self.loss_fn(y2, pred)
        result = {"loss": loss}
        if self.task == "binary":
            result["accuracy"] = float(np.mean((pred.reshape(-1) >= 0.5) == (y2.reshape(-1) >= 0.5)))
        elif self.task == "multiclass":
            result["accuracy"] = float(np.mean(np.argmax(pred, axis=1) == np.argmax(y2, axis=1)))
        if verbose:
            print(result)
        return result

    def count_params(self):
        return int(sum(layer.count_params() for layer in self.layers))

    def summary(self):
        if not self.layers:
            print(f'Model: "{self.name}" (no layers yet)')
            print("Call fit(X, y) and Pybms can build a model automatically.")
            return {"layers": 0, "params": 0}

        print(f'\nModel: "{self.name}"')
        print("=" * 70)
        print(f"{'Layer':<38}{'Output shape':<18}{'Params':>12}")
        print("-" * 70)
        for i, layer in enumerate(self.layers, 1):
            shape = f"(None, {layer.units})"
            params = layer.count_params()
            label = f"{i}. {repr(layer)}"
            print(f"{label:<38}{shape:<18}{params:>12,}")
        print("-" * 70)
        total = self.count_params()
        print(f"Total params: {total:,}")
        print(f"Trainable params: {total:,}")
        print(f"Task: {self.task}")
        print("=" * 70)
        return {"layers": len(self.layers), "params": total, "task": self.task}

    def info(self):
        return {
            "name": self.name,
            "task": self.task,
            "layers": len(self.layers),
            "params": self.count_params(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "compiled": self.compiled,
            "trained": self._trained,
            "loss": self.loss_name,
            "optimizer": getattr(self.optimizer, "name", None),
        }


Sequential = Model


def easy(X, y, epochs=100, **kwargs):
    model = Model.auto()
    model.fit(X, y, epochs=epochs, **kwargs)
    return model
