from __future__ import annotations

import difflib
import warnings
import copy
import json
import numpy as np

from .exceptions import (
    ConfigurationError,
    DataError,
    InvalidLayerError,
    ModelNotBuiltError,
    ModelNotCompiledError,
    ShapeError,
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
        "add",
        "compile",
        "fit",
        "predict",
        "evaluate",
        "summary",
        "count_params",
        "info",
        "build",
        "auto",
        "save",
    }

    def __init__(self, layers=None, task="auto", name="PybmsModel", strict=False, seed=None):
        if task not in {"auto", "binary", "multiclass", "regression"}:
            raise ConfigurationError("task must be auto, binary, multiclass, or regression.")
        self.layers = []
        self.name = name
        self.task = task
        self.strict = bool(strict)
        self.seed = seed
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.history_ = History(loss=[])
        self._auto_loss = True
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
        if name.startswith("_") or self.strict:
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
                raise InvalidLayerError("Pybms currently supports Dense layers only.")
        elif isinstance(layer, str):
            kind = difflib.get_close_matches(layer.lower(), ["dense"], n=1, cutoff=0.50)
            if not kind:
                raise InvalidLayerError(f"Unknown layer '{layer}'. Pybms currently supports Dense.")
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
                raise InvalidLayerError(
                    "Use model.add(8, activation='relu') or model.add(Dense(...))."
                )
            obj = Dense(units, **kwargs)
        else:
            raise InvalidLayerError(
                "Unsupported layer. Use Dense(...), an integer unit count, a dict, or 'dense'."
            )

        obj.rng = self.rng
        if self.layers and self.layers[-1].built and not obj.built:
            obj.build(self.layers[-1].units)
        self.layers.append(obj)
        return self

    def build(self, input_dim):
        if not self.layers:
            raise ModelNotBuiltError(
                "No layers exist yet. Add layers or call fit(X, y) for auto-build."
            )
        current = int(input_dim)
        for layer in self.layers:
            layer.rng = self.rng
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
            kwargs,
            "optimizer",
            aliases=("optimzer", "optmizer", "optimiser", "opt"),
            default=optimizer,
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

        self._auto_loss = loss is None
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
        if y_arr.ndim == 0 or y_arr.ndim > 2 or not y_arr.size:
            raise DataError("y must contain a nonempty 1D or 2D target array.")
        if len(y_arr) != len(X):
            raise DataError(f"X has {len(X)} samples but y has {len(y_arr)}.")
        import pandas as pd

        if pd.isna(y_arr).any() or (
            np.issubdtype(y_arr.dtype, np.number) and not np.isfinite(y_arr).all()
        ):
            raise DataError("y contains missing or non-finite values.")

        if training:
            if self.task == "auto":
                self.task = infer_task(y_arr)
            if self.task == "binary":
                values = np.unique(y_arr.reshape(-1))
                if values.size != 2:
                    if self.strict:
                        raise DataError("Binary task requires exactly 2 target classes.")
                    warnings.warn(
                        "Target no longer looks binary; switching task automatically.",
                        RuntimeWarning,
                    )
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
            if (
                self.task in {"binary", "multiclass"}
                and self._class_values is not None
                and (y_arr.ndim == 1 or y_arr.shape[1] == 1)
            ):
                index = {value: i for i, value in enumerate(self._class_values)}
                try:
                    encoded = np.array([index[value] for value in y_arr.reshape(-1)], dtype=int)
                except KeyError as exc:
                    raise DataError(f"Unknown target class: {exc.args[0]}") from exc
                y_arr = (
                    encoded.astype(float).reshape(-1, 1)
                    if self.task == "binary"
                    else one_hot(encoded, len(index))
                )
            else:
                y_arr = y_arr.astype(float)
                if y_arr.ndim == 1:
                    y_arr = y_arr.reshape(-1, 1)
        if self.task == "multiclass" and (
            y_arr.ndim != 2
            or not np.all(np.isin(y_arr, [0, 1]))
            or not np.allclose(y_arr.sum(axis=1), 1)
        ):
            raise DataError("Multiclass targets must be labels or one-hot rows.")
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
            self.layers[-1] = Dense(expected_units, activation=expected_activation)
            self.layers[-1].rng = self.rng
            self.layers[-1].build(prev)
            last = self.layers[-1]
        elif last.activation_name != expected_activation and not self.strict:
            warnings.warn(
                f"Pybms changed final activation '{last.activation_name}' -> '{expected_activation}' "
                f"for task '{task}'.",
                RuntimeWarning,
                stacklevel=3,
            )
            prev = input_dim if len(self.layers) == 1 else self.layers[-2].units
            self.layers[-1] = Dense(expected_units, activation=expected_activation)
            self.layers[-1].rng = self.rng
            self.layers[-1].build(prev)

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

    def fit(
        self,
        X,
        y,
        epochs=100,
        batch_size=32,
        verbose=1,
        shuffle=True,
        validation_split=0.0,
        validation_data=None,
        patience=None,
        min_delta=0.0,
        restore_best=True,
        clipnorm=None,
        **kwargs,
    ):
        """Train using mini-batches; return History and store it as history_.

        Optional validation_split or validation_data=(X_val, y_val) enables
        val_loss. patience stops after that many epochs without improvement;
        restore_best restores the weights/optimizer at best validation loss.
        clipnorm clips the global gradient norm before each optimizer update.
        """
        epochs = pop_fuzzy(kwargs, "epochs", aliases=("epoch", "epocs", "epoachs"), default=epochs)
        batch_size = pop_fuzzy(
            kwargs, "batch_size", aliases=("batch", "batchsize", "batch_siz"), default=batch_size
        )
        verbose = pop_fuzzy(kwargs, "verbose", aliases=("verbos", "verbosity"), default=verbose)
        if kwargs:
            raise TypeError(f"Unknown fit arguments: {', '.join(kwargs)}")

        if (
            isinstance(epochs, bool)
            or isinstance(batch_size, bool)
            or not isinstance(epochs, (int, np.integer))
            or not isinstance(batch_size, (int, np.integer))
        ):
            raise TrainingError("epochs and batch_size must be integers.")
        if epochs <= 0 or batch_size <= 0:
            raise TrainingError("epochs and batch_size must be > 0.")

        X, y = self._prepare_xy(X, y, training=True)
        self._ensure_architecture(X, y)
        if not 0 <= validation_split < 1:
            raise ConfigurationError("validation_split must be in [0,1).")
        if validation_data is not None and validation_split:
            raise ConfigurationError("Use validation_split or validation_data, not both.")
        X_val = y_val = None
        if validation_data is not None:
            X_val, y_val = self._prepare_xy(*validation_data, training=False)
        elif validation_split:
            from sklearn.model_selection import train_test_split

            labels = np.argmax(y, axis=1) if self.task == "multiclass" else y.ravel()
            try:
                X, X_val, y, y_val = train_test_split(
                    X,
                    y,
                    test_size=validation_split,
                    random_state=self.seed if self.seed is not None else 42,
                    stratify=labels if self.task in {"binary", "multiclass"} else None,
                )
            except ValueError as exc:
                raise DataError(f"Cannot split validation rows: {exc}") from exc
        if X_val is not None and (X_val.shape[1] != X.shape[1] or y_val.shape[1] != y.shape[1]):
            raise ShapeError("Validation feature/target dimensions must match training data.")
        if patience is not None and (
            isinstance(patience, bool)
            or not isinstance(patience, (int, np.integer))
            or patience <= 0
        ):
            raise ConfigurationError("patience must be a positive integer.")
        if patience is not None and X_val is None:
            raise ConfigurationError("patience needs validation_data or validation_split.")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ConfigurationError("min_delta must be finite and nonnegative.")
        if clipnorm is not None and (not np.isfinite(clipnorm) or clipnorm <= 0):
            raise ConfigurationError("clipnorm must be positive and finite.")

        if not self.compiled:
            self.compile(loss=None, optimizer="adam")
        elif self._auto_loss:
            expected = {
                "binary": "binary_crossentropy",
                "multiclass": "categorical_crossentropy",
                "regression": "mse",
            }[self.task]
            self.loss_name, self.loss_fn, self.loss_grad = get_loss(expected)

        if verbose:
            print("\nPybms auto setup")
            print(f"  task: {self.task}")
            print(f"  loss: {self.loss_name}")
            print(f"  optimizer: {getattr(self.optimizer, 'name', type(self.optimizer).__name__)}")
            self.summary()

        history = History(loss=[])
        if X_val is not None:
            history["val_loss"] = []
        best_loss, best_state, waiting = float("inf"), None, 0
        n = len(X)
        batch_size = min(batch_size, n)

        for epoch in range(epochs):
            if shuffle:
                idx = np.random.permutation(n) if self.rng is None else self.rng.permutation(n)
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
                if clipnorm is not None:
                    gradients = [g for layer in self.layers for g in (layer.dW, layer.db)]
                    norm = np.sqrt(sum(float(np.sum(g**2)) for g in gradients))
                    scale = min(1, clipnorm / max(norm, 1e-12))
                    for layer in self.layers:
                        layer.dW *= scale
                        layer.db *= scale
                self.optimizer.step(self.layers)
                losses.append(loss)

            epoch_loss = self.loss_fn(y, self._forward(X))
            if not np.isfinite(epoch_loss):
                raise TrainingError("Training loss became non-finite after an update.")
            history["loss"].append(epoch_loss)
            if X_val is not None:
                val_loss = self.loss_fn(y_val, self._forward(X_val))
                if not np.isfinite(val_loss):
                    raise TrainingError("Validation loss became non-finite.")
                history["val_loss"].append(val_loss)
                if val_loss < best_loss - min_delta:
                    best_loss, waiting = val_loss, 0
                    best_state = (
                        [(layer.W.copy(), layer.b.copy()) for layer in self.layers],
                        copy.deepcopy(self.optimizer),
                    )
                else:
                    waiting += 1
                if patience is not None and waiting >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1}; best val_loss: {best_loss:.6f}"
                        )
                    break
            if verbose and (
                epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 10) == 0
            ):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        if restore_best and best_state is not None:
            for layer, (W, b) in zip(self.layers, best_state[0]):
                layer.W, layer.b = W, b
            self.optimizer = best_state[1]
        self._trained = True
        self.history_ = history
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
        if not self.compiled:
            raise ModelNotCompiledError("Compile or fit the model before evaluation.")
        X, y2 = self._prepare_xy(X, y, training=False)
        pred = self._forward(X, training=False)
        loss = self.loss_fn(y2, pred)
        result = {"loss": loss}
        if self.task == "binary":
            result["accuracy"] = float(
                np.mean((pred.reshape(-1) >= 0.5) == (y2.reshape(-1) >= 0.5))
            )
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

    def save(self, path):
        """Save JSON metadata and NumPy weights to .npz without pickle.

        Optimizer moment buffers and RNG position are not persisted; resumed
        training starts with fresh optimizer state. Inference is unchanged.
        """
        from pathlib import Path

        path = Path(path)
        if path.suffix != ".npz":
            raise ConfigurationError("Native model files use .npz.")
        if not self.layers or any(not layer.built for layer in self.layers):
            raise ModelNotBuiltError("Build or fit the model before saving.")
        optimizer = (
            None
            if self.optimizer is None
            else {"name": self.optimizer.name, "learning_rate": self.optimizer.lr}
        )
        metadata = {
            "format": "pybms-native-v1",
            "name": self.name,
            "task": self.task,
            "strict": self.strict,
            "seed": self.seed,
            "trained": self._trained,
            "layers": [
                {
                    "units": layer.units,
                    "activation": layer.activation_name,
                    "input_dim": layer.input_dim,
                }
                for layer in self.layers
            ],
            "class_values": None if self._class_values is None else self._class_values.tolist(),
            "loss": self.loss_name,
            "optimizer": optimizer,
            "history": dict(self.history_),
        }
        arrays = {"metadata": np.array(json.dumps(metadata))}
        for i, layer in enumerate(self.layers):
            arrays[f"W_{i}"], arrays[f"b_{i}"] = layer.W, layer.b
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        """Load an array/JSON .npz model with allow_pickle=False."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                if metadata.get("format") != "pybms-native-v1":
                    raise DataError("Unknown native model format.")
                model = cls(
                    task=metadata["task"],
                    name=metadata["name"],
                    strict=metadata["strict"],
                    seed=metadata["seed"],
                )
                for i, spec in enumerate(metadata["layers"]):
                    layer = Dense(**spec)
                    W, b = archive[f"W_{i}"], archive[f"b_{i}"]
                    if (
                        W.shape != (spec["input_dim"], spec["units"])
                        or b.shape != (1, spec["units"])
                        or not np.isfinite(W).all()
                        or not np.isfinite(b).all()
                    ):
                        raise DataError("Invalid saved weight dimensions or values.")
                    layer.W, layer.b = W.copy(), b.copy()
                    model.add(layer)
                model.build(metadata["layers"][0]["input_dim"])
                model._class_values = (
                    None
                    if metadata["class_values"] is None
                    else np.asarray(metadata["class_values"])
                )
                if metadata["optimizer"] is not None:
                    model.compile(
                        optimizer=metadata["optimizer"]["name"],
                        learning_rate=metadata["optimizer"]["learning_rate"],
                        loss=metadata["loss"],
                    )
                model._trained = metadata["trained"]
                model.history_ = History(metadata["history"])
                return model
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise DataError(f"Cannot load native model: {exc}") from exc


Sequential = Model


def easy(X, y, epochs=100, **kwargs):
    """Infer task/architecture, fit a native model, and return it.

    Accepts seed/task/strict and all Model.fit options.
    """
    model = Model.auto(
        seed=kwargs.pop("seed", None),
        task=kwargs.pop("task", "auto"),
        strict=kwargs.pop("strict", False),
    )
    model.fit(X, y, epochs=epochs, **kwargs)
    return model
