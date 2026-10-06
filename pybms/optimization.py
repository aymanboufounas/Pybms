"""Inspectable gradient descent and numerical learning utilities."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._registry import api
from ._shared import finite_array, positive_int
from .exceptions import ConfigurationError, ShapeError, TrainingError
from .losses import get_loss


@dataclass
class OptimizeResult:
    """Final parameters, objective trace, convergence status, and update count."""

    x: np.ndarray
    loss: float
    history: list[float]
    converged: bool
    steps: int
    message: str


def _value(fn, x):
    value = np.asarray(fn(x))
    if value.ndim != 0 or not np.isfinite(value):
        raise TrainingError("Objective must return one finite scalar.")
    return float(value)


@api("Optimization", "result = pb.gradient_descent(lambda x: (x ** 2).sum(), [3.0, -2.0])")
def gradient_descent(
    objective,
    start,
    *,
    gradient=None,
    method="sgd",
    learning_rate="auto",
    steps=1000,
    tolerance=1e-8,
):
    """Minimize a scalar objective with SGD or Adam and optional analytical gradient.

    start is a finite numeric array/scalar. Without gradient use central finite
    differences. learning_rate='auto' uses backtracking on the objective;
    convergence means gradient norm <= tolerance, not global optimality.
    """
    positive_int(steps, "steps")
    if method not in {"sgd", "adam"}:
        raise ConfigurationError("method must be sgd or adam.")
    automatic = isinstance(learning_rate, str) and learning_rate == "auto"
    if not automatic and (
        not isinstance(learning_rate, (int, float))
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ConfigurationError("learning_rate must be auto or a positive finite number.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ConfigurationError("tolerance must be positive and finite.")
    x = finite_array(start).copy()
    history = [_value(objective, x)]
    m, v, converged, count = np.zeros_like(x), np.zeros_like(x), False, 0
    message = "maximum steps reached"
    for step in range(1, steps + 1):
        g = finite_array(gradient(x) if gradient is not None else numerical_gradient(objective, x))
        if g.shape != x.shape:
            raise ShapeError(f"Gradient shape {g.shape} must match parameters {x.shape}.")
        if np.linalg.norm(g.reshape(-1)) <= tolerance:
            converged, message = True, "gradient tolerance reached"
            break
        direction = g
        if method == "adam":
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g**2
            direction = (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        rate = (1.0 if method == "sgd" else 0.05) if automatic else float(learning_rate)
        if automatic:
            slope = float(np.sum(g * direction))
            if slope <= 0:
                direction, slope = g, float(np.sum(g**2))
            accepted = False
            for _ in range(40):
                candidate = x - rate * direction
                try:
                    loss = _value(objective, candidate)
                except (TrainingError, FloatingPointError, OverflowError):
                    loss = float("inf")
                if loss <= history[-1] - 1e-4 * rate * slope:
                    accepted = True
                    break
                rate *= 0.5
            if not accepted:
                message = "backtracking could not find a decreasing step"
                break
        else:
            candidate = x - rate * direction
            loss = _value(objective, candidate)
        x, count = candidate, step
        history.append(loss)
    if not converged:
        final_g = finite_array(
            gradient(x) if gradient is not None else numerical_gradient(objective, x)
        )
        if final_g.shape != x.shape:
            raise ShapeError("Gradient shape must match parameters.")
        converged = bool(np.linalg.norm(final_g.reshape(-1)) <= tolerance)
        if converged:
            message = "gradient tolerance reached"
    return OptimizeResult(x, history[-1], history, converged, count, message)


@api("Optimization", "g = pb.numerical_gradient(lambda x: (x ** 2).sum(), [2.0, 3.0])")
def numerical_gradient(objective, values, *, epsilon=1e-5):
    """Central-difference gradient of a scalar objective, preserving parameter shape."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ConfigurationError("epsilon must be positive and finite.")
    x = finite_array(values).copy()
    result = np.empty_like(x)
    for index in np.ndindex(x.shape):
        high, low = x.copy(), x.copy()
        high[index] += epsilon
        low[index] -= epsilon
        result[index] = (_value(objective, high) - _value(objective, low)) / (2 * epsilon)
    return result


@api("Optimization", "report = pb.check_gradient(lambda x: (x ** 2).sum(), lambda x: 2*x, [2.0])")
def check_gradient(objective, gradient, values, *, epsilon=1e-5, tolerance=1e-4):
    """Compare analytical/numerical gradients; return relative error and passed flag."""
    x = finite_array(values)
    analytical = finite_array(gradient(x))
    numerical = numerical_gradient(objective, x, epsilon=epsilon)
    if analytical.shape != numerical.shape:
        raise ShapeError("Analytical gradient shape must match the parameters.")
    error = np.linalg.norm((analytical - numerical).reshape(-1)) / max(
        1e-12, np.linalg.norm(analytical.reshape(-1)) + np.linalg.norm(numerical.reshape(-1))
    )
    return {
        "passed": bool(error <= tolerance),
        "relative_error": float(error),
        "analytical": analytical,
        "numerical": numerical,
    }


@api("Optimization", "rate = pb.learning_rate_schedule(0.01, epoch=20, mode='cosine', total=100)")
def learning_rate_schedule(initial, epoch, *, mode="exponential", decay=0.95, total=100):
    """Return a constant/exponential/cosine learning rate; epoch is zero-based."""
    if (
        not np.isfinite(initial)
        or initial <= 0
        or not isinstance(epoch, (int, np.integer))
        or epoch < 0
    ):
        raise ConfigurationError("initial must be positive and epoch a nonnegative integer.")
    if mode == "constant":
        return float(initial)
    if mode == "exponential":
        if not 0 < decay <= 1:
            raise ConfigurationError("decay must be in (0,1].")
        return float(initial * decay**epoch)
    if mode == "cosine":
        positive_int(total, "total")
        return float(initial * (1 + np.cos(np.pi * min(epoch, total) / total)) / 2)
    raise ConfigurationError("mode must be constant, exponential, or cosine.")


@api("Optimization", "clipped = pb.clip_gradients([pb.array([3, 4])], max_norm=1)")
def clip_gradients(gradients, *, max_norm=1.0):
    """Clip one gradient array or a sequence using their joint L2 norm; return copies."""
    if not np.isfinite(max_norm) or max_norm <= 0:
        raise ConfigurationError("max_norm must be positive and finite.")
    sequence = isinstance(gradients, (list, tuple))
    arrays = [finite_array(g) for g in gradients] if sequence else [finite_array(gradients)]
    norm = np.sqrt(sum(float(np.sum(g**2)) for g in arrays))
    scale = min(1.0, max_norm / max(norm, 1e-12))
    clipped = [g * scale for g in arrays]
    return (tuple(clipped) if isinstance(gradients, tuple) else clipped) if sequence else clipped[0]


@api("Optimization", "weights = pb.initialize_weights((4, 8), method='he')")
def initialize_weights(shape, *, method="xavier", seed=42):
    """Create a 2D weight matrix using Xavier/He normal initialization or zeros."""
    if len(shape) != 2:
        raise ShapeError("Weight shape must be (input_features, output_units).")
    for dimension in shape:
        positive_int(dimension, "shape dimension")
    if method == "zeros":
        return np.zeros(shape)
    if method not in {"xavier", "he"}:
        raise ConfigurationError("method must be xavier, he, or zeros.")
    scale = np.sqrt(2 / sum(shape)) if method == "xavier" else np.sqrt(2 / shape[0])
    return np.random.default_rng(seed).normal(size=shape) * scale


@api("Optimization", "p = pb.sigmoid([-1000, 0, 1000])")
def sigmoid(values):
    """Stable elementwise sigmoid of finite scalar/array values."""
    x = finite_array(values)
    positive = x >= 0
    z = np.exp(-np.abs(x))
    return np.where(positive, 1 / (1 + z), z / (1 + z))


@api("Optimization", "a = pb.relu([-1, 0, 2])")
def relu(values):
    """Rectified linear activation: max(0, value), preserving array shape."""
    return np.maximum(0, finite_array(values))


@api("Optimization", "p = pb.softmax([[1, 2, 3]])")
def softmax(values, *, axis=-1):
    """Stable softmax normalized along axis (defaults to last dimension)."""
    x = finite_array(values)
    if x.ndim == 0:
        raise ShapeError("Softmax needs an array with at least one dimension.")
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


@api("Optimization", "loss = pb.loss_value([1, 2], [1.1, 1.9], name='mse')")
def loss_value(y_true, y_pred, *, name="mse"):
    """Compute mse/mae/binary_crossentropy/categorical_crossentropy on equal-shaped arrays.

    For categorical loss supply 2D one-hot targets; for cross-entropy supply
    probabilities in [0,1] rather than logits.
    """
    actual, predicted = finite_array(y_true), finite_array(y_pred)
    if actual.shape != predicted.shape:
        raise ShapeError("Target and prediction shapes must match exactly.")
    canonical, fn, _ = get_loss(name)
    if "crossentropy" in canonical:
        if np.any((predicted < 0) | (predicted > 1)) or np.any((actual < 0) | (actual > 1)):
            raise ConfigurationError("Cross-entropy expects targets and probabilities in [0,1].")
        if canonical == "categorical_crossentropy" and (
            actual.ndim != 2
            or not np.allclose(actual.sum(axis=1), 1)
            or not np.allclose(predicted.sum(axis=1), 1)
        ):
            raise ShapeError(
                "Categorical targets/probabilities must be 2D and each row must sum to 1."
            )
    return fn(actual, predicted)
