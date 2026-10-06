"""Pybms: 100 beginner-friendly workflows and a native NumPy dense learner."""

from __future__ import annotations

import warnings
from .model import Model, Sequential, easy, History
from .layers import Dense
from .optimizers import SGD, Adam
from .device import device
from .exceptions import (
    PyBMSError,
    ConfigurationError,
    DataError,
    ShapeError,
    InvalidLayerError,
    UnknownActivationError,
    UnknownLossError,
    UnknownOptimizerError,
    ModelNotBuiltError,
    ModelNotCompiledError,
    TrainingError,
    NotSupportedError,
)
from . import data, cleaning, arrays, plotting, training, optimization, syntax
from .data import DataSplit
from .training import TrainingResult
from .optimization import OptimizeResult
from .syntax import SyntaxResult
from ._registry import REGISTRY

__version__ = "0.2.0"
globals().update({name: info["function"] for name, info in REGISTRY.items()})


def about():
    """Return version/backend/workflow count and print a short description."""
    info = {"version": __version__, "backend": "NumPy / CPU", "workflows": len(REGISTRY)}
    print(f"Pybms {__version__}: {len(REGISTRY)} simple data/ML workflows + native dense networks")
    return info


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    suggestions = syntax.suggest_function(name, limit=2, cutoff=0.82)
    if len(suggestions) == 1:
        corrected = suggestions[0]
        warnings.warn(
            f"Pybms corrected function '{name}' -> '{corrected}'.", RuntimeWarning, stacklevel=2
        )
        return REGISTRY[corrected]["function"]
    raise AttributeError(
        f"Pybms has no function '{name}'. Suggestions: {suggestions}. Use list_functions()."
    )


__all__ = list(REGISTRY) + [
    "Model",
    "Sequential",
    "Dense",
    "SGD",
    "Adam",
    "easy",
    "History",
    "device",
    "about",
    "__version__",
    "TrainingResult",
    "DataSplit",
    "OptimizeResult",
    "SyntaxResult",
    "PyBMSError",
    "ConfigurationError",
    "DataError",
    "ShapeError",
    "InvalidLayerError",
    "UnknownActivationError",
    "UnknownLossError",
    "UnknownOptimizerError",
    "ModelNotBuiltError",
    "ModelNotCompiledError",
    "TrainingError",
    "NotSupportedError",
]
