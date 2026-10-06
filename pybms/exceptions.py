class PyBMSError(Exception):
    """Base exception for Pybms."""


class ConfigurationError(PyBMSError):
    """An option, task, optimizer parameter or persistence choice is invalid."""


class DataError(PyBMSError):
    """A table, target, numeric array, file or URL cannot be used as supplied."""


class ShapeError(PyBMSError):
    """Array dimensions or target/prediction shapes do not match."""


class InvalidLayerError(PyBMSError):
    """A layer specification is not supported by the native dense model."""


class UnknownActivationError(PyBMSError):
    """An activation is not among linear/relu/sigmoid/tanh/softmax."""


class UnknownLossError(PyBMSError):
    """A loss is not among mse/mae/binary/categorical cross-entropy."""


class UnknownOptimizerError(PyBMSError):
    """An optimizer is not SGD/Adam or an object with a step method."""


class ModelNotBuiltError(PyBMSError):
    """Build/add/fit is needed before the requested native model operation."""


class ModelNotCompiledError(PyBMSError):
    """Compile or fit is required before evaluating loss."""


class TrainingError(PyBMSError):
    """Optimization diverged, training arguments are invalid, or all candidates failed."""


class NotSupportedError(PyBMSError):
    """A feature or file format is unsupported, or an optional reader is missing."""
