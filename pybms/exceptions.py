class PyBMSError(Exception):
    """Base exception for Pybms."""


class ConfigurationError(PyBMSError):
    pass


class DataError(PyBMSError):
    pass


class ShapeError(PyBMSError):
    pass


class InvalidLayerError(PyBMSError):
    pass


class UnknownActivationError(PyBMSError):
    pass


class UnknownLossError(PyBMSError):
    pass


class UnknownOptimizerError(PyBMSError):
    pass


class ModelNotBuiltError(PyBMSError):
    pass


class ModelNotCompiledError(PyBMSError):
    pass


class TrainingError(PyBMSError):
    pass


class NotSupportedError(PyBMSError):
    pass
