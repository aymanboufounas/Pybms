"""Pybms - beginner-first deep learning in pure Python + NumPy."""

from .model import Model, Sequential, easy, History
from .layers import Dense
from .optimizers import SGD, Adam
from .device import device
from .exceptions import *

__version__ = "0.1.0"


def about():
    print(f"Pybms {__version__} - simple, forgiving deep learning for beginners")
    print("Backend: NumPy / CPU")


__all__ = [
    "Model", "Sequential", "Dense", "SGD", "Adam", "easy", "History",
    "device", "about", "__version__"
]
