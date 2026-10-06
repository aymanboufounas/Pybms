from __future__ import annotations

import platform


def device(verbose=True):
    """Return the current execution device.

    Pybms 0.1 uses NumPy, so computation is CPU-based.
    """
    info = {
        "device": "cpu",
        "backend": "numpy",
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if verbose:
        print("Pybms device: CPU (NumPy backend)")
    return info
