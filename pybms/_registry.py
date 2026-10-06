"""Metadata for the documented public workflows."""

from __future__ import annotations

from functools import wraps
import difflib
import inspect
import warnings

REGISTRY: dict[str, dict] = {}


def api(category: str, example: str):
    def register(fn):
        signature = inspect.signature(fn)
        explicit = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
        }
        passthrough = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())

        @wraps(fn)
        def friendly(*args, **kwargs):
            for name in list(kwargs):
                if name in explicit or passthrough:
                    continue
                matches = difflib.get_close_matches(name, sorted(explicit), n=2, cutoff=0.75)
                if len(matches) == 1:
                    corrected = matches[0]
                    if corrected in kwargs:
                        raise TypeError(
                            f"Provide '{corrected}' once; both '{name}' and '{corrected}' were supplied."
                        )
                    warnings.warn(
                        f"Pybms corrected parameter '{name}' -> '{corrected}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    kwargs[corrected] = kwargs.pop(name)
            return fn(*args, **kwargs)

        REGISTRY[fn.__name__] = {
            "function": friendly,
            "category": category,
            "example": example,
        }
        return friendly

    return register
