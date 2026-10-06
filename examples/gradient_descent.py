"""Inspect optimization of a simple quadratic objective."""

import pybms as pb

result = pb.gradient_descent(lambda x: ((x - 3) ** 2).sum(), [10.0, -5.0])
print("Parameters:", result.x)
print("Loss:", result.loss)
print("Converged:", result.converged, result.message)
