import numpy as np
import pybms as pb

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

# These are intentional, valid-Python typos.
model = pb.Model()
model.add("dence", unit=8, activaton="reul")
model.add(1, activaton="sigmod")
model.complie(optmizer="adm", los="bce", lern_rate=0.03)
model.fitt(X, y, epocs=500, verbose=0)
model.summery()
