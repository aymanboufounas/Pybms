import numpy as np
import pybms

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

model = pybms.easy(X, y, epochs=500, verbose=0)
model.summary()
print(model.predict(X, classes=True))
