import numpy as np
import pybms as pb

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

model = pb.Model()
model.add(8, activation="tanh")
model.add(1, activation="sigmoid")
model.compile(optimizer="adam", loss="binary_crossentropy", learning_rate=0.03)
model.fit(X, y, epochs=1500, verbose=0)

model.summary()
print("Probabilities:")
print(model.predict(X))
print("Classes:", model.predict(X, classes=True))
