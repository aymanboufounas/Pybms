import numpy as np
import pybms as pb


def test_summary_params():
    model = pb.Model()
    model.add(pb.Dense(3, input_dim=2, activation="relu"))
    model.add(pb.Dense(1, activation="sigmoid"))
    model.build(2)
    assert model.count_params() == 13


def test_xor_learns():
    np.random.seed(7)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = pb.Model()
    model.add(8, activation="tanh")
    model.add(1, activation="sigmoid")
    model.compile("adam", "binary_crossentropy", learning_rate=0.03)
    model.fit(X, y, epochs=1200, batch_size=4, verbose=0, shuffle=True)
    pred = model.predict(X, classes=True)
    assert np.mean(pred == y) >= 0.75


def test_auto_model_runs():
    np.random.seed(3)
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = pb.Model(task="regression")
    model.fit(X, y, epochs=5, verbose=0)
    assert model.count_params() > 0
    assert model.predict(X).shape == (4, 1)
