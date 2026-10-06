import numpy as np
import pytest
import pybms as pb
from pybms.losses import get_loss


@pytest.mark.parametrize(
    "activation, loss, y",
    [
        ("linear", "mse", [[0.2, 0.4], [0.3, 0.7]]),
        ("tanh", "mae", [[0.2, 0.4], [0.3, 0.7]]),
        ("sigmoid", "binary_crossentropy", [[0, 1], [1, 0]]),
        ("softmax", "categorical_crossentropy", [[0, 1], [1, 0]]),
        ("relu", "mse", [[0.2, 0.4], [0.3, 0.7]]),
    ],
)
def test_native_backprop_matches_finite_difference(activation, loss, y):
    layer = pb.Dense(2, activation=activation, input_dim=3)
    layer.W[:] = [[0.2, 0.3], [0.1, 0.2], [0.3, 0.1]]
    layer.b[:] = 0.1
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    y = np.array(y, dtype=float)
    _, loss_fn, loss_grad = get_loss(loss)
    p = layer.forward(X, training=True)
    layer.backward(loss_grad(y, p))
    analytical = layer.dW.copy()
    original = layer.W.copy()

    def objective(weights):
        layer.W = weights
        return loss_fn(y, layer.forward(X))

    numerical = pb.numerical_gradient(objective, original)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)


def test_native_seed_and_auto_output_repair():
    X, y = np.array([[0], [1], [2], [3]]), np.array([0, 0, 1, 1])

    def fit():
        network = pb.Model([pb.Dense(4, activation="tanh"), pb.Dense(1)], seed=12)
        with pytest.warns(RuntimeWarning):
            network.fit(X, y, epochs=5, verbose=0)
        return network

    a, b = fit(), fit()
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_compile_before_task_inference_uses_correct_loss():
    network = pb.Model(seed=42).compile()
    network.fit([[0], [1], [2], [3]], [0, 0, 1, 1], epochs=2, verbose=0)
    assert network.loss_name == "binary_crossentropy"


def test_native_string_labels_validation_and_roundtrip(tmp_path):
    X = np.repeat([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], 10, axis=0)
    y = np.repeat(["no", "yes", "yes", "no"], 10)
    network = pb.fit_network(
        X, y, epochs=25, seed=7, verbose=0, validation_split=0.2, patience=5, clipnorm=1
    )
    assert len(network.history_["val_loss"]) == len(network.history_["loss"])
    metrics = network.evaluate(X, y, verbose=0)
    assert np.isfinite(metrics["loss"])
    path = pb.save_model(network, tmp_path / "network.npz")
    restored = pb.load_model(path)
    np.testing.assert_array_equal(network.predict(X), restored.predict(X))
    np.testing.assert_array_equal(
        network.predict(X, classes=True), restored.predict(X, classes=True)
    )
    assert restored.evaluate(X, y, verbose=0) == metrics
    assert pb.explain_model(restored)["backend"] == "NumPy CPU"
    with pytest.raises(pb.DataError):
        restored.evaluate(X, ["unknown"] * len(X), verbose=0)


def test_multiclass_strings_and_multioutput_regression():
    X = np.arange(36, dtype=float).reshape(12, 3) / 36
    labels = np.array(["a", "b", "c"] * 4)
    network = pb.fit_network(X, labels, epochs=2, seed=2, verbose=0)
    assert network.task == "multiclass" and network.predict(X).shape == (12, 3)
    assert "accuracy" in pb.evaluate(network, X, labels)
    regression = pb.fit_network(
        X, np.column_stack([X[:, 0], X[:, 1]]), epochs=2, task="regression", verbose=0
    )
    assert regression.predict(X).shape == (12, 2)
    with pytest.raises(pb.DataError):
        pb.fit_network(X, np.full(12, np.nan), epochs=1, verbose=0)


def test_early_stopping_restores_best_validation_state():
    network = pb.build_network(1, hidden=(), task="regression", learning_rate=0.001, seed=7)
    history = network.fit(
        [[1.0], [2.0], [3.0]],
        [1.0, 2.0, 3.0],
        epochs=20,
        verbose=0,
        validation_data=([[1.0], [2.0]], [1.0, 2.0]),
        patience=2,
        min_delta=1e6,
    )
    assert len(history["loss"]) == 3
    assert network.evaluate([[1.0], [2.0]], [1.0, 2.0], verbose=0)["loss"] == pytest.approx(
        history["val_loss"][0]
    )


def test_strict_methods_and_invalid_native_options():
    model = pb.Model(strict=True)
    with pytest.raises(AttributeError):
        model.fitt
    with pytest.raises(pb.ConfigurationError):
        pb.Adam(learning_rate=-1)
    with pytest.raises(pb.ConfigurationError):
        pb.SGD(momentum=2)
    with pytest.raises(pb.ModelNotCompiledError):
        model.evaluate([[1]], [1])
    with pytest.raises(pb.TrainingError):
        model.fit([[1]], [1], epochs=1.5)


def test_original_typo_friendly_api():
    model = pb.Model(seed=7)
    with pytest.warns(RuntimeWarning):
        model.add("dence", unit=4, activaton="reul")
        model.add(1, activaton="sigmod")
        model.complie(optmizer="adm", los="bce", lern_rate=0.01)
        model.fitt([[0], [1], [2], [3]], [0, 0, 1, 1], epocs=2, verbose=0)
    assert model.count_params() > 0
