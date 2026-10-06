import numpy as np
import pytest
import pybms as pb


def test_array_construction_and_shapes():
    assert pb.array([1, 2]).dtype == float
    assert pb.zeros((2, 3)).sum() == 0
    assert pb.ones((2, 3)).sum() == 6
    assert pb.reshape([1, 2, 3, 4], (2, 2)).shape == (2, 2)
    assert pb.flatten([[1, 2], [3, 4]]).tolist() == [1, 2, 3, 4]
    np.testing.assert_array_equal(pb.random_array((2, 3), seed=7), pb.random_array((2, 3), seed=7))
    uniform = pb.random_array((2, 3), distribution="uniform")
    assert np.all((uniform >= 0) & (uniform < 1))
    encoded, labels = pb.one_hot_encode(["b", "a", "b"], return_labels=True)
    np.testing.assert_array_equal(labels[np.argmax(encoded, axis=1)], ["b", "a", "b"])


def test_array_math_and_constant_slices():
    np.testing.assert_array_equal(pb.safe_divide([1, 2], [0, 2], default=-1), [-1, 1])
    np.testing.assert_allclose(pb.moving_average([1, 2, 3, 4], 2), [1.5, 2.5, 3.5])
    values = [[1, 5], [2, 5], [3, 5]]
    scaled = pb.standardize(values)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(scaled[:, 1], 0)
    np.testing.assert_array_equal(pb.minmax(values)[:, 0], [0, 0.5, 1])
    np.testing.assert_allclose(pb.correlation([[1, 2], [2, 4], [3, 6]]), 1)
    assert pb.distance([0, 0], [3, 4]) == 5
    assert pb.distance([0, 0], [3, 4], metric="manhattan") == 7
    assert pb.distance([1, 0], [0, 1], metric="cosine") == 1
    indices, top = pb.top_k([2, 4, 4, 1], 2)
    assert indices.tolist() == [1, 2] and top.tolist() == [4, 4]
    assert pb.array_stats([1, 2, 3])["mean"] == 2


@pytest.mark.parametrize(
    "call",
    [
        lambda: pb.reshape([1, 2], (3, 1)),
        lambda: pb.moving_average([1], 2),
        lambda: pb.top_k([1], 2),
        lambda: pb.standardize([np.nan]),
        lambda: pb.distance([0, 0], [1, 1], metric="cosine"),
    ],
)
def test_array_bad_inputs(call):
    with pytest.raises(pb.PyBMSError):
        call()


def test_gradient_estimation_and_check():
    def objective(x):
        return float(np.sum(x**2 + 2 * x))

    def gradient(x):
        return 2 * x + 2

    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(
        pb.numerical_gradient(objective, values), gradient(values), rtol=1e-7
    )
    assert pb.check_gradient(objective, gradient, values)["passed"]
    assert not pb.check_gradient(objective, lambda x: 3 * x, values)["passed"]
    np.testing.assert_allclose(pb.numerical_gradient(lambda x: x**2, 3.0), 6)


@pytest.mark.parametrize("method", ["sgd", "adam"])
def test_gradient_descent_converges_and_decreases(method):
    result = pb.gradient_descent(
        lambda x: np.sum((x - 2) ** 2),
        [8.0, -4.0],
        gradient=lambda x: 2 * (x - 2),
        method=method,
        steps=2000,
        tolerance=1e-5,
    )
    assert result.converged
    np.testing.assert_allclose(result.x, 2, atol=1e-4)
    assert np.all(np.diff(result.history) <= 1e-9)
    assert result.loss < 1e-8


def test_fixed_step_descent_and_finite_difference():
    result = pb.gradient_descent(lambda x: np.sum(x**2), [2.0], learning_rate=0.1, steps=200)
    assert result.loss < 1e-10 and result.steps > 0
    with pytest.raises(pb.TrainingError):
        pb.gradient_descent(lambda x: np.nan, [1.0])
    with pytest.raises(pb.ShapeError):
        pb.gradient_descent(lambda x: np.sum(x**2), [1.0, 2.0], gradient=lambda x: np.ones(3))


def test_schedules_gradient_clipping_and_initialization():
    assert pb.learning_rate_schedule(0.1, 0) == 0.1
    assert pb.learning_rate_schedule(0.1, 100, mode="cosine", total=100) == 0
    assert pb.learning_rate_schedule(0.1, 5, mode="constant") == 0.1
    gradients = [np.array([3.0, 4.0]), np.array([0.0])]
    clipped = pb.clip_gradients(gradients, max_norm=1)
    assert np.sqrt(sum(np.sum(x * x) for x in clipped)) == pytest.approx(1)
    np.testing.assert_array_equal(gradients[0], [3, 4])
    assert pb.initialize_weights((4, 8)).shape == (4, 8)
    assert pb.initialize_weights((4, 8), method="he").std() > 0
    assert pb.initialize_weights((4, 8), method="zeros").sum() == 0


def test_activations_and_loss_validation():
    np.testing.assert_allclose(pb.sigmoid([-1000, 0, 1000]), [0, 0.5, 1])
    np.testing.assert_array_equal(pb.relu([-1, 0, 2]), [0, 0, 2])
    p = pb.softmax([[1000, 1001, 1002], [-1000, -1001, -1002]])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert pb.loss_value([1.0, 2.0], [2.0, 4.0]) == 2.5
    assert pb.loss_value([1.0, 2.0], [2.0, 4.0], name="mae") == 1.5
    assert pb.loss_value([1.0, 0.0], [0.8, 0.2], name="bce") > 0
    assert pb.loss_value([[1.0, 0.0]], [[0.8, 0.2]], name="cce") > 0
    with pytest.raises(pb.ShapeError):
        pb.loss_value([1, 2], [[1], [2]])
    with pytest.raises(pb.ConfigurationError):
        pb.loss_value([1, 0], [5, -2], name="bce")
