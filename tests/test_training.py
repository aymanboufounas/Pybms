import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
import pybms as pb


def test_auto_classification_and_reproducibility():
    df = pb.sample_data("iris")
    result = pb.auto_train(df, trials=1, cv=2, seed=7)
    repeated = pb.auto_train(df, trials=1, cv=2, seed=7)
    assert len(result.leaderboard) == 4
    assert result.metrics["accuracy"] >= 0.8
    assert result.estimator == repeated.estimator and result.params == repeated.params
    np.testing.assert_array_equal(pb.predict(result, df), repeated.predict(df))
    assert pb.predict(result, df, probabilities=True).shape == (150, 3)
    assert pb.evaluate(result, df)["accuracy"] > 0.8
    assert pb.explain_model(result)["train_rows"] == 120


def test_holdout_not_used_for_imputation():
    df = pb.regression_data(rows=40, features=2, seed=8)
    train_index, test_index = train_test_split(np.arange(len(df)), test_size=0.2, random_state=7)
    df.loc[test_index, "feature_0"] = 1_000_000
    df.loc[train_index[0], "feature_0"] = np.nan
    result = pb.train(df, task="regression", cv=2, trials=1, seed=7)
    imputer = (
        result.model.named_steps["prepare"].named_transformers_["numeric"].named_steps["impute"]
    )
    assert imputer.statistics_[0] == pytest.approx(df.loc[train_index, "feature_0"].median())
    assert imputer.statistics_[0] != pytest.approx(df["feature_0"].median())


def test_mixed_categories_and_unseen_values():
    df = pd.DataFrame(
        {
            "number": [1.0, 2.0, 3.0, np.nan] * 10,
            "city": [" A ", "B", None, "A"] * 10,
            "empty": [np.nan] * 40,
            "target": ["yes", "no", "yes", "no"] * 10,
        }
    )
    result = pb.train(df, trials=1, cv=2)
    new = pd.DataFrame({"number": [5.0], "city": ["never_seen"], "empty": [None]})
    assert result.predict(new)[0] in {"yes", "no"}
    with pytest.raises(pb.DataError):
        result.predict(new.drop(columns="city"))
    assert df.loc[0, "city"] == " A "


@pytest.mark.parametrize("estimator", ["linear", "forest", "extra_trees", "sgd"])
def test_regression_estimators(estimator):
    df = pb.regression_data(rows=100, features=3, seed=7, noise=0.01)
    result = pb.train(df, task="regression", estimator=estimator, trials=1, cv=2)
    assert result.task == "regression" and np.isfinite(result.metrics["rmse"])
    assert len(result.predict(df)) == len(df)


def test_compare_tune_cv_and_helpers():
    df = pb.sample_data()
    assert len(pb.compare_models(df, estimators=["linear", "sgd"], cv=2, trials=1)) == 2
    result = pb.tune_model(df, estimator="forest", params={"max_depth": [2]}, cv=2, trials=1)
    assert result.params["model__max_depth"] == 2
    report = pb.cross_validate(df, cv=3)
    assert len(report["scores"]) == 3 and report["mean"] > 0.8
    importance = pb.feature_importance(result, df, repeats=2)
    assert len(importance) == 4 and set(importance["feature"]) == set(result.features)
    params = pb.recommend_params(df)
    assert params["task"] == "classification" and params["optimizer"] == "adam"
    assert pb.infer_task([1.1, 2.2, 3.3]) == "regression"
    assert pb.infer_task([0, 1, 0]) == "classification"


def test_joblib_roundtrip_and_trust_gate(tmp_path):
    df = pb.sample_data()
    result = pb.train(df, trials=1, cv=2)
    path = pb.save_model(result, tmp_path / "model.joblib")
    with pytest.raises(pb.ConfigurationError):
        pb.load_model(path)
    restored = pb.load_model(path, trusted=True)
    np.testing.assert_array_equal(result.predict(df), restored.predict(df))
    assert restored.metrics == result.metrics


def test_train_arrays_and_integer_regression_override():
    X = np.arange(60).reshape(30, 2)
    y = np.arange(30)
    result = pb.train(X, y, task="regression", trials=1, cv=2)
    assert result.predict(X).shape == (30,)


@pytest.mark.parametrize(
    "call",
    [
        lambda: pb.train({"x": [1, 2]}),
        lambda: pb.train({"x": [1, 2, 3], "target": [0, 1, None]}),
        lambda: pb.train({"x": list(range(10)), "target": [1] * 10}),
        lambda: pb.train(pb.sample_data(), estimator="unknown"),
        lambda: pb.train(pb.sample_data(), cv=1),
        lambda: pb.auto_train(pb.sample_data(), estimator="linear"),
        lambda: pb.infer_task([]),
    ],
)
def test_training_errors(call):
    with pytest.raises(pb.PyBMSError):
        call()
