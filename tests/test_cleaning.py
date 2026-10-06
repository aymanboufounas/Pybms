import numpy as np
import pandas as pd
import pytest
import pybms as pb


def test_cleaning_preserves_target_and_input():
    original = pd.DataFrame(
        {
            "price": [" 2 ", "4", "", "4"],
            "city": [" A ", "B", "", "B"],
            "target": [0.0, 1.0, np.nan, 1.0],
        }
    )
    before = original.copy(deep=True)
    cleaned, report = pb.clean_data(original, target="target", report=True)
    pd.testing.assert_frame_equal(original, before)
    assert len(cleaned) == 3 and report["dropped_duplicates"] == 1
    assert cleaned["price"].tolist() == [2, 4, 3]
    assert cleaned["city"].tolist() == ["A", "B", "A"]
    assert cleaned["target"].isna().sum() == 1


def test_clean_names_collisions():
    names = pb.clean_names(pd.DataFrame([[1, 2, 3]], columns=[" A ", "a!", "a_2"]))
    assert names.columns.tolist() == ["a", "a_2", "a_2_2"]


@pytest.mark.parametrize(
    "strategy, expected", [("auto", 3), ("median", 3), ("mean", 3), ("mode", 2), ("constant", 9)]
)
def test_imputation(strategy, expected):
    df = pd.DataFrame({"x": [2.0, 4.0, np.nan], "city": ["A", "B", None]})
    result = pb.fill_missing(df, strategy=strategy, value=9)
    assert result.loc[2, "x"] == expected
    assert result.isna().sum().sum() == 0


def test_empty_and_nullable_column_imputation():
    df = pd.DataFrame(
        {
            "all_missing": [np.nan, np.nan],
            "integers": pd.Series([1, None], dtype="Int64"),
            "category": pd.Categorical(["a", None]),
        }
    )
    result = pb.fill_missing(df)
    assert result.isna().sum().sum() == 0
    assert result["all_missing"].tolist() == [0, 0]


def test_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 1, None], "y": ["a", "a", "b"]})
    assert pb.missing_report(df).loc["x", "count"] == 1
    assert len(pb.drop_missing(df, names="x")) == 2
    assert len(pb.drop_duplicates(df)) == 2


def test_coercion_and_strings():
    df = pd.DataFrame(
        {"number": ["1", "bad"], "date": ["2026-10-06", "bad"], "text": [" A ", " B "]}
    )
    assert np.isnan(pb.coerce_numbers(df, names="number").loc[1, "number"])
    assert pd.isna(pb.coerce_dates(df, names="date", format="%Y-%m-%d").loc[1, "date"])
    assert pb.trim_strings(df, names="text", lower=True)["text"].tolist() == ["a", "b"]
    encoded = pb.encode_categories(df[["text"]])
    assert encoded.shape == (2, 3) and encoded.to_numpy().sum() == 2


@pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
def test_scalers_preserve_non_numeric_and_return_scaler(method):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "city": ["A", "B", "C"]})
    scaled, scaler = pb.scale_data(df, method=method, return_scaler=True)
    assert scaled["city"].tolist() == df["city"].tolist()
    np.testing.assert_allclose(scaler.transform(df[["x"]]).ravel(), scaled["x"])
    normalized = pb.normalize_data(df)
    assert normalized["x"].tolist() == [1, 1, 1]


def test_outliers_and_column_reductions():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0], "constant": [1] * 5, "sparse": [None] * 5})
    report = pb.outlier_report(df, names="x")
    assert report.loc["x", "count"] == 1
    clipped = pb.clip_outliers(df, names="x")
    assert clipped["x"].max() == report.loc["x", "upper"]
    assert len(pb.drop_outliers(df, names="x")) == 4
    assert list(pb.drop_constant_columns(df)) == ["x"]
    assert "sparse" not in pb.drop_sparse_columns(df)
    assert pb.replace_values(df, {100.0: 5.0}, names="x")["x"].max() == 5
    assert not pb.validate_data(df)["valid"]
    with pytest.raises(pb.DataError):
        pb.validate_data(df, strict=True)


@pytest.mark.parametrize("method, count", [("over", 6), ("under", 2)])
def test_balancing_is_reproducible(method, count):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 0, 0, 1]})
    balanced = pb.balance_data(df, method=method, seed=7)
    assert len(balanced) == count and balanced["target"].value_counts().nunique() == 1
    pd.testing.assert_frame_equal(balanced, pb.balance_data(df, method=method, seed=7))


@pytest.mark.parametrize(
    "call",
    [
        lambda: pb.fill_missing([[1]], strategy="bad"),
        lambda: pb.scale_data([[1]], method="bad"),
        lambda: pb.drop_sparse_columns([[1]], threshold=2),
        lambda: pb.balance_data({"target": [1, 1]}),
        lambda: pb.clip_outliers([[1]], factor=0),
    ],
)
def test_cleaning_bad_inputs(call):
    with pytest.raises(pb.PyBMSError):
        call()
