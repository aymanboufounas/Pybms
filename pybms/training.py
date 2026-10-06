"""Small-budget model selection with preprocessing inside every CV fold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

from ._registry import api
from ._shared import frame, positive_int
from .exceptions import ConfigurationError, DataError, TrainingError
from .model import Model


def _normalize_X(data, feature_names=None):
    if feature_names is not None and not isinstance(data, pd.DataFrame):
        try:
            df = pd.DataFrame(data, columns=feature_names)
        except ValueError as exc:
            raise DataError(f"Expected {len(feature_names)} feature columns.") from exc
    else:
        df = frame(data)
        df.columns = df.columns.map(str)
        if not df.columns.is_unique:
            raise DataError("Column names become ambiguous when converted to strings.")
    if feature_names is not None:
        missing = set(feature_names) - set(df.columns)
        if missing:
            raise DataError(f"Prediction data is missing features: {sorted(missing)}")
        df = df.loc[:, feature_names].copy()
    if df.empty:
        raise DataError("Feature data must have rows and columns.")
    for name in df:
        if pd.api.types.is_numeric_dtype(df[name]):
            df[name] = pd.to_numeric(df[name]).astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            df[name] = (
                df[name]
                .map(
                    lambda value: (
                        np.nan
                        if pd.isna(value) or (isinstance(value, str) and not value.strip())
                        else str(value).strip()
                    )
                )
                .astype(object)
            )
    return df


def _xy(data, target):
    df = frame(data)
    if isinstance(target, (str, int, np.integer)):
        if target not in df:
            raise DataError(
                f"Target column '{target}' does not exist. Available: {list(df.columns)}"
            )
        X, y = df.drop(columns=target), df[target].to_numpy()
    else:
        X, y = df, np.asarray(target)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1 or len(y) != len(X) or not len(y):
        raise DataError("Provide one target per row; targets must be 1D and match the features.")
    if pd.isna(y).any():
        raise DataError("Target contains missing values; drop those rows explicitly.")
    if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
        raise DataError("Target must not contain NaN or infinity.")
    return _normalize_X(X), y


@api("Training", "task = pb.infer_task([0, 1, 1, 0])")
def infer_task(y):
    """Infer classification or regression from a 1D target; integer targets classify.

    Integer-valued regression is ambiguous. Pass task='regression' to train()
    explicitly for counts/prices that happen to be whole numbers.
    """
    values = np.asarray(y)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.ravel()
    if values.ndim != 1 or not values.size or pd.isna(values).any():
        raise DataError("Expected nonmissing, nonempty 1D targets.")
    kind = type_of_target(values)
    if kind in {"binary", "multiclass"}:
        return "classification"
    if kind == "continuous":
        return "regression"
    raise DataError(f"Unsupported target type '{kind}'; use a single target column.")


def _task(y, task):
    if task == "auto":
        task = infer_task(y)
    if task not in {"classification", "regression"}:
        raise ConfigurationError("task must be auto, classification, or regression.")
    if task == "classification" and len(np.unique(y)) < 2:
        raise DataError("Classification requires at least two target classes.")
    if task == "regression":
        try:
            if not np.isfinite(y.astype(float)).all():
                raise DataError("Regression targets must be finite.")
        except (TypeError, ValueError) as exc:
            raise DataError("Regression targets must be numeric.") from exc
    return task


def _preprocessor(X):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__missing__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def _candidates(task, seed):
    if task == "classification":
        return {
            "linear": (
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
                {"model__C": [0.01, 0.1, 1, 10]},
            ),
            "forest": (
                RandomForestClassifier(
                    n_estimators=80, class_weight="balanced", n_jobs=1, random_state=seed
                ),
                {"model__max_depth": [None, 5, 12], "model__min_samples_leaf": [1, 3]},
            ),
            "extra_trees": (
                ExtraTreesClassifier(
                    n_estimators=80, class_weight="balanced", n_jobs=1, random_state=seed
                ),
                {"model__max_depth": [None, 5, 12], "model__min_samples_leaf": [1, 3]},
            ),
            "sgd": (
                SGDClassifier(
                    loss="log_loss", max_iter=2000, class_weight="balanced", random_state=seed
                ),
                {"model__alpha": [0.0001, 0.001, 0.01], "model__penalty": ["l2", "l1"]},
            ),
        }
    return {
        "linear": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10]}),
        "forest": (
            RandomForestRegressor(n_estimators=80, n_jobs=1, random_state=seed),
            {"model__max_depth": [None, 5, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "extra_trees": (
            ExtraTreesRegressor(n_estimators=80, n_jobs=1, random_state=seed),
            {"model__max_depth": [None, 5, 12], "model__min_samples_leaf": [1, 3]},
        ),
        "sgd": (
            SGDRegressor(max_iter=2000, random_state=seed),
            {"model__alpha": [0.0001, 0.001, 0.01], "model__penalty": ["l2", "l1"]},
        ),
    }


def _folds(y, task, cv, seed):
    positive_int(cv, "cv")
    if cv < 2:
        raise ConfigurationError("cv must be >= 2.")
    if task == "classification":
        folds = min(cv, int(pd.Series(y).value_counts().min()))
        if folds < 2:
            raise DataError("Need at least two training rows in every class for cross-validation.")
        return StratifiedKFold(folds, shuffle=True, random_state=seed)
    folds = min(cv, len(y))
    if folds < 2:
        raise DataError("Need at least two training rows for cross-validation.")
    return KFold(folds, shuffle=True, random_state=seed)


def _search(X, y, task, names, trials, cv, seed, scoring=None, param_grid=None):
    positive_int(trials, "trials")
    candidates = _candidates(task, seed)
    splitter = _folds(y, task, cv, seed)
    metric = scoring or (
        "balanced_accuracy" if task == "classification" else "neg_root_mean_squared_error"
    )
    rows, fitted = [], {}
    for name in names:
        if name not in candidates:
            raise ConfigurationError(
                f"Unknown estimator '{name}'. Choose {list(candidates)} or auto."
            )
        estimator, space = candidates[name]
        space = param_grid if param_grid is not None else space
        # Parameters without a prefix apply to the estimator, not preprocessing.
        space = {key if "__" in key else f"model__{key}": values for key, values in space.items()}
        pipeline = Pipeline([("prepare", _preprocessor(X)), ("model", estimator)])
        combinations = (
            int(np.prod([len(v) for v in space.values()]))
            if all(isinstance(v, (list, tuple, np.ndarray)) for v in space.values())
            else trials
        )
        search = RandomizedSearchCV(
            pipeline,
            space,
            n_iter=min(trials, max(1, combinations)),
            cv=splitter,
            scoring=metric,
            random_state=seed,
            n_jobs=1,
            error_score="raise",
            refit=True,
        )
        try:
            search.fit(X, y)
            if not np.isfinite(search.best_score_):
                raise TrainingError("Validation produced a non-finite score.")
        except (ValueError, TypeError, TrainingError) as exc:
            rows.append(
                {
                    "estimator": name,
                    "cv_score": np.nan,
                    "cv_std": np.nan,
                    "params": {},
                    "error": str(exc),
                }
            )
            continue
        index = search.best_index_
        rows.append(
            {
                "estimator": name,
                "cv_score": float(search.best_score_),
                "cv_std": float(search.cv_results_["std_test_score"][index]),
                "params": search.best_params_,
                "error": None,
            }
        )
        fitted[name] = search.best_estimator_
    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("cv_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if not fitted:
        raise TrainingError(
            f"All candidate models failed: {leaderboard[['estimator', 'error']].to_dict('records')}"
        )
    return leaderboard, fitted, metric


def _metrics(y, predicted, task):
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "f1_weighted": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        }
    return {
        "mae": float(mean_absolute_error(y, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y, predicted))),
        "r2": float(r2_score(y, predicted)) if len(y) > 1 else float("nan"),
    }


@dataclass
class TrainingResult:
    """Fitted pipeline plus CV leaderboard and an untouched holdout evaluation.

    model remains fitted to the training partition. The holdout is never used
    to choose models or parameters. Extra prediction columns are ignored.
    """

    model: Pipeline
    task: str
    estimator: str
    features: list[str]
    leaderboard: pd.DataFrame
    metrics: dict
    params: dict
    scoring: str
    train_rows: int
    test_rows: int
    seed: int

    def predict(self, data):
        return self.model.predict(_normalize_X(data, self.features))

    def predict_proba(self, data):
        if self.task != "classification" or not hasattr(self.model, "predict_proba"):
            raise ConfigurationError("This estimator does not support class probabilities.")
        return self.model.predict_proba(_normalize_X(data, self.features))

    def evaluate(self, data, target="target"):
        X, y = _xy(data, target)
        return _metrics(y, self.predict(X), self.task)

    def summary(self):
        return {
            "task": self.task,
            "estimator": self.estimator,
            "features": self.features,
            "params": self.params,
            "cv_scoring": self.scoring,
            "holdout_metrics": self.metrics,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
        }


@api("Training", "result = pb.train(df, target='target')")
def train(
    data,
    target="target",
    *,
    task="auto",
    estimator="linear",
    test_size=0.2,
    trials=4,
    cv=3,
    seed=42,
    scoring=None,
    param_grid=None,
):
    """Train a pipeline with a held-out test set and CV hyperparameter search.

    data is a table with target column, or pass train(X, y). estimator is linear,
    forest, extra_trees, sgd, or auto. trials limits configurations per estimator.
    cv limits folds. Default classification scoring is balanced_accuracy;
    regression uses negative RMSE (larger is better). Imputation/encoding/scaling
    fit within each fold. Independent rows are required; grouped/time-series
    datasets need an explicit evaluation strategy outside this convenience API.
    """
    X, y = _xy(data, target)
    task = _task(y, task)
    if len(y) < 6:
        raise DataError(
            "Use at least six rows for held-out CV training; use easy() for tiny neural demos."
        )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y if task == "classification" else None,
        )
    except ValueError as exc:
        raise DataError(f"Cannot create holdout split: {exc}") from exc
    names = list(_candidates(task, seed)) if estimator == "auto" else [estimator]
    leaderboard, fitted, metric = _search(
        X_train, y_train, task, names, trials, cv, seed, scoring, param_grid
    )
    best = leaderboard.iloc[0]
    model = fitted[best["estimator"]]
    metrics = _metrics(y_test, model.predict(X_test), task)
    return TrainingResult(
        model,
        task,
        best["estimator"],
        list(X.columns),
        leaderboard,
        metrics,
        best["params"],
        metric,
        len(y_train),
        len(y_test),
        seed,
    )


@api("Training", "result = pb.auto_train(df, target='target', trials=4)")
def auto_train(data, target="target", **kwargs):
    """Choose among four estimators and their bounded CV search spaces, then evaluate holdout.

    The winner is best among the tested configurations, not a guaranteed global
    optimum. All options except estimator are the same as train().
    """
    if "estimator" in kwargs:
        raise ConfigurationError(
            "auto_train chooses estimators; use train(estimator=...) to select one."
        )
    return train(data, target, estimator="auto", **kwargs)


@api("Training", "table = pb.compare_models(df, target='target', trials=2)")
def compare_models(
    data, target="target", *, task="auto", estimators=None, trials=2, cv=3, seed=42, scoring=None
):
    """Return a CV ranking without a holdout test. Larger cv_score is better.

    Use auto_train() for a final independent test metric. Do not repeatedly
    choose changes based on test-set results.
    """
    X, y = _xy(data, target)
    task = _task(y, task)
    names = list(_candidates(task, seed)) if estimators is None else list(estimators)
    if not names:
        raise ConfigurationError("Provide at least one estimator.")
    return _search(X, y, task, names, trials, cv, seed, scoring)[0]


@api("Training", "result = pb.tune_model(df, estimator='forest', trials=4)")
def tune_model(data, target="target", *, estimator="forest", params=None, **kwargs):
    """Tune one named estimator; params maps parameter names to candidate lists.

    Example: params={'max_depth': [3, 8], 'min_samples_leaf': [1, 3]}.
    The result includes an independent holdout metric like train().
    """
    return train(data, target, estimator=estimator, param_grid=params, **kwargs)


@api("Training", "predictions = pb.predict(result, df)")
def predict(model, data, *, probabilities=False):
    """Predict with a TrainingResult, native Model, or sklearn estimator.

    Native classification returns original class labels by default. Set
    probabilities=True for probability outputs; regression returns predictions.
    """
    if isinstance(model, Model):
        return model.predict(data, classes=not probabilities and model.task != "regression")
    if probabilities:
        if not hasattr(model, "predict_proba"):
            raise ConfigurationError("Model has no predict_proba method.")
        return model.predict_proba(data)
    return model.predict(data)


@api("Training", "metrics = pb.evaluate(result, df, target='target')")
def evaluate(model, data, target="target", *, task="auto"):
    """Evaluate a trained model on user-supplied labeled rows or evaluate(model, X, y).

    For native Model return loss/accuracy. For sklearn return classification
    accuracy/F1 or regression MAE/RMSE/R2. Evaluation does not fit the model.
    """
    if isinstance(model, TrainingResult):
        return model.evaluate(data, target)
    X, y = _xy(data, target)
    if isinstance(model, Model):
        return model.evaluate(X.to_numpy(dtype=float), y, verbose=0)
    return _metrics(y, model.predict(X), _task(y, task))


@api("Training", "report = pb.cross_validate(df, target='target', estimator='linear')")
def cross_validate(
    data, target="target", *, estimator="linear", task="auto", cv=5, seed=42, scoring=None
):
    """Cross-validate one default estimator pipeline; return scores, mean, std, and scoring."""
    X, y = _xy(data, target)
    task = _task(y, task)
    candidates = _candidates(task, seed)
    if estimator not in candidates:
        raise ConfigurationError(f"Unknown estimator '{estimator}'.")
    pipeline = Pipeline([("prepare", _preprocessor(X)), ("model", clone(candidates[estimator][0]))])
    scoring = scoring or (
        "balanced_accuracy" if task == "classification" else "neg_root_mean_squared_error"
    )
    scores = cross_val_score(
        pipeline, X, y, scoring=scoring, cv=_folds(y, task, cv, seed), error_score="raise", n_jobs=1
    )
    return {
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scoring": scoring,
    }


@api("Training", "importance = pb.feature_importance(result, test_df, target='target')")
def feature_importance(model, data, target="target", *, repeats=5, seed=42):
    """Measure original-column permutation importance on supplied labeled evaluation rows.

    Values describe this model/metric on these rows, not causal effects. Prefer
    a separate validation set for interpretation.
    """
    positive_int(repeats, "repeats")
    if not isinstance(model, TrainingResult):
        raise ConfigurationError("feature_importance expects a TrainingResult.")
    X, y = _xy(data, target)
    X = _normalize_X(X, model.features)
    result = permutation_importance(
        model.model, X, y, scoring=model.scoring, n_repeats=repeats, random_state=seed, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": model.features,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@api("Training", "path = pb.save_model(result, 'trained.joblib')")
def save_model(model, path):
    """Save native Model as .npz, or TrainingResult/sklearn as a .joblib file.

    Native .npz uses arrays and JSON. Joblib files must be trusted before loading.
    """
    path = Path(path)
    if isinstance(model, Model):
        if path.suffix != ".npz":
            raise ConfigurationError("Save native neural models with a .npz extension.")
        model.save(path)
    else:
        if path.suffix != ".joblib":
            raise ConfigurationError("Save sklearn models with a .joblib extension.")
        joblib.dump(model, path)
    return path


@api("Training", "result = pb.load_model('trained.joblib', trusted=True)")
def load_model(path, *, trusted=False):
    """Load native .npz safely or trusted .joblib files with trusted=True.

    Joblib uses pickle and can execute code. Only load files you created or
    otherwise trust. Persisted models should use matching dependency versions.
    """
    path = Path(path)
    if path.suffix == ".npz":
        return Model.load(path)
    if path.suffix != ".joblib":
        raise ConfigurationError("Supported model extensions are .npz and .joblib.")
    if not trusted:
        raise ConfigurationError(
            "Joblib can execute code; pass trusted=True only for a trusted model file."
        )
    return joblib.load(path)


@api("Training", "params = pb.recommend_params(df, target='target')")
def recommend_params(data, target="target", *, task="auto"):
    """Return transparent starter heuristics (task, batch size, layers, CV, optimizer).

    These are starting values, not measured optimal parameters. Use auto_train()
    for measured CV selection. Native networks expect numeric features.
    """
    X, y = _xy(data, target)
    task = _task(y, task)
    return {
        "task": task,
        "features": len(X.columns),
        "batch_size": min(32, max(1, len(X) // 8)),
        "hidden_units": [min(128, max(8, 2 * len(X.columns)))],
        "learning_rate": 0.001,
        "optimizer": "adam",
        "cv": min(3, len(X), int(pd.Series(y).value_counts().min()))
        if task == "classification"
        else min(3, len(X)),
        "basis": "starter heuristics; tune and evaluate on independent rows",
    }


@api("Training", "network = pb.build_network(input_dim=4, output_dim=3, task='multiclass')")
def build_network(
    input_dim,
    output_dim=1,
    *,
    hidden=(32, 16),
    task="regression",
    optimizer="adam",
    learning_rate=0.001,
    seed=42,
):
    """Build/compile a native CPU dense network; task is binary/multiclass/regression.

    This is a sequential dense learner, not a TensorFlow/GPU/autograd engine.
    Multiclass output_dim is the number of classes; binary requires one output.
    """
    positive_int(input_dim, "input_dim")
    positive_int(output_dim, "output_dim")
    if task not in {"binary", "multiclass", "regression"}:
        raise ConfigurationError("Native task must be binary, multiclass, or regression.")
    if task == "binary" and output_dim != 1:
        raise ConfigurationError("Binary networks use output_dim=1.")
    if task == "multiclass" and output_dim < 2:
        raise ConfigurationError("Multiclass networks need output_dim >= 2.")
    model = Model(task=task, seed=seed)
    for units in hidden:
        model.add(positive_int(units, "hidden units"), activation="relu")
    model.add(
        output_dim,
        activation={"binary": "sigmoid", "multiclass": "softmax", "regression": "linear"}[task],
    )
    model.build(input_dim).compile(optimizer=optimizer, learning_rate=learning_rate)
    return model


@api("Training", "network = pb.fit_network(X, y, epochs=200, seed=7)")
def fit_network(X, y, *, epochs=100, task="auto", seed=42, learning_rate=0.001, **kwargs):
    """Auto-build and fit a native dense network on finite numeric arrays.

    epochs/batch_size/validation_split/patience are supported. Returns Model;
    training history is available as model.history_. Scale raw features first.
    """
    model = Model(task=task, seed=seed)
    model._prepare_and_auto_build(X, y)
    model.compile(learning_rate=learning_rate)
    model.history_ = model.fit(X, y, epochs=epochs, **kwargs)
    return model


@api("Training", "explanation = pb.explain_model(result)")
def explain_model(model):
    """Return human-readable model setup, selection method, parameters, and limitations."""
    if isinstance(model, TrainingResult):
        return {
            **model.summary(),
            "selection": "training-only cross-validation; highest mean CV score",
            "preprocessing": "fold-local numeric median + scaling; categorical missing token + one-hot encoding",
            "limitation": "best tested configuration; independent shuffled rows assumed",
        }
    if isinstance(model, Model):
        return {
            **model.info(),
            "backend": "NumPy CPU",
            "layers_detail": [repr(layer) for layer in model.layers],
        }
    warnings.warn("Unknown estimator type; returning its repr only.", RuntimeWarning, stacklevel=2)
    return {"model": repr(model)}
