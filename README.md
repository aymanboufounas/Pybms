# Pybms

**Simple data and machine learning in a few lines of Python.**

Pybms 0.2.0 provides **100 documented workflow functions** for datasets, cleaning, arrays, plots, automatic tabular training, optimization and syntax assistance. It also includes a small Keras-style `Model` / `Sequential` / `Dense` API implemented with NumPy.

Built on NumPy, pandas, scikit-learn and Matplotlib. **Status: alpha.** Native networks support dense layers on CPU. Automatic model selection chooses among a bounded set of tested configurations.

## Install

Python 3.10 or newer:

```bash
python -m pip install "git+https://github.com/aymanboufounas/Pybms.git"
```

Without Git:

```bash
python -m pip install "https://github.com/aymanboufounas/Pybms/archive/refs/heads/main.zip"
```

For Excel and Parquet, install the optional readers:

```bash
python -m pip install "pybms[io] @ git+https://github.com/aymanboufounas/Pybms.git"
```

The short command `pip install pybms` becomes available after a PyPI release. The [publishing guide](docs/PUBLISHING.md) explains the included Trusted Publisher workflow and the owner's PyPI setup.

## Start here

```python
import pybms as pb

data = pb.load_data("iris")
result = pb.auto_train(data, target="target")

print(result.metrics)  # independent held-out test metrics
print(result.leaderboard)  # training-only cross-validation ranking
print(result.predict(data.head()))
```

Pybms infers classification/regression, handles missing features and categories, scales numeric features, compares estimators and searches a small parameter space. Preprocessing is learned within each training fold. The holdout is reserved for evaluation.

Defaults are small and reproducible. Override only what you need:

```python
result = pb.auto_train(data, target="price", task="regression", trials=8)
result = pb.train(data, target="target", estimator="forest")
```

Whole-number regression targets need an explicit `task="regression"`. Tabular convenience training assumes independent shuffled rows and one target; forecasting/grouped observations require a separate evaluation strategy.

## Load local files or URLs

```python
data = pb.load_data("customers.csv")
data = pb.from_url(
    "https://raw.githubusercontent.com/aymanboufounas/Pybms/main/examples/data/demo.csv"
)
result = pb.auto_train(data, target="passed")
```

Supported formats: CSV/TSV, JSON/JSONL, Excel and Parquet. URLs must return a dataset file. Use `format="csv"` when needed; reader options such as `sep=";"` pass through to pandas.

Offline samples: iris, wine, breast_cancer, diabetes, digits. Synthetic generators:

```python
data = pb.classification_data(rows=500, features=6, classes=3, seed=7)
data = pb.regression_data(rows=500, features=6, noise=0.1, seed=7)
data = pb.random_data(rows=100, columns=4, missing=0.1)
```

## Simple cleaning and plotting

```python
cleaned, report = pb.clean_data(data, report=True)
print(report)
print(pb.missing_report(data))

ax = pb.hist_plot(cleaned)
ax.figure.savefig("histogram.png")
```

Cleaning returns copies. Supply `target="name"` to preserve a target column. For model evaluation, give raw data to `auto_train` so imputation/encoding/scaling are learned only from training rows.

Plots return Matplotlib Axes, and `pair_plot` returns a Figure. You control display and export.

## Keras-style dense learning

```python
X = pb.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = pb.array([0, 1, 1, 0])

model = pb.Sequential(
    [
        pb.Dense(8, activation="tanh"),
        pb.Dense(1, activation="sigmoid"),
    ],
    seed=7,
)
model.compile(optimizer="adam", loss="binary_crossentropy", learning_rate=0.03)
history = model.fit(X, y, epochs=1200, batch_size=4, verbose=0)

print(model.predict(X, classes=True))
model.summary()
```

Or let Pybms build the network:

```python
model = pb.easy(X, y, epochs=500, seed=7, verbose=0)
```

Native training supports mini-batches, backpropagation, SGD/Adam, string class labels, validation loss, early stopping, gradient clipping and save/load. Native task names are `binary`, `multiclass`, `regression` and `auto`.

See [native_validation.py](examples/native_validation.py) for separate training/validation/test rows.

## Gradient descent you can inspect

```python
minimum = pb.gradient_descent(lambda x: ((x - 3) ** 2).sum(), [10.0, -5.0])
print(minimum.x, minimum.loss, minimum.converged)
```

Use analytical gradients or automatic central differences. Automatic step-size backtracking, Adam, gradient checks, clipping and learning-rate schedules are included.

## Save models

```python
pb.save_model(model, "network.npz")
model = pb.load_model("network.npz")
```

Native files contain JSON and arrays. Tabular models use Joblib:

```python
pb.save_model(result, "trained.joblib")
result = pb.load_model("trained.joblib", trusted=True)
```

Joblib uses pickle: only load your own or otherwise trusted files. Saved native networks preserve predictions/history and resume with fresh optimizer state.

## Forgiving spelling and reviewable syntax repair

Valid-Python model typos still work with warnings:

```python
model = pb.Model(seed=7)
model.add("dence", unit=8, activaton="reul")
model.add(1, activaton="sigmod")
model.complie(optmizer="adm", los="bce", lern_rate=0.01)
model.fitt(X, y, epocs=100, verbose=0)
```

Unambiguous top-level function and keyword corrections also warn. Use `Model(strict=True)` to disable method-name recovery and output-size repair.

A library import cannot intercept invalid Python grammar. Use the explicit source helper:

```python
repair = pb.fix_syntax("import pybms as pb\nvalues = pb.array([1, 2]")
print(repair.source)
print(repair.changes, repair.valid)
```

It repairs supported missing brackets/colons and recognized Pybms function typos. It returns source for review and never executes it. Ambiguous grammar, quotes and indentation may need manual repair.

## Discover all 100 functions

```python
print(pb.list_functions())
print(pb.list_functions(category="Cleaning"))
print(pb.function_help("auto_train"))
```

| Category | Functions |
|---|---:|
| Dataset loading and table operations | 20 |
| Cleaning and preprocessing | 20 |
| NumPy helpers | 15 |
| Matplotlib plots | 15 |
| Training, model selection and persistence | 15 |
| Optimization and activations | 10 |
| Syntax repair and function help | 5 |
| **Total workflow functions** | **100** |

Native classes, methods and the original `easy`, `device`, `about` entry points are additional.

CLI:

```bash
python -m pybms functions --category Cleaning
python -m pybms help auto_train
python -m pybms fix broken.py --output corrected.py
```

## Documentation and development

- [Getting started](docs/GETTING_STARTED.md)
- [100-function API reference](docs/API_REFERENCE.md)
- [Exceptions and recovery](docs/EXCEPTIONS.md)
- [Implementation sources and bibliography](docs/BIBLIOGRAPHY.md)
- [Publishing to PyPI](docs/PUBLISHING.md)
- [Examples](examples/)
- [Changelog](CHANGELOG.md)

```bash
python -m pip install ".[dev,io]"
python -m ruff check .
python -m pytest -q
python tools/generate_docs.py
python -m build
python -m twine check dist/*
```

CI checks Linux/Windows, Python 3.10/3.12/3.13, tests, documentation freshness and package builds. A separate job checks the declared minimum core dependencies.

## License

MIT — Ayman Boufounas.
