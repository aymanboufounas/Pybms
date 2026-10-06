# Getting started

Install Python 3.10 or newer. Use a virtual environment for your project.

```bash
python -m pip install "git+https://github.com/aymanboufounas/Pybms.git"
```

This command requires Git. A ZIP installation works without Git:

```bash
python -m pip install "https://github.com/aymanboufounas/Pybms/archive/refs/heads/main.zip"
```

The short command `python -m pip install pybms` becomes available after the maintainer publishes to PyPI. See [Publishing](PUBLISHING.md).

## Train in three lines

```python
import pybms as pb

data = pb.load_data("iris")
result = pb.auto_train(data, target="target")
print(result.metrics)
print(result.predict(data.head()))
```

`result` contains a fitted preprocessing/model pipeline, a cross-validation leaderboard, parameters and held-out metrics.
Predicting these example rows shows the interface. The independent performance estimate is `result.metrics`.

## Your CSV or URL

```python
data = pb.load_data("customers.csv")
result = pb.auto_train(data, target="churn")
```

```python
url = "https://raw.githubusercontent.com/aymanboufounas/Pybms/main/examples/data/demo.csv"
data = pb.from_url(url)
result = pb.auto_train(data, target="passed")
```

The URL must return a dataset file. For endpoints without file extensions, use `format='csv'` or `format='json'`.
Reader settings such as `sep=';'`, `encoding='utf-8'` and `lines=True` are supported.
Downloads have a 20-second timeout and 50 MB default limit; both are adjustable.

## Clean for exploration

```python
data = pb.random_data(rows=200, columns=4, missing=0.15)
cleaned, report = pb.clean_data(data, report=True)
print(report)
print(pb.missing_report(data))
ax = pb.hist_plot(cleaned)
ax.figure.savefig("distributions.png")
```

`clean_data` trims whitespace, converts blank/infinite feature values to missing, removes duplicate rows, and fills numeric medians/category modes.
When a target exists, pass `target='name'` to preserve it. Missing targets require an explicit decision, commonly dropping those rows.
To evaluate models, pass the raw table to `auto_train`: it fits learned cleaning inside each training fold.

## Choose a task explicitly when needed

```python
result = pb.auto_train(data, target="price", task="regression")
```

Whole-number targets may be class labels or numeric regression outputs. Explicit task selection removes that ambiguity.
Automatic tabular training supports one target, assumes independent rows, and uses shuffled splits.
For forecasting, repeated subjects or grouped observations, create an appropriate chronological/grouped evaluation yourself.

## Learn a dense network

```python
X = pb.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = pb.array([0, 1, 1, 0])
network = pb.Sequential([pb.Dense(8, activation="tanh"), pb.Dense(1, activation="sigmoid")], seed=7)
network.compile(optimizer="adam", loss="binary_crossentropy", learning_rate=0.03)
network.fit(X, y, epochs=1200, batch_size=4, verbose=0)
print(network.predict(X, classes=True))
```

This XOR example illustrates fitting, not performance on new data.
Native task names are `binary`, `multiclass`, `regression` or `auto`. Tabular task names are `classification`, `regression` or `auto`.
Networks accept finite numeric arrays; their built-in dense backend runs on CPU.
See [native_validation.py](../examples/native_validation.py) for separate train/validation/test partitions and training-only scaling.

## Save and load

```python
pb.save_model(network, "network.npz")
network = pb.load_model("network.npz")
```

Native persistence stores JSON and arrays without pickle. It preserves inference and history, and starts resumed training with fresh optimizer state.
Tabular results use Joblib, which uses pickle and can execute code on load:

```python
pb.save_model(result, "result.joblib")
result = pb.load_model("result.joblib", trusted=True)  # only your own or otherwise trusted file
```

## Get help and repair typos

```python
print(pb.list_functions())
print(pb.function_help("gradient_descent"))
print(pb.suggest_function("cleen_data"))
repair = pb.fix_syntax("import pybms as pb\nvalues = pb.array([1, 2]")
print(repair.source, repair.changes, repair.valid)
```

Pybms warns on unambiguous function/keyword spelling corrections. Reader functions that forward arbitrary options to pandas preserve those options.
Native model methods retain the original typo-friendly API; `Model(strict=True)` disables method-name recovery and output-size repair.
Invalid Python grammar fails before a library import runs. `fix_syntax` and the CLI handle a supplied source string/file explicitly.
They can append missing closing brackets, add missing block-header colons and repair recognized Pybms function names.
They return a reviewable result and never execute the repaired code. Quotes, indentation and ambiguous errors may still need manual correction.

```bash
python -m pybms functions --category Cleaning
python -m pybms help auto_train
python -m pybms fix broken.py --output corrected.py
```

The CLI refuses to overwrite an existing file. Every public workflow has a signature, explanation and example in the [100-function reference](API_REFERENCE.md).
