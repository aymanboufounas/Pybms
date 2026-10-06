# Exceptions and recovery

Catch `pb.PyBMSError` to handle package-specific failures, or a more specific class below.
Python/pandas/NumPy can also raise standard `TypeError`, `ValueError` and `OSError` for invalid arguments or filesystem access.
Typo corrections emit `RuntimeWarning` so corrections are visible.

| Exception | Typical cause | Next step |
|---|---|---|
| `PyBMSError` | Base class for package failures | Catch this when a broad package error handler is suitable. |
| `ConfigurationError` | Unsupported option, invalid learning rate, unsafe Joblib load | Inspect `function_help`; choose a valid option. |
| `DataError` | Missing target, malformed file, failed URL download, empty/nonfinite input | Inspect the table and reader settings; clean feature values explicitly. |
| `ShapeError` | Incompatible array dimensions | Check `.shape`, target length and number of output units. |
| `InvalidLayerError` | Unsupported native layer | Use `Dense` or an integer unit count. |
| `UnknownActivationError` | Unknown activation | Use linear/relu/sigmoid/tanh/softmax. |
| `UnknownLossError` | Unknown native loss | Use mse/mae/binary_crossentropy/categorical_crossentropy. |
| `UnknownOptimizerError` | Unknown native optimizer | Use sgd/adam or an object implementing step. |
| `ModelNotBuiltError` | Model has no built layers | Add/build/fit before predicting or saving. |
| `ModelNotCompiledError` | Loss evaluation before compile/fit | Call compile or fit. |
| `TrainingError` | Diverging loss or failed estimator search | Scale features, lower learning rate, or inspect leaderboard error details. |
| `NotSupportedError` | Unsupported file type or absent optional reader | Use supported formats or install the `[io]` extra. |

```python
import pybms as pb

try:
    data = pb.from_url("https://example.org/data.csv")
except pb.DataError as error:
    print("Dataset could not be loaded:", error)
```

Targets are never automatically imputed by tabular training. Decide how to handle unknown targets before fitting.
Native networks require finite numeric feature arrays; `clean_data` and scaling helpers are available for explicit preparation.
