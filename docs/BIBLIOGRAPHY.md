# Implementation sources and bibliography

Pybms builds on established algorithms and libraries. Its 100 public functions provide a simplified workflow interface.
The API reference links each function to its implementation. These sources explain the underlying tools and numerical methods.

| Area | Primary source | Used for |
|---|---|---|
| Arrays and numerical operations | [NumPy reference](https://numpy.org/doc/stable/reference/) | Array creation, shapes, statistics, broadcasting and matrix operations. |
| Dataset IO and table cleaning | [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html) | CSV/JSON/Excel/Parquet readers, missing values and table operations. |
| Plotting | [Matplotlib API](https://matplotlib.org/stable/api/index.html) | Lines, scatter, histograms, box plots, heatmaps and figure export. |
| Preprocessing and estimators | [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) | Imputation, scaling, one-hot encoding, linear models, forests and SGD. |
| Leakage prevention | [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Training-only preprocessing and cross-validation pipelines. |
| Evaluation | [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) | Stratified classification folds, model comparison and independent evaluation. |
| Persistence | [Joblib persistence](https://joblib.readthedocs.io/en/latest/persistence.html) | Tabular model serialization and pickle trust constraints. |
| Package distribution | [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/) | Project metadata, wheels, source distributions and pip installation. |
| PyPI publishing | [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) | GitHub Actions OIDC publishing configuration. |

## Native neural learning

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature, 323, 533–536. [Paper](https://www.nature.com/articles/323533a0).
- Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS. [Paper](https://proceedings.mlr.press/v9/glorot10a.html). The standalone `initialize_weights(method='xavier')` uses a fan-in/fan-out normal scaling; existing Dense initialization uses a fan-in scaling for non-ReLU layers.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. [Paper](https://arxiv.org/abs/1502.01852). Used for ReLU fan-in weight scaling; Pybms implements dense networks.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR. [Paper](https://arxiv.org/abs/1412.6980). Used for first/second moment adaptation and bias correction.

The builtin sample datasets come from scikit-learn. The repository's small `examples/data/demo.csv` is synthetic demonstration data.
