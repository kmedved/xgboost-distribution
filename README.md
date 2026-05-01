[![Tests](https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/CDonnerer/xgboost-distribution/actions/workflows/test.yml)
[![Coverage](https://coveralls.io/repos/github/CDonnerer/xgboost-distribution/badge.svg?branch=main)](https://coveralls.io/github/CDonnerer/xgboost-distribution?branch=main)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docs](https://readthedocs.org/projects/xgboost-distribution/badge/?version=latest)](https://xgboost-distribution.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/xgboost-distribution.svg)](https://pypi.org/project/xgboost-distribution/)

# xgboost-distribution

XGBoost for **probabilistic prediction**. Like [NGBoost](https://github.com/stanfordmlgroup/ngboost), but [faster](https://xgboost-distribution.readthedocs.io/en/latest/experiments.html), and using the [XGBoost scikit-learn API](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn).

<p align="center">
  <img src="https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/xgb_dist.png" width="600" alt="XGBDistribution example">
</p>

## Quickstart

```bash
pip install xgboost-distribution
pip install xgboost-distribution[gpu]   # for GPU support
```

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from xgboost_distribution import XGBDistribution

data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = XGBDistribution(
    distribution="normal",
    n_estimators=500,
    early_stopping_rounds=10,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

preds = model.predict(X_test)
mean, std = preds.loc, preds.scale
```

`predict` returns a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple) of NumPy arrays — one per parameter of the chosen distribution, named per [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html) conventions.

> **Tip:** use `early_stopping_rounds` — without it, distribution parameters can become unreliable from overfitting.

## Documentation

| Topic | Where to look |
| --- | --- |
| Architecture and how it works | [docs/architecture.md](docs/architecture.md) |
| Supported distributions and their parameters | [docs/distributions.md](docs/distributions.md) |
| Running the examples | [docs/examples.md](docs/examples.md) |
| Local development, testing, contributing | [docs/development.md](docs/development.md) |
| Common issues and pitfalls | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Benchmarks vs. NGBoost / XGBRegressor | [Experiments](https://xgboost-distribution.readthedocs.io/en/latest/experiments.html) |
| Full API reference | [Read the Docs](https://xgboost-distribution.readthedocs.io/en/latest/) |
| Release notes | [CHANGELOG.rst](CHANGELOG.rst) |

## Why use it

- **Probabilistic regression** with `XGBDistribution` performs comparably to `NGBRegressor` on negative log-likelihood, while being **~15x faster** on California Housing and **~20x faster** on the 500k-row MSD dataset (18 minutes vs. 6.7 hours).
- Drop-in support for the **full XGBoost feature set** — including [monotonic constraints](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html), GPU training, and early stopping.
- Six built-in distributions: `normal`, `laplace`, `log-normal`, `exponential`, `poisson`, `negative-binomial`.

<p align="center">
  <img src="https://raw.githubusercontent.com/CDonnerer/xgboost-distribution/main/imgs/performance_comparison.png" width="600" alt="XGBDistribution vs NGBoost">
</p>

See the [experiments page](https://xgboost-distribution.readthedocs.io/en/latest/experiments.html) for benchmarks across multiple datasets.

## Requirements

- Python `>=3.10`
- `scikit-learn`
- `xgboost>=3.0.0` (uses `xgboost-cpu` on x86_64 Windows/Linux for a smaller footprint; the `[gpu]` extra installs full `xgboost`)

## Acknowledgements

This package builds on the work of:

- [NGBoost](https://github.com/stanfordmlgroup/ngboost) — demonstrated gradient boosting with natural gradients for distributional estimation; much of the gradient code was adapted from there.
- [XGBoost](https://github.com/dmlc/xgboost) — the underlying gradient boosting engine. The scikit-learn API here mirrors XGBoost's.

## License

[MIT](LICENSE.txt) © Christian Donnerer. See [AUTHORS.rst](AUTHORS.rst) for the contributor list.

---

This project was scaffolded with [PyScaffold](https://pyscaffold.org/) 4.0.1.
