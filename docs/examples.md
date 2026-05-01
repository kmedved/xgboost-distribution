# Examples

Runnable scripts live in [`examples/`](../examples). All require the package and `matplotlib` (used for plots).

## Basic walkthrough

[`basic_walkthrough.py`](../examples/basic_walkthrough.py) — fits a Normal `XGBDistribution` on California Housing and plots residuals with predicted standard deviations as error bars.

```bash
python examples/basic_walkthrough.py
```

Useful as a sanity check that the package is installed correctly and that early stopping / fit / predict all work end-to-end.

## Count data with a distribution heatmap

[`count_data.py`](../examples/count_data.py) — generates synthetic data sampled from a Negative-Binomial distribution that depends on `X`, fits an `XGBDistribution(distribution="negative-binomial")`, and visualises the predicted PMF as a heatmap over `(X, y)`.

```bash
python examples/count_data.py
```

Try changing `distribution="negative-binomial"` to `"poisson"` or `"normal"` to see how mis-specification affects the fit.

## Hyperparameter tuning with sklearn

[`hyperparameter_tuning.py`](../examples/hyperparameter_tuning.py) — runs `GridSearchCV` over `XGBDistribution`, scoring each fold by log-likelihood via [`get_ll_score_func`](../src/xgboost_distribution/metrics.py).

```bash
python examples/hyperparameter_tuning.py
```

This is the canonical pattern for selecting `n_estimators` / `max_depth` / `learning_rate` on a probabilistic objective.

## Reproducing the benchmarks

[`experiments.py`](../examples/experiments.py) — runs the full `XGBDistribution` vs. `NGBRegressor` vs. `XGBRegressor` benchmark over the UCI datasets used in [Read the Docs experiments](https://xgboost-distribution.readthedocs.io/en/latest/experiments.html). Heavy — the MSD dataset alone takes ~20 minutes.

```bash
python examples/experiments.py
```

You'll need `ngboost` installed on top of the testing extras:

```bash
pip install -e ".[testing]" ngboost
```
