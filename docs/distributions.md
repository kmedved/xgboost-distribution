# Distributions reference

`XGBDistribution(distribution=...)` accepts the names below. The `predict()` return value is a `namedtuple` whose fields exactly match the `scipy.stats` parameter names — so you can pass them straight back into `scipy.stats.<dist>`.

| `distribution=` | Parameters returned | Target support | scipy.stats |
| --- | --- | --- | --- |
| `"normal"` | `loc`, `scale` | real-valued | [`scipy.stats.norm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) |
| `"laplace"` | `loc`, `scale` | real-valued | [`scipy.stats.laplace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) |
| `"log-normal"` | `scale`, `s` | strictly positive | [`scipy.stats.lognorm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html) |
| `"exponential"` | `scale` | non-negative | [`scipy.stats.expon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html) |
| `"poisson"` | `mu` | non-negative integers | [`scipy.stats.poisson`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html) |
| `"negative-binomial"` | `n`, `p` | non-negative integers | [`scipy.stats.nbinom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html) |

Default is `"normal"`. The list is generated dynamically from `BaseDistribution.__subclasses__()` (see [distributions/\_\_init\_\_.py](../src/xgboost_distribution/distributions/__init__.py)).

## Choosing a distribution

- **Continuous, symmetric residuals** → `normal`. Sensitive to outliers.
- **Continuous, heavy-tailed or with outliers** → `laplace`.
- **Continuous and strictly positive** (e.g. durations, prices) → `log-normal` or `exponential`.
- **Count data, low overdispersion** → `poisson` (assumes `mean == variance`).
- **Count data, overdispersed** → `negative-binomial` (extra dispersion parameter).

`XGBDistribution` validates the target on `fit()` and raises if the target is incompatible (e.g. negative values for `poisson`, non-integers for count distributions).

## Returned namedtuples

```python
from xgboost_distribution import XGBDistribution

model = XGBDistribution(distribution="normal").fit(X, y)
preds = model.predict(X_test)
preds.loc        # mean
preds.scale      # std
preds._fields    # ('loc', 'scale')
```

Because the fields are positional and named, you can also unpack:

```python
loc, scale = model.predict(X_test)
```

Or feed them straight to `scipy.stats`:

```python
from scipy.stats import norm
percentile_90 = norm.ppf(0.9, loc=preds.loc, scale=preds.scale)
```

For count data:

```python
model = XGBDistribution(distribution="negative-binomial").fit(X, y)
preds = model.predict(X_test)         # Params(n=..., p=...)

from scipy.stats import nbinom
prob_zero = nbinom.pmf(0, n=preds.n, p=preds.p)
```

See [count_data.py](../examples/count_data.py) for a worked end-to-end example with a heatmap visualisation.

## Scoring with log-likelihood

For cross-validation or hyperparameter search, use the log-likelihood scorer from [`metrics`](../src/xgboost_distribution/metrics.py):

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from xgboost_distribution import XGBDistribution
from xgboost_distribution.metrics import get_ll_score_func

distribution = "normal"
xgb_cv = GridSearchCV(
    XGBDistribution(distribution=distribution),
    param_grid={"n_estimators": [5, 10, 20], "max_depth": [1, 2, 3]},
    cv=5,
    scoring={f"{distribution}_ll": make_scorer(get_ll_score_func(distribution))},
    refit=False,
)
xgb_cv.fit(X, y)
```

`get_ll_score_func` is a thin wrapper around the matching `scipy.stats.<dist>.logpdf` / `logpmf`, with parameter names mapped to the distribution's namedtuple fields.

## Math behind each distribution

Each distribution's docstring includes the full derivation of the gradient, Hessian, and reparameterised Fisher information. See:

- [`normal.py`](../src/xgboost_distribution/distributions/normal.py)
- [`laplace.py`](../src/xgboost_distribution/distributions/laplace.py)
- [`log_normal.py`](../src/xgboost_distribution/distributions/log_normal.py)
- [`exponential.py`](../src/xgboost_distribution/distributions/exponential.py)
- [`poisson.py`](../src/xgboost_distribution/distributions/poisson.py)
- [`negative_binomial.py`](../src/xgboost_distribution/distributions/negative_binomial.py)

For an overview of how distributions plug into the boosting loop, see [architecture.md](architecture.md).
