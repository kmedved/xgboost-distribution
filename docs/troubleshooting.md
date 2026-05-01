# Troubleshooting

Issues people commonly run into when using `XGBDistribution`.

## "Distribution is not implemented!"

```
ValueError: Distribution is not implemented! Please choose one of {...}
```

The `distribution=` argument must be one of the keys in [`AVAILABLE_DISTRIBUTIONS`](../src/xgboost_distribution/distributions/__init__.py). Names use hyphens — `"negative-binomial"`, `"log-normal"`. See [distributions.md](distributions.md) for the full list.

## Target validation errors

Each distribution checks that the target is compatible with its support:

- **`poisson`, `negative-binomial`** require non-negative **integers**. Floating-point counts (even integer-valued) will raise unless you cast.
- **`exponential`** requires non-negative reals.
- **`log-normal`** requires strictly positive reals.

Cast the target explicitly if needed:

```python
y_int = y.astype(int)
```

## "Please do not set objective directly!"

```
ValueError: Please do not set objective directly! Use the `distribution` kwarg
```

`XGBDistribution` controls XGBoost's `objective` internally based on the chosen `distribution`. Pass `distribution="..."` instead of `objective=...`.

## Predictions look unreliable / standard deviations are huge

`XGBDistribution` is more sensitive to overfitting than a point-estimate regressor — overfitting tends to collapse the predicted scale toward zero or blow it up. Mitigations, in order of importance:

1. **Use early stopping.** Always pass `eval_set=` and `early_stopping_rounds=`. The README example uses `early_stopping_rounds=10`.
2. **Lower `max_depth` and `learning_rate`.** Defaults are inherited from XGBoost and are tuned for point estimation; probabilistic estimation is happier with shallower trees and smaller steps.
3. **Use `GridSearchCV`** with the log-likelihood scorer ([`get_ll_score_func`](../src/xgboost_distribution/metrics.py)) — see [examples.md → Hyperparameter tuning](examples.md#hyperparameter-tuning-with-sklearn).

## NaN losses or gradients

Most often caused by the booster pushing distribution parameters to extreme values. Each distribution clips the reparameterised inputs to a safe range (`MIN_EXPONENT` / `MAX_EXPONENT` in [`distributions/utils.py`](../src/xgboost_distribution/distributions/utils.py)) so that `exp()` cannot overflow `float32`. If you still see NaN:

- Check for NaN/Inf in `X` or `y`.
- Reduce `learning_rate`.
- Try `natural_gradient=False` to compare against vanilla gradients.

## "Found dtype Float64 but expected Float32"-style XGBoost warnings

Internally, gradients and Hessians are computed as `float32` to match XGBoost. If you pass features in a different dtype, XGBoost may copy them. This is not an error, just a warning — pass `np.float32` arrays if you want to silence it.

## `save_model` / `load_model` round-trip

Use the dedicated methods, not `pickle`:

```python
model.save_model("model.json")

restored = XGBDistribution(distribution="normal")
restored.load_model("model.json")
```

`pickle` will work in many cases but is brittle across XGBoost versions. The JSON path uses XGBoost's native serialisation and stores the distribution name as a Booster attribute (see [`model.py`](../src/xgboost_distribution/model.py)).

## Sample weights with very low values

Sample weights are supported (since v0.2.6). Avoid weights of exactly zero on every sample of a class/region — it can prevent the booster from updating those parameters.

## Installation

```bash
pip install xgboost-distribution
```

On x86_64 Linux/Windows, this pulls `xgboost-cpu` (smaller wheel, no GPU). For GPU support:

```bash
pip install "xgboost-distribution[gpu]"
```

The `[gpu]` extra installs full `xgboost` (which supersedes `xgboost-cpu`) — not a separate GPU-only build.

## Still stuck?

- Search [open and closed issues](https://github.com/CDonnerer/xgboost-distribution/issues).
- Open a new issue with a minimal reproducible example and the package versions (`xgboost`, `scikit-learn`, `numpy`, `xgboost-distribution`).
