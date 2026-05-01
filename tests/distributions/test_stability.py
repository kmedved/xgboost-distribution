"""Stability tests at the edges of the float32 reparameterisation range.

`safe_exp` clips parameters to roughly `(e^-73, e^87)`, which means downstream
quantities like `scale**2`, `n*p`, and `mu` can sit near float32 overflow or
underflow. These tests check that natural-gradient computations stay finite at
those extremes.
"""
import numpy as np
import pytest
from scipy.special import logit

from xgboost_distribution.distributions import (
    Laplace,
    NegativeBinomial,
    Normal,
    Poisson,
)
from xgboost_distribution.distributions.normal import MAX_LOG_SCALE
from xgboost_distribution.distributions.utils import MAX_EXPONENT, MIN_EXPONENT


def _all_finite(arr):
    return np.all(np.isfinite(arr))


# -- Laplace -------------------------------------------------------------------


@pytest.mark.parametrize("log_scale", [MAX_EXPONENT - 1, MAX_EXPONENT])
def test_laplace_natural_gradient_finite_at_large_scale(log_scale):
    """Laplace.predict() applies `safe_exp` (clip to MAX_EXPONENT ≈ 87),
    not the tighter MAX_LOG_SCALE Normal uses. So `scale` can reach ~1e38
    and `scale**2` overflows float32. The implementation uses the
    algebraically simplified `sign(diff) * scale` form, which does not.
    """
    y = np.array([0.0, 1.0, -1.0], dtype="float32")
    params = np.tile(
        np.array([0.0, log_scale], dtype="float32"), (3, 1)
    )
    grad, hess = Laplace().gradient_and_hessian(y, params, natural_gradient=True)
    assert _all_finite(grad), (
        f"Laplace natural grad has inf/nan at log_scale={log_scale}"
    )
    assert _all_finite(hess)


# -- NegativeBinomial ----------------------------------------------------------


@pytest.mark.parametrize(
    "log_n, logit_p",
    [
        # Mid-range / typical training values.
        (np.log(0.5), logit(0.5)),
        (np.log(0.5), logit(0.99)),
        (-30.0, logit(0.5)),
        # Pathological boundary cases. `expit(MAX_EXPONENT)` saturates to
        # exactly 1.0 in float32 (so `1-p == 0`), and the lower-bound
        # combination underflows `n*p` to zero. The DIAG_FLOOR in
        # `negative_binomial.py` keeps these finite.
        (np.log(0.5), MAX_EXPONENT),
        (MIN_EXPONENT, MIN_EXPONENT),
        (MIN_EXPONENT, MAX_EXPONENT),
        # Upper-clip combination: at log_n=MAX_EXPONENT (n ≈ 1.25e38) and
        # logit_p=MIN_EXPONENT (p ≈ 1e-32), the raw gradient `-n * log(p)`
        # would overflow float32 if computed naively, and `n*(1-p)/p`
        # would also overflow. The float64 raw-gradient computation +
        # algebraic rewrite for column 1 keep the result finite.
        (MAX_EXPONENT, MIN_EXPONENT),
    ],
)
def test_negative_binomial_natural_gradient_finite(log_n, logit_p):
    """The Fisher diagonal involves `n*p/(p+1)` and `n*(1-p)`. Each can
    underflow to zero in float32 at the parameter clipping boundaries; the
    DIAG_FLOOR in `negative_binomial.py` keeps the natural gradient finite.
    """
    y = np.array([0, 1, 2])
    params = np.tile(
        np.array([log_n, logit_p], dtype="float64"), (3, 1)
    )
    grad, hess = NegativeBinomial().gradient_and_hessian(
        y, params, natural_gradient=True
    )
    assert _all_finite(grad), (
        f"NegBinom natural grad has inf/nan at log_n={log_n}, logit_p={logit_p}; "
        f"got {grad}"
    )
    assert _all_finite(hess)


# -- Poisson -------------------------------------------------------------------


@pytest.mark.parametrize("log_mu", [MIN_EXPONENT, MIN_EXPONENT + 1, 0.0, MAX_EXPONENT])
def test_poisson_natural_gradient_finite(log_mu):
    """The Fisher diagonal entry is mu = safe_exp(log_mu). At the lower clip
    (~e^-73) it's tiny but nonzero; division by it should produce a finite
    (possibly large) gradient.
    """
    y = np.array([0, 1, 5])
    params = np.full(3, log_mu, dtype="float64")
    grad, hess = Poisson().gradient_and_hessian(y, params, natural_gradient=True)
    assert _all_finite(grad), f"Poisson natural grad has inf/nan at log_mu={log_mu}; got {grad}"
    assert _all_finite(hess)


# -- Normal --------------------------------------------------------------------


@pytest.mark.parametrize("log_scale", [-30.0, 0.0, MAX_LOG_SCALE - 1])
def test_normal_natural_gradient_finite(log_scale):
    """Normal already had a partial overflow test; this rounds out the
    suite to cover the whole `_safe_params` range.
    """
    y = np.array([-1.0, 0.0, 1.0], dtype="float32")
    params = np.tile(
        np.array([0.0, log_scale], dtype="float32"), (3, 1)
    )
    grad, hess = Normal().gradient_and_hessian(y, params, natural_gradient=True)
    assert _all_finite(grad), f"Normal natural grad has inf/nan at log_scale={log_scale}"
    assert _all_finite(hess)
