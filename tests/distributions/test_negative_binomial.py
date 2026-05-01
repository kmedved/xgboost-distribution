import pytest

import numpy as np
import pandas as pd
import scipy.special
from scipy.special import logit

from xgboost_distribution.distributions import NegativeBinomial


@pytest.fixture
def negative_binomial():
    return NegativeBinomial()


def test_gradient_calculation_natural(negative_binomial):
    """Natural-gradient column 1 has a clean closed form: with n=4, p=0.5
    and y=2, raw grad[1] = p*(y - n*(1-p)/p) = 0.5*(2 - 4) = -1, and
    F_bb = n*(1-p) = 2, so natural grad[1] = -0.5.

    Column 0 involves `digamma` and is checked against an inline
    closed-form expectation derived from the same Fisher diagonal.
    """
    y = np.array([2])
    n_param, p_param = 4.0, 0.5
    params = np.array([[np.log(n_param), logit(p_param)]])

    grad, _ = negative_binomial.gradient_and_hessian(
        y, params, natural_gradient=True
    )

    raw_grad0 = -n_param * (
        scipy.special.digamma(y + n_param)
        - scipy.special.digamma(n_param)
        + np.log(p_param)
    )
    expected_natural_grad0 = raw_grad0 / ((n_param * p_param) / (p_param + 1))
    expected_natural_grad1 = np.array([-0.5])

    np.testing.assert_array_almost_equal(grad[:, 0], expected_natural_grad0, decimal=5)
    np.testing.assert_array_almost_equal(grad[:, 1], expected_natural_grad1, decimal=6)


def test_target_validation(negative_binomial):
    valid_target = np.array([0, 1, 4, 5, 10])
    negative_binomial.check_target(valid_target)


@pytest.mark.parametrize(
    "invalid_target",
    [np.array([-0.1, 1.2]), pd.Series([1.1, 0.4, 2.3])],
)
def test_target_validation_raises(negative_binomial, invalid_target):
    with pytest.raises(ValueError):
        negative_binomial.check_target(invalid_target)


@pytest.mark.parametrize(
    "y, params",
    [
        (
            np.array([20], dtype="float32"),
            np.array([[113.1, 11.2]], dtype="float32"),
        ),
        (
            np.array([20], dtype="float32"),
            np.array([[13.1, -111.2]], dtype="float32"),
        ),
    ],
)
def test_overflow_stability(negative_binomial, y, params):
    """Test stability against large/small values produced by xgboost"""
    grad, _ = negative_binomial.gradient_and_hessian(y, params)
    assert isinstance(grad, np.ndarray)

    n, p = negative_binomial.predict(params)
    assert all(np.isfinite(n))
    assert n.all()
    assert p.all()
