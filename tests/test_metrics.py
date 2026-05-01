"""Test suite for metrics"""

import pytest

import numpy as np
import scipy

from xgboost_distribution.distributions import AVAILABLE_DISTRIBUTIONS
from xgboost_distribution.distributions.exponential import Params as ExponentialParams
from xgboost_distribution.distributions.laplace import Params as LaplaceParams
from xgboost_distribution.distributions.log_normal import Params as LogNormalParams
from xgboost_distribution.distributions.negative_binomial import (
    Params as NegativeBinomialParams,
)
from xgboost_distribution.distributions.normal import Params as NormalParams
from xgboost_distribution.distributions.poisson import Params as PoissonParams
from xgboost_distribution.metrics import get_ll_score_func


def test_get_ll_score_func_distribution_exist():
    for distribution in AVAILABLE_DISTRIBUTIONS:
        score_func = get_ll_score_func(distribution=distribution)
        assert callable(score_func)


@pytest.mark.parametrize(
    ("distribution", "y", "y_pred", "expected"),
    [
        (
            "exponential",
            np.array([0.25, 1.5]),
            ExponentialParams(scale=np.array([1.0, 2.0])),
            scipy.stats.expon.logpdf(
                np.array([0.25, 1.5]), scale=np.array([1.0, 2.0])
            ).mean(),
        ),
        (
            "laplace",
            np.array([-1.0, 1.0]),
            LaplaceParams(loc=np.array([0.0, 0.5]), scale=np.array([1.0, 2.0])),
            scipy.stats.laplace.logpdf(
                np.array([-1.0, 1.0]),
                loc=np.array([0.0, 0.5]),
                scale=np.array([1.0, 2.0]),
            ).mean(),
        ),
        (
            "log-normal",
            np.array([0.5, 2.0]),
            LogNormalParams(scale=np.array([1.0, 2.0]), s=np.array([0.5, 0.8])),
            scipy.stats.lognorm.logpdf(
                np.array([0.5, 2.0]),
                s=np.array([0.5, 0.8]),
                scale=np.array([1.0, 2.0]),
            ).mean(),
        ),
        (
            "negative-binomial",
            np.array([1, 3]),
            NegativeBinomialParams(n=np.array([5.0, 7.0]), p=np.array([0.4, 0.6])),
            scipy.stats.nbinom.logpmf(
                np.array([1, 3]), n=np.array([5.0, 7.0]), p=np.array([0.4, 0.6])
            ).mean(),
        ),
        (
            "normal",
            np.array([-1.0, 1.0]),
            NormalParams(loc=np.array([0.0, 0.5]), scale=np.array([1.0, 2.0])),
            scipy.stats.norm.logpdf(
                np.array([-1.0, 1.0]),
                loc=np.array([0.0, 0.5]),
                scale=np.array([1.0, 2.0]),
            ).mean(),
        ),
        (
            "poisson",
            np.array([1, 3]),
            PoissonParams(mu=np.array([2.0, 4.0])),
            scipy.stats.poisson.logpmf(
                np.array([1, 3]), mu=np.array([2.0, 4.0])
            ).mean(),
        ),
    ],
)
def test_get_ll_score_func_uses_distribution_parameter_names(
    distribution, y, y_pred, expected
):
    score_func = get_ll_score_func(distribution=distribution)

    assert score_func(y, y_pred) == pytest.approx(expected)
    assert score_func(y, tuple(y_pred)) == pytest.approx(expected)
