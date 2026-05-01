"""Metrics for evaluating xgboost-distribution scores
"""
from collections.abc import Callable

import numpy as np
import scipy
from xgboost._typing import ArrayLike


def get_ll_score_func(
    distribution: str,
) -> Callable[[ArrayLike, tuple[np.ndarray, ...]], float]:
    """Get log-likelihood scoring function for a given distribution

    Parameters
    ----------
    distribution : str

    Returns
    -------
    Callable
        Scoring function
    """
    dist_ll = {
        "exponential": (scipy.stats.expon.logpdf, ("scale",)),
        "laplace": (scipy.stats.laplace.logpdf, ("loc", "scale")),
        "log-normal": (scipy.stats.lognorm.logpdf, ("scale", "s")),
        "negative-binomial": (scipy.stats.nbinom.logpmf, ("n", "p")),
        "normal": (scipy.stats.norm.logpdf, ("loc", "scale")),
        "poisson": (scipy.stats.poisson.logpmf, ("mu",)),
    }

    def score_func(y, y_pred):
        ll_func, param_names = dist_ll[distribution]
        if hasattr(y_pred, "_asdict"):
            params = y_pred._asdict()
        else:
            params = dict(zip(param_names, y_pred))
        return ll_func(y, **params).mean()

    return score_func
