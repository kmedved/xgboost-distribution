"""Property tests verifying bit-identity of the natural-gradient path
to the legacy `numpy.linalg.solve(F, g)` formulation it replaced.

For each distribution where the reparameterised Fisher is provably diagonal
(Normal, LogNormal, Laplace, Exponential, Poisson) and for the legacy
diagonal approximation used by NegativeBinomial, these tests build the
old-style `(n, p, p)` float32 Fisher matrix, solve `F · x = g`, and check
that the production `gradient_and_hessian(natural_gradient=True)` produces
the same `float32` array element-for-element on randomly generated inputs.
"""
import pytest

import numpy as np
from scipy.special import digamma, expit, logit

from xgboost_distribution.distributions import (
    Exponential,
    Laplace,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
)


def _solve_diag_fisher(grad, *diag_entries):
    """Reproduce legacy `linalg.solve` on a diagonal Fisher matrix.

    Builds a `(n, p, p)` float32 matrix whose only non-zero entries are the
    diagonals supplied, then solves. Equivalent (and bitwise-identical) to
    `grad[:, i] /= diag_entries[i].astype('float32')` per column.
    """
    n_samples, p = grad.shape
    F = np.zeros((n_samples, p, p), dtype="float32")
    for i, entry in enumerate(diag_entries):
        F[:, i, i] = entry
    return np.linalg.solve(F, grad[..., np.newaxis])[..., 0]


# -- Normal --------------------------------------------------------------------


def _ref_normal_natural_grad(y, params):
    dist = Normal()
    grad, _ = dist.gradient_and_hessian(y, params, natural_gradient=False)
    _, log_scale = dist._safe_params(params)
    var = np.exp(2 * log_scale)
    return _solve_diag_fisher(grad, 1 / var, np.full_like(var, 2))


# -- LogNormal -----------------------------------------------------------------


def _ref_log_normal_natural_grad(y, params):
    dist = LogNormal()
    grad, _ = dist.gradient_and_hessian(y, params, natural_gradient=False)
    _, log_s = dist._safe_params(params)
    var = np.exp(2 * log_s)
    return _solve_diag_fisher(grad, 1 / var, np.full_like(var, 2))


# -- Laplace -------------------------------------------------------------------


def _ref_laplace_natural_grad(y, params):
    dist = Laplace()
    grad, _ = dist.gradient_and_hessian(y, params, natural_gradient=False)
    _, scale = dist.predict(params)
    return _solve_diag_fisher(grad, 1 / scale**2, np.ones_like(scale))


# -- Exponential ---------------------------------------------------------------


def _ref_exponential_natural_grad(y, params):
    # Reparameterised Fisher = 1; natural gradient equals raw gradient.
    grad, _ = Exponential().gradient_and_hessian(y, params, natural_gradient=False)
    return grad


# -- Poisson -------------------------------------------------------------------


def _ref_poisson_natural_grad(y, params):
    dist = Poisson()
    grad, _ = dist.gradient_and_hessian(y, params, natural_gradient=False)
    (mu,) = dist.predict(params)
    F = np.zeros((len(y), 1, 1), dtype="float32")
    F[:, 0, 0] = mu
    return np.linalg.solve(F, grad[..., np.newaxis])[..., 0]


# -- NegativeBinomial ----------------------------------------------------------
# NegativeBinomial's `gradient_and_hessian(natural=False)` raises, so we
# inline the raw gradient here, matching the implementation in
# `negative_binomial.py`.


def _ref_negative_binomial_natural_grad(y, params):
    dist = NegativeBinomial()
    n, p = dist.predict(params)
    grad = np.zeros((len(y), 2), dtype="float32")
    grad[:, 0] = -n * (digamma(y + n) - digamma(n) + np.log(p))
    grad[:, 1] = p * (y - n * (1 - p) / p)
    return _solve_diag_fisher(grad, (n * p) / (p + 1), n * p)


# -- Test cases ----------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_normal_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = rng.standard_normal(200).astype(dtype)
    params = rng.standard_normal((200, 2)).astype(dtype) * 0.5
    new_grad, _ = Normal().gradient_and_hessian(y, params, natural_gradient=True)
    ref = _ref_normal_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_log_normal_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = np.exp(rng.standard_normal(200)).astype(dtype)
    params = rng.standard_normal((200, 2)).astype(dtype) * 0.5
    new_grad, _ = LogNormal().gradient_and_hessian(y, params, natural_gradient=True)
    ref = _ref_log_normal_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_laplace_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = rng.standard_normal(200).astype(dtype)
    params = rng.standard_normal((200, 2)).astype(dtype) * 0.5
    new_grad, _ = Laplace().gradient_and_hessian(y, params, natural_gradient=True)
    ref = _ref_laplace_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exponential_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = rng.exponential(size=200).astype(dtype)
    params = rng.standard_normal(200).astype(dtype) * 0.5
    new_grad, _ = Exponential().gradient_and_hessian(y, params, natural_gradient=True)
    ref = _ref_exponential_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_poisson_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = rng.poisson(lam=4.0, size=200).astype(dtype)
    params = rng.standard_normal(200).astype(dtype) * 0.5 + np.log(4.0)
    new_grad, _ = Poisson().gradient_and_hessian(y, params, natural_gradient=True)
    ref = _ref_poisson_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_negative_binomial_natural_gradient_bit_identical_to_legacy_solve(rng, dtype):
    y = rng.negative_binomial(n=5, p=0.4, size=200).astype(dtype)
    log_n = (rng.standard_normal(200) * 0.3 + np.log(5)).astype(dtype)
    logit_p = (rng.standard_normal(200) * 0.5 + logit(0.4)).astype(dtype)
    params = np.stack([log_n, logit_p], axis=1)
    new_grad, _ = NegativeBinomial().gradient_and_hessian(
        y, params, natural_gradient=True
    )
    ref = _ref_negative_binomial_natural_grad(y, params)
    np.testing.assert_array_equal(new_grad, ref)
