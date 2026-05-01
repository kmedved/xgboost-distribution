"""Negative binomial distribution"""

from collections import namedtuple

import numpy as np
from scipy.special import digamma, expit
from scipy.stats import nbinom

from xgboost_distribution.distributions.base import BaseDistribution
from xgboost_distribution.distributions.utils import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    check_all_ge_zero,
    check_all_integer,
    safe_exp,
)

Params = namedtuple("Params", ("n", "p"))

# Floor for the Fisher diagonal entries. Both `n*p/(p+1)` and `n*(1-p)` can
# round to zero in float32 at the parameter clipping boundaries — `expit`
# saturates to exactly 1.0 at MAX_EXPONENT (so `1-p == 0`), and `n*p` can
# underflow when both `log_n` and `logit_p` hit their MIN_EXPONENT clips.
# Flooring at 1e-30 keeps the natural gradient finite for ordinary finite
# targets at the clipping boundaries. It is intended to engage only near
# degenerate parameter regions where the diagonal approximation is
# numerically singular or near-singular.
DIAG_FLOOR = np.float32(1e-30)


class NegativeBinomial(BaseDistribution):
    """Negative binomial distribution with log score

    Definition:

        f(k) = p^n (1 - p)^k binomial(n + k - 1, n - 1)

    with parameter (n, p), where n >= 0 and 1 >= p >= 0

    We reparameterize:

        n -> log(n) = a        |  e^a = n
        p -> log(p/(1-p)) = b  |  e^b = p / (1-p)   |  p = 1 / (1 + e^-b)

    The gradients are:

        d/da -log[f(k)] = -e^a [ digamma(k+e^a) - digamma(e^a) + log(p) ]
                        = -n   [ digamma(k+n) - digamma(n) + log(p) ]

        d/db -log[f(k)] = (k e^b - e^a) / (e^b + 1)
                        = (k - e^a e^-b) / (e^-b + 1)
                        = p * (k - e^a e^-b)
                        = p * (k - n e^-b)

    Under the stated pmf, the true reparameterised Fisher information has a
    nonzero off-diagonal term `F_ab = -n(1-p)`. We use a **diagonal
    approximation** — the cross term is dropped — to keep the natural-gradient
    update cheap (no per-sample matrix solve). The two diagonal entries are
    derived independently below.

    F_bb (b = logit(p) parameter):

        d/db -log[f(k)] = p*k - n*(1-p)
        d²/db² -log[f(k)] = p*(1-p)*(k + n)
        F_bb = E[ d²/db² -log[f(k)] ] = p*(1-p) * (E[K] + n)
             = p*(1-p) * (n*(1-p)/p + n)              [E[K] = n(1-p)/p]
             = n*(1-p)

    F_aa (a = log(n) parameter):

        d/da -log[f(k)] = -n*[ψ(k+n) - ψ(n) + log(p)]
        F_aa involves E[trigamma(K+n)]; we use the literature approximation

            I(n) ~ p / [ n*(p+1) ]   ->   F_aa = I_r(n) = n*p / (p+1)

        from http://erepository.uonbi.ac.ke:8080/xmlui/handle/123456789/33803

    Hence the diagonal Fisher approximation used in code:

        [ n*p / (p+1), 0       ]
        [ 0,           n*(1-p) ]

    Each diagonal entry is also floored at `DIAG_FLOOR` (≈ 1e-30, defined at
    module level) before division. This is a numerical safeguard — at the
    parameter clipping boundaries (e.g. `expit(MAX_EXPONENT)` saturates to
    exactly 1.0 in float32, making `1-p == 0`), the diagonals would otherwise
    round to zero and produce inf/nan natural gradients. The floor is
    intended to engage only near degenerate parameter regions where the
    diagonal approximation is numerically singular or near-singular.

    The raw gradient is computed in float64 (and the column-1 form is
    rewritten as `p*y - n*(1-p)` rather than `p*(y - n*(1-p)/p)`) to keep
    the intermediates from overflowing float32 at the upper-clip boundary
    (`log_n = MAX_EXPONENT`, `logit_p = MIN_EXPONENT`). The final
    natural gradient still casts back into the float32 destination. As a
    result, NegativeBinomial is **no longer bit-identical** to the legacy
    `numpy.linalg.solve` path; it matches within a few float32 ulps on
    non-extreme inputs and stays finite at the clipping boundaries.

    Note: the legacy code (xgboost-distribution <= 0.4.0) used `F_bb = n*p`,
    which is incorrect under the stated pmf; this was corrected in commit
    history. The off-diagonal cross term is still dropped as an
    approximation.

    Ref:

        https://www.wolframalpha.com/input/?i=d%2Fda+-log%28+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D+%5E%28e%5Ea%29+%281+-+%5B1+%2F+%281+%2B+e%5E%28-b%29%29%5D%29%5Ek+binomial%28%28e%5Ea%29+%2B+k+-+1%2C+%28e%5Ea%29+-+1%29+%29

    """

    @property
    def params(self):
        return Params._fields

    def check_target(self, y):
        check_all_integer(y)
        check_all_ge_zero(y)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        n, p = self.predict(params)
        grad = np.zeros(shape=(len(y), 2), dtype="float32")

        if natural_gradient:
            # Diagonal Fisher approximation: F = diag(n*p/(p+1), n*(1-p)).
            # See class docstring for the derivation and a note on why the
            # cross term is dropped.
            #
            # Two numerical safeguards:
            #   1. Compute the raw gradient in float64. At the upper-clip
            #      boundary `log_n = MAX_EXPONENT, logit_p = MIN_EXPONENT`,
            #      `n ≈ 1.25e38` and `log(p) ≈ -73`, so `-n * log(p)`
            #      overflows float32 even though the natural gradient
            #      (after dividing by F_aa) is well within float32 range.
            #   2. Use the algebraic rewrite `p*y - n*(1-p)` for column 1
            #      instead of `p*(y - n*(1-p)/p)`. The latter has an
            #      `n*(1-p)/p` intermediate that overflows when `p → 0`.
            # Diagonal entries are cast to float32 and floored at
            # DIAG_FLOOR to avoid float32 underflow when `expit` saturates
            # to 1.0 or `n*p` underflows.
            n64 = np.asarray(n, dtype="float64")
            p64 = np.asarray(p, dtype="float64")
            y64 = np.asarray(y, dtype="float64")

            raw0 = -n64 * (digamma(y64 + n64) - digamma(n64) + np.log(p64))
            raw1 = p64 * y64 - n64 * (1 - p64)

            diag0 = np.maximum(
                np.asarray((n64 * p64) / (p64 + 1), dtype="float32"), DIAG_FLOOR
            ).astype("float64")
            diag1 = np.maximum(
                np.asarray(n64 * (1 - p64), dtype="float32"), DIAG_FLOOR
            ).astype("float64")

            grad[:, 0] = raw0 / diag0
            grad[:, 1] = raw1 / diag1
            hess = np.ones(shape=(len(y), 2), dtype="float32")  # constant hessian
        else:
            raise NotImplementedError(
                "Normal gradients are currently not supported by this "
                "distribution. Please use natural gradients!"
            )

        return grad, hess

    def loss(self, y, params):
        n, p = self.predict(params)
        return "NegativeBinomial-NLL", -nbinom.logpmf(y, n=n, p=p)

    def predict(self, params):
        log_n, logits = params[:, 0], params[:, 1]

        n = safe_exp(log_n)
        logits = np.clip(logits, a_min=MIN_EXPONENT, a_max=MAX_EXPONENT)

        p = expit(logits)
        return Params(n=n, p=p)

    def starting_params(self, y):
        # TODO: starting params can matter a lot?
        return Params(n=np.log(np.mean(y)), p=0)  # expit(0) = 0.5
