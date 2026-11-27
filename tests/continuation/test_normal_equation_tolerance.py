"""A step the dense ``rcond`` fallback truncated cannot certify convergence (DEQ-156).

A correction proves convergence only if it is the least-squares step, so it
must satisfy the normal equations, ``|J^T (J dx + r)| <= tol |J|_F |r|`` (plus
``|J|_F^2 |dx|`` for rounding, which a truncated step, near zero, does not get).
``np.linalg.lstsq`` with ``rcond = 1e-8``, the last fallback, zeroes every
singular direction below ``1e-8 sigma_max``. Where the residual lies along one
of them, the true Gauss-Newton step is huge and the truncated step is zero.
The defect is then ``sigma |r|``. At ``tol = 1e-6`` the DEQ-120 verifier's
case, ``sigma = 1e-10``, was certified with a correction of 1.6e-16 against a
true step of 1e10.

The synthetic matrices are built from their SVD, so the true step is known
without the code under test.
"""

import math
from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix

from discrecontinual_equations.continuation.connecting_orbit import (
    _NORMAL_EQUATIONS,
    _RCOND,
    _relative_correction,
)


def _matrix(smallest: float):
    """A ``12 x 10`` matrix with singular values ``1..2`` and one ``smallest``."""
    rng = np.random.default_rng(0)
    left, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    right, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    values = np.concatenate([np.linspace(1.0, 2.0, 9), [smallest]])
    return left[:, :10] @ np.diag(values) @ right.T, left, rng.normal(size=10)


class TestTruncatedStepIsNotCertified(TestCase):
    def _truncated(self, smallest: float):
        jacobian, left, unknowns = _matrix(smallest)
        residual = left[:, 9]  # along the direction lstsq truncates
        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=_RCOND)
        return jacobian, residual, step, unknowns

    def test_the_verifiers_case_is_refused(self):
        jacobian, residual, step, unknowns = self._truncated(1.0e-10)
        # The truncated step is ~0; the true one is 1/sigma = 1e10.
        assert np.linalg.norm(step) < 1.0e-12
        correction = _relative_correction(
            csr_matrix(jacobian),
            residual,
            step,
            unknowns,
        )
        assert math.isinf(correction)

    def test_every_truncated_direction_down_to_the_resolved_limit_is_refused(self):
        # lstsq truncates below rcond * sigma_max = 2e-8. The defect sigma |r|
        # is visible to the test down to sigma = tol * |J|_F, 4.6e-11.
        for smallest in (1.5e-8, 1.0e-9, 1.0e-10):
            with self.subTest(smallest=smallest):
                jacobian, residual, step, unknowns = self._truncated(smallest)
                assert np.linalg.norm(step) < 1.0e-6
                correction = _relative_correction(
                    csr_matrix(jacobian),
                    residual,
                    step,
                    unknowns,
                )
                assert math.isinf(correction)

    def test_the_least_squares_step_itself_is_trusted(self):
        jacobian, residual, _, unknowns = self._truncated(1.0e-10)
        exact = -np.linalg.pinv(jacobian, rcond=1.0e-15) @ residual
        correction = _relative_correction(
            csr_matrix(jacobian),
            residual,
            exact,
            unknowns,
        )
        # Trusted, and huge: the iterate is far from converged, as it should say.
        assert correction > 1.0e8

    def test_the_tolerance_is_pinned(self):
        assert _NORMAL_EQUATIONS <= 1.0e-10
        assert _NORMAL_EQUATIONS >= 1.0e-12

    def test_a_step_that_is_not_finite_is_refused(self):
        jacobian, residual, step, unknowns = self._truncated(1.0e-10)
        for bad in (math.inf, math.nan):
            with self.subTest(bad=bad):
                broken = np.full_like(step, bad)
                correction = _relative_correction(
                    csr_matrix(jacobian),
                    residual,
                    broken,
                    unknowns,
                )
                assert math.isinf(correction)
