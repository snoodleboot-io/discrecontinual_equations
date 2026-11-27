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
from unittest import TestCase, mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from discrecontinual_equations.continuation.connecting_orbit import (
    _DAMPING,
    _NORMAL_EQUATIONS,
    _RCOND,
    HeteroclinicOrbit,
    NotConvergedError,
    Terminus,
    _dense_step,
    _relative_correction,
)
from tests.continuation.test_connecting_convergence import _tanh_problem
from tests.continuation.test_heteroclinic_network_subspace import _leak_network


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


class TestRankDeficientStepIsNeverCertified(TestCase):
    """A step computed without full rank certifies nothing (DEQ-180).

    The normal-equation test sees a truncated direction only when the residual
    lies mostly along it. The DEQ-155 verifier's mixed cases
    (outputs/verification/verify-deq-155/normal_eq_mixed_residual.out): with
    ``sigma = 1e-9`` and ten times as much residual in resolved directions, a
    step whose truncated part is 0.1 was certified with a correction of 5e-10;
    at ``sigma = 1e-10`` an equal split was enough. The dense fallback now
    reports its rank, and a rank-deficient step is never certified.
    """

    _SIGMAS = (1.5e-8, 1.0e-9, 1.0e-10)
    _RATIOS = (0.0, 1.0, 10.0, 100.0)

    def _mixed(self, smallest: float, ratio: float):
        jacobian, left, unknowns = _matrix(smallest)
        resolved = left[:, :9] @ np.linspace(1.0, 2.0, 9)
        resolved /= np.linalg.norm(resolved)
        # The verifier's scale: 1e-10 along the truncated direction, so the
        # true step there is 1e-10 / sigma (0.0067, 0.1 and 1).
        residual = 1.0e-10 * (left[:, 9] + ratio * resolved)
        return csr_matrix(jacobian), residual, unknowns

    def test_the_verifiers_mixed_cases_are_refused(self):
        for smallest in self._SIGMAS:
            for ratio in self._RATIOS:
                with self.subTest(sigma=smallest, ratio=ratio):
                    jacobian, residual, unknowns = self._mixed(smallest, ratio)
                    step, full_rank = _dense_step(jacobian, residual)
                    assert not full_rank
                    correction = _relative_correction(
                        jacobian,
                        residual,
                        step,
                        unknowns,
                        resolved=full_rank,
                    )
                    assert math.isinf(correction)

    def test_the_normal_equations_alone_were_blind_to_them(self):
        """The gap the rank report closes: these pass the defect test."""
        for smallest, ratio in ((1.0e-9, 10.0), (1.0e-10, 1.0)):
            with self.subTest(sigma=smallest, ratio=ratio):
                jacobian, residual, unknowns = self._mixed(smallest, ratio)
                step, _ = _dense_step(jacobian, residual)
                blind = _relative_correction(jacobian, residual, step, unknowns)
                # Measured 5e-10 and 5e-11, under the 1e-7 that certifies.
                assert blind < 1.0e-7

    def test_a_full_rank_dense_step_is_still_trusted(self):
        jacobian, residual, unknowns = self._mixed(1.0e-3, 10.0)
        step, full_rank = _dense_step(jacobian, residual)
        assert full_rank
        exact = -np.linalg.pinv(jacobian.toarray(), rcond=1.0e-15) @ residual
        np.testing.assert_allclose(step, exact, rtol=1e-9, atol=0.0)
        correction = _relative_correction(
            jacobian,
            residual,
            step,
            unknowns,
            resolved=full_rank,
        )
        assert math.isfinite(correction)

    def test_the_sparse_step_reports_rank_too(self):
        """Its rank test is ``sigma_min > _RCOND sigma_max`` on its estimates.

        In the singular basis the step is the Levenberg-Marquardt one with
        ``lambda = (_DAMPING sigma_max)^2``: resolved directions get the
        Gauss-Newton component ``-(u . r) / sigma``, and a direction far below
        the floor gets at most ``|u . r| / (2 sqrt(lambda))``, the filter
        ``sigma / (sigma^2 + lambda)`` at its peak.
        """
        solver = _tanh_problem()[0]
        for smallest, full_rank in (
            (1.0e-3, True),
            (1.0e-6, True),
            (1.0e-12, False),
            (1.0e-17, False),
        ):
            with self.subTest(sigma=smallest):
                jacobian, residual, _ = self._mixed(smallest, 1.0)
                step, resolved = solver._newton_step(  # noqa: SLF001
                    jacobian,
                    residual,
                )
                assert resolved == full_rank
                left, values, right = np.linalg.svd(jacobian.toarray())
                components = right @ step
                projected = left[:, :10].T @ residual
                gauss_newton = -projected / values
                # The augmented solve is as accurate as QR: eps cond(J) |x|.
                accuracy = 1.0e-9 * float(np.linalg.norm(step))
                if full_rank:
                    np.testing.assert_allclose(
                        components,
                        gauss_newton,
                        rtol=1e-8,
                        atol=accuracy,
                    )
                    continue
                np.testing.assert_allclose(
                    components[:9],
                    gauss_newton[:9],
                    rtol=1e-8,
                    atol=accuracy,
                )
                peak = abs(projected[9]) / (2.0 * _DAMPING * values[0])
                assert abs(components[9]) <= 1.01 * peak
                # Gauss-Newton's is 1e-10 / sigma, 100 and 1e7; measured 2.8e-3
                # of it at sigma = 1e-12, just under the floor, and 5e-11 at 1e-17.
                assert abs(components[9]) < 1.0e-2 * abs(gauss_newton[9])

    def test_a_solve_whose_steps_are_rank_deficient_is_certified_only_by_residual(
        self,
    ):
        """The leaking plane converges to a least-squares minimum, certified by
        its correction (measured 1e-14). Reported rank-deficient, the same
        steps certify nothing, and the solve refuses."""
        network, connection = _leak_network(0.05)
        source = network.saddle("xi1")
        target = network.saddle("xi2")
        mesh = network.mesh(connection, half_length=20.0, intervals=100)
        solver = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        seed = network.seed(connection, mesh)
        assert not solver.solve(seed).residual.is_trajectory
        full = HeteroclinicOrbit._newton_step  # noqa: SLF001 (wrapping the step)

        def deficient(self, jacobian, residual):
            return full(self, jacobian, residual)[0], False

        with (
            mock.patch.object(HeteroclinicOrbit, "_newton_step", deficient),
            pytest.raises(NotConvergedError) as caught,
        ):
            solver.solve(seed)
        assert math.isinf(caught.value.correction)
