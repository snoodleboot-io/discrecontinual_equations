"""Gauss-Legendre multipliers by periodic QR, and a tight error estimate (DEQ-55).

References are built outside the code under test: DOP853 at ``rtol = 1e-13``
for the period and orbit, and the nontrivial multiplier of a planar cycle as
the exponential of the field's divergence integrated round it (Liouville), at
the same tolerance.
"""

import math
from unittest import TestCase, mock

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation import cycle_continuation
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    _jacobian_monodromy,
    _multiplier_disagreement,
    _product_multipliers,
)
from discrecontinual_equations.continuation.periodic_schur import PeriodicSchurError
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import Alpha, State, Time, VanDerPolField


def _equation(mu: float) -> DifferentialEquation:
    parameters = [Alpha(value=mu)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=VanDerPolField(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _reference(mu: float):
    """Period, orbit on ``[0, 1]`` and both multipliers, by DOP853."""

    def field(_t, u):
        return [u[1], mu * (1.0 - u[0] ** 2) * u[1] - u[0]]

    settled = solve_ivp(
        field,
        [0.0, 400.0],
        [2.0, 0.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )

    def section(_t, u):
        return u[1]

    section.direction = -1.0
    run = solve_ivp(
        field,
        [0.0, 200.0],
        settled.y[:, -1],
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        events=section,
        dense_output=True,
    )
    start, end = run.t_events[0][0], run.t_events[0][1]
    period = end - start

    def with_divergence(t, z):
        return [*field(t, z[:2]), mu * (1.0 - z[0] ** 2)]

    divergence = solve_ivp(
        with_divergence,
        [0.0, period],
        [*run.sol(start), 0.0],
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
    ).y[2, -1]
    multipliers = np.array([1.0, math.exp(divergence)])
    return period, (lambda s: run.sol(start + s * period).T), multipliers


def _point(mu: float, intervals: int, points: int):
    period, orbit, multipliers = _reference(mu)
    continuation = CycleContinuation(
        _equation(mu),
        0,
        intervals,
        phase_index=1,
        collocation="gauss_legendre",
        collocation_points=points,
    )
    seed = CycleSeed(orbit(np.linspace(0.0, 1.0, intervals + 1)), period, mu)
    trace, _ = continuation.trace(seed, 0.05, 0)
    return continuation, trace[0], multipliers


class TestSmallMultiplier(TestCase):
    @pytest.mark.slow
    def test_van_der_pol_at_mu_5_to_its_own_relative_accuracy(self):
        """Exact 7.7385e-38; the explicit product reads it as 1e-15 noise."""
        continuation, point, exact = _point(5.0, 160, 4)
        small = float(np.min(np.abs(point.multipliers)))
        assert abs(small - exact[1]) / exact[1] < 1e-4, small
        unknowns = np.concatenate(
            [point.solution.states.ravel(), [point.solution.period], [5.0]],
        )
        nodes = unknowns.size // 2 - 1
        continuation._blocks(unknowns, nodes)  # noqa: SLF001
        assembled = continuation._assembled  # noqa: SLF001
        assert assembled is not None
        explicit = np.linalg.eigvals(_jacobian_monodromy(assembled[1], 2))
        assert float(np.min(np.abs(explicit))) > 1e-20


class TestEstimateIsTight(TestCase):
    """An upper bound, and within 100 times of the true error, at every m.

    The cases are the verifier grid's worst over-reads under the estimate it
    replaced, which took the largest of the trivial drift, a fourth-order
    Hermite cross-check and twice the doubled-mesh gap: 2.6e4 times at
    ``m = 4`` on 48 intervals, 7.6e4 at ``m = 3`` on 32, and a false refusal
    of van der Pol at ``mu = 6`` that Hermite-Simpson on 320 nodes passed.
    """

    def _ratio(self, mu: float, intervals: int, points: int) -> float:
        _, point, exact = _point(mu, intervals, points)
        true = _multiplier_disagreement(point.multipliers, exact)
        assert point.floquet_error >= true, (mu, intervals, points)
        return point.floquet_error / true

    def test_at_one_point_per_interval(self):
        assert self._ratio(0.5, 32, 1) < 100.0

    def test_at_two_points(self):
        assert self._ratio(0.5, 24, 2) < 100.0

    def test_at_three_points(self):
        assert self._ratio(0.5, 32, 3) < 100.0

    def test_at_four_points(self):
        assert self._ratio(0.5, 48, 4) < 100.0

    @pytest.mark.slow
    def test_no_false_refusal_at_mu_6(self):
        """Measured 4.65e-2 against a true 6.25e-3 under the old estimate."""
        _, point, exact = _point(6.0, 80, 4)
        true = _multiplier_disagreement(point.multipliers, exact)
        assert true <= point.floquet_error < 100.0 * true
        assert point.resolved()


class TestEstimateReadings(TestCase):
    """Each reading in the estimate counts on its own, and a failed one refuses.

    The doubled-mesh solve is replaced here, so the other reading is the only
    one left to see: with the doubled solve agreeing exactly, the estimate is
    the trivial multiplier's drift alone (the Brusselator at ``b = 3``, ``m =
    4`` on 80 intervals is a real case where the drift is the larger); with it
    failing, the point reads infinite and is refused.
    """

    def _trace(self, doubled):
        period, orbit, _ = _reference(0.5)
        continuation = CycleContinuation(
            _equation(0.5),
            0,
            16,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=2,
        )
        seed = CycleSeed(orbit(np.linspace(0.0, 1.0, 17)), period, 0.5)
        with mock.patch.object(CycleContinuation, "_doubled_multipliers", doubled):
            trace, _ = continuation.trace(seed, 0.05, 0)
        return continuation, trace[0]

    def test_the_drift_alone_when_the_doubled_mesh_agrees(self):
        def agreeing(continuation, states, period, parameter):
            dimension = continuation._dimension  # noqa: SLF001
            unknowns = np.concatenate([states.ravel(), [period], [parameter]])
            nodes = unknowns.size // dimension - 1
            reading = continuation._jacobian_multipliers  # noqa: SLF001
            multipliers, _ = reading(unknowns, nodes)
            return multipliers

        _, point = self._trace(agreeing)
        drift = float(np.min(np.abs(point.multipliers - 1.0)))
        assert drift > 0.0
        assert point.floquet_error == drift

    def test_a_failed_doubled_solve_reads_infinite_and_is_refused(self):
        def failing(continuation, states, period, parameter):  # noqa: ARG001
            return np.full(2, np.nan)

        _, point = self._trace(failing)
        assert np.all(np.isfinite(point.multipliers))
        assert point.floquet_error == float("inf")
        assert not point.resolved()


class TestProductMultipliers(TestCase):
    def test_falls_back_to_the_explicit_product_when_periodic_qr_fails(self):
        rng = np.random.default_rng(5)
        transfers = np.eye(3) + 0.2 * rng.standard_normal((6, 3, 3))
        product = np.eye(3)
        for transfer in transfers:
            product = transfer @ product

        def failing(_factors):
            message = "forced"
            raise PeriodicSchurError(message)

        with (
            mock.patch.object(cycle_continuation, "periodic_eigenvalues", failing),
            pytest.warns(RuntimeWarning, match="explicit product"),
        ):
            computed, method = _product_multipliers(transfers)
        np.testing.assert_array_equal(computed, np.linalg.eigvals(product))
        assert method == "explicit_product"

    def test_a_point_reports_the_fallback_it_was_read_by(self):
        """Not silent: the point says which reading its multipliers are."""

        def failing(_factors):
            message = "forced"
            raise PeriodicSchurError(message)

        _, point, _ = _point(0.5, 16, 2)
        assert point.multipliers_method == "periodic_qr"
        with (
            mock.patch.object(cycle_continuation, "periodic_eigenvalues", failing),
            pytest.warns(RuntimeWarning, match="explicit product"),
        ):
            _, fallen, _ = _point(0.5, 16, 2)
        assert fallen.multipliers_method == "explicit_product"
        np.testing.assert_allclose(fallen.multipliers, point.multipliers, atol=1e-9)

    def test_a_real_spectrum_comes_back_real_and_a_pair_stays_complex(self):
        rng = np.random.default_rng(6)
        frames = [np.linalg.qr(rng.standard_normal((2, 2)))[0] for _ in range(5)]
        real = np.array(
            [frames[(k + 1) % 5] @ np.diag([1.1, 0.5]) @ frames[k].T for k in range(5)],
        )
        computed, method = _product_multipliers(real)
        assert method == "periodic_qr"
        assert computed.dtype == float
        assert _multiplier_disagreement(computed, np.array([1.1**5, 0.5**5])) < 1e-12
        angle = 0.3
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        )
        pair, _ = _product_multipliers(np.array([rotation] * 5))
        assert np.iscomplexobj(pair)
        expected = np.exp(1j * 5 * angle) * np.array([1.0, 0.0]) + np.exp(
            -1j * 5 * angle,
        ) * np.array([0.0, 1.0])
        assert _multiplier_disagreement(pair, expected) < 1e-12
