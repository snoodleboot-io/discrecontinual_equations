"""The homoclinic curve refuses a saddle whose eigenvalue counts change.

The projection boundary conditions give one row per stable and one per unstable
eigenvalue, so the Newton matrix is square only while the saddle keeps the
counts it started with. On the Bogdanov-Takens field ``x' = y,
y' = b1 + b2 y + x^2 + x y`` the equilibria are ``(+-sqrt(-b1), 0)``: there are
none for ``b1 > 0``, and the one at ``x`` is a saddle exactly when ``x > 0``.
"""

import math
import warnings
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.connecting_orbit import MeshSpec
from discrecontinual_equations.continuation.homoclinic_curve import (
    EigenvalueCountError,
    HomoclinicCurve,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, BetaOne, State, Time
from tests.continuation.test_connecting_newton import _bogdanov_takens_problem

_EPSILON = 0.25


class _CentralDirection(DeterministicFunction):
    """``x' = x, y' = -y, z' = a z + b``: at ``a = b = 0`` a saddle with a centre."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        return [point[0], -point[1], first * point[2] + second]


def _central_curve() -> HomoclinicCurve:
    parameters = [Alpha(value=0.0), BetaOne(value=0.0)]
    variables = [State(), State(), State()]
    equation = DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=_CentralDirection(
            variables=variables,
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        ),
    )
    return HomoclinicCurve(equation, 0, 1, MeshSpec(10.0, 20, 0, 0.0))


def _seed(curve_problem):
    curve, times, unknowns, sweep = curve_problem
    nodes = times.size
    orbit = unknowns[: nodes * 2].reshape(nodes, 2)
    equilibrium = unknowns[nodes * 2 : nodes * 2 + 2]
    return curve, orbit, equilibrium, float(unknowns[-1]), sweep


class TestEigenvalueCountRefusal(TestCase):
    def test_b2_free_from_the_shared_seed_is_refused_with_counts_and_parameter(self):
        """The shared seed puts ``b1 = 5 eps^2 / 7 > 0``: no saddle exists there.

        Newton wanders off in ``b2`` until the saddle stops being one; that used
        to surface as a broadcast ``ValueError`` from the Jacobian assembly.
        """
        curve, orbit, equilibrium, continuation, sweep = _seed(
            _bogdanov_takens_problem(30, continuation_index=1),
        )
        with pytest.raises(EigenvalueCountError) as caught:
            curve.solve_point(sweep, orbit, equilibrium, continuation)
        error = caught.value
        assert isinstance(error, ValueError)
        assert error.expected == (1, 1)
        assert error.found != (1, 1)
        assert error.sweep == sweep
        message = str(error)
        assert "broadcast" not in message
        assert "1 stable and 1 unstable" in message
        stable, unstable = error.found
        assert f"{stable} stable and {unstable} unstable" in message
        assert f"b2 = {error.continuation:.6g}" in message
        assert f"b1 = {sweep:.6g}" in message

    def test_b2_free_converges_from_a_seed_on_the_curve(self):
        """Swap the seed's parameter values and ``b2`` free reaches the same point."""
        curve, orbit, equilibrium, continuation, sweep = _seed(
            _bogdanov_takens_problem(30, continuation_index=0),
        )
        held = curve.solve_point(sweep, orbit, equilibrium, continuation)
        assert held is not None
        free, *_ = _seed(_bogdanov_takens_problem(30, continuation_index=1))
        point = free.solve_point(
            held.continuation,
            held.orbit.states,
            held.equilibrium,
            sweep,
        )
        assert point is not None
        assert abs(point.continuation - sweep) < 1.0e-9 * sweep
        assert np.max(np.abs(point.orbit.states - held.orbit.states)) < 1.0e-9

    def test_a_finite_difference_step_across_a_count_change_is_refused(self):
        """Within one step of ``x = 0`` the saddle test ``det = -2x`` flips sign.

        The field Jacobian is itself a forward difference, which reads ``2x`` as
        ``2x + dx``; so at ``x = -dx`` the unshifted point is a repelling node and
        the point shifted by ``dx`` is a saddle. Their boundary rows have
        different lengths.
        """
        curve, times, unknowns, sweep = _bogdanov_takens_problem(30)
        nodes = times.size
        shifted = unknowns.copy()
        shifted[nodes * 2] = -1.0e-7
        shifted[-1] = -(shifted[nodes * 2] ** 2)
        residual = curve._residual(  # noqa: SLF001
            shifted,
            times,
            nodes,
            2,
            sweep,
        )
        with pytest.raises(EigenvalueCountError) as caught:
            curve._jacobian(shifted, times, 2, sweep, residual)  # noqa: SLF001
        assert caught.value.expected == (0, 2)
        assert caught.value.found == (1, 1)

    def test_a_seed_that_is_not_a_hyperbolic_saddle_is_refused(self):
        """At ``x = -eps^2`` the trace ``b2 + x`` is negative: a stable focus."""
        curve, orbit, _, _, sweep = _seed(_bogdanov_takens_problem(30))
        node = np.array([-(_EPSILON**2), 0.0])
        with pytest.raises(
            EigenvalueCountError,
            match="not a hyperbolic saddle",
        ) as caught:
            curve.solve_point(sweep, orbit, node, -(_EPSILON**4))
        assert caught.value.expected is None
        assert caught.value.found == (2, 0)

    def test_a_seed_with_a_central_eigenvalue_is_refused(self):
        """One stable and one unstable eigenvalue, but a zero third: not hyperbolic.

        The field is linear about the origin, so its differenced Jacobian is
        exact there and the zero eigenvalue is read as zero.
        """
        curve = _central_curve()
        with pytest.raises(EigenvalueCountError, match="none central") as caught:
            curve.solve_point(0.0, np.zeros((21, 3)), np.zeros(3), 0.0)
        assert caught.value.found == (1, 1)
        assert "a = 0" in str(caught.value)

    def test_an_iterate_that_loses_only_an_unstable_eigenvalue_is_refused(self):
        """For ``a > 0`` the counts are one stable and two unstable; at ``a = 0``
        the third eigenvalue is zero, so the stable count is unchanged and only
        the unstable one drops. The boundary rows are one shorter all the same.
        """
        curve = _central_curve()
        zero = np.zeros(3)
        with pytest.raises(EigenvalueCountError) as caught:
            curve._tail(zero, zero, zero, (0.0, 0.0), (1, 2))  # noqa: SLF001
        assert caught.value.expected == (1, 2)
        assert caught.value.found == (1, 1)

    def test_trace_stops_where_the_saddle_is_lost(self):
        """The curve keeps the points it solved, as the other curve tracers do."""
        curve, orbit, equilibrium, continuation, sweep = _seed(
            _bogdanov_takens_problem(30, continuation_index=0),
        )
        held = curve.solve_point(sweep, orbit, equilibrium, continuation)
        assert held is not None
        free, *_ = _seed(_bogdanov_takens_problem(30, continuation_index=1))
        sweeps = [held.continuation, 5.0 / 7.0 * _EPSILON**2, held.continuation]
        with self.assertWarnsRegex(RuntimeWarning, "1 stable and 1 unstable"):
            points = free.trace(
                sweeps,
                held.orbit.states,
                held.equilibrium,
                sweep,
            )
        assert len(points) == 1
        assert math.isclose(points[0].continuation, sweep, rel_tol=1.0e-9)

    def test_a_trace_that_keeps_its_saddle_does_not_warn(self):
        curve, orbit, equilibrium, continuation, sweep = _seed(
            _bogdanov_takens_problem(30, continuation_index=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            points = curve.trace(
                [sweep, 1.02 * sweep],
                orbit,
                equilibrium,
                continuation,
            )
        assert len(points) == 2
