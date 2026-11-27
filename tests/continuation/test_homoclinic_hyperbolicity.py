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
    _held_partition,
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

    def test_a_finite_difference_shift_across_a_count_change_is_not_refused(self):
        """Within one step of ``x = 0`` the saddle test ``det = -2x`` changes sign.

        At ``x = -dx`` the iterate is a repelling node, and the equilibrium
        shifted by ``dx`` in ``x`` has a zero eigenvalue. The shift only
        differences the iterate's own rows, so it keeps the iterate's partition
        into stable and unstable eigenvalues and the matrix stays square; a count
        change is refused only where an iterate itself reaches it.
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
        matrix = curve._jacobian(shifted, times, 2, sweep, residual)  # noqa: SLF001
        assert matrix.shape == (residual.size, shifted.size)
        assert np.all(np.isfinite(matrix.data))

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


class _NearlyCentral(DeterministicFunction):
    """``x' = x, y' = -y, z' = lam z + c z^2``: a saddle, eigenvalues ``1, -1, lam``.

    The verifier's probe for DEQ-154: a forward difference with step ``dx``
    reads ``lam`` as ``lam + c dx`` at the saddle, and as ``lam + 3 c dx`` at the
    saddle shifted by ``dx`` in ``z``.
    """

    lam = 0.0
    c = 0.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        z = point[2]
        return [
            point[0] + 0 * first,
            -point[1] + 0 * second,
            self.lam * z + self.c * z * z,
        ]


def _nearly_central_curve(lam: float, c: float) -> HomoclinicCurve:
    parameters = [Alpha(value=0.0), BetaOne(value=0.0)]
    variables = [State(), State(), State()]
    field = _NearlyCentral(
        variables=variables,
        parameters=parameters,
        results=[State(), State(), State()],
        time=None,
    )
    field.lam, field.c = lam, c
    equation = DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=field,
    )
    return HomoclinicCurve(equation, 0, 1, MeshSpec(10.0, 20, 0, 0.0))


class TestExactSaddleJacobian(TestCase):
    """The counts come from the exact Jacobian, so a small ``|lam|`` is still read.

    Each case failed on the forward-difference Jacobian: the seed was refused,
    miscounted, or a shift of the Newton matrix was refused (DEQ-154).
    """

    _CASES = (
        (-1.0e-7, 1.0, "seed refused: lam + c dx = 0, a central eigenvalue"),
        (-5.0e-8, 1.0, "seed miscounted: lam + c dx > 0, wrong rows"),
        (-1.0e-6, 10.0, "seed refused at the larger curvature"),
        (-2.0e-7, 1.0, "seed counted, but the z shift reads lam + 3 c dx > 0"),
        (-3.0e-7, 1.0, "seed counted, but the z shift reads lam + 3 c dx = 0"),
    )

    def test_a_small_stable_eigenvalue_is_counted_as_stable(self):
        for lam, c, case in self._CASES:
            with self.subTest(case):
                curve = _nearly_central_curve(lam, c)
                counts = curve._seed_counts(np.zeros(3), 0.0, 0.0)  # noqa: SLF001
                assert counts == (2, 1)

    def test_the_newton_matrix_at_the_saddle_is_square_and_not_refused(self):
        nodes = 21
        times = np.linspace(-10.0, 10.0, nodes)
        unknowns = np.zeros(nodes * 3 + 3 + 1)
        for lam, c, case in self._CASES:
            with self.subTest(case):
                curve = _nearly_central_curve(lam, c)
                residual = curve._residual(  # noqa: SLF001
                    unknowns,
                    times,
                    nodes,
                    3,
                    0.0,
                    (2, 1),
                )
                matrix = curve._jacobian(  # noqa: SLF001
                    unknowns,
                    times,
                    3,
                    0.0,
                    residual,
                )
                assert matrix.shape == (unknowns.size, unknowns.size)

    def test_the_saddle_solves(self):
        for lam, c, case in self._CASES:
            with self.subTest(case):
                curve = _nearly_central_curve(lam, c)
                point = curve.solve_point(0.0, np.zeros((21, 3)), np.zeros(3), 0.0)
                assert point is not None

    def test_a_field_without_jets_is_counted_by_the_forward_difference(self):
        """``math.pow`` refuses a Taylor jet, so the counts fall back to a difference.

        At ``lam = -1e-6, c = 1`` the difference reads ``lam + c dx < 0``: still
        a saddle with two stable eigenvalues, now read without automatic
        differentiation.
        """
        curve = _nearly_central_curve(-1.0e-6, 1.0)
        field = curve._equation.derivative  # noqa: SLF001
        field.eval = lambda point, time=None: [  # noqa: ARG005 (base signature)
            point[0],
            -point[1],
            field.lam * point[2] + field.c * math.pow(point[2], 2),
        ]
        assert curve._seed_counts(np.zeros(3), 0.0, 0.0) == (2, 1)  # noqa: SLF001
        jacobian = curve._saddle_jacobian(np.zeros(3), 0.0, 0.0)  # noqa: SLF001
        assert abs(jacobian[2, 2] - (-1.0e-6 + 1.0e-7)) < 1.0e-12


class TestHeldPartition(TestCase):
    """A shifted saddle's eigenvalues inherit the partition of the unshifted one."""

    def test_a_reordered_shift_keeps_the_reference_rows_in_their_order(self):
        """Shifted eigenvector ``j`` lies along reference eigenvector ``2, 0, 1``."""
        reference = (np.eye(3), [1, 2], [0])
        shifted = np.eye(3)[:, [2, 0, 1]]
        assert _held_partition(shifted, reference) == ([2, 0], [1])

    def test_a_complex_pair_keeps_its_two_members_apart(self):
        """A perturbed saddle-focus, its eigenvectors listed in reverse order."""
        jacobian = np.array([[-0.5, 2.0, 0.0], [-2.0, -0.5, 0.0], [0.0, 0.0, 3.0]])
        values, vectors = np.linalg.eig(jacobian)
        stable = [i for i in range(3) if values[i].real < 0]
        unstable = [i for i in range(3) if values[i].real > 0]
        perturbation = 1.0e-7 * np.arange(9.0).reshape(3, 3)
        moved, moved_vectors = np.linalg.eig(jacobian + perturbation)
        order = np.argsort(-moved.imag)[::-1]
        held = _held_partition(
            moved_vectors[:, order],
            (np.linalg.inv(vectors), stable, unstable),
        )
        for reference, shifted in zip(
            [*stable, *unstable],
            [*held[0], *held[1]],
            strict=True,
        ):
            assert abs(moved[order][shifted] - values[reference]) < 1.0e-5


class _CrossingPair(DeterministicFunction):
    """``x' = x, y' = -y, z' = -d z + z^2, w' = d w + w^2 - z w``.

    At the origin the eigenvalues are ``1, -1, -d, d``: a saddle with a stable
    and an unstable eigenvalue both near zero. The Jacobian is diagonal at every
    point with ``w = 0``, so the left eigenvectors are the unit vectors there,
    and shifting the saddle by ``dx`` in ``z`` moves the ``z`` eigenvalue to
    ``-d + 2 dx`` and the ``w`` one to ``d - dx``: for ``d < dx`` they pass each
    other, and each lies nearer the other's unshifted eigenvalue.
    """

    d = 0.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        z, w = point[2], point[3]
        return [
            point[0] + 0 * first,
            -point[1] + 0 * second,
            -self.d * z + z * z,
            self.d * w + w * w - z * w,
        ]


class TestHeldPartitionFollowsEigenvectors(TestCase):
    def test_a_stable_and_an_unstable_eigenvalue_passing_keep_their_rows(self):
        """The differenced tail column of ``z_eq`` is exact: ``-1`` in the
        departure row of ``z``, ``0`` in the arrival row of ``w``.

        Matching the shifted eigenvalues to the nearest unshifted ones swaps the
        ``z`` and ``w`` rows here and gives entries of order ``|z - w| / dx``.
        """
        parameters = [Alpha(value=0.0), BetaOne(value=0.0)]
        variables = [State() for _ in range(4)]
        field = _CrossingPair(
            variables=variables,
            parameters=parameters,
            results=[State() for _ in range(4)],
            time=None,
        )
        field.d = 1.0e-8
        equation = DifferentialEquation(
            variables=variables,
            time=Time(),
            parameters=parameters,
            derivative=field,
        )
        curve = HomoclinicCurve(equation, 0, 1, MeshSpec(10.0, 4, 0, 0.0))
        nodes, dimension = 5, 4
        times = np.linspace(-10.0, 10.0, nodes)
        states = np.zeros((nodes, dimension))
        states[0] = [0.0, 0.1, 0.2, -0.3]
        states[-1] = [0.1, 0.0, -0.2, 0.3]
        unknowns = np.concatenate([states.ravel(), np.zeros(dimension), [0.0]])
        residual = curve._residual(  # noqa: SLF001
            unknowns,
            times,
            nodes,
            dimension,
            0.0,
            (2, 2),
        )
        matrix = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            dimension,
            0.0,
            residual,
        ).toarray()
        boundary = (nodes - 1) * dimension + dimension
        column = nodes * dimension + 2
        # rows: departure y, departure z (stable); arrival x, arrival w (unstable)
        assert abs(matrix[boundary + 1, column] + 1.0) < 1.0e-6
        assert abs(matrix[boundary + 3, column]) < 1.0e-6


class _SaddleLoop(DeterministicFunction):
    """``x' = y, y' = (1 + b) x - x^2 + a y, z' = g(z, a)``: a loop times a line.

    At ``a = 0`` the ``(x, y)`` plane has the homoclinic loop
    ``x = 1.5 (1 + b) sech^2(sqrt(1 + b) t / 2)`` to the origin for every ``b``,
    and ``z = 0`` is an equilibrium of ``z' = g``. The saddle's third eigenvalue
    is ``g_z(0, a)``: the verifier's public-API probe for DEQ-154 and DEQ-172.
    """

    third = staticmethod(lambda z, a: 0.0 * z * a)

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        a = self.parameters[0].value
        b = self.parameters[1].value
        x, y, z = point
        return [y, (1 + b) * x - x * x + a * y, self.third(z, a)]


def _saddle_loop_curve(third, curve_class=HomoclinicCurve) -> HomoclinicCurve:
    parameters = [Alpha(value=0.0), BetaOne(value=0.0)]
    variables = [State(), State(), State()]
    field = _SaddleLoop(
        variables=variables,
        parameters=parameters,
        results=[State(), State(), State()],
        time=None,
    )
    field.third = third
    equation = DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=field,
    )
    return curve_class(equation, 0, 1, MeshSpec(12.0, 120, 1, 0.0))


def _perturbed_loop() -> np.ndarray:
    """The ``b = 0`` loop with ``1e-3`` noise on every node, ``a`` seeded at 0.01.

    Newton's first step takes ``a`` from 0.01 past the curve to ``-3.6e-5``,
    then back to ``0``: an iterate far from convergence that the next corrects.
    """
    nodes = 121
    times = np.linspace(-12.0, 12.0, nodes)
    x = 1.5 / np.cosh(times / 2.0) ** 2
    seed = np.column_stack([x, np.gradient(x, times), np.zeros(nodes)])
    seed += 1.0e-3 * np.random.default_rng(0).standard_normal((nodes, 3))
    seed[nodes // 2, 1] = 0.0
    return seed


# Seed and starting parameter of every solve below.
_LOOP_SEED_A = 0.01


class _RecordingCurve(HomoclinicCurve):
    """Records the saddle's counted ``(stable, unstable)`` at every iterate."""

    __slots__ = ["iterates"]

    def _follow(self, equilibrium, continuation, sweep, reference):
        if not hasattr(self, "iterates"):
            self.iterates = []
        self.iterates.append(
            (self._counts(equilibrium, float(continuation), sweep), continuation),
        )
        return super()._follow(equilibrium, continuation, sweep, reference)


class TestCountsAreEnforcedAtTheAcceptedPoint(TestCase):
    """DEQ-172: a far iterate's counts do not stop a solve; the accepted point's do."""

    def test_the_verifier_probe_saddles_solve_and_trace(self):
        """The two of the DEQ-154 verifier's 24 probe saddles that failed.

        Each has ``lam = -c dx`` for the difference step ``dx = 1e-7``: the
        differenced equilibrium row read ``lam + c dx = 0``, the sparse LU was
        exactly singular, and the least-squares fallback moved the saddle by
        ``2e-5`` and ``1e-4`` in ``z``, far past ``|lam| / 2c``. Both were
        refused on that first iterate.
        """
        for lam, c in ((-1.0e-7, 1.0), (-1.0e-6, 10.0)):
            with self.subTest(lam=lam, c=c):
                curve = _saddle_loop_curve(
                    lambda z, _a, lam=lam, c=c: lam * z + c * z * z,
                )
                point = curve.solve_point(
                    0.0,
                    _perturbed_loop(),
                    np.zeros(3),
                    _LOOP_SEED_A,
                )
                assert point is not None
                assert abs(point.continuation) < 1.0e-9
                assert np.max(np.abs(point.orbit.states[:, 2])) < 1.0e-9
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    points = curve.trace(
                        [0.0, 0.05, 0.1],
                        _perturbed_loop(),
                        np.zeros(3),
                        _LOOP_SEED_A,
                    )
                assert len(points) == 3

    def test_the_equilibrium_rows_hold_the_exact_saddle_jacobian(self):
        """At ``lam = -c dx`` a forward difference reads the ``z`` entry as 0."""
        lam, c = -1.0e-7, 1.0
        curve = _saddle_loop_curve(lambda z, _a: lam * z + c * z * z)
        nodes = 121
        times = np.linspace(-12.0, 12.0, nodes)
        unknowns = np.concatenate(
            [_perturbed_loop().ravel(), np.zeros(3), [_LOOP_SEED_A]],
        )
        residual = curve._residual(unknowns, times, nodes, 3, 0.0)  # noqa: SLF001
        matrix = curve._jacobian(unknowns, times, 3, 0.0, residual)  # noqa: SLF001
        row = (nodes - 1) * 3 + 2
        column = nodes * 3 + 2
        assert abs(matrix[row, column] - lam) < 1.0e-20

    def test_an_iterate_that_loses_the_counts_and_regains_them_is_not_refused(self):
        """``z' = (-1e-5 - a) z``: stable at the seed and on the curve (``a = 0``).

        Newton's first step overshoots to ``a = -3.6e-5``, where ``-1e-5 - a``
        is positive: that iterate has one stable and two unstable eigenvalues.
        It was refused there; the next step returns to ``a = 0``.
        """
        curve = _saddle_loop_curve(
            lambda z, a: (-1.0e-5 - a) * z,
            curve_class=_RecordingCurve,
        )
        point = curve.solve_point(0.0, _perturbed_loop(), np.zeros(3), _LOOP_SEED_A)
        assert point is not None
        assert abs(point.continuation) < 1.0e-9
        crossed = [a for counts, a in curve.iterates if counts == (1, 2)]
        assert crossed, curve.iterates
        assert all(a < -1.0e-5 for a in crossed)

    def test_a_count_change_at_the_converged_point_is_refused(self):
        """``z' = (1e-3 - a) z``: stable at the seed ``a = 0.01``, unstable at 0.

        The rows built for the seed's partition converge at ``a = 0``, where
        the saddle has one stable and two unstable eigenvalues: a solution of
        a different problem, refused there and not at an iterate on the way.
        """
        curve = _saddle_loop_curve(lambda z, a: (1.0e-3 - a) * z)
        with pytest.raises(EigenvalueCountError, match="converged where") as caught:
            curve.solve_point(0.0, _perturbed_loop(), np.zeros(3), _LOOP_SEED_A)
        assert caught.value.expected == (2, 1)
        assert caught.value.found == (1, 2)
        assert abs(caught.value.continuation) < 1.0e-9
        with self.assertWarnsRegex(RuntimeWarning, "1 stable and 2 unstable"):
            points = curve.trace(
                [0.0, 0.05],
                _perturbed_loop(),
                np.zeros(3),
                _LOOP_SEED_A,
            )
        assert points == []
