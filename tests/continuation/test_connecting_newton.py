"""The Newton step of the connecting-orbit and homoclinic-curve solvers.

Trapezoidal collocation couples only neighbouring nodes, so both Newton systems
are block-banded, bordered by the projection boundary conditions, the phase
condition and - for the curve - the equilibrium and the free parameter. These
tests hold the solvers to that structure: the Jacobian is assembled sparse with
O(N) nonzeros, it agrees with central differences of the residual (a reference
that shares nothing with the assembly), the step is the least-squares step a
dense factorisation gives, and the work of one iteration grows linearly in N.
"""

import math
import tracemalloc
from unittest import TestCase

import numpy as np
from scipy.sparse import issparse

from discrecontinual_equations.continuation.connecting_orbit import (
    HomoclinicOrbit,
    MeshSpec,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import (
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    State,
    Time,
    _saddle,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
_DIFFERENCE = 1.0e-6


class NonAnalyticSaddle(DeterministicFunction):
    """The saddle ``x' = y, y' = x - x^2`` written so Taylor jets are refused.

    ``math.pow`` needs a real number, so automatic differentiation raises and the
    solver must fall back to finite differences - on the same sparse pattern.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [y, x - math.pow(x, 2)]


class CountingBogdanovTakens(BogdanovTakensField):
    """The Bogdanov-Takens field, counting how often it is evaluated."""

    calls = 0

    def eval(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        CountingBogdanovTakens.calls += 1
        return super().eval(point, time)


def _saddle_problem(intervals: int, field=None):
    half = 15.0
    solver = HomoclinicOrbit(
        field if field is not None else _saddle(),
        np.zeros(2),
        _SADDLE_JACOBIAN,
        MeshSpec(half, intervals, phase_index=1, phase_value=0.0),
    )
    times = np.linspace(-half, half, intervals + 1)
    seed = np.column_stack(
        [
            1.2 / np.cosh(times / 2.5) ** 2,
            -0.5 * np.tanh(times) / np.cosh(times) ** 2,
        ],
    )
    return solver, times, seed


def _central_differences(residual, unknowns: np.ndarray) -> np.ndarray:
    """Reference Jacobian by central differences of the residual, column by column."""
    base = residual(unknowns)
    columns = np.empty((base.size, unknowns.size))
    for j in range(unknowns.size):
        ahead, behind = unknowns.copy(), unknowns.copy()
        ahead[j] += _DIFFERENCE
        behind[j] -= _DIFFERENCE
        columns[:, j] = (residual(ahead) - residual(behind)) / (2.0 * _DIFFERENCE)
    return columns


def _bogdanov_takens_problem(intervals: int, field_type=BogdanovTakensField):
    parameters = [BetaOne(value=0.0), BetaTwo(value=0.0)]
    equation = DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=field_type(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )
    epsilon, span = 0.25, 5.0
    nodes = intervals + 1
    half = span * math.sqrt(2.0) / epsilon
    times = np.linspace(-half, half, nodes)
    scaled = epsilon * times / math.sqrt(2.0)
    profile = 1.0 / np.cosh(scaled) ** 2
    orbit = np.column_stack(
        [
            epsilon**2 * (1.0 - 3.0 * profile),
            epsilon**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(scaled),
        ],
    )
    curve = HomoclinicCurve(equation, 0, 1, MeshSpec(half, intervals, 1, 0.0))
    unknowns = np.concatenate(
        [orbit.flatten(), [epsilon**2, 0.0], [-(epsilon**4)]],
    )
    sweep = (5.0 / 7.0) * epsilon**2
    return curve, times, unknowns, sweep


class TestConnectingOrbitNewton(TestCase):
    def test_jacobian_is_sparse_with_banded_nonzeros(self):
        intervals = 400
        solver, times, seed = _saddle_problem(intervals)
        unknowns = seed.flatten()
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        jacobian = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        )
        assert issparse(jacobian)
        assert jacobian.shape == (residual.size, unknowns.size)
        dimension = 2
        banded = intervals * 2 * dimension * dimension
        bordered = 2 * dimension * dimension + 1
        assert jacobian.nnz <= banded + bordered

    def test_jacobian_matches_central_differences(self):
        solver, times, seed = _saddle_problem(40)
        unknowns = seed.flatten()

        def residual(values):
            return solver._residual(values, times, seed.shape)  # noqa: SLF001

        jacobian = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual(unknowns),
            True,  # noqa: FBT003
        )
        reference = _central_differences(residual, unknowns)
        assert np.max(np.abs(jacobian.toarray() - reference)) < 1.0e-7

    def test_finite_difference_fallback_is_sparse_and_matches(self):
        solver, times, seed = _saddle_problem(
            40,
            NonAnalyticSaddle(
                variables=[State(), State()],
                parameters=[],
                results=[State(), State()],
                time=None,
            ),
        )
        unknowns = seed.flatten()
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        assert (
            solver._jacobian(  # noqa: SLF001
                unknowns,
                times,
                seed.shape,
                residual,
                True,  # noqa: FBT003
            )
            is None
        )
        numerical = solver._numerical_jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual.size,
        )
        assert issparse(numerical)
        analytic_solver, _, _ = _saddle_problem(40)
        analytic = analytic_solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        )
        assert np.max(np.abs(numerical.toarray() - analytic.toarray())) < 1.0e-5

    def test_non_analytic_field_reaches_the_analytic_orbit(self):
        intervals = 160
        analytic, _, seed = _saddle_problem(intervals)
        numerical, _, _ = _saddle_problem(
            intervals,
            NonAnalyticSaddle(
                variables=[State(), State()],
                parameters=[],
                results=[State(), State()],
                time=None,
            ),
        )
        expected = analytic.solve(seed).states
        found = numerical.solve(seed).states
        assert np.max(np.abs(found - expected)) < 1.0e-6

    def test_step_is_the_dense_least_squares_step(self):
        solver, times, seed = _saddle_problem(200)
        unknowns = seed.flatten()
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        jacobian = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        )
        step = solver._least_squares(jacobian, residual)  # noqa: SLF001
        dense = np.asarray(jacobian.toarray())
        reference, *_ = np.linalg.lstsq(dense, -residual, rcond=None)
        assert np.max(np.abs(step - reference)) < 1.0e-9 * np.max(np.abs(reference))

    def test_iteration_memory_grows_linearly(self):
        """Quadrupling the nodes must not multiply the peak memory by sixteen."""
        peaks = []
        for intervals in (400, 1600):
            solver, times, seed = _saddle_problem(intervals)
            unknowns = seed.flatten()
            tracemalloc.start()
            residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
            jacobian = solver._jacobian(  # noqa: SLF001
                unknowns,
                times,
                seed.shape,
                residual,
                True,  # noqa: FBT003
            )
            solver._least_squares(jacobian, residual)  # noqa: SLF001
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak)
        assert peaks[1] / peaks[0] < 6.0


class TestHomoclinicCurveNewton(TestCase):
    def test_jacobian_is_sparse_square_and_matches_differences(self):
        curve, times, unknowns, sweep = _bogdanov_takens_problem(30)
        nodes, dimension = times.size, 2

        def residual(values):
            return curve._residual(  # noqa: SLF001
                values,
                times,
                nodes,
                dimension,
                sweep,
            )

        jacobian = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            dimension,
            sweep,
            residual(unknowns),
        )
        assert issparse(jacobian)
        assert jacobian.shape == (unknowns.size, unknowns.size)
        reference = _central_differences(residual, unknowns)
        scale = np.max(np.abs(reference))
        assert np.max(np.abs(jacobian.toarray() - reference)) < 1.0e-6 * scale

    def test_step_solves_the_newton_system(self):
        curve, times, unknowns, sweep = _bogdanov_takens_problem(300)
        nodes, dimension = times.size, 2
        residual = curve._residual(  # noqa: SLF001
            unknowns,
            times,
            nodes,
            dimension,
            sweep,
        )
        jacobian = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            dimension,
            sweep,
            residual,
        )
        step = curve._step(jacobian, residual)  # noqa: SLF001
        assert np.linalg.norm(jacobian @ step + residual) < 1.0e-10 * np.linalg.norm(
            residual,
        )

    def test_field_evaluations_per_iteration_grow_linearly(self):
        """A dense finite-difference Newton evaluates the field O(N^2) times."""
        counts = []
        for intervals in (100, 200):
            curve, times, unknowns, sweep = _bogdanov_takens_problem(
                intervals,
                CountingBogdanovTakens,
            )
            nodes, dimension = times.size, 2
            CountingBogdanovTakens.calls = 0
            residual = curve._residual(  # noqa: SLF001
                unknowns,
                times,
                nodes,
                dimension,
                sweep,
            )
            jacobian = curve._jacobian(  # noqa: SLF001
                unknowns,
                times,
                dimension,
                sweep,
                residual,
            )
            curve._step(jacobian, residual)  # noqa: SLF001
            counts.append(CountingBogdanovTakens.calls)
        assert counts[1] / counts[0] < 2.2
