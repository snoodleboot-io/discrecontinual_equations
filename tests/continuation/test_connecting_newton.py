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
import warnings
from unittest import TestCase, mock

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, issparse

from discrecontinual_equations.continuation import connecting_orbit
from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    Terminus,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import (
    _TILT,
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    HeteroclinicField,
    State,
    TiltedHomoclinicField,
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


def _bogdanov_takens_problem(
    intervals: int,
    field_type=BogdanovTakensField,
    continuation_index: int = 0,
):
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
    curve = HomoclinicCurve(
        equation,
        continuation_index,
        1 - continuation_index,
        MeshSpec(half, intervals, 1, 0.0),
    )
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
        """Either parameter may be the free one.

        ``b1`` enters the field as a constant, so its column cannot tell which
        nodes an interval averages; ``b2`` multiplies ``y``, so it can.
        """
        for continuation_index in (0, 1):
            with self.subTest(continuation_index=continuation_index):
                self._check_against_differences(continuation_index)

    @staticmethod
    def _check_against_differences(continuation_index: int) -> None:
        curve, times, unknowns, sweep = _bogdanov_takens_problem(
            30,
            continuation_index=continuation_index,
        )
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


def _tilted_problem(intervals: int):
    half = 12.0
    times = np.linspace(-half, half, intervals + 1)
    field = TiltedHomoclinicField(
        variables=[State(), State(), State()],
        parameters=[],
        results=[State(), State(), State()],
        time=None,
    )
    jacobian = _TILT @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]])
    solver = HomoclinicOrbit(
        field,
        np.zeros(3),
        jacobian @ _TILT.T,
        MeshSpec(half, intervals, phase_index=0, phase_value=0.3),
    )
    zeros = np.zeros_like(times)
    seed = np.column_stack([1.2 / np.cosh(times / 2.5) ** 2, zeros, zeros]) @ _TILT.T
    return solver, times, seed


def _double_well_jacobian(x: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])


def _heteroclinic_problem(intervals: int):
    half = 8.0
    times = np.linspace(-half, half, intervals + 1)
    solver = HeteroclinicOrbit(
        HeteroclinicField(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        ),
        Terminus(np.array([-1.0, 0.0]), _double_well_jacobian(-1.0)),
        Terminus(np.array([1.0, 0.0]), _double_well_jacobian(1.0)),
        MeshSpec(half, intervals),
    )
    seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
    return solver, times, seed


def _bogdanov_takens_gap(first: float, second: float) -> float:
    """Shooting defect of the BT homoclinic: where each saddle manifold crosses y = 0.

    The unstable branch is integrated forward and the stable branch backward with
    DOP853 from the saddle ``(sqrt(-b1), 0)``, each to its first crossing of the
    section ``y = 0`` at the loop's leftmost point. Their ``x`` values agree exactly
    when the loop closes. Nothing here is shared with the collocation solver.
    """
    saddle = math.sqrt(-first)
    values, vectors = np.linalg.eig(
        np.array([[0.0, 1.0], [2.0 * saddle, second + saddle]]),
    )

    def leftward(vector: np.ndarray) -> np.ndarray:
        return vector if vector[0] < 0.0 else -vector

    def field(_: float, state: np.ndarray) -> list[float]:
        x, y = state
        return [y, first + second * y + x * x + x * y]

    def crossing(direction: int, vector: np.ndarray, horizon: float) -> float:
        def section(_: float, state: np.ndarray) -> float:
            return state[1]

        section.terminal = True  # type: ignore[attr-defined]
        section.direction = direction  # type: ignore[attr-defined]
        start = np.array([saddle, 0.0]) + 1.0e-7 * saddle * vector
        path = solve_ivp(
            field,
            (0.0, horizon),
            start,
            method="DOP853",
            events=section,
            rtol=1.0e-12,
            atol=1.0e-16,
        )
        return float(path.y_events[0][0][0])

    unstable = leftward(vectors[:, np.argmax(values)].real)
    stable = leftward(vectors[:, np.argmin(values)].real)
    return crossing(1, unstable, 1.0e4) - crossing(-1, stable, -1.0e4)


class TestIndependentReferences(TestCase):
    """References that share no code with the sparse assembly or the collocation."""

    def test_jacobians_match_differences_on_tilted_and_heteroclinic(self):
        """Off the solution, in 3-D and saddle-to-saddle, exact and differenced paths.

        Entrywise against central differences of the residual at a randomly
        perturbed iterate, so no term vanishes because the iterate is converged.
        """
        generator = np.random.default_rng(7)
        for name, problem in (
            ("tilted", _tilted_problem),
            ("heteroclinic", _heteroclinic_problem),
        ):
            solver, times, seed = problem(50)
            unknowns = seed.flatten() + 0.05 * generator.standard_normal(seed.size)

            def residual(values, solver=solver, times=times, shape=seed.shape):
                return solver._residual(values, times, shape)  # noqa: SLF001

            reference = _central_differences(residual, unknowns)
            exact = solver._jacobian(  # noqa: SLF001
                unknowns,
                times,
                seed.shape,
                residual(unknowns),
                True,  # noqa: FBT003
            )
            differenced = solver._numerical_jacobian(  # noqa: SLF001
                unknowns,
                times,
                seed.shape,
                reference.shape[0],
            )
            with self.subTest(problem=name):
                assert np.max(np.abs(exact.toarray() - reference)) < 1.0e-8
                assert np.max(np.abs(differenced.toarray() - reference)) < 1.0e-6

    def test_curve_converges_to_the_shooting_value_at_second_order(self):
        """The collocated b1 approaches an independent DOP853 shooting value as h^2.

        Measured: errors 3.51e-5 and 8.79e-6 at 100 and 200 intervals (ratio 4.00)
        against the shooting value -3.8859126e-3 at b2 = (5/7)/16.
        """
        epsilon = 0.25
        sweep = (5.0 / 7.0) * epsilon**2
        guess = -(49.0 / 25.0) * sweep**2
        shooting = brentq(
            lambda first: _bogdanov_takens_gap(first, sweep),
            1.3 * guess,
            0.7 * guess,
            xtol=1.0e-16,
            rtol=1.0e-14,
        )
        errors = []
        for intervals in (100, 200):
            curve, times, unknowns, _ = _bogdanov_takens_problem(intervals)
            nodes = times.size
            point = curve.solve_point(
                sweep,
                unknowns[: nodes * 2].reshape(nodes, 2),
                unknowns[nodes * 2 : nodes * 2 + 2],
                float(unknowns[-1]),
            )
            assert point is not None
            errors.append(point.continuation - shooting)
        assert abs(errors[1]) < 3.0e-3 * abs(shooting)
        assert 3.6 < errors[0] / errors[1] < 4.4


class TestDirectSolvePath(TestCase):
    """The sparse LU must be the path taken, and its fill must stay linear."""

    def test_well_posed_step_never_reaches_the_fallbacks(self):
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
        refuse = mock.Mock(side_effect=AssertionError("fallback reached"))
        with (
            mock.patch.object(connecting_orbit, "lsmr", refuse),
            mock.patch.object(np.linalg, "lstsq", refuse),
        ):
            solver._least_squares(jacobian, residual)  # noqa: SLF001

    def test_curve_step_never_reaches_the_dense_fallback(self):
        curve, times, unknowns, sweep = _bogdanov_takens_problem(200)
        residual = curve._residual(  # noqa: SLF001
            unknowns,
            times,
            times.size,
            2,
            sweep,
        )
        jacobian = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            2,
            sweep,
            residual,
        )
        refuse = mock.Mock(side_effect=AssertionError("dense fallback reached"))
        with mock.patch.object(np.linalg, "lstsq", refuse):
            curve._step(jacobian, residual)  # noqa: SLF001

    def test_rank_deficient_step_falls_back_to_the_minimum_norm_step(self):
        """An exactly singular augmented matrix still yields the lstsq step."""
        solver, times, seed = _saddle_problem(60)
        unknowns = seed.flatten()
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        dense = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        ).toarray()
        dense[:, 5] = dense[:, 0]
        with self.assertWarns(RuntimeWarning):
            step = solver._least_squares(csr_matrix(dense), residual)  # noqa: SLF001
        reference, *_ = np.linalg.lstsq(dense, -residual, rcond=None)
        assert np.max(np.abs(step - reference)) < 1.0e-6

    def test_factor_fill_is_flat_in_the_nodes(self):
        """tracemalloc cannot see SuperLU's C allocations, so measure the fill itself.

        Measured nnz(L+U)/nnz(A): 2.59 at 200 nodes and 2.61 at 3200 with COLAMD on
        scipy 1.15.3, 1.17.1 and 1.18.1; MMD_AT_PLUS_A grows from 22 to 322.
        """
        ratios = []
        for intervals in (400, 1600):
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
            augmented = connecting_orbit._augmented(jacobian)  # noqa: SLF001
            factor = connecting_orbit._factor(augmented)  # noqa: SLF001
            ratios.append((factor.L.nnz + factor.U.nnz) / augmented.nnz)
        assert max(ratios) < 4.0
        assert ratios[1] / ratios[0] < 1.1


class TestAugmentedConditioning(TestCase):
    """The augmented least-squares solve must be as accurate as a QR solve.

    With an unscaled identity block and ``sigma_min < 1`` the augmented matrix is
    conditioned like ``J^T J``, not ``J``: measured on these problems its forward
    error was 5e-7 at ``cond(J) = 1e6`` and 2e-3 at ``1e8``, where QR gave 1e-9
    and 2e-5. Scaling the block by ``sigma_min / sqrt(2)`` (Bjorck) brings it to
    ``cond(J)`` and the error to QR's.
    """

    def test_ill_conditioned_step_matches_qr_accuracy(self):
        solver, _, _ = _saddle_problem(20)
        generator = np.random.default_rng(3)
        rows, columns = 60, 40
        left, _ = np.linalg.qr(generator.standard_normal((rows, rows)))
        right, _ = np.linalg.qr(generator.standard_normal((columns, columns)))
        for decades in (6, 8):
            singular = np.logspace(0, -decades, columns)
            dense = left[:, :columns] @ np.diag(singular) @ right.T
            exact = generator.standard_normal(columns)
            # The orthogonal complement of the range carries the least-squares
            # residual, so ``exact`` is the least-squares solution.
            target = dense @ exact + 1.0e-3 * left[:, columns:] @ (
                generator.standard_normal(rows - columns)
            )
            step = solver._least_squares(  # noqa: SLF001
                csr_matrix(dense),
                -target,
            )
            factor_q, factor_r = np.linalg.qr(dense)
            reference = np.linalg.solve(factor_r, factor_q.T @ target)

            def error(candidate, exact=exact):
                return np.linalg.norm(candidate - exact) / np.linalg.norm(exact)

            with self.subTest(condition=f"1e{decades}"):
                assert error(step) < 10.0 * error(reference)


class TestFallbacksAreAnnounced(TestCase):
    """A fallback changes the cost and the step, so it must not happen silently."""

    def test_singular_connecting_step_warns_and_names_the_reason(self):
        solver, times, seed = _saddle_problem(60)
        unknowns = seed.flatten()
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        dense = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        ).toarray()
        dense[:, 5] = dense[:, 0]
        with self.assertWarnsRegex(RuntimeWarning, "singular"):
            solver._least_squares(csr_matrix(dense), residual)  # noqa: SLF001

    def test_singular_curve_step_warns_and_names_the_reason(self):
        curve, times, unknowns, sweep = _bogdanov_takens_problem(40)
        residual = curve._residual(  # noqa: SLF001
            unknowns,
            times,
            times.size,
            2,
            sweep,
        )
        dense = curve._jacobian(  # noqa: SLF001
            unknowns,
            times,
            2,
            sweep,
            residual,
        ).toarray()
        dense[:, 5] = 0.0
        with self.assertWarnsRegex(RuntimeWarning, "singular"):
            step = curve._step(csr_matrix(dense), residual)  # noqa: SLF001
        reference, *_ = np.linalg.lstsq(dense, -residual, rcond=None)
        assert np.max(np.abs(step - reference)) < 1.0e-8 * np.max(np.abs(reference))

    def test_well_posed_steps_do_not_warn(self):
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
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            solver._least_squares(jacobian, residual)  # noqa: SLF001
