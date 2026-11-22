"""The planar quasipotential and minimum action paths, against closed forms.

Every number here is checked against an exact quasipotential, not recorded output
(see :mod:`tests.continuation.quasipotential_systems` for the two systems and why
their quasipotentials are what they are). The normalisation is the library's: for
``dx = f dt + G dW`` with ``D = G G^T`` the quasipotential solves
``grad U . f + (1/2) grad U^T D grad U = 0``, so a gradient drift ``f = -grad V``
with ``G = I`` has ``U = 2 V`` and the stationary density of
:class:`~...continuation.fokker_planck.StationaryFokkerPlanck` is ``exp(-U)`` at
leading order - which one test asserts directly.

Discretisation order is measured, not assumed: the same box at spacings ``h`` and
``h / 2`` and ``h / 4``, and the observed order is ``log2`` of the ratio of
successive errors.
"""

import functools
import itertools
import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.quasipotential import (
    CycleAttractor,
    EquilibriumAttractor,
    GeometricMinimumAction,
    OrderedLineIntegral,
    _action,
    _stationary_fraction,
)
from discrecontinual_equations.continuation.stochastic import Ito
from tests.continuation.fields import (
    Alpha,
    ConformalColumn,
    HopfField,
    Sigma,
    State,
)
from tests.continuation.quasipotential_systems import (
    OTHER_WELL,
    SADDLE,
    WELL,
    constant_noise,
    double_well,
    potential,
    rotating_drift,
    rotating_noise_matrix,
    rotating_quasipotential,
)

_IDENTITY = np.eye(2)
# Each refinement halves the spacing on the same box: 2^k * 20 intervals.
_COUNTS = (41, 81, 161)
# U = 2 reaches x = -sqrt(3) and |y| = sqrt(2), inside [-2, 1] x [-1.5, 1.5].
_GRADIENT_LEVEL = 2.0
# The largest sublevel set of (1/2) x^T S^-1 x inside [-1.5, 1.5]^2 with a margin.
_ROTATING_LEVEL = 1.5
# Observed orders sit at 1.87, 1.89 and 1.23, 0.95 (module docstring): the
# double well is dominated by a second-order quadrature term, the rotating case
# by the method's first-order interpolation.
_GRADIENT_ORDER = 1.7
_ROTATING_ORDER = 0.9


def _orders(errors):
    return [math.log2(a / b) for a, b in itertools.pairwise(errors)]


@functools.cache
def _double_well_field(count):
    grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [count, count])
    solver = OrderedLineIntegral(double_well(), constant_noise(_IDENTITY), grid)
    return grid, solver, solver.solve(EquilibriumAttractor(WELL))


def _double_well_error(count):
    grid, _solver, field = _double_well_field(count)
    points = grid.points()
    exact = 2.0 * (potential(points[:, 0], points[:, 1]) - potential(*WELL))
    # The basin of the left well, on the sublevel set the box contains: the path
    # to x climbs V monotonically, so it never leaves {U <= U(x)}.
    inside = (points[:, 0] < 0.0) & (exact <= _GRADIENT_LEVEL)
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


def _rotating_error(count):
    grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [count, count])
    solver = OrderedLineIntegral(
        rotating_drift(),
        constant_noise(rotating_noise_matrix()),
        grid,
    )
    field = solver.solve(EquilibriumAttractor([0.0, 0.0]))
    points = grid.points()
    exact = rotating_quasipotential(points)
    # The minimum action path to x stays inside {U <= U(x)}, so on a sublevel set
    # the box contains, the answer does not depend on the boundary at all.
    inside = exact <= _ROTATING_LEVEL
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


class TestGradientQuasipotential(TestCase):
    """Oracle 1: ``f = -grad V``, ``D = I`` gives ``U = 2 V`` exactly."""

    def test_u_is_twice_the_potential_on_the_basin(self):
        error = _double_well_error(81)
        assert error < 5.0e-2

    def test_u_converges_on_doubling_the_grid(self):
        errors = [_double_well_error(count) for count in _COUNTS]
        orders = _orders(errors)
        assert errors[0] > errors[1] > errors[2]
        assert min(orders) > _GRADIENT_ORDER

    def test_the_barrier_is_twice_the_potential_difference(self):
        _grid, _solver, field = _double_well_field(81)
        barrier = 2.0 * (potential(*SADDLE) - potential(*WELL))
        assert abs(field.at(SADDLE) - barrier) < 2.0e-3


class TestRotatingOrnsteinUhlenbeck(TestCase):
    """Oracle 2: ``U = (1/2) x^T S^{-1} x``, ``A S + S A^T + G G^T = 0``."""

    def test_u_is_the_lyapunov_quadratic_form(self):
        assert _rotating_error(81) < 1.2e-3

    def test_u_converges_on_doubling_the_grid(self):
        errors = [_rotating_error(count) for count in _COUNTS]
        orders = _orders(errors)
        assert errors[0] > errors[1] > errors[2]
        assert min(orders) > _ROTATING_ORDER

    def test_u_is_minus_the_log_of_the_library_density(self):
        # The normalisation the module states, checked against the module it sits
        # beside rather than against a formula: -log p, shifted to vanish at the
        # attractor, is the same U on the same grid. The Gaussian makes it exact.
        grid = DensityGrid.box([-3.0, -3.0], [3.0, 3.0], [81, 81])
        drift, noise = rotating_drift(), constant_noise(rotating_noise_matrix())
        density = StationaryFokkerPlanck(drift, noise, Ito(), grid).solve()
        field = OrderedLineIntegral(drift, noise, grid).solve(
            EquilibriumAttractor([0.0, 0.0]),
        )
        centre = grid.nearest([0.0, 0.0])
        inside = rotating_quasipotential(grid.points()).reshape(grid.shape) <= 1.5
        logarithm = -np.log(density.values[inside] / density.at(centre))
        assert np.max(np.abs(field.values[inside] - logarithm)) < 1.5e-2


class TestMinimumActionPath(TestCase):
    """Oracle 3: in the gradient case the path is steepest descent via the saddle."""

    def test_the_path_from_a_point_is_reversed_steepest_descent(self):
        _grid, solver, field = _double_well_field(161)
        start = np.array([-0.3, 0.6])
        path = solver.minimum_action_path(field, start)
        assert np.allclose(path.states[0], start)
        assert np.linalg.norm(path.states[-1] - np.array(WELL)) < 0.05
        x, y = path.states[:, 0], path.states[:, 1]
        # y0 e^{-t}, and x^2 = 1 / (1 + (1/x0^2 - 1) e^{-2t}) along the descent.
        ratio = (y / start[1]) ** 2
        exact = -1.0 / np.sqrt(1.0 + (1.0 / start[0] ** 2 - 1.0) * ratio)
        assert np.max(np.abs(x - exact)) < 1.0e-2

    def test_descent_from_beside_the_saddle_follows_the_axis_to_the_well(self):
        _grid, solver, field = _double_well_field(161)
        path = solver.minimum_action_path(field, np.array([-0.05, 0.0]))
        assert np.max(np.abs(path.states[:, 1])) < 1.0e-2
        assert np.linalg.norm(path.states[-1] - np.array(WELL)) < 0.05
        assert abs(path.action - 2.0 * (potential(-0.05, 0.0) - potential(*WELL))) < (
            1.0e-2
        )

    def test_tracing_from_the_other_basin_stalls_at_the_saddle_and_says_so(self):
        # Through a saddle -(f + D grad U) vanishes and U stops falling, so the
        # trace stops beside it. It must say so and must not jump to the well.
        grid, solver, field = _double_well_field(161)
        path = solver.minimum_action_path(field, [0.5, 0.3], end=WELL)
        assert not path.arrived
        last = path.states[-1]
        assert np.linalg.norm(last - np.array(SADDLE)) < 0.15
        assert np.linalg.norm(last - np.array(WELL)) > 0.8
        steps = np.linalg.norm(np.diff(path.states, axis=0), axis=1)
        assert np.max(steps) <= grid.spacings[0]

    def test_a_trace_that_arrives_ends_on_the_equilibrium(self):
        _grid, solver, field = _double_well_field(161)
        path = solver.minimum_action_path(field, [-0.3, 0.6], end=WELL)
        assert path.arrived
        assert np.allclose(path.states[-1], WELL)

    def test_gmam_between_the_wells_crosses_the_saddle(self):
        # The initial guess bows far off the axis; the method has to find it.
        gmam = GeometricMinimumAction(double_well(), constant_noise(_IDENTITY))
        angle = np.linspace(0.0, math.pi, 81)
        initial = np.column_stack([-np.cos(angle), 0.8 * np.sin(angle)])
        path = gmam.path(initial)
        distance = np.min(np.linalg.norm(path.states - np.array(SADDLE), axis=1))
        assert distance < 2.0e-2
        assert np.max(np.abs(path.states[:, 1])) < 1.0e-2
        assert np.allclose(path.states[0], WELL)
        assert np.allclose(path.states[-1], OTHER_WELL)
        barrier = 2.0 * (potential(*SADDLE) - potential(*WELL))
        assert abs(path.action - barrier) < 5.0e-3

    def test_gmam_reproduces_the_quadratic_form_off_the_gradient_case(self):
        gmam = GeometricMinimumAction(
            rotating_drift(),
            constant_noise(rotating_noise_matrix()),
        )
        target = np.array([0.8, -0.3])
        initial = np.linspace([0.0, 0.0], target, 81)
        path = gmam.path(initial)
        exact = float(rotating_quasipotential(target[None, :])[0])
        assert abs(path.action - exact) / exact < 3.0e-3

    def test_gmam_reports_whether_it_converged(self):
        gmam = GeometricMinimumAction(double_well(), constant_noise(_IDENTITY))
        angle = np.linspace(0.0, math.pi, 41)
        initial = np.column_stack([-np.cos(angle), 0.8 * np.sin(angle)])
        assert gmam.path(initial).converged
        cut = GeometricMinimumAction(
            double_well(),
            constant_noise(_IDENTITY),
            iterations=5,
        ).path(initial)
        assert not cut.converged
        assert cut.iterations == 5

    def test_gmam_lands_on_the_same_path_whatever_its_step(self):
        # The step is how fast the chain relaxes, not where it ends; a fixed point
        # that moved with it was the tangential force fighting the
        # reparametrisation.
        target = np.array([0.8, -0.3])
        actions = [
            GeometricMinimumAction(
                rotating_drift(),
                constant_noise(rotating_noise_matrix()),
                step=step,
            )
            .path(np.linspace([0.0, 0.0], target, 41))
            .action
            for step in (0.05, 0.01)
        ]
        assert abs(actions[0] - actions[1]) < 1.0e-6

    def test_gmam_refuses_a_state_dependent_diffusion(self):
        parameters = [Alpha(value=1.0), Sigma(value=0.7)]
        gmam = GeometricMinimumAction(
            HopfField([State(), State()], parameters, [State(), State()], None),
            _conformal(parameters),
        )
        with pytest.raises(ValueError, match="constant"):
            gmam.path(np.linspace([0.5, 0.0], [1.2, 0.0], 21))


def _conformal(parameters):
    return NoiseMatrix(
        [
            ConformalColumn(index, [State(), State()], parameters, [State(), State()])
            for index in range(2)
        ],
    )


class TestCycleAttractor(TestCase):
    """The noisy Hopf with conformal noise, ``D = s^2 r^2 I``, about its cycle.

    The rotation is orthogonal to every radial gradient, so ``f = -(1/2) D grad U
    + l`` with ``l . grad U = 0`` and ``U`` is radial:
    ``U = (r^2 - mu - mu log(r^2 / mu)) / s^2``, zero on the cycle ``r = sqrt(mu)``
    and infinite at the origin where the noise vanishes.
    """

    def test_u_about_the_cycle_is_the_radial_closed_form(self):
        mu, sigma = 1.0, 0.7
        parameters = [Alpha(value=mu), Sigma(value=sigma)]
        drift = HopfField([State(), State()], parameters, [State(), State()], None)
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [100, 100])
        angle = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
        cycle = math.sqrt(mu) * np.column_stack([np.cos(angle), np.sin(angle)])
        field = OrderedLineIntegral(drift, _conformal(parameters), grid).solve(
            CycleAttractor(cycle),
        )
        points = grid.points()
        squared = np.sum(points * points, axis=1)
        exact = (squared - mu - mu * np.log(squared / mu)) / sigma**2
        band = (squared > 0.25) & (squared < 2.25)
        error = np.abs(field.values.ravel()[band] - exact[band])
        assert np.max(error) < 5.0e-2

    def test_a_trace_stalled_off_a_small_cycle_has_not_arrived(self):
        # mu = 0.025 on the film's grid: the cycle is three cells from the
        # singular origin and the trace from outside stalls about three cells
        # short of it, at U = 0.244 against a floor of 0.223. Judged against the
        # cycle's starting values instead (their largest is 0.79) it reported
        # arrived.
        grid, solver, field = _hopf_cycle_field(0.025)
        path = solver.minimum_action_path(field, [1.35, 0.85])
        off = abs(np.linalg.norm(path.states[-1]) - math.sqrt(0.025))
        assert off > 2.5 * grid.spacings[0]
        assert not path.arrived

    def test_a_trace_onto_a_resolved_cycle_has_arrived(self):
        grid, solver, field = _hopf_cycle_field(1.0)
        path = solver.minimum_action_path(field, [1.35, 0.85])
        off = abs(np.linalg.norm(path.states[-1]) - 1.0)
        assert off < 2.0 * grid.spacings[0]
        assert path.arrived

    def test_the_floor_is_what_the_march_left_on_the_band(self):
        _grid, _solver, field = _hopf_cycle_field(1.0)
        assert field.floor < 0.04  # 0.032; the starting values' largest is 0.175


@functools.cache
def _hopf_cycle_field(mu):
    """The noisy Hopf film's own march: s = 0.7, 88 points over [-2.2, 2.2]^2."""
    parameters = [Alpha(value=mu), Sigma(value=0.7)]
    drift = HopfField([State(), State()], parameters, [State(), State()], None)
    grid = DensityGrid.box([-2.2, -2.2], [2.2, 2.2], [88, 88])
    angle = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    cycle = math.sqrt(mu) * np.column_stack([np.cos(angle), np.sin(angle)])
    solver = OrderedLineIntegral(drift, _conformal(parameters), grid)
    return grid, solver, solver.solve(CycleAttractor(cycle))


class TestRefusals(TestCase):
    def test_a_grid_that_is_not_planar_is_refused(self):
        grid = DensityGrid.box([-1.0] * 3, [1.0] * 3, [5, 5, 5])
        with pytest.raises(ValueError, match="planar"):
            OrderedLineIntegral(double_well(), constant_noise(_IDENTITY), grid)

    def test_a_singular_diffusion_is_refused(self):
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [11, 11])
        solver = OrderedLineIntegral(
            double_well(),
            constant_noise(np.array([[1.0], [0.5]])),
            grid,
        )
        with pytest.raises(ValueError, match="singular"):
            solver.solve(EquilibriumAttractor(WELL))

    def test_an_attractor_off_the_grid_is_refused(self):
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [11, 11])
        solver = OrderedLineIntegral(double_well(), constant_noise(_IDENTITY), grid)
        with pytest.raises(ValueError, match="outside"):
            solver.solve(EquilibriumAttractor([3.0, 0.0]))


class TestTriangleUpdate(TestCase):
    """The closed-form minimiser over a triangle's base, against brute force.

    The double well never needs the triangle update - with it switched off its
    errors agree to five digits - so this checks the root directly: for random
    metrics, drifts, base edges, targets and rises, ``F`` at the closed-form
    ``s`` must be the minimum of ``F`` over a fine grid of ``s`` in ``[0, 1]``.
    """

    def test_the_closed_form_root_is_the_minimum_over_the_base(self):
        rng = np.random.default_rng(0)
        count = 4000
        factor = rng.normal(size=(count, 2, 2))
        metric = factor @ factor.transpose(0, 2, 1) + 0.1 * np.eye(2)
        drift = rng.normal(size=(count, 2)) * rng.choice([0.0, 1.0, 3.0], (count, 1))
        coefficients = np.column_stack(
            [drift, metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]],
        )
        edge = tuple(rng.integers(-1, 2, count).astype(float) for _ in range(2))
        reach = tuple(rng.integers(-8, 9, count).astype(float) for _ in range(2))
        rise = 0.5 * rng.normal(size=count)
        fraction = _stationary_fraction(edge, reach, rise, coefficients)

        def cost(s):
            return s * rise + _action(
                reach[0] - s * edge[0],
                reach[1] - s * edge[1],
                coefficients,
            )

        brute = np.min([cost(np.full(count, s)) for s in np.linspace(0, 1, 2001)], 0)
        moving = (edge[0] != 0.0) | (edge[1] != 0.0)
        degenerate = ((reach[0] == 0.0) & (reach[1] == 0.0)) | (
            (reach[0] == edge[0]) & (reach[1] == edge[1])
        )
        kept = moving & ~degenerate
        assert kept.sum() > count // 2
        assert np.max(cost(fraction)[kept] - brute[kept]) < 1.0e-9
