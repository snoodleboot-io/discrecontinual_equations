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
from unittest import TestCase, mock

import numpy as np
import pytest
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix, fields_at
from discrecontinual_equations.continuation.quasipotential import (
    _ACCEPTED,
    CHECKED,
    GRID,
    MIDPOINT,
    QUADRATIC,
    REFINED,
    CycleAttractor,
    EquilibriumAttractor,
    GeometricMinimumAction,
    GridQuasipotential,
    OrderedLineIntegral,
    _action,
    _base_cost,
    _base_slope,
    _cubic_coefficients,
    _curved_triangle,
    _guide,
    _HalfLattice,
    _March,
    _relax_from,
    _stationary_fraction,
    default_radius,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.function.function import Function
from tests.continuation.fields import (
    Alpha,
    ConformalColumn,
    HopfField,
    Sigma,
    State,
)
from tests.continuation.quasipotential_systems import (
    ANISOTROPIC_AMPLITUDE,
    CUBIC_TERMS,
    MAIER_STEIN_BARRIER,
    MAIER_STEIN_VALUES,
    NON_NORMAL_AMPLITUDE,
    NON_NORMAL_DRIFT,
    OTHER_WELL,
    SADDLE,
    WELL,
    AnisotropicGradientField,
    DoubleWellField,
    MaierSteinField,
    MultiplicativeField,
    NonNormalField,
    PaperCycleField,
    PaperLinearField,
    RotatingCubicField,
    constant_noise,
    double_well,
    kronecker_covariance,
    multiplicative_noise,
    potential,
    rotating_drift,
    rotating_noise_matrix,
    rotating_quasipotential,
    system,
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
# by the default straight base's first-order interpolation along the front
# (the opt-in parabola takes it to 2.00, 1.87; TestQuadraticUpdate).
_GRADIENT_ORDER = 1.7
_ROTATING_ORDER = 0.9
# The convergence tests hold the radius at 8 cells as the grid is refined, so
# that they measure the update itself. It is the default too; default_radius,
# Dahiya and Cameron's Eq. (28), grows with the grid and trades the two error
# terms differently for each problem (see TestDefaultRadius and the module
# docstring).
_FIXED_RADIUS = 8


def _orders(errors):
    return [math.log2(a / b) for a, b in itertools.pairwise(errors)]


@functools.cache
def _double_well_field(count):
    grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [count, count])
    solver = OrderedLineIntegral(
        double_well(),
        constant_noise(_IDENTITY),
        grid,
        _FIXED_RADIUS,
    )
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
        _FIXED_RADIUS,
    )
    field = solver.solve(EquilibriumAttractor([0.0, 0.0]))
    points = grid.points()
    exact = rotating_quasipotential(points)
    # The minimum action path to x stays inside {U <= U(x)}, so on a sublevel set
    # the box contains, the answer does not depend on the boundary at all.
    inside = exact <= _ROTATING_LEVEL
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


# --- Grids shared by the gate tests (DEQ-162) ----------------------------------
#
# A march on a 161 x 161 or 161 x 129 grid takes about ten seconds. Several gate
# tests read the same one; a slow test used to build it first in the same worker,
# through the caches above, and once the slow tests left the gate whichever gate
# test came first paid for it in its own time. These fixtures build each grid
# once per module, in setup, from the same caches - so the slow tests that call
# the helpers directly share them too - and the xdist group keeps every test
# that reads one grid on one worker (pytest -n auto --dist loadgroup), so the
# gate builds each grid once.


@pytest.fixture(scope="module")
def double_well_161():
    """The double well about its left well at 161 x 161: (grid, solver, field)."""
    return _double_well_field(161)


_DOUBLE_WELL_161 = pytest.mark.xdist_group("quasipotential-double-well-161")


class TestGradientQuasipotential(TestCase):
    """Oracle 1: ``f = -grad V``, ``D = I`` gives ``U = 2 V`` exactly."""

    def test_u_is_twice_the_potential_on_the_basin(self):
        error = _double_well_error(81)
        assert error < 5.0e-2

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_u_is_the_lyapunov_quadratic_form(self):
        # 9.68e-4 at 81 points with the straight base of OLIM-MID, the
        # default (4.25e-5 with the parabolic one; TestQuadraticUpdate).
        assert _rotating_error(81) < 1.2e-3

    @pytest.mark.slow
    def test_u_converges_on_doubling_the_grid(self):
        # Measured 2.27e-3, 9.68e-4, 5.00e-4: orders 1.23, 0.95, the first
        # order of the straight base (parabolic: 1.70e-4, 4.25e-5, 1.16e-5,
        # orders 2.00, 1.87; TestQuadraticUpdate).
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


class TestMinimumActionPath:
    """Oracle 3: in the gradient case the path is steepest descent via the saddle."""

    @_DOUBLE_WELL_161
    def test_the_path_from_a_point_is_reversed_steepest_descent(
        self,
        double_well_161,
    ):
        _grid, solver, field = double_well_161
        start = np.array([-0.3, 0.6])
        path = solver.minimum_action_path(field, start)
        assert np.allclose(path.states[0], start)
        assert np.linalg.norm(path.states[-1] - np.array(WELL)) < 0.05
        x, y = path.states[:, 0], path.states[:, 1]
        # y0 e^{-t}, and x^2 = 1 / (1 + (1/x0^2 - 1) e^{-2t}) along the descent.
        ratio = (y / start[1]) ** 2
        exact = -1.0 / np.sqrt(1.0 + (1.0 / start[0] ** 2 - 1.0) * ratio)
        assert np.max(np.abs(x - exact)) < 1.0e-2

    @_DOUBLE_WELL_161
    def test_descent_from_beside_the_saddle_follows_the_axis_to_the_well(
        self,
        double_well_161,
    ):
        _grid, solver, field = double_well_161
        path = solver.minimum_action_path(field, np.array([-0.05, 0.0]))
        assert np.max(np.abs(path.states[:, 1])) < 1.0e-2
        assert np.linalg.norm(path.states[-1] - np.array(WELL)) < 0.05
        assert abs(path.action - 2.0 * (potential(-0.05, 0.0) - potential(*WELL))) < (
            1.0e-2
        )

    @_DOUBLE_WELL_161
    def test_tracing_from_the_other_basin_stalls_at_the_saddle_and_says_so(
        self,
        double_well_161,
    ):
        # Through a saddle -(f + D grad U) vanishes and U stops falling, so the
        # trace stops beside it. It must say so and must not jump to the well.
        grid, solver, field = double_well_161
        path = solver.minimum_action_path(field, [0.5, 0.3], end=WELL)
        assert not path.arrived
        last = path.states[-1]
        assert np.linalg.norm(last - np.array(SADDLE)) < 0.15
        assert np.linalg.norm(last - np.array(WELL)) > 0.8
        steps = np.linalg.norm(np.diff(path.states, axis=0), axis=1)
        assert np.max(steps) <= grid.spacings[0]

    @_DOUBLE_WELL_161
    def test_a_trace_that_arrives_ends_on_the_equilibrium(self, double_well_161):
        _grid, solver, field = double_well_161
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

    @pytest.mark.slow
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

    def test_gmam_on_a_strongly_rotating_drift_settles_at_the_default_step(self):
        # Dahiya and Cameron's SDE (21), rotation ten times the gradient: the
        # explicit part of the flow is stiff there, and at the default step
        # the chain once blew up to nan and scipy's banded solve raised. The
        # answer must not depend on the step it took to get there.
        target = np.array([0.5, 0.3])
        paths = [
            GeometricMinimumAction(
                system(PaperLinearField),
                constant_noise(_IDENTITY),
                **options,
            ).path(np.linspace([0.0, 0.0], target, 41))
            for options in ({}, {"step": 0.01})
        ]
        assert all(path.converged for path in paths)
        assert np.all(np.isfinite(paths[0].states))
        assert abs(paths[0].action - paths[1].action) < 1.0e-6

    @pytest.mark.slow
    def test_gmam_follows_a_state_dependent_diffusion(self):
        # The conformal noise of the noisy Hopf, D = s^2 r^2 I, about its cycle:
        # the grid-free reference relaxes gMAM in its general form from the
        # band's edge, against U = (r^2 - mu - mu log(r^2 / mu)) / s^2.
        _grid, solver, field = _hopf_cycle_field(1.0)
        target = np.array([1.35, 0.85])
        squared = float(target @ target)
        exact = (squared - 1.0 - math.log(squared)) / 0.7**2
        value, own = solver.reference(field, target)
        assert abs(value - exact) < 1.0e-3
        assert abs(value - exact) <= own
        assert abs(value - exact) < abs(field.at(target) - exact)


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

    @pytest.mark.slow
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
        # short of it, at U = 0.067 against a floor of 0.0156. (With the old
        # straight-climb start the floor was 0.223 and the stall at 0.244, and
        # before the floor was read after the march it reported arrived.)
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

    def test_the_floor_is_the_level_of_u_on_the_band(self):
        # The largest exact U on the band at mu = 1 is 0.0229; the floor the
        # march leaves is 0.0219 (it was 0.032 from the old straight-climb
        # start, whose own largest value was 0.175).
        grid, _solver, field = _hopf_cycle_field(1.0)
        parameters = [Alpha(value=1.0), Sigma(value=0.7)]
        angle = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
        indices, _values = CycleAttractor(
            np.column_stack([np.cos(angle), np.sin(angle)]),
        ).initial(
            grid,
            HopfField([State(), State()], parameters, [State(), State()], None),
            _conformal(parameters),
        )
        squared = np.sum(grid.points()[indices] ** 2, axis=1)
        exact = (squared - 1.0 - np.log(squared)) / 0.7**2
        assert abs(field.floor - np.max(exact)) < 2.0e-3


@functools.cache
def _hopf_cycle_field(mu):
    """The noisy Hopf film's own march: s = 0.7, 88 points over [-2.2, 2.2]^2."""
    parameters = [Alpha(value=mu), Sigma(value=0.7)]
    drift = HopfField([State(), State()], parameters, [State(), State()], None)
    grid = DensityGrid.box([-2.2, -2.2], [2.2, 2.2], [88, 88])
    angle = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    cycle = math.sqrt(mu) * np.column_stack([np.cos(angle), np.sin(angle)])
    solver = OrderedLineIntegral(
        drift,
        _conformal(parameters),
        grid,
        update_order=MIDPOINT,
    )
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


# --- Independent references --------------------------------------------------
#
# Each oracle below is computed without the library: the published closed forms
# of Dahiya and Cameron, a gradient drift whose noise is anisotropic, a Lyapunov
# covariance by a Kronecker solve, and a Maier-Stein barrier from an L-BFGS
# minimisation of the discretised geometric action done outside the library.
# Numbers quoted in comments were measured on this branch.


@functools.cache
def _maier_stein_field(scale):
    """Maier-Stein about the left well, on a box of spacing ``0.05 / scale``."""
    grid = DensityGrid.box([-1.6, -0.8], [0.4, 0.8], [40 * scale + 1, 32 * scale + 1])
    solver = OrderedLineIntegral(
        system(MaierSteinField),
        constant_noise(_IDENTITY),
        grid,
    )
    return grid, solver, solver.solve(EquilibriumAttractor(WELL))


def _paper_cycle():
    angle = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    return CycleAttractor(np.column_stack([np.cos(angle), np.sin(angle)]))


@functools.cache
def _paper_cycle_field(count):
    """SDE (24) about the unit circle, and ``(r^2, U - exact)`` per grid point."""
    grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [count, count])
    field = OrderedLineIntegral(
        system(PaperCycleField),
        constant_noise(_IDENTITY),
        grid,
        _FIXED_RADIUS,
    ).solve(_paper_cycle())
    points = grid.points()
    squared = np.sum(points * points, axis=1)
    return grid, squared, field.values.ravel() - 0.5 * (squared - 1.0) ** 2


def _paper_cycle_error(count):
    _grid, squared, error = _paper_cycle_field(count)
    # {U <= 4.5} is the disc r <= 2, which the box contains; 1.9 keeps a margin.
    return float(np.max(np.abs(error[squared <= 1.9**2])))


def _paper_cycle_band_error(count):
    _grid, squared, error = _paper_cycle_field(count)
    return float(np.max(np.abs(error[np.abs(np.sqrt(squared) - 1.0) < 0.1])))


def _anisotropic_gradient_error(count):
    grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [count, count])
    field = OrderedLineIntegral(
        system(AnisotropicGradientField),
        constant_noise(ANISOTROPIC_AMPLITUDE),
        grid,
        _FIXED_RADIUS,
    ).solve(EquilibriumAttractor(WELL))
    points = grid.points()
    exact = 2.0 * (potential(points[:, 0], points[:, 1]) - potential(*WELL))
    inside = (points[:, 0] < -0.5) & (exact <= _GRADIENT_LEVEL)
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


class TestPublishedCases(TestCase):
    """The two test problems of Dahiya and Cameron (2018), Section 3."""

    @pytest.mark.slow
    def test_the_linear_sde_21_reaches_the_published_accuracy(self):
        # Their Table 4 fits OLIM-MID's max error on SDE (21) by 0.817 N^-1.39,
        # 9.5e-4 at N = 129, with K = 10 by their rule of thumb (28). Here
        # 1.14e-3 on the sublevel set the box contains (their march stops at
        # the boundary of the mesh, so it is the same set).
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [129, 129])
        field = OrderedLineIntegral(
            system(PaperLinearField),
            constant_noise(_IDENTITY),
            grid,
            update_radius=10,
        ).solve(EquilibriumAttractor([0.0, 0.0]))
        points = grid.points()
        exact = 2.0 * points[:, 0] ** 2 + points[:, 1] ** 2
        inside = exact <= 0.9
        error = np.abs(field.values.ravel()[inside] - exact[inside])
        assert np.max(error) < 1.5e-3

    @pytest.mark.slow
    def test_u_about_the_cycle_of_sde_24_converges_at_second_order(self):
        # Additive noise, unlike the conformal case above: U = (r^2 - 1)^2 / 2.
        # Measured 0.0899, 0.0221 (and 0.00453 at 321): orders 2.02, 2.29.
        errors = [_paper_cycle_error(count) for count in (81, 161)]
        assert errors[1] < 3.0e-2
        assert min(_orders(errors)) > 1.8

    def test_the_cycle_starts_from_the_quasipotential_not_a_straight_climb(self):
        # Dahiya and Cameron's start (their Eqs. (9)-(12)): grad U from the
        # split b = -(1/2) D grad U + l with l along b at the foot, integrated
        # from the foot by Simpson's rule. A straight one-point climb from the
        # foot pays |l| per unit of distance across the flow - O(h) - and
        # overestimated these values by 0.0396 at 161 points.
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [161, 161])
        indices, values = _paper_cycle().initial(
            grid,
            system(PaperCycleField),
            constant_noise(_IDENTITY),
        )
        points = grid.points()[indices]
        exact = 0.5 * (np.sum(points * points, axis=1) - 1.0) ** 2
        assert indices.size > 100
        assert np.max(np.abs(values - exact)) < 1.0e-4

    @pytest.mark.slow
    def test_u_next_to_the_cycle_converges_at_second_order(self):
        # On |r - 1| < 0.1. With the straight-climb start the march could not
        # repair the values it accepted first: 5.45e-3 and 3.27e-3 at 161 and
        # 321 points, order 0.74, ten times the error once the start is right.
        errors = [_paper_cycle_band_error(count) for count in (161, 321)]
        assert errors[1] < 1.0e-3
        assert min(_orders(errors)) > 1.8

    def test_the_stencil_reaches_its_full_radius(self):
        # Radius 1 is the four axis neighbours, the points exactly one cell
        # away; a stencil that stopped short of its radius would have none and
        # leave every point but the attractor's at infinity.
        grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [41, 41])
        field = OrderedLineIntegral(
            double_well(),
            constant_noise(_IDENTITY),
            grid,
            update_radius=1,
        ).solve(EquilibriumAttractor(WELL))
        assert np.all(np.isfinite(field.values))
        barrier = 2.0 * (potential(*SADDLE) - potential(*WELL))
        assert abs(field.at(SADDLE) - barrier) < 2.0e-2


class TestAnisotropicReferences(TestCase):
    """``D`` that is not a multiple of the identity, against independent forms."""

    @pytest.mark.slow
    def test_a_gradient_drift_in_the_noise_metric_has_u_twice_the_potential(self):
        # f = -D grad V: the density is exp(-2 V) for any constant D, so a
        # solver that used D for D^-1, or G^T G for G G^T, misses it by O(1).
        # Measured 0.0391, 0.00867: order 2.17 (quadrature-dominated, as the
        # double well with D = I).
        errors = [_anisotropic_gradient_error(count) for count in (81, 161)]
        assert errors[1] < 1.2e-2
        assert min(_orders(errors)) > 1.8

    @pytest.mark.slow
    def test_a_non_normal_linear_drift_has_the_kronecker_lyapunov_form(self):
        grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [81, 81])
        field = OrderedLineIntegral(
            system(NonNormalField),
            constant_noise(NON_NORMAL_AMPLITUDE),
            grid,
        ).solve(EquilibriumAttractor([0.0, 0.0]))
        diffusion = NON_NORMAL_AMPLITUDE @ NON_NORMAL_AMPLITUDE.T
        inverse = np.linalg.inv(kronecker_covariance(NON_NORMAL_DRIFT, diffusion))
        points = grid.points()
        exact = 0.5 * np.einsum("ki,ij,kj->k", points, inverse, points)
        # The level 0.9 x the least U on the box's edge, so the sublevel set is
        # inside the box. Measured max error 8.98e-4 at 81 points.
        edge = np.linspace(-1.5, 1.5, 301)
        ring = np.concatenate(
            [np.column_stack([edge, np.full_like(edge, side)]) for side in (-1.5, 1.5)]
            + [
                np.column_stack([np.full_like(edge, side), edge])
                for side in (-1.5, 1.5)
            ],
        )
        level = 0.9 * np.min(np.einsum("ki,ij,kj->k", ring, 0.5 * inverse, ring))
        inside = exact <= level
        assert np.max(np.abs(field.values.ravel()[inside] - exact[inside])) < 1.2e-3


@pytest.fixture(scope="module")
def maier_stein_4():
    """Maier-Stein about the left well at spacing 0.0125: (grid, solver, field)."""
    return _maier_stein_field(4)


class TestMaierStein:
    """Maier-Stein at beta = 10, where the escape path leaves the symmetry axis."""

    @pytest.mark.slow
    def test_the_barrier_converges_to_the_independent_minimum(self):
        # Measured 0.341561, 0.340604 against 0.340042 with OLIM-MID, the
        # default, at radius 8: errors 1.52e-3, 5.6e-4, order 1.44. With the
        # parabolic base 0.340175, 0.340084: errors 1.33e-4, 4.16e-5, order
        # 1.68. The on-axis path costs 1/2.
        errors = [
            abs(_maier_stein_field(scale)[2].at(SADDLE) - MAIER_STEIN_BARRIER)
            for scale in (2, 4)
        ]
        assert errors[1] < 1.0e-3
        assert min(_orders(errors)) > 1.2

    def test_gmam_finds_the_off_axis_path_and_its_action(self):
        gmam = GeometricMinimumAction(
            system(MaierSteinField),
            constant_noise(_IDENTITY),
        )
        fraction = np.linspace(0.0, 1.0, 161)
        initial = np.column_stack([-1.0 + fraction, 0.4 * np.sin(math.pi * fraction)])
        path = gmam.path(initial)
        assert path.converged
        # 0.3400577; the independent chain gives 0.340042 to 0.340052.
        assert abs(path.action - MAIER_STEIN_BARRIER) < 1.0e-4
        assert 0.28 < np.max(np.abs(path.states[:, 1])) < 0.34

    @pytest.mark.xdist_group("quasipotential-maier-stein-4")
    def test_the_grid_the_trace_and_gmam_agree_on_a_point(self, maier_stein_4):
        # Three estimates of U at one point by three routes - the march, the
        # action along the traced characteristic, and the relaxed chain - and
        # the traced path lies on the chain to within a cell.
        _grid, solver, field = maier_stein_4
        target = np.array([-0.4, 0.3])
        traced = solver.minimum_action_path(field, target, end=WELL)
        relaxed = GeometricMinimumAction(
            system(MaierSteinField),
            constant_noise(_IDENTITY),
        ).path(np.linspace(WELL, target, 81))
        assert traced.arrived
        assert relaxed.converged
        # Measured 0.25920 (grid), 0.25879 (trace), 0.25875 (gMAM) with
        # OLIM-MID, the default; 0.25883 on the grid with the parabolic base.
        assert abs(field.at(target) - relaxed.action) < 1.0e-3
        assert abs(traced.action - relaxed.action) < 3.0e-4
        distance = cKDTree(relaxed.states).query(traced.states)[0]
        assert np.max(distance) < _grid.spacings[0]


# --- Error estimates and refinement ----------------------------------------
#
# The check of a grid value is a grid-free value at the same point: the
# minimum action path relaxed from a small surface about the attractor, where U
# is known from the attractor's own expansion. These tests hold the estimate
# against the true error on the four reference problems of the verification,
# wherever that error is above 1e-4: each estimate must be within a factor of
# three of it or above it.

_INDICATOR_FLOOR = 1.0e-4
_INDICATOR_FACTOR = 3.0


def _nonnormal_exact(points):
    diffusion = NON_NORMAL_AMPLITUDE @ NON_NORMAL_AMPLITUDE.T
    inverse = np.linalg.inv(kronecker_covariance(NON_NORMAL_DRIFT, diffusion))
    return 0.5 * np.einsum("ki,ij,kj->k", points, inverse, points)


def _gradient_exact(points):
    return 2.0 * (potential(points[:, 0], points[:, 1]) - potential(*WELL))


def _cycle_exact(points):
    return 0.5 * (np.sum(points * points, axis=1) - 1.0) ** 2


@functools.cache
def _checked(name):
    """A reference problem's field at 81 points, checked at four grid points."""
    if name == "nonnormal":
        grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [81, 81])
        solver = OrderedLineIntegral(
            system(NonNormalField),
            constant_noise(NON_NORMAL_AMPLITUDE),
            grid,
            _FIXED_RADIUS,
            update_order=MIDPOINT,
        )
        attractor, exact = EquilibriumAttractor([0.0, 0.0]), _nonnormal_exact
        targets = [(0.4, 0.3), (-0.375, 0.1875), (0.225, -0.3), (-0.6, 0.0)]
    elif name == "gradient":
        grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [81, 81])
        solver = OrderedLineIntegral(
            system(AnisotropicGradientField),
            constant_noise(ANISOTROPIC_AMPLITUDE),
            grid,
            _FIXED_RADIUS,
            update_order=MIDPOINT,
        )
        attractor, exact = EquilibriumAttractor(WELL), _gradient_exact
        targets = [(-0.7, 0.6), (-1.5, -0.8), (-0.6, -0.3), (-1.2, 0.9)]
    else:
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [81, 81])
        solver = OrderedLineIntegral(
            system(PaperCycleField),
            constant_noise(_IDENTITY),
            grid,
            _FIXED_RADIUS,
            update_order=MIDPOINT,
        )
        attractor, exact = _paper_cycle(), _cycle_exact
        targets = [(0.5, 0.0), (1.0, 1.0), (-1.2, -1.2), (0.0, -0.6)]
    field = solver.solve(attractor)
    indices = [np.ravel_multi_index(grid.nearest(t), grid.shape) for t in targets]
    checked = solver.check(field, indices)
    points = grid.points()[indices]
    true = field.values.ravel()[indices] - exact(points)
    return solver, field, checked, indices, true


def _assert_tracks(estimate, true):
    large = np.abs(true) > _INDICATOR_FLOOR
    assert large.any()
    above = estimate[large] >= np.abs(true[large])
    close = estimate[large] >= np.abs(true[large]) / _INDICATOR_FACTOR
    close &= estimate[large] <= _INDICATOR_FACTOR * np.abs(true[large])
    assert np.all(above | close), (estimate, true)


@pytest.fixture(scope="module")
def checked_gradient():
    """The anisotropic gradient at 81 points, checked at four points."""
    return _checked("gradient")


_CHECKED_GRADIENT = pytest.mark.xdist_group("quasipotential-checked-gradient")


class TestErrorEstimate:
    """DEQ-57: an estimated error per checked point, against the true one."""

    @pytest.mark.slow
    def test_a_field_from_the_march_has_measured_nothing(self):
        _solver, field, _checked_field, _indices, _true = _checked("gradient")
        assert np.all(np.isnan(field.errors))
        assert np.all(field.status == GRID)
        assert isinstance(field.attractor, EquilibriumAttractor)

    @_CHECKED_GRADIENT
    def test_checking_marks_the_points_and_leaves_the_field_alone(
        self,
        checked_gradient,
    ):
        _solver, field, checked, indices, _true = checked_gradient
        assert np.all(checked.status.ravel()[indices] == CHECKED)
        assert np.all(np.isfinite(checked.errors.ravel()[indices]))
        others = np.ones(field.grid.size, dtype=bool)
        others[indices] = False
        assert np.all(checked.status.ravel()[others] == GRID)
        assert np.array_equal(checked.values, field.values)
        assert np.all(np.isnan(field.errors))

    @pytest.mark.slow
    def test_the_estimate_tracks_the_error_on_the_non_normal_system(self):
        _solver, _field, checked, indices, true = _checked("nonnormal")
        _assert_tracks(checked.errors.ravel()[indices], true)

    @_CHECKED_GRADIENT
    def test_the_estimate_tracks_the_error_on_the_anisotropic_gradient(
        self,
        checked_gradient,
    ):
        _solver, _field, checked, indices, true = checked_gradient
        _assert_tracks(checked.errors.ravel()[indices], true)

    @pytest.mark.slow
    def test_the_estimate_tracks_the_error_about_the_cycle_of_sde_24(self):
        _solver, _field, checked, indices, true = _checked("cycle")
        _assert_tracks(checked.errors.ravel()[indices], true)

    def test_the_estimate_tracks_the_error_on_maier_stein(self):
        # The truth here is the independent L-BFGS minimisation of the
        # discretised action (MAIER_STEIN_VALUES), not a closed form.
        grid, solver, field = _maier_stein_field(2)
        targets = list(MAIER_STEIN_VALUES)
        indices = [np.ravel_multi_index(grid.nearest(t), grid.shape) for t in targets]
        checked = solver.check(field, indices)
        true = field.values.ravel()[indices] - np.array(
            [MAIER_STEIN_VALUES[t] for t in targets],
        )
        _assert_tracks(checked.errors.ravel()[indices], true)

    @pytest.mark.slow
    def test_the_grid_free_value_is_closer_than_the_grid(self):
        # Where the path from a rotating equilibrium winds round it: from the
        # surface a quarter of the way out, the chain never sees the winding.
        grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [81, 81])
        solver = OrderedLineIntegral(
            rotating_drift(),
            constant_noise(rotating_noise_matrix()),
            grid,
            update_order=MIDPOINT,
        )
        field = solver.solve(EquilibriumAttractor([0.0, 0.0]))
        target = np.array([0.8, -0.3])
        exact = float(rotating_quasipotential(target[None, :])[0])
        value, own = solver.reference(field, target)
        # Measured 1.8e-5 (own estimate 4.3e-5) against the grid's 6.9e-4.
        assert abs(value - exact) < 3.0e-5
        assert abs(value - exact) <= own
        assert abs(field.at(target) - exact) > 10.0 * abs(value - exact)

    @pytest.mark.slow
    def test_a_strongly_rotating_drift_can_be_checked(self):
        # SDE (21), rotation ten times the gradient: the chain's flow is stiff
        # at the default step, and checking a point once raised from scipy.
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [41, 41])
        solver = OrderedLineIntegral(
            system(PaperLinearField),
            constant_noise(_IDENTITY),
            grid,
        )
        field = solver.solve(EquilibriumAttractor([0.0, 0.0]))
        index = np.ravel_multi_index(grid.nearest([0.5, 0.3]), grid.shape)
        target = grid.points()[index]
        exact = 2.0 * target[0] ** 2 + target[1] ** 2
        value, own = solver.reference(field, target)
        checked = solver.check(field, [index])
        # The chain from the surface is no better than the grid here - the
        # path spirals out of the origin ten times faster than it climbs -
        # but its own estimate says so, and the grid's estimate stays a bound.
        assert abs(value - exact) <= own
        grid_error = abs(field.values.ravel()[index] - exact)
        assert checked.errors.ravel()[index] >= grid_error

    @pytest.mark.slow
    def test_the_references_own_estimate_bounds_its_error(self):
        # About a nonlinear well, where the chain's error goes at second order
        # in the nodes rather than the fourth of the linear systems: measured
        # -4.9e-5 at 81 nodes and -1.3e-5 at 161 at (-1.5, -0.8).
        solver, field, _checked_field, _indices, _true = _checked("gradient")
        target = np.array([[-1.5, -0.8]])
        value, own = solver.reference(field, target[0])
        error = abs(value - float(_gradient_exact(target)[0]))
        assert error < 1.0e-4
        assert error <= own <= 10.0 * error

    @pytest.mark.slow
    def test_the_reference_beside_a_cycle_starts_on_the_targets_side(self):
        # From the band's edge on the side of the target, with the march's
        # Simpson-rule start there: at r = sqrt(2), U = 1/2 exactly.
        solver, field, _checked_field, _indices, _true = _checked("cycle")
        value, own = solver.reference(field, [1.0, 1.0])
        # Measured 5.5e-5 below; the grid at 81 points is 3.3e-2 below here.
        assert abs(value - 0.5) < 1.0e-4
        assert abs(value - 0.5) <= own

    @pytest.mark.slow
    def test_the_estimate_is_the_distance_to_the_reference_plus_its_error(self):
        solver, field, checked, indices, _true = _checked("nonnormal")
        target = field.grid.points()[indices[0]]
        value, own = solver.reference(field, target)
        estimate = abs(field.values.ravel()[indices[0]] - value) + own
        assert checked.errors.ravel()[indices[0]] == pytest.approx(estimate, rel=1e-9)

    @_CHECKED_GRADIENT
    def test_the_start_slides_to_where_u_plus_the_action_is_least(
        self,
        checked_gradient,
    ):
        # Started half a radian round the surface from the traced crossing, the
        # secant must still bring the start to the stationary point: anywhere
        # else U + S is larger, by the square of the distance. About the
        # nonlinear well the surface is not a level set of U, so only the root
        # of (grad U - p) . t is that point (on a linear system grad U . t
        # vanishes on the surface and (grad U + p) . t has the same root).
        solver, field, _checked_field, _indices, _true = checked_gradient
        drift = system(AnisotropicGradientField)
        noise = constant_noise(ANISOTROPIC_AMPLITUDE)
        target = np.array([-1.5, -0.8])
        surface = field.attractor.start(field.grid, drift, noise, target)
        chain, parameter = _guide(
            surface,
            solver.minimum_action_path(field, target).states,
            81,
        )
        gmam = GeometricMinimumAction(drift, noise, tolerance=1.0e-8)
        value, _chain, _place = _relax_from(gmam, surface, (chain, parameter + 0.5))
        exact = float(_gradient_exact(target[None, :])[0])
        assert abs(value - exact) < 1.0e-4

    @_CHECKED_GRADIENT
    def test_a_field_without_its_attractor_cannot_be_checked(self, checked_gradient):
        solver, field, _checked_field, _indices, _true = checked_gradient
        bare = GridQuasipotential(field.grid, field.values, field.floor)
        with pytest.raises(ValueError, match="attractor"):
            solver.check(bare, [0])


class TestRefinement(TestCase):
    """DEQ-57: past a tolerance, the grid value is replaced by the grid-free one."""

    @pytest.mark.slow
    def test_every_refined_point_is_within_the_tolerance(self):
        solver, field, _checked_field, indices, _true = _checked("gradient")
        tolerance = 5.0e-4
        refined = solver.check(field, indices, tolerance=tolerance)
        status = refined.status.ravel()[indices]
        assert np.any(status == REFINED)
        points = field.grid.points()[indices]
        error = np.abs(refined.values.ravel()[indices] - _gradient_exact(points))
        assert np.all(error[status == REFINED] <= tolerance)
        assert np.all(refined.errors.ravel()[indices][status == REFINED] <= tolerance)
        # A point refined was one whose grid value was off by more than that.
        grid_error = np.abs(field.values.ravel()[indices] - _gradient_exact(points))
        assert np.all(grid_error[status == REFINED] > error[status == REFINED])

    @pytest.mark.slow
    def test_a_reference_not_good_to_the_tolerance_only_flags(self):
        # No reference here is good to 1e-7, so nothing may be replaced, however
        # far off the grid value is; each point is flagged by its error instead.
        solver, field, _checked_field, indices, _true = _checked("gradient")
        flagged = solver.check(field, indices[:2], tolerance=1.0e-7)
        assert np.all(flagged.status.ravel()[indices[:2]] == CHECKED)
        assert np.array_equal(flagged.values, field.values)


class TestCubicTerm(TestCase):
    def test_the_cubic_term_at_a_well_is_twice_the_potentials(self):
        # U = 2 (V - V_min) and V - V_min = d^2 - d^3 + d^4 / 4 + y^2 / 2 with
        # d = x + 1, whatever the constant D: U3 = -2 d^3.
        coefficients = _cubic_coefficients(
            system(AnisotropicGradientField),
            constant_noise(ANISOTROPIC_AMPLITUDE),
            np.array(WELL),
            np.diag([2.0, 1.0]),
        )
        assert np.allclose(coefficients, [-2.0, 0.0, 0.0, 0.0], atol=1.0e-5)

    def test_the_cubic_term_of_a_rotating_well_is_its_own(self):
        # Off the gradient case, with every coefficient of U3 nonzero: U is
        # exact by construction (see RotatingCubicField), M = diag(1, 2).
        drift = system(RotatingCubicField)
        noise = constant_noise(_IDENTITY)
        origin = np.zeros(2)
        start = EquilibriumAttractor(origin).start(None, drift, noise, [0.4, 0.2])
        assert start.value([0.1, 0.0]) == pytest.approx(0.01 + 0.3e-3)
        coefficients = _cubic_coefficients(drift, noise, origin, np.diag([1.0, 2.0]))
        assert np.allclose(coefficients, CUBIC_TERMS, atol=1.0e-6)

    def test_the_inner_surface_is_at_half_the_distance(self):
        # The reference's second surface, which measures the expansion's error,
        # is the same level set scaled by one half about the equilibrium.
        drift = system(RotatingCubicField)
        noise = constant_noise(_IDENTITY)
        outer = EquilibriumAttractor([0.0, 0.0]).start(None, drift, noise, [0.4, 0.2])
        inner = outer.shrunk()
        for angle in (0.0, 1.0, 2.5):
            assert np.allclose(inner.point(angle), 0.5 * outer.point(angle))


class TestArrival(TestCase):
    """A trace has reached a cycle when it stops inside the band, by distance."""

    @pytest.mark.slow
    def test_a_trace_stopping_inside_the_band_has_arrived(self):
        # mu = 0.2: the trace from inside stops 1.48 cells from the cycle, U
        # 0.002 above the floor. The band is 1.5 cells wide, so it is in it;
        # by the old reading of U against the floor it had stalled.
        grid, solver, field = _hopf_cycle_field(0.2)
        path = solver.minimum_action_path(field, [0.3, 0.0])
        off = abs(np.linalg.norm(path.states[-1]) - math.sqrt(0.2))
        assert off < 1.5 * grid.spacings[0]
        assert path.arrived

    def test_a_trace_stopping_outside_the_band_has_not(self):
        # The trace from outside stops 2.1 cells from the cycle at mu = 0.2.
        grid, solver, field = _hopf_cycle_field(0.2)
        path = solver.minimum_action_path(field, [1.35, 0.85])
        off = abs(np.linalg.norm(path.states[-1]) - math.sqrt(0.2))
        assert off > 1.5 * grid.spacings[0]
        assert not path.arrived


# --- The update radius, the order of the update, batched fields (DEQ-58) -----


class TestDefaultRadius(TestCase):
    """The solver's radius is 8 at every size; Eq. (28) is there to opt into.

    ``default_radius`` is Dahiya and Cameron's Eq. (28), held at 8 below 128
    points a side. It is not the solver's default: at 321 points it made the
    gradient-dominated problems three to four times worse (module docstring).
    """

    def test_the_radius_follows_the_paper_rule_on_larger_grids(self):
        # log2 rounds: 182 is 7.51 and 200 is 7.64, nearer 8 than 7; 181 is 7.4998.
        expected = {
            41: 8,
            88: 8,
            90: 8,
            128: 10,
            161: 10,
            181: 10,
            182: 14,
            200: 14,
            256: 14,
            321: 14,
            513: 18,
        }
        for count, radius in expected.items():
            grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [count, count])
            assert default_radius(grid) == radius, count

    def test_a_rectangular_grid_takes_its_longer_side(self):
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [41, 300])
        assert default_radius(grid) == 14

    def test_the_solver_holds_the_radius_at_8_on_every_grid(self):
        drift, noise = rotating_drift(), constant_noise(rotating_noise_matrix())
        for count in (41, 161, 321):
            grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [count, count])
            assert OrderedLineIntegral(drift, noise, grid).update_radius == 8, count

    def test_the_paper_rule_and_any_other_radius_are_taken_when_asked(self):
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [321, 321])
        drift, noise = rotating_drift(), constant_noise(rotating_noise_matrix())
        chosen = OrderedLineIntegral(drift, noise, grid, default_radius(grid))
        assert chosen.update_radius == 14
        assert OrderedLineIntegral(drift, noise, grid, 4).update_radius == 4


def _rotating_error_by_order(count, order):
    grid = DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [count, count])
    field = OrderedLineIntegral(
        rotating_drift(),
        constant_noise(rotating_noise_matrix()),
        grid,
        update_order=order,
    ).solve(EquilibriumAttractor([0.0, 0.0]))
    exact = rotating_quasipotential(grid.points())
    inside = exact <= _ROTATING_LEVEL
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


# Values of the default march before the parabola existed, recorded on
# feature/deq-57 (90b9207): a diagonal, an antidiagonal, a row and the largest.
# The double well's were re-recorded on DEQ-94, whose Dennis-Schnabel Jacobian
# step moves them by at most 2.3e-9 relative; with the old 1e-7 step the march
# reproduces the 90b9207 values exactly. The rotating field's are unchanged.
_BEFORE_PARABOLA = {
    "double well": (
        ((0, 0), 6.592977174415461),
        ((4, 4), 3.138754914189343),
        ((8, 8), 1.2407144159023322),
        ((12, 12), 0.3820392304364576),
        ((16, 16), 0.1556341439168195),
        ((20, 20), 0.2823444336216896),
        ((24, 24), 0.5564219618722723),
        ((28, 28), 0.8554289194374021),
        ((32, 32), 1.2744712781313812),
        ((36, 36), 1.8875599869278659),
        ((40, 40), 2.653149218777939),
        ((2, 38), 4.632251169307202),
        ((6, 34), 2.026984828668291),
        ((10, 30), 0.7070492425305566),
        ((14, 26), 0.20743695324452222),
        ((18, 22), 0.19363762915612523),
        ((22, 18), 0.4133059316476094),
        ((26, 14), 0.7002162374448025),
        ((30, 10), 1.0395550235656736),
        ((34, 6), 1.5589037956181766),
        ((38, 2), 2.2515468245524075),
        ((20, 1), 2.312965383272623),
        ((20, 7), 1.2338645131422021),
        ((20, 13), 0.5590143057039301),
        ((20, 19), 0.2912606718449867),
        ((20, 25), 0.425661426184016),
        ((20, 31), 0.96417561723987),
        ((20, 37), 1.908077109473103),
    ),
    "rotating": (
        ((0, 0), 3.9704277385511832),
        ((4, 4), 2.541368106050344),
        ((8, 8), 1.4291755097261658),
        ((12, 12), 0.6351145165090196),
        ((16, 16), 0.15888327451376946),
        ((20, 20), 0.0),
        ((24, 24), 0.15888327451376946),
        ((28, 28), 0.6351145165090195),
        ((32, 32), 1.4291755097261651),
        ((36, 36), 2.5413681060503435),
        ((40, 40), 3.970427738551183),
        ((2, 38), 5.422778255160429),
        ((6, 34), 3.002889865402032),
        ((10, 30), 1.5327575847215404),
        ((14, 26), 0.5516771978674945),
        ((18, 22), 0.061479648191520624),
        ((22, 18), 0.061479648191520624),
        ((26, 14), 0.5516771978674946),
        ((30, 10), 1.5327575847215407),
        ((34, 6), 3.002889865402033),
        ((38, 2), 5.422778255160429),
        ((20, 1), 2.0397144719889644),
        ((20, 7), 0.9551798888580677),
        ((20, 13), 0.27713069085562236),
        ((20, 19), 0.005648904347000301),
        ((20, 25), 0.14130268600271823),
        ((20, 31), 0.683733659805187),
        ((20, 37), 1.632744522869766),
        ((0, 40), 7.552360706427722),
    ),
}


class TestDefaultUpdate(TestCase):
    """OLIM-MID, as published, is the default; the parabola is opt-in.

    The default march must be the march before the parabola existed: the
    values in ``_BEFORE_PARABOLA`` are from the solver on ``feature/deq-57``
    (90b9207), at radius 8 and 41 points. They are compared within a few ulps
    (``rtol=1e-12``), not bit for bit. On this machine the march is identical
    to the last bit, but a stored value is from one machine and the test runs
    on others: NumPy's vectorised ``pow`` and ``sqrt`` can differ from libm by
    an ulp on another CPU, and an exact digest of float output has already
    failed on CI runners. A change to the update itself moves these values by
    far more than that - the parabola moves them by 1e-4 and more.
    """

    def _fields(self):
        return {
            "double well": (
                double_well(),
                constant_noise(_IDENTITY),
                DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [41, 41]),
                WELL,
            ),
            "rotating": (
                rotating_drift(),
                constant_noise(rotating_noise_matrix()),
                DensityGrid.box([-1.5, -1.5], [1.5, 1.5], [41, 41]),
                [0.0, 0.0],
            ),
        }

    def test_the_default_update_is_the_midpoint_one(self):
        drift, noise, grid, _ = self._fields()["rotating"]
        solver = OrderedLineIntegral(drift, noise, grid)
        midpoint = OrderedLineIntegral(drift, noise, grid, update_order=MIDPOINT)
        attractor = EquilibriumAttractor([0.0, 0.0])
        assert np.array_equal(
            solver.solve(attractor).values,
            midpoint.solve(attractor).values,
        )

    def test_the_default_march_is_the_one_before_the_parabola(self):
        for name, (drift, noise, grid, well) in self._fields().items():
            values = (
                OrderedLineIntegral(drift, noise, grid)
                .solve(
                    EquilibriumAttractor(well),
                )
                .values
            )
            places, before = zip(*_BEFORE_PARABOLA[name], strict=True)
            now = [values[place] for place in places]
            np.testing.assert_allclose(
                now,
                before,
                rtol=1.0e-12,
                atol=1.0e-14,
                err_msg=name,
            )


def _random_rows(rng, count):
    """Half-lattice rows - ``f`` and a positive definite ``A`` - at random."""
    factor = rng.normal(size=(count, 2, 2))
    metric = factor @ factor.transpose(0, 2, 1) + 0.1 * np.eye(2)
    drift = rng.normal(size=(count, 2)) * rng.choice([0.3, 1.0, 3.0], (count, 1))
    return np.column_stack([drift, metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]])


def _kept(edge, reach):
    """Bases that move, to targets off them: the cases a march can offer."""
    moving = (edge[0] != 0.0) | (edge[1] != 0.0)
    degenerate = ((reach[0] == 0.0) & (reach[1] == 0.0)) | (
        (reach[0] == edge[0]) & (reach[1] == edge[1])
    )
    return moving & ~degenerate


class TestQuadraticUpdate(TestCase):
    """``U`` a parabola along the triangle's base, through a third front point."""

    @pytest.mark.slow
    def test_the_rotating_case_is_exact_at_every_size(self):
        # DEQ-116. On a linear drift with a constant D the parabola's F is
        # exact, so only the minimisation over s was left: Newton steps with f
        # and A frozen gave 1.70e-4, 4.25e-5, 1.16e-5 at 41, 81, 161 points;
        # the exact root of F' gives 3.1e-15, 4.7e-15, 8.4e-15 - rounding.
        errors = [_rotating_error_by_order(count, QUADRATIC) for count in _COUNTS]
        assert max(errors) < 1.0e-10

    @pytest.mark.slow
    def test_it_is_ten_times_closer_than_the_midpoint_update(self):
        quadratic = _rotating_error_by_order(81, QUADRATIC)
        midpoint = _rotating_error_by_order(81, MIDPOINT)
        assert 10.0 * quadratic < midpoint

    def test_the_parabola_is_the_minimum_over_the_base(self):
        # As for the straight base: against brute force over s in [0, 1].
        rng = np.random.default_rng(1)
        count = 2000
        factor = rng.normal(size=(count, 2, 2))
        metric = factor @ factor.transpose(0, 2, 1) + 0.1 * np.eye(2)
        drift = rng.normal(size=(count, 2)) * rng.choice([1.0, 3.0], (count, 1))
        rows = np.column_stack(
            [drift, metric[:, 0, 0], metric[:, 0, 1], metric[:, 1, 1]],
        )
        edge = tuple(rng.integers(-1, 2, count).astype(float) for _ in range(2))
        reach = tuple(rng.integers(-8, 9, count).astype(float) for _ in range(2))
        rise = 0.5 * rng.normal(size=count)
        # A parabola bending up, so that F is convex along the base. One that
        # bends down makes F concave there and its minimum an end of the base,
        # which the one-point updates from x0 and from x1 already offer.
        bend = 0.05 * np.abs(rng.normal(size=count))
        value = _curved_triangle(0.0, rise, bend, edge, reach, (rows, rows))

        def cost(s):
            return (
                s * rise
                + bend * s * (s - 1.0)
                + _action(
                    reach[0] - s * edge[0],
                    reach[1] - s * edge[1],
                    rows,
                )
            )

        brute = np.min([cost(np.full(count, s)) for s in np.linspace(0, 1, 2001)], 0)
        moving = (edge[0] != 0.0) | (edge[1] != 0.0)
        degenerate = ((reach[0] == 0.0) & (reach[1] == 0.0)) | (
            (reach[0] == edge[0]) & (reach[1] == edge[1])
        )
        kept = moving & ~degenerate
        assert kept.sum() > count // 2
        assert np.max(value[kept] - brute[kept]) < 1.0e-8

    def test_the_base_is_minimised_with_its_coefficients_moving_along_it(self):
        # DEQ-116. On the grid f and A change along the base, from the midpoint
        # of x0-x to that of x1-x, half a cell apart. With them frozen at the
        # current s the old Newton steps missed the minimum on 18% of these
        # (3,500 of 20,000, by up to 54); F' with their change included finds
        # it on all but the few where F has a second local minimum that F' at
        # the start and at the end does not bracket.
        rng = np.random.default_rng(7)
        count = 20000
        near = _random_rows(rng, count)
        far = near + 0.05 * _random_rows(rng, count)
        edge = tuple(rng.integers(-1, 2, count).astype(float) for _ in range(2))
        reach = tuple(rng.integers(-8, 9, count).astype(float) for _ in range(2))
        rise = 0.5 * rng.normal(size=count)
        bend = 0.05 * rng.normal(size=count)
        value = _curved_triangle(0.0, rise, bend, edge, reach, (near, far))

        def cost(s):
            return (
                s * rise
                + bend * s * (s - 1.0)
                + _action(
                    reach[0] - s * edge[0],
                    reach[1] - s * edge[1],
                    near + s[:, None] * (far - near),
                )
            )

        samples = [cost(np.full(count, s)) for s in np.linspace(0, 1, 2001)]
        brute = np.min(samples, 0)
        kept = _kept(edge, reach)
        missed = value[kept] - brute[kept] > 1.0e-8
        assert missed.sum() < 1.0e-3 * kept.sum()
        # The ends are the one-point updates: the triangle never offers more.
        ends = np.minimum(samples[0], samples[-1])
        assert np.all(value[kept] <= ends[kept] + 1.0e-12)

    def test_the_root_takes_secant_steps_not_only_bisections(self):
        # Bisection alone would reach the same root, but in some 40 steps;
        # the secant steps settle these in well under twenty.
        rng = np.random.default_rng(5)
        count = 2000
        near = _random_rows(rng, count)
        far = near + 0.05 * _random_rows(rng, count)
        edge = tuple(rng.integers(-1, 2, count).astype(float) for _ in range(2))
        reach = tuple(rng.integers(-8, 9, count).astype(float) for _ in range(2))
        rise = 0.5 * rng.normal(size=count)
        bend = 0.05 * np.abs(rng.normal(size=count))
        with mock.patch(
            "discrecontinual_equations.continuation.quasipotential._base_slope",
            side_effect=_base_slope,
        ) as slope:
            _curved_triangle(0.0, rise, bend, edge, reach, (near, far))
        assert slope.call_count < 20

    def test_the_slope_along_the_base_is_the_derivative_of_f(self):
        # F' with f and A linear in s, against central differences of F.
        rng = np.random.default_rng(11)
        count = 2000
        near = _random_rows(rng, count)
        far = near + 0.3 * _random_rows(rng, count)
        edge = tuple(rng.integers(-1, 2, count).astype(float) for _ in range(2))
        reach = tuple(rng.integers(-8, 9, count).astype(float) for _ in range(2))
        rise = 0.5 * rng.normal(size=count)
        bend = 0.05 * rng.normal(size=count)
        place = rng.uniform(0.05, 0.95, count)
        step = 1.0e-6
        midpoints = (near, far)
        ahead = _base_cost(0.0, rise, bend, edge, reach, midpoints, place + step)
        behind = _base_cost(0.0, rise, bend, edge, reach, midpoints, place - step)
        slope = _base_slope(rise, bend, edge, reach, (near, far - near), place)
        kept = _kept(edge, reach)
        np.testing.assert_allclose(
            slope[kept],
            ((ahead - behind) / (2.0 * step))[kept],
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_the_bend_is_the_parabola_through_the_third_point(self):
        # U = x^2 along a row: U(x0 + s h) = U0 + s dU + h^2 s (s - 1), so the
        # bend is h^2 whichever third point it is fitted through.
        grid = DensityGrid.box([0.0, 0.0], [1.0, 1.0], [5, 5])
        lattice = _HalfLattice(grid, double_well(), constant_noise(_IDENTITY))
        march = _March(grid, lattice, (np.array([1]), np.array([0])), quadratic=True)
        first = grid.axis(0)
        march.values[:] = (first**2)[:, None]
        spacing = first[1] - first[0]
        status, bends = march._status, march._bends  # noqa: SLF001 (the fit itself)
        status[1:4, 2] = _ACCEPTED  # x0 - e, x0, x1 accepted: fit behind x0
        assert bends(2, 2, [(3, 2)])[0] == pytest.approx(spacing**2)
        status[1, 2] = 0
        status[4, 2] = _ACCEPTED  # x0, x1, x1 + e accepted: fit beyond x1
        assert bends(2, 2, [(3, 2)])[0] == pytest.approx(spacing**2)
        status[4, 2] = 0  # neither: the straight base
        assert bends(2, 2, [(3, 2)])[0] == 0.0

    def test_the_point_behind_x0_is_preferred_when_both_are_accepted(self):
        # U = x^3 along a row, where the two fits differ: through x0 - e the
        # bend is (U(-1) - 2 U0 + U1) / 2, through x1 + e it is (U(2) - U0 -
        # 2 dU) / 2. With all four points accepted the first is taken.
        grid = DensityGrid.box([0.0, 0.0], [1.0, 1.0], [5, 5])
        lattice = _HalfLattice(grid, double_well(), constant_noise(_IDENTITY))
        march = _March(grid, lattice, (np.array([1]), np.array([0])), quadratic=True)
        values = grid.axis(0) ** 3
        march.values[:] = values[:, None]
        march._status[1:5, 2] = _ACCEPTED  # noqa: SLF001 (the fit itself)
        behind = 0.5 * (values[1] - 2.0 * values[2] + values[3])
        beyond = 0.5 * (values[4] - values[2] - 2.0 * (values[3] - values[2]))
        assert abs(behind - beyond) > 1.0e-3
        bend = march._bends(2, 2, [(3, 2)])[0]  # noqa: SLF001
        assert bend == pytest.approx(behind)

    def test_an_unknown_order_is_refused(self):
        grid = DensityGrid.box([-1.0, -1.0], [1.0, 1.0], [11, 11])
        with pytest.raises(ValueError, match="order"):
            OrderedLineIntegral(
                double_well(),
                constant_noise(_IDENTITY),
                grid,
                update_order="simpson",
            )


class _Unbatched(DoubleWellField):
    """The double well with ``eval`` only, as a user's function might be.

    It keeps the base class's ``eval_batch``, which calls ``eval`` per point.
    """

    eval_batch = Function.eval_batch


class TestBatchedFields(TestCase):
    def test_a_batch_is_the_field_point_by_point(self):
        points = np.random.default_rng(2).normal(size=(50, 2))
        field = system(MaierSteinField)
        batch = fields_at(field, points)
        single = np.array([field.eval(list(p)) for p in points])
        assert np.array_equal(batch, single)

    def test_a_function_without_a_batch_is_evaluated_point_by_point(self):
        points = np.random.default_rng(3).normal(size=(20, 2))
        assert np.array_equal(
            fields_at(system(_Unbatched), points),
            fields_at(double_well(), points),
        )

    def test_a_batch_of_the_wrong_shape_is_refused(self):
        class Flat(DoubleWellField):
            def eval_batch(self, points, time=None):  # noqa: ARG002
                return np.zeros(np.asarray(points).size)

        with pytest.raises(ValueError, match="one row per state"):
            fields_at(system(Flat), np.zeros((4, 2)))

    def test_the_noise_columns_come_as_one_batch(self):
        amplitude = np.array([[1.0, 0.5], [0.0, 0.8]])
        points = np.zeros((3, 2))
        batch = constant_noise(amplitude).columns_along(points)
        assert batch.shape == (3, 2, 2)
        assert np.array_equal(batch[1], amplitude.T)

    def test_the_march_is_the_same_with_and_without_batches(self):
        grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [41, 41])
        noise = constant_noise(_IDENTITY)
        batched = OrderedLineIntegral(double_well(), noise, grid).solve(
            EquilibriumAttractor(WELL),
        )
        single = OrderedLineIntegral(system(_Unbatched), noise, grid).solve(
            EquilibriumAttractor(WELL),
        )
        assert np.array_equal(batched.values, single.values)


# --- State-dependent noise (DEQ-59) ------------------------------------------


@functools.cache
def _multiplicative_field(count):
    grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [count, count])
    solver = OrderedLineIntegral(
        system(MultiplicativeField),
        multiplicative_noise(),
        grid,
    )
    return grid, solver, solver.solve(EquilibriumAttractor(WELL))


def _multiplicative_error(count):
    grid, _solver, field = _multiplicative_field(count)
    points = grid.points()
    exact = _gradient_exact(points)
    inside = (points[:, 0] < -0.5) & (exact <= _MULTIPLICATIVE_LEVEL)
    return float(np.max(np.abs(field.values.ravel()[inside] - exact[inside])))


# Below the double well's level of 2: on {U <= 2} the largest error at 161
# points (2.3e-2 with the default update, 2.2e-2 with the parabola, at
# (-0.52, 1.31) for the latter) sits at the top of the set, 0.19 from the
# box's edge, and held the order to 0.77 (0.86); that point is in the left
# well's basin - the flow from it reaches (-1, 0).
_MULTIPLICATIVE_LEVEL = 1.5


class TestMultiplicativeNoise(TestCase):
    """``f = -D(x) grad V + c R grad V`` with a state-dependent ``D``: ``U = 2 V``."""

    def test_the_construction_solves_the_hamilton_jacobi_equation(self):
        # Checked here rather than trusted: grad U . f + (1/2) grad U^T D grad U
        # at U = 2 (V - V_min) vanishes to rounding at random states.
        drift, noise = system(MultiplicativeField), multiplicative_noise()
        rng = np.random.default_rng(4)
        for state in rng.uniform([-2.0, -1.5], [1.0, 1.5], (50, 2)):
            slope = 2.0 * np.array([state[0] ** 3 - state[0], state[1]])
            diffusion = noise.columns_at(state).T @ noise.columns_at(state)
            residual = slope @ np.array(drift.eval(list(state))) + 0.5 * (
                slope @ diffusion @ slope
            )
            assert abs(residual) < 1.0e-12

    def test_the_cubic_term_sees_the_varying_noise(self):
        # U = 2 (V - V_min) whatever D(x) is, so U3 = -2 (x + 1)^3 here too;
        # without the D1 term of the cubic equation it is not.
        coefficients = _cubic_coefficients(
            system(MultiplicativeField),
            multiplicative_noise(),
            np.array(WELL),
            np.diag([2.0, 1.0]),
        )
        assert np.allclose(coefficients, [-2.0, 0.0, 0.0, 0.0], atol=1.0e-5)

    @pytest.mark.slow
    def test_the_grid_converges_to_twice_the_potential(self):
        # Measured 0.139, 0.0327, 0.00776 on {U <= 1.5}: orders 2.09, 2.07
        # (parabolic: 0.141, 0.0333, 0.00808). On {U <= 2} 0.156, 0.0392,
        # 0.0231, order 0.77 at the last (see _MULTIPLICATIVE_LEVEL) - not
        # from the radius, which is 8 on every grid here.
        errors = [_multiplicative_error(count) for count in (41, 81, 161)]
        assert errors[0] > errors[1] > errors[2]
        assert min(_orders(errors)) > 1.8

    @pytest.mark.slow
    def test_gmam_converges_to_twice_the_potential(self):
        gmam = GeometricMinimumAction(
            system(MultiplicativeField),
            multiplicative_noise(),
        )
        target = np.array([-0.6, 0.5])
        exact = float(_gradient_exact(target[None, :])[0])
        # Measured 1.36e-4, 3.11e-5, 1.04e-5: orders 2.12, 1.58.
        errors = [
            abs(gmam.path(np.linspace(WELL, target, nodes)).action - exact)
            for nodes in (41, 81, 161)
        ]
        assert errors[2] < 2.0e-5
        assert min(_orders(errors)) > 1.0

    @pytest.mark.slow
    def test_gmam_agrees_with_the_grid_within_its_error_estimate(self):
        grid, solver, field = _multiplicative_field(81)
        index = np.ravel_multi_index(grid.nearest([-0.6, 0.5]), grid.shape)
        checked = solver.check(field, [index])
        target = grid.points()[index]
        value, _own = solver.reference(field, target)
        assert abs(field.values.ravel()[index] - value) <= checked.errors.ravel()[index]
        exact = float(_gradient_exact(target[None, :])[0])
        assert abs(value - exact) < abs(field.values.ravel()[index] - exact)
