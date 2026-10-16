"""The ensemble solver against the per-path solvers and against closed forms.

A single member of an ensemble driven by the same Brownian path as the per-path
solver must be that solver's path - hex for hex for diagonal noise, where the
arithmetic is the same operations in the same order, and to the last bit or two
for matrix noise, where a batched product and a plain one may round once
differently. The statistical oracles are geometric Brownian motion's moments and
the rotating Ornstein-Uhlenbeck process's stationary covariance, both closed
forms, at an ensemble size that makes the Monte-Carlo error smaller than the
tolerance and in a time the suite can afford.
"""

import math
import time

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.ensemble import (
    BatchedCoefficients,
    StochasticEnsemble,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase
from tests.solver.milstein.test_milstein_solver import _shear_equation
from tests.solver.rotating_ornstein_uhlenbeck import (
    RotatingOrnsteinUhlenbeck,
    equation,
    relative_deviation,
    stationary_covariance,
)

_STEP = 2.0**-6
_RESOLUTION = 2.0**-10


def _per_path(solver_type, config, case, start, path):
    solver = solver_type(config, wiener=path)
    solver.solve(case.equation(), start)
    return np.array([result.discretization[-1] for result in solver.solution.results])


def _hex(values):
    return [float(v).hex() for v in values]


class TestOneMemberIsThePerPathSolver:
    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    @pytest.mark.parametrize(
        ("scheme", "solver_type", "config_type"),
        [
            ("euler_maruyama", EulerMaruyamaSolver, EulerMaruyamaConfig),
            ("milstein", MilsteinSolver, MilsteinConfig),
        ],
    )
    def test_diagonal_noise_is_hex_identical(
        self,
        calculus,
        scheme,
        solver_type,
        config_type,
    ):
        case = GeometricBrownianMotionCase(dimension=2, calculus=calculus)
        path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 2, seed=5)
        config = config_type(
            start_time=0.0,
            end_time=1.0,
            step_size=_STEP,
            calculus=calculus,
        )
        expected = _per_path(solver_type, config, case, [1.0, 1.0], path)
        ensemble = StochasticEnsemble(config, scheme=scheme, wiener=path)
        result = ensemble.solve(case.equation(), [1.0, 1.0])
        assert _hex(result.final[0]) == _hex(expected)

    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    @pytest.mark.parametrize(
        ("scheme", "solver_type", "config_type"),
        [
            ("euler_maruyama", EulerMaruyamaSolver, EulerMaruyamaConfig),
            ("milstein", MilsteinSolver, MilsteinConfig),
        ],
    )
    def test_matrix_noise_agrees_to_rounding(
        self,
        calculus,
        scheme,
        solver_type,
        config_type,
    ):
        case = ConformalCase(calculus=calculus)
        path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 2, seed=5)
        config = config_type(
            start_time=0.0,
            end_time=1.0,
            step_size=_STEP,
            calculus=calculus,
        )
        expected = _per_path(solver_type, config, case, [1.0, 0.0], path)
        ensemble = StochasticEnsemble(config, scheme=scheme, wiener=path)
        result = ensemble.solve(case.equation(), [1.0, 0.0])
        np.testing.assert_allclose(result.final[0], expected, rtol=1.0e-14, atol=0.0)


class TestBatchedCoefficients:
    def test_batched_stratonovich_drift_matches_the_per_point_one(self):

        case = GeometricBrownianMotionCase(
            mu=0.0,
            sigma=0.5,
            dimension=2,
            calculus="stratonovich",
        )
        points = np.array([[1.0, 4.0], [2.0, -1.0], [0.5, 0.25]])
        batched = BatchedCoefficients(case.equation(), "stratonovich").drift(
            points,
            0.0,
        )
        single = ItoCoefficients(case.equation(), "stratonovich")
        for point, row in zip(points, batched, strict=True):
            assert row == pytest.approx(single.drift(point, 0.0), rel=1e-12)

    def test_batched_column_jacobians_of_the_conformal_noise(self):
        points = np.array([[1.0, 2.0], [0.3, -0.7]])
        jacobians = BatchedCoefficients(
            ConformalCase(a=0.0, sigma=0.5).equation(),
            "ito",
        )
        result = jacobians.column_jacobians(points, 0.0)
        assert result.shape == (2, 2, 2, 2)
        for row in result:
            assert row[0] == pytest.approx(0.5 * np.eye(2), abs=1e-7)
            assert row[1] == pytest.approx(
                0.5 * np.array([[0.0, -1.0], [1.0, 0.0]]),
                abs=1e-7,
            )

    def test_drivers_and_spread_for_both_forms(self):
        vector = BatchedCoefficients(
            GeometricBrownianMotionCase(dimension=3).equation(),
            "ito",
        )
        matrix = BatchedCoefficients(ConformalCase().equation(), "ito")
        assert vector.drivers(np.ones((4, 3)), 0.0) == 3
        assert matrix.drivers(np.ones((4, 2)), 0.0) == 2
        diffusion = matrix.diffusion(np.array([[1.0, 2.0]]), 0.0)
        increments = np.array([[0.1, -0.2]])
        assert matrix.spread(diffusion, increments)[0] == pytest.approx(
            diffusion[0] @ increments[0],
        )


class TestStatisticalOracles:
    def test_geometric_brownian_motion_moments(self):
        mu, sigma, horizon, paths = 1.0, 0.5, 1.0, 20000
        case = GeometricBrownianMotionCase(mu=mu, sigma=sigma, dimension=1)
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=horizon,
            step_size=0.005,
            random_seed=17,
        )
        final = (
            StochasticEnsemble(config)
            .solve(case.equation(), [1.0], paths=paths)
            .final[:, 0]
        )
        mean = math.exp(mu * horizon)
        variance = math.exp(2.0 * mu * horizon) * (math.exp(sigma**2 * horizon) - 1.0)
        assert abs(final.mean() - mean) / mean < 0.03
        assert abs(final.var() - variance) / variance < 0.10

    def test_rotating_ornstein_uhlenbeck_covariance_quickly(self):
        # 20,000 paths over 2,000 steps: 4 s on a quiet machine, within 1% of the
        # exact covariance. The speed claim is made relative to the per-path solver
        # measured in the same run, so load on the machine cannot fail it: per path
        # the ensemble is a few hundred times cheaper, and 50x is asserted.
        exact = stationary_covariance(
            RotatingOrnsteinUhlenbeck.drift_matrix,
            RotatingOrnsteinUhlenbeck.noise_matrix,
        )
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=10.0,
            step_size=0.005,
            random_seed=3,
        )
        started = time.perf_counter()
        result = StochasticEnsemble(config).solve(
            equation(RotatingOrnsteinUhlenbeck),
            [0.0, 0.0],
            paths=20000,
        )
        ensemble_seconds = time.perf_counter() - started
        started = time.perf_counter()
        EulerMaruyamaSolver(config).solve(
            equation(RotatingOrnsteinUhlenbeck), [0.0, 0.0],
        )
        single_seconds = time.perf_counter() - started
        assert relative_deviation(np.cov(result.final.T), exact) < 0.03
        per_path_speedup = single_seconds * 20000 / ensemble_seconds
        assert per_path_speedup > 50.0, (ensemble_seconds, single_seconds)


class TestEnsembleMechanics:
    def test_keeps_samples_every_k_steps_and_broadcasts_the_start(self):
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=0.1,
            random_seed=1,
        )
        result = StochasticEnsemble(config).solve(
            GeometricBrownianMotionCase(dimension=2).equation(),
            [1.0, 2.0],
            paths=5,
            keep_every=4,
        )
        assert result.final.shape == (5, 2)
        assert result.times.tolist() == pytest.approx([0.0, 0.4, 0.8])
        assert result.samples.shape == (3, 5, 2)
        assert result.samples[0].tolist() == [[1.0, 2.0]] * 5

    def test_rows_of_initial_are_paths(self):
        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=0.1,
            step_size=0.1,
            random_seed=1,
        )
        start = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = StochasticEnsemble(config).solve(
            GeometricBrownianMotionCase(dimension=2).equation(),
            start,
        )
        assert result.final.shape == (2, 2)
        with pytest.raises(ValueError, match="rows"):
            StochasticEnsemble(config).solve(
                GeometricBrownianMotionCase(dimension=2).equation(),
                start,
                paths=3,
            )

    def test_milstein_refuses_non_commuting_noise(self):
        config = MilsteinConfig(start_time=0.0, end_time=1.0, step_size=0.01)
        with pytest.raises(ValueError, match="commuting noise"):
            StochasticEnsemble(config, scheme="milstein").solve(
                _shear_equation(),
                [1.0, 0.0],
                paths=3,
            )

    def test_unknown_scheme_is_refused(self):
        config = EulerMaruyamaConfig(start_time=0.0, end_time=1.0, step_size=0.01)
        with pytest.raises(ValueError, match="unknown ensemble scheme"):
            StochasticEnsemble(config, scheme="heun")
