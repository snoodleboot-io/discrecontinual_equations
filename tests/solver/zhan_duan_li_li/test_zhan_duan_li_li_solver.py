import functools

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope


class TestStochasticFunction(StochasticFunction):
    """Test SDE: dZ = μ dt + σ dW (basic stochastic equation)"""

    def __init__(self, variables, parameters, time=None):
        results = [
            Variable(name=f"d{variable.name}/dt", discretization=[])
            for variable in variables
        ]
        super().__init__(
            variables=variables,
            parameters=parameters,
            results=results,
            time=time,
        )
        self.mu = parameters[0].value  # drift coefficient
        self.sigma = parameters[1].value  # diffusion coefficient

    def eval(self, point, time=None):
        """Drift term: μ(z,t) = μ"""
        return [self.mu]

    def diffusion(self, point, time=None):
        """Diffusion term: σ(z,t) = σ"""
        return [self.sigma]


class TestZhanDuanLiLiConfig:
    def test_valid_config(self):
        """Test valid configuration parameters."""
        config = ZhanDuanLiLiConfig(
            method_parameter=1.5,
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=42,
        )
        assert config.method_parameter == 1.5
        assert config.start_time == 0.0
        assert config.end_time == 1.0
        assert config.step_size == 0.01
        assert config.random_seed == 42

    def test_default_config(self):
        """Test default configuration values."""
        config = ZhanDuanLiLiConfig()
        assert config.method_parameter == 0.5  # Default to trapezoidal rule
        assert config.start_time == 0.0
        assert config.end_time == 1.0
        assert config.step_size == 0.01
        assert config.random_seed is None
        assert config.calculus == "ito"


class TestZhanDuanLiLiSolver:
    def test_solver_initialization(self):
        """Test solver initialization with valid config."""
        config = ZhanDuanLiLiConfig(
            method_parameter=2.0,
            random_seed=42,
        )
        solver = ZhanDuanLiLiSolver(config)
        assert solver.solver_config == config
        assert solver.solution is None

    def test_solve_basic_sde(self):
        """Test solving a basic SDE using placeholder Euler-Maruyama method."""
        # Set up equation: dZ = μ dt + σ dW
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [
            Parameter(name="mu", value=0.1),  # drift
            Parameter(name="sigma", value=0.2),  # diffusion
        ]

        func = TestStochasticFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        config = ZhanDuanLiLiConfig(
            method_parameter=1.0,
            start_time=0.0,
            end_time=1.0,
            step_size=0.1,
            random_seed=42,
        )
        solver = ZhanDuanLiLiSolver(config)

        initial_values = [1.0]  # Start at 1.0

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert len(solver.solution.time.discretization) == 11  # 0 to 1 with step 0.1
        assert len(solver.solution.results[0].discretization) == 11

        # Check that solution exists and is reasonable
        final_value = solver.solution.results[0].discretization[-1]
        assert isinstance(final_value, float)

        # For basic SDE starting at 1 with small drift, should be close to 1
        # but with diffusion, there's stochastic variation
        assert 0.5 < final_value < 2.0  # Allow reasonable range

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Set up equation
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [
            Parameter(name="mu", value=0.0),
            Parameter(name="sigma", value=0.5),
        ]

        func = TestStochasticFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        # Solve twice with same seed
        config1 = ZhanDuanLiLiConfig(
            method_parameter=1.0,
            start_time=0.0,
            end_time=0.5,
            step_size=0.1,
            random_seed=42,
        )
        solver1 = ZhanDuanLiLiSolver(config1)
        solver1.solve(equation, [0.0])

        config2 = ZhanDuanLiLiConfig(
            method_parameter=1.0,
            start_time=0.0,
            end_time=0.5,
            step_size=0.1,
            random_seed=42,
        )
        solver2 = ZhanDuanLiLiSolver(config2)
        solver2.solve(equation, [0.0])

        # Results should be identical (same random seed)
        np.testing.assert_array_equal(
            solver1.solution.results[0].discretization,
            solver2.solution.results[0].discretization,
        )

    def test_solution_structure(self):
        """Test that the solution has the correct structure."""
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [
            Parameter(name="mu", value=0.0),
            Parameter(name="sigma", value=1.0),
        ]

        func = TestStochasticFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        config = ZhanDuanLiLiConfig(
            method_parameter=1.0,
            start_time=0.0,
            end_time=1.0,
            step_size=0.2,
            random_seed=123,
        )
        solver = ZhanDuanLiLiSolver(config)
        solver.solve(equation, [0.5])

        # Check solution structure
        assert solver.solution is not None
        assert (
            len(solver.solution.time.discretization) == 6
        )  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        assert len(solver.solution.results) == 1  # One variable
        assert len(solver.solution.results[0].discretization) == 6

        # Check that time values are correct
        expected_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        np.testing.assert_array_almost_equal(
            solver.solution.time.discretization,
            expected_times,
        )

    def test_method_parameter_usage(self):
        """Test that method parameter is accessible."""
        config = ZhanDuanLiLiConfig(method_parameter=3.14)
        solver = ZhanDuanLiLiSolver(config)

        # The method parameter should be stored in config
        assert solver.solver_config.method_parameter == 3.14

        # Currently just a placeholder - actual usage would depend on paper implementation
        assert hasattr(solver.solver_config, "method_parameter")


# The calculus (DEQ-133). Until DEQ-133 the config forced calculus="ito" while the
# step, a two-stage Runge-Kutta step whose second stage carries weight theta on
# the increment, converged to the Ito equation with drift a + theta b b' - at
# theta = 1/2 the Stratonovich solution (Rumelin's theorem, as Saito and Mitsui
# state it). These tests pin the limit in each calculus.

_STEPS = [2.0**-k for k in range(3, 8)]
_RESOLUTION = 2.0**-10
_MEAN_PATHS = 1000


class _MultiplicativeNoise(StochasticFunction):
    """``dX = X dW`` (one driver) or ``dX = X dW_1 + X dW_2`` (a 1 x 2 matrix)."""

    def __init__(self, variables, time, drivers):
        super().__init__(variables, [], [Variable(name="dX/dt")], time)
        self._drivers = drivers

    def eval(self, point, time=None):
        return [0.0]

    def diffusion(self, point, time=None):
        if self._drivers == 1:
            return [point[0]]
        return [[point[0]] * self._drivers]


def _multiplicative(drivers: int = 1) -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="X", discretization=[])]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=_MultiplicativeNoise(variables, time, drivers),
    )


def _theta_config(theta: float):
    return functools.partial(ZhanDuanLiLiConfig, method_parameter=theta)


class TestZhanDuanLiLiCalculus:
    def test_the_configured_calculus_is_kept(self):
        config = ZhanDuanLiLiConfig(calculus="stratonovich")
        assert config.calculus == "stratonovich"

    @pytest.mark.parametrize("theta", [0.5, 1.0])
    @pytest.mark.parametrize(
        ("calculus", "expected", "other"),
        [("ito", 1.0, np.exp(0.5)), ("stratonovich", np.exp(0.5), 1.0)],
    )
    def test_geometric_brownian_motion_mean_pins_the_calculus(
        self, theta, calculus, expected, other
    ):
        # dX = X dW from X(0) = 1: E X(1) is 1 in the Ito sense and e^(1/2) in the
        # Stratonovich sense. The scheme's bias at h = 0.05 is below 0.02 in either
        # (its mean is a polynomial in h that matches the exact one to O(h)), and
        # the standard error over 1000 paths is about 0.05, so the bound of four
        # standard errors plus that bias still sits more than eight standard errors
        # from the other calculus's mean.
        equation = _multiplicative()
        finals = np.array(
            [
                ZhanDuanLiLiSolver(
                    ZhanDuanLiLiConfig(
                        method_parameter=theta,
                        step_size=0.05,
                        random_seed=seed,
                        calculus=calculus,
                    )
                )
                .solve(equation, [1.0])
                .results[0]
                .discretization[-1]
                for seed in range(_MEAN_PATHS)
            ]
        )
        error = finals.std(ddof=1) / np.sqrt(_MEAN_PATHS)
        assert abs(finals.mean() - expected) < 4.0 * error + 0.02, finals.mean()
        assert abs(finals.mean() - other) > 8.0 * error, finals.mean()

    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    def test_strong_order_one_at_one_half(self, calculus):
        # At theta = 1/2 the step is the stochastic Heun step, strong order 1.0 for
        # scalar noise. A drift built for the wrong calculus leaves an error that
        # does not shrink with h, which the bound on the finest step catches.
        errors = GeometricBrownianMotionCase(seed=300, calculus=calculus).errors(
            ZhanDuanLiLiSolver,
            ZhanDuanLiLiConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert 0.85 < strong < 1.3, errors
        assert errors[_STEPS[-1]][0] < 0.02, errors

    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    def test_other_thetas_converge_to_the_same_solution(self, calculus):
        # Away from 1/2 the step's increment term is theta b b' dW^2 rather than
        # (1/2) b b' dW^2, a mean-zero error of size h per step, so the strong
        # order is 1/2; but the drift correction puts the limit on the right
        # solution, so the error still falls to zero.
        errors = GeometricBrownianMotionCase(seed=400, calculus=calculus).errors(
            ZhanDuanLiLiSolver,
            _theta_config(1.0),
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert 0.35 < strong < 0.8, errors
        assert errors[_STEPS[-1]][0] < 0.2, errors

    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    def test_systems_with_diagonal_noise(self, calculus):
        errors = GeometricBrownianMotionCase(
            seed=500, calculus=calculus, dimension=2
        ).errors(
            ZhanDuanLiLiSolver,
            ZhanDuanLiLiConfig,
            _STEPS[-2:],
            paths=10,
            resolution=_RESOLUTION,
        )
        assert errors[_STEPS[-1]][0] < 0.02, errors

    @pytest.mark.parametrize("theta", [0.5, 1.0])
    @pytest.mark.parametrize("calculus", ["ito", "stratonovich"])
    def test_matrix_noise_takes_the_correction_over_every_driver(self, theta, calculus):
        # dX = X dW_1 + X dW_2 has X(1) = exp(W_1 + W_2 - 1) (Ito) or
        # exp(W_1 + W_2) (Stratonovich): the correction is the sum over both
        # columns, and dropping either column's half leaves an error near
        # |e^(-1/2) - 1| X(1) at every step size.
        path = BrownianPath.sample(0.0, 1.0, _RESOLUTION, 2, seed=11)
        w = float(np.sum(path.value(1.0)))
        exact = np.exp(w - (1.0 if calculus == "ito" else 0.0))
        config = ZhanDuanLiLiConfig(
            method_parameter=theta, step_size=2.0**-9, calculus=calculus
        )
        final = (
            ZhanDuanLiLiSolver(config, wiener=path)
            .solve(_multiplicative(drivers=2), [1.0])
            .results[0]
            .discretization[-1]
        )
        assert abs(final - exact) < 0.1 * exact, (final, exact)

    def test_additive_noise_is_unchanged_by_the_calculus(self):
        # b' = 0, so no correction: both calculi give the same numbers.
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [Parameter(name="mu", value=0.1), Parameter(name="s", value=0.2)]
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=TestStochasticFunction(variables, parameters, time),
        )
        finals = [
            ZhanDuanLiLiSolver(
                ZhanDuanLiLiConfig(method_parameter=0.7, random_seed=3, calculus=c)
            )
            .solve(equation, [0.0])
            .results[0]
            .discretization
            for c in ("ito", "stratonovich")
        ]
        np.testing.assert_array_equal(finals[0], finals[1])
