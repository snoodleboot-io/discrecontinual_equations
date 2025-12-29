import pytest
import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable


class TestStochasticFunction(StochasticFunction):
    """Test SDE: dZ = μ dt + σ dW (basic stochastic equation)"""

    def __init__(self, variables, parameters, time=None):
        results = [
            Variable(name=f"d{variable.name}/dt", discretization=[])
            for variable in variables
        ]
        super().__init__(variables=variables, parameters=parameters, results=results, time=time)
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
            Parameter(name="mu", value=0.1),    # drift
            Parameter(name="sigma", value=0.2), # diffusion
        ]

        func = TestStochasticFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables, time=time, parameters=parameters, derivative=func
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
            variables=variables, time=time, parameters=parameters, derivative=func
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
            variables=variables, time=time, parameters=parameters, derivative=func
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
        assert len(solver.solution.time.discretization) == 6  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
        assert len(solver.solution.results) == 1  # One variable
        assert len(solver.solution.results[0].discretization) == 6

        # Check that time values are correct
        expected_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        np.testing.assert_array_almost_equal(
            solver.solution.time.discretization, expected_times
        )

    def test_method_parameter_usage(self):
        """Test that method parameter is accessible."""
        config = ZhanDuanLiLiConfig(method_parameter=3.14)
        solver = ZhanDuanLiLiSolver(config)

        # The method parameter should be stored in config
        assert solver.solver_config.method_parameter == 3.14

        # Currently just a placeholder - actual usage would depend on paper implementation
        assert hasattr(solver.solver_config, 'method_parameter')
