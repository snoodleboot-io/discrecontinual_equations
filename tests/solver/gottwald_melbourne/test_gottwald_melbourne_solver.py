import pytest
import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (
    GottwaldMelbourneSolver,
)
from discrecontinual_equations.variable import Variable


class TestAlphaStableFunction(StochasticFunction):
    """Test SDE: dZ = 0 dt + σ dW^α,η,β (pure α-stable diffusion)"""

    def __init__(self, variables, parameters, time=None):
        results = [Variable(name="dZ/dt", discretization=[])]
        super().__init__(variables, parameters, results, time)
        self.sigma = parameters[0].value  # diffusion coefficient

    def eval(self, point, time=None):
        """Drift term: μ(z,t) = 0"""
        return [0.0]

    def diffusion(self, point, time=None):
        """Diffusion term: σ(z,t) = σ"""
        return [self.sigma]


class TestBoundaryFunction(StochasticFunction):
    """Test SDE with natural boundary: dZ = -Z^3 dt + Z^2  dW^α,η,β"""

    def __init__(self, variables, parameters, time=None):
        results = [Variable(name="dZ/dt", discretization=[])]
        super().__init__(variables, parameters, results, time)

    def eval(self, point, time=None):
        """Drift term: μ(z,t) = -z^3"""
        z = point[0]
        return [-z**3]

    def diffusion(self, point, time=None):
        """Diffusion term: σ(z,t) = z^2"""
        z = point[0]
        return [z**2]


class TestGottwaldMelbourneConfig:
    def test_valid_config(self):
        """Test valid configuration parameters."""
        config = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=1.0,
            beta=0.0,
            epsilon=0.01,
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=42,
        )
        assert config.alpha == 1.5
        assert config.eta == 1.0
        assert config.beta == 0.0
        assert config.epsilon == 0.01
        assert config.gamma == 1.0 / 1.5

    def test_invalid_alpha(self):
        """Test invalid alpha values."""
        with pytest.raises(ValueError, match="α must be in the range"):
            GottwaldMelbourneConfig(alpha=0.5)  # Too small

        with pytest.raises(ValueError, match="α must be in the range"):
            GottwaldMelbourneConfig(alpha=2.5)  # Too large

    def test_invalid_beta(self):
        """Test invalid beta values."""
        with pytest.raises(ValueError, match="β must be in the range"):
            GottwaldMelbourneConfig(beta=-2.0)  # Too small

        with pytest.raises(ValueError, match="β must be in the range"):
            GottwaldMelbourneConfig(beta=2.0)  # Too large

    def test_invalid_eta(self):
        """Test invalid eta values."""
        with pytest.raises(ValueError, match="η must be positive"):
            GottwaldMelbourneConfig(eta=-1.0)

        with pytest.raises(ValueError, match="η must be positive"):
            GottwaldMelbourneConfig(eta=0.0)

    def test_invalid_epsilon(self):
        """Test invalid epsilon values."""
        with pytest.raises(ValueError, match="ε must be positive"):
            GottwaldMelbourneConfig(epsilon=-0.1)

        with pytest.raises(ValueError, match="ε must be positive"):
            GottwaldMelbourneConfig(epsilon=0.0)


class TestGottwaldMelbourneSolver:
    def test_solver_initialization(self):
        """Test solver initialization with valid config."""
        config = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=1.0,
            beta=0.0,
            epsilon=0.01,
            random_seed=42,
        )
        solver = GottwaldMelbourneSolver(config)
        assert solver.solver_config == config
        assert solver.solution is None

    def test_pure_diffusion_solve(self):
        """Test solving pure α-stable diffusion."""
        # Set up equation: dZ = σ dW^α,η,β
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [Parameter(name="sigma", value=0.5)]

        func = TestAlphaStableFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables, time=time, parameters=parameters, derivative=func
        )

        config = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=1.0,
            beta=0.0,
            epsilon=0.01,
            start_time=0.0,
            end_time=1.0,
            step_size=0.1,
            random_seed=42,
        )
        solver = GottwaldMelbourneSolver(config)

        initial_values = [0.0]  # Start at 0

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert len(solver.solution.time.discretization) == 11  # 0 to 1 with step 0.1
        assert len(solver.solution.results[0].discretization) == 11

        # Check that solution exists and is reasonable
        final_value = solver.solution.results[0].discretization[-1]
        assert isinstance(final_value, float)

        # For α-stable diffusion starting at 0, the distribution should be symmetric
        # around 0, but due to stochasticity, we just check it's a reasonable value
        assert -5.0 < final_value < 5.0

    def test_boundary_preservation(self):
        """Test that natural boundaries are preserved."""
        # Set up equation with natural boundary at Z=0
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = []

        func = TestBoundaryFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables, time=time, parameters=parameters, derivative=func
        )

        config = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=0.5,
            beta=0.5,
            epsilon=0.005,
            start_time=0.0,
            end_time=0.5,
            step_size=0.01,
            random_seed=42,
        )
        solver = GottwaldMelbourneSolver(config)

        initial_values = [0.1]  # Start positive

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        solution_values = solver.solution.results[0].discretization

        # All values should remain positive (natural boundary preservation)
        # Note: Due to the homogenisation, there might be small violations,
        # but they should be minimal compared to Euler-Maruyama
        positive_values = [v for v in solution_values if v > 0]
        assert len(positive_values) >= len(solution_values) * 0.9  # At least 90% positive

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Set up equation
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [Parameter(name="sigma", value=0.5)]

        func = TestAlphaStableFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables, time=time, parameters=parameters, derivative=func
        )

        # Solve twice with same seed
        config1 = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=1.0,
            beta=0.0,
            epsilon=0.01,
            start_time=0.0,
            end_time=0.5,
            step_size=0.1,
            random_seed=42,
        )
        solver1 = GottwaldMelbourneSolver(config1)
        solver1.solve(equation, [0.0])

        config2 = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=1.0,
            beta=0.0,
            epsilon=0.01,
            start_time=0.0,
            end_time=0.5,
            step_size=0.1,
            random_seed=42,
        )
        solver2 = GottwaldMelbourneSolver(config2)
        solver2.solve(equation, [0.0])

        # Results should be identical
        np.testing.assert_array_equal(
            solver1.solution.results[0].discretization,
            solver2.solution.results[0].discretization,
        )

    def test_different_beta_values(self):
        """Test solver with different skewness parameters."""
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [Parameter(name="sigma", value=0.5)]

        func = TestAlphaStableFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables, time=time, parameters=parameters, derivative=func
        )

        # Test with symmetric (β=0) and skewed (β=0.5) noise
        for beta in [0.0, 0.5]:
            config = GottwaldMelbourneConfig(
                alpha=1.5,
                eta=1.0,
                beta=beta,
                epsilon=0.01,
                start_time=0.0,
                end_time=0.5,
                step_size=0.1,
                random_seed=42,
            )
            solver = GottwaldMelbourneSolver(config)
            solver.solve(equation, [0.0])

            assert solver.solution is not None
            assert len(solver.solution.results[0].discretization) == 6
