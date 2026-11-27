import numpy as np
import pytest

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
        return [-(z**3)]

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
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
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
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
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
        assert (
            len(positive_values) >= len(solution_values) * 0.9
        )  # At least 90% positive

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        # Set up equation
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        parameters = [Parameter(name="sigma", value=0.5)]

        func = TestAlphaStableFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
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
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
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


# The ticket's reproduction (DEQ-86): ε = 1 on a grid of 20,000 steps over [0, 1].
# Before the fix each grid step took one fast step of the map, so the path ran
# 20,000 units of slow time with ε = 1 and overflowed in the z update at step
# 17,773, whatever the grid.
_FINE = 1.0 / 20_000


class _Linear(StochasticFunction):
    def __init__(self, variables, parameters, time=None):
        super().__init__(variables, parameters, [Variable(name="dZ/dt")], time)
        self.sigma = parameters[0].value

    def eval(self, point, time=None):
        return [-point[0]]

    def diffusion(self, point, time=None):
        return [self.sigma * point[0]]


def _linear_equation() -> DifferentialEquation:
    """dZ = -Z dt + 0.3 Z ⋄ dW^α,η,β, the equation the overflow was found on."""
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="Z", discretization=[])]
    parameters = [Parameter(name="sigma", value=0.3)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=parameters,
        derivative=_Linear(variables, parameters, time),
    )


def _solve_linear(epsilon: float, step_size: float) -> list[float]:
    """The path of :func:`_linear_equation` from Z = 1 over [0, 1]."""
    config = GottwaldMelbourneConfig(
        epsilon=epsilon,
        start_time=0.0,
        end_time=1.0,
        step_size=step_size,
        random_seed=1,
    )
    curve = GottwaldMelbourneSolver(config).solve(_linear_equation(), [1.0])
    return curve.results[0].discretization


class TestEpsilonAgainstTheStepSize:
    """ε may not exceed the step size (DEQ-86).

    The map advances slow time by ε per fast step: the path at time t is
    z_⌊t/ε⌋ (Gottwald & Melbourne 2021, Theorem 4.1). A grid step shorter than
    ε cannot be resolved, and the solver took one fast step per grid step
    regardless, so the path ran ε/step_size times faster than its own clock.
    """

    def test_epsilon_one_on_a_fine_grid_is_refused_rather_than_overflowing(self):
        # Act / Assert: refused before any step is taken, not a path of inf/nan.
        with pytest.raises(ValueError, match="ε must not exceed step_size"):
            _solve_linear(epsilon=1.0, step_size=_FINE)

    def test_the_error_names_both_values(self):
        with pytest.raises(ValueError, match=r"epsilon=0\.02.*step_size=0\.01"):
            GottwaldMelbourneConfig(epsilon=0.02, step_size=0.01)

    def test_epsilon_equal_to_the_step_size_is_accepted(self):
        # Arrange / Act
        config = GottwaldMelbourneConfig(epsilon=0.01, step_size=0.01)

        # Assert
        assert config.epsilon == config.step_size

    def test_epsilon_just_above_the_step_size_is_refused(self):
        with pytest.raises(ValueError, match="ε must not exceed step_size"):
            GottwaldMelbourneConfig(epsilon=0.010000000000000002, step_size=0.01)

    def test_the_solver_refuses_a_config_changed_after_it_was_built(self):
        # Arrange: a valid config whose step is then shortened below ε.
        config = GottwaldMelbourneConfig(epsilon=0.01, step_size=0.01)
        config.step_size = 0.005

        # Act / Assert
        with pytest.raises(ValueError, match="ε must not exceed step_size"):
            GottwaldMelbourneSolver(config)

    @pytest.mark.parametrize("epsilon", [0.005, 0.01])
    def test_a_path_with_epsilon_within_the_step_is_solved(self, epsilon):
        # Arrange: ε below and equal to the step. The values themselves changed on
        # purpose with the paper's observable (DEQ-108), so the path is checked for
        # what it must be rather than pinned.
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="Z", discretization=[])]
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=[],
            derivative=TestBoundaryFunction(variables, [], time),
        )
        config = GottwaldMelbourneConfig(
            alpha=1.5,
            eta=0.5,
            beta=0.5,
            epsilon=epsilon,
            start_time=0.0,
            end_time=0.05,
            step_size=0.01,
            random_seed=42,
        )

        # Act
        curve = GottwaldMelbourneSolver(config).solve(equation, [0.1])

        # Assert: the start, then a finite value at every node.
        path = curve.results[0].discretization
        assert path[0] == 0.1
        assert len(path) == 6
        assert np.all(np.isfinite(path))
