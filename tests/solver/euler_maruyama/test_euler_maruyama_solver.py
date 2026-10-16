import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope
from tests.solver.rotating_ornstein_uhlenbeck import (
    OneDriverOrnsteinUhlenbeck,
    RotatingOrnsteinUhlenbeck,
    relative_deviation,
    sample_covariance,
    stationary_covariance,
)

_STEPS = [2.0**-k for k in range(3, 8)]
_RESOLUTION = 2.0**-10


class StochasticTestFunction(StochasticFunction):
    """Test SDE function: dX = -alpha X dt + sigma dW (Ornstein-Uhlenbeck process)"""

    def __init__(self, variables, parameters, time=None):
        results = [Variable(name="dX/dt", discretization=[])]
        super().__init__(variables, parameters, results, time)
        self.alpha = parameters[0].value  # drift coefficient
        self.sigma = parameters[1].value  # diffusion coefficient

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        """Drift term: a(x, t) = -alpha x"""
        x = point[0]
        return [-self.alpha * x]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        """Diffusion term: b(x, t) = sigma"""
        return [self.sigma]


class TestEulerMaruyamaSolver:
    def test_solve_stochastic_ode(self):
        # Arrange - Ornstein-Uhlenbeck process: dX = -alpha X dt + sigma dW
        time = Variable(name="t", discretization=[])
        variables = [Variable(name="X", discretization=[])]
        parameters = [
            Parameter(name="alpha", value=1.0),  # drift coefficient
            Parameter(name="sigma", value=0.5),  # diffusion coefficient
        ]

        func = StochasticTestFunction(variables, parameters, time)
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=func,
        )

        config = EulerMaruyamaConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=42,  # For reproducible results
        )
        solver = EulerMaruyamaSolver(config)

        initial_values = [1.0]  # X(0) = 1

        # Act
        solver.solve(equation, initial_values)

        # Assert
        assert solver.solution is not None
        assert len(solver.solution.time.discretization) == 101  # 0 to 1 with step 0.01
        assert len(solver.solution.results[0].discretization) == 101

        # Check that solution exists and is reasonable
        final_value = solver.solution.results[0].discretization[-1]
        assert isinstance(final_value, float)

        # For Ornstein-Uhlenbeck, the solution should be mean-reverting
        # With alpha=1, sigma=0.5, starting at x=1, it should tend toward 0
        # But due to stochasticity, we just check it's a reasonable value
        assert -2.0 < final_value < 3.0  # Allow reasonable range due to noise


class TestEulerMaruyamaConvergence:
    def test_stratonovich_converges_to_the_stratonovich_solution(self):
        # Pins the sign of the drift correction (DEQ-27): the Stratonovich solution
        # X0 exp(mu t + sigma W) is only reached if (1/2) b b' is added to the drift,
        # not taken off. With the sign inverted the scheme converges to a different
        # process: the slope measures about 0.05 and the finest-step error stays
        # near 0.7, against about 0.6 and 0.03 with the sign right. The thresholds
        # sit between, with room for the sampling noise of 40 paths.
        errors = GeometricBrownianMotionCase(seed=200, calculus="stratonovich").errors(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 0.35, errors
        assert errors[_STEPS[-1]][0] < 0.1, errors


class TestEulerMaruyamaMatrixNoise:
    """A matrix diffusion is stepped as ``G dW``: same scheme, coupled noise.

    Complex geometric Brownian motion is the strong oracle (two drivers, coupled,
    multiplicative; see :mod:`tests.solver.conformal_geometric_brownian_motion`):
    the slope should be Euler-Maruyama's 0.5 (measured 0.59, where the diagonal
    GBM measures the same). The rotating Ornstein-Uhlenbeck process is the weak
    one: its stationary covariance has a closed form that the off-diagonal of
    ``G G^T`` shapes, and a one-driver variant has a ``D`` of rank one.
    """

    def test_strong_order_half_on_conformal_noise(self):
        errors = ConformalCase().errors(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        order = slope(errors)
        assert order > 0.35, errors
        assert order < 0.8, errors
        assert errors[_STEPS[-1]] < 0.1, errors

    def test_conformal_noise_has_one_solution_in_either_calculus(self):
        # The Stratonovich correction (1/2) sum_k (dG_k/dx) G_k vanishes for this G,
        # so the two readings are the same equation and must agree to the
        # difference estimate's own error, not merely to the scheme's.
        ito = ConformalCase(calculus="ito").errors(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            [_STEPS[-1]],
            paths=5,
            resolution=_RESOLUTION,
        )
        stratonovich = ConformalCase(calculus="stratonovich").errors(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            [_STEPS[-1]],
            paths=5,
            resolution=_RESOLUTION,
        )
        assert abs(ito[_STEPS[-1]] - stratonovich[_STEPS[-1]]) < 1.0e-6

    def test_stationary_covariance_of_coupled_noise(self):
        sample = sample_covariance(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            RotatingOrnsteinUhlenbeck,
            horizon=500.0,
            seed=7,
        )
        drift = RotatingOrnsteinUhlenbeck.drift_matrix
        noise = RotatingOrnsteinUhlenbeck.noise_matrix
        exact = stationary_covariance(drift, noise)
        # What a diagonal reading of the same G would produce: the 0.5 dropped.
        diagonal_only = stationary_covariance(drift, np.diag(np.diag(noise)))
        assert relative_deviation(sample, exact) < 0.15
        assert relative_deviation(sample, exact) < relative_deviation(
            sample,
            diagonal_only,
        )

    def test_one_driver_for_two_components(self):
        sample = sample_covariance(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            OneDriverOrnsteinUhlenbeck,
            horizon=500.0,
            seed=11,
        )
        exact = stationary_covariance(
            OneDriverOrnsteinUhlenbeck.drift_matrix,
            OneDriverOrnsteinUhlenbeck.noise_matrix,
        )
        assert relative_deviation(sample, exact) < 0.15
