"""A planar Ornstein-Uhlenbeck process with coupled noise, and its exact covariance.

``dx = A x dt + G dW`` with ``A`` stable and ``G`` constant has a Gaussian
stationary law whose covariance ``S`` solves the Lyapunov equation
``A S + S A^T + G G^T = 0``. Only ``D = G G^T`` survives into ``S``, so this is a
check of exactly what a matrix diffusion must get right in a scheme: that ``G``
is applied to ``m`` independent increments, coupling the components the way the
off-diagonal of ``D`` says. The time average of one long path estimates ``S``
with a relative error of a few percent at a horizon of a few hundred correlation
times (measured 3-9% at ``T = 500``, ``dt = 0.01``), and a diagonal-only reading
of the same ``G`` - the off-diagonal entry dropped - gives an ``S`` the sample is
measurably farther from, which is what the test on this module asserts.

The second system has one driver for two components, ``m = 1 < n``. Its ``D`` has
rank one; the vector interface could not have expressed it at all.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.variable import Variable


class RotatingOrnsteinUhlenbeck(StochasticFunction):
    """``dx = A x dt + G dW``: a damped rotation driven by two coupled channels."""

    drift_matrix = np.array([[-1.0, -2.0], [2.0, -1.0]])
    noise_matrix = np.array([[1.0, 0.5], [0.0, 0.8]])

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
            time,
        )

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return (self.drift_matrix @ np.asarray(point, dtype=float)).tolist()

    def diffusion(
        self,
        point: list[float],  # noqa: ARG002
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return self.noise_matrix.tolist()


class OneDriverOrnsteinUhlenbeck(RotatingOrnsteinUhlenbeck):
    """The same drift with a single Brownian motion pushing both components."""

    noise_matrix = np.array([[1.0], [0.5]])


def stationary_covariance(drift: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """``S`` with ``A S + S A^T + G G^T = 0``."""
    return solve_continuous_lyapunov(drift, -noise @ noise.T)


def equation(function_type) -> DifferentialEquation:
    """The planar equation for one of the two systems above."""
    time = Variable(name="t", discretization=[])
    variables = [
        Variable(name="x", discretization=[]),
        Variable(name="y", discretization=[]),
    ]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], time),
    )


def sample_covariance(
    solver_type,
    config_type,
    function_type,
    horizon: float,
    seed: int,
) -> np.ndarray:
    """Covariance of one path over ``[0.05 T, T]`` at ``dt = 0.01``, started at rest."""
    config = config_type(
        start_time=0.0,
        end_time=horizon,
        step_size=0.01,
        random_seed=seed,
    )
    solver = solver_type(config)
    solver.solve(equation(function_type), [0.0, 0.0])
    samples = np.array([result.discretization for result in solver.solution.results])
    samples = samples[:, int(samples.shape[1] * 0.05) :]
    return np.cov(samples)


def relative_deviation(sample: np.ndarray, exact: np.ndarray) -> float:
    """Largest entry-wise deviation, relative to the largest entry of ``exact``."""
    return float(np.abs(sample - exact).max() / np.abs(exact).max())
