"""Milstein with Levy areas against references computed here, not by the library.

``test_milstein_levy_area.py`` measures the scheme against the same scheme on a
finer grid of a :class:`BrownianPath`, whose areas the path itself computes. Here
the reference never goes through the library:

* A Brownian path is simulated on ``2^-17``; the library is given its nodes on
  ``2^-12`` and the area inside each of those cells, computed here as the
  polygon area of the ``32`` sub-cells, so the library's readings (its polygon,
  its cell sums, Chen's relation between them) are all under test.
* ``dx = dW_1, dy = x dW_2, dz = z x dW_2`` has the exact solution ``y = I_(1,2)
  = int W_1 dW_2`` and ``z = exp(y - (1/2) int W_1^2 dt)`` (Ito's formula), both
  computed here as sums over the fine grid. Milstein must reproduce ``y`` at
  every step size (it is ``sum W_1 dW_2 + I_(1,2)`` per step, which by Chen's
  relation is the whole integral), and ``z`` is a genuinely nonlinear,
  non-commuting test of the scheme.
* ``dx = -x/2 dt + cos(y) dW_1``, ``dy = -y/2 dt + cos(x) dW_2`` against a
  Milstein written out here with analytic derivatives, on the ``2^-12`` grid
  with the same cell areas.

Measured with 300 paths over ``h = 2^-2 ... 2^-7`` while writing these: the
cosine system's mean error 1.13e-01 ... 3.48e-03, slope 1.00 (Euler-Maruyama
0.51); ``z`` 1.08e-01 ... 4.75e-03, slope 0.88 with local slopes between 0.55
and 1.12 (``z`` is lognormal-like and its error is dominated by a few paths;
Euler-Maruyama 0.45); ``y`` exact to 7e-12.
"""

import warnings

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
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
from discrecontinual_equations.variable import Variable
from tests.solver.geometric_brownian_motion import slope

_FINE = 17
_NODES = 12
_SUB = 2 ** (_FINE - _NODES)
_STEPS = [2.0**-k for k in range(2, 8)]


class ExponentialOfIteratedIntegral(StochasticFunction):
    """``dx = dW_1``, ``dy = x dW_2``, ``dz = z x dW_2``: z = exp(y - int x^2 / 2)."""

    def eval(
        self,
        point: list[float],  # noqa: ARG002
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [0.0, 0.0, 0.0]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, point[0]], [0.0, point[2] * point[0]]]


class CrossCosineNoise(StochasticFunction):
    """``dx = -x/2 dt + cos(y) dW_1``, ``dy = -y/2 dt + cos(x) dW_2``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-0.5 * point[0], -0.5 * point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[np.cos(point[1]), 0.0], [0.0, np.cos(point[0])]]


def _final(  # noqa: PLR0913 (a solver, its config, a step, a path, a problem)
    solver_type,
    config_type,
    step,
    path,
    function_type,
    initial,
):
    time = Variable(name="t", discretization=[])
    variables = [Variable(name=f"v{i}", discretization=[]) for i in initial]
    results = [Variable(name=f"d{i}", discretization=[]) for i in initial]
    equation = DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], results, time),
    )
    solver = solver_type(
        config_type(start_time=0.0, end_time=1.0, step_size=step),
        wiener=path,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(equation, list(initial))
    return np.array([r.discretization[-1] for r in solver.solution.results])


class _FinePath:
    """A ``2^-17`` path, its ``2^-12`` library view, and functionals computed here."""

    def __init__(self, seed: int) -> None:
        generator = np.random.default_rng(seed)
        steps = generator.standard_normal((2**_FINE, 2)) * np.sqrt(2.0**-_FINE)
        values = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
        left = values[:-1]
        area = 0.5 * np.sum(left[:, 0] * steps[:, 1] - left[:, 1] * steps[:, 0])
        self.y = 0.5 * values[-1, 0] * values[-1, 1] + area
        self.z = np.exp(
            self.y - 0.5 * np.trapezoid(values[:, 0] ** 2, dx=2.0**-_FINE),
        )
        cells = steps.reshape(2**_NODES, _SUB, 2)
        relative = np.cumsum(cells, axis=1) - cells
        self.cell_areas = 0.5 * np.sum(
            relative[:, :, 0] * cells[:, :, 1] - relative[:, :, 1] * cells[:, :, 0],
            axis=1,
        )
        matrix = np.zeros((2**_NODES, 2, 2))
        matrix[:, 0, 1] = self.cell_areas
        matrix[:, 1, 0] = -self.cell_areas
        self.nodes = values[::_SUB]
        self.path = BrownianPath(0.0, 2.0**-_NODES, self.nodes, areas=matrix)

    def cross_cosine_reference(self, initial: tuple[float, float]) -> np.ndarray:
        """Milstein with analytic derivatives on the ``2^-12`` grid, written out.

        ``L^1 G_2 = (0, -sin x cos y)``, ``L^2 G_1 = (-sin y cos x, 0)`` and
        ``L^j G_j = 0``, so the step adds ``L^1 G_2 I_(1,2) + L^2 G_1 I_(2,1)``.
        """
        step = 2.0**-_NODES
        x, y = initial
        for dw, cell in zip(np.diff(self.nodes, axis=0), self.cell_areas, strict=True):
            i12 = 0.5 * dw[0] * dw[1] + cell
            i21 = 0.5 * dw[0] * dw[1] - cell
            x, y = (
                x - 0.5 * x * step + np.cos(y) * dw[0] - np.sin(y) * np.cos(x) * i21,
                y - 0.5 * y * step + np.cos(x) * dw[1] - np.sin(x) * np.cos(y) * i12,
            )
        return np.array([x, y])


class TestAgainstIndependentReferences:
    def test_iterated_integral_is_exact_and_its_exponential_converges(self):
        paths = 40
        milstein = dict.fromkeys(_STEPS, 0.0)
        euler = dict.fromkeys(_STEPS, 0.0)
        for index in range(paths):
            fine = _FinePath(seed=500 + index)
            for step in _STEPS:
                final = _final(
                    MilsteinSolver,
                    MilsteinConfig,
                    step,
                    fine.path,
                    ExponentialOfIteratedIntegral,
                    (0.0, 0.0, 1.0),
                )
                # I_(1,2) with I_(2,1) in its place, or the cell areas left out,
                # is an error of order 0.1 here.
                assert abs(final[1] - fine.y) < 1e-9, (step, final, fine.y)
                milstein[step] += abs(final[2] - fine.z) / paths
                euler[step] += (
                    abs(
                        _final(
                            EulerMaruyamaSolver,
                            EulerMaruyamaConfig,
                            step,
                            fine.path,
                            ExponentialOfIteratedIntegral,
                            (0.0, 0.0, 1.0),
                        )[2]
                        - fine.z,
                    )
                    / paths
                )
        assert slope(milstein) > 0.7, milstein
        assert slope(euler) < 0.65, euler
        assert milstein[_STEPS[-1]] < 0.2 * euler[_STEPS[-1]], (milstein, euler)

    def test_cross_cosine_noise_converges_at_order_one(self):
        paths = 40
        initial = (0.3, -0.2)
        milstein = dict.fromkeys(_STEPS, 0.0)
        for index in range(paths):
            fine = _FinePath(seed=900 + index)
            reference = fine.cross_cosine_reference(initial)
            for step in _STEPS:
                final = _final(
                    MilsteinSolver,
                    MilsteinConfig,
                    step,
                    fine.path,
                    CrossCosineNoise,
                    initial,
                )
                milstein[step] += np.linalg.norm(final - reference) / paths
        assert 0.85 < slope(milstein) < 1.15, milstein
        # The reference's own error is O(2^-12): far below the finest error.
        assert milstein[_STEPS[-1]] < 6e-3, milstein
