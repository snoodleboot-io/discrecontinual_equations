"""Geometric Brownian motion as the yardstick for the stochastic Runge-Kutta solvers.

``dX = mu X dt + sigma X dW`` has multiplicative noise, so every term a scheme
carries (or omits) shows up in its error, and it has the closed form
``X(t) = X(0) exp((mu - sigma^2 / 2) t + sigma W(t))`` in the Ito sense, or
``X(0) exp(mu t + sigma W(t))`` in the Stratonovich sense. Given the Brownian path
the exact value at the horizon is one evaluation, so a scheme can be compared
with the truth path by path.

That comparison is only a strong-order measurement when every step size is
driven by the same Brownian path: the error ``E|X_h(T) - X(T)|`` is a distance
between two functionals of one path, and comparing different random paths at
different step sizes measures the spread of ``X(T)``, not the scheme. So each
path is drawn once on a fine grid and the solvers read their increments off it
through :class:`BrownianPath`. The weak error ``|E[X_h(T)] - E[X(T)]|`` is
estimated on the same paths as the mean of ``X_h(T) - X(T)``, whose variance is
of the order of the squared strong error rather than of ``Var X(T)``; this is
what lets a few thousand paths resolve a second-order weak error.
"""

from dataclasses import dataclass

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from discrecontinual_equations.variable import Variable


class GeometricBrownianMotion(StochasticFunction):
    """``dX_i = mu X_i dt + sigma X_i dW_i`` on each component independently."""

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
            time,
        )

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return [self.parameters[0].value * x for x in point]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return [self.parameters[1].value * x for x in point]


@dataclass
class GeometricBrownianMotionCase:
    """One parameter set, its equation, and its exact solution on a path."""

    mu: float = 1.0
    sigma: float = 0.5
    x0: float = 1.0
    horizon: float = 1.0
    dimension: int = 1
    calculus: str = "ito"
    seed: int = 0

    def equation(self) -> DifferentialEquation:
        time = Variable(name="t", discretization=[])
        variables = [
            Variable(name=f"X{i}", discretization=[]) for i in range(self.dimension)
        ]
        parameters = [
            Parameter(name="mu", value=self.mu),
            Parameter(name="sigma", value=self.sigma),
        ]
        return DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=GeometricBrownianMotion(variables, parameters, time),
        )

    def exact(self, path: BrownianPath) -> np.ndarray:
        w = path.value(self.horizon)
        stratonovich = self.calculus == "stratonovich"
        drift = self.mu if stratonovich else self.mu - self.sigma**2 / 2
        return self.x0 * np.exp(drift * self.horizon + self.sigma * w)

    def final_value(self, solver_type, config_type, step, path):
        config = config_type(
            start_time=0.0,
            end_time=self.horizon,
            step_size=step,
            calculus=self.calculus,
        )
        solver = solver_type(config, wiener=path)
        solver.solve(self.equation(), [self.x0] * self.dimension)
        return np.array(
            [result.discretization[-1] for result in solver.solution.results],
        )

    def errors(
        self,
        solver_type,
        config_type,
        steps: list[float],
        paths: int,
        resolution: float,
    ) -> dict[float, tuple[float, float]]:
        """``{h: (strong error, weak error)}`` over ``paths`` shared Brownian paths."""
        strong = dict.fromkeys(steps, 0.0)
        weak = dict.fromkeys(steps, 0.0)
        for index in range(paths):
            path = BrownianPath.sample(
                0.0,
                self.horizon,
                resolution,
                self.dimension,
                seed=self.seed + index,
            )
            exact = self.exact(path)
            for step in steps:
                difference = (
                    self.final_value(solver_type, config_type, step, path) - exact
                )
                strong[step] += float(np.mean(np.abs(difference)))
                weak[step] += float(np.mean(difference))
        return {step: (strong[step] / paths, abs(weak[step]) / paths) for step in steps}


def slope(errors: dict[float, float]) -> float:
    """Least-squares slope of ``log error`` against ``log h``: the measured order."""
    steps = np.array(sorted(errors))
    values = np.array([errors[step] for step in steps])
    return float(np.polyfit(np.log(steps), np.log(values), 1)[0])
