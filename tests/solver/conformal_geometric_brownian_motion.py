"""Complex geometric Brownian motion: the yardstick for matrix noise.

``dz = a z dt + sigma z dW_c`` with ``W_c = B_1 + i B_2`` two independent Brownian
motions is, as a planar system in ``(x, y) = (Re z, Im z)``, the equation with
drift ``(a x, a y)`` and the ``2 x 2`` diffusion ``sigma [[x, -y], [y, x]]``: two
drivers, each pushing both components, so nothing about it is diagonal. It is
the conformal multiplicative noise of the noisy-Hopf film, and it has a closed
form because ``(dW_c)^2 = dB_1^2 - dB_2^2 + 2i dB_1 dB_2 = dt - dt + 0 = 0``: Ito's
formula for ``log z`` carries no second-order term, so
``z(t) = z(0) exp(a t + sigma W_c(t))`` exactly, in the Ito sense. The
Stratonovich correction ``(1/2) sum_k (dG_k/dx) G_k`` is identically zero for
this ``G`` (the two columns' contributions cancel), so the same formula is the
Stratonovich solution too, and a scheme in either calculus is measured against it.

The noise is commutative - ``(dG_2/dx) G_1 = (dG_1/dx) G_2 = sigma^2 (-y, x)`` -
which is what lets the Milstein scheme reach strong order 1.0 on it without Levy
areas; Euler-Maruyama sits at 0.5. Both are measured path by path on the same
two-channel Brownian path, as :mod:`tests.solver.geometric_brownian_motion`
explains.
"""

from dataclasses import dataclass

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from discrecontinual_equations.variable import Variable


class ConformalGeometricBrownianMotion(StochasticFunction):
    """``dz = a z dt + sigma z (dB_1 + i dB_2)`` as a planar system, two drivers."""

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
            time,
        )

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        a = self.parameters[0].value
        return [a * point[0], a * point[1]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        sigma = self.parameters[1].value
        x, y = point
        return [[sigma * x, -sigma * y], [sigma * y, sigma * x]]


@dataclass
class ConformalCase:
    """One parameter set, its equation, and its exact solution on a two-channel path."""

    a: float = 0.5
    sigma: float = 0.5
    z0: tuple[float, float] = (1.0, 0.0)
    horizon: float = 1.0
    calculus: str = "ito"
    seed: int = 0

    def equation(self) -> DifferentialEquation:
        time = Variable(name="t", discretization=[])
        variables = [
            Variable(name="x", discretization=[]),
            Variable(name="y", discretization=[]),
        ]
        parameters = [
            Parameter(name="a", value=self.a),
            Parameter(name="sigma", value=self.sigma),
        ]
        return DifferentialEquation(
            variables=variables,
            time=time,
            parameters=parameters,
            derivative=ConformalGeometricBrownianMotion(variables, parameters, time),
        )

    def exact(self, path: BrownianPath) -> np.ndarray:
        """``z0 exp(a T + sigma (B_1 + i B_2))`` as ``(Re, Im)``, in either calculus."""
        w = path.value(self.horizon)
        z = complex(*self.z0) * np.exp(
            self.a * self.horizon + self.sigma * complex(w[0], w[1]),
        )
        return np.array([z.real, z.imag])

    def final_value(self, solver_type, config_type, step, path):
        config = config_type(
            start_time=0.0,
            end_time=self.horizon,
            step_size=step,
            calculus=self.calculus,
        )
        solver = solver_type(config, wiener=path)
        solver.solve(self.equation(), list(self.z0))
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
    ) -> dict[float, float]:
        """``{h: strong error}`` over ``paths`` shared two-channel Brownian paths."""
        strong = dict.fromkeys(steps, 0.0)
        for index in range(paths):
            path = BrownianPath.sample(
                0.0,
                self.horizon,
                resolution,
                2,
                seed=self.seed + index,
            )
            exact = self.exact(path)
            for step in steps:
                difference = (
                    self.final_value(solver_type, config_type, step, path) - exact
                )
                strong[step] += float(np.linalg.norm(difference))
        return {step: strong[step] / paths for step in steps}
