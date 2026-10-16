"""The stepping loop shared by the stochastic Runge-Kutta solvers.

The four ``srk*`` solvers differ only in the formula for one step. Everything
around that formula - owning a Wiener source seeded from the configuration,
translating the equation into Ito coefficients, walking the time grid and
recording the curve - is the same, and lives here so that the step formula is the
whole of each solver file and can be read against its derivation.
"""

from abc import abstractmethod

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.wiener import (
    GaussianWienerSource,
    WienerIncrements,
    WienerSource,
)
from discrecontinual_equations.variable import Variable


class StochasticRungeKuttaSolver(Solver):
    """Base of the ``srk*`` solvers: one step formula, one shared driver."""

    __slots__ = ["_wiener"]

    def __init__(
        self,
        solver_config: StochasticConfig,
        wiener: WienerSource | None = None,
    ) -> None:
        super().__init__(solver_config=solver_config)
        # The increments come from a source this solver owns, seeded from the
        # configuration, so that no solver can alter another's stream and the same
        # seed repeats the same path. A caller may pass a source of its own - a
        # Brownian path fixed in advance, which is what a convergence study needs.
        self._wiener = wiener or GaussianWienerSource(solver_config.random_seed)

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        results = [
            Variable(name=f"Integral of {variable.name}")
            for variable in equation.derivative.variables
        ]
        self.solution = Curve(
            time=equation.derivative.time,
            variables=equation.derivative.variables,
            results=results,
        )
        config = self.solver_config
        coefficients = ItoCoefficients(equation, config.calculus)

        y = np.array(initial_values, dtype=float)
        self._check_dimension(len(y))
        self._refuse_matrix_noise(coefficients, y, config.start_time)
        self.solution.append([config.start_time, [0] * len(y), y.tolist()])

        dt = config.dt
        for i in range(config.n_steps):
            t_current = config.times[i]
            increments = self._wiener.increments(t_current, dt, len(y))
            y = self._step(y, t_current, dt, coefficients, increments)
            self.solution.append([config.times[i + 1], [0] * len(y), y.tolist()])

    def _check_dimension(self, dimension: int) -> None:
        """Refuse systems a scheme was not derived for; the default accepts all."""

    def _refuse_matrix_noise(
        self,
        coefficients: ItoCoefficients,
        state: np.ndarray,
        time: float,
    ) -> None:
        """Refuse an ``n x m`` diffusion: these steps read one coefficient per channel.

        Every ``srk*`` step formula reads ``diffusion[channel]`` and perturbs one
        component per supporting value, which is the diagonal case and nothing
        else; their orders are stated for scalar noise because the Levy areas a
        coupled system needs have no sampler (DEQ-25). A matrix diffusion would
        be misread, not integrated at a lower order, so it is refused up front.
        """
        if ItoCoefficients.is_matrix(coefficients.diffusion(state, time)):
            message = (
                f"{type(self).__name__} integrates diagonal noise only: the diffusion "
                "returned an n x m matrix. Use EulerMaruyamaSolver for any matrix "
                "noise, or MilsteinSolver when the noise commutes (DEQ-28)."
            )
            raise ValueError(message)

    @staticmethod
    @abstractmethod
    def _step(
        y: np.ndarray,
        t: float,
        h: float,
        coefficients: ItoCoefficients,
        increments: WienerIncrements,
    ) -> np.ndarray:
        """Advance ``y`` from ``t`` to ``t + h`` with the given increments."""
        raise NotImplementedError
