"""The stepping loop shared by the stochastic Runge-Kutta solvers.

The four ``srk*`` solvers differ only in the formula for one step. Everything
around that formula - owning a Wiener source seeded from the configuration,
translating the equation into Ito coefficients, walking the time grid and
recording the curve - is the same, and lives here so that the step formula is the
whole of each solver file and can be read against its derivation.

The schemes are derivative-free forms of Ito-Taylor schemes: each derivative of
the drift ``a`` or the diffusion ``b`` that the Taylor scheme needs is replaced
by a difference of ``a`` or ``b`` at supporting values near the current state.
Their stage counts say nothing about their orders. The orders follow from which
Wiener functionals they consume (see :mod:`.wiener`).

References:
    Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic Differential
    Equations. Springer, 1992. Chapters 11 and 15, the explicit strong and weak
    schemes. DOI 10.1007/978-3-662-12616-5.
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


class StochasticRungeKuttaSolver(Solver):
    """Base of the ``srk*`` solvers: one step formula, one shared driver.

    A subclass supplies ``_step``, and may refuse dimensions it was not derived
    for by overriding ``_check_dimension``.

    References:
        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer, 1992. Chapters 11 and 15.
    """

    __slots__ = ["_wiener"]

    def __init__(
        self,
        solver_config: StochasticConfig,
        wiener: WienerSource | None = None,
    ) -> None:
        """Set up the solver and its Wiener source.

        Args:
            solver_config: The time grid, the seed and the calculus.
            wiener: Where the increments come from. ``None``, the default,
                draws them from a :class:`~.wiener.GaussianWienerSource`
                seeded with ``random_seed``.
        """
        super().__init__(solver_config=solver_config)
        # The increments come from a source this solver owns, seeded from the
        # configuration, so that no solver can alter another's stream and the same
        # seed repeats the same path. A caller may pass a source of its own - a
        # Brownian path fixed in advance, which is what a convergence study needs.
        self._wiener = wiener or GaussianWienerSource(solver_config.random_seed)

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """Integrate the equation over the configured time grid.

        Args:
            equation: The SDE, with a vector ``diffusion`` (diagonal noise).
            initial_values: The state at ``start_time``.

        Returns:
            The solution curve, one point for each grid time.

        Raises:
            ValueError: When the diffusion is an ``n x m`` matrix, or when the
                scheme refuses the dimension (srk4 takes one variable only).
        """
        self.solution = self._new_solution(equation)
        config = self.solver_config
        coefficients = ItoCoefficients(equation, config.calculus)

        y = np.array(initial_values, dtype=float)
        self._check_dimension(len(y))
        self._refuse_matrix_noise(coefficients, y, config.start_time)
        self.solution.append([config.start_time, [0] * len(y), y.tolist()])

        dt = config.dt
        # The time grid is read once: ``times`` builds it afresh on every read,
        # so reading it per step made a path of K steps cost O(K^2) (DEQ-47).
        times = config.times
        for i in range(len(times) - 1):
            t_current = times[i]
            increments = self._wiener.increments(t_current, dt, len(y))
            y = self._step(y, t_current, dt, coefficients, increments)
            self.solution.append([times[i + 1], [0] * len(y), y.tolist()])
        return self.solution

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
        else; their orders are stated for scalar or diagonal noise because their
        several-channel forms (extra supporting values per pair of channels) are
        not implemented; the Levy areas themselves are sampled since DEQ-36, for
        the Milstein solver. A matrix diffusion would be misread, not integrated
        at a lower order, so it is refused up front. (srk3 has its several-channel
        form since DEQ-66 and sends only a single scalar variable through here.)
        """
        if ItoCoefficients.is_matrix(coefficients.diffusion(state, time)):
            message = (
                f"{type(self).__name__} integrates diagonal noise only: the diffusion "
                "returned an n x m matrix. Use MilsteinSolver (strong order 1.0, "
                "with Levy areas when the noise does not commute; DEQ-36) or "
                "EulerMaruyamaSolver for matrix noise."
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
