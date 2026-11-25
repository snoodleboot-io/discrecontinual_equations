"""A theta-weighted Euler predictor-corrector step for Gaussian SDEs.

The step predicts with Euler-Maruyama and corrects with a weighted average of
the drift and diffusion at the start and at the prediction, reusing the same
Wiener increment. ``theta = 0`` is Euler-Maruyama and ``theta = 1/2`` is the
stochastic Heun step. The name is historical: see the class docstring for how
the step relates to the paper it was named after, and for the calculus it
converges to.

References:
    Zhan, Q., Duan, J., Li, X. and Li, Y. Symplectic method for Hamiltonian
    stochastic differential equations with multiplicative Levy noise in the sense
    of Marcus. arXiv:2010.07691 (2020).
"""

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)


class ZhanDuanLiLiSolver(Solver):
    """A theta-weighted Euler predictor-corrector step, one Wiener increment a step.

    With ``dW ~ N(0, h)`` drawn once per step and per component (diagonal
    noise), and ``theta`` the configuration's ``method_parameter``::

        Y_pred = Y + a(Y, t) h + b(Y, t) dW
        Y_next = Y + ((1 - theta) a(Y, t) + theta a(Y_pred, t + h)) h
                   + ((1 - theta) b(Y, t) + theta b(Y_pred, t + h)) dW

    Every ``theta`` is explicit. ``theta = 0`` is Euler-Maruyama, and ``theta =
    1/2`` the stochastic Heun step.

    **Not the cited paper's scheme.** The class used to describe itself as "a
    stochastic theta method based on the trapezoidal rule" from
    arXiv:2010.07691. That preprint is Zhan, Duan, Li and Li, "Symplectic method
    for Hamiltonian stochastic differential equations with multiplicative Levy
    noise in the sense of Marcus". It builds a symplectic Euler scheme for
    Hamiltonian systems driven by Levy noise in the Marcus sense (its scheme (9),
    Theorems 3.2 and 4.1), and contains no theta or trapezoidal method. The step
    above uses Gaussian increments only, has no Hamiltonian structure, and is
    not symplectic. The reference is kept so that the name can be traced.

    **Calculus.** The configuration fixes ``calculus = "ito"``, but for
    multiplicative noise and ``theta > 0`` the step does not converge to the Ito
    solution. Expanding ``b(Y_pred) = b + b' b dW + ...``, the corrector carries
    ``theta b b' dW^2``, whose mean ``theta b b' h`` is a drift the Ito equation
    does not have. At ``theta = 1/2`` that is exactly the Stratonovich
    correction, so the stochastic Heun step converges to the Stratonovich
    solution. Other ``theta > 0`` converge to neither. On ``dX = X dW``, ``X(0) =
    1``, ``h = 0.05``, 4000 paths, the mean of ``X(1)`` is 1.62 (standard
    error 0.03) at ``theta = 1/2``. The Ito mean is 1 and the Stratonovich mean
    ``e^(1/2) = 1.65``; Euler-Maruyama on the same seeds gives 0.98 (0.02). For
    additive noise ``b' = 0`` and every ``theta`` converges to the one solution.
    For an Ito equation with multiplicative noise, use
    :class:`~.euler_maruyama.euler_maruyama_solver.EulerMaruyamaSolver` or
    :class:`~.milstein.milstein_solver.MilsteinSolver`.

    References:
        Zhan, Q., Duan, J., Li, X. and Li, Y. Symplectic method for Hamiltonian
        stochastic differential equations with multiplicative Levy noise in the
        sense of Marcus. arXiv:2010.07691 (2020). The source of the name, not
        of the step.
    """

    def __init__(self, solver_config: ZhanDuanLiLiConfig):
        """Set up the step and its generator.

        Args:
            solver_config: The time grid, the seed, and ``method_parameter``,
                the weight ``theta``.
        """
        super().__init__(solver_config=solver_config)

        # A generator of this solver's own, never np.random.seed. Seeding the global
        # stream is a correctness hazard rather than a convenience: it changes every
        # draw made anywhere afterwards, so one solver's result can depend on how many
        # draws something else happened to take first, and two solvers constructed with
        # the same seed share one stream instead of repeating one another. A seed of
        # None still means fresh entropy, as before.
        self._generator = np.random.default_rng(self.solver_config.random_seed)

        # Theta parameter from config (controls implicit/explicit balance)
        self.theta = self.solver_config.method_parameter

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """
        Integrate the equation over the configured time grid.

        Args:
            equation: The SDE to solve. Its derivative must be a stochastic
                function, with a vector ``diffusion`` (diagonal noise).
            initial_values: Initial values for all variables.

        Returns:
            The solution curve, one point for each grid time.
        """
        self.solution = self._new_solution(equation)

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # Time stepping loop
        dt = self.solver_config.dt

        # The time grid is read once: ``times`` builds it afresh on every read,
        # so reading it per step made a path of K steps cost O(K^2) (DEQ-47).
        times = self.solver_config.times
        for i in range(len(times) - 1):
            # Current time
            t_current = times[i]
            t_next = times[i + 1]

            # Generate Wiener increment: ΔW ~ N(0, dt)
            dW = self._generator.normal(0, np.sqrt(dt), size=len(y))

            # Stochastic theta method with trapezoidal rule
            y_new = self._stochastic_theta_step(y, t_current, t_next, dt, dW, equation)

            # Update state
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
        return self.solution

    def _stochastic_theta_step(self, y, t_current, t_next, dt, dW, equation):
        """
        One predictor-corrector step, as in the class docstring.

        The step is explicit for every ``theta``. The predictor is evaluated at
        ``t_next``.
        """
        theta = self.theta

        # Evaluate functions at current point
        drift_current = np.array(
            equation.derivative.eval(point=y.tolist(), time=t_current),
        )
        diffusion_current = np.array(
            equation.derivative.diffusion(point=y.tolist(), time=t_current),
        )

        # Predictor step (explicit Euler)
        y_predictor = y + drift_current * dt + diffusion_current * dW

        # Evaluate functions at predicted point
        drift_predictor = np.array(
            equation.derivative.eval(point=y_predictor.tolist(), time=t_next),
        )
        diffusion_predictor = np.array(
            equation.derivative.diffusion(point=y_predictor.tolist(), time=t_next),
        )

        # Corrector step (theta method combining current and predicted values)
        # This implements a trapezoidal-like rule for stochastic systems
        drift_combined = theta * drift_predictor + (1 - theta) * drift_current
        diffusion_combined = (
            theta * diffusion_predictor + (1 - theta) * diffusion_current
        )

        # Final update using the combined values
        y_new = y + drift_combined * dt + diffusion_combined * dW

        return y_new
