"""Euler-Maruyama, the stochastic Euler step.

The simplest consistent scheme for ``dX = a dt + G dW``: freeze the drift and
the diffusion at the start of the step and add one Wiener increment per driver,
``X_next = X + a h + G dW``. It keeps only the first two terms of the Ito-Taylor
expansion, so it has strong order 0.5 and weak order 1.0. It needs no
derivative of ``G`` and no iterated integral, so it accepts any noise shape:
diagonal, matrix, commuting or not.

References:
    Maruyama, G. Continuous Markov processes and stochastic equations.
    Rendiconti del Circolo Matematico di Palermo 4, 48-90, 1955.
    DOI 10.1007/BF02846028.

    Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic Differential
    Equations. Springer, 1992. DOI 10.1007/978-3-662-12616-5.
"""

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.wiener import WienerSource


class EulerMaruyamaSolver(Solver):
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs).

    The stochastic analogue of the Euler method for ODEs. Supports both Ito and
    Stratonovich interpretations of SDEs.

    For Ito SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    The discretization is: X_{n+1} = X_n + μ(X_n, t_n) Δt + σ(X_n, t_n) ΔW_n

    For Stratonovich SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t
    The equivalent Ito form is used, with drift μ_corrected = μ + (1/2)σ ∂σ/∂x
    (the sign is argued in
    :mod:`discrecontinual_equations.solver.stochastic.coefficients`; with
    diagonal noise the correction applies to systems component by component, with
    matrix noise it is (1/2) Σ_k (∂G_k/∂x) G_k over the columns).

    Where ΔW_n ~ N(0, Δt) is a Wiener increment. This method achieves strong order 0.5
    and weak order 1.0 convergence.

    **Noise shape.** ``diffusion()`` may return a vector (diagonal noise, one Wiener
    process per component, the step above) or an ``n x m`` matrix ``G`` (DEQ-28):
    ``m`` independent drivers and the step ``X_{n+1} = X_n + μ Δt + G ΔW_n`` with
    ``ΔW_n`` of length ``m``. Euler-Maruyama needs nothing beyond that - no Levy
    areas, no commutativity - which is why it is the scheme to reach for with
    coupled noise, and the one the Milstein and ``srk*`` solvers name when they
    refuse a matrix. The vector form is stepped exactly as before.

    References:
        Maruyama, G. Continuous Markov processes and stochastic equations.
        Rendiconti del Circolo Matematico di Palermo 4, 48-90, 1955.
        DOI 10.1007/BF02846028.

        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer-Verlag, 1992.
        DOI 10.1007/978-3-662-12616-5.
    """

    def __init__(
        self,
        solver_config: EulerMaruyamaConfig,
        wiener: WienerSource | None = None,
    ):
        """Set up the step and its generator.

        Args:
            solver_config: The time grid, the seed and the calculus.
            wiener: Where the increments come from. ``None``, the default,
                draws them from a generator seeded with ``random_seed``. A
                :class:`~.wiener.BrownianPath` drives the solver along a path
                fixed in advance, as a strong-convergence study needs.
        """
        super().__init__(solver_config=solver_config)

        # A generator of this solver's own, never np.random.seed. Seeding the global
        # stream is a correctness hazard rather than a convenience: it changes every
        # draw made anywhere afterwards, so one solver's result can depend on how many
        # draws something else happened to take first, and two solvers constructed with
        # the same seed share one stream instead of repeating one another. A seed of
        # None still means fresh entropy, as before.
        self._generator = np.random.default_rng(self.solver_config.random_seed)

        # A caller may supply the increments instead - a Brownian path fixed in
        # advance, which is what a strong-convergence measurement needs, since the
        # scheme and the exact solution must be driven by the same path. The solver's
        # own draw stays the default rather than being routed through a
        # GaussianWienerSource, which takes two normals per step, so that a seeded run
        # gives the same numbers it always has.
        self._wiener = wiener

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """Integrate the equation over the configured time grid.

        Args:
            equation: The SDE. Its ``diffusion`` may return a vector (diagonal
                noise) or an ``n x m`` matrix (``m`` drivers).
            initial_values: The state at ``start_time``.

        Returns:
            The solution curve, one point for each grid time.
        """
        self.solution = self._new_solution(equation)

        coefficients = ItoCoefficients(equation, self.solver_config.calculus)

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)
        # One Wiener channel per driver: the component count for a vector
        # diffusion, the column count for a matrix. The shape is a property of the
        # equation, so it is read once.
        drivers = coefficients.drivers(y, t)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # The time grid is read once: ``times`` builds it afresh on every read,
        # so reading it per step made a path of K steps cost O(K^2) (DEQ-47).
        times = self.solver_config.times

        # Time stepping loop
        for i in range(len(times) - 1):
            # Current time
            t_current = times[i]

            # Evaluate drift and diffusion in the Ito sense. The step below is an Ito
            # scheme, so a Stratonovich equation is integrated as the Ito equation
            # with drift a + (1/2) b b' - the correction is added, not subtracted
            # (DEQ-27). ItoCoefficients holds that sign and the derivative estimate
            # for every stochastic solver, so they agree by construction; in Ito mode
            # it returns a and b untouched.
            drift = coefficients.drift(y, t_current)
            diffusion = coefficients.diffusion(y, t_current)

            # Generate Wiener increment: delta_w ~ N(0, dt), one per driver
            dt = self.solver_config.dt
            delta_w = (
                self._generator.normal(0, np.sqrt(dt), size=drivers)
                if self._wiener is None
                else self._wiener.increments(t_current, dt, drivers).delta_w
            )

            # Euler-Maruyama step: y_{n+1} = y_n + a(y_n, t_n) dt + b(y_n, t_n) dW,
            # the last term b * dW for a vector diffusion and G @ dW for a matrix.
            y_new = y + drift * dt + coefficients.spread(diffusion, delta_w)

            # Update time and state
            t_next = times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
        return self.solution
