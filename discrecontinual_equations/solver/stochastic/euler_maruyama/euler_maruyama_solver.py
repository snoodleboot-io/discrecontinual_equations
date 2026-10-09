import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.wiener import WienerSource
from discrecontinual_equations.variable import Variable


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
    :mod:`discrecontinual_equations.solver.stochastic.coefficients`; with the
    diagonal noise this interface implies, the correction applies to systems
    component by component).

    Where ΔW_n ~ N(0, Δt) is a Wiener increment. This method achieves strong order 0.5
    and weak order 1.0 convergence.

    References:
    - Maruyama, Gisiro. "Continuous Markov processes and stochastic equations"
      Rendiconti del Circolo Matematico di Palermo, 1955
    - Kloeden, Peter E. and Platen, Eckhard. "Numerical Solution of Stochastic
      Differential Equations" Springer-Verlag, 1992
    """

    def __init__(
        self,
        solver_config: EulerMaruyamaConfig,
        wiener: WienerSource | None = None,
    ):
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

        coefficients = ItoCoefficients(equation, self.solver_config.calculus)

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # Time stepping loop
        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]

            # Evaluate drift and diffusion in the Ito sense. The step below is an Ito
            # scheme, so a Stratonovich equation is integrated as the Ito equation
            # with drift a + (1/2) b b' - the correction is added, not subtracted
            # (DEQ-27). ItoCoefficients holds that sign and the derivative estimate
            # for every stochastic solver, so they agree by construction; in Ito mode
            # it returns a and b untouched.
            drift = coefficients.drift(y, t_current)
            diffusion = coefficients.diffusion(y, t_current)

            # Generate Wiener increment: ΔW ~ N(0, dt)
            dt = self.solver_config.dt
            dW = (
                self._generator.normal(0, np.sqrt(dt), size=len(y))
                if self._wiener is None
                else self._wiener.increments(t_current, dt, len(y)).delta_w
            )

            # Euler-Maruyama step: y_{n+1} = y_n + μ(y_n, t_n) dt + σ(y_n, t_n) dW
            y_new = y + drift * dt + diffusion * dW

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
