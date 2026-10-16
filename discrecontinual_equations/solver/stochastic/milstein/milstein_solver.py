import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.wiener import WienerSource
from discrecontinual_equations.variable import Variable

# How far (dG_k/dx) G_j may differ from (dG_j/dx) G_k, relative to the largest
# such product, before the noise is called non-commuting and refused.
_COMMUTATIVITY_TOLERANCE = 1.0e-6


class MilsteinSolver(Solver):
    """
    Milstein method for solving stochastic differential equations (SDEs).

    The Milstein method is a higher-order method that improves upon Euler-Maruyama
    by including second-order terms from the Ito-Taylor expansion.

    Supports both Ito and Stratonovich interpretations of SDEs.

    For Ito SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    The discretization is: X_{n+1} = X_n + μ Δt + σ ΔW + (1/2)σ ∂σ/∂x (ΔW² - Δt)

    For Stratonovich SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t
    The Ito equivalent has the drift μ_corrected = μ + (1/2)σ ∂σ/∂x (the sign is
    argued in :mod:`discrecontinual_equations.solver.stochastic.coefficients`), and
    the step is written in its Stratonovich form, which absorbs that correction:
    X_{n+1} = X_n + μ Δt + σ ΔW + (1/2)σ ∂σ/∂x ΔW², since the extra
    (1/2)σ ∂σ/∂x Δt of the corrected drift is exactly the Δt the Ito step takes
    off its last term.

    This method achieves strong order 1.0 and weak order 1.0 convergence.

    **Systems and matrix noise (DEQ-28).** For ``dX = a dt + G dW`` with ``m``
    drivers the Milstein term is ``(1/2) Σ_{j,k} (∂G_k/∂x) G_j I_(j,k)`` over
    pairs of channels, with ``I_(j,k)`` the iterated Ito integrals. Only the
    diagonal ones, ``I_(k,k) = (ΔW_k² - Δt) / 2``, have a closed form; the
    off-diagonal ones are the Levy areas, which have no exact sampler. When the
    noise *commutes* - ``(∂G_k/∂x) G_j = (∂G_j/∂x) G_k`` for every pair - the
    Levy areas drop out and the term is ``(1/2) Σ_{j,k} (∂G_k/∂x) G_j (ΔW_j ΔW_k -
    δ_jk Δt)``, which is what this solver steps, with ``δ_jk Δt`` left off in the
    Stratonovich form as above. The identity is checked at every step from the
    column Jacobians the step computes anyway (central differences, the estimate
    every stochastic solver shares through ``ItoCoefficients``), and a violation
    raises ``ValueError`` naming Euler-Maruyama rather than dropping the cross
    terms silently. A vector diffusion is ``G = diag(b)``: with each ``b_i``
    depending only on ``x_i`` the noise commutes and the term is the familiar
    per-component ``(1/2) b_i b_i' (ΔW_i² - Δt)``; with some ``b_i`` depending on
    another component it does not, and the equation is refused. (Before DEQ-28 a
    system of any kind was stepped by Euler-Maruyama with a printed warning.)

    References:
    - Milstein, Grigori N. "Numerical Integration of Stochastic Differential Equations"
      Kluwer Academic Publishers, 1995
    """

    def __init__(
        self,
        solver_config: MilsteinConfig,
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
        ito = self.solver_config.calculus == "ito"

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)
        drivers = coefficients.drivers(y, t)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # Time stepping loop
        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]

            # The drift as written: the Stratonovich form of the step below absorbs
            # the (1/2) sum_k (dG_k/dx) G_k correction, so the drift is not corrected
            # here in either calculus.
            drift = coefficients.written_drift(y, t_current)
            diffusion = coefficients.diffusion(y, t_current)
            products = self._commuting_products(coefficients, y, t_current, diffusion)

            # Generate Wiener increment: delta_w ~ N(0, dt), one per driver
            dt = self.solver_config.dt
            delta_w = (
                self._generator.normal(0, np.sqrt(dt), size=drivers)
                if self._wiener is None
                else self._wiener.increments(t_current, dt, drivers).delta_w
            )

            # Milstein term: (1/2) sum_{j,k} (dG_k/dx) G_j (dW_j dW_k - delta_jk dt),
            # the delta_jk dt only in the Ito form (see the class docstring).
            correction = 0.5 * np.einsum("jki,j,k->i", products, delta_w, delta_w)
            if ito:
                correction -= 0.5 * dt * np.einsum("kki->i", products)

            # Milstein step: y_{n+1} = y_n + a dt + b dW + correction
            y_new = (
                y + drift * dt + coefficients.spread(diffusion, delta_w) + correction
            )

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    @staticmethod
    def _commuting_products(
        coefficients: ItoCoefficients,
        state: np.ndarray,
        time: float,
        diffusion: np.ndarray,
    ) -> np.ndarray:
        """``(dG_k/dx) G_j`` for every pair as ``products[j, k, :]``, or refuse.

        The check is relative to the largest product, so additive noise (every
        Jacobian zero) passes trivially and a coupling that is small in absolute
        terms is still caught when it is the whole of what the Jacobians see. The
        tolerance sits far above the central differences' own error (about 1e-10
        relative) and far below any coupling worth the name.
        """
        jacobians = coefficients.column_jacobians(state, time)
        columns = (
            diffusion.T if ItoCoefficients.is_matrix(diffusion) else np.diag(diffusion)
        )
        products = np.einsum("kil,jl->jki", jacobians, columns)
        scale = float(np.max(np.abs(products)))
        asymmetry = float(np.max(np.abs(products - products.transpose(1, 0, 2))))
        if scale > 0.0 and asymmetry > _COMMUTATIVITY_TOLERANCE * scale:
            message = (
                "MilsteinSolver needs commuting noise: (dG_k/dx) G_j must equal "
                f"(dG_j/dx) G_k for every driver pair; they differ by {asymmetry:.3g} "
                f"against a scale of {scale:.3g}. The cross terms of a non-commuting "
                "noise need Levy areas, which no Wiener source here samples; use "
                "EulerMaruyamaSolver, which needs none (DEQ-28)."
            )
            raise ValueError(message)
        return products
