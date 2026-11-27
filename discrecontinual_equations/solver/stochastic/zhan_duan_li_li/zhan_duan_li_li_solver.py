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
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.wiener import WienerSource
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)


class ZhanDuanLiLiSolver(Solver):
    """
    A theta-weighted stochastic Heun step, in either calculus.

    **The step.** An Euler predictor and a corrector that weights the predicted
    point by ``theta`` (``method_parameter``), reusing one increment::

        Y~      = Y + F(Y, t) h + G(Y, t) dW
        Y_{n+1} = Y + [(1 - theta) F(Y, t) + theta F(Y~, t + h)] h
                    + [(1 - theta) G(Y, t) + theta G(Y~, t + h)] dW

    ``theta = 0`` is Euler-Maruyama and ``theta = 1/2`` the stochastic Heun
    step. Every ``theta`` keeps the step explicit.

    **The calculus.** This is a two-stage stochastic Runge-Kutta step. Rumelin's
    theorem says such a step converges to the Ito equation with drift
    ``F + lambda G'G``, with ``lambda`` the second stage's weight on the increment
    times its increment coefficient, here ``theta * 1``. So the step does not
    integrate ``dX = F dt + G dW`` in either calculus unless ``F`` is adjusted.
    With ``a`` the drift as written, the solver uses::

        F = a + (kappa - theta) c,    c = sum_k (dG_k/dx) G_k

    Here ``kappa`` is 0 for an Ito equation and 1/2 for a Stratonovich one, and
    ``c`` is ``b_i b_i'`` per component for diagonal noise. The limit's drift is
    then ``a + kappa c``, which is the Ito form of the equation as written. At
    ``theta = 1/2`` a Stratonovich equation needs no correction, so no derivative
    is taken. An Ito equation uses ``F = a - (1/2) c``, which is Saito and
    Mitsui's ``f - (1/2) g'g``. The derivatives are central differences from
    :class:`~.coefficients.ItoCoefficients`, as in every stochastic solver here.
    Additive noise has ``c = 0``, so both calculi give the same numbers there.

    Before DEQ-133 the config forced ``calculus = "ito"`` and ``F = a``. That
    step converged to the drift ``a + theta c``, which is the Stratonovich
    solution at ``theta = 1/2`` and matches neither calculus otherwise. On
    ``dX = X dW`` the Ito mean ``E X(1) = 1`` came out as ``(1 + h/2)^(1/h)``,
    about 1.64.

    **Order.** At ``theta = 1/2`` the step reproduces the Ito-Taylor expansion up
    to ``(1/2) c dW^2``, so it has strong order 1.0 for scalar and diagonal
    noise. For other ``theta`` that term is ``theta c dW^2``. The difference
    ``(theta - 1/2) c (dW^2 - h)`` has mean zero but size ``h`` per step, so the
    strong order is 0.5. The weak order is 1.0 for every ``theta``.

    Measured on ``dX = X dW``, ``X(0) = 1``, ``T = 1``, with 4000 Brownian
    paths shared by ``h = 2^-3 ... 2^-7``. The strong error ``E|X_h(1) - X(1)|``
    is taken against the closed form, ``exp(W(1) - 1/2)`` (Ito) or
    ``exp(W(1))`` (Stratonovich). The weak error is ``|E[X_h(1) - X(1)]|`` on
    the same paths. The last column is the slope of the weak error computed
    exactly from the scheme's mean, ``(1 + f h + theta h + theta f^2 h^2)^(1/h)``
    with ``f = kappa - theta``, against ``E X(1) = 1`` or ``e^(1/2)``:

    ======== ============ ======== ======== ==============
    theta    calculus     strong   weak     weak, exact
    ======== ============ ======== ======== ==============
    0.5      Ito          0.95     1.02     1.00
    0.5      Stratonovich 0.93     0.97     0.98
    1.0      Ito          0.65     1.04     1.02
    1.0      Stratonovich 0.58     0.89     0.97
    ======== ============ ======== ======== ==============

    At ``theta = 1`` the strong slopes are still falling towards 0.5 (between
    successive step sizes 0.73, 0.66, 0.64, 0.57 for Ito). The Stratonovich
    weak errors at ``theta = 1`` are within one standard error of zero at the
    finest steps, which is why that row's Monte Carlo slope is low. Before
    DEQ-133 the weak error on the Ito equation did not shrink: the step's mean
    was ``(1 + theta h)^(1/h)``, 1.6471 at ``theta = 1/2`` and ``h = 2^-7``,
    against 1.

    **Matrix noise.** ``diffusion()`` may return an ``n x m`` matrix ``G``. The
    increments then act as ``G dW`` with ``dW`` of length ``m``, and ``c`` sums
    over the columns. The limit's drift is right for any ``G``, which the tests
    pin on ``dX = X dW_1 + X dW_2``. With non-commuting noise the step has no
    Levy areas, so its strong order should be 0.5 at every ``theta``; that case
    is not measured here.

    **The name.** The solver keeps its historical name. arXiv:2010.07691 by Zhan,
    Duan, Li and Li, which the docstring cited before DEQ-133, is a symplectic
    Euler scheme for Hamiltonian Marcus SDEs with Levy noise. It contains no theta
    method. This step is also not Higham's stochastic theta method, whose drift is
    implicit and whose diffusion is evaluated only at ``Y``.

    References:
    - Saito, Y. and Mitsui, T. "Discrete approximations for stochastic
      differential equations", RIMS Kokyuroku 746, 251-260 (1991): the m-stage
      Runge-Kutta form (10)-(11), Theorem 1 (Rumelin's limit drift ``f + lambda
      g'g``, ``lambda`` by (13)), and the stochastic Heun scheme (15) with the Ito
      drift (16), ``F = f - (1/2) g'g``.
    - Rumelin, W. "Numerical treatment of stochastic differential equations",
      SIAM J. Numer. Anal. 19(3), 604-613 (1982). Read through Saito and Mitsui
      above; not fetched.
    - Higham, D. J. "Mean-square and asymptotic stability of the stochastic
      theta method", SIAM J. Numer. Anal. 38(3), 753-769 (2000), eq. (1.2): the
      drift-implicit theta method this step is not.
    - Zhan, Q., Duan, J., Li, X. and Li, Y. "Symplectic method for Hamiltonian
      stochastic differential equations with multiplicative Levy noise in the
      sense of Marcus", arXiv:2010.07691 (2020): cited for the name only.
    """

    def __init__(
        self,
        solver_config: ZhanDuanLiLiConfig,
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

        # A caller may supply the increments instead, a Brownian path fixed in
        # advance, as Euler-Maruyama allows: a strong-order measurement drives the
        # scheme and the exact solution with the same path.
        self._wiener = wiener

        # Theta parameter from config (the corrector's weight on the predicted point)
        self.theta = self.solver_config.method_parameter

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """
        Solve the SDE with the theta-weighted stochastic Heun step.

        Args:
            equation: The SDE to solve. Its derivative must be a stochastic
                function, with a vector ``diffusion`` (diagonal noise).
            initial_values: Initial values for all variables.

        Returns:
            The solution curve, one point for each grid time.
        """
        self.solution = self._new_solution(equation)

        # The calculus is applied here, not in ItoCoefficients' drift: the step
        # needs a + (kappa - theta) c, which is neither the written drift nor
        # the Ito one.
        coefficients = ItoCoefficients(
            equation,
            "ito",
            typical=self.solver_config.typical_sizes,
        )
        kappa = 0.5 if self.solver_config.calculus == "stratonovich" else 0.0
        weight = kappa - self.theta

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)
        # One Wiener channel per driver: the component count for a vector
        # diffusion, the column count for a matrix.
        drivers = coefficients.drivers(y, t)

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

            # Generate Wiener increment: dW ~ N(0, dt), one per driver
            dW = (
                self._generator.normal(0, np.sqrt(dt), size=drivers)
                if self._wiener is None
                else self._wiener.increments(t_current, dt, drivers).delta_w
            )

            y = self._stochastic_theta_step(
                y, t_current, t_next, dt, dW, coefficients, weight
            )

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
        return self.solution

    def _stochastic_theta_step(
        self,
        y: np.ndarray,
        t_current: float,
        t_next: float,
        dt: float,
        dW: np.ndarray,
        coefficients: ItoCoefficients,
        weight: float,
    ) -> np.ndarray:
        """One step: Euler predictor, then the theta-weighted corrector."""
        theta = self.theta

        # The drift the step must use, a + (kappa - theta) c: zero weight (a
        # Stratonovich equation at theta = 1/2) takes no derivative at all.
        def drift(point: np.ndarray, time: float) -> np.ndarray:
            written = coefficients.written_drift(point, time)
            if weight == 0.0:
                return written
            return written + weight * coefficients.noise_drift(point, time)

        drift_current = drift(y, t_current)
        diffusion_current = coefficients.diffusion(y, t_current)

        # Predictor step (explicit Euler)
        y_predictor = (
            y + drift_current * dt + coefficients.spread(diffusion_current, dW)
        )

        drift_predictor = drift(y_predictor, t_next)
        diffusion_predictor = coefficients.diffusion(y_predictor, t_next)

        # Corrector: theta on the predicted point, 1 - theta on the current one
        drift_combined = theta * drift_predictor + (1 - theta) * drift_current
        diffusion_combined = (
            theta * diffusion_predictor + (1 - theta) * diffusion_current
        )
        return y + drift_combined * dt + coefficients.spread(diffusion_combined, dW)
