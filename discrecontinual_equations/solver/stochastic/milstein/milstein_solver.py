import warnings

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.wiener import (
    LevyAreaSampler,
    WienerSource,
)
from discrecontinual_equations.variable import Variable

# How far (dG_k/dx) G_j may differ from (dG_j/dx) G_k, relative to the largest
# such product, before the noise is called non-commuting and the Levy areas drawn.
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

    **Systems and matrix noise (DEQ-28, DEQ-36).** For ``dX = a dt + G dW`` with
    ``m`` drivers the Milstein term is ``Σ_{j,k} (∂G_k/∂x) G_j I_(j,k)`` over pairs
    of channels (the Ito-Taylor ``L^j G_k I_(j,k)``), with ``I_(j,k) = ∫∫ dW_j
    dW_k`` the iterated Ito integrals, ``j`` the inner one - the convention the
    ``dx = dW_1, dy = x dW_2`` test pins, since the transpose is an order-0.5
    error there. Each is ``(1/2)(ΔW_j ΔW_k - δ_jk Δt) + A_(j,k)`` with ``A`` the
    antisymmetric matrix of Levy areas, which :class:`~.wiener.LevyAreaSampler`
    draws.

    When the noise *commutes* - ``(∂G_k/∂x) G_j = (∂G_j/∂x) G_k`` for every pair -
    the areas cancel between ``(j, k)`` and ``(k, j)`` and the term is ``(1/2)
    Σ_{j,k} (∂G_k/∂x) G_j (ΔW_j ΔW_k - δ_jk Δt)``; the step is then exactly the
    commuting step of DEQ-28, bit for bit, and no area is drawn. The identity is
    checked at every step from the column Jacobians the step computes anyway
    (central differences, the estimate every stochastic solver shares through
    ``ItoCoefficients``). When it fails the step adds ``Σ_{j,k} (∂G_k/∂x) G_j
    A_(j,k)``, with the areas from the Wiener source if one was given (a
    :class:`~.wiener.BrownianPath` reads them off its grid) and otherwise from a
    sampler on a stream spawned from ``random_seed``, so the increments a seeded
    run draws are the ones Euler-Maruyama draws with that seed. That costs
    ``2 m n + m (m + 1) / 2`` extra normals a step with ``n = ceil(0.092 sqrt(m /
    Δt))`` Fourier terms (``LevyAreaSampler.terms``), which is why a warning says
    so once per solve;
    before DEQ-36 such noise was refused. A vector diffusion is ``G = diag(b)``:
    with each ``b_i`` depending only on ``x_i`` the noise commutes and the term is
    the familiar per-component ``(1/2) b_i b_i' (ΔW_i² - Δt)``.

    **Order with non-commuting noise.** Strong order 1.0, with the truncation
    keeping the areas' L2 error at ``Δt^{3/2}`` per step. Measured on ``dx = -x/2
    dt + cos(y) dW_1``, ``dy = -y/2 dt + cos(x) dW_2``, ``(x, y)(0) = (0.3,
    -0.2)``, over ``[0, 1]`` against the scheme on the path's own grid (``2^-11``,
    whose cell areas have the exact law), 100 paths, ``Δt = 2^-2 ... 2^-7``: mean
    error 1.11e-01 ... 4.00e-03, slope 0.96; Euler-Maruyama 1.54e-01 ... 2.60e-02,
    slope 0.51. On ``dx = dW_1, dy = x dW_2``, whose ``y(T)`` is the iterated
    integral ``I_(1,2)(0, T)`` itself, the scheme reproduces a path's own
    ``I_(1,2)(0, T)`` at every step size (largest difference 3.8e-11 over 200
    paths and ``Δt = 2^-2 ... 2^-8``, the central-difference Jacobians' error),
    while Euler-Maruyama measures 0.49. Driven by the sampler, that equation's
    error is the sampler's alone: below ``0.44 Δt`` for ``Δt = 2^-3 ... 2^-10``
    (60 paths, against a reference coupled to 64 times as many Fourier terms),
    with slope 0.78 there because ``n`` is at most 5 and the sampler's error falls
    more slowly than ``1 / n`` for so few terms; for ``Δt = 2^-14 ... 2^-20`` the
    slope is 0.98.

    References:
    - Milstein, Grigori N. "Numerical Integration of Stochastic Differential Equations"
      Kluwer Academic Publishers, 1995
    - Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
      Equations", Springer, 1992 (the Milstein scheme for several drivers).
    - Mrongowius, J. and Rossler, A. "On the approximation and simulation of
      iterated stochastic integrals and the corresponding Levy areas in terms of a
      multidimensional Brownian motion", arXiv:2101.09542 (2021).
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

        # Levy areas for non-commuting noise when no source is given, on a stream of
        # their own so that drawing them leaves the increment stream untouched.
        sequence = np.random.SeedSequence(self.solver_config.random_seed)
        self._areas = LevyAreaSampler(np.random.default_rng(sequence.spawn(1)[0]))

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
        warned = False
        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]

            # The drift as written: the Stratonovich form of the step below absorbs
            # the (1/2) sum_k (dG_k/dx) G_k correction, so the drift is not corrected
            # here in either calculus.
            drift = coefficients.written_drift(y, t_current)
            diffusion = coefficients.diffusion(y, t_current)
            products = self._products(coefficients, y, t_current, diffusion)
            commuting = self._commute(products)
            if not commuting and not warned:
                self._warn_cost(drivers)
                warned = True

            # Generate Wiener increment: delta_w ~ N(0, dt), one per driver, and
            # the Levy areas when the noise does not commute.
            dt = self.solver_config.dt
            area = None
            if self._wiener is None:
                delta_w = self._generator.normal(0, np.sqrt(dt), size=drivers)
                if not commuting:
                    area = self._areas.sample(delta_w, dt)
            else:
                increments = self._wiener.increments(
                    t_current,
                    dt,
                    drivers,
                    levy_area=not commuting,
                )
                delta_w = increments.delta_w
                area = increments.levy_area

            # Milstein term: (1/2) sum_{j,k} (dG_k/dx) G_j (dW_j dW_k - delta_jk dt),
            # the delta_jk dt only in the Ito form (see the class docstring), plus
            # sum_{j,k} (dG_k/dx) G_j A_(j,k) when the noise does not commute.
            correction = 0.5 * np.einsum("jki,j,k->i", products, delta_w, delta_w)
            if ito:
                correction -= 0.5 * dt * np.einsum("kki->i", products)
            if area is not None:
                correction += np.einsum("jki,jk->i", products, area)

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
    def _products(
        coefficients: ItoCoefficients,
        state: np.ndarray,
        time: float,
        diffusion: np.ndarray,
    ) -> np.ndarray:
        """``(dG_k/dx) G_j`` for every pair as ``products[j, k, :]``."""
        jacobians = coefficients.column_jacobians(state, time)
        columns = (
            diffusion.T if ItoCoefficients.is_matrix(diffusion) else np.diag(diffusion)
        )
        return np.einsum("kil,jl->jki", jacobians, columns)

    @staticmethod
    def _commute(products: np.ndarray) -> bool:
        """Whether ``(dG_k/dx) G_j = (dG_j/dx) G_k`` for every pair.

        The check is relative to the largest product, so additive noise (every
        Jacobian zero) passes trivially and a coupling that is small in absolute
        terms is still caught when it is the whole of what the Jacobians see. The
        tolerance sits far above the central differences' own error (about 1e-10
        relative) and far below any coupling worth the name; below it the area
        terms it leaves out are at most that fraction of the products times an
        area.
        """
        scale = float(np.max(np.abs(products)))
        asymmetry = float(np.max(np.abs(products - products.transpose(1, 0, 2))))
        return scale == 0.0 or asymmetry <= _COMMUTATIVITY_TOLERANCE * scale

    @staticmethod
    def _warn_cost(drivers: int) -> None:
        message = (
            "MilsteinSolver: the noise does not commute ((dG_k/dx) G_j differs from "
            "(dG_j/dx) G_k), so every such step draws the Levy areas between its "
            f"{drivers} drivers: 2 m n + m (m + 1) / 2 extra normals a step with "
            "n = ceil(0.092 sqrt(m / dt)) Fourier terms (DEQ-36). "
            "EulerMaruyamaSolver needs none, at strong order 0.5."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
