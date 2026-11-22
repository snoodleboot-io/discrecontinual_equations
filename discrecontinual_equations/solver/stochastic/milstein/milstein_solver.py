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
    Δt) / c1)`` Fourier terms (``LevyAreaSampler.terms``, ``c1`` the config's
    ``levy_area_accuracy``, default 1), which is why a warning says so once per
    solve;
    before DEQ-36 such noise was refused. A vector diffusion is ``G = diag(b)``:
    with each ``b_i`` depending only on ``x_i`` the noise commutes and the term is
    the familiar per-component ``(1/2) b_i b_i' (ΔW_i² - Δt)``. That case is
    stepped from the ``n x n`` Jacobian of ``b`` rather than the ``n x n x n``
    array of products, ``O(n C_b)`` time and ``O(n^2)`` memory a step where the
    general form takes ``O(n^4)`` and ``O(n^3)``, and agrees with it to rounding
    (DEQ-48); vector noise that does not commute takes the general form.

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
    with slope 0.78 there. Mostly that is the truncation rule's ``max(1,
    ceil(.))``: ``n`` stays at 1 for ``Δt >= 2^-5``, and with ``n`` fixed the
    per-step error is of order ``Δt`` and the global one of order ``Δt^(1/2)``
    (still far below the bound: ``0.07`` to ``0.21 Δt`` there); order 1.0 needs ``n``
    to grow like ``Δt^(-1/2)``, which the ceiling only does in steps. The rest is
    that for ``n <= 5`` the sampler's error falls more slowly than ``1 / n``. For
    ``Δt = 2^-14 ... 2^-20`` the slope is 0.98.

    **The accuracy setting (DEQ-67).** ``levy_area_accuracy`` is ``c1`` in the
    rule's per-step bound ``c1 Δt^(3/2)`` (Kastner and Rossler (64)-(67)); ``n``
    grows like ``1 / c1``. Sampler-driven on the same equation, ``Δt = 2^-3 ...
    2^-10``, 300-400 paths against a reference coupled to ``16 n`` terms
    (bootstrap standard error of the slope 0.01): ``c1 = 1`` slope 0.75, error
    at most ``0.40 Δt``, ``n = 1 ... 5``; ``0.2``: 0.91, ``0.10 Δt``, ``n = 2
    ... 21``; ``0.1``: 0.96, ``0.05 Δt``, ``n = 4 ... 42``; ``0.05``: 0.96,
    ``0.026 Δt``, ``n = 8 ... 84`` (0.975 from the closed form of Mrongowius
    and Rossler's Theorem 4.4 (i); ``0.02`` gives 0.99 there). The extra cost
    is ``2 m`` normals per extra term: at ``Δt = 2^-10`` 339 normals a step at
    ``0.05`` against 23 at 1, the sampler alone 1.8 times slower, a two-driver
    solve no slower within timing noise (the per-step Python work dominates).
    The default stays 1: it has order 1.0 as ``Δt -> 0`` and is within its bound
    at every ``Δt``, the normals grow like ``m^(3/2) / c1`` and so matter at
    fine steps and many drivers, and changing it would change every seeded
    non-commuting run. Wiktorsson's tail approximation (2001) is not an option
    to add: Mrongowius and Rossler's second remainder is its refinement, with an
    error constant ``sqrt(5)`` times smaller for the same ``n`` (Kastner and
    Rossler, Theorem 2 (iii), (iv)), and it is used at every ``n``.

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
        sequence = np.random.SeedSequence(solver_config.random_seed)
        self._areas = LevyAreaSampler(
            np.random.default_rng(sequence.spawn(1)[0]),
            accuracy=solver_config.levy_area_accuracy,
        )

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        self.solution = self._new_solution(equation)

        coefficients = ItoCoefficients(equation, self.solver_config.calculus)
        ito = self.solver_config.calculus == "ito"

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values, dtype=float)
        drivers = coefficients.drivers(y, t)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # The time grid is read once: ``times`` builds it afresh on every read,
        # so reading it per step made a path of K steps cost O(K^2) (DEQ-47).
        times = self.solver_config.times

        # Time stepping loop
        warned = False
        for i in range(len(times) - 1):
            # Current time
            t_current = times[i]

            # The drift as written: the Stratonovich form of the step below absorbs
            # the (1/2) sum_k (dG_k/dx) G_k correction, so the drift is not corrected
            # here in either calculus.
            drift = coefficients.written_drift(y, t_current)
            diffusion = coefficients.diffusion(y, t_current)
            # Diagonal noise whose products commute takes the n^2 form of the
            # term (DEQ-48); a matrix, or a vector whose b_k depends on another
            # component than its own so that the noise does not commute, takes
            # the general form, which this leaves as it was.
            diagonal = self._commuting_diagonal(coefficients, y, t_current, diffusion)
            if diagonal is None:
                products = self._products(coefficients, y, t_current, diffusion)
                commuting = self._commute(products)
            else:
                commuting = True
            if not commuting and not warned:
                self._warn_cost(drivers, self.solver_config.levy_area_accuracy)
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
            if diagonal is not None:
                correction = self._diagonal_correction(
                    diagonal,
                    delta_w,
                    dt,
                    ito=ito,
                )
            else:
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
            t_next = times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
        return self.solution

    @staticmethod
    def _commuting_diagonal(
        coefficients: ItoCoefficients,
        state: np.ndarray,
        time: float,
        diffusion: np.ndarray,
    ) -> np.ndarray | None:
        """``b_j d b_k / d x_j`` as ``[j, k]`` for commuting diagonal noise, else None.

        For a vector diffusion ``G = diag(b)`` the product ``(dG_k/dx) G_j`` of
        :meth:`_products` is ``b_j (d b_k / d x_j) e_k``: one nonzero entry, so the
        ``(n, n, n)`` array that method builds in ``O(n^4)`` holds these ``n^2``
        numbers and zeros (DEQ-48). Its commutativity check reduces exactly:
        entry ``(j, k)`` of the antisymmetric part is ``P[j, k] e_k - P[k, j] e_j``,
        so the asymmetry is the largest off-diagonal ``|P[j, k]|`` and the scale
        the largest ``|P[j, k]|``, the same two numbers :meth:`_commute` compares,
        and the same verdict. They commute exactly when every ``b_k`` depends on
        ``x_k`` alone wherever ``b_j`` is not zero - Kloeden and Platen's diagonal
        noise (Section 10.3) - and then ``P`` is diagonal. None for a matrix, and
        for a vector that does not commute: its Levy-area step stays general.
        An equation that declares diagonal noise gets ``P``'s diagonal alone, a
        vector, from ``2`` evaluations instead of ``2 n``, once the declaration
        has been checked at the first state (``2 n`` evaluations, once a solve).
        """
        if ItoCoefficients.is_matrix(diffusion):
            return None
        coefficients.confirm_declaration(state, time)
        if coefficients.declared_diagonal:
            # P is diagonal by the declaration: its diagonal alone, from two
            # evaluations, in O(n).
            return diffusion * coefficients.diagonal_slopes(state, time)
        products = diffusion[:, None] * coefficients.diffusion_jacobian(state, time).T
        scale = float(np.max(np.abs(products)))
        cross = np.abs(products)
        np.fill_diagonal(cross, 0.0)
        asymmetry = float(np.max(cross))
        if scale == 0.0 or asymmetry <= _COMMUTATIVITY_TOLERANCE * scale:
            return products
        return None

    @staticmethod
    def _diagonal_correction(
        products: np.ndarray,
        delta_w: np.ndarray,
        dt: float,
        *,
        ito: bool,
    ) -> np.ndarray:
        """The Milstein term from ``products[j, k] = b_j d b_k / d x_j``, in O(n^2).

        ``(1/2) sum_{j,k} (dG_k/dx) G_j (dW_j dW_k - delta_jk dt)`` with only the
        ``e_k`` entries nonzero is, in component ``k``, ``(1/2) dW_k sum_j P[j, k]
        dW_j - (1/2) dt P[k, k]``. With ``P`` diagonal that is Kloeden and Platen's
        diagonal-noise term ``(1/2) b_k (d b_k / d x_k) (dW_k^2 - dt)``; the
        off-diagonal entries the tolerance admits are kept rather than dropped,
        so the step is the general one to rounding. The sum over ``j`` runs in
        the order the ensemble's does, which keeps the two hex-identical. A
        vector ``products`` is the diagonal of a diagonal ``P``, whose sum has
        the one term ``P[k, k] dW_k``.
        """
        if products.ndim == 1:
            own = products
            pushed = own * delta_w
        else:
            own = np.diagonal(products)
            pushed = np.zeros(delta_w.size)
            for j in range(delta_w.size):
                pushed += products[j] * delta_w[j]
        correction = 0.5 * delta_w * pushed
        if ito:
            correction -= 0.5 * dt * own
        return correction

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
    def _warn_cost(drivers: int, accuracy: float) -> None:
        message = (
            "MilsteinSolver: the noise does not commute ((dG_k/dx) G_j differs from "
            "(dG_j/dx) G_k), so every such step draws the Levy areas between its "
            f"{drivers} drivers: 2 m n + m (m + 1) / 2 extra normals a step with "
            f"n = ceil(0.092 sqrt(m / dt) / c1) Fourier terms, c1 = {accuracy:g} "
            "(levy_area_accuracy; DEQ-36, DEQ-67). "
            "EulerMaruyamaSolver needs none, at strong order 0.5."
        )
        warnings.warn(message, UserWarning, stacklevel=3)
