import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.iterated_integrals import (
    GaussianIteratedIntegralSource,
    IteratedIntegrals,
    IteratedIntegralSource,
)
from discrecontinual_equations.solver.stochastic.runge_kutta import (
    StochasticRungeKuttaSolver,
)
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.wiener import (
    WienerIncrements,
    WienerSource,
)


class SRK3Solver(StochasticRungeKuttaSolver):
    """Platen's explicit strong order 1.5 scheme, for one or several noise channels.

    This is the explicit order 1.5 strong scheme of Kloeden and Platen, "Numerical
    Solution of Stochastic Differential Equations" (Springer, 1992), equation
    (11.2.1), for one state variable driven by one Wiener process. It is the order
    1.5 strong Ito-Taylor scheme (their (10.4.1)) with every derivative of ``a`` and
    ``b`` replaced by a difference of supporting values::

        Y+-    = Y + a h +- b sqrt(h)
        Phi+-  = Y+ +- b(Y+) sqrt(h)
        Y_next = Y + b dW
                 + (a(Y+) - a(Y-)) dZ / (2 sqrt(h))
                 + (a(Y+) + 2 a + a(Y-)) h / 4
                 + (b(Y+) - b(Y-)) (dW^2 - h) / (4 sqrt(h))
                 + (b(Y+) - 2 b + b(Y-)) (dW h - dZ) / (2 h)
                 + (b(Phi+) - b(Phi-) - b(Y+) + b(Y-)) (dW^2 / 3 - h) dW / (4 h)

    with ``dW = W(t + h) - W(t)`` and ``dZ = integral_t^{t+h} (W(s) - W(t)) ds``, the
    pair being jointly Gaussian with ``Var dZ = h^3 / 3`` and ``Cov(dW, dZ) = h^2 / 2``
    (see :mod:`discrecontinual_equations.solver.stochastic.wiener`). Term by term:
    the ``dZ`` line is ``a' b dZ``; the ``h / 4`` line is ``a h + (1/2)(a a' + (1/2)
    b^2 a'') h^2``; the ``(dW^2 - h)`` line is the Milstein term ``(1/2) b b' (dW^2 -
    h)``; the ``(dW h - dZ)`` line is ``(a b' + (1/2) b^2 b'')(dW h - dZ)``; and the
    last line is ``(1/2) b (b b'' + b'^2)((1/3) dW^2 - h) dW``, the ``I_(1,1,1)``
    term. Those are exactly the terms of the order 1.5 strong Taylor scheme, each
    reproduced to a remainder of strong order 2.0, which is how the scheme earns its
    order. Six evaluations of ``b`` and three of ``a`` per step.

    **Order, as measured.** Strong order 1.5 for scalar noise. The previous version
    of this file claimed "order 3 convergence" from a "simplified Butcher tableau"
    and, further down, "strong order 1.5 and weak order 3.0"; it was in fact
    deterministic (DEQ-25). No weak order is claimed here: the scheme is a strong
    one, its weak order is not separately established in the derivation and was not
    separately measured. Measured on geometric Brownian motion ``dX = X dt + 0.5 X
    dW``, ``X(0) = 1``, over ``[0, 1]``, against the exact solution ``X0 exp((mu -
    sigma^2/2) t + sigma W(t))`` driven by the same Brownian path at every step size
    (100000 paths):

    ====== ============ ====================
    h      strong       weak (std. error)
    ====== ============ ====================
    1/8    1.818e-02    6.366e-03 (9e-05)
    1/16   6.609e-03    1.664e-03 (3e-05)
    1/32   2.397e-03    4.091e-04 (1e-05)
    1/64   8.665e-04    1.036e-04 (4e-06)
    1/128  3.088e-04    2.618e-05 (1e-06)
    1/256  1.090e-04    6.691e-06 (5e-07)
    slope  1.48         1.98
    ====== ============ ====================

    In the same run Euler-Maruyama measured strong 0.59 / weak 0.97 and Milstein
    strong 0.97 / weak 0.97, so the harness resolves the orders it is meant to.

    **Several channels (DEQ-66).** For ``m`` Wiener processes - an ``n x m``
    diffusion ``G`` with columns ``b^j``, or a vector of ``n > 1`` coefficients,
    which is ``G = diag(b)`` - the step is the order 1.5 strong Ito-Taylor scheme
    for several channels (Kloeden and Platen (10.4.6); Kuznetsov,
    arXiv:1802.00888, (5)) with its derivatives replaced by differences the same
    way::

        Ups+-^j       = Y + a h / m +- b^j sqrt(h)                (time t + h / m)
        Phi+-^(j1,j2) = Ups+^j1 +- b^j2(Ups+^j1) sqrt(h)
        Y_next = Y + a h + sum_j b^j dW_j
                 + sum_j (a(Ups+^j) - a(Ups-^j)) I_(j,0) / (2 sqrt(h))
                 + sum_j (a(Ups+^j) - 2 a + a(Ups-^j)) h / 4
                 + sum_j sum_k (b^j(Ups+^k) - 2 b^j + b^j(Ups-^k)) I_(0,j) / (2 h)
                 + sum_(j1,j2) (b^j2(Ups+^j1) - b^j2(Ups-^j1)) I_(j1,j2) / (2 sqrt(h))
                 + sum_(j1,j2,j3) (b^j3(Phi+^(j1,j2)) - b^j3(Phi-^(j1,j2))
                                   - b^j3(Ups+^j2) + b^j3(Ups-^j2)) I_(j1,j2,j3) / (2 h)

    with ``I_(0,j) = h dW_j - I_(j,0)``. Expanding each supporting value about
    ``Y``: the ``I_(j,0)`` line is ``L^j a`` up to ``O(h)``; the ``h / 4`` line is
    ``(1/2) L^0 a h^2`` up to ``O(h^3)`` - the ``1 / m`` in ``Ups`` is what makes
    the ``m`` first differences add up to one ``a' a`` (and, with time as a state
    of drift one, to one ``da / dt``) while each channel contributes its own
    ``(1/2) a''[b^j, b^j]``; the ``I_(0,j)`` line is ``L^0 b^j`` up to ``O(h)`` by
    the same count; the ``I_(j1,j2)`` line is ``L^j1 b^j2`` up to ``O(h)``; and in
    the last line ``b^j3(Phi+-)`` differ by ``2 sqrt(h) L^j2 b^j3(Ups+^j1)``, from
    which ``b^j3(Ups+-^j2)`` remove ``2 sqrt(h) L^j2 b^j3(Y)``, leaving ``2 h
    L^j1 L^j2 b^j3`` up to ``O(h^(3/2))``. Every coefficient is thus reproduced to
    a remainder whose product with its integral has mean zero and an L2 size of
    ``O(h^2)``, which is what order 1.5 asks of a one-step scheme. (Subtracting
    ``b^j3(Ups+-^j1)`` instead, as the scalar formula's ``b(Y+-)`` might suggest,
    leaves ``(L^j2 - L^j1) b^j3 / sqrt(h)`` for ``j1 != j2``: order 1.0.) With
    ``m = 1`` the lines are (11.2.1)'s, term for term. Per step: ``1 + 2 m`` drift
    and ``1 + 2 m + 2 m^2`` diffusion evaluations, and the ``m^3`` triple
    integrals.

    The integrals come from the Wiener source, which must then be an
    :class:`~.iterated_integrals.IteratedIntegralSource`: by default a
    :class:`~.iterated_integrals.GaussianIteratedIntegralSource` seeded from the
    configuration, which draws the twofold and threefold integrals jointly by
    Kuznetsov's Legendre series, truncated for order 1.5; or an
    :class:`~.iterated_integrals.IteratedIntegralPath` for a convergence study. A
    plain :class:`~.wiener.BrownianPath` holds no triple integrals and is refused
    for several channels. Rossler's order 1.5 stochastic Runge-Kutta methods (SIAM
    J. Numer. Anal. 48, 2010) cover scalar, diagonal and additive noise only, so
    they do not serve non-commuting noise. Kloeden and Platen give a
    derivative-free several-channel scheme in Section 11.2; the book was not to
    hand, so the form above is derived here from the Taylor scheme as quoted by
    Kuznetsov, and checked against it term by term in the tests.

    MEASURED_TABLE

    A single variable with vector noise is integrated by (11.2.1) itself, the code
    path it always took, number for number.

    **Stratonovich.** A Stratonovich equation is integrated as the Ito equation with
    drift ``a + (1/2) b b'`` (see
    :mod:`discrecontinual_equations.solver.stochastic.coefficients`), so the order
    above applies to either calculus.

    References:
    - Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
      Equations", Springer, 1992, Section 11.2, equation (11.2.1); the Taylor scheme
      it is derived from is Section 10.4, equation (10.4.1).
    """

    def __init__(self, solver_config: SRK3Config, wiener: WienerSource | None = None):
        # The default source gives the increments GaussianWienerSource gives for
        # the seed, so a scalar run is unchanged, and the triple integrals too.
        super().__init__(
            solver_config=solver_config,
            wiener=wiener or GaussianIteratedIntegralSource(solver_config.random_seed),
        )
        self._config = solver_config

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        config = self._config
        coefficients = ItoCoefficients(equation, config.calculus)
        y = np.array(initial_values, dtype=float)
        diffusion = coefficients.diffusion(y, config.start_time)
        if y.size == 1 and not ItoCoefficients.is_matrix(diffusion):
            return super().solve(equation, initial_values)
        if not isinstance(self._wiener, IteratedIntegralSource):
            message = (
                "SRK3Solver needs the triple integrals I_(j,k,l) for several noise "
                f"channels, and {type(self._wiener).__name__} does not supply them. "
                "Pass a GaussianIteratedIntegralSource or an IteratedIntegralPath "
                "(iterated_integrals.py), or leave the source to the solver."
            )
            raise ValueError(message)  # noqa: TRY004 (as every refusal here)
        channels = coefficients.drivers(y, config.start_time)

        self.solution = self._new_solution(equation)
        self.solution.append([config.start_time, [0] * len(y), y.tolist()])
        dt = config.dt
        times = config.times
        for i in range(len(times) - 1):
            integrals = self._wiener.iterated_integrals(times[i], dt, channels)
            y = self._channel_step(y, times[i], dt, coefficients, integrals)
            self.solution.append([times[i + 1], [0] * len(y), y.tolist()])
        return self.solution

    @staticmethod
    def _columns(diffusion: np.ndarray) -> np.ndarray:
        """``G``, ``n x m``: the matrix as given, or ``diag(b)`` for a vector."""
        if ItoCoefficients.is_matrix(diffusion):
            return diffusion
        return np.diag(diffusion)

    @staticmethod
    def _channel_step(
        y: np.ndarray,
        t: float,
        h: float,
        coefficients: ItoCoefficients,
        integrals: IteratedIntegrals,
    ) -> np.ndarray:
        """One step of the several-channel scheme in the class docstring."""
        columns = SRK3Solver._columns
        root = np.sqrt(h)
        drift = coefficients.drift(y, t)
        g = columns(coefficients.diffusion(y, t))
        channels = g.shape[1]
        # Time is a state of drift one and no diffusion, so it moves by h / m too.
        later = t + h / channels
        base = y + drift * h / channels
        plus = base[np.newaxis, :] + g.T * root  # plus[j] = Ups+^j
        minus = base[np.newaxis, :] - g.T * root
        drift_plus = np.array([coefficients.drift(p, later) for p in plus])
        drift_minus = np.array([coefficients.drift(p, later) for p in minus])
        g_plus = np.array([columns(coefficients.diffusion(p, later)) for p in plus])
        g_minus = np.array([columns(coefficients.diffusion(p, later)) for p in minus])
        # phi[j1, j2] = Ups+^j1 +- b^j2(Ups+^j1) sqrt(h); g_phi[j1, j2] = G there.
        shift = np.swapaxes(g_plus, 1, 2) * root
        g_phi_plus = np.array(
            [
                [
                    columns(coefficients.diffusion(plus[j1] + s, later))
                    for s in shift[j1]
                ]
                for j1 in range(channels)
            ],
        )
        g_phi_minus = np.array(
            [
                [
                    columns(coefficients.diffusion(plus[j1] - s, later))
                    for s in shift[j1]
                ]
                for j1 in range(channels)
            ],
        )

        delta_w = integrals.delta_w
        delta_z = integrals.delta_z  # I_(j,0)
        mixed = h * delta_w - delta_z  # I_(0,j)
        # (b^j3(Phi+^(j1,j2)) - b^j3(Phi-^(j1,j2))) - (b^j3(Ups+^j2) - b^j3(Ups-^j2))
        third = (g_phi_plus - g_phi_minus) - (g_plus - g_minus)[np.newaxis, :, :, :]
        return (
            y
            + drift * h
            + g @ delta_w
            + (drift_plus - drift_minus).T @ delta_z / (2.0 * root)
            + np.sum(drift_plus - 2.0 * drift + drift_minus, axis=0) * h / 4.0
            + np.sum(g_plus - 2.0 * g + g_minus, axis=0) @ mixed / (2.0 * h)
            + np.einsum("aij,aj->i", g_plus - g_minus, integrals.double) / (2.0 * root)
            + np.einsum("abij,abj->i", third, integrals.triple) / (2.0 * h)
        )

    @staticmethod
    def _step(
        y: np.ndarray,
        t: float,
        h: float,
        coefficients: ItoCoefficients,
        increments: WienerIncrements,
    ) -> np.ndarray:
        """One step of Kloeden-Platen (11.2.1)."""
        root = np.sqrt(h)
        delta_w = increments.delta_w
        delta_z = increments.delta_z
        drift = coefficients.drift(y, t)
        diffusion = coefficients.diffusion(y, t)

        # Supporting values sit at t + h: a non-autonomous equation is the autonomous
        # one with time as a component of zero diffusion, so the second difference
        # (b(Y+) - 2 b + b(Y-)) / (2 h) picks up d b / d t along with (a b' + ...),
        # which is the L^0 b term the derivation asks for.
        later = t + h
        plus = y + drift * h + diffusion * root
        minus = y + drift * h - diffusion * root
        drift_plus = coefficients.drift(plus, later)
        drift_minus = coefficients.drift(minus, later)
        diffusion_plus = coefficients.diffusion(plus, later)
        diffusion_minus = coefficients.diffusion(minus, later)
        phi_plus = plus + diffusion_plus * root
        phi_minus = plus - diffusion_plus * root
        diffusion_phi_plus = coefficients.diffusion(phi_plus, later)
        diffusion_phi_minus = coefficients.diffusion(phi_minus, later)

        return (
            y
            + diffusion * delta_w
            + (drift_plus - drift_minus) * delta_z / (2.0 * root)
            + (drift_plus + 2.0 * drift + drift_minus) * h / 4.0
            + (diffusion_plus - diffusion_minus) * (delta_w**2 - h) / (4.0 * root)
            + (diffusion_plus - 2.0 * diffusion + diffusion_minus)
            * (delta_w * h - delta_z)
            / (2.0 * h)
            + (
                diffusion_phi_plus
                - diffusion_phi_minus
                - diffusion_plus
                + diffusion_minus
            )
            * (delta_w**2 / 3.0 - h)
            * delta_w
            / (4.0 * h)
        )
