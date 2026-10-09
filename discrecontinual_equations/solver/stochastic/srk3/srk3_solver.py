import numpy as np

from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.runge_kutta import (
    StochasticRungeKuttaSolver,
)
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.wiener import (
    WienerIncrements,
    WienerSource,
)


class SRK3Solver(StochasticRungeKuttaSolver):
    """Platen's explicit strong scheme of order 1.5 for scalar Ito SDEs.

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

    **Systems are refused.** The scheme is derived for one Wiener process. For a
    system with diagonal noise the order 1.5 expansion contains the iterated
    integrals ``I_(j,k)``, ``I_(j,0)``, ``I_(0,j)`` and ``I_(j,k,l)`` across channels
    ``j != k``, and the Levy areas among them have no exact sampler; applying the
    scalar formula component-wise would be a scheme of unknown order, which is the
    defect this file is being cured of. ``solve`` therefore raises for more than one
    variable and points at :class:`SRK2Solver`, whose order 1.0 for commuting diagonal
    noise is established.

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
        super().__init__(solver_config=solver_config, wiener=wiener)

    def _check_dimension(self, dimension: int) -> None:
        if dimension != 1:
            message = (
                "SRK3Solver is the scalar-noise scheme of Kloeden-Platen (11.2.1); "
                f"it was asked to integrate {dimension} variables. Its order 1.5 "
                "needs the Levy areas between noise channels, which cannot be "
                "sampled, so systems are refused rather than integrated at an "
                "unknown order. Use SRK2Solver (strong order 1.0 for diagonal noise "
                "whose coefficients commute) or EulerMaruyamaSolver for systems."
            )
            raise ValueError(message)

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
