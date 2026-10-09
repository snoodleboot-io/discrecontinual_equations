import numpy as np

from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.runge_kutta import (
    StochasticRungeKuttaSolver,
)
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.wiener import (
    WienerIncrements,
    WienerSource,
)


class SRK4Solver(StochasticRungeKuttaSolver):
    """Platen's explicit weak scheme of order 2.0 for scalar Ito SDEs.

    This is the explicit order 2.0 weak scheme of Kloeden and Platen, "Numerical
    Solution of Stochastic Differential Equations" (Springer, 1992), equation
    (15.1.4), for one state variable driven by one Wiener process. It is the
    derivative-free form of the order 2.0 weak Taylor scheme (their (14.2.1))::

        Y_bar  = Y + a h + b dW
        Y+-    = Y + a h +- b sqrt(h)
        Y_next = Y + (a(Y_bar) + a) h / 2
                 + (b(Y+) + b(Y-) + 2 b) dW / 4
                 + (b(Y+) - b(Y-)) (dW^2 - h) / (4 sqrt(h))

    with ``dW = W(t + h) - W(t)``, ``N(0, h)``. Why this is weak order 2.0 and not
    strong order 2.0: expanding the stages, the first line is ``a h + (1/2) a a' h^2 +
    (1/2) a' b h dW + (1/4) a'' b^2 h dW^2``, the second is ``b dW + (1/2)(a b' + (1/2)
    b^2 b'') h dW``, and the third is the Milstein term ``(1/2) b b' (dW^2 - h)``.
    Against the order 2.0 weak Taylor scheme, ``(1/2) h dW`` stands in for
    ``I_(1,0) = dZ`` and ``I_(0,1) = h dW - dZ``: it has the same mean and the same
    covariance with ``dW`` (``h^2 / 2``) but variance ``h^3 / 4`` instead of ``h^3 /
    3``. That mismatch is invisible to any expectation at order ``h^2`` - which is
    all a weak order 2.0 scheme promises - but it is a pathwise error of order
    ``h^{3/2}``, so in the strong sense the scheme is the Milstein scheme and nothing
    more. Four evaluations of ``b`` and two of ``a`` per step.

    **Order, as measured.** Weak order 2.0 and strong order 1.0. The previous version
    of this file claimed "strong order 2.0 and weak order 4.0" and was deterministic
    (DEQ-25). Strong order 2.0 for multiplicative noise would require the iterated
    integrals ``I_(1,1,0)``, ``I_(1,0,1)``, ``I_(0,1,1)`` and ``I_(1,1,1,1)`` with
    their exact joint law; weak order 4.0 is not a scheme anyone has written down
    for a general scalar SDE, only an extrapolation result (Kloeden-Platen Section
    15.3) that applies to expectations, not to a path. Measured on geometric Brownian
    motion ``dX = X dt + 0.5 X dW``, ``X(0) = 1``, over ``[0, 1]``, against the exact
    solution ``X0 exp((mu - sigma^2/2) t + sigma W(t))``: the strong error is the
    mean absolute pathwise difference with the same Brownian path at every step size,
    and the weak error is ``|E[X_h - X_exact]|`` estimated path-coupled over the same
    paths, which removes the variance of ``X`` itself and leaves the standard error in
    brackets (100000 paths):

    ====== ============ ====================
    h      strong       weak (std. error)
    ====== ============ ====================
    1/8    1.679e-02    6.388e-03 (1e-04)
    1/16   7.584e-03    1.675e-03 (4e-05)
    1/32   3.582e-03    4.119e-04 (2e-05)
    1/64   1.751e-03    9.584e-05 (9e-06)
    1/128  8.660e-04    2.420e-05 (4e-06)
    1/256  4.320e-04    8.141e-06 (2e-06)
    slope  1.05         1.96
    ====== ============ ====================

    In the same run Euler-Maruyama measured strong 0.59 / weak 0.97 and Milstein
    strong 0.97 / weak 0.97, so the harness resolves the orders it is meant to.

    **Systems are refused.** Kloeden and Platen's scheme for several Wiener processes
    ((15.1.5)) needs, for each pair of channels, extra supporting values and an
    independent two-point variable standing in for the Levy area ``I_(j,k)``; this
    file implements the scalar scheme only, so ``solve`` raises for more than one
    variable rather than run a component-wise variant of unknown order, and points at
    :class:`SRK2Solver`.

    **Stratonovich.** A Stratonovich equation is integrated as the Ito equation with
    drift ``a + (1/2) b b'`` (see
    :mod:`discrecontinual_equations.solver.stochastic.coefficients`), so the order
    above applies to either calculus.

    References:
    - Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
      Equations", Springer, 1992, Section 15.1, equation (15.1.4); the Taylor scheme
      it is derived from is Section 14.2, equation (14.2.1).
    - Talay, D. and Tubaro, L. "Expansion of the global error for numerical schemes
      solving stochastic differential equations", Stochastic Analysis and
      Applications 8, 1990 - the extrapolation route to higher weak order.
    """

    def __init__(self, solver_config: SRK4Config, wiener: WienerSource | None = None):
        super().__init__(solver_config=solver_config, wiener=wiener)

    def _check_dimension(self, dimension: int) -> None:
        if dimension != 1:
            message = (
                "SRK4Solver is the scalar-noise weak scheme of Kloeden-Platen "
                f"(15.1.4); it was asked to integrate {dimension} variables. The "
                "several-channel form needs a two-point variable per pair of channels "
                "standing in for their Levy area, which this solver does not "
                "implement, so systems are refused rather than integrated at an "
                "unknown order. "
                "Use SRK2Solver or EulerMaruyamaSolver for systems."
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
        """One step of Kloeden-Platen (15.1.4)."""
        root = np.sqrt(h)
        delta_w = increments.delta_w
        drift = coefficients.drift(y, t)
        diffusion = coefficients.diffusion(y, t)

        # Supporting values sit at t + h, as for the other schemes: time is a
        # component of zero diffusion in the autonomous form of the equation.
        later = t + h
        predictor = y + drift * h
        drift_bar = coefficients.drift(predictor + diffusion * delta_w, later)
        diffusion_plus = coefficients.diffusion(predictor + diffusion * root, later)
        diffusion_minus = coefficients.diffusion(predictor - diffusion * root, later)

        return (
            y
            + (drift_bar + drift) * h / 2.0
            + (diffusion_plus + diffusion_minus + 2.0 * diffusion) * delta_w / 4.0
            + (diffusion_plus - diffusion_minus) * (delta_w**2 - h) / (4.0 * root)
        )
