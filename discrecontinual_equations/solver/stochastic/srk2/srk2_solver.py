import numpy as np

from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.runge_kutta import (
    StochasticRungeKuttaSolver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.wiener import (
    WienerIncrements,
    WienerSource,
)


class SRK2Solver(StochasticRungeKuttaSolver):
    """Platen's two-stage explicit strong scheme of order 1.0 for Ito SDEs.

    This is the explicit order 1.0 strong scheme of Kloeden and Platen, "Numerical
    Solution of Stochastic Differential Equations" (Springer, 1992), equation
    (11.1.7), due to Platen (1984). It is the Milstein scheme with the derivative
    of the diffusion replaced by a difference between two stages, which is where the
    "Runge-Kutta" comes from and what the second stage is for::

        Y_bar  = Y + a h + b sqrt(h)
        Y_next = Y + a h + b dW + (b(Y_bar) - b) (dW^2 - h) / (2 sqrt(h))

    where ``dW = W(t + h) - W(t)`` is one Wiener increment, ``N(0, h)``. Expanding
    ``b(Y_bar)`` shows ``(b(Y_bar) - b) / (2 sqrt(h)) = (1/2) b b' + O(sqrt(h))``, so
    the last term is the Milstein correction ``(1/2) b b' (dW^2 - h)`` up to a
    remainder of strong order 1.5. The ``sqrt(h)`` in the supporting value is
    therefore a legitimate part of the scheme - it is a probe distance, not an
    increment - and the defect of the previous version of this file was to use it
    *as* the increment as well, so that nothing was random (DEQ-25).

    **Order, as measured.** Strong order 1.0 and weak order 1.0, the same as the
    Milstein scheme it is a derivative-free form of, and the same as the body of this
    docstring always claimed; the "2" in the name counts stages, not an order, and
    there is no sense in which this scheme is second order. Measured on geometric
    Brownian motion ``dX = X dt + 0.5 X dW``, ``X(0) = 1``, over ``[0, 1]``, against
    the exact solution ``X0 exp((mu - sigma^2/2) t + sigma W(t))`` driven by the same
    Brownian path at every step size (100000 paths; the weak error is the mean of the
    path-coupled difference, whose standard error is in brackets):

    ====== ============ ====================
    h      strong       weak (std. error)
    ====== ============ ====================
    1/8    2.130e-01    1.521e-01 (8e-04)
    1/16   1.109e-01    8.015e-02 (5e-04)
    1/32   5.616e-02    4.139e-02 (2e-04)
    1/64   2.809e-02    2.096e-02 (1e-04)
    1/128  1.397e-02    1.054e-02 (6e-05)
    1/256  6.936e-03    5.271e-03 (3e-05)
    slope  0.99         0.97
    ====== ============ ====================

    In the same run Euler-Maruyama measured strong 0.59 / weak 0.97 and Milstein
    strong 0.97 / weak 0.97, so the harness resolves the orders it is meant to.

    **Systems.** The diffusion interface returns one coefficient per component, so
    the noise is diagonal: component ``i`` has its own Wiener process. The scheme is
    applied with one supporting value per channel, ``Y_bar_j = Y + a h + b_j e_j
    sqrt(h)``, which reproduces exactly the terms ``(1/2) b_j (d b_j / d x_j)`` of
    the diagonal Milstein scheme. What it omits are the cross terms ``b_k (d b_j /
    d x_k) I_(k,j)`` for ``k != j``, which need the Levy areas ``I_(k,j)`` that have
    no exact sampler. So for a system the measured order 1.0 holds when each ``b_j``
    depends only on ``x_j`` (the noise then commutes and the omitted terms are zero);
    if some ``b_j`` depends on another component, the omission costs a local error of
    order ``h`` and the strong order falls to 0.5, which is what Euler-Maruyama gives.

    **Stratonovich.** A Stratonovich equation is integrated as the Ito equation with
    drift ``a + (1/2) b b'`` (the sign is argued in
    :mod:`discrecontinual_equations.solver.stochastic.coefficients`), so the order
    above applies to either calculus.

    References:
    - Kloeden, P. E. and Platen, E. "Numerical Solution of Stochastic Differential
      Equations", Springer, 1992, Section 11.1, equation (11.1.7).
    - Platen, E. "Zur zeitdiskreten Approximation von Itoprozessen", Diss. B,
      Akademie der Wissenschaften der DDR, 1984.
    """

    def __init__(self, solver_config: SRK2Config, wiener: WienerSource | None = None):
        super().__init__(solver_config=solver_config, wiener=wiener)

    @staticmethod
    def _step(
        y: np.ndarray,
        t: float,
        h: float,
        coefficients: ItoCoefficients,
        increments: WienerIncrements,
    ) -> np.ndarray:
        """One step of Kloeden-Platen (11.1.7), one supporting value per channel."""
        root = np.sqrt(h)
        drift = coefficients.drift(y, t)
        diffusion = coefficients.diffusion(y, t)
        delta_w = increments.delta_w

        # Supporting values are evaluated at t + h: a non-autonomous equation is the
        # autonomous one with time as a component of zero diffusion, whose supporting
        # value is t + 1 * h.
        predictor = y + drift * h
        correction = np.empty_like(y)
        for channel in range(len(y)):
            support = predictor.copy()
            support[channel] += diffusion[channel] * root
            supported = coefficients.diffusion(support, t + h)[channel]
            correction[channel] = (
                (supported - diffusion[channel])
                * (delta_w[channel] ** 2 - h)
                / (2.0 * root)
            )
        return y + drift * h + diffusion * delta_w + correction
