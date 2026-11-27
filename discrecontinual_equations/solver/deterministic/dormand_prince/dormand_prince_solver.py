"""The Dormand-Prince 5(4) embedded Runge-Kutta pair, with adaptive steps.

An explicit Runge-Kutta step evaluates the field ``f(t, y)`` at ``s`` stages
``k_i = f(t + c_i h, y + h sum_j a_ij k_j)`` and combines them as
``y + h sum_i b_i k_i``. An *embedded* pair carries two weight rows on the same
stages, here a fifth-order ``b`` and a fourth-order ``b_hat``, so the difference
``y5 - y4`` estimates the local error at no extra cost. The step is accepted when
that estimate is within tolerance, and the next step is scaled from it.

Dormand and Prince (1980) chose their seven-stage 5(4) tableau so that the
fifth-order solution has small error coefficients: it is the one to advance with
("local extrapolation"), and this solver does so. Its last stage is evaluated at
the new point (the "first same as last" property); this solver does not reuse
it, so a step costs seven evaluations, not six.

The node ``c_5`` is ``8/9``, the sum of ``a_5j`` and Hairer and Wanner's
``C5=8.D0/9.D0``. It was ``7/9`` until DEQ-137: the ``a`` and ``b`` rows were
right, so an autonomous field was integrated at fifth order, but a field that
depends on ``t`` saw the fifth stage at the wrong time and the global error fell
to first order (1.3e-2 to 1.6e-3 on ``y' = y cos t`` to ``t = 2`` with steps 0.1
to 0.0125; now 2.9e-9 to 7.3e-14, order 5).

Steps are taken, measured in the RMS norm and sized by the PI controller of
:mod:`~discrecontinual_equations.solver.deterministic.embedded_runge_kutta`; a
step rejected at ``min_step_size`` raises ``StepSizeTooSmallError`` (DEQ-140).

References:
    Dormand, J. R. and Prince, P. J. (1980). A family of embedded Runge-Kutta
    formulae. Journal of Computational and Applied Mathematics 6(1):19-26.
    DOI: 10.1016/0771-050X(80)90013-3. The seven-stage tableau (not read
    directly: checked against Hairer and Wanner's ``CDOPRI`` and by the order
    conditions in exact arithmetic).

    Hairer, E. and Wanner, G. (1996). DOPRI5, Fortran code described in
    Hairer, Norsett and Wanner, Solving Ordinary Differential Equations I,
    2nd ed., Springer 1993. Routine ``CDOPRI`` (coefficients) and ``DOPCOR``
    (step-size control; the defaults ``SAFE = 0.9`` and ``BETA = 0.04``).

    See ``docs/references.md`` (DormandPrince1980, HairerWanner1996).
"""

import numpy as np

from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
    EmbeddedRungeKuttaSolver,
)

# The seven-stage tableau as transcribed here; see the class docstring.
DORMAND_PRINCE_54 = ButcherTableau(
    name="Dormand-Prince 5(4)",
    a=(
        row(""),
        row("1/5"),
        row("3/40 9/40"),
        row("44/45 -56/15 32/9"),
        row("19372/6561 -25360/2187 64448/6561 -212/729"),
        row("9017/3168 -355/33 46732/5247 49/176 -5103/18656"),
        row("35/384 0 500/1113 125/192 -2187/6784 11/84"),
    ),
    c=row("0 1/5 3/10 4/5 8/9 1 1"),
    b=row("35/384 0 500/1113 125/192 -2187/6784 11/84 0"),
    b_hat=row("5179/57600 0 7571/16695 393/640 -92097/339200 187/2100 1/40"),
    order=5,
    embedded_order=4,
)


class DormandPrinceSolver(EmbeddedRungeKuttaSolver):
    """Dormand-Prince 5(4): advance at fifth order, estimate the error at fourth.

    Seven stages per step. ``a`` and ``c`` hold the tableau, ``b5`` the
    fifth-order weights the step advances with and ``b4`` the embedded
    fourth-order ones. Stepping, error norm and the PI controller are those of
    :class:`EmbeddedRungeKuttaSolver`.

    References:
        Dormand and Prince (1980), the seven-stage tableau; Hairer and Wanner
        (1996), ``DOPRI5``. See ``docs/references.md`` (DormandPrince1980,
        HairerWanner1996).

    Complexity:
        Per attempted step, seven field evaluations and ``O(49 n)`` array work;
        ``O(K n)`` memory for ``K`` accepted steps.
    """

    tableau = DORMAND_PRINCE_54

    def __init__(self, solver_config: DormandPrinceConfig) -> None:
        """Store the configuration and build the tableau.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.
        """
        super().__init__(solver_config=solver_config)

    @property
    def b5(self) -> np.ndarray:
        """The fifth-order weights the step advances with: ``b``."""
        return self.b

    @property
    def b4(self) -> np.ndarray:
        """The embedded fourth-order weights: ``b_hat``."""
        return self.b_hat
