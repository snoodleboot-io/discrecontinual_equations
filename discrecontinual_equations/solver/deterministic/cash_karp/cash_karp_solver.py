"""The Cash-Karp embedded Runge-Kutta pair of orders 4 and 5, with adaptive steps.

Cash and Karp (1990) give a six-stage tableau for a variable-order method
aimed at right-hand sides that vary rapidly. This solver uses one fixed pair
from it: two weight rows on the same six stages, of orders 5 and 4, whose
difference estimates the local error. Steps are accepted and resized as
in the other embedded pairs (see the Dormand-Prince 5(4) solver).

The fifth-order row ``b = (37/378, 0, 250/621, 125/594, 0, 512/1771)`` advances
the solution and the fourth-order row ``b_hat = (2825/27648, 0, 18575/48384,
13525/55296, 277/14336, 1/4)`` is the embedded one; ``|y5 - y4|`` estimates the
error of the fourth-order solution. Advancing with the higher-order row is
local extrapolation, the choice of GSL's ``rkck`` and Boost.Odeint's
``runge_kutta_cash_karp54``, whose weights these are: the estimate is then
pessimistic for the solution returned, which is one order better. Until
DEQ-138 the rows were swapped - the fourth-order row was labelled ``b5`` and
advanced with - so the solver was order 4: on ``y' = y cos t`` to ``t = 2`` the
error at ``h = 0.0125`` was 3.1e-11 and is now 3.7e-13 (order 5).

Steps are taken, measured in the RMS norm and sized by the PI controller of
:mod:`~discrecontinual_equations.solver.deterministic.embedded_runge_kutta`; a
step rejected at ``min_step_size`` raises ``StepSizeTooSmallError`` (DEQ-140).

References:
    Cash, J. R. and Karp, A. H. (1990). A variable order Runge-Kutta method
    for initial value problems with rapidly varying right-hand sides. ACM
    Transactions on Mathematical Software 16(3):201-222.
    DOI: 10.1145/79505.79507. The six-stage tableau (not read directly:
    checked against GSL and Boost.Odeint below and by the order conditions in
    exact arithmetic). The code's earlier citation gave the year as 1978.

    Galassi, M. et al. GNU Scientific Library, ``ode-initval2/rkck.c``: the
    tableau, the fifth-order weights ``c1 = 37/378 ...`` the step advances
    with, and the error row as their difference from the fourth-order ones.

    Ahnert, K. and Mulansky, M. Boost.Odeint,
    ``stepper/runge_kutta_cash_karp54.hpp``: the same tableau, advancing with
    ``b = (37/378, ...)``.

    See ``docs/references.md`` (CashKarp1990, GSL2024, BoostOdeint2024,
    HairerWanner1996).
"""

import numpy as np

from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
    EmbeddedRungeKuttaSolver,
)

# Cash and Karp's six-stage tableau: b is the fifth-order row the step advances
# with, b_hat the embedded fourth-order row.
CASH_KARP_54 = ButcherTableau(
    name="Cash-Karp 5(4)",
    a=(
        row(""),
        row("1/5"),
        row("3/40 9/40"),
        row("3/10 -9/10 6/5"),
        row("-11/54 5/2 -70/27 35/27"),
        row("1631/55296 175/512 575/13824 44275/110592 253/4096"),
    ),
    c=row("0 1/5 3/10 3/5 1 7/8"),
    b=row("37/378 0 250/621 125/594 0 512/1771"),
    b_hat=row("2825/27648 0 18575/48384 13525/55296 277/14336 1/4"),
    order=5,
    embedded_order=4,
)


class CashKarpSolver(EmbeddedRungeKuttaSolver):
    """Cash-Karp 5(4) pair on six stages, with adaptive steps.

    ``a`` and ``c`` hold the tableau, ``b5`` the fifth-order weights the step
    advances with and ``b4`` the embedded fourth-order ones (module docstring).
    Stepping, error norm and the PI controller are those of
    :class:`EmbeddedRungeKuttaSolver`.

    References:
        Cash and Karp (1990), the six-stage tableau. See ``docs/references.md``
        (CashKarp1990).
    """

    tableau = CASH_KARP_54

    def __init__(self, solver_config: CashKarpConfig) -> None:
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
