"""The Runge-Kutta-Fehlberg 4(5) pair, with adaptive steps.

Fehlberg (1969) chose six stages that carry two formulas: a fourth-order ``y``
to advance and a fifth-order ``y_hat``, with ``y_hat - y`` an estimate of the
leading error of ``y`` (his eqs. (2)-(3)). His "Formula 2" (Table III, nodes
``0, 1/4, 3/8, 12/13, 1, 1/2``) is the RKF45 pair in common use, and every
coefficient here matches it.

Fehlberg minimized the error of the fourth-order formula and advanced with it.
This solver advances with the fifth-order ``y_hat`` instead (local
extrapolation, the usual practice with the Dormand-Prince pairs): the returned
solution is fifth order, and the error estimate is a bound for the formula it
does not use. On ``y' = y cos t`` with fixed steps the global error falls as
``h^5``.

Steps are taken, measured in the RMS norm and sized by the PI controller of
:mod:`~discrecontinual_equations.solver.deterministic.embedded_runge_kutta`; a
step rejected at ``min_step_size`` raises ``StepSizeTooSmallError`` (DEQ-140).

References:
    Fehlberg, E. (1969). Low-order classical Runge-Kutta formulas with
    stepsize control and their application to some heat transfer problems.
    NASA Technical Report R-315, George C. Marshall Space Flight Center.
    Eqs. (2)-(3) and Table III, "Coefficients for RK4(5), Formula 2" (p. 13).
    The code's earlier citation was Fehlberg's NASA TR R-287 (1968), whose
    title names fifth- to eighth-order formulas; the coefficients here are
    those of R-315.

    See ``docs/references.md`` (Fehlberg1969, HairerWanner1996).
"""

import numpy as np

from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
    EmbeddedRungeKuttaSolver,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)

# Table III of Fehlberg (1969), "Coefficients for RK4(5), Formula 2".
FEHLBERG_45 = ButcherTableau(
    name="Runge-Kutta-Fehlberg 4(5)",
    a=(
        row(""),
        row("1/4"),
        row("3/32 9/32"),
        row("1932/2197 -7200/2197 7296/2197"),
        row("439/216 -8 3680/513 -845/4104"),
        row("-8/27 2 -3544/2565 1859/4104 -11/40"),
    ),
    c=row("0 1/4 3/8 12/13 1 1/2"),
    b=row("16/135 0 6656/12825 28561/56430 -9/50 2/55"),
    b_hat=row("25/216 0 1408/2565 2197/4104 -1/5 0"),
    order=5,
    embedded_order=4,
)


class FehlbergSolver(EmbeddedRungeKuttaSolver):
    """Runge-Kutta-Fehlberg 4(5) pair on six stages, with adaptive steps.

    ``a`` and ``c`` are Fehlberg's ``beta_kl`` and ``alpha_k``, ``b4`` his
    fourth-order ``c_k`` and ``b5`` his fifth-order ``c_hat_k`` (Table III).
    The solver advances with ``b5``; ``|y5 - y4|`` is the error estimate.

    References:
        Fehlberg (1969), Table III. See ``docs/references.md`` (Fehlberg1969).
    """

    tableau = FEHLBERG_45

    def __init__(self, solver_config: FehlbergConfig) -> None:
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
