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

Known defects, found while documenting and not yet fixed:

* The step-size controller and the endless retry at ``min_step_size`` are as
  in the Dormand-Prince 5(4) solver: an integral controller
  ``h S err^(-(1 + beta)/5)`` with no previous error, a max-norm error, and a
  ``RuntimeError("Step size became too small")`` that cannot be reached.

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

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.solver import Solver

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


class CashKarpSolver(Solver):
    """Cash-Karp 5(4) pair on six stages, with adaptive steps.

    ``a`` and ``c`` hold the tableau, ``b5`` the fifth-order weights the step
    advances with and ``b4`` the embedded fourth-order ones; ``|y5 - y4|`` is
    the error estimate (module docstring).

    References:
        Cash and Karp (1990), the six-stage tableau. See ``docs/references.md``
        (CashKarp1990).
    """

    def __init__(self, solver_config: CashKarpConfig):
        """Store the configuration and build the tableau.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.
        """
        super().__init__(solver_config=solver_config)

        self.a, self.c, self.b5, self.b4 = CASH_KARP_54.arrays()

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """Integrate from ``start_time`` to ``end_time`` with adaptive steps.

        Starting from ``initial_step_size``, each step is tried, accepted when
        the scaled error estimate is at most 1, and the next step size is set
        from that estimate; a rejected step is retried with the smaller size.
        The last step is shortened to land on ``end_time``. The accepted
        points are recorded, so their spacing varies.

        Args:
            equation: The equation; its ``derivative.eval(point, time)`` is the
                field ``f(t, y)``.
            initial_values: The state at ``start_time``.

        Returns:
            The curve of accepted points: ``time`` holds ``t_n``, ``results``
            the state ``y_n`` and ``variables`` zeros (this solver records no
            derivative). It is also kept as ``solution``.

        Raises:
            RuntimeError: Declared for a step below ``min_step_size``, but not
                reached: a step rejected at ``min_step_size`` is retried there
                indefinitely (see the module docstring).
        """
        self.solution = self._new_solution(equation)

        # Initialize
        t = self.solver_config.start_time
        y = np.array(initial_values)
        h = self.solver_config.initial_step_size

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), initial_values])

        # Adaptive integration loop
        while t < self.solver_config.end_time:
            if t + h > self.solver_config.end_time:
                h = self.solver_config.end_time - t

            # Perform one step
            y_new, error, h_new = self._step(equation, t, y, h)

            # Check if step is accepted
            if self._accept_step(error, y, y_new, h):
                # Accept step
                t += h
                y = y_new
                self.solution.append([t, [0] * len(initial_values), y.tolist()])
                h = h_new
            else:
                # Reject step, try smaller step
                h = max(h_new, self.solver_config.min_step_size)
                if h < self.solver_config.min_step_size:
                    raise RuntimeError("Step size became too small")
        return self.solution

    def _step(self, equation: DifferentialEquation, t: float, y: np.ndarray, h: float):
        """Take one trial step of size ``h`` from ``(t, y)``.

        Evaluates the six stages, forms both weight rows and returns the
        advancing solution ``y5``, the componentwise estimate ``|y4 - y5|``
        and the proposed next step from :meth:`_compute_new_step_size`.
        """
        k = np.zeros((6, len(y)))

        # Compute stages
        for i in range(6):
            t_stage = t + self.c[i] * h
            y_stage = y + h * np.dot(self.a[i, : i + 1], k[: i + 1])
            k[i] = np.array(
                equation.derivative.eval(point=y_stage.tolist(), time=t_stage),
            )

        # Compute solutions
        y4 = y + h * np.dot(self.b4, k)  # 4th order solution
        y5 = y + h * np.dot(self.b5, k)  # 5th order solution

        # Error estimate
        error = np.abs(y5 - y4)

        # Step size control
        h_new = self._compute_new_step_size(error, y, h)

        return y5, error, h_new

    def _accept_step(
        self,
        error: np.ndarray,
        y: np.ndarray,
        y_new: np.ndarray,
        h: float,
    ) -> bool:
        """Accept the step when ``max_i err_i / (atol + rtol max(|y_i|, |y_new_i|))``
        is at most 1 (a max norm; Hairer and Wanner's ``DOPRI5`` uses the RMS
        norm of the same ratios)."""
        # Compute scaled error
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance
            * np.maximum(np.abs(y), np.abs(y_new))
        )
        scaled_error = np.max(error / tol)

        return scaled_error <= 1.0

    def _compute_new_step_size(
        self,
        error: np.ndarray,
        y: np.ndarray,
        h: float,
    ) -> float:
        """Propose the next step: ``h S r^(-1/5) r^(-beta/5)``, clipped to the bounds.

        ``r`` is the max-norm error ratio, scaled by ``atol + rtol |y|`` (the old
        state only), and ``S`` is ``safety_factor``; ``r = 0`` gives
        ``max_step_size``. This is an integral controller with exponent
        ``(1 + beta)/5``: although the name says PI, no earlier error enters,
        unlike the Lund-stabilized ``err^(-(0.2 - 0.75 beta)) err_old^beta``
        of Hairer and Wanner's ``DOPRI5`` for its 5(4) pair.
        """
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance * np.abs(y)
        )
        scaled_error = np.max(error / tol)

        if scaled_error == 0:
            return self.solver_config.max_step_size

        # PI controller
        h_new = (
            h
            * self.solver_config.safety_factor
            * (1 / scaled_error) ** (1 / 5)
            * (1 / scaled_error) ** (self.solver_config.beta / 5)
        )

        # Limit step size
        h_new = np.clip(
            h_new,
            self.solver_config.min_step_size,
            self.solver_config.max_step_size,
        )

        return h_new
