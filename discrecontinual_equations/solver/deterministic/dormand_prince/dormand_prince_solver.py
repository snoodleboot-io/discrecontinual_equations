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

Known defects, found while documenting and not yet fixed:

* The controller is not the PI ("Lund-stabilized") controller its comment
  names: it scales by ``err^(-(1 + beta)/5)`` with no previous error, where
  Hairer and Wanner's ``DOPRI5`` uses ``err^(-(0.2 - 0.75 beta)) err_old^beta``.
  The error is measured in the max norm, not the RMS norm.
* A step rejected at ``min_step_size`` is retried at ``min_step_size`` without
  end; the ``RuntimeError("Step size became too small")`` cannot be reached.

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

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
    row,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.solver import Solver

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


class DormandPrinceSolver(Solver):
    """Dormand-Prince 5(4): advance at fifth order, estimate the error at fourth.

    Seven stages per step. ``a``, ``c``, ``b5`` and ``b4`` hold the tableau;
    ``b5`` advances the solution and ``|y5 - y4|`` is the error estimate. See the
    module docstring for the step control.

    References:
        Dormand and Prince (1980), the seven-stage tableau; Hairer and Wanner
        (1996), ``DOPRI5``. See ``docs/references.md`` (DormandPrince1980,
        HairerWanner1996).
    """

    def __init__(self, solver_config: DormandPrinceConfig):
        """Store the configuration and build the tableau.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.
        """
        super().__init__(solver_config=solver_config)

        self.a, self.c, self.b5, self.b4 = DORMAND_PRINCE_54.arrays()

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

        Evaluates the seven stages, forms both weight rows and returns the
        advancing solution ``y5``, the componentwise estimate ``|y4 - y5|``
        and the proposed next step from :meth:`_compute_new_step_size`.
        """
        k = np.zeros((7, len(y)))

        # Compute stages
        for i in range(7):
            t_stage = t + self.c[i] * h
            y_stage = y + h * np.dot(self.a[i, : i + 1], k[: i + 1])
            k[i] = np.array(
                equation.derivative.eval(point=y_stage.tolist(), time=t_stage),
            )

        # Compute solutions
        y5 = y + h * np.dot(self.b5, k)  # 5th order solution
        y4 = y + h * np.dot(self.b4, k)  # 4th order solution

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
