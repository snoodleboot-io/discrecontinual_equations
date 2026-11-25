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

Known defects, found while documenting and not yet fixed (no behaviour changes
in this pass):

* The step-size controller and the endless retry at ``min_step_size`` are as
  in the Dormand-Prince 5(4) solver: an integral controller
  ``h S err^(-(1 + beta)/5)`` with no previous error, a max-norm error, and a
  ``RuntimeError("Step size became too small")`` that cannot be reached.

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

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)
from discrecontinual_equations.solver.solver import Solver


class FehlbergSolver(Solver):
    """Runge-Kutta-Fehlberg 4(5) pair on six stages, with adaptive steps.

    ``a`` and ``c`` are Fehlberg's ``beta_kl`` and ``alpha_k``, ``b4`` his
    fourth-order ``c_k`` and ``b5`` his fifth-order ``c_hat_k`` (Table III).
    The solver advances with ``b5``; ``|y5 - y4|`` is the error estimate.

    References:
        Fehlberg (1969), Table III. See ``docs/references.md`` (Fehlberg1969).
    """

    def __init__(self, solver_config: FehlbergConfig):
        """Store the configuration and build the tableau.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.
        """
        super().__init__(solver_config=solver_config)

        # Fehlberg coefficients
        self.a = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 4, 0, 0, 0, 0, 0],
                [3 / 32, 9 / 32, 0, 0, 0, 0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
            ],
        )

        self.b4 = np.array(
            [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0],
        )  # 4th order
        self.b5 = np.array(
            [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
        )  # 5th order

        self.c = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])

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
