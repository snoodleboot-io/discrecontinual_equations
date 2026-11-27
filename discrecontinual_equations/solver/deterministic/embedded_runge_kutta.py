"""The adaptive stepping shared by the embedded Runge-Kutta pairs.

The four adaptive solvers - Cash-Karp 5(4), Fehlberg 4(5), Dormand-Prince 5(4)
and Prince-Dormand 8(7) - differ only in their tableau. Everything around it
lives here: the trial step, the error norm, the acceptance test, the step-size
controller and what happens when the step cannot be made small enough.

**Error norm.** A trial step from ``y`` to ``y1`` yields the estimate
``err = h sum_i (b_i - b_hat_i) k_i``. Each component is scaled by
``sk_i = atol + rtol max(|y_i|, |y1_i|)`` and the step is accepted when the
root-mean-square ratio ``E = sqrt(mean((err_i / sk_i)^2))`` is at most 1, as in
Hairer and Wanner's ``DOPRI5`` (the norm and acceptance rule of its ``DOPCOR``).

**Step-size controller.** A proportional-integral ("PI") controller in the
form ``DOPRI5`` uses, which Hairer and Wanner call Lund stabilization after
Gustafsson's control-theoretic analysis. With ``q`` one more than the order of
the error estimate (5 for the 5(4) and 4(5) pairs, 8 for the 8(7) pair),
``alpha = 1/q - 0.75 beta`` and ``E_old`` the ratio of the last accepted step
(at least ``1e-4``)::

    h_new = h / clip(E^alpha / E_old^beta / S, 1/10, 1/0.2)

that is, ``h_new = h S E^(-alpha) E_old^beta`` with the change in step held to
``[0.2, 10]`` times the old one, then to ``[min_step_size, max_step_size]``.
The integral term ``E^(-alpha)`` steers ``E`` towards ``S^q``; the proportional
term ``E_old^beta`` damps the oscillation a pure integral controller shows when
the step is limited by stability rather than accuracy. ``beta = 0`` recovers
the classical ``h S E^(-1/q)``. After a rejection the step shrinks by
``E^alpha / S`` (at most by 5) with no ``E_old`` term, and the step that
follows a rejection may not grow. The defaults ``S = safety_factor = 0.9`` and
``beta = 0.04`` are ``DOPRI5``'s; for the 8(7) pair the same form is used with
``q = 8`` (Hairer and Wanner's ``DOP853``, a different 8th-order pair, uses
``alpha = 1/8 - 0.2 beta`` with ``beta = 0`` by default; this package keeps
one controller for all four pairs).

Until DEQ-140 the controller was integral only - ``h S E^(-(1 + beta)/q)``,
with no previous error, no limit on the change, and a max norm - although it
was called PI.

**Too small a step.** A step rejected when it is already at ``min_step_size``
(or one so small that ``t + h`` cannot be told from ``t``) raises
:class:`StepSizeTooSmallError` naming ``t``, ``h`` and the error ratio. The
integration is not continued: a step that fails its tolerance at the smallest
size allowed means the solution is singular or too stiff for an explicit
method there, and accepting it would return a value the error estimate has
already disowned. (Until DEQ-140 the step was retried at ``min_step_size``
forever, and the error meant for this case could not be raised.) The last step
is stretched or shortened to land on ``end_time`` when it would otherwise stop
within 1% of it, as ``DOPRI5`` does.

References:
    Hairer, E. and Wanner, G. (1996). DOPRI5, Fortran code, as vendored in
    SciPy v1.11.4, ``scipy/integrate/dop/dopri5.f``. Subroutine ``DOPCOR``:
    the error norm ``ERR = SQRT(ERR/N)`` with ``SK = ATOLI + RTOLI *
    MAX(ABS(Y(I)), ABS(Y1(I)))``; ``FAC11 = ERR**EXPO1``,
    ``EXPO1 = 0.2D0 - BETA*0.75D0``, ``FAC = FAC11/FACOLD**BETA`` ("LUND-
    STABILIZATION"), ``FAC = MAX(FACC2, MIN(FACC1, FAC/SAFE))``,
    ``HNEW = H/FAC``; on acceptance ``FACOLD = MAX(ERR, 1.0D-4)`` and, after a
    rejection, ``HNEW = MIN(ABS(HNEW), ABS(H))``; on rejection
    ``HNEW = H/MIN(FACC1, FAC11/SAFE)``; the defaults ``SAFE = 0.9``,
    ``FAC1 = 0.2``, ``FAC2 = 10``, ``BETA = 0.04``; the last-step rule
    ``X + 1.01 H - XEND``; and the exit "STEP SIZE TOO SMALL" when
    ``0.1 ABS(H) <= ABS(X) UROUND``.

    Hairer, E. and Wanner, G. DOP853, Fortran code, as vendored in SciPy
    v1.11.4, ``scipy/integrate/dop/dop853.f``: ``EXPO1 = 1/8 - BETA*0.2``,
    ``BETA = 0`` by default, the step-change limits ``[0.333, 6]``.

    Gustafsson, K. (1991). Control theoretic techniques for stepsize selection
    in explicit Runge-Kutta methods. ACM Transactions on Mathematical
    Software 17(4):533-554. DOI: 10.1145/210232.210242. Cited by ``DOPRI5``
    for the stabilized controller (via Hairer, Norsett and Wanner, Solving
    Ordinary Differential Equations, Section IV.2); not read here.

    See ``docs/references.md`` (HairerWanner1996, HairerWanner853,
    Gustafsson1991).
"""

from typing import ClassVar, Protocol, cast

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.butcher_tableau import (
    ButcherTableau,
)
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.solver_config import SolverConfig

# DOPRI5's FAC1 and FAC2: a step may shrink to a fifth or grow tenfold.
MIN_FACTOR = 0.2
MAX_FACTOR = 10.0
# DOPRI5's floor on the remembered error ratio, FACOLD = MAX(ERR, 1.0D-4).
ERROR_FLOOR = 1.0e-4
# DOPRI5 ends the integration with the next step when it would stop within 1%
# of the end: (X + 1.01 H - XEND) > 0.
LAST_STEP_STRETCH = 1.01
_ROUNDING = float(np.finfo(float).eps)


class EmbeddedPairSettings(Protocol):
    """The configuration fields an embedded pair reads; every adaptive config has them."""

    start_time: float
    end_time: float
    initial_step_size: float
    absolute_tolerance: float
    relative_tolerance: float
    min_step_size: float
    max_step_size: float
    safety_factor: float
    beta: float


class StepSizeTooSmallError(RuntimeError):
    """A step failed its tolerance at the smallest step allowed.

    Raised instead of retrying the same step forever. ``time`` is where the
    step starts, ``step`` its size and ``error_ratio`` the scaled error norm it
    reached (``None`` when the step was too small to advance ``time`` at all).
    """

    def __init__(self, time: float, step: float, error_ratio: float | None) -> None:
        self.time = time
        self.step = step
        self.error_ratio = error_ratio
        reached = (
            "it cannot advance t in floating point"
            if error_ratio is None
            else f"its scaled error is {error_ratio:.3g} (at most 1 is accepted)"
        )
        message = (
            f"Step size became too small at t = {time!r}: the step h = {step:.3g} "
            f"is at min_step_size or below, and {reached}. The solution may be "
            "singular there, or too stiff for an explicit method; lower "
            "min_step_size or loosen the tolerances to go on."
        )
        super().__init__(message)


class EmbeddedRungeKuttaSolver(Solver):
    """Adaptive integration with an embedded explicit Runge-Kutta pair.

    A subclass names its :class:`ButcherTableau` as ``tableau``; this class
    takes the steps, measures the error in the RMS norm and sizes the next
    step with the PI controller of ``DOPRI5`` (module docstring). ``a``, ``c``,
    ``b`` (the advancing weights) and ``b_hat`` (the embedded ones) are the
    tableau as float arrays.

    References:
        Hairer and Wanner (1996), ``DOPRI5``, subroutine ``DOPCOR``. See
        ``docs/references.md`` (HairerWanner1996).
    """

    tableau: ClassVar[ButcherTableau]

    def __init__(self, solver_config: SolverConfig) -> None:
        """Store the configuration and build the tableau's float arrays.

        Args:
            solver_config: The interval, first step, tolerances, step bounds,
                ``safety_factor`` and ``beta``.

        Raises:
            ValueError: If ``beta >= 4 / (3 q)``: the exponent ``alpha`` would
                not be positive, and a rejected step would not shrink.
        """
        super().__init__(solver_config=solver_config)
        self.a, self.c, self.b, self.b_hat = self.tableau.arrays()
        if not self.controller_exponent > 0:
            q = self.tableau.error_order + 1
            message = (
                f"beta = {self.settings.beta} is too large for {self.tableau.name}: "
                f"the controller exponent 1/{q} - 0.75 beta must be positive, "
                f"so beta < {4 / (3 * q):.4g} (DOPRI5 uses 0.04)"
            )
            raise ValueError(message)

    @property
    def settings(self) -> EmbeddedPairSettings:
        """The configuration, typed as the fields this solver reads."""
        return cast("EmbeddedPairSettings", self.solver_config)

    @property
    def controller_exponent(self) -> float:
        """``alpha = 1/q - 0.75 beta``, ``q`` one more than the estimate's order."""
        q = self.tableau.error_order + 1
        return 1.0 / q - 0.75 * self.settings.beta

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """Integrate from ``start_time`` to ``end_time`` with adaptive steps.

        Starting from ``initial_step_size``, each step is tried and accepted
        when the RMS error ratio is at most 1; the next step comes from the PI
        controller (module docstring). The last step lands on ``end_time``.
        The accepted points are recorded, so their spacing varies.

        Args:
            equation: The equation; its ``derivative.eval(point, time)`` is the
                field ``f(t, y)``.
            initial_values: The state at ``start_time``.

        Returns:
            The curve of accepted points: ``time`` holds ``t_n``, ``results``
            the state ``y_n`` and ``variables`` zeros (no derivative is
            recorded). It is also kept as ``solution``.

        Raises:
            StepSizeTooSmallError: A step was rejected at ``min_step_size`` or
                below, or the step no longer advances ``t``.
        """
        settings = self.settings
        self.solution = self._new_solution(equation)
        zeros = [0.0] * len(initial_values)

        t = settings.start_time
        end = settings.end_time
        y = np.array(initial_values, dtype=float)
        h = settings.initial_step_size
        self.solution.append([t, zeros, list(initial_values)])

        previous_ratio = ERROR_FLOOR
        rejected = False
        while t < end:
            if 0.1 * abs(h) <= abs(t) * _ROUNDING:
                raise StepSizeTooSmallError(t, h, None)
            last = t + LAST_STEP_STRETCH * h >= end
            if last:
                h = end - t

            y_new, error = self._trial_step(equation, t, y, h)
            ratio = self.error_ratio(error, y, y_new)

            if ratio <= 1.0:
                t = end if last else t + h
                y = y_new
                self.solution.append([t, zeros, y.tolist()])
                h_next = self.accepted_step_size(h, ratio, previous_ratio)
                if rejected:
                    h_next = min(h_next, h)
                previous_ratio = max(ratio, ERROR_FLOOR)
                rejected = False
                h = h_next
            else:
                if h <= settings.min_step_size:
                    raise StepSizeTooSmallError(t, h, ratio)
                h = max(self.rejected_step_size(h, ratio), settings.min_step_size)
                rejected = True
        return self.solution

    def _trial_step(
        self,
        equation: DifferentialEquation,
        t: float,
        y: np.ndarray,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One trial step: the advancing solution and the error estimate.

        Evaluates the ``s`` stages ``k_i = f(t + c_i h, y + h sum_j a_ij k_j)``
        and returns ``y + h b.k`` with ``h (b - b_hat).k``.
        """
        stages = self.tableau.stages
        k = np.zeros((stages, len(y)))
        for i in range(stages):
            stage = y + h * (self.a[i, :i] @ k[:i])
            k[i] = equation.derivative.eval(
                point=stage.tolist(), time=t + self.c[i] * h
            )
        return y + h * (self.b @ k), h * ((self.b - self.b_hat) @ k)

    def error_ratio(
        self,
        error: np.ndarray,
        y: np.ndarray,
        y_new: np.ndarray,
    ) -> float:
        """``sqrt(mean((err_i / sk_i)^2))``, ``sk_i = atol + rtol max(|y_i|, |y1_i|)``.

        A NaN or infinite estimate comes back as infinity, so the step is
        rejected.
        """
        settings = self.settings
        scale = settings.absolute_tolerance + settings.relative_tolerance * np.maximum(
            np.abs(y),
            np.abs(y_new),
        )
        ratio = float(np.sqrt(np.mean((error / scale) ** 2)))
        return ratio if np.isfinite(ratio) else float("inf")

    def accepted_step_size(
        self,
        h: float,
        ratio: float,
        previous_ratio: float,
    ) -> float:
        """The PI step after an accepted step: ``h S E^(-alpha) E_old^beta``, limited.

        The change is held to ``[MIN_FACTOR, MAX_FACTOR]`` and the step to
        ``[min_step_size, max_step_size]``.
        """
        settings = self.settings
        factor = ratio**self.controller_exponent / previous_ratio**settings.beta
        factor = min(
            1.0 / MIN_FACTOR, max(1.0 / MAX_FACTOR, factor / settings.safety_factor)
        )
        return min(settings.max_step_size, max(settings.min_step_size, h / factor))

    def rejected_step_size(self, h: float, ratio: float) -> float:
        """The step after a rejection: ``h S E^(-alpha)``, shrinking at most fivefold."""
        if not np.isfinite(ratio):
            return h * MIN_FACTOR
        shrink = ratio**self.controller_exponent / self.settings.safety_factor
        return h / min(1.0 / MIN_FACTOR, shrink)
