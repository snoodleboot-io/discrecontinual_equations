"""Arclength step-size control.

A predictor-corrector method needs a step small enough that the corrector
converges from the predicted point, and no smaller than it must be. The rule here
is the convergence-dependent one Kuznetsov (1998, Section 10.2.1, "Step-size
control") recommends as simple and reliable: shrink the step and retry when the
corrector fails, grow it when the corrector converged in only a few iterations, and
keep it when convergence took a moderate number. It is the bare idea behind the
error-model strategy of Den Heijer and Rheinboldt, which steers the step towards a
desired number of corrector iterations ``k~`` (Allgower and Georg (1990), Section
6.2, Algorithm (6.2.1)); the factor here is fixed instead of being derived from a
contraction model.

:class:`AdaptiveStepController` with target ``k``, slack ``s``, growth ``g`` and
shrink ``c``:

* converged in fewer than ``k`` iterations: ``h <- min(g h, h_max)``;
* converged in more than ``k + s``: ``h <- max(c h, h_min)``;
* otherwise ``h`` is kept;
* corrector failed: ``h <- h / 2``, and continuation stops once ``h < h_min``.

Limits: the iteration count is a coarse signal. A step can converge in few
iterations and still jump to a nearby branch (the controller cannot see that), and
near a fold of very high curvature the step can halve down to ``h_min`` and stop.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.1
  ("Step-size control").
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Chapter 6, Algorithm (6.2.1).
"""

from abc import ABC, abstractmethod

_STEP_HALVING = 0.5


class StepController(ABC):
    """Decide the next arclength step from the previous step's behaviour."""

    @abstractmethod
    def on_success(self, step: float, iterations: int) -> float:
        """Return the next step after a converged corrector.

        Args:
            step: The step just taken.
            iterations: Newton iterations the corrector needed.

        Returns:
            The step to try next.
        """
        raise NotImplementedError

    @abstractmethod
    def on_failure(self, step: float) -> float:
        """Return a reduced step after a failed corrector.

        Args:
            step: The step that failed.

        Returns:
            The smaller step to retry with.
        """
        raise NotImplementedError

    @abstractmethod
    def is_exhausted(self, step: float) -> bool:
        """Return whether the step has shrunk below the usable minimum.

        Args:
            step: The step about to be tried.

        Returns:
            ``True`` when continuation should stop.
        """
        raise NotImplementedError


class AdaptiveStepController(StepController):
    """Grow or shrink the step to keep the corrector near a target iteration count.

    Args:
        target: Iteration count below which the step grows.
        growth: Factor applied on a fast convergence (above one).
        shrink: Factor applied on a slow convergence (below one).
        bounds: ``(minimum, maximum)`` step; a successful step is clamped to it, and
            a failed one that halves below the minimum ends continuation.
        slack: Iterations above ``target`` tolerated before the step shrinks.

    References:
    - Kuznetsov (1998), Section 10.2.1 ("Step-size control").
    - Allgower and Georg (1990), Algorithm (6.2.1).
    """

    __slots__ = [
        "_growth",
        "_maximum",
        "_minimum",
        "_shrink",
        "_slack",
        "_target",
    ]

    def __init__(
        self,
        target: int,
        growth: float,
        shrink: float,
        bounds: tuple[float, float],
        slack: int,
    ) -> None:
        self._target = target
        self._growth = growth
        self._shrink = shrink
        self._minimum, self._maximum = bounds
        self._slack = slack

    def on_success(self, step: float, iterations: int) -> float:
        if iterations < self._target:
            return min(step * self._growth, self._maximum)
        if iterations > self._target + self._slack:
            return max(step * self._shrink, self._minimum)
        return step

    def on_failure(self, step: float) -> float:
        return step * _STEP_HALVING

    def is_exhausted(self, step: float) -> bool:
        return step < self._minimum
