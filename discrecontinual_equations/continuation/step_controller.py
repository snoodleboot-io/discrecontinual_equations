"""Arclength step-size control."""

from abc import ABC, abstractmethod

_STEP_HALVING = 0.5


class StepController(ABC):
    """Decide the next arclength step from the previous step's behaviour."""

    @abstractmethod
    def on_success(self, step: float, iterations: int) -> float:
        """Return the next step after a converged corrector."""
        raise NotImplementedError

    @abstractmethod
    def on_failure(self, step: float) -> float:
        """Return a reduced step after a failed corrector."""
        raise NotImplementedError

    @abstractmethod
    def is_exhausted(self, step: float) -> bool:
        """Return whether the step has shrunk below the usable minimum."""
        raise NotImplementedError


class AdaptiveStepController(StepController):
    """Grow or shrink the step to keep the corrector near a target iteration count."""

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
