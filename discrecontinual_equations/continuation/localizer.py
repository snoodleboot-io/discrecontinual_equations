"""Refine a bracketed sign change into a precise bifurcation location.

The localizer owns the *geometry* - mapping a fraction along the segment to a
corrected point and a test value - and delegates the *root-finding* on that
fraction to an injected :class:`~.root_finder.ScalarRootFinder`.

When a detector's test function ``psi`` changes sign between accepted points
``y_k`` and ``y_(k+1)`` (Kuznetsov (1998), eq. (10.35)), :class:`RefiningLocalizer`
parametrises the chord ``y(sigma) = y_k + sigma (y_(k+1) - y_k)``,
``sigma in [0, 1]``, corrects each trial point back onto the curve with the
continuation corrector (on the hyperplane through the trial point normal to
``y_k``'s tangent, for the default pseudo-arclength constraint), and root-finds
``psi`` of the corrected point in ``sigma``. This is the one-dimensional location
along the curve of Kuznetsov (1998, Section 10.2.2) and Allgower and Georg (1990,
Section 9.2), rather than Newton on the augmented system
``F(y) = 0, psi(y) = 0`` (Kuznetsov's (10.36)), so it works for test functions
that are not differentiable, or whose augmented Jacobian is singular at the zero.

Limits: the located point is as accurate as the corrector tolerance and the
root-finder's tolerances, not more. If the corrector fails at a trial point, the
test value at ``y_k`` is used instead, which keeps the bracket's sign but can make
bisection stop at the failure rather than the zero. Two zeros of ``psi`` within one
step cancel and are not seen.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.2,
  eqs. (10.35)-(10.36).
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Section 9.2.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.continuation_state import ContinuationState
from discrecontinual_equations.continuation.corrector import Corrector
from discrecontinual_equations.continuation.detection import (
    BifurcationDetector,
    DetectionContext,
)
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.root_finder import ScalarRootFinder
from discrecontinual_equations.continuation.tangent import TangentComputer


class LocatedBifurcation:
    """A refined bifurcation: its state, its kind, and any frequency."""

    __slots__ = ["_frequency", "_kind", "_state"]

    def __init__(
        self,
        state: ContinuationState,
        kind: str,
        frequency: float | None,
    ) -> None:
        self._state = state
        self._kind = kind
        self._frequency = frequency

    @property
    def state(self) -> ContinuationState:
        """Located point on the branch."""
        return self._state

    @property
    def kind(self) -> str:
        """Bifurcation label."""
        return self._kind

    @property
    def frequency(self) -> float | None:
        """Oscillation frequency for a Hopf point, else ``None``."""
        return self._frequency


class Localizer(ABC):
    """Pin a bifurcation between two bracketing states for a given detector."""

    @abstractmethod
    def localize(
        self,
        lower: ContinuationState,
        upper: ContinuationState,
        detector: BifurcationDetector,
    ) -> LocatedBifurcation | None:
        """Return the refined bifurcation, or ``None`` if it cannot be pinned.

        Args:
            lower: The accepted point before the sign change.
            upper: The accepted point after it.
            detector: The detector whose test function changed sign.

        Returns:
            The located bifurcation, or ``None`` when the two points coincide or
            the corrector fails at the located fraction.
        """
        raise NotImplementedError


class RefiningLocalizer(Localizer):
    """Localize by re-correcting onto the branch and root-finding the test value.

    Args:
        corrector: Corrector used to project chord points onto the branch.
        tangent_computer: Recomputes the tangent at each corrected point.
        jacobian: Jacobian source handed to the detector.
        root_finder: Scalar root-finder on the chord fraction.

    References:
    - Kuznetsov (1998), Section 10.2.2.
    - Allgower and Georg (1990), Section 9.2.
    """

    __slots__ = ["_corrector", "_jacobian", "_root_finder", "_tangent_computer"]

    def __init__(
        self,
        corrector: Corrector,
        tangent_computer: TangentComputer,
        jacobian: JacobianProvider,
        root_finder: ScalarRootFinder,
    ) -> None:
        self._corrector = corrector
        self._tangent_computer = tangent_computer
        self._jacobian = jacobian
        self._root_finder = root_finder

    def localize(
        self,
        lower: ContinuationState,
        upper: ContinuationState,
        detector: BifurcationDetector,
    ) -> LocatedBifurcation | None:
        direction_vector = np.concatenate(
            [upper.state - lower.state, [upper.parameter - lower.parameter]],
        )
        distance = float(np.linalg.norm(direction_vector))
        if distance == 0.0:
            return None
        unit = direction_vector / distance
        fallback = detector.test_value(DetectionContext(lower, self._jacobian))

        def test_at(fraction: float) -> float:
            state = self._corrected_state(lower, unit, distance, fraction)
            if state is None:
                return fallback
            return detector.test_value(DetectionContext(state, self._jacobian))

        fraction = self._root_finder.locate(test_at, 0.0, 1.0)
        if fraction is None:
            return None
        found = self._corrected_state(lower, unit, distance, fraction)
        if found is None:
            return None
        context = DetectionContext(found, self._jacobian)
        return LocatedBifurcation(
            found,
            detector.refine_kind(context),
            detector.characteristic_frequency(context),
        )

    def _corrected_state(
        self,
        lower: ContinuationState,
        unit: np.ndarray,
        distance: float,
        fraction: float,
    ) -> ContinuationState | None:
        offset = fraction * distance
        predicted_state = lower.state + offset * unit[:-1]
        predicted_parameter = lower.parameter + offset * float(unit[-1])
        result = self._corrector.correct(
            predicted_state,
            predicted_parameter,
            lower.tangent,
        )
        if not result.converged:
            return None
        tangent = self._tangent_computer.compute(
            result.state,
            result.parameter,
            lower.tangent,
        )
        return ContinuationState(
            result.state,
            result.parameter,
            tangent,
            lower.arclength,
        )
