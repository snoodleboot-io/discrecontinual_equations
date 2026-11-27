"""``HomoclinicShooting.locate`` refuses a bracket it cannot bisect (DEQ-135).

``gap`` is NaN wherever the unstable manifold does not return to the saddle,
and NaN compares false with everything. The old sign test,
``low_gap * middle_gap <= 0``, was then always false. With NaN at the upper
end, every step moved ``lower`` up and the result came back as the upper end.
With NaN at the lower end, a root in the lower half was lost.

On the damped well ``y' = x - x^2 + mu y`` the gap is finite only near the
homoclinic value ``mu = 0``: measured NaN from ``|mu| = 0.1`` outwards, and
+0.257 and -0.267 at ``mu = -0.05`` and ``0.05``.
"""

import math
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)
from tests.continuation.fields import Alpha, DampedWellField, State


def _shooter() -> HomoclinicShooting:
    field = DampedWellField(
        variables=[State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State()],
        time=None,
    )
    departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
    return HomoclinicShooting(field, 0, departure)


class _Scripted(HomoclinicShooting):
    """A shooter whose gap is a given function: ``root - p`` with NaN holes."""

    def __init__(self, root: float, holes: tuple[tuple[float, float], ...]) -> None:
        self.root, self.holes, self.calls = root, holes, []
        # locate reads its jump tolerance from the settings (DEQ-174)
        self._settings = ReturnSettings()

    def gap(self, parameter: float) -> float:
        self.calls.append(parameter)
        if any(low < parameter < high for low, high in self.holes):
            return float("nan")
        return self.root - parameter


class TestBracketEndsAreChecked(TestCase):
    """On the damped well, whose gap is NaN away from the homoclinic value."""

    def test_a_nan_at_the_upper_end_is_refused(self):
        # Before: the result came back as the upper end, 0.3.
        with pytest.raises(ValueError, match="upper end"):
            _shooter().locate(-0.05, 0.3)

    def test_a_nan_at_the_lower_end_is_refused(self):
        # Before: the root at 0 was lost and the result was the upper end.
        with pytest.raises(ValueError, match="lower end"):
            _shooter().locate(-0.3, 0.05)

    def test_a_bracket_without_a_sign_change_is_refused(self):
        with pytest.raises(ValueError, match="same sign"):
            _shooter().locate(0.01, 0.05)


class TestBisection(TestCase):
    """On a scripted gap ``0.3 - p``, so the root and the NaN regions are known."""

    def test_the_root_is_found_in_either_orientation_of_the_sign(self):
        assert _Scripted(0.3, ()).locate(0.0, 1.0) == pytest.approx(0.3, abs=1e-9)
        # The reported case: a root at 0.3 that came back as 1.0.
        assert _Scripted(0.3, ()).locate(1.0, 0.0) == pytest.approx(0.3, abs=1e-9)

    def test_a_nan_midpoint_is_refused_not_followed(self):
        shooter = _Scripted(0.3, ((0.45, 0.55),))
        with pytest.raises(ValueError, match=r"not finite at 0\.5, inside"):
            shooter.locate(0.0, 1.0)

    def test_a_nan_midpoint_later_in_the_search_is_refused(self):
        # The first midpoints 0.5 and 0.25 are finite; 0.375 is in the hole.
        shooter = _Scripted(0.3, ((0.37, 0.38),))
        with pytest.raises(ValueError, match="inside the bracket"):
            shooter.locate(0.0, 1.0)
        assert shooter.calls[-1] == pytest.approx(0.375)

    def test_an_end_with_a_zero_gap_is_the_root(self):
        assert _Scripted(0.3, ()).locate(0.3, 1.0) == 0.3
        assert _Scripted(0.3, ()).locate(0.0, 0.3) == 0.3


class _Wave(HomoclinicShooting):
    """A traced gap ``cos(pi (p - c))``, roots at ``c + 1/2 + k``, NaN past ``edge``.

    ``c`` is the control parameter, so the roots move with it as a curve would.
    """

    def __init__(self, edge: float) -> None:
        self.edge = edge
        self._settings = ReturnSettings()
        self._function = SimpleNamespace(
            parameters=[SimpleNamespace(value=0.0), SimpleNamespace(value=0.0)],
        )

    def gap(self, parameter: float) -> float:
        if parameter > self.edge:
            return float("nan")
        control = self._function.parameters[1].value
        return math.cos(math.pi * (parameter - control))


class TestTraceScansItsWindow(TestCase):
    """``trace`` hands ``locate`` a sampled sign change, never a bare window."""

    def test_a_nan_end_of_the_window_is_stepped_over(self):
        # The window [0, 1.2] ends past the NaN edge at 1.1; the root at 0.5
        # is still found, as the melnikov trace's root is beside its NaN end.
        curve = _Wave(1.1).trace(1, [0.0], 0.6, 0.6)
        assert curve[0][1] == pytest.approx(0.5, abs=1e-8)

    def test_the_branch_followed_is_the_nearest_root(self):
        # Roots at 0.5 and 1.5 both lie in [0.4, 2.4]; the one nearest the
        # previous point, 1.4, is the branch being followed.
        curve = _Wave(10.0).trace(1, [0.0, 0.1, 0.2], 1.4, 1.0)
        located = [point for _, point in curve]
        assert located == pytest.approx([1.5, 1.6, 1.7], abs=1e-8)

    def test_a_window_without_a_finite_sign_change_is_refused(self):
        with pytest.raises(ValueError, match="no sign change of finite"):
            _Wave(1.1).trace(1, [0.0], 1.3, 0.1)

    def test_a_nan_sample_never_makes_a_sign_change(self):
        # Gaps are positive up to the NaN edge at 0.3: no finite sign change,
        # though a NaN beside a positive gap differs from it in sign test.
        with pytest.raises(ValueError, match="no sign change of finite"):
            _Wave(0.3).trace(1, [0.0], 0.2, 0.2)

    def test_the_damped_well_trace_is_refused_beyond_the_return(self):
        # Every sample of [0.2, 0.4] is past the return: NaN throughout.
        with pytest.raises(ValueError, match="no sign change of finite"):
            _shooter().trace(0, [0.0], 0.3, 0.1)
