"""Locating a stochastic bifurcation in the parameter, not just observing it.

:mod:`.stochastic` and :mod:`.fokker_planck` answer what the system does at one
parameter value. A bifurcation is a statement about a *parameter*, and the
deterministic side of this library has long been able to say where one is: a detector
in :mod:`.detection` supplies a scalar test function whose sign change brackets the
event, and :class:`~.localizer.RefiningLocalizer` refines the bracket with an injected
:class:`~.root_finder.ScalarRootFinder`. This module gives the two stochastic notions
the same shape. Each threshold is a test function of the parameter plus a root finder;
what differs between them is only what the test measures.

* **Phenomenological.** The test is a probe of the stationary density's shape. For the
  planar systems of interest the question is whether the peak still sits at a
  reference state or has moved off it onto a ring, and the sharpest form of that
  question is the sign of the second radial derivative of ``log p`` at the reference
  state. Since symmetry kills the first derivative there, a one-cell difference of
  ``log p`` carries that sign, and it is a smooth function of the parameter - unlike
  the location of the discrete argmax, which moves in grid-sized jumps and would stall
  a root finder on a staircase.
* **Dynamical.** The test is the top Lyapunov exponent itself, which already crosses
  zero at the threshold by definition.

The dynamical test needs one thing the deterministic ones do not: **common random
numbers**. The exponent is a Monte-Carlo estimate, and if each evaluation drew fresh
noise the test function would be a different random function at every parameter value,
with no reproducible sign change to bracket - root finding would have to be replaced
by stochastic approximation. Reusing a fixed seed set makes the estimate a
deterministic, smooth function of the parameter, so bisection converges and the located
threshold is reproducible. What this does *not* do is remove the error: for the linear
multiplicative system the estimate is exactly ``lambda + s W(T) / T``, so every
evaluation shares one offset of order ``s / sqrt(T)``, and the located threshold is
displaced by it. That makes *differences* of located thresholds - between the two noise
conventions, or along a curve - far more accurate than their absolute level, and it is
why :class:`MeanTopExponent` averages over several seeds: the offset shrinks like
``s / sqrt(paths * T)``, and nothing else about the answer changes.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    GridDensity,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    matrix_noise_lyapunov,
)
from discrecontinual_equations.continuation.root_finder import ScalarRootFinder

_DEFAULT_SEEDS = (0,)


class DensityProbe(ABC):
    """A scalar whose sign change marks a change in the shape of a density."""

    @abstractmethod
    def value(self, density: GridDensity) -> float:
        """Evaluate the probe on a solved density."""
        raise NotImplementedError


class RadialCrater(DensityProbe):
    """``log p`` one cell out from a reference state, minus ``log p`` at it.

    Negative while the density still peaks at the reference state, positive once the
    peak has split off it, and zero exactly at the phenomenological threshold. Working
    in ``log p`` rather than ``p`` matters: the densities here are exponentials of a
    potential, so the logarithm is the smooth object and its curvature is the thing the
    bifurcation changes the sign of. The probe inherits an ``O(h^2)`` error from the
    one-cell difference, which is the accuracy of any threshold located with it.

    ``offset`` moves the reference cell that many cells out from the one nearest
    the centre before the difference is taken. It exists for a grid that straddles
    the centre rather than landing a cell on it - the only grid a noise that
    vanishes at the centre allows, since a cell there would make the diffusion
    singular and the solve refuse. On such a grid the nearest cell is a tie
    broken by rounding, and when it falls on the lower side its neighbour sits at
    the same radius, so the difference says nothing about the curvature. There is
    a second reason to step out: a finite-volume solve overshoots the few cells
    around a singular centre by a factor fixed in cells rather than in length,
    so a probe within them carries a bias no refinement removes, while a probe
    several cells out trades it for the ``O(h^2)`` one above, which refinement
    does remove. The default keeps the cell nearest the centre, as before.
    """

    __slots__ = ["_axis", "_centre", "_offset"]

    def __init__(
        self,
        centre: Sequence[float],
        axis: int = 0,
        offset: int = 0,
    ) -> None:
        self._centre = list(centre)
        self._axis = axis
        self._offset = offset

    def value(self, density: GridDensity) -> float:
        """Difference of ``log p`` between the neighbour cell and the reference cell."""
        index = list(density.grid.nearest(self._centre))
        index[self._axis] += self._offset
        here = density.at(index)
        index[self._axis] += 1
        there = density.at(index)
        if here <= 0.0 or there <= 0.0:
            message = (
                "the density is not positive at the probe; the grid is too coarse "
                "or the reference state is outside its support"
            )
            raise ValueError(message)
        return math.log(there) - math.log(here)


class ParameterThreshold(ABC):
    """Locate the parameter at which a stochastic diagnostic crosses zero."""

    __slots__ = ["_root_finder"]

    def __init__(self, root_finder: ScalarRootFinder) -> None:
        self._root_finder = root_finder

    @abstractmethod
    def test_value(self, parameter: float) -> float:
        """Scalar whose sign change in the parameter brackets the threshold."""
        raise NotImplementedError

    def locate(self, low: float, high: float) -> float | None:
        """Refine the threshold in ``[low, high]``, or ``None`` if it is not in it.

        The bracket is checked before the root finder runs. A bracketing method handed
        an interval with no sign change still returns a number, and for a test function
        that costs a sparse solve or a simulated path that number would be an
        expensively obtained fiction; refusing is the only honest answer. Evaluations
        are memoised because the bracket check and the root finder ask about the same
        endpoints.
        """
        cache: dict[float, float] = {}

        def test(parameter: float) -> float:
            if parameter not in cache:
                cache[parameter] = self.test_value(parameter)
            return cache[parameter]

        if test(low) * test(high) >= 0.0:
            return None
        return self._root_finder.locate(test, low, high)


class PhenomenologicalThreshold(ParameterThreshold):
    """Where the stationary density changes shape: the P-bifurcation parameter."""

    __slots__ = ["_density_at", "_probe"]

    def __init__(
        self,
        density_at: Callable[[float], StationaryFokkerPlanck],
        probe: DensityProbe,
        root_finder: ScalarRootFinder,
    ) -> None:
        super().__init__(root_finder)
        self._density_at = density_at
        self._probe = probe

    def test_value(self, parameter: float) -> float:
        """Solve the stationary equation at this parameter and probe the shape."""
        return self._probe.value(self._density_at(parameter).solve())


class MeanTopExponent:
    """Top Lyapunov exponent of a parameter family, averaged over a fixed seed set.

    The seeds are fixed and shared across parameter values on purpose; see this
    module's docstring for why that is what makes the exponent usable as a test
    function at all, and for what it does and does not buy.
    """

    __slots__ = ["_family", "_seeds", "_settings", "_start"]

    def __init__(
        self,
        family: Callable[[float], MatrixNoiseSystem],
        start: np.ndarray,
        settings: StochasticLyapunovSettings | None = None,
        seeds: Sequence[int] = _DEFAULT_SEEDS,
    ) -> None:
        if len(seeds) == 0:
            message = "at least one seed is needed to estimate an exponent"
            raise ValueError(message)
        self._family = family
        self._start = start
        self._settings = (
            settings if settings is not None else StochasticLyapunovSettings()
        )
        self._seeds = list(seeds)

    def at(self, parameter: float) -> float:
        """Mean top exponent of the system at this parameter value."""
        system = self._family(parameter)
        total = 0.0
        for seed in self._seeds:
            control = StochasticLyapunovSettings(
                dt=self._settings.dt,
                horizon=self._settings.horizon,
                transient=self._settings.transient,
                seed=seed,
            )
            total += matrix_noise_lyapunov(
                system,
                self._start,
                settings=control,
            ).top
        return total / len(self._seeds)


class DynamicalThreshold(ParameterThreshold):
    """Where the top Lyapunov exponent crosses zero: the D-bifurcation parameter."""

    __slots__ = ["_exponent_at"]

    def __init__(
        self,
        exponent_at: Callable[[float], float],
        root_finder: ScalarRootFinder,
    ) -> None:
        super().__init__(root_finder)
        self._exponent_at = exponent_at

    def test_value(self, parameter: float) -> float:
        """The top exponent, whose own zero is the threshold."""
        return self._exponent_at(parameter)


class ThresholdPoint:
    """One located threshold: the second parameter, and where the threshold sat."""

    __slots__ = ["_control", "_parameter"]

    def __init__(self, control: float, parameter: float) -> None:
        self._control = control
        self._parameter = parameter

    @property
    def control(self) -> float:
        """Value of the parameter that was held while the threshold was located."""
        return self._control

    @property
    def parameter(self) -> float:
        """Located threshold."""
        return self._parameter


class ThresholdCurve:
    """Follow a located threshold as a second parameter varies.

    A threshold in one parameter is a point; in two it is a curve, exactly as a fold
    or a Hopf point becomes a fold or Hopf curve. The continuation here is the simplest
    one that deserves the name: the previously located threshold predicts the next, and
    the next is corrected inside a window around it, falling back to the original
    bracket if the window does not contain a sign change. Predicting from the previous
    point is not just an economy - for a test function that costs a simulated path, a
    narrow window is the difference between a traceable curve and an unaffordable one.
    Tracing stops where even the full bracket fails, because the curve has then left
    the range the caller said to look in, and inventing further points would be worse
    than returning a short curve.
    """

    __slots__ = ["_bracket", "_threshold_at", "_window"]

    def __init__(
        self,
        threshold_at: Callable[[float], ParameterThreshold],
        bracket: tuple[float, float],
        window: float,
    ) -> None:
        self._threshold_at = threshold_at
        self._bracket = bracket
        self._window = window

    def trace(self, controls: Sequence[float]) -> list[ThresholdPoint]:
        """Locate the threshold at each control value, predicting from the last."""
        points: list[ThresholdPoint] = []
        previous: float | None = None
        for control in controls:
            threshold = self._threshold_at(control)
            located = None
            if previous is not None:
                located = threshold.locate(
                    previous - self._window,
                    previous + self._window,
                )
            if located is None:
                located = threshold.locate(*self._bracket)
            if located is None:
                break
            points.append(ThresholdPoint(control, located))
            previous = located
        return points


class StochasticSample:
    """Both stochastic bifurcation diagnostics at one parameter value."""

    __slots__ = ["_density", "_exponent", "_parameter"]

    def __init__(
        self,
        parameter: float,
        density: GridDensity,
        exponent: float,
    ) -> None:
        self._parameter = parameter
        self._density = density
        self._exponent = exponent

    @property
    def parameter(self) -> float:
        """Parameter value this sample was taken at."""
        return self._parameter

    @property
    def density(self) -> GridDensity:
        """Stationary density there."""
        return self._density

    @property
    def exponent(self) -> float:
        """Top Lyapunov exponent there."""
        return self._exponent


class StochasticScan:
    """Both diagnostics across a parameter range, as one record per value.

    The two notions of stochastic bifurcation are kept as separate objects everywhere
    else in this package because they are separate phenomena and can occur at different
    parameters. They are reported together here because deciding *which* of them a
    given system undergoes, and in which order, requires seeing both along the same
    range - which is the whole content of the statement that the noisy Hopf system has
    a density that craters at one parameter and an origin that destabilises at another.
    """

    __slots__ = ["_density_at", "_exponent_at"]

    def __init__(
        self,
        density_at: Callable[[float], StationaryFokkerPlanck],
        exponent_at: Callable[[float], float],
    ) -> None:
        self._density_at = density_at
        self._exponent_at = exponent_at

    def run(self, values: Sequence[float]) -> list[StochasticSample]:
        """Solve for the density and estimate the exponent at every value."""
        return [
            StochasticSample(
                value,
                self._density_at(value).solve(),
                self._exponent_at(value),
            )
            for value in values
        ]
