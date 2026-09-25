"""Continue a limit cycle in one parameter and detect its bifurcations.

The cycle is continued by pseudo-arclength so the branch can round a turning point.
The augmented unknown carries the cycle nodes, the period, and the active parameter;
the extra equation is the arclength constraint. Along the branch the Floquet
multipliers are tracked, and their crossings mark the codimension-one bifurcations
of a cycle:

* **fold of cycles** - a real nontrivial multiplier passes ``+1`` (two cycles meet
  and annihilate; the branch turns);
* **period-doubling** - a real nontrivial multiplier passes ``-1``;
* **Neimark-Sacker** - a complex-conjugate pair of multipliers crosses the unit
  circle (a torus is born).
"""

import numpy as np

from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_TOLERANCE = 1.0e-10
_ITERATIONS = 40
_STEP = 1.0e-7
_TRIVIAL_BAND = 5.0e-2
_CROSS_MARGIN = 1.0e-6
_IMAGINARY_TOLERANCE = 1.0e-6
_NOISE_FLOOR = 1.0e-9
_ANGLE_BAND = 0.12


class CycleSeed:
    """Initial guess for a cycle branch: node states, period, and parameter."""

    __slots__ = ["parameter", "period", "states"]

    def __init__(self, states: np.ndarray, period: float, parameter: float) -> None:
        self.states = states
        self.period = period
        self.parameter = parameter


class _Arc:
    """Pseudo-arclength context: anchor point, tangent, and step length."""

    __slots__ = ["anchor", "arclength", "tangent"]

    def __init__(
        self,
        anchor: np.ndarray,
        tangent: np.ndarray,
        arclength: float,
    ) -> None:
        self.anchor = anchor
        self.tangent = tangent
        self.arclength = arclength


class CyclePoint:
    """A cycle on the branch: its parameter, orbit, and Floquet multipliers."""

    __slots__ = ["_multipliers", "_parameter", "_solution"]

    def __init__(
        self,
        parameter: float,
        solution: PeriodicOrbitSolution,
        multipliers: np.ndarray,
    ) -> None:
        self._parameter = parameter
        self._solution = solution
        self._multipliers = multipliers

    @property
    def parameter(self) -> float:
        """Active parameter value on the branch."""
        return self._parameter

    @property
    def solution(self) -> PeriodicOrbitSolution:
        """The limit cycle at this point."""
        return self._solution

    @property
    def multipliers(self) -> np.ndarray:
        """Floquet multipliers of the cycle."""
        return self._multipliers

    @property
    def floquet_error(self) -> float:
        """How far the trivial multiplier is from one: the multipliers' accuracy.

        One multiplier is exactly ``1`` along the orbit, so its computed value
        measures the error of the whole set: the monodromy is integrated along
        the *discrete* orbit, and an under-resolved mesh shifts every multiplier
        by about this fraction. Measured on a Bogdanov-Takens cycle hugging a
        saddle (nontrivial multiplier 6.16): 8.9% at 80 nodes, 2.5% at 160,
        0.6% at 320, 0.2% at 640 - second order in the mesh, and forty times
        more sensitive than the period, which was 0.2% off at 80 nodes. The
        monodromy quadrature itself is not the limit: on the exact orbit it
        returns the multipliers to four figures at 80 nodes.
        """
        return float(np.min(np.abs(self._multipliers - 1.0)))

    @property
    def amplitude(self) -> float:
        """Peak radius of the cycle about its centre."""
        states = self._solution.states
        return float(np.max(np.linalg.norm(states - states.mean(axis=0), axis=1)))


class CycleBifurcation:
    """A detected cycle bifurcation: its kind and parameter value."""

    __slots__ = ["_kind", "_parameter"]

    def __init__(self, kind: str, parameter: float) -> None:
        self._kind = kind
        self._parameter = parameter

    @property
    def kind(self) -> str:
        """``fold_of_cycles``, ``period_doubling``, or ``neimark_sacker``."""
        return self._kind

    @property
    def parameter(self) -> float:
        """Parameter value at the bifurcation (linear estimate)."""
        return self._parameter


def _nontrivial(multipliers: np.ndarray) -> np.ndarray:
    """Drop the trivial ``+1`` multiplier (the one nearest ``+1``)."""
    order = np.argsort(np.abs(multipliers - 1.0))
    return multipliers[order[1:]]


def _excess_near_angle(multipliers: np.ndarray, target: float) -> float:
    """Largest ``|mu| - 1`` among nontrivial multipliers whose angle is near target.

    Used with target ``0`` (fold of cycles, ``mu -> +1``) and ``pi``
    (period-doubling, ``mu -> -1``); returns ``-1`` when none lie near the angle.
    """
    excess = -1.0
    for multiplier in _nontrivial(multipliers):
        angle = abs(float(np.angle(multiplier)))
        if abs(angle - target) < _ANGLE_BAND:
            excess = max(excess, abs(multiplier) - 1.0)
    return excess


def _torus_excess(multipliers: np.ndarray) -> float:
    """Largest ``|mu| - 1`` among complex pairs with angle away from 0 and pi."""
    excess = -1.0
    for multiplier in _nontrivial(multipliers):
        angle = abs(float(np.angle(multiplier)))
        if (
            _ANGLE_BAND < angle < np.pi - _ANGLE_BAND
            and abs(multiplier.imag) >= _IMAGINARY_TOLERANCE
        ):
            excess = max(excess, abs(multiplier) - 1.0)
    return excess


def _crossing(before: float, after: float) -> bool:
    if abs(before) < _NOISE_FLOOR and abs(after) < _NOISE_FLOOR:
        return False  # a flat segment sitting on zero is noise, not a crossing
    return before <= 0.0 <= after or after <= 0.0 <= before


def _interpolate(
    lower: CyclePoint,
    upper: CyclePoint,
    before: float,
    after: float,
) -> float:
    fraction = before / (before - after) if (before - after) != 0.0 else 0.5
    return lower.parameter + fraction * (upper.parameter - lower.parameter)


def classify_transition(
    lower: CyclePoint,
    upper: CyclePoint,
) -> CycleBifurcation | None:
    """Classify a bifurcation between two adjacent cycle-branch points, if any.

    A Floquet multiplier crossing the unit circle is a fold of cycles at ``+1``
    (angle ``0``), a period-doubling at ``-1`` (angle ``pi``), and a Neimark-Sacker
    when a genuine complex pair crosses elsewhere.
    """
    fold_before = _excess_near_angle(lower.multipliers, 0.0)
    fold_after = _excess_near_angle(upper.multipliers, 0.0)
    if _crossing(fold_before, fold_after):
        return CycleBifurcation(
            "fold_of_cycles",
            _interpolate(lower, upper, fold_before, fold_after),
        )
    flip_before = _excess_near_angle(lower.multipliers, float(np.pi))
    flip_after = _excess_near_angle(upper.multipliers, float(np.pi))
    if _crossing(flip_before, flip_after):
        return CycleBifurcation(
            "period_doubling",
            _interpolate(lower, upper, flip_before, flip_after),
        )
    torus_before = _torus_excess(lower.multipliers)
    torus_after = _torus_excess(upper.multipliers)
    if _crossing(torus_before, torus_after):
        return CycleBifurcation(
            "neimark_sacker",
            _interpolate(lower, upper, torus_before, torus_after),
        )
    return None


class CycleContinuation:
    """Pseudo-arclength continuation of a limit cycle in one parameter."""

    __slots__ = [
        "_dimension",
        "_equation",
        "_intervals",
        "_parameter_index",
        "_phase_index",
        "_phase_value",
    ]

    def __init__(
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
    ) -> None:
        self._equation = equation
        self._parameter_index = parameter_index
        self._intervals = intervals
        self._phase_index = phase_index
        self._phase_value = phase_value
        self._dimension = 0

    def _field(self, state: np.ndarray, parameter: float) -> np.ndarray:
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        return np.array(
            self._equation.derivative.eval(point=list(state), time=None),
            dtype=float,
        )

    def _cycle_residual(self, unknowns: np.ndarray, nodes: int) -> np.ndarray:
        dimension = self._dimension
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[nodes * dimension])
        parameter = float(unknowns[-1])
        step = 1.0 / self._intervals
        blocks = []
        for i in range(self._intervals):
            here = self._field(states[i], parameter)
            ahead = self._field(states[i + 1], parameter)
            blocks.append(
                states[i + 1] - states[i] - 0.5 * period * step * (here + ahead),
            )
        blocks.append(states[-1] - states[0])
        blocks.append(np.array([states[0, self._phase_index] - self._phase_value]))
        return np.concatenate(blocks)

    def _augmented(self, unknowns: np.ndarray, nodes: int, arc: _Arc) -> np.ndarray:
        cycle = self._cycle_residual(unknowns, nodes)
        constraint = float(arc.tangent @ (unknowns - arc.anchor) - arc.arclength)
        return np.concatenate([cycle, [constraint]])

    def _numerical_jacobian(
        self,
        unknowns: np.ndarray,
        nodes: int,
        arc: _Arc,
        residual: np.ndarray,
    ) -> np.ndarray:
        jacobian = np.empty((residual.size, unknowns.size))
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += _STEP
            jacobian[:, j] = (self._augmented(shifted, nodes, arc) - residual) / _STEP
        return jacobian

    def _correct(
        self,
        prediction: np.ndarray,
        nodes: int,
        arc: _Arc,
    ) -> np.ndarray | None:
        unknowns = prediction
        for _ in range(_ITERATIONS):
            residual = self._augmented(unknowns, nodes, arc)
            if np.linalg.norm(residual) < _TOLERANCE:
                return unknowns
            jacobian = self._numerical_jacobian(
                unknowns,
                nodes,
                arc,
                residual,
            )
            update, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            unknowns = unknowns + update
        residual = self._augmented(unknowns, nodes, arc)
        return unknowns if np.linalg.norm(residual) < _TOLERANCE else None

    def _tangent(
        self,
        unknowns: np.ndarray,
        nodes: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        residual = self._cycle_residual(unknowns, nodes)
        jacobian = np.empty((residual.size, unknowns.size))
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += _STEP
            jacobian[:, j] = (self._cycle_residual(shifted, nodes) - residual) / _STEP
        system = np.vstack([jacobian, previous])
        rhs = np.zeros(unknowns.size)
        rhs[-1] = 1.0
        tangent, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        tangent = tangent / np.linalg.norm(tangent)
        if tangent @ previous < 0.0:
            tangent = -tangent
        return tangent

    def _multipliers(
        self,
        states: np.ndarray,
        period: float,
        parameter: float,
    ) -> np.ndarray:
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        orbit = PeriodicOrbit(
            self._equation.derivative,
            self._intervals,
            self._phase_index,
            self._phase_value,
        )
        return orbit.floquet_multipliers(PeriodicOrbitSolution(states, period))

    def trace(
        self,
        seed: CycleSeed,
        arclength: float,
        steps: int,
        direction: float = 1.0,
    ) -> tuple[list[CyclePoint], list[CycleBifurcation]]:
        """Trace the cycle branch, returning its points and detected bifurcations.

        ``direction`` sets the initial parameter sense (+1 increasing, -1 decreasing);
        the branch then follows its own tangent and may round a turning point.
        """
        self._dimension = seed.states.shape[1]
        nodes = self._intervals + 1
        orbit = PeriodicOrbit(
            self._equation.derivative,
            self._intervals,
            self._phase_index,
            self._phase_value,
        )
        self._equation.derivative.parameters[
            self._parameter_index
        ].value = seed.parameter
        solved = orbit.solve(seed.states, seed.period)
        if solved is None:
            return [], []
        unknowns = np.concatenate(
            [solved.states.flatten(), [solved.period], [seed.parameter]],
        )
        seed_direction = np.zeros(unknowns.size)
        seed_direction[-1] = direction
        tangent = self._tangent(unknowns, nodes, seed_direction)
        points = [self._point(unknowns, nodes)]
        bifurcations: list[CycleBifurcation] = []
        for _ in range(steps):
            prediction = unknowns + arclength * tangent
            arc = _Arc(unknowns, tangent, arclength)
            corrected = self._correct(prediction, nodes, arc)
            if corrected is None:
                break
            tangent = self._tangent(corrected, nodes, tangent)
            unknowns = corrected
            point = self._point(unknowns, nodes)
            transition = classify_transition(points[-1], point)
            if transition is not None:
                bifurcations.append(transition)
            points.append(point)
        return points, bifurcations

    def _point(self, unknowns: np.ndarray, nodes: int) -> CyclePoint:
        dimension = self._dimension
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[nodes * dimension])
        parameter = float(unknowns[-1])
        multipliers = self._multipliers(states, period, parameter)
        return CyclePoint(
            parameter,
            PeriodicOrbitSolution(states, period),
            multipliers,
        )
