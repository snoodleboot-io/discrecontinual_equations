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

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
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
_HALF = 0.5
_AUTODIFF = AutomaticDifferentiation()


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

    def _dense_tangent_system(
        self,
        unknowns: np.ndarray,
        nodes: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        """The tangent system built by finite differences, for the fallback only."""
        residual = self._cycle_residual(unknowns, nodes)
        jacobian = np.empty((residual.size, unknowns.size))
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += _STEP
            jacobian[:, j] = (self._cycle_residual(shifted, nodes) - residual) / _STEP
        return np.vstack([jacobian, previous])

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

    def _state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """``df/dx`` at one node, analytically where the field allows it.

        Automatic differentiation carries Taylor jets through the field, which a
        field written with ``math.sqrt`` or a fractional power rejects. Those fall
        back to finite differences - still one column per state rather than one
        per unknown of the whole augmented system.
        """
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        try:
            return _AUTODIFF.jacobian(self._equation.derivative, state, 0.0)
        except (TypeError, ValueError, AttributeError):
            base = self._field(state, parameter)
            columns = []
            for j in range(state.size):
                shifted = np.array(state, dtype=float)
                shifted[j] += _STEP
                columns.append((self._field(shifted, parameter) - base) / _STEP)
            return np.column_stack(columns)

    def _blocks(
        self,
        unknowns: np.ndarray,
        nodes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The cycle residual's Jacobian as sparse triplets.

        Trapezoidal collocation makes this bordered almost-block-diagonal: a band
        from the interval blocks, one dense column for the period and another for
        the parameter, the periodicity rows coupling the first and last nodes, and
        the phase row. Interval ``i`` contributes ``-I - w Df(x_i)`` against node
        ``i`` and ``I - w Df(x_{i+1})`` against node ``i+1``.

        This is what DEQ-4 and DEQ-6 did for the periodic orbit solver, which the
        cycle *continuation* never got: it perturbed every unknown and re-evaluated
        the whole augmented residual, so a Newton step cost ``nodes * dimension +
        2`` residuals of ``O(nodes)`` field evaluations each, then a dense solve.
        """
        dimension = self._dimension
        intervals = self._intervals
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[nodes * dimension])
        parameter = float(unknowns[-1])
        size = unknowns.size
        identity = np.eye(dimension)
        step = 1.0 / intervals
        weight = _HALF * period * step

        derivatives = np.array(
            [self._state_jacobian(states[i], parameter) for i in range(nodes)],
        )
        fields = np.array([self._field(states[i], parameter) for i in range(nodes)])
        # df/dp at every node: one field evaluation each, against the whole
        # augmented residual a finite-difference column would have cost.
        ahead = np.array(
            [self._field(states[i], parameter + _STEP) for i in range(nodes)],
        )
        parameter_fields = (ahead - fields) / _STEP

        left = -identity[None] - weight * derivatives[:-1]
        right = identity[None] - weight * derivatives[1:]
        span = np.arange(dimension)
        starts = np.arange(intervals) * dimension
        block_rows = np.broadcast_to(
            (starts[:, None] + span)[:, :, None],
            (intervals, dimension, dimension),
        )
        left_columns = np.broadcast_to(
            (starts[:, None] + span)[:, None, :],
            (intervals, dimension, dimension),
        )
        right_columns = left_columns + dimension
        border_rows = starts[:, None] + span
        period_values = -_HALF * step * (fields[:-1] + fields[1:])
        parameter_values = -weight * (parameter_fields[:-1] + parameter_fields[1:])
        periodic = intervals * dimension + span

        rows = np.concatenate(
            [
                block_rows.ravel(),
                block_rows.ravel(),
                border_rows.ravel(),
                border_rows.ravel(),
                periodic,
                periodic,
                [intervals * dimension + dimension],
            ],
        )
        columns = np.concatenate(
            [
                left_columns.ravel(),
                right_columns.ravel(),
                np.full(intervals * dimension, size - 2),
                np.full(intervals * dimension, size - 1),
                intervals * dimension + span,
                span,
                [self._phase_index],
            ],
        )
        values = np.concatenate(
            [
                left.ravel(),
                right.ravel(),
                period_values.ravel(),
                parameter_values.ravel(),
                np.ones(dimension),
                -np.ones(dimension),
                [1.0],
            ],
        )
        return rows, columns, values

    def _bordered(
        self,
        unknowns: np.ndarray,
        nodes: int,
        border: np.ndarray,
    ) -> coo_matrix:
        """The cycle Jacobian with ``border`` as its last row.

        Both systems the continuation solves have this shape: the corrector borders
        it with the arclength constraint's tangent, and the tangent solve borders it
        with the previous tangent. Either way the result is square, so a sparse LU
        applies directly.
        """
        rows, columns, values = self._blocks(unknowns, nodes)
        size = unknowns.size
        last = nodes * self._dimension + 1
        return coo_matrix(
            (
                np.concatenate([values, border]),
                (
                    np.concatenate([rows, np.full(size, last)]),
                    np.concatenate([columns, np.arange(size)]),
                ),
            ),
            shape=(size, size),
        )

    def _solve_bordered(
        self,
        unknowns: np.ndarray,
        nodes: int,
        border: np.ndarray,
        rhs: np.ndarray,
        dense: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """One sparse solve, falling back to a dense least squares.

        Where the Jacobian is singular the sparse LU returns a non-finite vector
        rather than raising, so the result is checked before it is trusted - the
        same guard the periodic orbit solver uses.

        ``dense`` is a callable and not an array on purpose. Building the
        finite-difference system costs a whole Newton step's worth of field
        evaluations, which is the expense this class exists to avoid; passing it
        eagerly would pay that cost on every step and leave the sparse path
        saving nothing at all.
        """
        sparse = self._bordered(unknowns, nodes, border).tocsc()
        update = spsolve(sparse, rhs)
        if np.all(np.isfinite(update)):
            return update
        fallback, *_ = np.linalg.lstsq(dense(), rhs, rcond=None)
        return fallback

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
            update = self._solve_bordered(
                unknowns,
                nodes,
                arc.tangent,
                -residual,
                # bound now, not at call time: both are rebound each iteration
                lambda u=unknowns, r=residual: self._numerical_jacobian(
                    u,
                    nodes,
                    arc,
                    r,
                ),
            )
            unknowns = unknowns + update
        residual = self._augmented(unknowns, nodes, arc)
        return unknowns if np.linalg.norm(residual) < _TOLERANCE else None

    def _tangent(
        self,
        unknowns: np.ndarray,
        nodes: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        rhs = np.zeros(unknowns.size)
        rhs[-1] = 1.0
        tangent = self._solve_bordered(
            unknowns,
            nodes,
            previous,
            rhs,
            lambda: self._dense_tangent_system(unknowns, nodes, previous),
        )
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
