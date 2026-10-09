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

The orbit is discretised by collocation on a uniform mesh, and the scheme is a
choice: fourth-order Hermite-Simpson by default, second-order trapezoidal on
request. See :class:`CycleContinuation` for why the order, and not the mesh, is
what decides whether the multipliers can be believed.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise

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
# Simpson's rule weights the end fields by h/6 and the midpoint field by four times
# that; the cubic Hermite interpolant places the midpoint h/8 along the difference
# of the end slopes.
_SIXTH = 6.0
_EIGHTH = 8.0
_MIDPOINT_WEIGHT = 4.0
_AUTODIFF = AutomaticDifferentiation()
# How far the trivial multiplier may drift from one before a cycle point is refused
# as unresolved. A crossing of the unit circle is located by linearly interpolating
# ``|mu| - 1`` between adjacent points, so an error of a couple of percent in ``|mu|``
# moves the reported crossing by a comparable fraction of one continuation step -
# tolerable. Tens of percent does not: it can invent a crossing, hide one, or place
# it a long way from where it is, and on the Bogdanov-Takens branch at 80 nodes
# under trapezoidal collocation the error reaches 59%. On a Bogdanov-Takens-like
# cycle this sits between what 160 and 320 trapezoidal nodes buy (8.9% at 80, 2.5%
# at 160, 0.6% at 320); Hermite-Simpson is under it on the whole branch at 80.
_FLOQUET_TOLERANCE = 2.0e-2


class _VectorField:
    """``f``, ``df/dx`` and ``df/dp`` of the equation, as three callables.

    Each takes ``(state, parameter)``. Bundled so that a collocation scheme asks
    one object for whatever it needs at a node or a midpoint, and so that the
    scheme itself holds no reference to the equation and stays a stateless
    singleton.
    """

    __slots__ = ["jacobian", "parameter_derivative", "value"]

    def __init__(
        self,
        value: Callable[[np.ndarray, float], np.ndarray],
        jacobian: Callable[[np.ndarray, float], np.ndarray],
        parameter_derivative: Callable[[np.ndarray, float], np.ndarray],
    ) -> None:
        self.value = value
        self.jacobian = jacobian
        self.parameter_derivative = parameter_derivative


class _Collocation:
    """One way of discretising ``x'(s) = T f(x(s), p)`` on the uniform mesh.

    A scheme is two things: the residual of one interval, and the Jacobian of that
    residual against the interval's two end nodes, the period, and the parameter.
    Both schemes here couple an interval only to its own two nodes, so the sparsity
    pattern :meth:`CycleContinuation._blocks` assembles is the same for each and
    only the block values differ. That is why the scheme supplies values and not
    triplets: the index arithmetic DEQ-15 got right stays in one place.
    """

    name = ""

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> np.ndarray:
        """The collocation residual, one row of ``dimension`` per interval."""
        raise NotImplementedError

    def blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-interval Jacobian blocks: ``(left, right, period, parameter)``.

        ``left`` and ``right`` are the ``dimension x dimension`` blocks against
        nodes ``i`` and ``i+1``; ``period`` and ``parameter`` are the interval's
        rows of the two dense columns.
        """
        raise NotImplementedError


class _Trapezoidal(_Collocation):
    """Second order: ``x_{i+1} - x_i = (hT/2)(f_i + f_{i+1})``."""

    name = "trapezoidal"

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> np.ndarray:
        fields = np.array([field.value(state, parameter) for state in states])
        weight = _HALF * period * step
        return states[1:] - states[:-1] - weight * (fields[:-1] + fields[1:])

    def blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        identity = np.eye(states.shape[1])
        weight = _HALF * period * step
        derivatives = np.array([field.jacobian(state, parameter) for state in states])
        fields = np.array([field.value(state, parameter) for state in states])
        parameter_fields = np.array(
            [field.parameter_derivative(state, parameter) for state in states],
        )
        left = -identity[None] - weight * derivatives[:-1]
        right = identity[None] - weight * derivatives[1:]
        period_values = -_HALF * step * (fields[:-1] + fields[1:])
        parameter_values = -weight * (parameter_fields[:-1] + parameter_fields[1:])
        return left, right, period_values, parameter_values


class _HermiteSimpson(_Collocation):
    """Fourth order: Simpson's rule with the midpoint on the cubic Hermite.

    The midpoint state is ``m = (x_i + x_{i+1})/2 + (hT/8)(f_i - f_{i+1})`` and the
    residual ``x_{i+1} - x_i - (hT/6)(f_i + 4 f(m) + f_{i+1})``, exactly as
    :class:`HermiteSimpsonOrbit` forms it, including the period scaling the slopes
    that place the midpoint. What makes the Jacobian wider than trapezoidal's is
    that ``m`` depends on everything: both end nodes through the mean and the
    slopes, the period through the slopes, and the parameter through the fields.
    Every block therefore carries a chain-rule term ``4a Df(m) dm/d(.)`` with
    ``a = hT/6``, and the end-node blocks pick up a product ``Df(m) Df(x_i)`` that
    trapezoidal never has. The pattern is unchanged - each interval still touches
    only its own two nodes - so the sparse assembly and solve are the same.
    """

    name = "hermite_simpson"

    def _midpoints(
        self,
        states: np.ndarray,
        fields: np.ndarray,
        period: float,
        step: float,
    ) -> np.ndarray:
        slope = period * step / _EIGHTH
        return _HALF * (states[:-1] + states[1:]) + slope * (fields[:-1] - fields[1:])

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> np.ndarray:
        fields = np.array([field.value(state, parameter) for state in states])
        midpoints = self._midpoints(states, fields, period, step)
        middles = np.array([field.value(point, parameter) for point in midpoints])
        weight = period * step / _SIXTH
        return (
            states[1:]
            - states[:-1]
            - weight * (fields[:-1] + _MIDPOINT_WEIGHT * middles + fields[1:])
        )

    def blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        identity = np.eye(states.shape[1])
        fields = np.array([field.value(state, parameter) for state in states])
        derivatives = np.array([field.jacobian(state, parameter) for state in states])
        parameter_fields = np.array(
            [field.parameter_derivative(state, parameter) for state in states],
        )
        midpoints = self._midpoints(states, fields, period, step)
        middles = np.array([field.value(point, parameter) for point in midpoints])
        middle_derivatives = np.array(
            [field.jacobian(point, parameter) for point in midpoints],
        )
        middle_parameter_fields = np.array(
            [field.parameter_derivative(point, parameter) for point in midpoints],
        )

        end_weight = period * step / _SIXTH
        middle_weight = _MIDPOINT_WEIGHT * end_weight
        slope = period * step / _EIGHTH
        # dm/dx_i and dm/dx_{i+1}: the mean, then the Hermite slope through Df.
        midpoint_left = _HALF * identity[None] + slope * derivatives[:-1]
        midpoint_right = _HALF * identity[None] - slope * derivatives[1:]
        # dm/dT and dm/dp: only the slope term moves, through f and df/dp.
        midpoint_period = (step / _EIGHTH) * (fields[:-1] - fields[1:])
        midpoint_parameter = slope * (parameter_fields[:-1] - parameter_fields[1:])

        chain = middle_weight * middle_derivatives
        left = -identity[None] - end_weight * derivatives[:-1] - chain @ midpoint_left
        right = identity[None] - end_weight * derivatives[1:] - chain @ midpoint_right
        period_values = -(step / _SIXTH) * (
            fields[:-1] + _MIDPOINT_WEIGHT * middles + fields[1:]
        ) - np.einsum("ijk,ik->ij", chain, midpoint_period)
        parameter_values = -end_weight * (
            parameter_fields[:-1]
            + _MIDPOINT_WEIGHT * middle_parameter_fields
            + parameter_fields[1:]
        ) - np.einsum("ijk,ik->ij", chain, midpoint_parameter)
        return left, right, period_values, parameter_values


_COLLOCATIONS: dict[str, _Collocation] = {
    scheme.name: scheme for scheme in (_Trapezoidal(), _HermiteSimpson())
}


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
        by about this fraction. Measured under trapezoidal collocation on a
        Bogdanov-Takens cycle hugging a saddle (nontrivial multiplier 6.16): 8.9%
        at 80 nodes, 2.5% at 160, 0.6% at 320, 0.2% at 640 - second order in the
        mesh, and forty times more sensitive than the period, which was 0.2% off
        at 80 nodes. The monodromy quadrature itself is not the limit: on the
        exact orbit it returns the multipliers to four figures at 80 nodes.

        It is a measure of the *discretisation*, not of the parameter, so it
        varies enormously along one branch: on that same Bogdanov-Takens branch
        at 80 trapezoidal nodes it reads 8.9% on a typical cycle and 59% on the
        one nearest the homoclinic, whose period the same mesh gets 3.8% wrong.
        Under the default Hermite-Simpson scheme the same branch stays under 1%
        throughout, and what is left is the monodromy's own interpolation
        between nodes. Use :func:`resolved_branch` rather than a single sampled
        point to decide whether a branch's multipliers can be believed.
        """
        return float(np.min(np.abs(self._multipliers - 1.0)))

    def resolved(self, tolerance: float = _FLOQUET_TOLERANCE) -> bool:
        """Whether this cycle's multipliers are accurate enough to be believed.

        The counterpart of :meth:`AdaptivePeriodicOrbit.continue_to_resolved` for a
        cycle, and far cheaper for the same reason ``floquet_error`` is: the period
        has to be re-solved on a doubled mesh before anyone can say how wrong it is,
        whereas the trivial multiplier's exact value is known in advance, so this
        measurement arrives with the point and costs nothing.
        """
        return self.floquet_error < tolerance

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


class ResolvedBranch:
    """The stretch of a traced branch whose Floquet multipliers can be believed.

    ``refused`` is kept as a count rather than thrown away, because it is the useful
    half of the answer. A branch that was traced further than it can be trusted is
    not the same thing as a branch that ended: the first says raise the node count,
    the second says the cycle really is gone. Only the count can tell them apart.
    """

    __slots__ = ["_bifurcations", "_points", "_refused", "_tolerance"]

    def __init__(
        self,
        points: list[CyclePoint],
        bifurcations: list[CycleBifurcation],
        refused: int,
        tolerance: float,
    ) -> None:
        self._points = points
        self._bifurcations = bifurcations
        self._refused = refused
        self._tolerance = tolerance

    @property
    def points(self) -> list[CyclePoint]:
        """The longest unbroken run of cycles whose multipliers met the tolerance."""
        return self._points

    @property
    def bifurcations(self) -> list[CycleBifurcation]:
        """Bifurcations detected between adjacent *kept* points."""
        return self._bifurcations

    @property
    def refused(self) -> int:
        """How many of the branch's points fell outside that run."""
        return self._refused

    @property
    def tolerance(self) -> float:
        """The Floquet error the kept points were required to stay under."""
        return self._tolerance

    @property
    def worst_error(self) -> float:
        """Largest Floquet error among the kept points; ``0`` when none were kept."""
        return max((point.floquet_error for point in self._points), default=0.0)


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


def resolved_branch(
    points: Sequence[CyclePoint],
    tolerance: float = _FLOQUET_TOLERANCE,
) -> ResolvedBranch:
    """The stretch of a traced branch whose Floquet multipliers can be believed.

    :meth:`CycleContinuation.trace` answers "where did the branch go"; this answers
    "how much of it can be believed", which is the question a caller about to read
    stability off the multipliers actually has. A converged cycle proves only that the
    *discrete* collocation system was satisfied; an under-resolved mesh satisfies it
    perfectly well while reporting multipliers tens of percent out - far enough to
    move a crossing of the unit circle, which is the thing the branch was traced to
    find. Nothing in the residual can catch that, and the trivial multiplier can.

    An unbroken run and not a filter, because :func:`classify_transition` compares
    *adjacent* points: a branch with holes punched in it would have it interpolating a
    crossing across a gap neither of whose ends it looked at. So the bifurcations are
    re-derived over the kept run rather than sieved out of the full trace.

    The *longest* such run, rather than the leading one, because a branch is not
    always handed over in the order it was traced. Resolution does degrade
    monotonically outward from the seed, so for :meth:`CycleContinuation.trace`'s own
    output the leading run is the answer. But a film that traces both ways from a seed
    and sorts the result by parameter - which is the usual shape here - puts the
    *worst* point first, and a leading run would then refuse the whole branch
    including the well-resolved middle it was asked about. Taking the longest run
    gives the right answer to both.

    This is a gate, not a remedy. Raising the node count is the remedy, and nothing
    cheaper was found to be one - see the note on
    :class:`CycleContinuation` about the adaptive mesh that was measured and rejected.
    """
    best_start, best_length, start = 0, 0, 0
    for index, point in enumerate(points):
        if not point.resolved(tolerance):
            start = index + 1
            continue
        if index + 1 - start > best_length:
            best_start, best_length = start, index + 1 - start
    kept = list(points[best_start : best_start + best_length])
    bifurcations = [
        transition
        for lower, upper in pairwise(kept)
        if (transition := classify_transition(lower, upper)) is not None
    ]
    return ResolvedBranch(kept, bifurcations, len(points) - len(kept), tolerance)


class CycleContinuation:
    """Pseudo-arclength continuation of a limit cycle in one parameter.

    Every cycle on the branch is discretised on the same uniform mesh of
    ``intervals`` intervals in rescaled time. An adaptive mesh was tried here as the
    cure for the Floquet error that a near-homoclinic orbit carries, and rejected on
    measurement. Three things were found, in order of how much they settle:

    * The mesh is not the lever. On the Bogdanov-Takens branch's worst orbit at 80
      nodes, equidistributing every monitor tried - curvature and its square and cube
      roots, ``|dx/ds|`` and its roots, arclength-plus-curvature, each with and
      without a floor - moved the Floquet error from 59% to between 47% and 64%, the
      best of them a factor of 1.3. The monitor
      :class:`AdaptivePeriodicOrbit` actually uses lands on 55%. There is no
      80-node mesh that resolves this orbit, so no mesh monitor could have found one.
    * The order is the lever. The same orbit on the same 80 *uniform* nodes under
      fourth-order Hermite-Simpson collocation gives 0.12%, a factor of 500, with the
      period right to five figures where trapezoidal gets it 3.8% wrong. The error is
      the second-order collocation of the orbit itself, not the mesh it sits on and
      not the monodromy quadrature - on the exact orbit sampled at 80 nodes that
      quadrature returns the multipliers to 0.2%.
    * Re-meshing along the branch also costs more than it returns. Re-adapting at
      every point whose error exceeded tolerance, re-projecting the tangent and
      re-solving on the moved mesh, took the worst case from 59.3% to 49.3% while
      taking 7.5x the time (302 s against 40 s), leaving the median slightly worse
      (3.95% against 3.71%) and stopping marginally *sooner* on the branch
      (b1 = -0.4557 against -0.4575).

    So that higher-order collocation is now wired in, and is the default.
    ``collocation`` picks the scheme: ``"hermite_simpson"`` (fourth order) or
    ``"trapezoidal"`` (second order). It is a parameter and not a replacement
    because the two answer different questions. Trapezoidal is the oracle the
    fourth-order claim is measured against - the convergence-order test needs both
    on the same mesh - and it is what every film shipped until this landed, so a
    film that wants to keep its old output, or to show the two side by side, can
    say so at the call site rather than by checking out an old revision. The cost
    is small: both schemes share the sparse assembly, the solve, the corrector and
    the tangent, and differ only in the per-interval residual and block values,
    which :class:`_Collocation` keeps to about thirty lines each. Hermite-Simpson
    is the default because a new caller should get the scheme whose multipliers
    can be believed without having to know this history. :func:`resolved_branch`
    remains the way a caller finds out whether the mesh it chose was enough.
    """

    __slots__ = [
        "_collocation",
        "_dimension",
        "_equation",
        "_intervals",
        "_parameter_index",
        "_phase_index",
        "_phase_value",
    ]

    def __init__(  # noqa: PLR0913 (one scheme flag on the established signature)
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
        *,
        collocation: str = "hermite_simpson",
    ) -> None:
        if collocation not in _COLLOCATIONS:
            message = (
                f"collocation must be one of {sorted(_COLLOCATIONS)}, "
                f"got {collocation!r}"
            )
            raise ValueError(message)
        self._equation = equation
        self._parameter_index = parameter_index
        self._intervals = intervals
        self._phase_index = phase_index
        self._phase_value = phase_value
        self._collocation = _COLLOCATIONS[collocation]
        self._dimension = 0

    @property
    def collocation(self) -> str:
        """The collocation scheme: ``"hermite_simpson"`` or ``"trapezoidal"``."""
        return self._collocation.name

    def _field(self, state: np.ndarray, parameter: float) -> np.ndarray:
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        return np.array(
            self._equation.derivative.eval(point=list(state), time=None),
            dtype=float,
        )

    def _parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """``df/dp`` at one state, by a forward difference in the parameter.

        One extra field evaluation per node, against the whole augmented residual
        that a finite-difference column of the full Jacobian would have cost.
        """
        base = self._field(state, parameter)
        return (self._field(state, parameter + _STEP) - base) / _STEP

    def _vector_field(self) -> _VectorField:
        return _VectorField(
            self._field,
            self._state_jacobian,
            self._parameter_derivative,
        )

    def _cycle_residual(self, unknowns: np.ndarray, nodes: int) -> np.ndarray:
        dimension = self._dimension
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[nodes * dimension])
        parameter = float(unknowns[-1])
        collocation = self._collocation.residual(
            self._vector_field(),
            states,
            period,
            parameter,
            1.0 / self._intervals,
        )
        return np.concatenate(
            [
                collocation.ravel(),
                states[-1] - states[0],
                [states[0, self._phase_index] - self._phase_value],
            ],
        )

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

        Either collocation makes this bordered almost-block-diagonal: a band from
        the interval blocks, one dense column for the period and another for the
        parameter, the periodicity rows coupling the first and last nodes, and the
        phase row. Under trapezoidal collocation interval ``i`` contributes
        ``-I - w Df(x_i)`` against node ``i`` and ``I - w Df(x_{i+1})`` against node
        ``i+1``; Hermite-Simpson adds the midpoint's chain-rule terms to the same
        four blocks (see :class:`_HermiteSimpson`), so the scheme supplies the
        values and this method owns the pattern.

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

        left, right, period_values, parameter_values = self._collocation.blocks(
            self._vector_field(),
            states,
            period,
            parameter,
            1.0 / intervals,
        )
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

    def _solve_seed(self, seed: CycleSeed, nodes: int) -> np.ndarray | None:
        """The first point on the branch: the seed, solved at its own parameter.

        Two stages whatever the scheme. First the trapezoidal orbit solver every
        film has always seeded from; then the continuation's own sparse Newton,
        with the parameter held fixed, carries that cycle onto the scheme in use.
        For trapezoidal the second stage is a no-op - the orbit solver has already
        met the corrector's tolerance, so it returns after one residual - and the
        path is exactly what it was before a scheme could be chosen.

        Two stages and not one, because a seed is a rough guess - a circle, a few
        integration samples - and the fourth-order system's Newton basin around
        such a guess is smaller than the second-order one's. On van der Pol's
        circle seed at ``mu = 0.08`` the smallest singular value of the
        Hermite-Simpson seed Jacobian is 6e-7 against 2e-5 for trapezoidal, and
        the first undamped Newton step takes the period to -2592 and never comes
        back; trapezoidal's first step is itself 148 long, but it recovers. The
        trapezoidal cycle is within its own ``O(h^2)`` error of the fourth-order
        one, and from there the correction converges in a few quadratic steps.
        It is also the cheap way round: the correction uses the analytic sparse
        Jacobian, where a dense fourth-order orbit solve would cost seconds at
        160 nodes and dominate a short trace.
        """
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
            return None
        unknowns = np.concatenate(
            [solved.states.flatten(), [solved.period], [seed.parameter]],
        )
        fixed_parameter = np.zeros(unknowns.size)
        fixed_parameter[-1] = 1.0
        return self._correct(unknowns, nodes, _Arc(unknowns, fixed_parameter, 0.0))

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
        unknowns = self._solve_seed(seed, nodes)
        if unknowns is None:
            return [], []
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
