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
request, or AUTO's orthogonal collocation at ``m`` Gauss-Legendre points (order
``2m``). See :class:`CycleContinuation` for why the order, and not the mesh, is
what decides whether the multipliers can be believed.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
    _hermite_positions,
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
    Trapezoidal and Hermite-Simpson couple an interval only to its own two nodes,
    so the sparsity pattern :meth:`CycleContinuation._blocks` assembles is the
    same for each and only the block values differ. That is why the scheme
    supplies values and not triplets: the index arithmetic DEQ-15 got right stays
    in one place.

    ``width`` is how many points of the solution each interval adds: one for
    those two, ``m`` for :class:`_GaussLegendre`, whose interval carries ``m - 1``
    interior points besides its right end and so contributes ``m`` rows of
    equations against ``m + 1`` points. :meth:`interval_blocks` is the one block
    shape the assembly takes, and a one-wide scheme gets it for free from
    :meth:`blocks`.
    """

    name = ""
    width = 1
    # Whether the Floquet multipliers are read off this scheme's own Jacobian
    # (:func:`_jacobian_monodromy`) rather than the Hermite monodromy.
    jacobian_monodromy = False

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

    def interval_blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-interval Jacobian blocks in the one shape the assembly takes.

        ``(block, period, parameter)``: ``block`` is ``width * dimension`` rows by
        ``(width + 1) * dimension`` columns, against the interval's points in
        order; the other two are the rows of the dense columns. For a scheme
        with one row per interval this is ``left`` and ``right`` side by side.
        """
        left, right, period_values, parameter_values = self.blocks(
            field,
            states,
            period,
            parameter,
            float(step),
        )
        return np.concatenate([left, right], axis=2), period_values, parameter_values


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


class _GaussLegendre(_Collocation):
    """Order ``2m``: orthogonal collocation at ``m`` Gauss-Legendre points, as AUTO.

    On each interval the orbit is the polynomial of degree ``m`` through ``m + 1``
    equally spaced points - the two mesh points and ``m - 1`` between them - and
    the equation is imposed at the ``m`` Gauss-Legendre points of the interval:
    ``p'(z_k) = h T f(p(z_k))`` in the interval's local coordinate. That is the
    ``m``-stage Gauss Runge-Kutta method written as a boundary-value problem, and
    it is superconvergent: order ``2m`` at the mesh points, and for the period,
    where the polynomial in between is only order ``m + 1``.

    The interval's block is ``D_kj I - h T L_kj Df(p(z_k))`` against its ``m + 1``
    points, with ``L`` and ``D`` the Lagrange basis and its derivative at the
    Gauss points; the period column is ``-h f(p(z_k))`` and the parameter column
    ``-h T df/dp``. Nothing couples two intervals except the shared mesh point,
    so the pattern is still almost block diagonal, only with ``m`` rows per
    interval against ``m + 1`` points instead of one against two.

    ``step`` is an array of the intervals' widths here, not a scalar, so the
    blocks are right on a graded mesh as well; the continuation runs it on the
    uniform one, for the reason given in :class:`CycleContinuation`.

    Measured on the continued Hopf branch (``mu = 0.5``, period exactly 2 pi),
    period error and the order per mesh doubling, coarse meshes for large ``m``
    so the error stays above the corrector's tolerance::

        m = 2    8: 3.2e-3   16: 2.1e-4 (3.96)   32: 1.3e-5 (3.99)   64: 8.1e-7 (4.00)
        m = 3    3: 4.4e-3    6: 7.9e-5 (5.79)   12: 1.3e-6 (5.95)   24: 2.0e-8 (5.99)
        m = 4    3: 8.0e-5    6: 3.5e-7 (7.86)   12: 1.4e-9 (7.97)   24: 5.4e-12 (7.99)

    The nontrivial multiplier off the Jacobian against ``exp(-4 pi mu)`` at
    ``mu = 0.25`` goes the same way: orders 4.13, 4.04, 4.01, 4.00 for ``m = 2``
    from 4 to 64 intervals, 5.13, 5.82, 5.96 for ``m = 3`` from 4 to 32, and
    7.72, 7.94, 7.95 for ``m = 4`` from 3 to 24.
    """

    name = "gauss_legendre"
    jacobian_monodromy = True

    def __init__(self, points: int) -> None:
        self.width = points
        roots, _ = leggauss(points)
        gauss = _HALF * (roots + 1.0)
        powers = np.arange(points + 1)
        grid = np.linspace(0.0, 1.0, points + 1)
        # Column j of this holds the monomial coefficients of the j-th Lagrange
        # basis polynomial on the equally spaced grid.
        self._coefficients = np.linalg.inv(grid[:, None] ** powers[None, :])
        self._values = self._basis(gauss)
        self._derivatives = (
            powers[None, 1:] * gauss[:, None] ** powers[None, :-1]
        ) @ self._coefficients[1:]

    def _basis(self, fractions: np.ndarray) -> np.ndarray:
        """The Lagrange basis at positions in ``[0, 1]``: one row per position."""
        powers = np.arange(self.width + 1)
        return (fractions[:, None] ** powers[None, :]) @ self._coefficients

    def _windows(self, states: np.ndarray) -> np.ndarray:
        """Each interval's ``m + 1`` points: shape ``(intervals, m + 1, dimension)``."""
        intervals = (states.shape[0] - 1) // self.width
        index = np.arange(intervals)[:, None] * self.width + np.arange(self.width + 1)
        return states[index]

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> np.ndarray:
        windows = self._windows(states)
        steps = np.broadcast_to(np.asarray(step, dtype=float), windows.shape[:1])
        points = np.einsum("kj,njd->nkd", self._values, windows)
        slopes = np.einsum("kj,njd->nkd", self._derivatives, windows)
        fields = np.array(
            [
                [field.value(point, parameter) for point in interval]
                for interval in points
            ],
        )
        defect = slopes - (period * steps)[:, None, None] * fields
        return defect.reshape(-1, states.shape[1])

    def interval_blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        windows = self._windows(states)
        intervals, width, dimension = windows.shape[0], self.width, states.shape[1]
        steps = np.broadcast_to(np.asarray(step, dtype=float), (intervals,))
        points = np.einsum("kj,njd->nkd", self._values, windows)
        fields = np.array(
            [
                [field.value(point, parameter) for point in interval]
                for interval in points
            ],
        )
        derivatives = np.array(
            [
                [field.jacobian(point, parameter) for point in interval]
                for interval in points
            ],
        )
        parameter_fields = np.array(
            [
                [field.parameter_derivative(point, parameter) for point in interval]
                for interval in points
            ],
        )
        # Rows (stage k, component d) against columns (point j, component e).
        identity = np.einsum("kj,de->kdje", self._derivatives, np.eye(dimension))
        chain = np.einsum("kj,nkde->nkdje", self._values, derivatives)
        scale = period * steps
        block = identity[None] - scale[:, None, None, None, None] * chain
        period_values = -steps[:, None, None] * fields
        parameter_values = -scale[:, None, None] * parameter_fields
        return (
            block.reshape(intervals, width * dimension, (width + 1) * dimension),
            period_values.reshape(intervals, width * dimension),
            parameter_values.reshape(intervals, width * dimension),
        )

    def grid(self, mesh: np.ndarray) -> np.ndarray:
        """Where the solution's points sit on ``mesh``: ``m`` per interval, then 1."""
        fractions = np.arange(self.width) / self.width
        inner = mesh[:-1, None] + fractions[None, :] * np.diff(mesh)[:, None]
        return np.append(inner.ravel(), mesh[-1])


_COLLOCATIONS: dict[str, _Collocation] = {
    scheme.name: scheme for scheme in (_Trapezoidal(), _HermiteSimpson())
}
_SCHEMES = sorted([*_COLLOCATIONS, _GaussLegendre.name])
# AUTO's default point count (its NCOL), and the most this scheme is tested at.
_GAUSS_POINTS = 4


def _multiplier_disagreement(first: np.ndarray, second: np.ndarray) -> float:
    """Largest relative gap between two readings of one cycle's multipliers.

    Each multiplier of ``first`` is paired with one of ``second`` so that the
    pairs are as close as possible, and each gap is taken relative to the larger
    of one and the multiplier's size: the trivial multiplier is compared in
    absolute terms, as :attr:`CyclePoint.floquet_error` always has, and a
    multiplier of 140 is not called wrong for being off by 1e-3.
    """
    gaps = np.abs(first[:, None] - second[None, :]) / np.maximum(
        1.0,
        np.abs(first)[:, None],
    )
    rows, columns = linear_sum_assignment(gaps)
    return float(np.max(gaps[rows, columns]))


def _jacobian_monodromy(block: np.ndarray, dimension: int) -> np.ndarray:
    """The monodromy read off the collocation Jacobian (Fairgrieve and Jepson).

    The state columns of each interval's block are the linearised collocation
    system: ``A x_left + W (interior, x_right) = 0`` for any perturbation that
    keeps the equations satisfied. Solving it for the interval's far end gives
    the discrete transfer matrix ``-(W^-1 A)`` restricted to its last rows, and
    the product round the mesh is the derivative of the discrete period map,
    which inherits the scheme's order. Nothing is integrated and no field is
    evaluated: the block is the one the last tangent solve assembled.

    The product is formed explicitly rather than by a periodic Schur
    decomposition, which is what keeps this short. That loses accuracy only
    when the multipliers span many orders of magnitude, far beyond the 1e2-1e3
    the films see.
    """
    transfers = -np.linalg.solve(block[:, :, dimension:], block[:, :, :dimension])
    monodromy = np.eye(dimension)
    for transfer in transfers[:, -dimension:]:
        monodromy = transfer @ monodromy
    return monodromy


class _DifferentiatedOrbit(PeriodicOrbit):
    """The Hermite monodromy with ``df/dx`` by automatic differentiation.

    Falls back to the parent's forward difference for a field that automatic
    differentiation rejects, as :meth:`CycleContinuation._state_jacobian` does.
    """

    __slots__ = []

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        try:
            return _AUTODIFF.jacobian(self._function, state, 0.0)
        except (TypeError, ValueError, AttributeError):
            return super()._jacobian(state)


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
    """A cycle on the branch: its parameter, orbit, and Floquet multipliers.

    ``mesh`` is where the solution's states sit in rescaled time; ``None``, the
    default, means uniformly spaced, which is where the one-wide schemes put
    their nodes. A Gauss-Legendre point holds its interior points too and says
    where they are. ``error`` is the multipliers' accuracy where the trivial
    multiplier cannot say it (see :attr:`floquet_error`); ``None`` means it can.
    """

    __slots__ = ["_error", "_mesh", "_multipliers", "_parameter", "_solution"]

    def __init__(
        self,
        parameter: float,
        solution: PeriodicOrbitSolution,
        multipliers: np.ndarray,
        mesh: np.ndarray | None = None,
        error: float | None = None,
    ) -> None:
        self._parameter = parameter
        self._solution = solution
        self._multipliers = multipliers
        self._mesh = mesh
        self._error = error

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
    def mesh(self) -> np.ndarray:
        """Positions of the solution's states on ``[0, 1]`` in rescaled time."""
        if self._mesh is None:
            return np.linspace(0.0, 1.0, self._solution.states.shape[0])
        return self._mesh

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
        Under the default Hermite-Simpson scheme the same branch stays under 0.1%
        throughout, and what is left is the orbit's: the monodromy itself walks
        the cubic Hermite interpolant between nodes and is fourth order (see
        :meth:`PeriodicOrbit.monodromy`), which took every smooth film from
        0.3-4% to 1e-5-1e-3 percent. Use :func:`resolved_branch` rather than a
        single sampled point to decide whether a branch's multipliers can be
        believed.

        Under Gauss-Legendre collocation the trivial multiplier stops being a
        measure, and the point carries its error explicitly instead. Those
        multipliers are read off the collocation Jacobian, which is the
        derivative of the discrete period map, and a one-step method's map is
        the flow of a nearby modified field to all orders in ``h`` - so the map
        keeps the trivial multiplier at one far beyond the scheme's order while
        the others carry the full ``O(h^(2m))``. On van der Pol at ``mu = 1``
        with ``m = 2`` the trivial drift fell 3.4e-4, 1.4e-9, 4e-16 from 20 to
        80 intervals while the nontrivial multiplier's error was 2.0e-3, 9.8e-5,
        6.0e-6. So for those points this is the larger of the trivial drift and
        the relative disagreement between the Jacobian's multipliers and the
        Hermite monodromy's - two independent readings of one cycle (see
        :func:`_multiplier_disagreement`). Against a converged reference it was
        above the nontrivial multiplier's true error on van der Pol for every
        ``m`` from 2 to 4 and 10 to 80 intervals, and on the Bogdanov-Takens
        orbit at b1 = -0.4554 in every resolved case (``m = 4`` on 80 intervals:
        1.9e-7 against 5.4e-10; on 20: 8.1e-5 against 1.3e-5). The one case
        under it is the badly resolved ``m = 2`` on 40 equidistributed intervals
        there, 1.0% against 1.9%: a factor two, where the trivial drift alone
        said 0.2%.
        """
        if self._error is not None:
            return self._error
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
      quadrature returned the multipliers to about 1e-4 (re-measured for DEQ-29;
      the 0.2% once quoted here did not reproduce).
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

    ``"gauss_legendre"`` is AUTO's orthogonal collocation, order ``2m`` with
    ``collocation_points`` (default 4) points to an interval - see
    :class:`_GaussLegendre` for the measured orders. Its multipliers come off its
    own Jacobian, and its points carry their error explicitly because the
    trivial multiplier of that monodromy no longer measures it (see
    :attr:`CyclePoint.floquet_error`). On 80 mesh intervals it takes the two
    films the fourth-order schemes left at 0.09% far under 0.01%: worst error
    over the branch 4.8e-7 on Bogdanov-Takens and 4.6e-5 on the ring, against
    8.9e-4 and 8.8e-4 under Hermite-Simpson on 80 nodes. On Hermite-Simpson's
    own unknowns - 20 intervals of four points are 81 states - Bogdanov-Takens
    reaches 8.0e-5 and the ring does not improve (2.4e-3: its period-35 wave
    has ``T h = 1.7`` per interval). It is not the default: a step costs about
    four times as much on 80 intervals, and with ``m`` times the unknowns one
    arclength step covers about ``1/sqrt(m)`` of the parameter, so a caller
    after the same span passes about ``sqrt(m)`` times the step - the figures
    above doubled it, and then reached further along both branches.

    Mesh equidistribution was built for it and measured, and is left out. The
    Russell-Christiansen monitor ``|x^(m+1)|^(1/(m+1))``, re-applied every step
    with the polynomial and the tangent carried onto the new mesh and the point
    re-solved there, graded the Bogdanov-Takens mesh by a factor of six. Over
    the stretch of each branch both runs covered, the worst error with ``m = 4``
    on 80 intervals went from 1.44e-7 to 1.43e-7 on Bogdanov-Takens and from
    3.7e-6 to 4.3e-6 on the ring; on 20 intervals from 5.7e-5 to 4.5e-5 on
    Bogdanov-Takens, ending a little sooner, and from 2.4e-3 to 1.6e-2 on the
    ring. Under trapezoidal the mesh could not help because the order was the
    limit; under this scheme it does not need to. The blocks take a graded mesh
    all the same, and the implementation is in the history (DEQ-34) for
    whoever finds an orbit that needs it.
    """

    __slots__ = [
        "_assembled",
        "_collocation",
        "_dimension",
        "_equation",
        "_intervals",
        "_mesh",
        "_parameter_index",
        "_phase_index",
        "_phase_value",
    ]

    def __init__(  # noqa: PLR0913 (scheme flags on the established signature)
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
        *,
        collocation: str = "hermite_simpson",
        collocation_points: int | None = None,
    ) -> None:
        if collocation not in _SCHEMES:
            message = f"collocation must be one of {_SCHEMES}, got {collocation!r}"
            raise ValueError(message)
        if collocation == _GaussLegendre.name:
            points = _GAUSS_POINTS if collocation_points is None else collocation_points
            if points < 1:
                message = f"collocation_points must be positive, got {points}"
                raise ValueError(message)
            self._collocation: _Collocation = _GaussLegendre(points)
        else:
            if collocation_points is not None:
                message = (
                    f"collocation_points applies to {_GaussLegendre.name!r} only; "
                    f"{collocation!r} has one point per interval"
                )
                raise ValueError(message)
            self._collocation = _COLLOCATIONS[collocation]
        self._equation = equation
        self._parameter_index = parameter_index
        self._intervals = intervals
        self._phase_index = phase_index
        self._phase_value = phase_value
        self._mesh = np.linspace(0.0, 1.0, intervals + 1)
        self._dimension = 0
        # The last Jacobian blocks assembled and the unknowns they were taken
        # at, so the multipliers of a point can reuse the tangent solve's.
        self._assembled: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def collocation(self) -> str:
        """The scheme: ``hermite_simpson``, ``trapezoidal`` or ``gauss_legendre``."""
        return self._collocation.name

    @property
    def collocation_points(self) -> int:
        """Collocation points per interval: ``m`` for Gauss-Legendre, else one."""
        return self._collocation.width

    def _step(self) -> float | np.ndarray:
        """The interval widths the scheme is handed.

        A scalar for the one-wide schemes, exactly as before Gauss-Legendre
        existed, so that their arithmetic - and every film built on them - is the
        same to the last bit; the widths themselves for Gauss-Legendre.
        """
        if self._collocation.width == 1:
            return 1.0 / self._intervals
        return np.diff(self._mesh)

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
            self._step(),
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

        Every collocation makes this bordered almost-block-diagonal: a band from
        the interval blocks, one dense column for the period and another for the
        parameter, the periodicity rows coupling the first and last nodes, and the
        phase row. Under trapezoidal collocation interval ``i`` contributes
        ``-I - w Df(x_i)`` against node ``i`` and ``I - w Df(x_{i+1})`` against node
        ``i+1``; Hermite-Simpson adds the midpoint's chain-rule terms to the same
        four blocks (see :class:`_HermiteSimpson`), so the scheme supplies the
        values and this method owns the pattern. Gauss-Legendre widens the block
        to ``m`` rows of equations against ``m + 1`` points, of which only the
        last is shared with the next interval; the index arithmetic is the same
        with ``width`` points to an interval, and at one it is the old pattern.

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

        block, period_values, parameter_values = self._collocation.interval_blocks(
            self._vector_field(),
            states,
            period,
            parameter,
            self._step(),
        )
        self._assembled = (unknowns.copy(), block)
        span = np.arange(dimension)
        height = self._collocation.width * dimension
        width = height + dimension
        # Interval i's equations and its first point both start at i * height.
        starts = np.arange(intervals) * height
        block_rows = np.broadcast_to(
            (starts[:, None] + np.arange(height))[:, :, None],
            (intervals, height, width),
        )
        block_columns = np.broadcast_to(
            (starts[:, None] + np.arange(width))[:, None, :],
            (intervals, height, width),
        )
        border_rows = starts[:, None] + np.arange(height)
        collocation_rows = intervals * height
        periodic = collocation_rows + span

        rows = np.concatenate(
            [
                block_rows.ravel(),
                border_rows.ravel(),
                border_rows.ravel(),
                periodic,
                periodic,
                [collocation_rows + dimension],
            ],
        )
        columns = np.concatenate(
            [
                block_columns.ravel(),
                np.full(collocation_rows, size - 2),
                np.full(collocation_rows, size - 1),
                collocation_rows + span,
                span,
                [self._phase_index],
            ],
        )
        values = np.concatenate(
            [
                block.ravel(),
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
                lambda u=unknowns, r=residual: self._numerical_jacobian(  # type: ignore[misc]
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
        mesh: np.ndarray | None = None,
    ) -> np.ndarray:
        """The Hermite monodromy's multipliers of the states (DEQ-29).

        Under the one-wide schemes this is exactly the monodromy the films have
        always used, forward-difference Jacobian and all. Under Gauss-Legendre it
        is the cross-check, and takes ``df/dx`` analytically where the field
        allows (:class:`_DifferentiatedOrbit`): the forward difference is a
        floor there. On the Bogdanov-Takens orbit at b1 = -0.4554 under ``m = 4``
        its trivial drift sat at 3.0e-5 on 80, 160 and 320 intervals alike, and
        with the analytic Jacobian falls 1.9e-7, 1.1e-8, 7.1e-10 - sixteen per
        doubling, the interpolant's own order.
        """
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        kind = (
            _DifferentiatedOrbit
            if self._collocation.jacobian_monodromy
            else PeriodicOrbit
        )
        orbit = kind(
            self._equation.derivative,
            states.shape[0] - 1,
            self._phase_index,
            self._phase_value,
        )
        if mesh is not None:
            orbit.set_mesh(mesh)
        return orbit.floquet_multipliers(PeriodicOrbitSolution(states, period))

    def hermite_multipliers(self, point: CyclePoint) -> np.ndarray:
        """The Hermite monodromy's multipliers of a traced point: the cross-check.

        For trapezoidal and Hermite-Simpson these are the multipliers the point
        already carries. For Gauss-Legendre, whose points carry the multipliers of
        its own Jacobian, this is the independent second opinion: the variational
        equation integrated along the cubic Hermite interpolant through all the
        point's states, interior ones included, fourth order whatever ``m`` is,
        so the two agree to the Hermite monodromy's own error.
        """
        return self._multipliers(
            point.solution.states,
            point.solution.period,
            point.parameter,
            point.mesh,
        )

    def _jacobian_multipliers(self, unknowns: np.ndarray, nodes: int) -> np.ndarray:
        """Floquet multipliers off the collocation Jacobian at ``unknowns``.

        The blocks are the ones the tangent solve just assembled at this very
        point whenever :meth:`trace` asks, and are only rebuilt otherwise.
        """
        assembled = self._assembled
        if assembled is None or not np.array_equal(assembled[0], unknowns):
            self._blocks(unknowns, nodes)
            assembled = self._assembled
        if assembled is None:  # pragma: no cover (``_blocks`` always sets it)
            message = "the collocation Jacobian was not assembled"
            raise RuntimeError(message)
        monodromy = _jacobian_monodromy(assembled[1], self._dimension)
        return np.linalg.eigvals(monodromy)

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
        self._equation.derivative.parameters[self._parameter_index].value = (
            seed.parameter
        )
        solved = orbit.solve(seed.states, seed.period)
        if solved is None:
            return None
        states = self._filled(solved, seed.parameter)
        unknowns = np.concatenate(
            [states.flatten(), [solved.period], [seed.parameter]],
        )
        fixed_parameter = np.zeros(unknowns.size)
        fixed_parameter[-1] = 1.0
        return self._correct(unknowns, nodes, _Arc(unknowns, fixed_parameter, 0.0))

    def _filled(self, solved: PeriodicOrbitSolution, parameter: float) -> np.ndarray:
        """Mesh states with each interval's interior points put back in.

        The trapezoidal seed has only the mesh points; a scheme wider than one
        also needs the ``m - 1`` points between them, taken off the cubic Hermite
        interpolant through the mesh states and their slopes ``T f(x)``, which is
        within the trapezoidal cycle's own ``O(h^2)`` of the curve the wider
        scheme will converge.
        """
        width = self._collocation.width
        if width == 1:
            return solved.states
        states = solved.states
        slopes = solved.period * np.array(
            [self._field(state, parameter) for state in states],
        )
        fractions = np.arange(width) / width
        step = 1.0 / self._intervals
        pieces = [
            _hermite_positions(states[i : i + 2], slopes[i : i + 2], step, fractions)
            for i in range(self._intervals)
        ]
        return np.vstack([*pieces, states[-1:]])

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
        self._mesh = np.linspace(0.0, 1.0, self._intervals + 1)
        nodes = self._intervals * self._collocation.width + 1
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
        if not self._collocation.jacobian_monodromy:
            multipliers = self._multipliers(states, period, parameter)
            return CyclePoint(
                parameter,
                PeriodicOrbitSolution(states, period),
                multipliers,
            )
        # The point keeps every point of the solution, interior ones included,
        # with where each sits: the Hermite cross-check walks all of them.
        scheme = self._collocation
        positions = (
            scheme.grid(self._mesh)
            if isinstance(scheme, _GaussLegendre)
            else self._mesh.copy()
        )
        multipliers = self._jacobian_multipliers(unknowns, nodes)
        hermite = self._multipliers(states, period, parameter, positions)
        error = max(
            float(np.min(np.abs(multipliers - 1.0))),
            _multiplier_disagreement(multipliers, hermite),
        )
        return CyclePoint(
            parameter,
            PeriodicOrbitSolution(states, period),
            multipliers,
            positions,
            error,
        )
