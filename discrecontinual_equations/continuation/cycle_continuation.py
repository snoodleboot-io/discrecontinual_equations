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

The phase condition and the arclength norm are options. By default they are a
point (Poincare) condition ``x_0[i] = c`` and the plain Euclidean norm of the
whole unknown vector; ``phase="integral"`` and ``arclength="weighted"`` give
AUTO's integral phase condition and its ``L2`` arclength (Doedel, Keller and
Kernevez, IJBC 1, 1991), integrated by the active scheme's own quadrature.
"""

import warnings
from collections.abc import Callable, Sequence
from itertools import pairwise
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import SuperLU, splu

from discrecontinual_equations.continuation.batched_field import (
    field_jacobians,
    field_values,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
    _hermite_positions,
)
from discrecontinual_equations.continuation.periodic_schur import (
    PeriodicSchurError,
    periodic_eigenvalues,
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
# What a Gauss-Legendre point's gap to the same cycle on the doubled mesh is
# multiplied by to bound its own error. If doubling the mesh at least halves the
# error, then e_N <= |mu_N - mu_2N| + e_2N <= |mu_N - mu_2N| + e_N / 2, so
# e_N <= 2 |mu_N - mu_2N|. Any convergent order gives that once resolved; order
# 2m gives it with room to spare (the gap is then (1 - 4^-m) e_N, read 2/(1 -
# 4^-m) ~ 2.1 times high at m = 2). The fourth-order Hermite cross-check could
# not bound the error at m = 2, where both readings cancel, and over-read it by
# orders of magnitude at m = 4; since DEQ-55 it is not part of the estimate.
_DOUBLING_SAFETY = 2.0
# How far a solved seed's period may stray from the seed's own, either way,
# before it is refused as another cycle or none (see
# :meth:`CycleContinuation._plausible`).
_SEED_PERIOD_RATIO = 2.0
# The same for its size: the largest distance of a state from the mean state.
# A seed's size comes from the same integration or normal form as its period;
# a cycle many times smaller is a spurious one round an equilibrium (see
# :meth:`CycleContinuation._plausible`).
_SEED_EXTENT_RATIO = 2.0
# Imaginary parts this many machine epsilons per factor of a multiplier's size
# are the complex arithmetic's rounding, not a conjugate pair.
_REAL_TOLERANCE = 64.0 * float(np.finfo(float).eps)
# How a point's multipliers were read (:attr:`CyclePoint.multipliers_method`).
_HERMITE_MONODROMY = "hermite_monodromy"
_PERIODIC_QR = "periodic_qr"
_EXPLICIT_PRODUCT = "explicit_product"
_UNREAD = "none"
# The damped seed correction: as many iterations as the adaptive orbit solver
# allows, halving each step at most thirty times, the period kept above a tenth
# of the seed's - the floor :class:`RobustPeriodicOrbit` uses.
_DAMPED_ITERATIONS = 80
_BACKTRACK_STEPS = 30
_BACKTRACK_FACTOR = 0.5
_PERIOD_FLOOR_FRACTION = 0.1
# The adaptive step (DEQ-75): AUTO's ITNW, the iteration count its step rules
# are scaled to; the factor a failed or slow step shrinks by; and what is left
# of the arclength budget, in units of the base step, below which it is spent.
_NEWTON_LIMIT = 5
_STEP_DECREASE = 0.5
_BUDGET_SLACK = 1.0e-9


def _near(period: float, expected: float) -> bool:
    """A finite, positive period within ``_SEED_PERIOD_RATIO`` of ``expected``."""
    return bool(
        np.isfinite(period)
        and expected / _SEED_PERIOD_RATIO <= period <= expected * _SEED_PERIOD_RATIO,
    )


def _extent(states: np.ndarray) -> float:
    """The largest distance of a state from the mean state."""
    return float(np.max(np.linalg.norm(states - states.mean(axis=0), axis=1)))


class _VectorField:
    """``f``, ``df/dx`` and ``df/dp`` of the equation, at many states at once.

    Each takes ``(states, parameter)`` with one state to a row and returns one
    row per state; ``parameter_derivatives`` also takes the values at those
    states, which its difference starts from. Bundled so that a collocation
    scheme asks one object for whatever it needs at its nodes or midpoints, and
    so that the scheme itself holds no reference to the equation and stays a
    stateless singleton. Taking every state at once is what lets a field that
    accepts arrays be evaluated in one pass (DEQ-75, :mod:`.batched_field`).
    """

    __slots__ = ["jacobians", "parameter_derivatives", "values"]

    def __init__(
        self,
        values: Callable[[np.ndarray, float], np.ndarray],
        jacobians: Callable[[np.ndarray, float], np.ndarray],
        parameter_derivatives: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    ) -> None:
        self.values = values
        self.jacobians = jacobians
        self.parameter_derivatives = parameter_derivatives


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
        step: float | np.ndarray,
    ) -> np.ndarray:
        """The collocation residual, one row of ``dimension`` per interval."""
        raise NotImplementedError

    def blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
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

    def grid(self, mesh: np.ndarray) -> np.ndarray:
        """Where the solution's points sit on ``mesh``: ``width`` to an interval."""
        fractions = np.arange(self.width) / self.width
        inner = mesh[:-1, None] + fractions[None, :] * np.diff(mesh)[:, None]
        return np.append(inner.ravel(), mesh[-1])

    def weights(self, mesh: np.ndarray) -> np.ndarray:
        """Quadrature weights over ``[0, 1]`` for the solution's points.

        The trapezoidal rule on the nodes: the only points a one-wide scheme
        has. Hermite-Simpson's own Simpson rule would need the midpoints, which
        are not unknowns; and on a uniform periodic mesh the trapezoidal rule
        is spectrally accurate for a smooth periodic integrand anyway.
        """
        widths = np.diff(mesh)
        return _HALF * (np.append(widths, 0.0) + np.insert(widths, 0, 0.0))


class _Trapezoidal(_Collocation):
    """Second order: ``x_{i+1} - x_i = (hT/2)(f_i + f_{i+1})``."""

    name = "trapezoidal"

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> np.ndarray:
        fields = field.values(states, parameter)
        weight = _HALF * period * step
        return states[1:] - states[:-1] - weight * (fields[:-1] + fields[1:])

    def blocks(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        identity = np.eye(states.shape[1])
        weight = _HALF * period * step
        derivatives = field.jacobians(states, parameter)
        fields = field.values(states, parameter)
        parameter_fields = field.parameter_derivatives(states, parameter, fields)
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
        step: float | np.ndarray,
    ) -> np.ndarray:
        slope = period * step / _EIGHTH
        return _HALF * (states[:-1] + states[1:]) + slope * (fields[:-1] - fields[1:])

    def residual(
        self,
        field: _VectorField,
        states: np.ndarray,
        period: float,
        parameter: float,
        step: float | np.ndarray,
    ) -> np.ndarray:
        fields = field.values(states, parameter)
        midpoints = self._midpoints(states, fields, period, step)
        middles = field.values(midpoints, parameter)
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
        step: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        identity = np.eye(states.shape[1])
        fields = field.values(states, parameter)
        derivatives = field.jacobians(states, parameter)
        parameter_fields = field.parameter_derivatives(states, parameter, fields)
        midpoints = self._midpoints(states, fields, period, step)
        middles = field.values(midpoints, parameter)
        middle_derivatives = field.jacobians(midpoints, parameter)
        middle_parameter_fields = field.parameter_derivatives(
            midpoints,
            parameter,
            middles,
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

    The collocation is AUTO's; the phase condition and the arclength norm around
    it are AUTO's only on request (see :class:`CycleContinuation`).

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
        roots, gauss_weights = leggauss(points)
        gauss = _HALF * (roots + 1.0)
        powers = np.arange(points + 1)
        grid = np.linspace(0.0, 1.0, points + 1)
        # Column j of this holds the monomial coefficients of the j-th Lagrange
        # basis polynomial on the equally spaced grid.
        self._coefficients = np.linalg.inv(grid[:, None] ** powers[None, :])
        self._values = self.basis(gauss)
        self._derivatives = (
            powers[None, 1:] * gauss[:, None] ** powers[None, :-1]
        ) @ self._coefficients[1:]
        # Each of the interval's m + 1 points' weight in the Gauss rule on [0, 1]:
        # the integral of its Lagrange basis polynomial, exact since the basis
        # has degree m < 2m.
        self._quadrature = (_HALF * gauss_weights) @ self._values

    def weights(self, mesh: np.ndarray) -> np.ndarray:
        """Quadrature weights over ``[0, 1]`` for every point of the solution.

        Each interval integrates its own polynomial exactly by the Gauss rule,
        which is what Doedel's discretised integrals use (the ``omega_{j,i}``
        Lagrange quadrature coefficients); a mesh point shared by two intervals
        collects a weight from each.
        """
        widths = np.diff(mesh)
        total = np.zeros(widths.size * self.width + 1)
        for interval, width in enumerate(widths):
            start = interval * self.width
            total[start : start + self.width + 1] += width * self._quadrature
        return total

    def basis(self, fractions: np.ndarray) -> np.ndarray:
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
        flat = points.reshape(-1, points.shape[2])
        fields = field.values(flat, parameter).reshape(points.shape)
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
        flat = points.reshape(-1, dimension)
        flat_fields = field.values(flat, parameter)
        fields = flat_fields.reshape(points.shape)
        derivatives = field.jacobians(flat, parameter).reshape(
            (*points.shape, dimension),
        )
        parameter_fields = field.parameter_derivatives(
            flat,
            parameter,
            flat_fields,
        ).reshape(points.shape)
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


_COLLOCATIONS: dict[str, _Collocation] = {
    scheme.name: scheme for scheme in (_Trapezoidal(), _HermiteSimpson())
}
_SCHEMES = sorted([*_COLLOCATIONS, _GaussLegendre.name])
_PHASES = ("integral", "point")
_ARCLENGTHS = ("euclidean", "weighted")
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


def _transfer_matrices(block: np.ndarray, dimension: int) -> np.ndarray:
    """Each interval's transfer matrix, read off the collocation Jacobian.

    The state columns of each interval's block are the linearised collocation
    system: ``A x_left + W (interior, x_right) = 0`` for any perturbation that
    keeps the equations satisfied. Solving it for the interval's far end gives
    the discrete transfer matrix ``-(W^-1 A)`` restricted to its last rows
    (Fairgrieve and Jepson), shape ``(intervals, dimension, dimension)``.
    Nothing is integrated and no field is evaluated: the block is the one the
    last tangent solve assembled. A block too singular to solve raises
    :class:`numpy.linalg.LinAlgError`.
    """
    transfers = -np.linalg.solve(block[:, :, dimension:], block[:, :, :dimension])
    return transfers[:, -dimension:]


def _jacobian_monodromy(block: np.ndarray, dimension: int) -> np.ndarray:
    """The monodromy read off the collocation Jacobian (Fairgrieve and Jepson).

    The product round the mesh of :func:`_transfer_matrices`: the derivative of
    the discrete period map, which inherits the scheme's order. Formed
    explicitly, so its eigenvalues are accurate only relative to the largest;
    the multipliers come from :func:`_product_multipliers` instead, and this is
    kept as the plain reading tests compare against.
    """
    monodromy = np.eye(dimension)
    for transfer in _transfer_matrices(block, dimension):
        monodromy = transfer @ monodromy
    return monodromy


def _product_multipliers(transfers: np.ndarray) -> tuple[np.ndarray, str]:
    """Eigenvalues of the transfer product by periodic QR, never forming it.

    A multiplier far below the largest keeps its own relative accuracy this
    way (see :mod:`periodic_schur`): on van der Pol at ``mu = 5`` the explicit
    product read the 7.7e-38 multiplier as 1e-15 noise. Should the iteration
    not converge, the product's eigenvalues are the fallback, as before - with
    a :class:`RuntimeWarning`, since they are then accurate only relative to
    the largest. A spectrum whose imaginary parts are all at rounding level
    comes back real, as :func:`numpy.linalg.eigvals` would return it.

    Returns the multipliers and how they were read: ``"periodic_qr"`` or
    ``"explicit_product"`` (see :attr:`CyclePoint.multipliers_method`).
    """
    try:
        values = periodic_eigenvalues(transfers)
    except PeriodicSchurError as failure:
        warnings.warn(
            f"{failure}; the Floquet multipliers come from the explicit product, "
            "accurate only relative to the largest",
            RuntimeWarning,
            stacklevel=2,
        )
        product = np.eye(transfers.shape[1])
        for transfer in transfers:
            product = transfer @ product
        return np.linalg.eigvals(product), _EXPLICIT_PRODUCT
    rounding = _REAL_TOLERANCE * transfers.shape[0] * np.abs(values)
    if np.all(np.abs(values.imag) <= rounding):
        return values.real.copy(), _PERIODIC_QR
    return values, _PERIODIC_QR


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

    def _jacobians(self, states: np.ndarray) -> np.ndarray:
        batched = field_jacobians(self._function, states, self._jacobian)
        if batched is not None:
            return batched
        return np.array([self._jacobian(state) for state in states])


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

    __slots__ = [
        "_error",
        "_mesh",
        "_method",
        "_multipliers",
        "_parameter",
        "_solution",
    ]

    def __init__(  # noqa: PLR0913 (the point's readings, all optional)
        self,
        parameter: float,
        solution: PeriodicOrbitSolution,
        multipliers: np.ndarray,
        mesh: np.ndarray | None = None,
        error: float | None = None,
        method: str = _HERMITE_MONODROMY,
    ) -> None:
        self._parameter = parameter
        self._solution = solution
        self._multipliers = multipliers
        self._mesh = mesh
        self._error = error
        self._method = method

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
    def multipliers_method(self) -> str:
        """How :attr:`multipliers` were read.

        ``"hermite_monodromy"`` under the one-wide schemes; under
        Gauss-Legendre ``"periodic_qr"``, or ``"explicit_product"`` where the
        periodic QR iteration did not converge - those multipliers are accurate
        only relative to the largest, and a :class:`RuntimeWarning` said so -
        or ``"none"`` where an interval block could not be solved.
        """
        return self._method

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
        6.0e-6. That is asymptotic, not a rule: on the Brusselator at ``b = 3``
        the drift was the *larger* error, 2.9e-8 against 5.7e-12 with ``m = 4``
        on 80 intervals. Neither can stand for the other, so for those points
        this is the larger of two readings: the trivial drift, and twice the
        relative disagreement (see :func:`_multiplier_disagreement`) with the
        same cycle re-solved on twice the intervals (see ``_DOUBLING_SAFETY``
        for why twice). Both converge at the scheme's own order, so the
        estimate does too.

        Against DOP853 references with variational multipliers on the
        verifier's grid - van der Pol at ``mu`` 0.5, 2, 3, 4 and the
        Brusselator at ``b`` 2.5, 3, 4, ``m`` from 1 to 4 on 8 to 96
        intervals, 211 cycles - it never read below the true error, and its
        median over-read was 2.0 at every ``m``; 196 cycles were within 100
        times. The other 15 are cycles whose doubled-mesh solve failed (read
        infinite) or landed far off, with true errors 2.7e-3 to 1.1, all but
        one above the default gate anyway.
        On 1159 cycles of the fold-of-cycles normal form, against the exact
        multipliers, it read low only at ``m = 1`` on 3 and 4 intervals - up to
        11 times, on true errors of 5e-7 to 6e-6 - and no cycle above 2e-2
        passed anywhere. That is a measurement, not a proof: it holds wherever
        doubling the mesh at least halves the error.

        Until DEQ-55 a third reading was in the maximum: the disagreement with
        the fourth-order Hermite monodromy (:meth:`CycleContinuation.
        hermite_multipliers`, still there on request). Fourth order whatever
        ``m`` is, it over-read by a median of 5 at ``m = 3`` and 140 at
        ``m = 4`` - 1.6e4 times at worst on that grid - and falsely refused van
        der Pol at ``mu = 6`` with ``m = 4`` on 80 intervals (4.65e-2 against a
        true 6.25e-3). It was kept for one ``m = 1`` cycle on 3 intervals that
        read eight times low without it; that is the fold-branch case above,
        where the error it hides is under 1e-5.

        A point whose states have collapsed onto one another reads infinite
        under every scheme. Every collocation admits that degenerate solution -
        all states equal at period zero, or an equilibrium at any period - and
        an under-resolved seed can converge to it: van der Pol at ``mu = 2`` on
        8 intervals reaches period -4e-20 under the one-wide schemes and the
        origin at period -7.6e-4 under ``m = 4``. Its monodromy is the identity
        or within ``|T|`` of it, so both measures above read about zero there
        and would call it resolved.
        """
        if not self.amplitude > _NOISE_FLOOR * max(
            1.0,
            float(np.max(np.abs(self._solution.states))),
        ):
            return float("inf")
        if self._error is not None:
            return self._error
        return float(np.min(np.abs(self._multipliers - 1.0)))

    def resolved(self, tolerance: float = _FLOQUET_TOLERANCE) -> bool:
        """Whether this cycle's multipliers are accurate enough to be believed.

        The counterpart of :meth:`AdaptivePeriodicOrbit.continue_to_resolved` for a
        cycle, and far cheaper for the same reason ``floquet_error`` is: the period
        has to be re-solved on a doubled mesh before anyone can say how wrong it is,
        whereas the trivial multiplier's exact value is known in advance, so this
        measurement arrives with the point. Under the one-wide schemes it costs
        nothing more. Under Gauss-Legendre it costs a re-solve on the doubled
        mesh for every point, and it is conservative: no cycle measured passed
        with a true error above the tolerance (see :attr:`floquet_error`).
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
    find. Nothing in the residual can catch that, and the trivial multiplier can -
    under the one-wide schemes. Under Gauss-Legendre the trivial multiplier can sit
    at one however wrong the rest are, and what is gated on is the point's explicit
    error, a conservative estimate that may refuse more of a branch than its
    multipliers deserve (see :attr:`CyclePoint.floquet_error`).

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


def _step_limits(
    arclength: float,
    maximum: float | None,
    minimum: float | None,
) -> tuple[float, float] | None:
    """``(max, min)`` for the adaptive step, or ``None`` for the fixed one."""
    if maximum is None:
        if minimum is not None:
            message = "min_arclength applies to the adaptive step; pass max_arclength"
            raise ValueError(message)
        return None
    floor = arclength if minimum is None else minimum
    if not 0.0 < floor <= arclength <= maximum:
        message = (
            "need 0 < min_arclength <= arclength <= max_arclength, got "
            f"{floor!r}, {arclength!r}, {maximum!r}"
        )
        raise ValueError(message)
    return float(maximum), float(floor)


def _adapted_step(length: float, iterations: int) -> float:
    """The next step after a correction that took ``iterations`` Newton solves.

    AUTO-07p's ``ADPTDS`` (``src/mesh.f90``) with its recommended ``ITNW = 5``:
    double after at most one iteration, half as much again after two, a tenth
    more up to ``ITNW / 2`` (in integers, so never at ``ITNW = 5``), and halve
    at ``ITNW`` or more. MATCONT's rule is the same shape (manual 2019, section
    2.3: grow after fewer than a threshold of iterations, shrink on failure);
    Allgower and Georg (*Numerical Continuation Methods*, 1990, chapter 6)
    derive steplength rules from the corrector's contraction, which the
    iteration count stands in for here.
    """
    if iterations <= 1:
        return 2.0 * length
    if iterations == 2:  # noqa: PLR2004 (AUTO's table, entry by entry)
        return 1.5 * length
    if iterations <= _NEWTON_LIMIT // 2:
        return 1.1 * length
    if iterations >= _NEWTON_LIMIT:
        return _STEP_DECREASE * length
    return length


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
    bound over the branch 4.8e-7 on Bogdanov-Takens and 4.6e-5 on the ring,
    against 8.9e-4 and 8.8e-4 under Hermite-Simpson on 80 nodes. Those are the
    bound, the cross-check's fourth-order reading; on the worst
    Bogdanov-Takens orbit the true error against an exact reference is 1.3e-10
    where the bound says 1.9e-7. On Hermite-Simpson's own unknowns - 20
    intervals of four points are 81 states - the Bogdanov-Takens bound is
    8.0e-5 (true 1.3e-5 on that orbit) and the ring does not improve (2.4e-3,
    which there is the trivial drift itself: its period-35 wave has
    ``T h = 1.7`` per interval). These worst-error figures predate the
    doubled-mesh reading in the estimate; on the worst Bogdanov-Takens orbit it
    leaves them unchanged (1.87e-7 on 80 intervals, 8.06e-5 on 20), the
    Hermite term being the larger at ``m = 4``; DEQ-55 dropped that term, so
    the estimate there is now the doubled-mesh reading, about twice the true
    error. It is not the default: a step costs about four times as much on 80
    intervals, and the doubled-mesh reading more per point, and with ``m``
    times the unknowns one
    arclength step covers about ``1/sqrt(m)`` of the parameter, so a caller
    after the same span passes about ``sqrt(m)`` times the step - the figures
    above doubled it, and then reached further along both branches.

    ``phase`` and ``arclength`` (DEQ-56) choose what pins the cycle's phase
    and what measures a step. ``phase="point"``, the default, is
    ``x_0[phase_index] = phase_value``; it needs the orbit to cross that value
    transversally, and loses the branch where it stops: a circle sliding
    sideways past ``x = 0`` is followed to the tangency and then round an
    artificial fold back. ``phase="integral"`` is AUTO's
    ``int_0^1 <x - x_hat, x_hat'> dt = 0`` against the previous cycle, which
    needs no crossing and follows that circle on. ``arclength="euclidean"``,
    the default, is the norm of every unknown, so one step spans less of the
    parameter the more points a scheme has: on the Hopf branch at 24
    intervals 9.9e-3 under trapezoidal and Hermite-Simpson, 7.1e-3, 5.8e-3,
    5.1e-3 under Gauss-Legendre with ``m = 2, 3, 4``. ``"weighted"`` is
    AUTO's ``L2`` norm of the orbit plus the parameter (the period weighted
    zero, AUTO's default for cycles), and the same step spans 3.5959e-2 under
    all five to six figures. Both integrals use the scheme's own quadrature:
    the trapezoidal rule on the nodes for the one-wide schemes, the Gauss rule
    on each interval's polynomial for Gauss-Legendre. The defaults are kept
    because they reproduce every film to the last bit.

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
    whoever finds an orbit that needs it. (These comparisons are of the
    bound too.)
    """

    __slots__ = [
        "_arclength",
        "_assembled",
        "_collocation",
        "_dimension",
        "_equation",
        "_factored",
        "_intervals",
        "_iterations",
        "_mesh",
        "_parameter_index",
        "_phase",
        "_phase_index",
        "_phase_reference",
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
        phase: str = "point",
        arclength: str = "euclidean",
    ) -> None:
        if phase not in _PHASES:
            message = f"phase must be one of {list(_PHASES)}, got {phase!r}"
            raise ValueError(message)
        if arclength not in _ARCLENGTHS:
            message = f"arclength must be one of {list(_ARCLENGTHS)}, got {arclength!r}"
            raise ValueError(message)
        self._phase = phase
        self._arclength = arclength
        # The previous cycle the integral phase condition is taken against, as
        # (its states, the row of weights times its slope), both flattened.
        self._phase_reference: tuple[np.ndarray, np.ndarray] | None = None
        if collocation not in _SCHEMES:
            message = f"collocation must be one of {_SCHEMES}, got {collocation!r}"
            raise ValueError(message)
        if collocation == _GaussLegendre.name:
            points = _GAUSS_POINTS if collocation_points is None else collocation_points
            if (
                isinstance(points, bool)
                or not isinstance(points, Integral)
                or points < 1
            ):
                message = (
                    f"collocation_points must be a positive integer, got {points!r}"
                )
                raise ValueError(message)
            self._collocation: _Collocation = _GaussLegendre(int(points))
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
        # The last corrector iteration's factorisation and the border row it
        # was taken with, which the tangent solve reuses (see :meth:`_tangent`),
        # and how many Newton solves that correction took.
        self._factored: tuple[np.ndarray, SuperLU] | None = None
        self._iterations = 0

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

    def _values(self, states: np.ndarray, parameter: float) -> np.ndarray:
        """``f`` at every state: in one call where the field takes arrays.

        Otherwise a state at a time, exactly as before. Either way the same
        numbers to rounding: on some CPUs an array power rounds an ulp away
        from the scalar one (see :mod:`.batched_field`).
        """
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        batched = field_values(
            self._equation.derivative,
            states,
            lambda state: self._field(state, parameter),
        )
        if batched is not None:
            return batched
        return np.array([self._field(state, parameter) for state in states])

    def _state_jacobians(self, states: np.ndarray, parameter: float) -> np.ndarray:
        """``df/dx`` at every state: one dual-number pass where the field allows.

        A field the duals cannot pass falls back to :meth:`_state_jacobian` a
        state at a time - Taylor jets, or forward differences - as before.
        """
        self._equation.derivative.parameters[self._parameter_index].value = parameter
        batched = field_jacobians(
            self._equation.derivative,
            states,
            lambda state: self._state_jacobian(state, parameter),
        )
        if batched is not None:
            return batched
        return np.array([self._state_jacobian(state, parameter) for state in states])

    def _parameter_derivatives(
        self,
        states: np.ndarray,
        parameter: float,
        values: np.ndarray,
    ) -> np.ndarray:
        """``df/dp`` at every state, by a forward difference in the parameter.

        One extra field evaluation per state, against the whole augmented
        residual that a finite-difference column of the full Jacobian would have
        cost. ``values`` are ``f`` at the same states and parameter, which the
        difference starts from rather than evaluating them again.
        """
        return (self._values(states, parameter + _STEP) - values) / _STEP

    def _vector_field(self) -> _VectorField:
        return _VectorField(
            self._values,
            self._state_jacobians,
            self._parameter_derivatives,
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
                [self._phase_residual(states)],
            ],
        )

    def _phase_residual(self, states: np.ndarray) -> float:
        """The phase condition: a point condition, or AUTO's integral one.

        Integral: ``int_0^1 <x(t) - x_hat(t), x_hat'(t)> dt`` by the scheme's
        quadrature, where ``x_hat`` is the previous cycle and ``x_hat'`` its
        slope ``T f(x_hat)`` in rescaled time. It picks the phase that
        minimises the distance to ``x_hat`` (Doedel's derivation), so it needs
        no coordinate to cross any value. Taken against ``x - x_hat`` rather
        than ``x`` so that ``x_hat`` satisfies it exactly on the mesh too; the
        two differ by ``int <x_hat, x_hat'>``, zero for any periodic ``x_hat``.
        """
        if self._phase == "point":
            return float(states[0, self._phase_index] - self._phase_value)
        anchor, row = self._reference()
        return float(row @ (states.ravel() - anchor))

    def _reference(self) -> tuple[np.ndarray, np.ndarray]:
        if self._phase_reference is None:  # pragma: no cover (set before solves)
            message = "the integral phase condition has no reference cycle"
            raise RuntimeError(message)
        return self._phase_reference

    def _set_phase_reference(self, unknowns: np.ndarray, nodes: int) -> None:
        """Take the integral phase condition against the cycle in ``unknowns``."""
        if self._phase == "point":
            return
        dimension = self._dimension
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[nodes * dimension])
        parameter = float(unknowns[-1])
        slopes = period * self._values(states, parameter)
        weights = self._collocation.weights(self._mesh)
        row = (weights[:, None] * slopes).ravel()
        self._phase_reference = (states.ravel().copy(), row)

    def _norm_weights(self, size: int) -> np.ndarray | None:
        """The weighted arclength's inner product, as a diagonal; ``None`` if plain.

        Doedel's pseudo-arclength: ``int_0^1 <dx(t), x_dot(t)> dt`` by the
        scheme's quadrature, plus the parameter's term. The period's weight is
        zero, AUTO's default for periodic solutions (its ``THL = {11: 0}``),
        which keeps the step from being throttled near an infinite period.
        """
        if self._arclength == "euclidean":
            return None
        weights = self._collocation.weights(self._mesh)
        states = np.repeat(weights, self._dimension)
        diagonal = np.concatenate([states, [0.0], [1.0]])
        if diagonal.size != size:  # pragma: no cover (shapes are fixed by trace)
            message = f"{diagonal.size} weights for {size} unknowns"
            raise ValueError(message)
        return diagonal

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

        if self._phase == "point":
            phase_columns = np.array([self._phase_index])
            phase_values = np.array([1.0])
        else:
            phase_values = self._reference()[1]
            phase_columns = np.arange(phase_values.size)
        rows = np.concatenate(
            [
                block_rows.ravel(),
                border_rows.ravel(),
                border_rows.ravel(),
                periodic,
                periodic,
                np.full(phase_columns.size, collocation_rows + dimension),
            ],
        )
        columns = np.concatenate(
            [
                block_columns.ravel(),
                np.full(collocation_rows, size - 2),
                np.full(collocation_rows, size - 1),
                collocation_rows + span,
                span,
                phase_columns,
            ],
        )
        values = np.concatenate(
            [
                block.ravel(),
                period_values.ravel(),
                parameter_values.ravel(),
                np.ones(dimension),
                -np.ones(dimension),
                phase_values,
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

    def _solve_bordered(  # noqa: PLR0913 (the factorisation is kept on request)
        self,
        unknowns: np.ndarray,
        nodes: int,
        border: np.ndarray,
        rhs: np.ndarray,
        dense: Callable[[], np.ndarray],
        *,
        keep: bool = False,
    ) -> np.ndarray:
        """One sparse solve, falling back to a dense least squares.

        Where the Jacobian is singular the LU cannot be formed, or yields a
        non-finite vector, so the result is checked before it is trusted - the
        same guard the periodic orbit solver uses.

        ``dense`` is a callable and not an array on purpose. Building the
        finite-difference system costs a whole Newton step's worth of field
        evaluations, which is the expense this class exists to avoid; passing it
        eagerly would pay that cost on every step and leave the sparse path
        saving nothing at all.

        The LU is SuperLU's, factored once and kept when ``keep`` asks, so a
        later solve with the same matrix costs two triangular sweeps. It gives
        the same vector :func:`scipy.sparse.linalg.spsolve` did to the last bit
        (both are SuperLU with the same column ordering), which the corrector
        used before DEQ-75.
        """
        sparse = self._bordered(unknowns, nodes, border).tocsc()
        try:
            factor = splu(sparse)
        except RuntimeError:  # exactly singular: no LU to solve with
            factor = None
        if factor is not None:
            update = factor.solve(rhs)
            if np.all(np.isfinite(update)):
                if keep:
                    self._factored = (border.copy(), factor)
                return update
        fallback, *_ = np.linalg.lstsq(dense(), rhs, rcond=None)
        return fallback

    def _correct(
        self,
        prediction: np.ndarray,
        nodes: int,
        arc: _Arc,
    ) -> np.ndarray | None:
        """Newton on the arclength-augmented system from ``prediction``.

        Keeps the last iteration's factorisation for :meth:`_tangent`, and
        counts its solves in ``_iterations`` for the step control.
        """
        unknowns = prediction
        self._factored = None
        self._iterations = 0
        for _ in range(_ITERATIONS):
            residual = self._augmented(unknowns, nodes, arc)
            if np.linalg.norm(residual) < _TOLERANCE:
                return unknowns
            if not np.all(np.isfinite(residual)):
                # Diverged past the floating-point range. Nothing converges
                # from here, and the dense fallback would spend a
                # finite-difference Jacobian - seconds at 160 intervals -
                # finding that out, or raise.
                return None
            self._factored = None
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
                keep=True,
            )
            self._iterations += 1
            unknowns = unknowns + update
        residual = self._augmented(unknowns, nodes, arc)
        return unknowns if np.linalg.norm(residual) < _TOLERANCE else None

    def _tangent(
        self,
        unknowns: np.ndarray,
        nodes: int,
        previous: np.ndarray,
    ) -> np.ndarray:
        """The unit tangent at ``unknowns``, pointing the way ``previous`` did.

        It solves ``[J; b^T] t = e_last`` with ``J`` the cycle Jacobian and ``b``
        the previous tangent in the arclength's inner product - the very matrix
        the corrector's last iteration factored, except that the corrector took
        ``J`` one Newton update earlier. Where :meth:`_correct` left that
        factorisation behind (``_factored``, with the same border), it is
        reused (DEQ-75), as MATCONT's Moore-Penrose corrector takes its
        tangent off the corrector's own factorisation (MATCONT manual, 2019,
        section 2.2.2); AUTO goes further and takes the secant
        ``(u - u_old) / ds`` (``CONTBV`` in AUTO-07p's ``bvp.f90``). The
        tangent is then off by the order of that last update. It is only a
        direction to predict along and a plane to correct in, so the points
        stay on the branch to the corrector's tolerance, but they are not the
        same points: on the films the next point moved by up to ``1e-4`` in
        the parameter, the spacing of the branch and not its geometry. That is
        why :meth:`trace` reuses it only under the adaptive step, which moves
        the points anyway, and keeps the fixed step's tangent as it always was:
        rebuilt at the point, the same matrix and solve as before. Rebuilt and
        refactored where the corrector did not iterate, and under
        Gauss-Legendre, whose multipliers need the blocks at this very point
        anyway (see :meth:`_jacobian_multipliers`).
        """
        rhs = np.zeros(unknowns.size)
        rhs[-1] = 1.0
        weights = self._norm_weights(unknowns.size)
        border = previous if weights is None else weights * previous
        tangent = None
        factored = self._factored
        if (
            factored is not None
            and not self._collocation.jacobian_monodromy
            and np.array_equal(factored[0], border)
        ):
            reused = factored[1].solve(rhs)
            if np.all(np.isfinite(reused)):
                tangent = reused
        if tangent is None:
            tangent = self._solve_bordered(
                unknowns,
                nodes,
                border,
                rhs,
                lambda: self._dense_tangent_system(unknowns, nodes, border),
            )
        if weights is None:
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = tangent / np.sqrt(tangent @ (weights * tangent))
        if tangent @ border < 0.0:
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
        so the two agree to the Hermite monodromy's own error (3.86-3.99 per
        doubling on the Hopf cycle) and no further: agreement to the
        collocation order, which DEQ-34 asked for, is not something a
        fourth-order cross-check can show.
        """
        return self._multipliers(
            point.solution.states,
            point.solution.period,
            point.parameter,
            point.mesh,
        )

    def _jacobian_multipliers(
        self,
        unknowns: np.ndarray,
        nodes: int,
    ) -> tuple[np.ndarray, str]:
        """Floquet multipliers off the collocation Jacobian at ``unknowns``.

        The blocks are the ones the tangent solve just assembled at this very
        point whenever :meth:`trace` asks, and are only rebuilt otherwise.
        Returned with how they were read (:func:`_product_multipliers`), or
        ``nan`` and ``"none"`` where a block could not be solved.
        """
        assembled = self._assembled
        if assembled is None or not np.array_equal(assembled[0], unknowns):
            self._blocks(unknowns, nodes)
            assembled = self._assembled
        if assembled is None:  # pragma: no cover (``_blocks`` always sets it)
            message = "the collocation Jacobian was not assembled"
            raise RuntimeError(message)
        try:
            transfers = _transfer_matrices(assembled[1], self._dimension)
        except np.linalg.LinAlgError:
            return np.full(self._dimension, np.nan), _UNREAD
        if not np.all(np.isfinite(transfers)):
            return np.full(self._dimension, np.nan), _UNREAD
        return _product_multipliers(transfers)

    def _doubled_multipliers(
        self,
        states: np.ndarray,
        period: float,
        parameter: float,
    ) -> np.ndarray:
        """The same cycle's multipliers re-solved on twice the intervals.

        The Gauss-Legendre error estimate's second reading (see
        :data:`_DOUBLING_SAFETY`). Each interval's polynomial is evaluated at
        the doubled mesh's points, which puts the start within the coarse
        solution's own error of the fine one, and the cycle is corrected there
        with the parameter held fixed, as :meth:`_solve_seed` does. A fine
        solve that fails returns ``nan``, which the caller's disagreement turns
        into an infinite error, so the point is refused rather than believed.
        """
        width = self._collocation.width
        if not isinstance(self._collocation, _GaussLegendre):
            message = "the doubled-mesh reading is for Gauss-Legendre only"
            raise TypeError(message)
        basis = self._collocation.basis(np.arange(2 * width) / (2 * width))
        pieces = [
            basis @ states[i * width : (i + 1) * width + 1]
            for i in range(self._intervals)
        ]
        fine = CycleContinuation(
            self._equation,
            self._parameter_index,
            2 * self._intervals,
            self._phase_index,
            self._phase_value,
            collocation=_GaussLegendre.name,
            collocation_points=width,
            phase=self._phase,
        )
        fine._dimension = self._dimension
        unknowns = np.concatenate(
            [np.vstack([*pieces, states[-1:]]).ravel(), [period], [parameter]],
        )
        nodes = 2 * self._intervals * width + 1
        fine._set_phase_reference(unknowns, nodes)
        fixed_parameter = np.zeros(unknowns.size)
        fixed_parameter[-1] = 1.0
        corrected = fine._correct(unknowns, nodes, _Arc(unknowns, fixed_parameter, 0.0))
        if corrected is None:
            return np.full(self._dimension, np.nan)
        return fine._jacobian_multipliers(corrected, nodes)[0]

    def _solve_seed(self, seed: CycleSeed, nodes: int) -> np.ndarray | None:
        """The first point on the branch, or ``None`` if no believable one is found.

        The staged solve (:meth:`_staged_seed`) is tried first, exactly as it
        always ran, so every seed it already started is started the same way to
        the last bit. Its answer is kept only if it is a cycle: a positive
        period within a factor of two of the seed's, and states that have not
        collapsed onto one point (see :meth:`_plausible`). Otherwise the target
        scheme is corrected directly from the seed with a damped Newton
        (:meth:`_direct_seed`), and that answer faces the same test.

        A stiff cycle is what needs the second path. On van der Pol at
        ``mu = 5`` and ``10``, seeded from an accurate integration, the
        trapezoidal stage lands on cycles of period -1328 or 13.8 against the
        true 11.6, or fails, and Gauss-Legendre Newton cannot recover from
        those; started from the integration itself it converges. A seed path
        that finds nothing believable returns ``None`` rather than hand a
        negative period, or a point, to the continuation.
        """
        if not (np.isfinite(seed.period) and seed.period > 0.0):
            return None
        if self._phase == "integral":
            # Until there is a previous cycle, the seed is the reference: the
            # first point takes the phase nearest the guess it was given.
            self._equation.derivative.parameters[self._parameter_index].value = (
                seed.parameter
            )
            filled = self._filled(
                PeriodicOrbitSolution(
                    np.asarray(seed.states, dtype=float),
                    seed.period,
                ),
                seed.parameter,
            )
            self._set_phase_reference(
                np.concatenate([filled.ravel(), [seed.period], [seed.parameter]]),
                nodes,
            )
        try:
            staged = self._staged_seed(seed, nodes)
        except np.linalg.LinAlgError:
            # The trapezoidal stage handed on an iterate so far off that the
            # dense fallback could not factor it: no cycle there either.
            staged = None
        size = _extent(np.asarray(seed.states, dtype=float))
        if staged is not None and self._plausible(
            staged,
            nodes,
            seed.period,
            extent=size,
        ):
            return staged
        direct = self._direct_seed(seed, nodes)
        if direct is not None and self._plausible(
            direct,
            nodes,
            seed.period,
            extent=size,
        ):
            return direct
        return None

    def _plausible(
        self,
        unknowns: np.ndarray,
        nodes: int,
        expected: float,
        *,
        extent: float | None = None,
    ) -> bool:
        """Whether a solved seed is a cycle near the one the seed described.

        The period must be finite, positive and within ``_SEED_PERIOD_RATIO``
        of the seed's, either way: a seed's period comes from an integration or
        a normal form and is never that far out, while every collocation
        admits the collapsed solution at period zero or below. The states must
        also span more than :data:`_NOISE_FLOOR` of their size, which an
        equilibrium at a sensible period does not, and - given the seed's own
        ``extent`` - be within ``_SEED_EXTENT_RATIO`` of it either way. A
        coarse mesh round a degenerate equilibrium has spurious small cycles
        at a sensible period: the fold of cycles at ``mu = 0`` on 3 intervals
        with ``m = 1`` took one of radius 4e-4 for a seed of radius 1, and its
        multipliers, both about one, let it pass :meth:`CyclePoint.resolved`.
        """
        dimension = self._dimension
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        if not _near(float(unknowns[nodes * dimension]), expected):
            return False
        size = _extent(states)
        if size <= _NOISE_FLOOR * max(1.0, float(np.max(np.abs(states)))):
            return False
        return extent is None or (
            extent / _SEED_EXTENT_RATIO <= size <= extent * _SEED_EXTENT_RATIO
        )

    def _direct_seed(self, seed: CycleSeed, nodes: int) -> np.ndarray | None:
        """The seed corrected on the target scheme itself, with a damped Newton.

        The first of the approaches DEQ-54 lists: when the seed is already
        accurate - an integrated orbit - nothing is gained by detouring through
        a coarser scheme whose own cycle may be far from it. The interior
        points a wider scheme needs come off the seed's cubic Hermite
        interpolant, as in :meth:`_filled`; the parameter is held fixed.
        """
        self._equation.derivative.parameters[self._parameter_index].value = (
            seed.parameter
        )
        states = self._filled(
            PeriodicOrbitSolution(np.asarray(seed.states, dtype=float), seed.period),
            seed.parameter,
        )
        unknowns = np.concatenate(
            [states.flatten(), [float(seed.period)], [seed.parameter]],
        )
        fixed_parameter = np.zeros(unknowns.size)
        fixed_parameter[-1] = 1.0
        return self._damped_correct(
            unknowns,
            nodes,
            _Arc(unknowns, fixed_parameter, 0.0),
            _PERIOD_FLOOR_FRACTION * float(seed.period),
        )

    def _damped_correct(
        self,
        prediction: np.ndarray,
        nodes: int,
        arc: _Arc,
        floor: float,
    ) -> np.ndarray | None:
        """Newton with backtracking on the residual norm and a floor on the period.

        The same globalisation :class:`AdaptivePeriodicOrbit` uses: a step is
        halved until it lowers the residual norm without taking the period
        under the floor, and the iteration stops if no halving does.
        """
        unknowns = prediction
        period_index = nodes * self._dimension
        for _ in range(_DAMPED_ITERATIONS):
            residual = self._augmented(unknowns, nodes, arc)
            norm = float(np.linalg.norm(residual))
            if norm < _TOLERANCE:
                return unknowns
            try:
                update = self._solve_bordered(
                    unknowns,
                    nodes,
                    arc.tangent,
                    -residual,
                    lambda u=unknowns, r=residual: self._numerical_jacobian(  # type: ignore[misc]
                        u,
                        nodes,
                        arc,
                        r,
                    ),
                )
            except np.linalg.LinAlgError:
                # The dense fallback met a matrix with no finite solution: an
                # iterate the field cannot be evaluated near. Not a cycle.
                return None
            if not np.all(np.isfinite(update)):
                return None
            scale = 1.0
            for _ in range(_BACKTRACK_STEPS):
                candidate = unknowns + scale * update
                if candidate[period_index] > floor and (
                    float(np.linalg.norm(self._augmented(candidate, nodes, arc))) < norm
                ):
                    break
                scale *= _BACKTRACK_FACTOR
            else:
                return None
            unknowns = candidate
        residual = self._augmented(unknowns, nodes, arc)
        return unknowns if np.linalg.norm(residual) < _TOLERANCE else None

    def _staged_seed(self, seed: CycleSeed, nodes: int) -> np.ndarray | None:
        """The seed solved at its own parameter, trapezoidal first.

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
        Both stages are sparse and cost ``O(N)`` field calls per Newton
        iteration (DEQ-49): the orbit solver differences its Jacobian a column
        group at a time, and the correction uses the analytic blocks. On van
        der Pol at 160 nodes the seed takes no longer than one continuation
        step.
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
        if solved is None or not _near(solved.period, seed.period):
            # Not handed on: the wider scheme's Newton cannot recover from
            # another cycle, and spends its dense fallback finding that out.
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
        slopes = solved.period * self._values(states, parameter)
        fractions = np.arange(width) / width
        step = 1.0 / self._intervals
        pieces = [
            _hermite_positions(states[i : i + 2], slopes[i : i + 2], step, fractions)
            for i in range(self._intervals)
        ]
        return np.vstack([*pieces, states[-1:]])

    def trace(  # noqa: PLR0913 (step limits beside the established signature)
        self,
        seed: CycleSeed,
        arclength: float,
        steps: int,
        direction: float = 1.0,
        *,
        max_arclength: float | None = None,
        min_arclength: float | None = None,
    ) -> tuple[list[CyclePoint], list[CycleBifurcation]]:
        """Trace the cycle branch, returning its points and detected bifurcations.

        ``direction`` sets the initial parameter sense (+1 increasing, -1 decreasing);
        the branch then follows its own tangent and may round a turning point.

        By default every step is ``arclength`` long and the trace takes
        ``steps`` of them, stopping at the first the corrector cannot close.
        ``max_arclength`` turns on AUTO's step control instead (DEQ-75, see
        :func:`_adapted_step`): the step starts at ``arclength``, doubles while
        Newton converges in one iteration, grows by half at two, is halved at
        five or more, and never exceeds ``max_arclength``; a step that fails is
        halved and retried, down to ``min_arclength`` (default ``arclength``,
        so that the branch ends where the fixed step would have ended it).
        ``steps * arclength`` is then the arclength budget, the distance the
        fixed-step trace covers, and the last step is shortened to land on it;
        ``steps`` still caps the number of points.

        The adaptive step also takes each tangent off the corrector's last
        factorisation instead of a fresh one (see :meth:`_tangent`), one LU per
        step fewer.

        A crossing the step control would have stepped over is not: a step
        longer than ``arclength`` across which :func:`classify_transition`
        finds a bifurcation is discarded and retaken at ``arclength``, so the
        crossing is bracketed by points as close as the fixed step put them
        and located as accurately. Tracing at a few times the step everywhere
        else is where the time goes; see :meth:`trace`'s use in the films.
        """
        self._dimension = seed.states.shape[1]
        self._mesh = np.linspace(0.0, 1.0, self._intervals + 1)
        nodes = self._intervals * self._collocation.width + 1
        limits = _step_limits(arclength, max_arclength, min_arclength)
        unknowns = self._solve_seed(seed, nodes)
        if unknowns is None:
            return [], []
        self._set_phase_reference(unknowns, nodes)
        seed_direction = np.zeros(unknowns.size)
        seed_direction[-1] = direction
        # The seed's own correction held the parameter fixed; its factorisation
        # is not the tangent system's, whatever its border reads.
        self._factored = None
        tangent = self._tangent(unknowns, nodes, seed_direction)
        weights = self._norm_weights(unknowns.size)
        points = [self._point(unknowns, nodes)]
        bifurcations: list[CycleBifurcation] = []
        budget = steps * arclength
        travelled = 0.0
        length = arclength
        while len(points) <= steps:
            step = arclength if limits is None else min(length, budget - travelled)
            if step <= _BUDGET_SLACK * arclength:
                break
            prediction = unknowns + step * tangent
            # The constraint's row is the tangent in the arclength's own inner
            # product, so it reads ``step`` at the prediction either way.
            row = tangent if weights is None else weights * tangent
            corrected = self._correct(prediction, nodes, _Arc(unknowns, row, step))
            if corrected is None:
                if limits is None or length * _STEP_DECREASE < limits[1]:
                    break
                length *= _STEP_DECREASE
                continue
            iterations = self._iterations
            reference = self._phase_reference
            self._set_phase_reference(corrected, nodes)
            if limits is None:
                # The fixed step keeps the tangent rebuilt at the point, so
                # its points are those it always gave, up to the batched
                # fields' rounding (:mod:`.batched_field`).
                self._factored = None
            following = self._tangent(corrected, nodes, tangent)
            point = self._point(corrected, nodes)
            transition = classify_transition(points[-1], point)
            if transition is not None and step > arclength:
                self._phase_reference = reference
                length = arclength
                continue
            tangent = following
            unknowns = corrected
            if transition is not None:
                bifurcations.append(transition)
            points.append(point)
            travelled += step
            if limits is not None:
                length = min(
                    max(_adapted_step(length, iterations), limits[1]),
                    limits[0],
                )
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
        # with where each sits: the Hermite cross-check, asked for, walks them.
        positions = self._collocation.grid(self._mesh)
        multipliers, method = self._jacobian_multipliers(unknowns, nodes)
        if np.all(np.isfinite(multipliers)):
            doubled = self._doubled_multipliers(states, period, parameter)
            refinement = (
                _DOUBLING_SAFETY * _multiplier_disagreement(multipliers, doubled)
                if np.all(np.isfinite(doubled))
                else float("inf")
            )
            error = max(float(np.min(np.abs(multipliers - 1.0))), refinement)
        else:
            # An interval block too singular to solve: no multipliers to read,
            # so the point is reported and refused rather than the trace lost.
            error = float("inf")
        return CyclePoint(
            parameter,
            PeriodicOrbitSolution(states, period),
            multipliers,
            positions,
            error,
            method,
        )
