"""Periodic normal form coefficients of the codimension-one cycle bifurcations.

At a fold, a period-doubling or a Neimark-Sacker bifurcation of a limit cycle
the cycle's critical centre manifold carries Iooss's *periodic* normal form
(Kuznetsov, Govaerts, Doedel and Dhooge, SIAM J. Numer. Anal. 43, 2005; the
formulas as in Kuznetsov's lecture notes "Numerical periodic normalization for
codim 1 bifurcations of limit cycles", Utrecht, 2014)::

    fold   tau' = 1 - xi + a xi^2,          xi' = b xi^2
    flip   tau' = 1 + a xi^2,               xi' = c xi^3       (tau in [0, 2T])
    torus  tau' = 1 + a |xi|^2,             xi' = (i theta / T) xi + d xi |xi|^2

``b`` decides which side of the fold the two cycles live on, ``c < 0`` makes
a flip supercritical and ``Re d < 0`` a Neimark-Sacker one; each vanishes at
the codimension-two point that ends a curve's nondegenerate stretch (cusp,
generalised flip, Chenciner). Each is a Fredholm solvability condition on
linear periodic boundary-value problems along the critical cycle - nothing
about the Poincare map is used - and every one of those problems is solved
here by the same fourth-order Hermite-Simpson collocation the cycle
continuation uses by default, on the cycle's own uniform mesh. The multilinear
forms ``B`` and ``C`` come from Taylor jets (:mod:`.derivative_provider`), so
the field must be one automatic differentiation can carry.

Everything is in the cycle's own time ``t in [0, T]``; ``<u, v> = conj(u) . v``
and the integrals are Simpson's rule on the nodes and the Hermite midpoints.
"""

from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.function import Function

_AUTODIFF = AutomaticDifferentiation()
_HALF = 0.5
_SIXTH = 6.0
_EIGHTH = 8.0
_MIDPOINT_WEIGHT = 4.0
_SECOND = 2
_THIRD = 3
# The bordering vector of a singular problem: any fixed vector off its range.
_BORDER_SEED = 35


class _Profile:
    """A function along the cycle: its values at the nodes and the midpoints."""

    __slots__ = ["middles", "nodes"]

    def __init__(self, nodes: np.ndarray, middles: np.ndarray) -> None:
        self.nodes = nodes
        self.middles = middles

    def __add__(self, other: "_Profile") -> "_Profile":
        return _Profile(self.nodes + other.nodes, self.middles + other.middles)

    def __sub__(self, other: "_Profile") -> "_Profile":
        return _Profile(self.nodes - other.nodes, self.middles - other.middles)

    def scaled(self, factor: complex) -> "_Profile":
        """This profile times ``factor``."""
        return _Profile(factor * self.nodes, factor * self.middles)

    def conjugate(self) -> "_Profile":
        """The complex conjugate profile."""
        return _Profile(np.conj(self.nodes), np.conj(self.middles))

    def apply(self, matrices: "_Profile") -> "_Profile":
        """Pointwise matrix-vector product with a profile of matrices."""
        return _Profile(
            np.einsum("nij,nj->ni", matrices.nodes, self.nodes),
            np.einsum("nij,nj->ni", matrices.middles, self.middles),
        )


def _bordered(matrix: csc_matrix, right: np.ndarray, left: np.ndarray):
    """The LU of ``[[L, left], [right^T, 0]]``."""
    size = matrix.shape[0]
    full = coo_matrix(matrix)
    rows = np.concatenate([full.row, np.arange(size), np.full(size, size)])
    columns = np.concatenate([full.col, np.full(size, size), np.arange(size)])
    values = np.concatenate([full.data, left, right])
    return splu(
        coo_matrix((values, (rows, columns)), shape=(size + 1, size + 1)).tocsc(),
    )


def _bordered_solve(
    matrix: csc_matrix,
    rhs: np.ndarray,
    *,
    null: bool,
) -> np.ndarray:
    """A problem singular to the discretisation's error, bordered by its null vectors.

    A first bordering by a fixed vector finds the near-null vectors of ``L`` and
    ``L^T``; the second borders by those, the well-conditioned choice: the
    solution has no component along the right null vector, and what the
    discrete Fredholm condition misses goes into the corner, along the left
    one. ``null`` asks for the null vector itself, scaled to unit size.
    """
    size = matrix.shape[0]
    seed = np.random.default_rng(_BORDER_SEED).standard_normal(size).astype(complex)
    first = _bordered(matrix, seed, seed)
    corner = np.zeros(size + 1, dtype=complex)
    corner[size] = 1.0
    right = first.solve(corner)[:size]
    left = first.solve(corner, trans="T")[:size]
    right /= np.linalg.norm(right)
    left /= np.linalg.norm(left)
    second = _bordered(matrix, np.conj(right), np.conj(left))
    if null:
        return right
    return second.solve(np.concatenate([rhs, [0.0]]))


class CycleNormalForm:
    """The critical cycle and the tools its normal form coefficients share.

    ``states`` are the cycle's ``N + 1`` nodes on a uniform mesh of one period
    (first equal to last), ``period`` its period, and ``function`` the vector
    field with its parameters already at the bifurcation values.
    """

    __slots__ = [
        "_field",
        "_function",
        "_jacobians",
        "_orbit",
        "_period",
        "_step",
    ]

    def __init__(
        self,
        function: Function,
        states: np.ndarray,
        period: float,
    ) -> None:
        self._function = function
        self._period = float(period)
        intervals = states.shape[0] - 1
        self._step = self._period / intervals
        nodes = np.asarray(states, dtype=float)
        fields = np.array([self._value(state) for state in nodes])
        # The Hermite midpoints of the orbit, where Simpson's rule and the
        # collocation take their middle values.
        middles = _HALF * (nodes[:-1] + nodes[1:]) + (self._step / _EIGHTH) * (
            fields[:-1] - fields[1:]
        )
        self._orbit = _Profile(nodes, middles)
        self._field = _Profile(
            fields,
            np.array([self._value(state) for state in middles]),
        )
        self._jacobians = _Profile(
            np.array([_AUTODIFF.jacobian(function, s, 0.0) for s in nodes]),
            np.array([_AUTODIFF.jacobian(function, s, 0.0) for s in middles]),
        )

    @property
    def period(self) -> float:
        """The cycle's period."""
        return self._period

    def _value(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._function.eval(point=list(state), time=None), dtype=float)

    # -- multilinear forms -----------------------------------------------------

    def _taylor(
        self,
        state: np.ndarray,
        direction: np.ndarray,
        order: int,
    ) -> np.ndarray:
        return _AUTODIFF.taylor(self._function, state, direction, order, 0.0)[order]

    def _bilinear(self, state: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``B(u, v)`` at one state: ``[B(u+v)^2 - B(u-v)^2] / 4``.

        The jet's second coefficient is ``B(w, w) / 2``.
        """
        return _HALF * (
            self._taylor(state, u + v, _SECOND) - self._taylor(state, u - v, _SECOND)
        )

    def _trilinear(self, state: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """``C(u, u, v)`` at one state: ``[C(u+v)^3 - C(u-v)^3 - 2 C(v)^3] / 6``.

        The jet's third coefficient is ``C(w, w, w) / 6``.
        """
        return (
            self._taylor(state, u + v, _THIRD)
            - self._taylor(state, u - v, _THIRD)
            - 2.0 * self._taylor(state, v, _THIRD)
        )

    def _pointwise(
        self,
        form: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        first: _Profile,
        second: _Profile,
    ) -> _Profile:
        """``form`` at every node and midpoint of the cycle."""
        return _Profile(
            np.array(
                [
                    form(state, u, v)
                    for state, u, v in zip(
                        self._orbit.nodes,
                        first.nodes,
                        second.nodes,
                        strict=True,
                    )
                ],
            ),
            np.array(
                [
                    form(state, u, v)
                    for state, u, v in zip(
                        self._orbit.middles,
                        first.middles,
                        second.middles,
                        strict=True,
                    )
                ],
            ),
        )

    def _form(self, first: _Profile, second: _Profile) -> _Profile:
        """``B(t; u, v)`` along the cycle."""
        return self._pointwise(self._bilinear, first, second)

    def _cubic(self, first: _Profile, second: _Profile) -> _Profile:
        """``C(t; u, u, v)`` along the cycle."""
        return self._pointwise(self._trilinear, first, second)

    # -- integrals and linear problems ----------------------------------------

    def integral(self, first: _Profile, second: _Profile) -> complex:
        """``int_0^T <u, v> dt`` by Simpson's rule on nodes and midpoints."""
        nodes = np.sum(np.conj(first.nodes) * second.nodes, axis=1)
        middles = np.sum(np.conj(first.middles) * second.middles, axis=1)
        return complex(
            (self._step / _SIXTH)
            * np.sum(nodes[:-1] + _MIDPOINT_WEIGHT * middles + nodes[1:]),
        )

    def _operator(self, shift: complex, *, adjoint: bool) -> _Profile:
        """``P`` in ``y' = P y + g``: ``A - shift`` or ``-A^T - shift``."""
        identity = np.eye(self._jacobians.nodes.shape[1])
        if adjoint:
            nodes = -np.transpose(self._jacobians.nodes, (0, 2, 1))
            middles = -np.transpose(self._jacobians.middles, (0, 2, 1))
        else:
            nodes, middles = self._jacobians.nodes, self._jacobians.middles
        return _Profile(
            nodes.astype(complex) - shift * identity,
            middles.astype(complex) - shift * identity,
        )

    def solve(
        self,
        shift: complex = 0.0,
        *,
        adjoint: bool = False,
        twist: float = 1.0,
        source: _Profile | None = None,
        singular: bool = False,
    ) -> _Profile:
        """Solve ``y' = P y + g`` with ``y(T) = twist y(0)`` by Hermite-Simpson.

        ``P`` is ``A(t) - shift`` (or ``-A(t)^T - shift`` for the adjoint), ``g``
        is ``source`` (zero if none). A problem whose operator is singular - a
        null vector is wanted, or a Fredholm alternative holds - is bordered by
        its own near-null vectors (:func:`_bordered_solve`): for no source it
        returns the null vector, of unit size at the nodes; otherwise the
        solution with no component along it, which the caller then fixes.
        """
        operator = self._operator(shift, adjoint=adjoint)
        dimension = operator.nodes.shape[1]
        intervals = operator.nodes.shape[0] - 1
        step = self._step
        identity = np.eye(dimension)
        before, after, middle = (
            operator.nodes[:-1],
            operator.nodes[1:],
            operator.middles,
        )
        left = -identity - (step / _SIXTH) * (
            before
            + _MIDPOINT_WEIGHT * middle @ (_HALF * identity + (step / _EIGHTH) * before)
        )
        right = identity - (step / _SIXTH) * (
            after
            + _MIDPOINT_WEIGHT * middle @ (_HALF * identity - (step / _EIGHTH) * after)
        )
        size = (intervals + 1) * dimension
        rhs = np.zeros(size, dtype=complex)
        if source is not None:
            g_before, g_after = source.nodes[:-1], source.nodes[1:]
            rhs[: intervals * dimension] = (
                (step / _SIXTH)
                * (
                    g_before
                    + g_after
                    + _MIDPOINT_WEIGHT * source.middles
                    + _MIDPOINT_WEIGHT
                    * np.einsum(
                        "nij,nj->ni",
                        middle,
                        (step / _EIGHTH) * (g_before - g_after),
                    )
                )
            ).ravel()
        span = np.arange(dimension)
        starts = np.arange(intervals) * dimension
        block_rows = (
            starts[:, None, None] + span[None, :, None] + 0 * span[None, None, :]
        )
        block_columns = (
            starts[:, None, None] + 0 * span[None, :, None] + span[None, None, :]
        )
        rows = [
            block_rows.ravel(),
            block_rows.ravel(),
            intervals * dimension + span,
            intervals * dimension + span,
        ]
        columns = [
            block_columns.ravel(),
            (block_columns + dimension).ravel(),
            intervals * dimension + span,
            span,
        ]
        values = [
            left.ravel(),
            right.ravel(),
            np.ones(dimension, dtype=complex),
            -twist * np.ones(dimension, dtype=complex),
        ]
        operator_matrix = coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsc()
        if not singular:
            solution = splu(operator_matrix).solve(rhs)
        else:
            solution = _bordered_solve(
                operator_matrix,
                rhs[:size],
                null=source is None,
            )
        solution = solution[:size].reshape(intervals + 1, dimension)
        return self._completed(solution, operator, source)

    def _completed(
        self,
        nodes: np.ndarray,
        operator: _Profile,
        source: _Profile | None,
    ) -> _Profile:
        """The nodes with their Hermite midpoints, off the slopes ``P y + g``."""
        slopes = np.einsum("nij,nj->ni", operator.nodes, nodes)
        if source is not None:
            slopes = slopes + source.nodes
        middles = _HALF * (nodes[:-1] + nodes[1:]) + (self._step / _EIGHTH) * (
            slopes[:-1] - slopes[1:]
        )
        return _Profile(nodes, middles)

    def _field_profile(self) -> _Profile:
        return _Profile(
            self._field.nodes.astype(complex),
            self._field.middles.astype(complex),
        )

    def _normalised(
        self, profile: _Profile, against: _Profile, value: complex
    ) -> _Profile:
        """``profile`` rescaled so that ``int <profile, against> = value``."""
        return profile.scaled(value / np.conj(self.integral(profile, against)))

    def _orthogonal(self, profile: _Profile, weight: _Profile) -> _Profile:
        """``profile`` plus the multiple of ``f`` with ``int <weight, .> = 0``."""
        field = self._field_profile()
        return profile - field.scaled(
            self.integral(weight, profile) / self.integral(weight, field),
        )

    # -- the coefficients ----------------------------------------------------

    def fold(self) -> float:
        """``b`` of the fold of cycles (limit point of cycles)."""
        field = self._field_profile()
        generalised = self.solve(source=field, singular=True)
        # Only v's component along f is free; the normalisation picks it.
        generalised = self._orthogonal(generalised, field)
        adjoint = self.solve(adjoint=True, singular=True)
        adjoint = self._normalised(adjoint, generalised, 1.0)
        quadratic = self._form(generalised, generalised)
        linear = generalised.apply(self._jacobians)
        return float(
            _HALF * self.integral(adjoint, quadratic + linear.scaled(2.0)).real,
        )

    def flip(self) -> float:
        """``c`` of the period-doubling (flip) bifurcation."""
        eigen = self.solve(twist=-1.0, singular=True)
        eigen = eigen.scaled(1.0 / np.sqrt(self.integral(eigen, eigen).real))
        adjoint = self.solve(adjoint=True, twist=-1.0, singular=True)
        adjoint = self._normalised(adjoint, eigen, _HALF)
        field = self._field_profile()
        periodic = self.solve(adjoint=True, singular=True)
        periodic = self._normalised(periodic, field, _HALF)
        quadratic = self._form(eigen, eigen)
        a = self.integral(periodic, quadratic).real
        second = self.solve(source=quadratic - field.scaled(2.0 * a), singular=True)
        second = self._orthogonal(second, periodic)
        cubic = self._cubic(eigen, eigen)
        mixed = self._form(eigen, second)
        linear = eigen.apply(self._jacobians)
        return float(
            self.integral(
                adjoint,
                cubic + mixed.scaled(3.0) - linear.scaled(6.0 * a),
            ).real
            / 3.0,
        )

    def torus(self, angle: float) -> complex:
        """``d`` of the Neimark-Sacker bifurcation at multipliers ``e^(+-i angle)``."""
        period = self._period
        rate = 1j * angle / period
        eigen = self.solve(rate, singular=True)
        eigen = eigen.scaled(1.0 / np.sqrt(self.integral(eigen, eigen).real))
        adjoint = self.solve(rate, adjoint=True, singular=True)
        adjoint = self._normalised(adjoint, eigen, 1.0)
        conjugate = eigen.conjugate()
        double = self.solve(2.0 * rate, source=self._form(eigen, eigen))
        field = self._field_profile()
        periodic = self.solve(adjoint=True, singular=True)
        periodic = self._normalised(periodic, field, 1.0)
        mixed = self._form(eigen, conjugate)
        a = self.integral(periodic, mixed).real
        middle = self.solve(source=mixed - field.scaled(a), singular=True)
        middle = self._orthogonal(middle, periodic)
        cubic = self._cubic(eigen, conjugate)
        terms = _HALF * self.integral(adjoint, cubic)
        terms += _HALF * self.integral(
            adjoint,
            self._form(middle, eigen).scaled(2.0) + self._form(double, conjugate),
        )
        terms -= a * self.integral(adjoint, eigen.apply(self._jacobians))
        terms += 1j * a * angle / period
        return complex(terms)
