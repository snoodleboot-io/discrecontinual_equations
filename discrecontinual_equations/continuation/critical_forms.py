"""Multilinear forms of a vector field at a critical point, in state and parameters.

The normal-form coefficients at a codim-2 point are built from the symmetric
multilinear forms of ``f(x, alpha)`` at the critical point ``(x0, alpha0)``:

    f(x0 + x, alpha0 + k) = A x + J1 k + B(x, x)/2 + A1(x, k) + J2(k, k)/2
                            + C(x, x, x)/6 + ...

Every one of them is a single symmetric ``m``-linear form of ``F(x, alpha)`` on the
joint space ``R^n x R^2``, evaluated on joint directions ``(x, 0)`` and
``(0, k)``. :class:`CriticalForms` recovers it by polarisation from the directional
Taylor coefficient ``P_m(v) = D^m F(v, ..., v) / m!``,

    F_m(v1, ..., vm) = sum over nonempty S of (-1)^(m - |S|) P_m(sum_{i in S} v_i),

with ``P_m`` taken from jets (:meth:`.TwoParameterProblem.joint_taylor`), so each
form is exact for any field written in arithmetic. Directions may be complex.

The bordered solves of Kuznetsov's method live here as well, because every
normal form uses them: ``L^INV r`` is the solution of ``L h = r`` orthogonal to the
adjoint null vector, found from the nonsingular ``(n+1)`` system
``[[L, q], [p^H, 0]] (h, s) = (r, 0)`` (Kuznetsov 1999, eq. 4.6).
"""

import numpy as np

from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)

_PARAMETER_COUNT = 2


class CriticalForms:
    """Exact symmetric multilinear forms of ``f`` at a point, by polarisation."""

    __slots__ = [
        "_cache",
        "_dimension",
        "_parameter_a",
        "_parameter_b",
        "_problem",
        "_state",
    ]

    def __init__(
        self,
        problem: TwoParameterProblem,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> None:
        self._problem = problem
        self._state = np.asarray(state, dtype=float)
        self._parameter_a = parameter_a
        self._parameter_b = parameter_b
        self._dimension = self._state.size
        self._cache: dict[tuple[int, bytes], np.ndarray] = {}

    def _power(self, direction: np.ndarray, order: int) -> np.ndarray:
        key = (order, direction.tobytes())
        if key not in self._cache:
            self._cache[key] = self._problem.joint_taylor(
                self._state,
                direction,
                order,
                self._parameter_a,
                self._parameter_b,
            )[order]
        return self._cache[key]

    def joint(self, *directions: np.ndarray) -> np.ndarray:
        """The symmetric form on joint directions ``(x, k)`` of length ``n + 2``."""
        order = len(directions)
        total = np.zeros(self._dimension, dtype=complex)
        for mask in range(1, 2**order):
            chosen = [directions[i] for i in range(order) if mask >> i & 1]
            sign = -1.0 if (order - len(chosen)) % 2 else 1.0
            total = total + sign * self._power(np.sum(chosen, axis=0), order)
        return total

    def state(self, *vectors: np.ndarray) -> np.ndarray:
        """The pure state form ``B``, ``C``, ``D``, ... on state vectors."""
        return self.joint(*(self._lift(vector, None) for vector in vectors))

    def mixed(
        self,
        vectors: list[np.ndarray],
        parameters: list[np.ndarray],
    ) -> np.ndarray:
        """A form with some state and some parameter arguments, e.g. ``A1(x, k)``."""
        directions = [self._lift(vector, None) for vector in vectors]
        directions += [self._lift(None, parameter) for parameter in parameters]
        return self.joint(*directions)

    def parameter_jacobian(self) -> np.ndarray:
        """``J1 = f_alpha``, one column per active parameter."""
        columns = [
            self.mixed([], [np.eye(_PARAMETER_COUNT)[index]]).real
            for index in range(_PARAMETER_COUNT)
        ]
        return np.column_stack(columns)

    def _lift(
        self,
        vector: np.ndarray | None,
        parameter: np.ndarray | None,
    ) -> np.ndarray:
        joint = np.zeros(self._dimension + _PARAMETER_COUNT, dtype=complex)
        if vector is not None:
            joint[: self._dimension] = vector
        if parameter is not None:
            joint[self._dimension :] = parameter
        return joint


def bordered_solve(
    matrix: np.ndarray,
    column: np.ndarray,
    row: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Solve ``[[matrix, column], [row, 0]] (h, s) = (right, 0)`` and return ``h``.

    With ``column`` spanning the kernel and ``row`` annihilating it from the left
    (or the other way round), the bordered matrix is nonsingular even though
    ``matrix`` is not, and ``h`` is the solution of ``matrix h = right`` with
    ``row . h = 0`` whenever ``right`` is in the range.
    """
    size = matrix.shape[0]
    bordered = np.zeros((size + 1, size + 1), dtype=complex)
    bordered[:size, :size] = matrix
    bordered[:size, size] = column
    bordered[size, :size] = row
    extended = np.concatenate([np.asarray(right, dtype=complex), [0.0]])
    return np.linalg.solve(bordered, extended)[:size]


def inner(left: np.ndarray, right: np.ndarray) -> complex:
    """Kuznetsov's scalar product ``<p, q> = conj(p) . q``."""
    return complex(np.vdot(left, right))


def unit_phase(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length with the largest component real and positive.

    MatCont normalises critical eigenvectors to unit length; the phase (sign, for a
    real vector) is fixed here so that repeated calls agree.
    """
    vector = np.asarray(vector, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    largest = vector[int(np.argmax(np.abs(vector)))]
    return vector * (abs(largest) / largest)


def right_null(matrix: np.ndarray) -> np.ndarray:
    """The (approximate) right null vector: last right singular vector."""
    _left, _values, right = np.linalg.svd(matrix)
    return right[-1].conj()


def eigenvector(matrix: np.ndarray, eigenvalue: complex) -> np.ndarray:
    """The null vector of ``matrix - eigenvalue I``, robust near a defective pair."""
    size = matrix.shape[0]
    return right_null(matrix - eigenvalue * np.eye(size))


def adjoint(matrix: np.ndarray, eigenvalue: complex, right: np.ndarray) -> np.ndarray:
    """The adjoint eigenvector ``A^T p = conj(eigenvalue) p`` with ``<p, q> = 1``."""
    left = eigenvector(matrix.T, np.conj(eigenvalue))
    return left / np.conj(inner(left, right))
