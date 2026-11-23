"""Stable and unstable manifolds by the parameterization method.

The invariant manifold of an equilibrium tangent to a set of eigenvectors is
sought as a Taylor map ``P`` from the eigenspace coordinates ``theta`` into state
space satisfying the conjugacy ``f(P(theta)) = DP(theta) . Lambda theta``, where
``Lambda`` is the restricted linear dynamics (the selected eigenvalues). Writing
``P`` as a truncated series and matching order by order, each homogeneous term is a
per-monomial linear solve ``(A - (k . lambda) I) P_k = -N_k`` - non-singular
whenever ``k . lambda`` is not in the spectrum of the Jacobian ``A``
(non-resonance). This reuses the multivariate-series field expansion; selecting the
unstable spectrum yields ``W^u`` and the stable spectrum ``W^s``.

A complex pair ``lambda, conj(lambda)`` - a saddle-focus, whose manifold spirals
- is solved in the complex coordinates ``(z, conj(z))`` of the pair, with
eigenvectors ``v / 2`` and ``conj(v) / 2``: the coefficient of ``z^a conj(z)^b``
solves ``(A - (a lambda + b conj(lambda)) I) P = -N``, a complex system, and
the coefficient of ``z^b conj(z)^a`` comes out as its conjugate, so the chart
is real on ``z = u + i w``. Its real coordinates ``(u, w)`` start along
``Re v`` and ``-Im v``. (Cabre, Fontich and de la Llave, Indiana Univ. Math. J.
52, 2003; Haro, Canadell, Figueras, Luque and Mondelo, *The Parameterization
Method for Invariant Manifolds*, Springer 2016, Sec. 2.2 on complex
eigenvalues.) A resonance - a rate within ``_RESONANCE`` of the spectrum - is
solved by least squares when the remainder lies in the range it leaves, and
otherwise raises :class:`numpy.linalg.LinAlgError`: no chart of that order
exists.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.multivariate import Multivariate
from discrecontinual_equations.function.function import Function

_IMAGINARY = 1.0e-9
_LINEAR_DEGREE = 2
# A monomial's rate this close to an eigenvalue, relative to its size, is a
# resonance: its homological equation has no solution worth the name.
_RESONANCE = 1.0e-8
# A resonant monomial's least-squares term must cancel its remainder to this,
# relative to the remainder, or the resonance is an obstruction.
_UNFORCED = 1.0e-9


class ManifoldSelection(ABC):
    """Choose which eigenvalues span the invariant manifold."""

    @abstractmethod
    def indices(self, values: np.ndarray) -> list[int]:
        """Return the indices of the eigenvalues that span the manifold."""
        raise NotImplementedError


class UnstableManifold(ManifoldSelection):
    """The manifold tangent to eigenvectors with positive real part."""

    def indices(self, values: np.ndarray) -> list[int]:
        return [i for i in range(len(values)) if values[i].real > _IMAGINARY]


class StableManifold(ManifoldSelection):
    """The manifold tangent to eigenvectors with negative real part."""

    def indices(self, values: np.ndarray) -> list[int]:
        return [i for i in range(len(values)) if values[i].real < -_IMAGINARY]


class ManifoldChart:
    """A Taylor parameterization of an invariant manifold.

    Its coordinates are real. A complex pair's two coordinates ``(u, w)`` are
    the real and imaginary parts of the pair's complex coordinate ``z``; the
    series itself is in ``(z, conj(z))`` and real on that diagonal.
    """

    __slots__ = ["_coefficients", "_eigenvalues", "_equilibrium", "_order", "_pairs"]

    def __init__(
        self,
        coefficients: list[Multivariate],
        eigenvalues: list[complex],
        equilibrium: np.ndarray,
        order: int,
        pairs: list[int] | None = None,
    ) -> None:
        self._coefficients = coefficients
        self._eigenvalues = eigenvalues
        self._equilibrium = equilibrium
        self._order = order
        self._pairs = pairs or []

    @property
    def dimension(self) -> int:
        """Number of manifold coordinates."""
        return len(self._eigenvalues)

    @property
    def eigenvalues(self) -> list[complex]:
        """Restricted linear eigenvalues, in coordinate order.

        Real numbers where the manifold is spanned by real eigenvalues; a
        complex pair appears as ``lambda, conj(lambda)``.
        """
        if all(abs(complex(value).imag) <= _IMAGINARY for value in self._eigenvalues):
            return [float(complex(value).real) for value in self._eigenvalues]
        return list(self._eigenvalues)

    @property
    def order(self) -> int:
        """Truncation order of the parameterization."""
        return self._order

    def point(self, coordinates: list[float]) -> np.ndarray:
        """Evaluate ``P(theta)`` - the state on the manifold at ``theta``.

        The series already carries the equilibrium as its constant term, so the
        evaluation starts from zero. Starting from the equilibrium as well used to
        add it twice, which every chart of an equilibrium at the origin hid.
        """
        return self._evaluate(self._complex(coordinates))

    def flow_image(self, coordinates: list[float], time: float) -> np.ndarray:
        """The chart point whose coordinates have flowed for ``time``.

        On the true manifold ``P(theta)`` maps to ``P(e^{Lambda t} theta)`` under
        the flow, so this is what integrating the field from ``point(theta)`` must
        reproduce - the invariance test. A complex coordinate turns as it
        shrinks or grows: ``z e^{lambda t}``.
        """
        flowed = [
            value * np.exp(complex(rate) * time)
            for value, rate in zip(
                self._complex(coordinates),
                self._eigenvalues,
                strict=True,
            )
        ]
        return self._evaluate(flowed)

    def _complex(self, coordinates: list[float]) -> list[complex]:
        """The series' own coordinates: ``(u + i w, u - i w)`` for each pair."""
        values = [complex(value) for value in coordinates]
        for first in self._pairs:
            u, w = values[first], values[first + 1]
            values[first], values[first + 1] = u + 1j * w, u - 1j * w
        return values

    def _evaluate(self, values: list[complex]) -> np.ndarray:
        result = np.zeros(len(self._coefficients), dtype=complex)
        for i, series in enumerate(self._coefficients):
            for key, value in series.coefficients.items():
                monomial = complex(1.0)
                for axis, power in enumerate(key):
                    monomial *= values[axis] ** power
                result[i] += complex(value) * monomial
        return result.real


class TaylorManifold:
    """Compute an invariant-manifold chart order by order.

    Complexity:
        For order ``K``, manifold dimension ``d`` and ``M = C(d + K, d)``
        monomials: ``K - 1`` field evaluations on series whose products visit
        ``M^2`` term pairs (``O(K^3)`` for ``d = 1``, ``O(K^5)`` for ``d = 2``),
        and one dense ``O(n^3)`` solve per monomial; ``O(n M)`` memory.
    """

    __slots__ = ["_order"]

    def __init__(self, order: int = 5) -> None:
        self._order = order

    def compute(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        selection: ManifoldSelection,
    ) -> ManifoldChart:
        """Return the Taylor chart of the selected invariant manifold."""
        values, vectors = np.linalg.eig(jacobian)
        eigenvalues, basis, pairs = _modes(values, vectors, selection.indices(values))
        dimension = len(eigenvalues)
        state = jacobian.shape[0]
        coefficients = self._seed(equilibrium, basis, state, dimension)
        for degree in range(_LINEAR_DEGREE, self._order + 1):
            self._solve_degree(
                function,
                jacobian,
                eigenvalues,
                coefficients,
                degree,
            )
        series = [
            Multivariate(coefficients[i], self._order, dimension) for i in range(state)
        ]
        return ManifoldChart(series, eigenvalues, equilibrium, self._order, pairs)

    def _seed(
        self,
        equilibrium: np.ndarray,
        basis: np.ndarray,
        state: int,
        dimension: int,
    ) -> list[dict[tuple[int, ...], complex]]:
        coefficients: list[dict[tuple[int, ...], complex]] = []
        for i in range(state):
            terms: dict[tuple[int, ...], complex] = {
                (0,) * dimension: complex(equilibrium[i]),
            }
            for axis in range(dimension):
                key = tuple(1 if j == axis else 0 for j in range(dimension))
                terms[key] = complex(basis[i, axis])
            coefficients.append(terms)
        return coefficients

    def _solve_degree(
        self,
        function: Function,
        jacobian: np.ndarray,
        eigenvalues: list[complex],
        coefficients: list[dict[tuple[int, ...], complex]],
        degree: int,
    ) -> None:
        state = jacobian.shape[0]
        dimension = len(eigenvalues)
        point = [
            Multivariate(coefficients[i], self._order, dimension) for i in range(state)
        ]
        field = function.eval(point=point, time=None)  # type: ignore[arg-type]
        identity = np.eye(state)
        spectrum = np.linalg.eigvals(jacobian)
        for monomial in self._monomials(degree, dimension):
            rate = sum(monomial[axis] * eigenvalues[axis] for axis in range(dimension))
            remainder = np.array(
                [
                    complex(field[i].coefficients.get(monomial, 0.0))
                    for i in range(state)
                ],
            )
            gap = float(np.min(np.abs(spectrum - rate)))
            matrix = jacobian - rate * identity
            if gap <= _RESONANCE * max(1.0, abs(rate)):
                # A resonance: solvable only if the remainder has no part
                # along the resonant eigenvector - unforced, as in a linear
                # field, or forced only in other directions - and then the
                # least-squares solution is a chart term. Otherwise no term
                # of this degree can cancel it.
                solution = np.linalg.lstsq(matrix, -remainder, rcond=_RESONANCE)[0]
                defect = np.max(np.abs(matrix @ solution + remainder))
                if defect > _UNFORCED * max(1.0, float(np.max(np.abs(remainder)))):
                    message = f"resonance: the rate of {monomial} is {rate}"
                    raise np.linalg.LinAlgError(message)
            else:
                solution = np.linalg.solve(matrix, -remainder)
            for i in range(state):
                coefficients[i][monomial] = complex(solution[i])

    def _monomials(self, degree: int, dimension: int) -> list[tuple[int, ...]]:
        if dimension == 1:
            return [(degree,)]
        monomials: list[tuple[int, ...]] = []
        for first in range(degree + 1):
            monomials.extend(
                (first, *rest)
                for rest in self._monomials(degree - first, dimension - 1)
            )
        return monomials


def _modes(
    values: np.ndarray,
    vectors: np.ndarray,
    chosen: list[int],
) -> tuple[list[complex], np.ndarray, list[int]]:
    """The chart's rates and linear terms, and where its complex pairs start.

    A real eigenvalue contributes its real eigenvector. A complex pair
    contributes ``v / 2`` and ``conj(v) / 2`` for ``lambda`` (imaginary part
    positive) and ``conj(lambda)``, so that on ``z = u + i w`` the linear term
    ``(v z + conj(v z)) / 2 = u Re v - w Im v`` is real and spans the pair's
    plane. Taking the real part of both eigenvectors instead - what this did
    before (DEQ-71) - gives the same vector twice, a rank-one "plane".
    """
    rates: list[complex] = []
    columns: list[np.ndarray] = []
    pairs: list[int] = []
    for index in chosen:
        value = complex(values[index])
        if abs(value.imag) <= _IMAGINARY:
            rates.append(complex(value.real))
            columns.append(vectors[:, index].real.astype(complex))
        elif value.imag > 0.0:
            pairs.append(len(rates))
            half = vectors[:, index] / 2.0
            rates.extend([value, value.conjugate()])
            columns.extend([half, half.conjugate()])
    if len(rates) != len(chosen):
        message = "the selection takes one of a complex pair without the other"
        raise ValueError(message)
    return rates, np.column_stack(columns), pairs
