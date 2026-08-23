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

The current implementation covers manifolds spanned by real eigenvalues (saddle
type). Complex (spiral) manifolds need the conjugate-coordinate variant and are
deferred; see ``ROADMAP.md``.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.multivariate import Multivariate
from discrecontinual_equations.function.function import Function

_IMAGINARY = 1.0e-9
_LINEAR_DEGREE = 2


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
    """A Taylor parameterization of an invariant manifold."""

    __slots__ = ["_coefficients", "_eigenvalues", "_equilibrium", "_order"]

    def __init__(
        self,
        coefficients: list[Multivariate],
        eigenvalues: list[float],
        equilibrium: np.ndarray,
        order: int,
    ) -> None:
        self._coefficients = coefficients
        self._eigenvalues = eigenvalues
        self._equilibrium = equilibrium
        self._order = order

    @property
    def dimension(self) -> int:
        """Number of manifold coordinates."""
        return len(self._eigenvalues)

    @property
    def eigenvalues(self) -> list[float]:
        """Restricted linear eigenvalues, in coordinate order."""
        return self._eigenvalues

    @property
    def order(self) -> int:
        """Truncation order of the parameterization."""
        return self._order

    def point(self, coordinates: list[float]) -> np.ndarray:
        """Evaluate ``P(theta)`` - the state on the manifold at ``theta``."""
        result = np.array(self._equilibrium, dtype=float)
        for i, series in enumerate(self._coefficients):
            for key, value in series.coefficients.items():
                monomial = 1.0
                for axis, power in enumerate(key):
                    monomial *= coordinates[axis] ** power
                result[i] += complex(value).real * monomial
        return result

    def flow_image(self, coordinates: list[float], time: float) -> np.ndarray:
        """The chart point whose coordinates have flowed for ``time``.

        On the true manifold ``P(theta)`` maps to ``P(e^{Lambda t} theta)`` under
        the flow, so this is what integrating the field from ``point(theta)`` must
        reproduce - the invariance test.
        """
        flowed = [
            coordinates[axis] * np.exp(self._eigenvalues[axis] * time)
            for axis in range(self.dimension)
        ]
        return self.point(flowed)


class TaylorManifold:
    """Compute an invariant-manifold chart order by order."""

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
        chosen = selection.indices(values)
        eigenvalues = [float(values[i].real) for i in chosen]
        basis = np.column_stack([vectors[:, i].real for i in chosen])
        dimension = len(chosen)
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
        return ManifoldChart(series, eigenvalues, equilibrium, self._order)

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
        eigenvalues: list[float],
        coefficients: list[dict[tuple[int, ...], complex]],
        degree: int,
    ) -> None:
        state = jacobian.shape[0]
        dimension = len(eigenvalues)
        point = [
            Multivariate(coefficients[i], self._order, dimension) for i in range(state)
        ]
        field = function.eval(point=point, time=None)
        identity = np.eye(state)
        for monomial in self._monomials(degree, dimension):
            rate = sum(monomial[axis] * eigenvalues[axis] for axis in range(dimension))
            remainder = np.array(
                [
                    complex(field[i].coefficients.get(monomial, 0.0)).real
                    for i in range(state)
                ],
            )
            solution = np.linalg.solve(jacobian - rate * identity, -remainder)
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
