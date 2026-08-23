"""Center manifold reduction of an n-dimensional Hopf point to the plane.

At a Hopf point the asymptotic dynamics collapses onto a two-dimensional invariant
center manifold tangent to the eigenspace of the critical pair ``+/- i omega``.
Reducing to it lets the planar focal-value recursion - and therefore the
generalized-Hopf and degenerate-Bautin detectors - apply in any dimension.

The reduction is modelled as a pipeline of collaborators, each with a single
responsibility: a :class:`SpectralSplit` strategy produces the
:class:`CoordinateSplit` into center and hyperbolic parts; :class:`FieldExpansion`
expresses the field in those coordinates as a :class:`SplitField`;
:class:`ManifoldExpansion` solves the invariance equation order by order,
delegating each degree to a :class:`HomologicalEquation`; and
:class:`ReducedField` substitutes the manifold back to obtain the normalised
planar field. :class:`TaylorCenterManifold` composes them and hands the reduced
field to the verified planar recursion.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.continuation.focal import recurse
from discrecontinual_equations.continuation.multivariate import Multivariate
from discrecontinual_equations.function.function import Function

_ORDER = 6
_IMAGINARY = 1.0e-9
_PLANAR_DIMENSION = 2


class CoordinateSplit:
    """A linear change of coordinates separating center from hyperbolic parts."""

    __slots__ = ["_frequency", "_inverse", "_transform"]

    def __init__(self, transform: np.ndarray, frequency: float) -> None:
        self._transform = transform
        self._inverse = np.linalg.inv(transform)
        self._frequency = frequency

    @property
    def transform(self) -> np.ndarray:
        """Columns are the center basis followed by the hyperbolic basis."""
        return self._transform

    @property
    def inverse(self) -> np.ndarray:
        """Inverse of the coordinate transform."""
        return self._inverse

    @property
    def frequency(self) -> float:
        """Imaginary part of the critical eigenvalue."""
        return self._frequency

    @property
    def dimension(self) -> int:
        """Number of state variables."""
        return self._transform.shape[0]

    @property
    def hyperbolic_dimension(self) -> int:
        """Number of hyperbolic directions off the center manifold."""
        return self._transform.shape[0] - _PLANAR_DIMENSION


class SpectralSplit(ABC):
    """Choose the coordinates that split the center and hyperbolic subspaces."""

    @abstractmethod
    def split(self, jacobian: np.ndarray) -> CoordinateSplit:
        """Return the coordinate split induced by the Jacobian at the point."""
        raise NotImplementedError


class EigenvalueSplit(SpectralSplit):
    """Split using the eigenvectors of the Jacobian."""

    def split(self, jacobian: np.ndarray) -> CoordinateSplit:
        values, vectors = np.linalg.eig(jacobian)
        dimension = jacobian.shape[0]
        critical = self._critical(values, dimension)
        conjugate = self._nearest(values, np.conj(values[critical]), range(dimension))
        eigenvector = vectors[:, critical]
        center = np.column_stack([eigenvector.real, -eigenvector.imag])
        columns = self._hyperbolic(values, vectors, {critical, conjugate})
        transform = np.column_stack([center, *columns]) if columns else center
        return CoordinateSplit(transform, float(values[critical].imag))

    def _critical(self, values: np.ndarray, dimension: int) -> int:
        candidates = [i for i in range(dimension) if values[i].imag > _IMAGINARY]
        return min(candidates, key=lambda i: abs(values[i].real))

    def _nearest(
        self,
        values: np.ndarray,
        target: complex,
        pool: Iterable[int],
    ) -> int:
        return min(pool, key=lambda i: abs(values[i] - target))

    def _hyperbolic(
        self,
        values: np.ndarray,
        vectors: np.ndarray,
        used: set[int],
    ) -> list[np.ndarray]:
        columns: list[np.ndarray] = []
        seen = set(used)
        for index in range(len(values)):
            if index in seen:
                continue
            seen.add(index)
            if abs(values[index].imag) < _IMAGINARY:
                columns.append(vectors[:, index].real)
                continue
            available = [j for j in range(len(values)) if j not in seen]
            partner = self._nearest(values, np.conj(values[index]), available)
            seen.add(partner)
            columns.append(vectors[:, index].real)
            columns.append(vectors[:, index].imag)
        return columns


class SplitField:
    """The vector field in split coordinates, as truncated series."""

    __slots__ = ["_block", "_center", "_hyperbolic"]

    def __init__(
        self,
        center: list[Multivariate],
        hyperbolic: list[Multivariate],
        block: np.ndarray,
    ) -> None:
        self._center = center
        self._hyperbolic = hyperbolic
        self._block = block

    @property
    def center(self) -> list[Multivariate]:
        """The two center-coordinate components of the field."""
        return self._center

    @property
    def hyperbolic(self) -> list[Multivariate]:
        """The hyperbolic-coordinate components of the field."""
        return self._hyperbolic

    @property
    def block(self) -> np.ndarray:
        """Linear part of the hyperbolic dynamics."""
        return self._block

    @property
    def hyperbolic_dimension(self) -> int:
        """Number of hyperbolic directions."""
        return len(self._hyperbolic)


class FieldExpansion:
    """Expand a vector field in split coordinates as multivariate series."""

    __slots__ = ["_equilibrium", "_function", "_split", "_time"]

    def __init__(
        self,
        function: Function,
        equilibrium: np.ndarray,
        split: CoordinateSplit,
        time: float,
    ) -> None:
        self._function = function
        self._equilibrium = equilibrium
        self._split = split
        self._time = time

    def expand(self) -> SplitField:
        rows = self._rows()
        center = rows[:_PLANAR_DIMENSION]
        hyperbolic = rows[_PLANAR_DIMENSION:]
        return SplitField(center, hyperbolic, self._linear_block(hyperbolic))

    def _rows(self) -> list[Multivariate]:
        dimension = self._split.dimension
        transform = self._split.transform
        inverse = self._split.inverse
        variables = [
            Multivariate.variable(axis, 0.0, _ORDER, dimension)
            for axis in range(dimension)
        ]
        zero = Multivariate.constant(0.0, _ORDER, dimension)
        point = [
            Multivariate.constant(float(self._equilibrium[i]), _ORDER, dimension)
            + sum((transform[i, k] * variables[k] for k in range(dimension)), zero)
            for i in range(dimension)
        ]
        raw = self._function.eval(point=point, time=self._time)
        return [
            sum((inverse[i, k] * raw[k] for k in range(dimension)), zero)
            for i in range(dimension)
        ]

    def _linear_block(self, hyperbolic: list[Multivariate]) -> np.ndarray:
        size = len(hyperbolic)
        block = np.zeros((size, size))
        for row in range(size):
            for column in range(size):
                coefficient = hyperbolic[row].linear_coefficient(
                    _PLANAR_DIMENSION + column,
                )
                block[row, column] = complex(coefficient).real
        return block


class CenterCoordinates:
    """The two center coordinates as seed series, shared across the pipeline."""

    __slots__ = ["_order"]

    def __init__(self, order: int) -> None:
        self._order = order

    def variables(self) -> list[Bivariate]:
        """Seed series for the two center coordinates."""
        return [
            Bivariate({(1, 0): 1.0}, self._order),
            Bivariate({(0, 1): 1.0}, self._order),
        ]

    def arguments(self, manifold: list[Bivariate]) -> list[Bivariate]:
        """Full substitution: center coordinates followed by the manifold."""
        return self.variables() + manifold


class HomologicalEquation:
    """The degree-``m`` linear equation for one order of the center manifold.

    The operator ``P -> DP . A_c u - A_h P`` is invertible because the hyperbolic
    block ``A_h`` has no eigenvalues on the imaginary axis, so each order is a
    unique solve.
    """

    __slots__ = ["_block", "_degree", "_frequency", "_hyperbolic_dimension"]

    def __init__(self, frequency: float, block: np.ndarray, degree: int) -> None:
        self._frequency = frequency
        self._block = block
        self._degree = degree
        self._hyperbolic_dimension = block.shape[0]

    def solve(self, residual: list[Bivariate]) -> list[Bivariate]:
        """Return the degree-``m`` manifold terms that cancel the residual."""
        basis = [(a, self._degree - a) for a in range(self._degree + 1)]
        operator = self._operator(basis)
        right = self._right_hand_side(residual, basis)
        solution = np.linalg.solve(operator, right)
        return self._series(solution, basis)

    def _operator(self, basis: list[tuple[int, int]]) -> np.ndarray:
        index = {monomial: position for position, monomial in enumerate(basis)}
        width = len(basis)
        size = self._hyperbolic_dimension * width
        operator = np.zeros((size, size))
        for row in range(self._hyperbolic_dimension):
            for column, (a, b) in enumerate(basis):
                target = row * width + column
                if a > 0:
                    place = row * width + index[(a - 1, b + 1)]
                    operator[place, target] += -self._frequency * a
                if b > 0:
                    place = row * width + index[(a + 1, b - 1)]
                    operator[place, target] += self._frequency * b
                for other in range(self._hyperbolic_dimension):
                    operator[other * width + column, target] += -self._block[
                        other,
                        row,
                    ]
        return operator

    def _right_hand_side(
        self,
        residual: list[Bivariate],
        basis: list[tuple[int, int]],
    ) -> np.ndarray:
        width = len(basis)
        right = np.zeros(self._hyperbolic_dimension * width)
        for row in range(self._hyperbolic_dimension):
            homogeneous = residual[row].homogeneous(self._degree)
            for column, monomial in enumerate(basis):
                value = complex(homogeneous.get(monomial, 0.0)).real
                right[row * width + column] = -value
        return right

    def _series(
        self,
        solution: np.ndarray,
        basis: list[tuple[int, int]],
    ) -> list[Bivariate]:
        width = len(basis)
        return [
            Bivariate(
                {
                    basis[column]: solution[row * width + column]
                    for column in range(width)
                },
                _ORDER,
            )
            for row in range(self._hyperbolic_dimension)
        ]


class ManifoldExpansion:
    """Solve the center manifold as a Taylor series, order by order."""

    __slots__ = ["_coordinates", "_field", "_frequency"]

    def __init__(
        self,
        field: SplitField,
        frequency: float,
        coordinates: CenterCoordinates,
    ) -> None:
        self._field = field
        self._frequency = frequency
        self._coordinates = coordinates

    def expand(self) -> list[Bivariate]:
        hyperbolic = self._field.hyperbolic_dimension
        manifold = [Bivariate({}, _ORDER) for _ in range(hyperbolic)]
        for degree in range(_PLANAR_DIMENSION, _ORDER + 1):
            equation = HomologicalEquation(self._frequency, self._field.block, degree)
            correction = equation.solve(self._residual(manifold))
            manifold = [manifold[row] + correction[row] for row in range(hyperbolic)]
        return manifold

    def _residual(self, manifold: list[Bivariate]) -> list[Bivariate]:
        arguments = self._coordinates.arguments(manifold)
        centre = [
            component.compose(arguments, _ORDER) for component in self._field.center
        ]
        residual = []
        for row in range(self._field.hyperbolic_dimension):
            flow = (
                manifold[row].derivative(0) * centre[0]
                + manifold[row].derivative(1) * centre[1]
            )
            image = self._field.hyperbolic[row].compose(arguments, _ORDER)
            residual.append(flow - image)
        return residual


class ReducedField:
    """Substitute the manifold into the center rows and normalise to rotation."""

    __slots__ = ["_coordinates", "_field", "_frequency", "_manifold"]

    def __init__(
        self,
        field: SplitField,
        manifold: list[Bivariate],
        frequency: float,
        coordinates: CenterCoordinates,
    ) -> None:
        self._field = field
        self._manifold = manifold
        self._frequency = frequency
        self._coordinates = coordinates

    def planar(self) -> list[Bivariate]:
        """The reduced planar field, time-scaled so its linear part is a rotation."""
        arguments = self._coordinates.arguments(self._manifold)
        scale = 1.0 / self._frequency
        return [
            self._realise(component.compose(arguments, _ORDER), scale)
            for component in self._field.center
        ]

    def _realise(self, series: Bivariate, scale: float) -> Bivariate:
        return Bivariate(
            {
                key: complex(value).real * scale
                for key, value in series.coefficients.items()
            },
            _ORDER,
        )


class CenterManifold(ABC):
    """Reduce an n-dimensional Hopf point to its planar Lyapunov quantities."""

    @abstractmethod
    def lyapunov_quantities(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        time: float,
    ) -> tuple[float, float]:
        """Return the first and second Lyapunov quantities at the Hopf point."""
        raise NotImplementedError


class TaylorCenterManifold(CenterManifold):
    """Center manifold as a truncated Taylor series, reduced to the plane."""

    __slots__ = ["_coordinates", "_splitter"]

    def __init__(self, splitter: SpectralSplit | None = None) -> None:
        self._splitter = splitter if splitter is not None else EigenvalueSplit()
        self._coordinates = CenterCoordinates(_ORDER)

    def lyapunov_quantities(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        time: float,
    ) -> tuple[float, float]:
        split = self._splitter.split(jacobian)
        field = FieldExpansion(function, equilibrium, split, time).expand()
        manifold = ManifoldExpansion(
            field,
            split.frequency,
            self._coordinates,
        ).expand()
        reduced = ReducedField(
            field,
            manifold,
            split.frequency,
            self._coordinates,
        ).planar()
        return recurse(reduced)
