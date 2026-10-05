"""Stationary densities of a stochastic flow above one dimension.

:mod:`.stochastic` can only answer the phenomenological question in one dimension,
because the closed form ``p_s(x) = (C / g^2) exp(integral 2 f_ito / g^2 dx)`` it rests
on is an antiderivative, and an antiderivative of a vector field exists only when the
field is a gradient. In the plane the drift generally is not: the noisy Hopf system
carries a rotation that no potential can produce, so there is no formula to evaluate
and the stationary Fokker-Planck equation has to be *solved*.

Two routes were available. An ensemble of paths from the solvers in
:mod:`...solver.stochastic` reuses more of the library and degrades gracefully with
dimension, but it answers with a sampling error of order ``1 / sqrt(paths)`` on top of
a density-estimation bandwidth, and the quantity most wanted here - whether the peak
of the density has split off the origin - is a second difference of the density, which
is exactly where that error is worst. This module takes the other route and solves the
stationary equation on a grid, because its error is deterministic discretisation error:
it does not move when the code is run again, it shrinks on a known power of the
spacing, and it can therefore be checked against an analytic density to a tight
tolerance rather than to an error bar. In two dimensions the cost is a sparse linear
solve of a few tens of thousands of unknowns, which is cheaper than any ensemble that
would resolve the same feature. Above three dimensions the grid becomes the wrong
answer and the ensemble becomes the right one.

The discretisation is a finite-volume one, written in flux form so that mass is
conserved exactly by construction. For ``dx = f dt + G dW`` with ``D = G G^T`` the
Ito stationary equation is ``div J = 0`` with

    ``J_d = f_d p - (1/2) sum_e d_e (D_de p)``,

and the key observation for the diagonal part is that the derivative falls on the
*product* ``D_dd p``, not on ``p``: with state-dependent noise the flux is
``f_d p - (1/2) d_d(D_dd p)``, which in the variable ``w = D_dd p`` is an advection
with velocity ``f_d / D_dd`` and a constant diffusion of ``1/2``. Exponential fitting
(Scharfetter-Gummel) applied in ``w`` is then exact wherever ``f / D`` is locally
constant, gives a positive operator at any cell Peclet number, and reproduces the
one-dimensional closed form above term for term. Fitting ``p`` instead - the obvious
thing to write - is wrong by the factor ``D_dd`` and silently misplaces every
threshold once the noise is multiplicative. The off-diagonal part is added as a plain
central difference of ``D_de p``; it is second order and conservative but carries no
positivity guarantee, so a strongly correlated noise on a coarse grid can return small
negative values, which :meth:`GridDensity.minimum` exposes rather than hides.

Because ``div J = 0`` with reflecting boundaries is homogeneous, the operator is
singular by exactly one dimension: the fluxes cancel in pairs, so its rows sum to
zero and any single row is implied by the others. One row is therefore replaced by the
normalisation ``integral p = 1``, which turns the null-space problem into one square
sparse solve with a unique answer.

The scheme needs ``D`` non-singular on the whole grid. That rules out two cases worth
naming: scalar noise, where ``D`` has rank one by construction (see :mod:`.noise`),
and hypoelliptic systems with noise on a strict subset of the components - a noisy
oscillator forced only in its velocity has a perfectly good smooth density, but not
one this discretisation can produce.
"""

import itertools
from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    diffusion_matrix,
    field_at,
    state_drift_shift,
    stratonovich_factor,
)
from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function

_HALF = 0.5
_MIN_AXIS_POINTS = 3
_UNIFORM_TOLERANCE = 1.0e-9
_ELLIPTICITY_FLOOR = 1.0e-10
_SMALL_EXPONENT = 1.0e-8
_ANCHOR_ROW = 0


def _bernoulli(exponent: np.ndarray) -> np.ndarray:
    """``z / (exp(z) - 1)``, continued to ``1 - z / 2`` where the ratio cancels.

    This is the exponential-fitting weight. It is written with ``expm1`` and a small
    argument branch because the naive ratio loses all its significant digits exactly
    where the grid is finest, which is the regime the scheme is meant to be used in.
    """
    values = np.empty_like(exponent)
    small = np.abs(exponent) < _SMALL_EXPONENT
    values[small] = 1.0 - _HALF * exponent[small]
    large = ~small
    values[large] = exponent[large] / np.expm1(exponent[large])
    return values


class DensityGrid:
    """A uniform tensor-product grid of cell centres, one axis per state variable.

    Uniform spacing per axis is required, not a convenience: the finite-volume
    divergence below weights every cell by the same volume, and a graded grid would
    need that weight to appear in the normalisation row and in the conservation
    argument that makes the operator's rank deficiency exactly one.
    """

    __slots__ = ["_axes"]

    def __init__(self, axes: Sequence[np.ndarray]) -> None:
        if len(axes) == 0:
            message = "a density grid needs at least one axis"
            raise ValueError(message)
        prepared = [np.asarray(axis, dtype=float) for axis in axes]
        for axis in prepared:
            _check_axis(axis)
        self._axes = prepared

    @classmethod
    def box(
        cls,
        lower: Sequence[float],
        upper: Sequence[float],
        counts: Sequence[int],
    ) -> "DensityGrid":
        """A grid on the box ``[lower, upper]`` with ``counts`` points per axis."""
        return cls(
            [
                np.linspace(low, high, count)
                for low, high, count in zip(lower, upper, counts, strict=True)
            ],
        )

    @property
    def dimension(self) -> int:
        """Number of state variables."""
        return len(self._axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Points per axis."""
        return tuple(len(axis) for axis in self._axes)

    @property
    def size(self) -> int:
        """Total number of cells."""
        return int(np.prod(self.shape))

    @property
    def spacings(self) -> np.ndarray:
        """Spacing of each axis."""
        return np.array([axis[1] - axis[0] for axis in self._axes])

    @property
    def cell_volume(self) -> float:
        """Volume of one cell, the quadrature weight of every sum over the grid."""
        return float(np.prod(self.spacings))

    def axis(self, index: int) -> np.ndarray:
        """Coordinates along one axis."""
        return self._axes[index]

    def points(self) -> np.ndarray:
        """Every cell centre as a ``(size, dimension)`` array in C order."""
        mesh = np.meshgrid(*self._axes, indexing="ij")
        return np.stack([block.ravel() for block in mesh], axis=1)

    def nearest(self, state: Sequence[float]) -> tuple[int, ...]:
        """Multi-index of the cell whose centre is nearest ``state``."""
        return tuple(
            int(np.argmin(np.abs(axis - value)))
            for axis, value in zip(self._axes, state, strict=True)
        )


def _check_axis(axis: np.ndarray) -> None:
    if axis.ndim != 1 or axis.size < _MIN_AXIS_POINTS:
        message = "each grid axis needs at least three increasing points"
        raise ValueError(message)
    widths = np.diff(axis)
    if np.min(widths) <= 0.0:
        message = "grid axes must increase"
        raise ValueError(message)
    if np.max(np.abs(widths - widths[0])) > _UNIFORM_TOLERANCE * widths[0]:
        message = "grid axes must be uniformly spaced"
        raise ValueError(message)


def _shifted_slice(step: int) -> slice:
    """The interior window of a grid translated by ``step`` cells along one axis."""
    if step < 0:
        return slice(0, -2)
    if step == 0:
        return slice(1, -1)
    return slice(2, None)


class GridDensity:
    """A density sampled on a :class:`DensityGrid`, with its shape descriptors.

    The descriptors are what a bifurcation argument actually consumes, and they are
    deliberately the same ones :class:`~.stochastic.DensityModes` reports in one
    dimension, lifted to a multi-index. One lift behaves differently and is worth
    knowing about: a rotationally symmetric density whose peak has moved onto a ring
    has a whole circle of near-equal maxima, so :meth:`interior_maxima` returns a
    chain of cells around that ring rather than a single point. :meth:`dominant_mode`
    picks one representative of it, and the radial probes in
    :mod:`.stochastic_threshold` ask the question that stays well posed.
    """

    __slots__ = ["_grid", "_values"]

    def __init__(self, grid: DensityGrid, values: np.ndarray) -> None:
        self._grid = grid
        self._values = np.asarray(values, dtype=float).reshape(grid.shape)

    @property
    def grid(self) -> DensityGrid:
        """The grid the density lives on."""
        return self._grid

    @property
    def values(self) -> np.ndarray:
        """The density, shaped like the grid."""
        return self._values

    def at(self, index: Sequence[int]) -> float:
        """Density in one cell, by multi-index."""
        return float(self._values[tuple(index)])

    def minimum(self) -> float:
        """Smallest value on the grid.

        A density is non-negative, so a negative minimum is a statement about the
        discretisation rather than about the system: the central differencing of the
        off-diagonal diffusion has undershot and the grid needs refining. It is
        reported rather than clipped so that it cannot be mistaken for a feature.
        """
        return float(np.min(self._values))

    def mass(self) -> float:
        """Integral over the grid; ``1`` up to the solve, and a check that it is."""
        return float(np.sum(self._values) * self._grid.cell_volume)

    def mean(self) -> np.ndarray:
        """First moment of the density."""
        weights = self._values.ravel()
        return (self._grid.points() * weights[:, None]).sum(axis=0) * (
            self._grid.cell_volume / self.mass()
        )

    def covariance(self) -> np.ndarray:
        """Second central moment, the quantity a Gaussian oracle pins exactly."""
        centred = self._grid.points() - self.mean()
        weights = self._values.ravel()
        outer = centred[:, :, None] * centred[:, None, :] * weights[:, None, None]
        return outer.sum(axis=0) * (self._grid.cell_volume / self.mass())

    def marginal(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates and the density integrated over every other axis."""
        others = tuple(i for i in range(self._grid.dimension) if i != axis)
        weight = self._grid.cell_volume / self._grid.spacings[axis]
        return self._grid.axis(axis), np.sum(self._values, axis=others) * weight

    def dominant_mode(self) -> np.ndarray:
        """Coordinates of the global maximum."""
        return self._coordinates(
            np.unravel_index(int(np.argmax(self._values)), self._grid.shape),
        )

    def interior_maxima(self) -> list[np.ndarray]:
        """Coordinates of every interior cell that dominates its neighbours.

        A cell qualifies when it is strictly above each neighbour that precedes it in
        the offset ordering and at least equal to each that follows, which is the
        multi-index reading of the one-dimensional rule in
        :class:`~.stochastic.DensityModes` and keeps a flat plateau from reporting
        every one of its cells.
        """
        dimension = self._grid.dimension
        centre = self._values[tuple(slice(1, -1) for _ in range(dimension))]
        winner = np.ones(centre.shape, dtype=bool)
        origin = tuple(0 for _ in range(dimension))
        for offset in itertools.product((-1, 0, 1), repeat=dimension):
            if offset == origin:
                continue
            shifted = self._values[tuple(_shifted_slice(step) for step in offset)]
            winner &= centre > shifted if offset < origin else centre >= shifted
        return [
            self._coordinates(tuple(index + 1 for index in position))
            for position in zip(*np.nonzero(winner), strict=True)
        ]

    def _coordinates(self, index: Sequence[int]) -> np.ndarray:
        return np.array(
            [self._grid.axis(axis)[position] for axis, position in enumerate(index)],
        )


class StationaryFokkerPlanck:
    """The stationary density of ``dx = f dt + G dW`` on a grid, in any dimension."""

    __slots__ = ["_convention", "_drift", "_grid", "_noise"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        convention: NoiseConvention,
        grid: DensityGrid,
    ) -> None:
        self._drift = drift
        self._noise = noise
        self._convention = convention
        self._grid = grid

    def solve(self) -> GridDensity:
        """Solve ``div J = 0`` with reflecting boundaries and unit mass."""
        grid = self._grid
        points = grid.points()
        cells = self._diffusion(points)
        _check_elliptic(cells)
        builder = _Discretisation(grid, cells)
        for axis in range(grid.dimension):
            low, high = _face_cells(grid, axis)
            faces = _HALF * (points[low.ravel()] + points[high.ravel()])
            drift, diffusion = self._face_coefficients(faces, axis)
            builder.add_axis(axis, low, high, drift, diffusion)
        return GridDensity(grid, _stationary_solution(builder.triplets(), grid))

    def _diffusion(self, points: np.ndarray) -> np.ndarray:
        return np.array(
            [diffusion_matrix(self._noise.columns_at(state)) for state in points],
        )

    def _face_coefficients(
        self,
        faces: np.ndarray,
        axis: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ito drift component and ``D_dd`` at the face centres, in one pass.

        The two are gathered together because both need the noise columns at the same
        points, and the Stratonovich shift needs their Jacobians as well; evaluating
        each field once per face rather than once per quantity is the difference
        between a solve of a second and a solve of several.
        """
        factor = stratonovich_factor(self._convention)
        drift = np.empty(faces.shape[0])
        diffusion = np.empty(faces.shape[0])
        for row, state in enumerate(faces):
            columns = self._noise.columns_at(state)
            value = field_at(self._drift, state)[axis]
            if factor != 0.0:
                jacobians = self._noise.jacobians_at(state)
                value += state_drift_shift(jacobians, columns, factor)[axis]
            drift[row] = value
            diffusion[row] = diffusion_matrix(columns)[axis, axis]
        return drift, diffusion


def _check_elliptic(cells: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(cells)
    scale = float(np.max(eigenvalues))
    if float(np.min(eigenvalues)) <= _ELLIPTICITY_FLOOR * max(scale, 1.0):
        message = (
            "the diffusion matrix G G^T is singular somewhere on the grid; a "
            "degenerate diffusion (scalar noise, or noise on a subset of the "
            "components) has no density this discretisation can represent"
        )
        raise ValueError(message)


def _face_cells(grid: DensityGrid, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat indices of the cell pairs either side of every face normal to ``axis``."""
    indices = np.arange(grid.size).reshape(grid.shape)
    count = grid.shape[axis]
    return (
        np.take(indices, np.arange(count - 1), axis=axis),
        np.take(indices, np.arange(1, count), axis=axis),
    )


class _Discretisation:
    """Accumulate the sparse divergence operator one family of faces at a time."""

    __slots__ = ["_cells", "_columns", "_grid", "_rows", "_values"]

    def __init__(self, grid: DensityGrid, cells: np.ndarray) -> None:
        self._grid = grid
        self._cells = cells
        self._rows: list[np.ndarray] = []
        self._columns: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

    def add_axis(
        self,
        axis: int,
        low: np.ndarray,
        high: np.ndarray,
        drift: np.ndarray,
        face_diffusion: np.ndarray,
    ) -> None:
        """Add every flux through the faces normal to ``axis``."""
        spacing = float(self._grid.spacings[axis])
        own = self._cells[:, axis, axis]
        exponent = 2.0 * drift * spacing / face_diffusion
        scale = 1.0 / (2.0 * spacing * spacing)
        self._add(low, high, low, _bernoulli(-exponent) * own[low.ravel()] * scale)
        self._add(low, high, high, -_bernoulli(exponent) * own[high.ravel()] * scale)
        for other in range(self._grid.dimension):
            if other != axis:
                self._add_cross(axis, other, low, high)

    def _add_cross(
        self,
        axis: int,
        other: int,
        low: np.ndarray,
        high: np.ndarray,
    ) -> None:
        """Add ``-(1/2) d_other (D[axis, other] p)`` to the flux normal to ``axis``."""
        coupling = self._cells[:, axis, other]
        shape = low.shape
        below = coupling[low.ravel()].reshape(shape)
        above = coupling[high.ravel()].reshape(shape)
        if np.max(np.abs(below)) + np.max(np.abs(above)) == 0.0:
            return
        count = self._grid.shape[other]
        forward = np.minimum(np.arange(count) + 1, count - 1)
        backward = np.maximum(np.arange(count) - 1, 0)
        reach = np.ones(self._grid.dimension, dtype=int)
        reach[other] = count
        width = np.broadcast_to(
            ((forward - backward) * self._grid.spacings[other]).reshape(reach),
            shape,
        )
        scale = -0.25 / (width * self._grid.spacings[axis])
        for cells, coupled, shift, sign in (
            (low, below, forward, 1.0),
            (high, above, forward, 1.0),
            (low, below, backward, -1.0),
            (high, above, backward, -1.0),
        ):
            self._add(
                low,
                high,
                np.take(cells, shift, axis=other),
                scale * sign * np.take(coupled, shift, axis=other),
            )

    def _add(
        self,
        low: np.ndarray,
        high: np.ndarray,
        column: np.ndarray,
        coefficient: np.ndarray,
    ) -> None:
        """A flux ``coefficient * p[column]`` leaving ``low`` and entering ``high``."""
        self._rows.append(np.ravel(low))
        self._columns.append(np.ravel(column))
        self._values.append(np.ravel(coefficient))
        self._rows.append(np.ravel(high))
        self._columns.append(np.ravel(column))
        self._values.append(-np.ravel(coefficient))

    def triplets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The operator as row, column and value arrays."""
        return (
            np.concatenate(self._rows),
            np.concatenate(self._columns),
            np.concatenate(self._values),
        )


def _stationary_solution(
    triplets: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid: DensityGrid,
) -> np.ndarray:
    """Replace the redundant conservation row by unit mass, then solve.

    Every flux enters one row with a plus and another with a minus and the same
    spacing, so the rows of the divergence operator sum to zero exactly: one of them
    carries no information the others do not. Overwriting it with the normalisation
    gives a square system whose solution is the stationary density itself, already
    scaled, with no eigenvalue iteration and no shift to choose.
    """
    rows, columns, values = triplets
    keep = rows != _ANCHOR_ROW
    size = grid.size
    operator = coo_matrix(
        (
            np.concatenate([values[keep], np.full(size, grid.cell_volume)]),
            (
                np.concatenate([rows[keep], np.full(size, _ANCHOR_ROW)]),
                np.concatenate([columns[keep], np.arange(size)]),
            ),
        ),
        shape=(size, size),
    ).tocsr()
    right = np.zeros(size)
    right[_ANCHOR_ROW] = 1.0
    return spsolve(operator, right)
