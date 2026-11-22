"""Stable and unstable manifolds of a cycle by Fourier-Taylor parameterization.

Castelli, Lessard and Mireles James, "Parameterization of invariant manifolds
for periodic orbits I: efficient numerics via the Floquet normal form" (SIAM J.
Appl. Dyn. Syst. 14, 2015). The manifold of a cycle ``gamma`` of period ``T``
tangent to the bundle of one real Floquet exponent ``lambda`` is sought as a
map ``P(theta, sigma)``, ``2T``-periodic in the phase, satisfying their
invariance equation (11)

    dP/dtheta + lambda sigma dP/dsigma = f(P),

which makes it a conjugacy: the flow carries ``P(theta, sigma)`` to
``P(theta + t, e^(lambda t) sigma)``. Written as ``P = sum_n a_n(theta) sigma^n``
the equation splits into the cycle (``n = 0``), the Floquet bundle (``n = 1``)
and the homological equations (18) for ``n >= 2``,

    a_n' + n lambda a_n - Df(gamma) a_n = R_n,

where ``R_n`` is the order-``n`` coefficient of ``f`` along the series so far.
In the Floquet basis ``Xi(theta) = [xi_1 ... xi_d]`` - every bundle, each
solving ``xi_j' + lambda_j xi_j = Df(gamma) xi_j``, which is the paper's
``Q(theta) M`` - the equation is diagonal and constant-coefficient: with
``a_n = Xi v_n``, Fourier mode ``k`` of component ``j`` is (21)

    v_(n,k,j) = (Xi^-1 R_n)_(k,j) / (i pi k / T + n lambda - lambda_j).

No integration of the field is involved anywhere. The cycle is refined by
Newton's method on its Fourier collocation; the bundles and exponents are the
real eigenpairs of the Fourier collocation of ``Df(gamma) - d/dtheta`` on
``2T``-periodic functions (Castelli et al. II, Assumption 2), so a bundle that
twists - a Mobius band, multiplier negative - is as periodic as one that does
not; and ``R_n`` is evaluated pseudo-spectrally, by running the field on
:class:`~.fourier_taylor.GridSeries` at every phase of the grid at once.

One manifold coordinate is supported: a chart spans the bundle of a single real
exponent, which is a two-dimensional manifold - the tube around a saddle cycle
in three dimensions, the band around a repelling cycle in the plane. Complex
Floquet multipliers, and manifolds of several exponents, are not covered.
"""

import numpy as np

from discrecontinual_equations.continuation.fourier_taylor import (
    evaluate_field,
    field_jacobians,
)
from discrecontinual_equations.continuation.manifold import ManifoldSelection
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.function.function import Function

_DEFAULT_ORDER = 10
_DEFAULT_MODES = 48
_NEWTON_ITERATIONS = 30
# The cycle's Fourier collocation is converged when the Newton update is this
# small relative to the orbit's size.
_NEWTON_TOLERANCE = 1.0e-13
# A Galerkin eigenvalue whose imaginary part is below this (relative to the
# eigenvalue's size, at least one) is a real Floquet exponent.
_REAL_EXPONENT = 1.0e-7
# A Galerkin eigenvector holding more than this fraction of its size in its
# top third of Fourier modes is not resolved - a spurious eigenpair of the
# truncation, not a Floquet bundle. Spurious pairs live almost wholly up
# there, so the bar only has to separate O(1) from small.
_RESOLVED_TAIL = 1.0e-2
_TAIL_FRACTION = 3
# Inverse-iteration sweeps that polish each bundle after the dense solve.
_POLISH_SWEEPS = 2
# The exponent nearest zero is the trivial one; one further than this from
# zero, relative to the largest, means the orbit was not a cycle at all.
_TRIVIAL = 1.0e-6
# Two exponents closer than this (relative) make ``n lambda - lambda_j`` vanish:
# a resonance, for which the invariance equation has no solution.
_RESONANT = 1.0e-9
# The a-posteriori defect is expanded to this multiple of the chart's order.
_DEFECT_ORDER_FACTOR = 2


def _wavenumbers(points: int) -> np.ndarray:
    """Signed integer wavenumbers of an ``points``-point FFT."""
    return np.fft.fftfreq(points, 1.0 / points)


def _differentiation(points: int, length: float) -> np.ndarray:
    """The Fourier differentiation matrix on ``points`` nodes over ``length``.

    The node count is odd, so there is no Nyquist mode whose derivative the
    grid cannot represent.
    """
    frequencies = 2.0j * np.pi * _wavenumbers(points) / length
    identity = np.eye(points)
    spectrum = np.fft.fft(identity, axis=0)
    return np.real(np.fft.ifft(frequencies[:, None] * spectrum, axis=0))


def _interpolate(
    values: np.ndarray,
    length: float,
    phases: np.ndarray,
    derivative: int = 0,
) -> np.ndarray:
    """Evaluate the trigonometric interpolant of grid ``values`` at ``phases``.

    ``values`` is ``[points, ...]`` on the uniform grid over ``[0, length)``.
    """
    points = values.shape[0]
    spectrum = np.fft.fft(values, axis=0) / points
    frequencies = 2.0j * np.pi * _wavenumbers(points) / length
    basis = np.exp(np.outer(np.atleast_1d(phases), frequencies))
    if derivative:
        basis = basis * frequencies**derivative
    flat = spectrum.reshape(points, -1)
    result = np.real(basis @ flat)
    return result.reshape((np.atleast_1d(phases).size, *values.shape[1:]))


def _grid_field(function: Function, states: np.ndarray) -> np.ndarray:
    return evaluate_field(function, states[None, :, :])[0]


class FourierCycle:
    """A cycle as its values on a uniform phase grid over one period.

    ``values[j]`` is the state at time ``j T / points`` along the orbit; the
    trigonometric interpolant through them is the cycle between nodes.
    """

    __slots__ = ["_period", "_values"]

    def __init__(self, values: np.ndarray, period: float) -> None:
        self._values = values
        self._period = period

    @property
    def values(self) -> np.ndarray:
        """Node states, ``[points, d]``, uniform in time over one period."""
        return self._values

    @property
    def period(self) -> float:
        """The cycle's period."""
        return self._period

    def point(self, phase: float) -> np.ndarray:
        """The state at time ``phase`` along the cycle."""
        return _interpolate(self._values, self._period, np.array([phase]))[0]

    def sample(self, phases: np.ndarray, derivative: int = 0) -> np.ndarray:
        """States (or their time derivatives) at each of ``phases``."""
        return _interpolate(self._values, self._period, phases, derivative)

    def residual(self, function: Function) -> float:
        """Largest defect ``|gamma' - f(gamma)|`` at the nodes."""
        slope = self.sample(
            np.arange(self._values.shape[0]) * self._period / self._values.shape[0],
            derivative=1,
        )
        return float(np.max(np.abs(slope - _grid_field(function, self._values))))

    @classmethod
    def refine(
        cls,
        function: Function,
        solution: PeriodicOrbitSolution,
        harmonics: int,
        mesh: np.ndarray | None = None,
    ) -> "FourierCycle":
        """Refine a collocation cycle to a Fourier one with ``harmonics`` modes.

        The collocation states - from any of the library's cycle solvers, on
        ``mesh`` in rescaled time (uniform when ``None``) - are interpolated
        onto ``2 * harmonics + 1`` uniform nodes, then ``s``-scaled Fourier
        collocation ``D x = T f(x)`` is solved by Newton's method with the
        integral phase condition against the seed. A cycle resolved by its
        Fourier modes converges spectrally, so the result is far more accurate
        than the collocation it starts from.
        """
        states = np.asarray(solution.states, dtype=float)
        period = float(solution.period)
        nodes = states.shape[0]
        positions = np.linspace(0.0, 1.0, nodes) if mesh is None else np.asarray(mesh)
        points = 2 * harmonics + 1
        grid = np.arange(points) / points
        seed = np.column_stack(
            [
                np.interp(grid, positions, states[:, index])
                for index in range(states.shape[1])
            ],
        )
        return cls._newton(function, seed, period)

    @classmethod
    def _newton(
        cls,
        function: Function,
        seed: np.ndarray,
        period: float,
    ) -> "FourierCycle":
        points, dimension = seed.shape
        derivative = _differentiation(points, 1.0)
        reference = derivative @ seed
        values = seed.copy()
        size = points * dimension
        for _ in range(_NEWTON_ITERATIONS):
            field = _grid_field(function, values)
            residual = np.concatenate(
                [
                    (derivative @ values - period * field).ravel(),
                    [float(np.sum((values - seed) * reference)) / points],
                ],
            )
            jacobians = field_jacobians(function, values)
            matrix = np.zeros((size + 1, size + 1))
            matrix[:size, :size] = np.kron(derivative, np.eye(dimension))
            for node in range(points):
                block = slice(node * dimension, (node + 1) * dimension)
                matrix[block, block] -= period * jacobians[node]
            matrix[:size, size] = -field.ravel()
            matrix[size, :size] = reference.ravel() / points
            update = np.linalg.solve(matrix, -residual)
            values = values + update[:size].reshape(points, dimension)
            period += float(update[size])
            scale = max(1.0, float(np.max(np.abs(values))))
            if np.max(np.abs(update)) < _NEWTON_TOLERANCE * scale:
                break
        else:
            message = "the cycle's Fourier collocation did not converge"
            raise ValueError(message)
        return cls(values, period)


class FloquetBundles:
    """The real Floquet exponents of a cycle and their ``2T``-periodic bundles.

    ``bundles[j, :, i]`` is ``xi_i`` at node ``j`` of the uniform grid over
    ``[0, 2T)``: the solution of ``xi' + lambda_i xi = Df(gamma) xi`` with
    exponent ``exponents[i]``, normalised to unit sup norm. ``orientable[i]``
    says whether ``xi_i(theta + T) = xi_i(theta)`` (an annulus) rather than
    ``-xi_i(theta)`` (a Mobius band, Floquet multiplier negative).
    """

    __slots__ = ["_bundles", "_exponents", "_orientable", "_period"]

    def __init__(
        self,
        exponents: np.ndarray,
        bundles: np.ndarray,
        orientable: list[bool],
        period: float,
    ) -> None:
        self._exponents = exponents
        self._bundles = bundles
        self._orientable = orientable
        self._period = period

    @property
    def exponents(self) -> np.ndarray:
        """Real Floquet exponents, ascending; ``exp(lambda T)`` up to sign."""
        return self._exponents

    @property
    def bundles(self) -> np.ndarray:
        """``[points, d, d]``: column ``i`` at each node is ``xi_i``."""
        return self._bundles

    @property
    def orientable(self) -> list[bool]:
        """Per exponent: ``True`` for an annulus, ``False`` for a Mobius band."""
        return self._orientable

    @property
    def multipliers(self) -> np.ndarray:
        """Floquet multipliers ``+-exp(lambda T)``, negative where twisted."""
        signs = np.array([1.0 if flag else -1.0 for flag in self._orientable])
        return signs * np.exp(self._exponents * self._period)

    @classmethod
    def compute(
        cls, function: Function, cycle: FourierCycle, harmonics: int
    ) -> "FloquetBundles":
        """The real eigenpairs of ``Df(gamma) - d/dtheta`` on ``2T``-periodic grids.

        Its spectrum is every ``lambda_i + i pi k / T``; the real eigenvalues
        (``k = 0``) are the Floquet exponents, one per dimension when the
        multipliers are real. The dense eigensolve is polished by inverse
        iteration, and eigenpairs whose eigenvectors live in the top Fourier
        modes - artefacts of the truncation - are rejected.
        """
        period = cycle.period
        points = 2 * harmonics + 1
        length = 2.0 * period
        phases = np.arange(points) * length / points
        orbit = cycle.sample(phases)
        jacobians = field_jacobians(function, orbit)
        dimension = orbit.shape[1]
        operator = -np.kron(_differentiation(points, length), np.eye(dimension))
        for node in range(points):
            block = slice(node * dimension, (node + 1) * dimension)
            operator[block, block] += jacobians[node]
        values, vectors = np.linalg.eig(operator)
        candidates: list[tuple[float, np.ndarray]] = []
        for index in np.argsort(values.real):
            value = values[index]
            if abs(value.imag) > _REAL_EXPONENT * max(1.0, abs(value)):
                continue
            vector = vectors[:, index].reshape(points, dimension)
            if not _resolved(vector):
                continue
            exponent, polished = _polish(operator, float(value.real), vector.ravel())
            candidates.append((exponent, polished.reshape(points, dimension)))
        if len(candidates) != dimension:
            message = (
                f"found {len(candidates)} real Floquet exponents for a "
                f"{dimension}-dimensional cycle: the multipliers are complex or "
                "the bundles need more Fourier modes"
            )
            raise ValueError(message)
        candidates.sort(key=lambda pair: pair[0])
        exponents = np.array([pair[0] for pair in candidates])
        bundles = np.empty((points, dimension, dimension))
        orientable: list[bool] = []
        for column, (_, vector) in enumerate(candidates):
            bundles[:, :, column] = vector
            shifted = _interpolate(vector, length, phases + period)
            orientable.append(
                bool(
                    np.max(np.abs(shifted - vector)) < np.max(np.abs(shifted + vector))
                ),
            )
        return cls(exponents, bundles, orientable, period)


def _resolved(vector: np.ndarray) -> bool:
    spectrum = np.abs(np.fft.fft(vector, axis=0))
    wavenumbers = np.abs(_wavenumbers(vector.shape[0]))
    tail = wavenumbers > (vector.shape[0] // 2) * (1.0 - 1.0 / _TAIL_FRACTION)
    return float(spectrum[tail].sum()) <= _RESOLVED_TAIL * float(spectrum.sum())


def _polish(
    operator: np.ndarray,
    exponent: float,
    vector: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Real-normalise an eigenvector and sharpen it by inverse iteration."""
    vector = vector / vector[np.argmax(np.abs(vector))]
    vector = np.real(vector)
    shift = exponent
    identity = np.eye(operator.shape[0])
    for _ in range(_POLISH_SWEEPS):
        try:
            vector = np.linalg.solve(operator - shift * identity, vector)
        except np.linalg.LinAlgError:
            break
        vector = vector / np.max(np.abs(vector))
        image = operator @ vector
        shift = float(vector @ image / (vector @ vector))
    sign = np.sign(vector[np.argmax(np.abs(vector))])
    return shift, sign * vector / np.max(np.abs(vector))


class CycleManifoldChart:
    """A Fourier-Taylor parameterization ``P(theta, sigma)`` of a cycle manifold.

    ``theta`` is time along the cycle, ``2T``-periodic; ``sigma`` the manifold
    coordinate. The flow carries ``P(theta, sigma)`` to
    ``P(theta + t, e^(exponent t) sigma)``, forward in time on a stable
    manifold and backward on an unstable one. Coefficients are held as values
    on the uniform grid of ``points`` nodes over ``[0, 2T)``.
    """

    __slots__ = [
        "_defect",
        "_exponent",
        "_function",
        "_grid",
        "_orientable",
        "_period",
    ]

    def __init__(  # noqa: PLR0913 (the chart's defining data)
        self,
        function: Function,
        grid: np.ndarray,
        exponent: float,
        period: float,
        orientable: bool,  # noqa: FBT001 (the chart's own datum, not a switch)
        defect: np.ndarray | None = None,
    ) -> None:
        self._function = function
        self._grid = grid
        self._exponent = exponent
        self._period = period
        self._orientable = orientable
        self._defect = defect

    @property
    def exponent(self) -> float:
        """The Floquet exponent the chart's ``sigma`` contracts or grows with."""
        return self._exponent

    @property
    def period(self) -> float:
        """The cycle's period ``T``; the chart is ``2T``-periodic in ``theta``."""
        return self._period

    @property
    def order(self) -> int:
        """Truncation order in ``sigma``."""
        return self._grid.shape[0] - 1

    @property
    def orientable(self) -> bool:
        """``False`` when the manifold is a Mobius band (multiplier negative)."""
        return self._orientable

    @property
    def grid(self) -> np.ndarray:
        """``[order + 1, points, d]``: ``a_n`` at the nodes of ``[0, 2T)``."""
        return self._grid

    @property
    def coefficients(self) -> np.ndarray:
        """The Fourier-Taylor coefficients ``a_(n,k)``, ``[order + 1, 2K + 1, d]``.

        ``P(theta, sigma) = sum_n sum_k a_(n,k) exp(i pi k theta / T) sigma^n``,
        with ``k`` running ``-K .. K`` along the second axis.
        """
        points = self._grid.shape[1]
        spectrum = np.fft.fft(self._grid, axis=1) / points
        return np.fft.fftshift(spectrum, axes=1)

    def _taylor(self, theta: float, derivative: int = 0) -> np.ndarray:
        """``a_n(theta)`` (or its ``theta`` derivative), ``[order + 1, d]``."""
        moved = np.moveaxis(self._grid, 1, 0)
        return _interpolate(moved, 2.0 * self._period, np.array([theta]), derivative)[0]

    def point(self, theta: float, sigma: float) -> np.ndarray:
        """``P(theta, sigma)``: the state on the manifold."""
        powers = sigma ** np.arange(self.order + 1)
        return powers @ self._taylor(theta)

    def tangent(self, theta: float, sigma: float) -> np.ndarray:
        """``dP/dtheta`` at ``(theta, sigma)``: the chart's phase direction."""
        powers = sigma ** np.arange(self.order + 1)
        return powers @ self._taylor(theta, derivative=1)

    def flow_image(self, theta: float, sigma: float, time: float) -> np.ndarray:
        """Where the flow carries ``P(theta, sigma)`` in ``time``, by the conjugacy."""
        return self.point(theta + time, sigma * np.exp(self._exponent * time))

    def defect(self, theta: float, sigma: float) -> np.ndarray:
        """``dP/dtheta + lambda sigma dP/dsigma - f(P)`` at one point, on floats."""
        powers = sigma ** np.arange(self.order + 1)
        orders = np.arange(self.order + 1)
        along = powers @ self._taylor(theta, derivative=1)
        across = (orders * powers) @ self._taylor(theta)
        state = powers @ self._taylor(theta)
        field = np.array(self._function.eval(point=list(state), time=None), dtype=float)
        return along + self._exponent * across - field

    def invariance_error(self, radius: float) -> float:
        """The a-posteriori error (28) of Castelli et al. on ``|sigma| <= radius``.

        The weighted Wiener norm ``sum_n sum_k ||E_(n,k)|| radius^n`` of the
        defect of the invariance equation, with ``f(P)`` expanded to twice the
        chart's order. It bounds the defect's sup norm on the chart.
        """
        if self._defect is None:
            self._defect = self._defect_coefficients()
        weights = radius ** np.arange(self._defect.shape[0])
        spectrum = np.abs(np.fft.fft(self._defect, axis=1)) / self._defect.shape[1]
        per_order = np.sum(np.max(spectrum, axis=2), axis=1)
        return float(per_order @ weights)

    def _defect_coefficients(self) -> np.ndarray:
        order, points, dimension = self._grid.shape
        extended = np.zeros((_DEFECT_ORDER_FACTOR * (order - 1) + 1, points, dimension))
        extended[:order] = self._grid
        field = evaluate_field(self._function, extended)
        along = _interpolate(
            extended.transpose(1, 0, 2),
            2.0 * self._period,
            np.arange(points) * 2.0 * self._period / points,
            derivative=1,
        ).transpose(1, 0, 2)
        orders = np.arange(extended.shape[0])[:, None, None]
        return along + self._exponent * orders * extended - field

    def radius(self, tolerance: float) -> float:
        """The ``sigma`` at which the last retained term is ``tolerance`` in size.

        The tail of a convergent series is about its first neglected term, so
        this is where the truncated chart stops being trustworthy to
        ``tolerance``: the start radius for globalising the manifold.
        """
        last = float(np.max(np.abs(self._grid[-1])))
        if last == 0.0:
            return float("inf")
        return float((tolerance / last) ** (1.0 / self.order))

    def rescaled(self, factor: float) -> "CycleManifoldChart":
        """The same manifold with ``sigma`` stretched: ``a_n -> factor^n a_n``."""
        weights = factor ** np.arange(self.order + 1)
        return CycleManifoldChart(
            self._function,
            self._grid * weights[:, None, None],
            self._exponent,
            self._period,
            self._orientable,
        )


class CycleManifold:
    """Compute the Fourier-Taylor chart of a cycle's stable or unstable manifold.

    ``order`` is the Taylor order in ``sigma``; ``harmonics`` the number of
    Fourier modes ``K`` on the ``2T``-periodic grid of ``2K + 1`` nodes (the
    cycle itself carries ``K // 2`` modes of its own period). ``scale`` is the
    sup norm of the first-order term, which sets how far ``sigma = 1``
    reaches: the parameterization is unique up to that scaling (Castelli et al.
    I, Remark 2.8).
    """

    __slots__ = ["_harmonics", "_order", "_scale"]

    def __init__(
        self,
        order: int = _DEFAULT_ORDER,
        harmonics: int = _DEFAULT_MODES,
        scale: float = 1.0,
    ) -> None:
        self._order = order
        self._harmonics = harmonics
        self._scale = scale

    def compute(
        self,
        function: Function,
        solution: PeriodicOrbitSolution,
        selection: ManifoldSelection,
        mesh: np.ndarray | None = None,
    ) -> CycleManifoldChart:
        """The chart of the manifold tangent to the selected bundle.

        ``solution`` and ``mesh`` are a cycle from any of the library's cycle
        solvers - a :class:`~.cycle_continuation.CyclePoint`'s ``solution`` and
        ``mesh``, say. ``selection`` picks the exponent by sign, among the
        nontrivial ones, and must pick exactly one.
        """
        cycle = FourierCycle.refine(function, solution, self._harmonics // 2, mesh)
        return self.compute_on(function, cycle, selection)

    def compute_on(
        self,
        function: Function,
        cycle: FourierCycle,
        selection: ManifoldSelection,
    ) -> CycleManifoldChart:
        """The chart, from a cycle already refined to Fourier form."""
        bundles = FloquetBundles.compute(function, cycle, self._harmonics)
        exponents = bundles.exponents
        largest = max(1.0, float(np.max(np.abs(exponents))))
        trivial = int(np.argmin(np.abs(exponents)))
        if abs(exponents[trivial]) > _TRIVIAL * largest:
            message = "no trivial Floquet exponent: the orbit is not a cycle"
            raise ValueError(message)
        nontrivial = np.array(
            [np.nan if i == trivial else exponents[i] for i in range(exponents.size)],
        )
        chosen = [
            i
            for i in selection.indices(np.nan_to_num(nontrivial, nan=0.0))
            if i != trivial
        ]
        if len(chosen) != 1:
            message = (
                f"the selection picks {len(chosen)} exponents; a chart spans "
                "exactly one"
            )
            raise ValueError(message)
        index = chosen[0]
        exponent = float(exponents[index])
        grid = self._solve(function, cycle, bundles, index)
        return CycleManifoldChart(
            function,
            grid,
            exponent,
            cycle.period,
            bundles.orientable[index],
        )

    def _solve(
        self,
        function: Function,
        cycle: FourierCycle,
        bundles: FloquetBundles,
        index: int,
    ) -> np.ndarray:
        period = cycle.period
        frame = bundles.bundles
        points, dimension, _ = frame.shape
        phases = np.arange(points) * 2.0 * period / points
        grid = np.zeros((self._order + 1, points, dimension))
        grid[0] = cycle.sample(phases)
        grid[1] = self._scale * frame[:, :, index]
        if self._order < 2:  # noqa: PLR2004 (the first homological order)
            return grid
        inverse = np.linalg.inv(frame)
        exponents = bundles.exponents
        exponent = exponents[index]
        frequencies = 1.0j * np.pi * _wavenumbers(points) / period
        for order in range(2, self._order + 1):
            rates = order * exponent - exponents
            if np.min(np.abs(rates)) < _RESONANT * max(1.0, abs(order * exponent)):
                message = (
                    f"resonance at order {order}: {order} x {exponent:.6g} is a "
                    "Floquet exponent, and the invariance equation has no solution"
                )
                raise ValueError(message)
            remainder = evaluate_field(function, grid[: order + 1])[order]
            projected = np.einsum("jab,jb->ja", inverse, remainder)
            spectrum = np.fft.fft(projected, axis=0)
            spectrum /= frequencies[:, None] + rates[None, :]
            solved = np.real(np.fft.ifft(spectrum, axis=0))
            grid[order] = np.einsum("jab,jb->ja", frame, solved)
        return grid
