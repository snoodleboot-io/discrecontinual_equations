"""The noise amplitude of a stochastic flow, once the state is a vector.

In one dimension a stochastic differential equation needs only two scalar fields and
the noise term is unambiguous. Above one dimension it is not: ``dx = f(x) dt +
G(x) dW`` has an ``N x K`` amplitude driven by ``K`` *independent* Brownian motions,
and the number of drivers matters as much as their size. One shared driver applied
to every component (what :func:`~.lyapunov.stochastic_lyapunov` calls scalar noise)
gives ``D = G G^T`` of rank one, so the noise pushes the state along a single
direction at every point; the Fokker-Planck operator is then degenerate and has no
smooth stationary density at all. Independent drivers per component give a full-rank
``D`` and a genuine planar density. The same system therefore cannot be asked both
questions unless the amplitude is carried as a matrix, which is what this module
exists to do.

The columns are held as ordinary :class:`~.function.function.Function` objects, one
per driver, so every field already written for the library can be reused as a noise
column without a new evaluation protocol.

The solvers read the same amplitude differently: a
:class:`~.function.stochastic.StochasticFunction` whose ``diffusion()`` returns the
``N x K`` matrix ``G``. :class:`NoiseMatrixFunction` and :func:`noise_matrix_from`
carry a system from one form to the other, so it is written once (DEQ-44). Neither
form carries the calculus: the analysis side takes a
:class:`~.stochastic.NoiseConvention`, the solvers a ``calculus`` in their
configuration, and the two must be chosen to agree (``Ito()`` with ``"ito"``).
"""

from collections.abc import Sequence

import numpy as np

from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.function.stochastic import StochasticFunction

_JACOBIAN_STEP = 1.0e-7
_UNIT = 1.0
_VECTOR = 1


def stratonovich_factor(convention: NoiseConvention) -> float:
    """Read the ``1/2`` that Stratonovich contributes off the scalar correction.

    :class:`~.stochastic.NoiseConvention` was written for the scalar case, where the
    correction is ``factor * g g'``. Evaluating it at ``g = g' = 1`` recovers the
    factor itself, so the two existing conventions drive the vector case as well
    without a parallel class hierarchy that could disagree with the scalar one.
    """
    return convention.drift_correction(_UNIT, _UNIT)


def field_at(function: Function, state: np.ndarray) -> np.ndarray:
    """Evaluate a vector field at a state."""
    return np.array(function.eval(point=list(state), time=None), dtype=float)


def jacobian_at(function: Function, state: np.ndarray) -> np.ndarray:
    """Forward-difference Jacobian of a vector field, stepped relative to the state.

    The step is scaled by the size of the component it perturbs, floored at one. A
    fixed absolute step is a silent trap here: the reference orbit of a stochastic
    Benettin estimate need not be bounded - the linear multiplicative system above its
    dynamical threshold grows like ``exp(lambda t)`` and reaches ``1e30`` over a
    perfectly ordinary horizon - and once the state exceeds ``step / eps`` the
    perturbed state rounds back to the state itself, the difference is exactly zero,
    and the Jacobian comes back as a zero matrix. Nothing raises; the exponent just
    quietly stops depending on the parameter. Scaling the step keeps it meaningful at
    any magnitude.
    """
    base = field_at(function, state)
    dimension = state.size
    steps = _JACOBIAN_STEP * np.maximum(np.abs(state), 1.0)
    columns = np.empty((base.size, dimension))
    for j in range(dimension):
        shifted = state.copy()
        shifted[j] += steps[j]
        columns[:, j] = (field_at(function, shifted) - base) / steps[j]
    return columns


class NoiseMatrix:
    """The amplitude ``G`` of ``dx = f dt + G dW``, one ``Function`` per driver.

    A single column reproduces scalar noise exactly, so this is a strict
    generalisation rather than a second convention. The class deliberately offers only
    the two raw evaluations - the columns, and their Jacobians - and leaves the
    algebra built from them to the free functions below. Callers in this package
    evaluate ``G`` tens of thousands of times per answer, and a convenience method
    that recomputed the columns inside a Jacobian-using formula would double that
    cost invisibly.
    """

    __slots__ = ["_columns"]

    def __init__(self, columns: Sequence[Function]) -> None:
        if len(columns) == 0:
            message = "a noise matrix needs at least one driver"
            raise ValueError(message)
        self._columns = list(columns)

    @property
    def drivers(self) -> int:
        """Number of independent Brownian motions, ``K``."""
        return len(self._columns)

    def columns_at(self, state: np.ndarray) -> np.ndarray:
        """The ``K`` columns of ``G(x)`` as a ``(K, N)`` array."""
        return np.array([field_at(column, state) for column in self._columns])

    def jacobians_at(self, state: np.ndarray) -> np.ndarray:
        """The Jacobian of each column, as a ``(K, N, N)`` array."""
        return np.array([jacobian_at(column, state) for column in self._columns])


class NoiseMatrixFunction(StochasticFunction):
    """A drift ``Function`` and a :class:`NoiseMatrix` as one ``StochasticFunction``.

    The analysis side's two objects, handed to the solvers unchanged: ``eval`` is the
    drift's own, and ``diffusion`` is ``G`` as the ``N x K`` matrix the solvers read,
    built from :meth:`NoiseMatrix.columns_at` so that it is the very matrix the
    Fokker-Planck operator and the matrix-noise exponent see, not a re-derivation of
    it. Variables and parameters are the drift's, so a parameter changed on the drift
    reaches the solvers too. Like the rest of this module the columns are evaluated
    without a time; the drift receives the solver's time as it always would.
    """

    __slots__ = ["_drift", "_noise"]

    def __init__(self, drift: Function, noise: NoiseMatrix) -> None:
        super().__init__(
            drift.variables,
            drift.parameters,
            drift.curve.results,
            drift.time,
        )
        self._drift = drift
        self._noise = noise

    @property
    def drift(self) -> Function:
        """The drift ``f``, as given."""
        return self._drift

    @property
    def noise(self) -> NoiseMatrix:
        """The amplitude ``G``, as given."""
        return self._noise

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """The drift at ``point``, exactly as the drift computes it."""
        return self._drift.eval(point=point, time=time)

    def diffusion(  # type: ignore[override]
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (the columns are autonomous)
    ) -> list[list[float]]:
        """``G(x)`` as ``N`` rows of ``K`` entries: column ``k`` is driver ``k``."""
        columns = self._noise.columns_at(np.asarray(point, dtype=float))
        return [[float(value) for value in row] for row in columns.T]


class _DiffusionColumn(Function):
    """Column ``index`` of a ``StochasticFunction``'s diffusion, as a vector field.

    A vector-form diffusion ``b`` is the diagonal ``G = diag(b)``, so its column
    ``k`` is ``b_k e_k``; a matrix-form one is read as it stands, and its width must
    be the driver count the noise matrix was built with.
    """

    __slots__ = ["_drivers", "_index", "_source"]

    def __init__(self, source: StochasticFunction, index: int, drivers: int) -> None:
        super().__init__(
            source.variables,
            source.parameters,
            source.curve.results,
            source.time,
        )
        self._source = source
        self._index = index
        self._drivers = drivers

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        amplitude = np.asarray(self._source.diffusion(point=point, time=time))
        vector = amplitude.ndim == _VECTOR
        width = amplitude.shape[-1]
        if width != self._drivers:
            message = (
                f"the noise matrix was built with {self._drivers} drivers but "
                f"diffusion() has {width}"
            )
            raise ValueError(message)
        if vector:
            column = np.zeros(width)
            column[self._index] = amplitude[self._index]
            return [float(value) for value in column]
        return [float(value) for value in amplitude[:, self._index]]


def noise_matrix_from(
    function: StochasticFunction,
    drivers: int | None = None,
) -> NoiseMatrix:
    """The :class:`NoiseMatrix` of a solver's ``StochasticFunction``.

    The reverse of :class:`NoiseMatrixFunction`: ``function`` itself serves as the
    drift (its ``eval`` is the drift), and this supplies the amplitude, one column
    per driver, each evaluated from ``diffusion()`` so the two can never disagree.
    ``drivers`` is ``K``, the width of a matrix-form diffusion; it defaults to the
    state dimension, which is what a vector-form (diagonal) diffusion needs. A width
    that does not match is reported when the columns are first evaluated, since the
    shape of ``diffusion()`` is only known then.
    """
    count = len(function.variables) if drivers is None else drivers
    return NoiseMatrix([_DiffusionColumn(function, k, count) for k in range(count)])


def diffusion_matrix(columns: np.ndarray) -> np.ndarray:
    """``D = G G^T``, the only part of the amplitude a density can see.

    Two different amplitudes with the same product give the same density, so the
    drivers are not recoverable from it. That is why the thing to check before asking
    a system for a density is the rank of ``D`` and not the number of columns.
    """
    return columns.T @ columns


def state_drift_shift(
    jacobians: np.ndarray,
    columns: np.ndarray,
    factor: float,
) -> np.ndarray:
    """``factor * sum_k (dG_k/dx) G_k``: what the convention adds to the drift."""
    return factor * np.einsum("kij,kj->i", jacobians, columns)


def variational_drift_shift(jacobians: np.ndarray, factor: float) -> np.ndarray:
    """``factor * sum_k (dG_k/dx)^2``: what it adds to the variational drift.

    The two shifts are the same sum over drivers seen from the state and from the
    tangent frame, and they must come from the same convention factor or the exponent
    and the density would be reported for two different systems.
    """
    return factor * np.einsum("kij,kjl->il", jacobians, jacobians)
