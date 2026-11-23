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

from discrecontinual_equations.continuation.jacobian_field import JacobianField
from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.function.stochastic import StochasticFunction

# Dennis and Schnabel's relative steps: sqrt(eps) balances a forward difference's
# truncation against its rounding, eps^(1/3) a central difference's.
_FORWARD_STEP = float(np.sqrt(np.finfo(float).eps))
_CENTRAL_STEP = float(np.cbrt(np.finfo(float).eps))
_UNIT = 1.0
_VECTOR = 1
# A batch of vector fields is one row per point.
_BATCH_AXES = 2


def stratonovich_factor(convention: NoiseConvention) -> float:
    """Read the ``1/2`` that Stratonovich contributes off the scalar correction.

    :class:`~.stochastic.NoiseConvention` was written for the scalar case, where the
    correction is ``factor * g g'``. Evaluating it at ``g = g' = 1`` recovers the
    factor itself, so the two existing conventions drive the vector case as well
    without a parallel class hierarchy that could disagree with the scalar one.
    """
    return convention.drift_correction(_UNIT, _UNIT)


def field_at(function: Function, state: np.ndarray) -> np.ndarray:
    """Evaluate a vector field at a state.

    The state is handed over as Python floats: arithmetic on NumPy scalars rounds
    the same but costs several times as much, and a field is evaluated
    ``O(n)`` times per Jacobian.
    """
    point = np.asarray(state, dtype=float).tolist()
    return np.array(function.eval(point=point, time=None), dtype=float)


def fields_at(function: Function, states: np.ndarray) -> np.ndarray:
    """Evaluate a vector field at every row of ``states``, as ``(count, n)``.

    Row ``p`` equals :func:`field_at` at ``states[p]`` exactly; the evaluation goes
    through :meth:`Function.eval_batch`, so a field with array arithmetic does it in
    one call. A batch whose shape is not one row per state is refused rather than
    broadcast: an ``eval_batch`` written for one point at a time, returning the
    ``n`` components as a flat vector, would otherwise pass for a column.
    """
    states = np.asarray(states, dtype=float)
    values = np.asarray(function.eval_batch(states), dtype=float)
    if values.ndim != _BATCH_AXES or values.shape[0] != states.shape[0]:
        message = (
            f"eval_batch returned shape {values.shape} for {states.shape[0]} states; "
            f"it must return one row per state"
        )
        raise ValueError(message)
    return values


def jacobian_at(
    function: Function,
    state: np.ndarray,
    base: np.ndarray | None = None,
    *,
    typical: float | np.ndarray | None = None,
    central: bool = False,
) -> np.ndarray:
    """Jacobian of a vector field, exact if it carries one, else finite-differenced.

    A :class:`~.jacobian_field.JacobianField` returns its own Jacobian, and nothing
    else here applies. Otherwise the field is differenced with the steps of Dennis
    and Schnabel (*Numerical Methods for Unconstrained Optimization and Nonlinear
    Equations*, 1983, Sec. 5.4, Algorithm A5.4.1): component ``j`` is moved by
    ``h_j = sqrt(eps) max(|x_j|, typical_j) sign(x_j)``, and the column is divided
    by the step the machine actually took, ``(x_j + h_j) - x_j``, not the one asked
    for. ``typical`` - one per component, or one for all, one by default - is the
    size a component has where its value does not say: near zero, or on a
    component that lives at ``1e-8``, where a step floored at one is all
    truncation. ``central=True`` steps ``eps^(1/3)`` either side instead: twice
    the evaluations, error ``eps^(2/3)`` rather than ``sqrt(eps)``.

    ``base`` is the field at ``state`` when the caller has it already, which saves
    the forward difference one evaluation; a central difference does not use it.
    It stays the third positional argument it has always been, so ``typical`` and
    ``central`` are keyword-only.

    The step follows the size of the component because a fixed absolute step is a
    silent trap here: the reference orbit of a stochastic Benettin estimate need not
    be bounded - the linear multiplicative system above its dynamical threshold grows
    like ``exp(lambda t)`` and reaches ``1e30`` over a perfectly ordinary horizon -
    and once the state exceeds ``step / eps`` the perturbed state rounds back to the
    state itself, the difference is exactly zero, and the Jacobian comes back as a
    zero matrix. Nothing raises; the exponent just quietly stops depending on the
    parameter.

    No step fixes the rounding of the field itself: entry ``(i, j)`` is resolved to
    about ``sqrt(eps) |f_i| / |x_j|``, so a unit coupling in a row of size ``1e8``
    comes back anywhere in ``1 +- 3``. That is what the exact Jacobian is for.
    """
    if isinstance(function, JacobianField):
        return function.jacobian(state)
    point = np.asarray(state, dtype=float)
    scale = np.maximum(np.abs(point), _typical_sizes(typical, point.size))
    if central:
        return _central_jacobian(function, point, _CENTRAL_STEP * scale)
    steps = _FORWARD_STEP * scale * np.where(point < 0.0, -1.0, 1.0)
    if base is None:
        base = field_at(function, point)
    columns = np.empty((base.size, point.size))
    for j in range(point.size):
        shifted = point.copy()
        shifted[j] += steps[j]
        columns[:, j] = (field_at(function, shifted) - base) / (shifted[j] - point[j])
    return columns


def jacobians_along(
    function: Function,
    states: np.ndarray,
    base: np.ndarray | None = None,
    *,
    typical: float | np.ndarray | None = None,
    central: bool = False,
) -> np.ndarray:
    """:func:`jacobian_at` at every row of ``states``, as ``(count, out, n)``.

    ``base`` is the field at ``states`` (``fields_at``) when it is already in hand.
    Each column is one batched evaluation with the steps of :func:`jacobian_at` -
    the same ``typical`` sizes, the same sign rule, the same division by the step
    actually taken - formed elementwise by the same arithmetic, so row ``p``
    matches ``jacobian_at(function, states[p])`` to the bit wherever ``fields_at``
    matches ``field_at``. A :class:`~.jacobian_field.JacobianField` is asked for its
    own Jacobian at each row.
    """
    states = np.asarray(states, dtype=float)
    if isinstance(function, JacobianField):
        return np.stack([function.jacobian(state) for state in states])
    dimension = states.shape[1]
    scale = np.maximum(np.abs(states), _typical_sizes(typical, dimension))
    if central:
        return _central_jacobians(function, states, _CENTRAL_STEP * scale)
    steps = _FORWARD_STEP * scale * np.where(states < 0.0, -1.0, 1.0)
    if base is None:
        base = fields_at(function, states)
    jacobians = np.empty((states.shape[0], base.shape[1], dimension))
    for j in range(dimension):
        shifted = states.copy()
        shifted[:, j] += steps[:, j]
        taken = shifted[:, j] - states[:, j]
        jacobians[:, :, j] = (fields_at(function, shifted) - base) / taken[:, None]
    return jacobians


def _central_jacobians(
    function: Function,
    states: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """:func:`_central_jacobian` at every row, one batched evaluation a side."""
    columns: list[np.ndarray] = []
    for j in range(states.shape[1]):
        up = states.copy()
        down = states.copy()
        up[:, j] += steps[:, j]
        down[:, j] -= steps[:, j]
        difference = fields_at(function, up) - fields_at(function, down)
        columns.append(difference / (up[:, j] - down[:, j])[:, None])
    return np.stack(columns, axis=2)


def _typical_sizes(typical: float | np.ndarray | None, dimension: int) -> np.ndarray:
    """The typical size of each component, checked to be positive."""
    sizes = np.broadcast_to(
        np.asarray(_UNIT if typical is None else typical, dtype=float),
        (dimension,),
    )
    if not np.all(sizes > 0.0):
        message = "typical sizes must be positive"
        raise ValueError(message)
    return sizes


def _central_jacobian(
    function: Function,
    point: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    """Central differences, each over the width the machine actually stepped."""
    columns: list[np.ndarray] = []
    for j in range(point.size):
        up = point.copy()
        down = point.copy()
        up[j] += steps[j]
        down[j] -= steps[j]
        difference = field_at(function, up) - field_at(function, down)
        columns.append(difference / (up[j] - down[j]))
    return np.column_stack(columns)


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

    def columns_along(self, states: np.ndarray) -> np.ndarray:
        """:meth:`columns_at` at every row of ``states``, as ``(count, K, N)``."""
        return np.stack(
            [fields_at(column, states) for column in self._columns],
            axis=1,
        )

    def jacobians_along(self, states: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """:meth:`jacobians_at` at every row, as ``(count, K, N, N)``.

        ``columns`` is :meth:`columns_along` at the same states, reused as the base
        of the forward differences.
        """
        return np.stack(
            [
                jacobians_along(column, states, columns[:, index])
                for index, column in enumerate(self._columns)
            ],
            axis=1,
        )


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

    The ``K`` squares are taken as one batched ``matmul`` and then summed. A
    two-operand ``einsum`` runs the sum as a loop without BLAS, four times slower at
    ``n = K = 64``. Folding the sum into one ``(n x Kn) (Kn x n)`` product is no
    faster on one thread, and on a loaded machine its long inner dimension makes
    OpenBLAS start threads that measured up to 25 times slower than the loop.
    """
    return factor * np.matmul(jacobians, jacobians).sum(axis=0)
