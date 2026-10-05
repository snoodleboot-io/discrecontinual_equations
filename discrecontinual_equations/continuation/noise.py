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
"""

from collections.abc import Sequence

import numpy as np

from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.function.function import Function

_JACOBIAN_STEP = 1.0e-7
_UNIT = 1.0


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
