"""Jacobian abstraction, decoupled from how derivatives are obtained."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np

from discrecontinual_equations.continuation.residual import (
    DifferentiableResidual,
    ResidualFunction,
)
from discrecontinual_equations.continuation.sparsity import SparsityPattern
from discrecontinual_equations.finite_difference import step_scales, typical_sizes


class JacobianProvider(ABC):
    """Supply the derivatives of the residual needed for continuation."""

    @abstractmethod
    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``df/du`` of shape ``(n, n)``."""
        raise NotImplementedError

    @abstractmethod
    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``df/dlambda`` of shape ``(n,)``."""
        raise NotImplementedError

    def extended_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return the augmented ``[df/du | df/dlambda]`` of shape ``(n, n + 1)``."""
        state_part = self.state_jacobian(state, parameter)
        parameter_part = self.parameter_derivative(state, parameter).reshape(-1, 1)
        return np.hstack([state_part, parameter_part])

    @contextmanager
    def retained(self) -> Iterator[None]:
        """Scope within which a repeated request at one point may be remembered.

        The base provider remembers nothing. :class:`MemoizedJacobian` overrides this
        so that, inside the scope, the converged point's Jacobian is built once and
        shared by the tangent, the detectors and the point's stability.
        """
        yield


class FiniteDifferenceJacobian(JacobianProvider):
    """Central finite-difference Jacobian; needs no analytic derivatives.

    Column ``j`` is the central difference of the residual at step
    ``h_j = epsilon * max(|u_j|, typical_j)``, divided by the width the machine
    actually stepped, ``(u_j + h_j) - (u_j - h_j)``; the parameter column steps
    ``epsilon * max(|lambda|, parameter_typical)``. ``epsilon`` is therefore a
    *relative* step (DEQ-118). On a component no larger than its typical size -
    one by default, so every state and parameter of size one or less - the step
    is ``epsilon`` itself, which is what ``finite_difference_epsilon`` has
    always meant there; above it the step grows with the component, where an
    absolute step would round away (an ulp of ``1e12`` is ``1.2e-4``) and
    return a zero column. ``typical`` - one per state component, or one for all
    - is the size of a component that lives far below one, which a step floored
    at one would swamp. ``eps^(1/3)``, about ``6e-6``, balances truncation
    against rounding; the default configuration's ``1e-7`` trades some rounding
    for less truncation.

    Dense by default: ``2n`` residual evaluations, one pair per column. Given a
    ``sparsity`` - a :class:`~.sparsity.SparsityPattern`, for each equation the
    list of state indices it depends on, or ``"detect"`` to find the pattern from
    dense Jacobians at the first point it is asked about - it differences each
    group of structurally orthogonal columns in one pair instead (Curtis, Powell
    and Reid), each column of the group moved by its own step and divided by its
    own width, ``2g`` evaluations for ``g`` groups, and every entry is
    bit-identical to the dense one as long as the pattern covers the field's.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        scaled per component by ``max(|x_j|, typx_j)``. SciPy 1.14.1,
        ``scipy/optimize/_numdiff.py``: the central relative step and the
        division by the width actually stepped. See ``docs/references.md``.

    Complexity:
        ``state_jacobian`` costs ``2n`` residual evaluations dense, or ``2g`` for
        ``g`` column groups (a banded or ring field has ``g`` independent of
        ``n``), and ``parameter_derivative`` two more. The result is a dense
        ``n x n`` array either way, ``O(n^2)`` memory.
    """

    __slots__ = [
        "_epsilon",
        "_parameter_typical",
        "_pattern",
        "_residual",
        "_sparsity",
        "_typical",
    ]

    def __init__(
        self,
        residual: ResidualFunction,
        epsilon: float,
        sparsity: SparsityPattern | list[list[int]] | Literal["detect"] | None = None,
        typical: float | np.ndarray | None = None,
        parameter_typical: float = 1.0,
    ) -> None:
        self._residual = residual
        self._epsilon = epsilon
        self._sparsity = sparsity
        self._pattern: SparsityPattern | None = (
            sparsity if isinstance(sparsity, SparsityPattern) else None
        )
        self._typical = typical
        typical_sizes(parameter_typical, 1)
        self._parameter_typical = parameter_typical

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        pattern = self._resolved_pattern(state, parameter)
        if pattern is None:
            return self._dense(state, parameter)
        return self._grouped(state, parameter, pattern)

    def _resolved_pattern(
        self,
        state: np.ndarray,
        parameter: float,
    ) -> SparsityPattern | None:
        if self._pattern is not None or self._sparsity is None:
            return self._pattern
        if self._sparsity == "detect":
            self._pattern = SparsityPattern.detect(
                lambda point: self._dense(point, parameter),
                np.asarray(state, dtype=float),
            )
        elif isinstance(self._sparsity, list):
            self._pattern = SparsityPattern.from_rows(self._sparsity, state.size)
        return self._pattern

    def _steps(self, point: np.ndarray) -> np.ndarray:
        """``epsilon * max(|u_j|, typical_j)`` for every state component."""
        return self._epsilon * step_scales(point, self._typical)

    def _grouped(
        self,
        state: np.ndarray,
        parameter: float,
        pattern: SparsityPattern,
    ) -> np.ndarray:
        point = np.asarray(state, dtype=float)
        steps = self._steps(point)
        directions = pattern.directions()
        compressed = np.zeros((pattern.mask.shape[0], directions.shape[0]))
        width = np.empty(point.size)
        for group, direction in enumerate(directions):
            moved = direction != 0.0
            forward = point.copy()
            backward = point.copy()
            forward[moved] += steps[moved]
            backward[moved] -= steps[moved]
            width[moved] = forward[moved] - backward[moved]
            compressed[:, group] = self._residual.evaluate(
                forward,
                parameter,
            ) - self._residual.evaluate(backward, parameter)
        return pattern.expand(compressed) / width

    def _dense(self, state: np.ndarray, parameter: float) -> np.ndarray:
        point = np.asarray(state, dtype=float)
        dimension = point.size
        steps = self._steps(point)
        jacobian = np.zeros((dimension, dimension))
        for column in range(dimension):
            forward = point.copy()
            backward = point.copy()
            forward[column] += steps[column]
            backward[column] -= steps[column]
            jacobian[:, column] = (
                self._residual.evaluate(forward, parameter)
                - self._residual.evaluate(backward, parameter)
            ) / (forward[column] - backward[column])
        return jacobian

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        step = self._epsilon * max(abs(parameter), self._parameter_typical)
        up = parameter + step
        down = parameter - step
        forward = self._residual.evaluate(state, up)
        backward = self._residual.evaluate(state, down)
        return (forward - backward) / (up - down)


class AnalyticJacobian(JacobianProvider):
    """The extended Jacobian the residual computes itself.

    For augmented systems whose rows are bordered test functions, differencing
    every column would redo the whole bordered reduction per column; the residual
    instead differentiates its rows analytically (see
    :class:`~.residual.DifferentiableResidual`).

    ``epsilon`` is handed to the residual for the columns it still differences,
    as a *relative* step (DEQ-159): the unknown ``x`` such a column moves steps
    ``epsilon max(|x|, typical_x)``, as in :class:`FiniteDifferenceJacobian`.
    The builder passes ``finite_difference_epsilon``, so on unknowns of size
    one or less nothing changes, and on a curve whose parameters are of size
    ``1e12`` the differenced columns are still resolved.

    ``typical`` (one per state unknown, or one for all) and
    ``parameter_typical`` are the typical sizes of the residual's unknowns, as
    in :class:`FiniteDifferenceJacobian` (DEQ-158): on a cusp or
    generalized-Hopf curve whose parameters live near ``1e-8`` a typical size
    of one steps them by ``epsilon``, which swamps them. When either is set the
    residual is handed the ``m + 1`` sizes as ``typical=``; when neither is,
    it is called with ``epsilon`` alone, as before, so a
    :class:`~.residual.DifferentiableResidual` written before DEQ-158 still
    works.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        scaled per unknown by ``max(|x_j|, typx_j)``. See
        ``docs/references.md``.
    """

    __slots__ = ["_epsilon", "_parameter_typical", "_residual", "_typical"]

    def __init__(
        self,
        residual: ResidualFunction,
        epsilon: float,
        typical: float | np.ndarray | None = None,
        parameter_typical: float | None = None,
    ) -> None:
        if not isinstance(residual, DifferentiableResidual):
            message = "AnalyticJacobian needs a DifferentiableResidual"
            raise TypeError(message)
        self._residual = residual
        self._epsilon = epsilon
        if parameter_typical is not None:
            typical_sizes(parameter_typical, 1)
        self._typical = typical
        self._parameter_typical = parameter_typical

    def extended_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        if self._typical is None and self._parameter_typical is None:
            return self._residual.extended_jacobian(state, parameter, self._epsilon)
        point = np.asarray(state, dtype=float)
        sizes = np.append(
            typical_sizes(self._typical, point.size),
            typical_sizes(self._parameter_typical, 1),
        )
        return self._residual.extended_jacobian(
            point,
            parameter,
            self._epsilon,
            typical=sizes,
        )

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self.extended_jacobian(state, parameter)[:, :-1]

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self.extended_jacobian(state, parameter)[:, -1]


class MemoizedJacobian(JacobianProvider):
    """Remember the last few points' derivatives, but only inside :meth:`retained`.

    One accepted continuation point asks for the Jacobian at the converged state
    four times - the tangent, the branch-point determinant, the Hopf test and the
    stability - and the localizer asks again at the bracket's lower end. Inside a
    retained scope each of those after the first is a copy of the remembered array,
    so a point costs ``I + 1`` Jacobians instead of ``I + 4`` and every value is
    bit-identical to recomputing it. Points are keyed by the exact bytes of the
    state and the parameter.

    The memory lives only for the scope (one :meth:`Continuer.solve`): the residual
    reads the equation's other parameters live, so a Jacobian remembered across
    solves could be stale after the caller changes one of them.

    A provider that overrides :meth:`extended_jacobian` solves the whole extended
    Jacobian at once (as :class:`AnalyticJacobian` does), so its state part and
    parameter part are both slices of one solve. For such a provider the wrapper
    remembers the extended Jacobian itself and slices the parts from it, so each
    point costs one solve rather than one per part. Other providers keep their
    parts separate, since differencing the state columns alone is cheaper than the
    whole extended Jacobian. Outside a scope every request goes straight to the
    provider, one call per request.

    Args:
        provider: The provider whose derivatives are remembered.

    Attributes:
        capacity: How many points a scope remembers; the least recently used
            point is forgotten first.

    References:
        No published algorithm: this is a cache, and every value it returns is
        bit-identical to the provider's own. The analytic extended Jacobians it
        shares are those of the minimally augmented cusp and generalized-Hopf
        systems, whose sources are cited in :mod:`~.cusp_curve` and
        :mod:`~.generalized_hopf_curve`.
    """

    __slots__ = ["_active", "_joint", "_memory", "_provider"]

    capacity = 4

    def __init__(self, provider: JacobianProvider) -> None:
        self._provider = provider
        self._joint = (
            type(provider).extended_jacobian is not JacobianProvider.extended_jacobian
        )
        self._active = False
        self._memory: OrderedDict[tuple[bytes, bytes], dict[str, np.ndarray]] = (
            OrderedDict()
        )

    def extended_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        if not self._joint:
            return super().extended_jacobian(state, parameter)
        return self._remembered(
            "extended",
            state,
            parameter,
            self._provider.extended_jacobian,
        )

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        if self._joint and self._active:
            return self.extended_jacobian(state, parameter)[:, :-1]
        return self._remembered(
            "state",
            state,
            parameter,
            self._provider.state_jacobian,
        )

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        if self._joint and self._active:
            return self.extended_jacobian(state, parameter)[:, -1]
        return self._remembered(
            "parameter",
            state,
            parameter,
            self._provider.parameter_derivative,
        )

    @contextmanager
    def retained(self) -> Iterator[None]:
        self._active = True
        try:
            yield
        finally:
            self._active = False
            self._memory.clear()

    def _remembered(
        self,
        part: str,
        state: np.ndarray,
        parameter: float,
        compute: Callable[[np.ndarray, float], np.ndarray],
    ) -> np.ndarray:
        if not self._active:
            return compute(state, parameter)
        key = (
            np.ascontiguousarray(state, dtype=float).tobytes(),
            np.float64(parameter).tobytes(),
        )
        parts = self._memory.get(key)
        if parts is None:
            parts = {}
            self._memory[key] = parts
            if len(self._memory) > self.capacity:
                self._memory.popitem(last=False)
        else:
            self._memory.move_to_end(key)
        if part not in parts:
            parts[part] = np.array(compute(state, parameter), dtype=float)
        return parts[part].copy()
