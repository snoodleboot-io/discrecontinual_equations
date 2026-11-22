"""A vector field that carries its own Jacobian.

Every Lyapunov estimator, the covariant vectors and the quasipotential's
linearisation take the field's Jacobian from :func:`~.noise.jacobian_at`, which
finite-differences the field. That costs ``n + 1`` evaluations a call, and its
accuracy has a floor no step can lower: an entry ``df_i/dx_j`` is resolved only to
about ``sqrt(eps) |f_i| / |x_j|``, so a small coupling in a large row - the
``x0``-dependence of a component of size ``1e8`` - is lost in the rounding of
``f_i`` itself. A field wrapped in :class:`JacobianField` hands ``jacobian_at``
its exact Jacobian instead, so every estimator that reads one uses it without a
new argument anywhere.

The Jacobian is either supplied in closed form or, by default, computed by the
Taylor-jet automatic differentiation the continuation solvers already use
(:class:`~.derivative_provider.AutomaticDifferentiation`): exact to rounding for
any polynomial or rational field, with no step at all. Jets are the convenient
choice and the closed form the fast one; on Lorenz a jet Jacobian costs about
twice a finite difference, a closed form about one field evaluation.
"""

from collections.abc import Callable

import numpy as np

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.function import Function

_AUTODIFF = AutomaticDifferentiation()
# Jets are seeded at this time; the fields these estimators take are autonomous,
# as the finite difference, which evaluates them with no time, also assumes.
_JET_TIME = 0.0


class JacobianField(Function):
    """``function``, evaluated as itself, with an exact Jacobian attached.

    ``jacobian`` maps a state to the ``(n, n)`` Jacobian there; left out, the
    field is differentiated on Taylor jets, which raises ``TypeError`` for a field
    written with ``math`` functions or fractional powers - supply a closed form
    for those. The wrapper shares the wrapped field's variables and parameters, so
    a parameter changed on either is seen by both.
    """

    __slots__ = ["_function", "_jacobian"]

    def __init__(
        self,
        function: Function,
        jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__(
            variables=function.variables,
            parameters=function.parameters,
            results=function.curve.results,
            time=function.time,
        )
        self._function = function
        self._jacobian = jacobian

    @property
    def function(self) -> Function:
        """The field that is evaluated."""
        return self._function

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        return self._function.eval(point=point, time=time)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """The Jacobian at ``state``, as supplied or by Taylor jets."""
        if self._jacobian is not None:
            return np.asarray(self._jacobian(state), dtype=float)
        return _AUTODIFF.jacobian(self._function, state, _JET_TIME)
