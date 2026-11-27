"""The step of every finite difference in the package, sized by the state.

A difference quotient at step ``h`` errs by a truncation that grows with ``h`` and
a rounding that grows as ``eps |f| / h`` shrinks it, and both are measured against
the size of the component being moved. Dennis and Schnabel (*Numerical Methods
for Unconstrained Optimization and Nonlinear Equations*, 1983, Sec. 5.4 and
Algorithm A5.4.1) therefore step component ``j`` by ``h_j = r max(|x_j|,
typical_j)``: ``r = sqrt(eps)`` balances the two errors of a forward difference,
``r = eps^(1/3)`` those of a central one, and ``typical_j`` - one by default, as in
their ``typx`` - is the size a component has where its value does not say, near
zero or on a component that lives at ``1e-8``.

An absolute step is the trap this replaces. Once ``|x_j|`` exceeds ``h / eps`` the
moved state rounds back to the state itself, the difference is exactly zero, and
whatever reads the derivative carries on without a warning. Where a caller lets
the user set the step, the user's number is ``r``: on components no larger than
their typical size the step is that number exactly, as it always was, and above
it the step grows with the component.

The quotient is divided by the width the machine actually stepped,
``(x_j + h_j) - (x_j - h_j)``, rather than by ``2 h_j``; the two differ by the
rounding of the moved state, which is up to ``eps |x_j| / h_j`` relative.

:func:`central_column` is that central difference for one unknown of a
``(state, parameter)`` pair, for the residuals that difference only some of their
columns (the cusp and generalized-Hopf curves, DEQ-159).
"""

from collections.abc import Callable

import numpy as np

FORWARD_STEP = float(np.sqrt(np.finfo(float).eps))
"""``sqrt(eps)``: the relative step that balances a forward difference."""

# 6.0554544523933395e-06 = 2**(-52/3) correctly rounded. A literal, not
# np.cbrt(eps), whose SIMD kernels may round differently on another machine;
# SciPy's EPS**(1/3) is 4 ulps above it (0x1.965fea53d6e41p-18).
CENTRAL_STEP = float.fromhex("0x1.965fea53d6e3dp-18")
"""``eps^(1/3)``: the relative step that balances a central difference."""

# eps^(1/(k + 2)) for k = 1 .. 4, correctly rounded and written as literals for
# the same reason as CENTRAL_STEP (computed as Decimal(2) ** (-52 / (k + 2)) at
# 60 digits). eps^(1/4) = 2**-13 is exact.
ORDER_STEPS = (
    CENTRAL_STEP,
    float.fromhex("0x1.0000000000000p-13"),
    float.fromhex("0x1.8406003b2ae5cp-11"),
    float.fromhex("0x1.428a2f98d728bp-9"),
)
"""``eps^(1/(k+2))`` for derivative orders ``k = 1 .. 4``; see :func:`order_step`."""

_UNIT = 1.0


def order_step(order: int) -> float:
    """The relative step that balances a central difference of order ``order``.

    Every central stencil the package uses for an order-``k`` derivative
    (``k = 1 .. 4``: three points for ``k <= 2``, five for ``k = 3, 4``) is
    second-order accurate, so at a relative step ``r`` it errs by a truncation
    ``c_k r^2`` plus a rounding ``w_k eps / r^k``: the samples carry an error of
    ``eps`` relative, and the stencil divides their weighted sum by ``r^k``.
    The sum is smallest where the two balance, at ``r ~ eps^(1/(k+2))``:
    ``6.1e-6`` for a slope, ``1.2e-4`` for a second derivative, ``7.4e-4`` for
    a third and ``2.5e-3`` for a fourth. A step sized for the slope, used for a
    third derivative, leaves a rounding of ``eps / r^3``, about ``2e2``
    relative at ``r = 1e-6``: that is how ``l1`` of van der Pol came back
    ``+21`` instead of ``-0.93`` (DEQ-167).

    Args:
        order: The derivative order ``k``, one to four.

    Returns:
        ``eps^(1/(k+2))``, correctly rounded.

    Raises:
        ValueError: If ``order`` is not one to four.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        that balances truncation against rounding, ``eps^(1/3)`` for a central
        slope. Mikulski and Uhl (2026), arXiv:2608.11020: the error model
        ``a_k r^2 + b_k eps / r^k`` of a second-order central stencil for the
        ``k``-th derivative and its minimiser ``r ~ eps^(1/(k+2))``. Fornberg
        (1988): the stencils' weights and their order of accuracy. See
        ``docs/references.md``.
    """
    if not 1 <= order <= len(ORDER_STEPS):
        message = f"central steps are tabulated for orders 1 to 4, not {order}"
        raise ValueError(message)
    return ORDER_STEPS[order - 1]


def typical_sizes(typical: float | np.ndarray | None, dimension: int) -> np.ndarray:
    """The typical size of each of ``dimension`` components, checked to be positive.

    ``typical`` is one per component, one for all, or ``None`` for one.
    """
    sizes = np.broadcast_to(
        np.asarray(_UNIT if typical is None else typical, dtype=float),
        (dimension,),
    )
    if not np.all(sizes > 0.0):
        message = "typical sizes must be positive"
        raise ValueError(message)
    return sizes


def step_scales(
    point: np.ndarray,
    typical: float | np.ndarray | None = None,
) -> np.ndarray:
    """``max(|x_j|, typical_j)`` for every component of ``point``.

    ``point`` is one state, ``(n,)``, or a batch of them, ``(paths, n)``; the
    typical sizes are per component either way. A step is a relative step times
    this.
    """
    sizes = typical_sizes(typical, point.shape[-1])
    return np.maximum(np.abs(point), sizes)


def central_column(  # noqa: PLR0913 (the difference and its step)
    rows: Callable[[np.ndarray, float], np.ndarray],
    state: np.ndarray,
    parameter: float,
    column: int,
    epsilon: float,
    typical: float = _UNIT,
) -> np.ndarray:
    """Central difference of ``rows`` along one unknown of ``(state, parameter)``.

    ``column`` names a state component, or the parameter when it equals
    ``state.size``. That unknown ``x`` is moved by ``h = epsilon max(|x|,
    typical)`` either way, and the difference is divided by the width actually
    stepped, ``(x + h) - (x - h)``: ``epsilon`` is a relative step, and on an
    unknown no larger than its typical size (one by default) the step is
    ``epsilon typical``. An unknown that lives near ``1e-8`` needs ``typical``
    of that size, or a step floored at one swamps it (DEQ-158). The forward
    point is evaluated first. ``state`` is not modified.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        ``max(|x_j|, typx_j)`` times a relative step, with ``typx_j`` the
        ``typical`` size. SciPy 1.14.1, ``scipy/optimize/_numdiff.py``: the
        division by the width actually stepped. See ``docs/references.md``.
    """
    moves_parameter = column == state.size
    value = float(parameter) if moves_parameter else float(state[column])
    step = epsilon * float(step_scales(np.array([value]), typical)[0])
    up = value + step
    down = value - step
    if moves_parameter:
        upper = rows(state, up)
        lower = rows(state, down)
    else:
        forward = np.array(state, dtype=float)
        backward = np.array(state, dtype=float)
        forward[column] = up
        backward[column] = down
        upper = rows(forward, parameter)
        lower = rows(backward, parameter)
    return (upper - lower) / (up - down)
