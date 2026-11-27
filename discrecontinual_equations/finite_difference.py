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
"""

import numpy as np

FORWARD_STEP = float(np.sqrt(np.finfo(float).eps))
"""``sqrt(eps)``: the relative step that balances a forward difference."""

# 6.0554544523933395e-06 = 2**(-52/3) correctly rounded. A literal, not
# np.cbrt(eps), whose SIMD kernels may round differently on another machine;
# SciPy's EPS**(1/3) is 4 ulps above it (0x1.965fea53d6e41p-18).
CENTRAL_STEP = float.fromhex("0x1.965fea53d6e3dp-18")
"""``eps^(1/3)``: the relative step that balances a central difference."""

_UNIT = 1.0


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
